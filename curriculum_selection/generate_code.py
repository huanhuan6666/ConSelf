import os
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
from utils import *
from vllm_interface import VLLMRunner
from transformers import AutoTokenizer
from prompts.code_generation import get_code_generation_prompt
from prompts.observation_generation import get_observation_generation_prompt
logger = get_logger("curriculum_main")


def main():
    cfg = get_config()
    base_dataset_name = os.path.basename(cfg["data"]["dataset_path"]).replace('/', '_')
    save_dir = os.path.join(cfg["data"]["save_dir"], base_dataset_name, f"{cfg['llm']['local_model_path'].split('/')[-1]}")
    obs_file_name = f"{base_dataset_name}_with_observations.jsonl"
    os.makedirs(save_dir, exist_ok=True)
    obs_save_path = os.path.join(save_dir, obs_file_name)

    if os.path.exists(obs_save_path):
        logger.info(f"Found existing observations file. Loading from: {obs_save_path}")
        dataset_with_observations = load_jsonl(obs_save_path)
    else:
        logger.info(f"Observations file not found. Generating observations...")
        dataset = load_dataset(cfg["data"]["dataset_path"], split="train")
        logger.info(f"Loaded original dataset with {len(dataset)} problems")

        engine = VLLMRunner(args=cfg["llm"])
        logger.info("Initialized vLLM with model: %s", cfg["llm"]["local_model_path"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["local_model_path"], trust_remote_code=True)

        observation_prompts = []
        for example in tqdm(dataset, desc="Generating observation prompts"):
            obs_sys_prompt, obs_question_format = get_observation_generation_prompt(problem=example["question"])
            messages = [
                {"role": "system", "content": obs_sys_prompt},
                {"role": "user", "content": obs_question_format}
            ]
            prompt = get_conversation_prompt_by_messages(tokenizer, messages)
            observation_prompts.append(prompt)

        observation_outputs = engine.run_batch(
            prompts=observation_prompts,
            temperature=cfg["sampling"]["temperature_observations"],
            calculate_confidence=False
        )

        dataset_with_observations = []
        for example, output_list in tqdm(zip(dataset, observation_outputs), total=len(dataset), desc="Attaching observations"):
            if output_list:
                obs = extract_observation(output_list)
            else:
                obs = ""
            
            new_example = example.copy()
            new_example["observations"] = obs
            dataset_with_observations.append(new_example)

        save_jsonl(dataset_with_observations, obs_save_path)
        logger.info(f"Saved dataset with observations to {obs_save_path}")

    if 'engine' not in locals():
        engine = VLLMRunner(args=cfg["llm"])
        logger.info("Initialized vLLM for code generation: %s", cfg["llm"]["local_model_path"])
    if 'tokenizer' not in locals():
        tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["local_model_path"], trust_remote_code=True)

    code_prompts_data = []
    for i, ex in enumerate(tqdm(dataset_with_observations, desc="Generating code prompts")):
        if "observations" in ex and ex["observations"]:
            for obs in ex["observations"][:cfg["sampling"]["num_observations"]]:
                code_sys_prompt, code_question_format = get_code_generation_prompt(
                    problem=ex["question"], observation=obs, starter_code=ex.get("starter_code", ""))
                messages = [
                    {"role": "system", "content": code_sys_prompt},
                    {"role": "user", "content": code_question_format}
                ]
                prompt = get_conversation_prompt_by_messages(tokenizer, messages)
                code_prompts_data.append({
                    "index": i,
                    "observation": obs,
                    "prompt": prompt
                })

    code_prompts_strings = [item["prompt"] for item in code_prompts_data]

    code_outputs = engine.run_batch(
        prompts=code_prompts_strings,
        n=cfg["sampling"]["num_code_per_observation"],
        temperature=cfg["sampling"]["temperature_code"],
        calculate_confidence=True
    )
    index_to_aggregated_outputs = {}
    logger.info("Aggregating outputs and extracting codes...")
    assert len(code_prompts_data) == len(code_outputs), f"len(code_prompts_data) != len(code_outputs): {len(code_prompts_data)} != {len(code_outputs)}"
    for prompt_data, output_completions in tqdm(zip(code_prompts_data, code_outputs), total=len(code_prompts_data), desc="Aggregating results"):
        original_index = prompt_data["index"]
        if original_index not in index_to_aggregated_outputs:
            index_to_aggregated_outputs[original_index] = {
                "raw_outputs": [],
                "extracted_codes": []
            }
        for completion in output_completions:
            text = completion.get("text", "")
            if "```python" in text:
                try:
                    code = text.split("```python", 1)[1].split("```", 1)[0].strip()
                except IndexError:
                    try:
                        code = text.split("```python", 1)[1].strip()
                    except IndexError:
                        code = ""
            else:
                code = ""
            if code:
                index_to_aggregated_outputs[original_index]["raw_outputs"].append(completion)
                index_to_aggregated_outputs[original_index]["extracted_codes"].append(code)

    final_dataset = []
    logger.info("Attaching aggregated results to final dataset...")
    for i, ex in enumerate(tqdm(dataset_with_observations, desc="Creating final dataset")):
        ex_copy = ex.copy()
        aggregated_data = index_to_aggregated_outputs.get(i, {
            "raw_outputs": [],
            "extracted_codes": []
        })
        ex_copy["output_list"] = aggregated_data["raw_outputs"]
        ex_copy["code_list"] = aggregated_data["extracted_codes"]
        ex_copy.pop("observations", None)
        final_dataset.append(ex_copy)

    final_file_name = f"{base_dataset_name}_with_responses.jsonl"
    final_save_path = os.path.join(save_dir, final_file_name)
    save_jsonl(final_dataset, final_save_path)
    logger.info(f"Saved full dataset with responses to {final_save_path}")


if __name__ == "__main__":
    main()