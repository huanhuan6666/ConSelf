import argparse
import yaml
import os
import json
import logging
from jinja2 import Template
import importlib
import re
def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_observation(output: list[dict]) -> list[str]:
    pattern = r"Observation \d+:\s*(.*?)\s*(?=Observation \d+:|$)"
    matches = re.findall(pattern, output[0]["text"], re.DOTALL)
    observations = [m.strip() for m in matches]
    return observations

def extract_codes_from_outputs(output_completions: list[dict]) -> list[str]:
    """
    Extracts code blocks (specifically ```python ... ```) from a list of
    VLLM output dictionaries and returns a list of code strings.
    """
    extracted_codes = []
    for completion in output_completions:
        text = completion.get("text", "")
        if "```python" in text:
            try:
                code = text.split("```python", 1)[1].split("```", 1)[0].strip()
                extracted_codes.append(code)
            except IndexError:
                try:
                    code = text.split("```python", 1)[1].strip()
                    extracted_codes.append(code)
                except IndexError:
                    extracted_codes.append("")
        else:
            extracted_codes.append("")
    return [code for code in extracted_codes if code]


def save_jsonl(data_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data_list:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_prompts(action_name):
    file_path = os.path.join("curriculum_selection", "prompts", f"{action_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")
    return system_prompt, question_format

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_logger(name="default"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def get_config():
    def merge_config_with_args(cfg, args):
        if args.local_model_path is not None:
            cfg["llm"]["local_model_path"] = args.local_model_path
        if args.temperature_observations is not None:
            cfg["sampling"]["temperature_observations"] = args.temperature_observations
        if args.temperature_code is not None:
            cfg["sampling"]["temperature_code"] = args.temperature_code
        if args.num_code_per_observation is not None:
            cfg["sampling"]["num_code_per_observation"] = args.num_code_per_observation
        if args.num_observations is not None:
            cfg["sampling"]["num_observations"] = args.num_observations
        if args.dataset_path is not None:
            cfg["data"]["dataset_path"] = args.dataset_path
        if args.save_dir is not None:
            cfg["data"]["save_dir"] = args.save_dir
        return cfg
    parser = argparse.ArgumentParser(description="Curriculum Selection Config")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--local_model_path", type=str, help="Path to local model")
    parser.add_argument("--config", type=str, default="curriculum_selection/config/default.yaml", help="Path to YAML config")
    parser.add_argument("--temperature_observations", type=float, help="Sampling temperature for code generation")
    parser.add_argument("--temperature_code", type=float, help="Sampling temperature for code generation")
    parser.add_argument("--num_code_per_observation", type=int, help="Number of samples per observation")
    parser.add_argument("--num_observations", type=int, help="Number of observations per problem")
    parser.add_argument("--save_dir", type=str, help="Directory to save processed dataset")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    cfg = merge_config_with_args(cfg, args)

    return cfg