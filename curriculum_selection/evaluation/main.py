import json
import os
from tqdm import tqdm
import timeout_decorator
import traceback
from compute_code_generation_metrics import codegen_metrics
from pass_k_utils import extract_instance_results
from calculate_confidence import process_evaluation_item
def safe_exec(code_str, func_name="solve"):
    """
    Safely execute a code string and retrieve the function object.
    """
    local_vars = {}
    try:
        exec(code_str, {}, local_vars)
        if func_name not in local_vars:
            return None
        return local_vars[func_name]
    except Exception:
        return None

@timeout_decorator.timeout(3, timeout_exception=TimeoutError)
def run_single_test(solve_fn, test_input):
    """
    Run a single test input through the solution function.
    """
    if isinstance(test_input, (list, tuple)):
        return solve_fn(*test_input)
    else:
        return solve_fn(test_input)

def run_code_on_tests(code_str, inputs_list, expected_outputs):
    """
    Execute a code string on all tests and check if all pass.
    """
    try:
        solve_fn = safe_exec(code_str)
        if solve_fn is None:
            return False
        
        for input_item, expected_output in zip(inputs_list, expected_outputs):
            output = run_single_test(solve_fn, input_item)
            if output != expected_output:
                return False
        return True
    except Exception:
        return False

def evaluate_one_problem(example):
    """
    Evaluate all code samples for one problem.
    """
    inputs_list = json.loads(example["tests"])["inputs"]
    expected_outputs = json.loads(example["tests"])["outputs"]

    for response in example["responses"]:
        for sample in response["samples"]:
            code = sample["text"]
            passed = run_code_on_tests(code, inputs_list, expected_outputs)
            sample["passed"] = passed
    return example

def get_evaluation_sample(instance):
    return {
        "input_output": instance["input_output"],
    }

def get_generation_codes(instance):
    codes = []
    for response in instance["responses"]:
        for sample in response["samples"]:
            codes.append(sample["text"])
    return codes


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="curriculum_selection/data/processed/taco/CodeLlama-7b-Instruct-hf/taco_with_responses.jsonl", type=str, help="Path to input .jsonl file with responses")
    parser.add_argument("--chunk_size", default=5000, type=int, help="Chunk size for evaluation")
    parser.add_argument("--num_process_evaluate", default=16, type=int, help="Number of processes for evaluation")
    parser.add_argument("--start_index", default=0, type=int, help="Start index for evaluation")
    parser.add_argument("--end_index", default=None, type=int, help="End index for evaluation")
    args = parser.parse_args()
    chunk_size = args.chunk_size
    assert os.path.exists(args.input_file), f"Input file {args.input_file} not found."
    output_file = args.input_file.replace("with_responses.jsonl", "evaluated.jsonl")
    output_dir = os.path.dirname(args.input_file) + "/evaluate_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    benchmark = load_jsonl(args.input_file)
    print(f"Loaded {len(benchmark)} problems from {args.input_file}")
    existing_question_ids = [f.split(".")[0] for f in os.listdir(output_dir) if f.endswith(".json")]
    print(f"Found {len(existing_question_ids)} problems already evaluated")
    benchmark = [instance for instance in benchmark if instance["question_id"] not in existing_question_ids]
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(benchmark)
    print(f"Remaining {len(benchmark)} problems to be evaluated. Begin from {start_index} to {end_index}")

    eval_samples = [{"input_output": instance["input_output"]} for instance in benchmark]
    generations = [instance['code_list'] for instance in benchmark]
    assert len(eval_samples) == len(generations), f"len(eval_samples) != len(generations): {len(eval_samples)} != {len(generations)}"
    for i in range(len(generations) - 1, -1, -1):
        if len(generations[i]) == 0:
            print(f"Warning: problem {i} has no code. Skip it.")
            generations.pop(i)
            eval_samples.pop(i)
            benchmark.pop(i)
    assert len(eval_samples) == len(generations) == len(benchmark), f"len(eval_samples) != len(generations) != len(benchmark): {len(eval_samples)} != {len(generations)} != {len(benchmark)}"
    print(f"After removing problems with no code, {len(benchmark)} problems left")
    eval_samples = eval_samples[start_index:end_index]
    generations = generations[start_index:end_index]
    for i in range(start_index, end_index, chunk_size):
        metrics = codegen_metrics(
            eval_samples[i:i+chunk_size],
            generations[i:i+chunk_size],
            timeout=3,
            num_process_evaluate=args.num_process_evaluate,
        )
        print("In get_metrics print metrics[0]['pass@1']: ", metrics[0]["pass@1"])
        graded_chunk = extract_instance_results(metrics[1])
        metadatas = metrics[2]
        save_eval = []
        for instance, graded, meta in zip(
            benchmark[i:i+chunk_size], graded_chunk, metadatas
        ):
            instance["graded_list"] = graded
            instance["pass@1"] = graded.count(True) / len(graded)
            instance["metadata"] = meta
            process_evaluation_item(instance)
            if len(instance['code_list']) != len(meta['generation_results']):
                print(f"Warning: problem {instance['question_id']} has {len(instance['code_list'])} code, but {len(meta['generation_results'])} generation results")
                continue
            save_eval.append(instance)
            question_id = instance["question_id"]
            with open(f"{output_dir}/{question_id}.json", "w") as f:
                json.dump(instance, f, ensure_ascii=False, indent=4)
    print(f"Saved evaluated results to {output_dir}")

if __name__ == "__main__":
    main()
