import os
import json
import sys
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
dataset_path = "/xxx/datasets/TACO" # the path of the raw TACO dataset
split_name = "train"
output_dir = "curriculum_selection/data/taco"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "taco_cleaned.jsonl")
max_io_pairs = 30
max_str_len_threshold = 5000
min_pairs_for_length_filter = 5

print(f"Loading '{split_name}' split from {dataset_path}...")
taco_train_dataset = load_dataset(dataset_path, split=split_name, trust_remote_code=True)
print(f"Successfully loaded {len(taco_train_dataset)} examples from the '{split_name}' split.")

cleaned_data = []
skipped_picture_num = 0
skipped_json_error = 0
skipped_initial_zero_io = 0
skipped_all_io_filtered = 0
total_pairs_processed = 0
total_pairs_filtered_by_length = 0
final_test_case_counts = defaultdict(int)
skipped_missing_fn_name = 0

print(f"\nProcessing examples, filtering I/O pairs (threshold: {max_str_len_threshold} chars if >= {min_pairs_for_length_filter} pairs), and filtering examples...")
print("-" * 70)

for idx, example in tqdm(enumerate(taco_train_dataset), total=len(taco_train_dataset), desc="Processing examples"):
    picture_num = example.get("picture_num", 0)
    if picture_num is not None and picture_num != '0' or '<image>' in example.get("question", ""):
        skipped_picture_num += 1
        continue

    starter_code = example.get("starter_code", "")
    if starter_code and starter_code.strip():
        if not example.get("input_output"):  # No input_output at all
            skipped_missing_fn_name += 1
            continue
        try:
            input_output = json.loads(example["input_output"])
            if "fn_name" not in input_output:
                skipped_missing_fn_name += 1
                continue
        except (json.JSONDecodeError, ValueError) as e:
            skipped_missing_fn_name += 1
            continue

    input_output_dict = None
    final_inputs = []
    final_outputs = []
    initial_pair_count = 0

    if "input_output" in example and example["input_output"]:
        try:
            loaded_io = json.loads(example["input_output"])
            input_output_dict = loaded_io

            original_inputs = input_output_dict.get("inputs", [])
            original_outputs = input_output_dict.get("outputs", [])
            initial_pair_count = min(len(original_inputs), len(original_outputs))
            total_pairs_processed += initial_pair_count

            if initial_pair_count == 0:
                skipped_initial_zero_io += 1
                continue

            apply_length_filter = initial_pair_count >= min_pairs_for_length_filter
            intermediate_inputs = []
            intermediate_outputs = []
            pairs_filtered_in_this_example = 0

            if apply_length_filter:
                for i in range(initial_pair_count):
                    input_item = original_inputs[i]
                    output_item = original_outputs[i]
                    if len(str(input_item)) <= max_str_len_threshold and \
                       len(str(output_item)) <= max_str_len_threshold:
                        intermediate_inputs.append(input_item)
                        intermediate_outputs.append(output_item)
                    else:
                        pairs_filtered_in_this_example += 1
                        total_pairs_filtered_by_length += 1
            else:
                for i in range(initial_pair_count):
                    intermediate_inputs.append(original_inputs[i])
                    intermediate_outputs.append(original_outputs[i])
                pairs_filtered_in_this_example = 0

            final_inputs = intermediate_inputs[:max_io_pairs]
            final_outputs = intermediate_outputs[:max_io_pairs]

            if apply_length_filter and len(final_inputs) == 0:
                skipped_all_io_filtered += 1
                continue

            input_output_dict["inputs"] = final_inputs
            input_output_dict["outputs"] = final_outputs

        except (ValueError, json.JSONDecodeError, Exception) as e:
            error_type = type(e).__name__
            print(f"Warning: Skipping example {idx} due to {error_type} during I/O processing. Error: {e}. Question starts with: {example.get('question', '')[:50]}...")
            skipped_json_error += 1
            continue
    else:
        skipped_initial_zero_io += 1
        continue

    cleaned_example = {
        "question_id": f"taco_{idx}",
        "question": example.get("question", ""),
        "starter_code": example.get("starter_code", ""),
        "difficulty": example.get("difficulty", ""),
        "name": example.get("name", ""),
        "source": example.get("source", ""),
        "url": example.get("url", ""),
        "solutions": json.dumps(json.loads(example.get("solutions", ""))[:1]),
        "input_output": json.dumps(input_output_dict)
    }

    final_count = len(final_inputs)
    final_test_case_counts[final_count] += 1

    cleaned_data.append(cleaned_example)

print(f"\nSaving cleaned dataset to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in cleaned_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

