import json
import math
from collections import defaultdict
import os
import argparse
import sys
from tqdm import tqdm

def calculate_semantic_entropy(item):
    """
    Calculates the semantic entropy based on execution outputs.

    Groups code samples based on their output sequence across test cases.
    Calculates the entropy of the cluster distribution.

    Args:
        item (dict): The dictionary loaded from a per-problem JSON result file.

    Returns:
        float or None: The calculated semantic entropy, or None if calculation
                       is not possible (e.g., no valid generation results).
    """
    if "metadata" not in item or "generation_results" not in item["metadata"]:
        print(f"Warning: Missing 'metadata' or 'generation_results' for q_id {item.get('question_id')}")
        return None

    generation_results_str_list = item["metadata"]["generation_results"]
    num_samples = len(generation_results_str_list)
    if num_samples == 0:
        return None

    behavior_counts = defaultdict(int)
    total_valid_samples = 0
    full_error_count = 0

    for gen_result_str in generation_results_str_list:
        output_sequence = []
        try:
            test_case_results = json.loads(gen_result_str)
            if not isinstance(test_case_results, list):
                 if isinstance(test_case_results, dict) and "error_code" in test_case_results:
                    behavior_tuple = ("ERROR",)
                 else:
                    print(f"Warning: Unexpected format in generation_results for q_id {item.get('question_id')}. Skipping sample for entropy.")
                    continue
            else:
                for test_result in test_case_results:
                    if isinstance(test_result, dict):
                        # -1 CompileError -3 Timeout -4 RuntimeError -5 TestRunnerError
                        if test_result.get("error_code") in [-1, -3, -4, -5]:
                            output_sequence.append(None)
                        else: # -2 WrongAnswer
                            output_sequence.append(test_result.get("output"))
                    else:
                        output_sequence.append(None)
            if not all(x is None for x in output_sequence):
                behavior_str = str(output_sequence)
                behavior_counts[behavior_str] += 1
                total_valid_samples += 1
            else:
                full_error_count += 1
        except Exception as e:
            behavior_counts[("PROCESSING_ERROR",)] += 1
            total_valid_samples += 1
            print(f"Warning: Error processing generation_results for q_id {item.get('question_id')}: {e}")

    crash_rate = full_error_count/num_samples
    if total_valid_samples == 0:
        return None, crash_rate

    entropy = 0.0
    for behavior in behavior_counts:
        count = behavior_counts[behavior]
        probability = count / total_valid_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy, crash_rate

def calculate_average_confidences(item):
    """
    Calculates the average confidence metrics across all samples.

    Args:
        item (dict): The dictionary loaded from a per-problem JSON result file.

    Returns:
        dict: A dictionary containing the average metrics (avg_nll, avg_token_entropy).
              Values are None if no valid samples are found for a metric.
    """
    if "output_list" not in item or not isinstance(item["output_list"], list):
        print(f"Warning: Missing or invalid 'output_list' for q_id {item.get('question_id')}")
        return {"avg_all_samples_nll": None, "avg_all_samples_top_k_entropy": None}

    metrics_to_average = ["avg_nll", "avg_token_entropy"]
    sums = {key: 0.0 for key in metrics_to_average}
    counts = {key: 0 for key in metrics_to_average}

    for sample_output_info in item["output_list"]:
        if not isinstance(sample_output_info, dict):
            continue # Skip malformed entries

        for key in metrics_to_average:
            value = sample_output_info.get(key)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                sums[key] += value
                counts[key] += 1

    averages = {}
    for key in metrics_to_average:
        avg_key_name = f"avg_all_samples_{key.split('avg_')[-1]}"
        if counts[key] > 0:
            averages[avg_key_name] = sums[key] / counts[key]
        else:
            averages[avg_key_name] = None # No valid samples for this metric

    return averages

def process_evaluation_item(item):
    if not isinstance(item, dict):
        print("Warning: process_evaluation_item received non-dict input.")
        return item
    semantic_entropy, crash_rate = calculate_semantic_entropy(item)
    item['semantic_entropy'] = semantic_entropy
    item['crash_rate'] = crash_rate
    avg_confidences = calculate_average_confidences(item)
    item.update(avg_confidences)
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process evaluation result JSON files in a directory to add calculated metrics (semantic entropy, failure rates, avg confidences)."
    )
    parser.add_argument(
        "--input_dir",
        default="curriculum_selection/data/processed/taco/evaluate_results",
        help="Directory containing the per-problem JSON result files."
    )
    args = parser.parse_args()

    input_directory = args.input_dir

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found or is not a directory: {input_directory}")
        sys.exit(1)

    print(f"Processing JSON files in directory: {input_directory}")

    files_processed = 0
    files_failed = 0
    json_files = [f for f in os.listdir(input_directory) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the directory.")
        sys.exit(0)

    for filename in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(input_directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f_in:
                result_item_loaded = json.load(f_in)
            processed_item = process_evaluation_item(result_item_loaded)
            with open(file_path, 'w', encoding='utf-8') as f_out:
                json.dump(processed_item, f_out, indent=4, ensure_ascii=False)
            files_processed += 1

        except json.JSONDecodeError as json_err:
            files_failed += 1
        except IOError as io_err:
            files_failed += 1
        except Exception as e:
            files_failed += 1

    print(f"Total JSON files found: {len(json_files)}")
    print(f"Successfully processed and overwritten: {files_processed}")
    print(f"Failed/Skipped: {files_failed}")