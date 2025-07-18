import json
import os
import argparse
import math
import re
from collections import Counter
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re

sys.set_int_max_str_digits(10**6)
def extract_base_metrics(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
        metrics = {
            'filepath': file_path,
            'question_id': item.get('question_id', os.path.basename(file_path)),
            'semantic_entropy': item.get('semantic_entropy', np.nan),
            'avg_all_samples_nll': item.get('avg_all_samples_nll', np.nan),
            'crash_rate': item.get('crash_rate', np.nan)
        }
        for key in ['semantic_entropy', 'avg_all_samples_nll', 'crash_rate']:
            metrics[key] = pd.to_numeric(metrics[key], errors='coerce')
        return metrics
    except Exception as e:
        return None

def clean_surrogates(obj):
    if isinstance(obj, str):
        return re.sub(r'[\ud800-\udfff]', '', obj)
    elif isinstance(obj, list):
        return [clean_surrogates(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: clean_surrogates(v) for k, v in obj.items()}
    else:
        return obj

def process_problem_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
    except Exception as e:
        print(f"Error decoding JSON in {os.path.basename(file_path)}: {e}", file=sys.stderr)
        return None
    if not all(k in item for k in ["metadata", "output_list", "question"]):
        print(f"Warning: Missing required keys (metadata, output_list, question) in {os.path.basename(file_path)}. Skipping.", file=sys.stderr)
        return None
    crash_rate = item.get("crash_rate", 0)
    if crash_rate >= 0.8:
        return None
    semantic_entropy = item.get("semantic_entropy", 0)
    if semantic_entropy == 0:
        return None
    instruction = item.get("question", "")
    metadata = item.get("metadata", {})
    output_list = item.get("output_list", [])
    graded_list = item.get("graded_list", [])

    if not isinstance(metadata, dict) or \
       "input_output" not in metadata or \
       "generation_results" not in metadata or \
       not isinstance(output_list, list) or not output_list or \
       not instruction:
        return None
    try:
        if not isinstance(metadata["input_output"], list) or len(metadata["input_output"]) != 1:
            return None
        io_string = metadata["input_output"][0]
        io_data = json.loads(json.loads(io_string))
        if "inputs" not in io_data:
            return None
        problem_inputs = io_data["inputs"]
        actual_outputs = io_data["outputs"]
        num_test_cases = len(problem_inputs)
    except (json.JSONDecodeError, TypeError, IndexError, KeyError) as e:
        return None
    try:
        generation_results_str_list = metadata["generation_results"]
        if not isinstance(generation_results_str_list, list):
            return None
        parsed_generation_results = []
        crash_code_ids = []
        for k, gen_result_str in enumerate(generation_results_str_list):
            try:
                parsed_results = json.loads(gen_result_str)
            except json.JSONDecodeError:
                parsed_results = [{"error_code": -5, "error_message": "JSON Parsing Error"}] * num_test_cases
                crash_code_ids.append(k)
            if not isinstance(parsed_results, list):
                if isinstance(parsed_results, dict) and "error_code" in parsed_results:
                    parsed_results = [parsed_results] * num_test_cases
                    crash_code_ids.append(k)
                else:
                    return None
            if len(parsed_results)==1 and len(parsed_results) != num_test_cases:
                crash_code_ids.append(k)
                parsed_results = parsed_results * num_test_cases
            assert len(parsed_results) == num_test_cases
            parsed_generation_results.append(parsed_results)

        num_codes = len(parsed_generation_results)
        if num_codes == 0:
            return None
        if len(output_list) != num_codes:
            return None
    except (json.JSONDecodeError, TypeError) as e:
        return None

    crash_error_codes = {-1, -3, -4, -5}
    avg_output_confidence = [0.0] * num_codes
    for j in range(num_test_cases):
        outputs_for_input_j = []
        for k in range(num_codes):
            try:
                test_result_kj = parsed_generation_results[k][j]
                if isinstance(test_result_kj, dict) and \
                   test_result_kj.get("error_code") not in crash_error_codes and \
                   "output" in test_result_kj:
                    output_val = test_result_kj["output"]
                    if isinstance(output_val, (list, dict)): 
                        hashable_output = json.dumps(output_val, sort_keys=True)
                    else:
                        hashable_output = output_val
                    if isinstance(hashable_output, str):
                        hashable_output = hashable_output.strip()
                    outputs_for_input_j.append((hashable_output, k)) 
            except (IndexError, KeyError, TypeError):
                continue
        if not outputs_for_input_j:
            continue

        output_counts = Counter(out for out, k_idx in outputs_for_input_j)
        for k in range(num_codes):
            for out, k_idx in outputs_for_input_j:
                if k_idx == k:
                    avg_output_confidence[k] += output_counts.get(out, 0) / len(outputs_for_input_j)
    for k in range(num_codes):
        avg_output_confidence[k] /= num_test_cases
    rewards = avg_output_confidence 
    valid_rewards = [(r, i) for i, r in enumerate(rewards) if isinstance(r, (int, float)) and not math.isnan(r) and not math.isinf(r)]
    valid_rewards.sort(key=lambda x: x[0], reverse=True)
    max_reward = valid_rewards[0][0]
    chosen_indices = [idx for r, idx in valid_rewards if r == max_reward]
    non_crash_indices = [idx for r, idx in valid_rewards if idx not in crash_code_ids]
    dpo_pairs = []
    meta_data_list = []
    for chosen_idx in chosen_indices:
        chosen_reward = rewards[chosen_idx]
        potential_rejected = []
        for idx in non_crash_indices:
            if idx != chosen_idx and rewards[idx] < chosen_reward:
                potential_rejected.append(idx)
        if not potential_rejected:
            potential_rejected = [idx for idx in range(num_codes) 
                                 if idx != chosen_idx and rewards[idx] < chosen_reward]
        if not potential_rejected:
            continue
        potential_rejected.sort(key=lambda x: rewards[x])
        for rejected_idx in potential_rejected:
            try:
                chosen_response = output_list[chosen_idx].get("text", "")
                rejected_response = output_list[rejected_idx].get("text", "")
                chosen_output_confidence = avg_output_confidence[chosen_idx]
                rejected_output_confidence = avg_output_confidence[rejected_idx]
                chosen_grade = graded_list[chosen_idx] if chosen_idx < len(graded_list) else None
                rejected_grade = graded_list[rejected_idx] if rejected_idx < len(graded_list) else None
                starter_code = item.get("starter_code", "")
                question_format = f"""You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nQuestion: {instruction}\n\n"""
                if starter_code:
                    question_format += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n ```python\n{starter_code}\n```\n\n"
                else:
                    question_format += "Ensure that when the python program runs, it reads the inputs from stdin, runs the algorithm and print the output to stdout(do not directly test on the sample inputs).\n ```python\n# YOUR CODE HERE\n```\n\n"
                question_format += "You will NOT return anything except for the program.\n\n"
                dpo_pair = {
                    "problem_id": item.get("question_id"),
                    "instruction": question_format,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "chosen_output_confidence": chosen_output_confidence
                }
                meta_data = {
                    "problem_id": item.get("question_id"),
                    "chosen_grade": chosen_grade,
                    "rejected_grade": rejected_grade,
                    "chosen_reward": chosen_output_confidence,
                    "rejected_reward": rejected_output_confidence,
                    "chosen_output_confidence": chosen_output_confidence,
                    "rejected_output_confidence": rejected_output_confidence,
                }
                dpo_pairs.append(dpo_pair)
                meta_data_list.append(meta_data)
            except (IndexError, KeyError, TypeError) as e:
                print(f"Error extracting final code text for pair ({chosen_idx}, {rejected_idx}) in {os.path.basename(file_path)}: {e}.", file=sys.stderr)
                continue
    if not dpo_pairs:
        return None
    return dpo_pairs, meta_data_list


def analyze_threshold_impact(metrics_df_with_scores, args, output_dir):
    """
    Analyzes and plots the impact of varying fixed semantic_entropy thresholds
    on the number of DPO pairs and the oracle rate.

    Args:
        metrics_df_with_scores (pd.DataFrame): DataFrame containing 'filepath' and 'semantic_entropy'.
        args (argparse.Namespace): Parsed command-line arguments.
        output_dir (str): Directory to save the plot and summary table.
    """
    thresholds_to_test = [round(x, 2) for x in np.arange(1.0, 0.0, -0.1)]
    results = []

    if 'semantic_entropy' not in metrics_df_with_scores.columns:
        return
    if metrics_df_with_scores['semantic_entropy'].isnull().all():
        return
    for threshold in tqdm(thresholds_to_test):
        score_threshold = threshold
        num_pairs = 0
        correct_pairs = 0
        oracle_rate = 0.0

        selected_df = metrics_df_with_scores[
            metrics_df_with_scores['semantic_entropy'].notna() &
            (metrics_df_with_scores['semantic_entropy'] <= score_threshold)
        ]
        if not selected_df.empty:
            temp_dpo_pairs_count = 0
            temp_correct_pairs = 0
            for _, row in selected_df.iterrows():
                file_path = row['filepath']
                combined_score = row.get('semantic_entropy')
                result = process_problem_file(file_path)
                if result:
                    result = list(result)
                    result[0] = result[0][:3]
                    result[1] = result[1][:3]
                    dpo_pairs_from_file, meta_data_list_from_file = result
                    temp_dpo_pairs_count += len(dpo_pairs_from_file)
                    for meta_item in meta_data_list_from_file:
                        meta_item['semantic_entropy'] = combined_score
                    temp_correct_pairs += sum(1 for meta in meta_data_list_from_file if meta.get('chosen_grade') is True and meta.get('rejected_grade') is False)
            num_pairs = temp_dpo_pairs_count
            correct_pairs = temp_correct_pairs
            if num_pairs > 0:
                oracle_rate = correct_pairs / num_pairs
        results.append({
            "threshold": threshold,
            "num_pairs": num_pairs,
            "oracle_rate": oracle_rate,
            "oracle_count": correct_pairs,
        })
    if not results:
        return
    print("raw results: \n")
    print(results)
    df_results = pd.DataFrame(results).dropna()
    df_results = df_results.astype({
        "threshold": float,
        "num_pairs": int,
        "oracle_rate": float,
        "oracle_count": int
    })

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1_pairs = 'tab:blue'
    color1_oracle_count = 'tab:purple'
    ax1.set_xlabel('Semantic Entropy Score Threshold')
    ax1.set_ylabel('Number of DPO Pairs / Oracle Count', color=color1_pairs)
    ax1.plot(df_results['threshold'], df_results['num_pairs'], color=color1_pairs, marker='o', label='Num DPO Pairs')
    ax1.plot(df_results['threshold'], df_results['oracle_count'], color=color1_oracle_count, marker='d', linestyle=':', label='Oracle Count')
    ax1.tick_params(axis='y', labelcolor=color1_pairs)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    color2_oracle_rate = 'tab:green'
    ax2.set_ylabel('Oracle Rate (Chosen=T, Rejected=F)', color=color2_oracle_rate)
    ax2.plot(df_results['threshold'], df_results['oracle_rate'], color=color2_oracle_rate, marker='s', linestyle='--', label='Oracle Rate')
    ax2.tick_params(axis='y', labelcolor=color2_oracle_rate)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    max_rate = df_results['oracle_rate'].max() if not df_results['oracle_rate'].empty else 0.1
    ax2.set_ylim(bottom=0, top=min(max(max_rate * 1.1, 0.1), 1.05))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.title('Impact of Semantic Entropy Threshold on DPO Pair Quality')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "threshold_vs_pairs_oracle_rate.png")
    try:
        plt.savefig(plot_path)
    except Exception as e:
        print(f"Save picture Error: {e}")
    plt.close(fig)

    df_print = df_results.copy()
    df_print.rename(columns={
        "threshold": "Threshold",
        "num_pairs": "Num DPO Pairs",
        "oracle_rate": "Oracle Rate (%)",
        "oracle_count": "Oracle Count"
    }, inplace=True)
    df_print['Oracle Rate (%)'] = df_print['Oracle Rate (%)'].apply(lambda x: f"{x:.2%}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_print)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DPO pairs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the per-problem JSON evaluation result files."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save the generated DPO pairs JSON file."
    )
    parser.add_argument(
        "--semantic_entropy_threshold",
        type=float,
        default=0.3,
        help="Threshold for semantic entropy filtering. Only problems with semantic_entropy < threshold will be selected."
    )
    args = parser.parse_args()
    output_parent_dir = os.path.dirname(args.output_file) 
    os.makedirs(output_parent_dir, exist_ok=True)
    all_problem_metrics = []
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    if not json_files:
        sys.exit(0)
    for filename in tqdm(json_files):
        file_path = os.path.join(args.input_dir, filename)
        metrics = extract_base_metrics(file_path)
        if metrics:
            all_problem_metrics.append(metrics)
    if not all_problem_metrics:
        sys.exit(1)
    metrics_df = pd.DataFrame(all_problem_metrics)
    initial_count = len(metrics_df)
    required_metrics = ['semantic_entropy', 'crash_rate']
    metrics_df.dropna(subset=required_metrics, inplace=True)
    initial_count = len(metrics_df)

    metrics_df = metrics_df[metrics_df['crash_rate'] < 0.8]
    initial_count = len(metrics_df)
    metrics_df = metrics_df[metrics_df['semantic_entropy'] != 0]
    initial_count = len(metrics_df)
    metrics_df['semantic_entropy_raw'] = metrics_df['semantic_entropy'].copy()
    scaler = MinMaxScaler()
    metrics_df['semantic_entropy'] = scaler.fit_transform(metrics_df[['semantic_entropy']])
    
    # analyze_threshold_impact(metrics_df, args, output_parent_dir)

    selected_df = metrics_df[metrics_df['semantic_entropy'] < args.semantic_entropy_threshold]
    if selected_df.empty:
        sys.exit(1)

    dpo_pairs = []
    oracle_pairs = []
    meta_data_list = []
    correct_dpo_pairs = 0
    chosen_false_rejected_true_count = 0
    chosen_true_rejected_true_count = 0
    chosen_false_rejected_false_count = 0

    for index, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Generate DPO pairs"):
        file_path = row['filepath']
        semantic_entropy = row.get('semantic_entropy', None)
        result = process_problem_file(file_path)
        if result:
            dpo_pairs_from_file, meta_data_list_from_file = result
            for meta_item in meta_data_list_from_file:
                meta_item['semantic_entropy'] = semantic_entropy
            for dpo_pair in dpo_pairs_from_file:
                dpo_pair['1-semantic_entropy'] = 1-semantic_entropy
            dpo_pairs_from_file = dpo_pairs_from_file[:3]
            meta_data_list_from_file = meta_data_list_from_file[:3]
            dpo_pairs.extend(dpo_pairs_from_file)
            meta_data_list.extend(meta_data_list_from_file)
    if dpo_pairs:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            json.dump(dpo_pairs, f_out, ensure_ascii=False, indent=4)
        meta_output_file = args.output_file.replace('.json', '_meta.json')
        meta_data_list_clean = clean_surrogates(meta_data_list)
        with open(meta_output_file, 'w', encoding='utf-8') as f_out:
            json.dump(meta_data_list_clean, f_out, ensure_ascii=False, indent=4)
        oracle_output_file = args.output_file.replace('.json', '_oracle.json')
        with open(oracle_output_file, 'w', encoding='utf-8') as f_out:
            json.dump(oracle_pairs, f_out, ensure_ascii=False, indent=4)
