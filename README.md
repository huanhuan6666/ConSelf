# Self-Improving Code Generation via Semantic Entropy and Behavioral Consensus

This artifact provides the necessary code and instructions to reproduce the key experiments of our paper. The workflow is divided into five main stages: Data Preparation, Candidate Generation, Evaluation & Metrics Calculation, Training Data Construction, and Model Fine-tuning & Evaluation.

## Stage 1: Data Preparation

This stage processes the raw TACO dataset.

1.  Download the original TACO dataset from [Hugging Face Datasets](https://huggingface.co/datasets/BAAI/TACO).
2.  Run the preprocessing script to clean the data:
    ```bash
    python curriculum_selection/data/data_process.py --input_path /path/to/raw/taco.jsonl --output_path curriculum_selection/data/taco/taco_cleaned.jsonl
    ```
    For convenience, we also provide the pre-processed version at `curriculum_selection/data/taco/taco_cleaned.jsonl`.

## Stage 2: Candidate Generation

This stage uses a base model to generate candidate programs for each problem via Observation-Guided Sampling.

-   **Action**: Run the `generate_code.py` script.
-   **Configuration**: Key parameters (e.g., number of observations, temperature) are defined in `curriculum_selection/config/default.yaml`.
-   **Command**:
    ```bash
    python curriculum_selection/generate_code.py \
        --dataset_path curriculum_selection/data/taco/taco_cleaned.jsonl \
        --local_model_path /path/to/your/deepseek-coder-6.7b-instruct # Replace with your model path
    ```
-   **Output**: A file named `taco_with_responses.jsonl` will be created in the corresponding processed data directory.

## Stage 3: Execution and Metric Calculation

This stage executes all generated candidates and computes the metrics required for our analysis, including Code Semantic Entropy.

-   **Action**: Run the `evaluation/main.py` script.
-   **Input**: The `taco_with_responses.jsonl` file from the previous stage.
-   **Command**:
    ```bash
    python curriculum_selection/evaluation/main.py \
        --input_file /path/to/taco_with_responses.jsonl \
        --chunk_size 1000 \
        --num_process_evaluate 16
    ```
-   **Output**: A directory named `evaluate_results` containing a JSON file for each problem, detailing execution outcomes and calculated metrics (semantic entropy, NLL, etc.).

> **Optional**: To reproduce the correlation analysis from Section 3.3 of our paper, run the following script. It reads from the `evaluate_results` directory.
> ```bash
> python curriculum_selection/analyze/analyze_confidence_correlation.py --input_dir /path/to/evaluate_results
> ```

## Stage 4: Training Data Construction (DPO Pairs)

This stage applies our entropy-based filtering to create the final training dataset of preference pairs.

-   **Action**: Run the `create_dpo_pairs.py` script.
-   **Key Parameter**: `--semantic_entropy_threshold` corresponds to the hyperparameter \(\tau\) in the paper.
-   **Command**:
    ```bash
    python curriculum_selection/create_dpo_pairs.py \
        --input_dir /path/to/evaluate_results \
        --output_file /path/to/dpo_pairs.json \
        --semantic_entropy_threshold 0.3
    ```
-   **Output**: A file named `dpo_pairs.json` ready for training.

## Stage 5: Fine-tuning and Evaluation

### 5.1. Setup LLaMA-Factory

Our Consensus-Driven DPO (Con-DPO) is implemented within the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. Please follow their official [README](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md) to install the library. Our implementation is located in the `train/` directory.

### 5.2. Configure Dataset

Add a new entry for our dataset in `train/data/dataset_info.json`. Ensure the `file_name` points to the `dpo_pairs.json` created in Stage 4.

```json
"code_dpo_pairs": {
    "file_name": "/path/to/your/dpo_pairs.json",
    ...
  }
```

### 5.3. Train the Model

Use the provided example YAML configuration files to start training. Remember to set `model_name_or_path` and `output_dir` in the YAML file correctly.

```bash
llamafactory-cli train examples/code_condpo_lora_xx.yaml
```

### 5.4. Merge LoRA Weights

After training, merge the LoRA adapter with the base model to get the final fine-tuned model. Update the `export_dir` in the YAML file to match the `output_dir` from the training step.

```bash
llamafactory-cli export examples/merge_lora/merge_code_dpo_lora_xx.yaml
```

### 5.5. Evaluate on Benchmarks

Finally, evaluate the merged model on standard benchmarks. Ensure you have [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) and [EvalPlus](https://github.com/evalplus/evalplus) installed.

-   **LiveCodeBench**:
    ```bash
    python lcb_runner/runner/main.py \
        --model the_model_name \
        --local_model_path /path/to/your/merged_model \
        --scenario codegeneration \
        --evaluate \
        --n 1 \
        --temperature 0.0
    ```
-   **EvalPlus (MBPP)**:
    ```bash
    evalplus.evaluate \
        --model /path/to/your/merged_model \
        --dataset mbpp \
        --greedy
    ```