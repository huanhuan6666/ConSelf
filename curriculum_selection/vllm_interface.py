import numpy as np
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams


class VLLMRunner():
    def __init__(self, args, model_name: str=""):
        self.args = args
        model_tokenizer_path = (
            model_name if args['local_model_path'] is None else args['local_model_path']
        )
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        print(f"Available GPUs: {available_gpus}")
        self.llm = LLM(
            model=model_tokenizer_path,
            tokenizer=model_tokenizer_path,
            tensor_parallel_size=len(available_gpus),
            dtype=args['dtype'],
            enforce_eager=True,
            gpu_memory_utilization=0.95,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args['enable_prefix_caching'],
            trust_remote_code=args['trust_remote_code'],
        )

    def run_batch(
        self, prompts: list[str],
        **kwargs
    ) -> list[list[dict]]:
        """
        Returns:
            [
                [{"text": "...", "avg_nll": 0.5, "avg_token_entropy": 1.5}, ...],
                [{"text": "...", "avg_nll": 0.6, "avg_token_entropy": 1.8}, ...]
            ]
        """
        n = kwargs.get("n", self.args['n'])
        temperature = kwargs.get("temperature", self.args['temperature'])
        max_tokens = kwargs.get("max_tokens", self.args['max_tokens'])
        top_p = kwargs.get("top_p", self.args['top_p'])
        calculate_confidence = kwargs.get("calculate_confidence", True)
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>", "<|endoftext|>"],
            logprobs=20
        )
        print(f"Running generation for {len(prompts)} prompts...")
        vllm_outputs = self.llm.generate(prompts, sampling_params)
        print("Generation complete. Processing outputs...")
        final_results = []
        for i, vllm_output in enumerate(tqdm(vllm_outputs, desc="Calculating Confidence")):
            current_prompt_results = []
            for completion_output in vllm_output.outputs:
                full_text = completion_output.text
                if not calculate_confidence:
                    current_prompt_results.append(
                        {
                            "text": full_text,
                        }
                    )
                    continue
                token_ids = completion_output.token_ids
                logprobs_per_step = completion_output.logprobs
                confidence_metrics = self._calculate_confidence_metrics(
                    token_ids, logprobs_per_step, k=self.args['confidence_top_k']
                )
                current_prompt_results.append(
                    {
                        "text": full_text,
                        "avg_nll": confidence_metrics["avg_nll"],
                        "avg_token_entropy": confidence_metrics["avg_token_entropy"],
                    }
                )
            final_results.append(current_prompt_results)
        print("Metrics calculation complete.")
        return final_results


    def _calculate_confidence_metrics(
        self,
        token_ids: list[int],
        logprobs_per_step: list[dict[int, object] | None] | None,
        k: int
    ) -> dict:
        """
        Calculate confidence metrics for a single generation sequence.
        Args:
            token_ids: List of token IDs for the generated sequence.
            logprobs_per_step: List of logprobs dictionaries for each step (vLLM format).
                                [{token_id: logprob}, {token_id: logprob}, ...]
                                Need to request logprobs=k to get.
            k: k value for token-level entropy calculation.
        Returns:
            {
                "avg_nll": float | None,
                "avg_token_entropy": float | None
            }
        """
        avg_nll = None
        avg_token_entropy = None
        if not logprobs_per_step or k < 1:
            return {"avg_nll": None, "avg_token_entropy": None}
        token_nlls, step_top_k_entropies = [], []
        for i, step_logprobs_dict in enumerate(logprobs_per_step):
            if not step_logprobs_dict or i >= len(token_ids):
                continue
            generated_token_id = token_ids[i]
            # Calculate NLL
            if generated_token_id in step_logprobs_dict:
                token_nlls.append(-step_logprobs_dict[generated_token_id].logprob)
            else:
                token_nlls.append(None)
            # Token-level Entropy
            sorted_logprobs = sorted(
                step_logprobs_dict.items(), 
                key=lambda item: item[1].logprob if hasattr(item[1], 'logprob') else -float('inf'),
                reverse=True
            )
            top_k_logprobs = [lp.logprob for _, lp in sorted_logprobs[:k]]
            if top_k_logprobs:
                top_k_probs = np.exp(np.array(top_k_logprobs, dtype=np.float64))
                sum_top_k_probs = np.sum(top_k_probs)
                if sum_top_k_probs > 1e-9:
                    normalized_top_k_probs = top_k_probs / sum_top_k_probs
                    entropy = -np.sum(normalized_top_k_probs * np.log2(normalized_top_k_probs + 1e-9))
                    step_top_k_entropies.append(max(0.0, entropy))
                else:
                    step_top_k_entropies.append(None)
        if token_nlls:
            valid_nlls = [n for n in token_nlls if n is not None]
            if valid_nlls: avg_nll = np.mean(valid_nlls)
        if step_top_k_entropies:
            valid_entropies = [e for e in step_top_k_entropies if e is not None]
            if valid_entropies: avg_token_entropy = np.mean(valid_entropies)
        return {
            "avg_nll": avg_nll,
            "avg_token_entropy": avg_token_entropy,
        }
