"""Speculative Decoding from First Principles.

References:
- Leviathan et al. (Google Research, 2023) - Fast Inference from Transformers via Speculative Decoding.
- Chen et al. (DeepMind, 2023) - Accelerating Large Language Model Decoding with Speculative Sampling.

Hardware Principle:
    Target Model (e.g. 70B parameters) generates tokens one-by-one.
    Each token requires loading ALL 70B weights from VRAM to SRAM -> MEMORY BANDWIDTH SATURATION (GEMV).
    
    Speculative Decoding Solution:
    1. A small, fast Draft Model (e.g. 1B) generates K candidate tokens quickly.
    2. The large Target Model processes all K tokens in a SINGLE parallel forward pass (GEMM mode!).
    3. Rejection sampling tests each candidate:
           Accept if U ~ Uniform(0, 1) < min(1, P_target(x) / P_draft(x))
    4. If a token is rejected, sample from residual distribution max(0, P_target - P_draft) and stop.

Theoretical Guarantee:
    The output distribution is MATHEMATICALLY IDENTICAL to sampling directly from the Target Model alone!
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def rejection_sampling_step(
    p_draft: torch.Tensor,
    p_target: torch.Tensor,
    draft_token: int,
) -> Tuple[bool, Optional[int]]:
    """Applies exact rejection sampling condition for a single speculative token.

    Args:
        p_draft: Probability distribution over vocabulary from draft model, shape (vocab_size,)
        p_target: Probability distribution over vocabulary from target model, shape (vocab_size,)
        draft_token: Token proposed by draft model

    Returns:
        accepted: Boolean flag indicating if draft token was accepted
        resampled_token: If rejected, the newly sampled replacement token; else None
    """
    p_d = p_draft[draft_token].item()
    p_t = p_target[draft_token].item()

    # Acceptance threshold: min(1, p_target / p_draft)
    acceptance_prob = min(1.0, (p_t / max(p_d, 1e-10)))
    rand_val = torch.rand(1).item()

    if rand_val < acceptance_prob:
        return True, None
    else:
        # Rejection recovery: Sample from adjusted distribution max(0, P_target - P_draft)
        residual = torch.clamp(p_target - p_draft, min=0.0)
        norm_factor = residual.sum()

        if norm_factor > 0:
            adjusted_p = residual / norm_factor
        else:
            adjusted_p = p_target

        resampled_token = torch.multinomial(adjusted_p, num_samples=1).item()
        return False, resampled_token


def speculative_decoding_loop(
    draft_model: nn.Module,
    target_model: nn.Module,
    input_ids: torch.Tensor,
    gamma: int = 4,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Executes the complete speculative decoding algorithm.

    Args:
        draft_model: Small fast autoregressive model
        target_model: Large high-fidelity target model
        input_ids: Prompt token tensor of shape (1, seq_len)
        gamma: Speculation lookahead budget (number of draft tokens per iteration)
        max_new_tokens: Total tokens to generate
        temperature: Softmax sampling temperature

    Returns:
        generated_ids: Tensor containing prompt + generated tokens
        stats: Dictionary containing acceptance rate and target forward pass count
    """
    curr_ids = input_ids.clone()
    total_accepted_tokens = 0
    total_draft_tokens = 0
    target_forward_passes = 0

    while curr_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
        # Step 1: Draft model generates gamma candidate tokens autoregressively
        draft_tokens = []
        draft_probs = []

        draft_context = curr_ids.clone()
        for _ in range(gamma):
            with torch.no_grad():
                d_logits = draft_model(draft_context)
                d_probs = F.softmax(d_logits[:, -1, :] / temperature, dim=-1).squeeze(0)
                next_token = torch.multinomial(d_probs, num_samples=1).item()

            draft_tokens.append(next_token)
            draft_probs.append(d_probs)
            draft_context = torch.cat([draft_context, torch.tensor([[next_token]], device=curr_ids.device)], dim=1)

        # Step 2: Target model evaluates the prompt AND all gamma candidate tokens in ONE parallel forward pass!
        # (This is GEMM computation, fully saturating GPU Tensor Cores!)
        target_forward_passes += 1
        with torch.no_grad():
            t_logits = target_model(draft_context)
            # Logits for each position: curr_len - 1 up to curr_len + gamma - 1
            start_idx = curr_ids.shape[1] - 1
            t_probs_all = F.softmax(t_logits[:, start_idx:, :] / temperature, dim=-1).squeeze(0)

        # Step 3: Sequential rejection sampling
        accepted_this_round = 0
        all_accepted = True

        for i in range(gamma):
            total_draft_tokens += 1
            p_d = draft_probs[i]
            p_t = t_probs_all[i]
            candidate = draft_tokens[i]

            accepted, resampled_token = rejection_sampling_step(p_d, p_t, candidate)

            if accepted:
                curr_ids = torch.cat([curr_ids, torch.tensor([[candidate]], device=curr_ids.device)], dim=1)
                accepted_this_round += 1
                total_accepted_tokens += 1
            else:
                # Add corrected token and abort remaining draft tokens
                curr_ids = torch.cat([curr_ids, torch.tensor([[resampled_token]], device=curr_ids.device)], dim=1)
                all_accepted = False
                break

        # If all gamma tokens were accepted, sample one bonus token from target's final logits
        if all_accepted:
            bonus_token = torch.multinomial(t_probs_all[-1], num_samples=1).item()
            curr_ids = torch.cat([curr_ids, torch.tensor([[bonus_token]], device=curr_ids.device)], dim=1)
            total_accepted_tokens += 1

    stats = {
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "acceptance_rate": (total_accepted_tokens / max(total_draft_tokens, 1)),
        "target_forward_passes": target_forward_passes,
    }
    return curr_ids, stats
