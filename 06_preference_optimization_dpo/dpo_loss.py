"""Direct Preference Optimization (DPO) Loss from First Principles.

Reference: Rafailov et al. (Stanford, 2023) - Direct Preference Optimization:
Your Language Model is Secretly a Reward Model.

The Mathematical Breakthrough:
    Under the Bradley-Terry preference model, the ground-truth reward can be
    reparameterized directly via the optimal policy:
        r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)
        
    When computing the win probability between chosen y_w and rejected y_l:
        P(y_w > y_l | x) = sigma( r(x, y_w) - r(x, y_l) )
    The intractable partition function Z(x) CANCELS IDENTICALLY TO ZERO:
        r(y_w) - r(y_l) = beta * [ log(pi*(y_w)/pi_ref(y_w)) - log(pi*(y_l)/pi_ref(y_l)) ]

This eliminates the separate Reward Model and Critic Model entirely, reducing
GPU memory consumption from 4 simultaneous models (PPO) to just 1 base model with LoRA!
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """Computes the sum of log-probabilities for target completion tokens.

    Autoregressive shift:
        logits at index t predict token at index t+1.
        logits[:, :-1, :] aligns with labels[:, 1:].

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size)
        labels: Ground-truth token IDs of shape (batch, seq_len) with -100 on prompts
        label_pad_token_id: Token ID used to mask prompt tokens

    Returns:
        sum_log_probs: Sum of log probabilities per sequence, shape (batch,)
    """
    # 1. Shift for next-token prediction alignment
    shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len - 1, vocab_size)
    shift_labels = labels[:, 1:].contiguous()       # (batch, seq_len - 1)

    # 2. Compute log softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 3. Create loss mask: 1 for valid completion tokens, 0 for prompt tokens
    loss_mask = shift_labels != label_pad_token_id

    # 4. Clamp labels to 0 for invalid tokens so torch.gather does not crash
    clamped_labels = shift_labels.clone()
    clamped_labels[~loss_mask] = 0

    # 5. Gather log probabilities of the actual target tokens
    # (batch, seq_len - 1, 1)
    gathered_log_probs = torch.gather(
        log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 6. Zero out prompt positions and sum across sequence
    gathered_log_probs = gathered_log_probs * loss_mask.float()
    return gathered_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the Direct Preference Optimization (DPO) loss.

    Args:
        policy_chosen_logps: Log-probs of chosen completions under active policy pi_theta
        policy_rejected_logps: Log-probs of rejected completions under active policy pi_theta
        reference_chosen_logps: Log-probs of chosen completions under frozen ref pi_ref
        reference_rejected_logps: Log-probs of rejected completions under frozen ref pi_ref
        beta: KL-divergence penalty hyperparameter (typical range: 0.05 to 0.5)
        label_smoothing: Optional conservative label smoothing factor (default: 0.0)

    Returns:
        loss: Scalar mean DPO loss
        chosen_rewards: Implicit rewards for chosen responses (batch,)
        rejected_rewards: Implicit rewards for rejected responses (batch,)
        reward_margin: Margin between chosen and rejected rewards (batch,)
    """
    # 1. Compute policy-to-reference log ratio (Implicit Reward)
    # r_hat(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # 2. Reward margin: r_hat(y_w) - r_hat(y_l)
    reward_margin = chosen_rewards - rejected_rewards

    # 3. Bradley-Terry Negative Log-Likelihood Loss:
    # L_DPO = -E [ log sigma( r_hat(y_w) - r_hat(y_l) ) ]
    if label_smoothing == 0.0:
        loss = -F.logsigmoid(reward_margin)
    else:
        # Conservative DPO with label smoothing (c.f. Azar et al. 2023)
        loss = -(
            (1.0 - label_smoothing) * F.logsigmoid(reward_margin)
            + label_smoothing * F.logsigmoid(-reward_margin)
        )

    return loss.mean(), chosen_rewards.detach(), rejected_rewards.detach(), reward_margin.detach()


class DPOTrainer(nn.Module):
    """Modular DPO loss wrapper module."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
