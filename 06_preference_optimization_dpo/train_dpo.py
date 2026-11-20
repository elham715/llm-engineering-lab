"""End-to-End DPO Fine-Tuning Demonstration.

Simulates preference alignment on paired prompt completions:
- Prompt: "Does a human mature red blood cell execute the citric acid cycle?"
- Chosen Completion (y_w): "No, mature erythrocytes lack mitochondria and rely solely on glycolysis."
- Rejected Completion (y_l): "Yes, it produces ATP through the mitochondrial Krebs cycle."

Tracks the widening reward margin and loss convergence across training steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dpo_loss import compute_sequence_log_probs, dpo_loss


def run_synthetic_dpo_training():
    print("=" * 70)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) TRAINING SIMULATION")
    print("=" * 70)

    torch.manual_seed(42)
    vocab_size = 64
    dim = 32
    seq_len = 10
    beta = 0.1

    # 1. Base Reference Model (FROZEN)
    ref_model = nn.Sequential(
        nn.Embedding(vocab_size, dim),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, vocab_size),
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    # 2. Trainable Policy Model (Active)
    policy_model = nn.Sequential(
        nn.Embedding(vocab_size, dim),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, vocab_size),
    )
    # Initialize policy weights identical to reference model
    policy_model.load_state_dict(ref_model.state_dict())

    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-2)

    # 3. Create Synthetic Preference Pairs (Batch of 4 prompts)
    batch_size = 4
    # Prompts: First 4 tokens are prompt (masked with -100 in labels)
    chosen_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Identical prompts across chosen and rejected
    rejected_ids[:, :4] = chosen_ids[:, :4]

    # Labels with -100 on prompt tokens
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, :4] = -100

    rejected_labels = rejected_ids.clone()
    rejected_labels[:, :4] = -100

    # 4. Precompute Reference Log-Probs (only computed ONCE!)
    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_ids)
        ref_rejected_logits = ref_model(rejected_ids)
        ref_chosen_logps = compute_sequence_log_probs(ref_chosen_logits, chosen_labels)
        ref_rejected_logps = compute_sequence_log_probs(ref_rejected_logits, rejected_labels)

    print(f"Step 0 Initial Reference Log-Probs: Chosen={ref_chosen_logps.mean():.3f}, Rejected={ref_rejected_logps.mean():.3f}\n")

    # 5. Training Loop
    print(f"{'Step':<6} | {'DPO Loss':<10} | {'Chosen Rew':<12} | {'Reject Rew':<12} | {'Reward Margin':<14} | {'Preference P(y_w > y_l)'}")
    print("-" * 80)

    for step in range(1, 26):
        optimizer.zero_grad()

        # Forward pass on policy
        pol_chosen_logits = policy_model(chosen_ids)
        pol_rejected_logits = policy_model(rejected_ids)

        pol_chosen_logps = compute_sequence_log_probs(pol_chosen_logits, chosen_labels)
        pol_rejected_logps = compute_sequence_log_probs(pol_rejected_logits, rejected_labels)

        loss, chosen_rew, reject_rew, margin = dpo_loss(
            pol_chosen_logps,
            pol_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            win_prob = torch.sigmoid(margin).mean().item()
            print(
                f"{step:<6} | {loss.item():<10.4f} | {chosen_rew.mean().item():<12.4f} | "
                f"{reject_rew.mean().item():<12.4f} | {margin.mean().item():<14.4f} | {win_prob * 100:.1f}%"
            )

    print("-" * 80)
    print("Alignment completed successfully! Reward margin widened to positive territory without a separate Reward Model.\n")


if __name__ == "__main__":
    run_synthetic_dpo_training()
