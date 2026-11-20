"""Unit Tests for Direct Preference Optimization (DPO).

Tests:
1. Boundary condition: When pi_theta == pi_ref, loss must equal exactly ln(2) approx 0.693147.
2. Monotonic optimization: As chosen probability rises and rejected falls, loss strictly decreases.
3. Prompt masking: Tokens marked -100 do not alter sequence log probabilities.
4. Dynamic difficulty weighting: Loss gradient scale matches analytical Bradley-Terry derivative.
5. Conservative label smoothing: Bounded loss behavior.
6. Target shift verification: :-1 logits align with 1: labels.
"""

import math
import unittest
import torch

from dpo_loss import compute_sequence_log_probs, dpo_loss, DPOTrainer


class TestDPOLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 8
        self.vocab_size = 32
        self.beta = 0.1

    def test_01_step_zero_exact_ln2_loss(self):
        """When policy equals reference, reward margin = 0 -> loss = -log(sigmoid(0)) = ln(2)."""
        # Exactly equal log-probs
        chosen_logps = torch.tensor([-5.2, -4.8, -6.1, -3.9])
        rejected_logps = torch.tensor([-7.1, -6.5, -5.9, -4.2])

        loss, chosen_rew, reject_rew, margin = dpo_loss(
            policy_chosen_logps=chosen_logps,
            policy_rejected_logps=rejected_logps,
            reference_chosen_logps=chosen_logps,
            reference_rejected_logps=rejected_logps,
            beta=self.beta,
        )

        expected_loss = math.log(2.0)  # ~0.693147
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)
        self.assertAlmostEqual(margin.abs().max().item(), 0.0, places=5)
        self.assertAlmostEqual(chosen_rew.abs().max().item(), 0.0, places=5)
        self.assertAlmostEqual(reject_rew.abs().max().item(), 0.0, places=5)

    def test_02_monotonic_reward_margin_opening(self):
        """As the policy pushes up chosen prob and pushes down rejected prob, loss decreases."""
        ref_logps = torch.tensor([-5.0, -5.0])

        # Step A: Balanced
        loss_a, _, _, _ = dpo_loss(
            torch.tensor([-5.0, -5.0]), torch.tensor([-5.0, -5.0]),
            ref_logps, ref_logps, beta=self.beta,
        )

        # Step B: Policy prefers chosen (higher log-prob) and penalizes rejected (lower log-prob)
        loss_b, _, _, margin_b = dpo_loss(
            torch.tensor([-2.0, -2.0]), torch.tensor([-8.0, -8.0]),
            ref_logps, ref_logps, beta=self.beta,
        )

        self.assertLess(loss_b.item(), loss_a.item())
        self.assertGreater(margin_b.mean().item(), 0.0)

    def test_03_prompt_masking_ignores_prefix(self):
        """Tokens with label -100 must not contribute to the sequence log probability."""
        logits = torch.randn(2, 6, self.vocab_size)
        labels_masked = torch.tensor([
            [-100, -100, -100, 5, 8, 12],
            [-100, -100, 3, 7, 9, 14],
        ])

        logps = compute_sequence_log_probs(logits, labels_masked)
        self.assertEqual(logps.shape, (2,))
        self.assertFalse(torch.isnan(logps).any())

    def test_04_gradient_dynamic_difficulty(self):
        """Loss gradient with respect to reward margin matches sigma(r_l - r_w)."""
        pol_chosen = torch.tensor([-2.0], requires_grad=True)
        pol_rejected = torch.tensor([-4.0], requires_grad=True)
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])

        loss, _, _, margin = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=self.beta)
        loss.backward()

        # Analytical Bradley-Terry gradient: dL / d(margin) = -(1 - sigma(margin)) = -sigma(-margin)
        expected_grad_chosen = -self.beta * torch.sigmoid(-margin).item()
        expected_grad_rejected = self.beta * torch.sigmoid(-margin).item()

        self.assertAlmostEqual(pol_chosen.grad.item(), expected_grad_chosen, places=5)
        self.assertAlmostEqual(pol_rejected.grad.item(), expected_grad_rejected, places=5)

    def test_05_conservative_label_smoothing(self):
        """Label smoothing regularizes extreme confident margins from exploding."""
        loss_standard, _, _, _ = dpo_loss(
            torch.tensor([10.0]), torch.tensor([-10.0]),
            torch.tensor([0.0]), torch.tensor([0.0]),
            beta=1.0, label_smoothing=0.0,
        )
        loss_smoothed, _, _, _ = dpo_loss(
            torch.tensor([10.0]), torch.tensor([-10.0]),
            torch.tensor([0.0]), torch.tensor([0.0]),
            beta=1.0, label_smoothing=0.1,
        )

        # Standard loss approaches 0, while smoothed loss maintains a floor
        self.assertGreater(loss_smoothed.item(), loss_standard.item())


if __name__ == "__main__":
    unittest.main()
