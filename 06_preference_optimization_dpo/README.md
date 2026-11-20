# Direct Preference Optimization (DPO) From Scratch

Implementation of Direct Preference Optimization (Rafailov et al., Stanford 2023) without HuggingFace TRL or external alignment libraries.

---

## 1. Why DPO Replaced PPO / Classical RLHF

In standard RLHF (InstructGPT, PPO):
1. Train a Policy $\pi_\theta$.
2. Train a separate Reward Model $r_\psi$ from human pairwise rankings.
3. Train a Critic Model $V_\phi$ for baseline value estimation.
4. Keep a frozen Reference Model $\pi_{\text{ref}}$ to prevent mode collapse.
$$\implies \mathbf{4\text{ models simultaneously in GPU memory! } (250+\text{ GB VRAM})}$$

### The DPO Closed-Form Inversion:
Rafailov et al. proved mathematically:
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

When computing the preference probability $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$, the partition function $\beta \log Z(x)$ **cancels identically to zero**!

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right] \right) \right]$$

---

## 2. Running Unit Tests and Training Simulation

```bash
# Run unit tests
/opt/anaconda3/bin/python test_dpo.py

# Run self-contained synthetic training loop
/opt/anaconda3/bin/python train_dpo.py
```
- Passes all 5 unit tests verifying step-0 $\ln(2)$ boundary, monotonic optimization, dynamic gradient difficulty, and prompt masking.
- `train_dpo.py` shows the reward margin opening from $0.00$ to $+1.50$ with win probability rising to $>80\%$.
