# Project 09: NeuroMet-SLM Multi-Expert Engine & Research Paper

> **Flagship Capstone & Scientific Publication Track**  
> *Target Venue: Nature Machine Intelligence / Bioinformatics / NeurIPS AI for Science*  
> *Hardware Constraint: Strictly < 1.0 GB RAM / Consumer Edge Deployment*

---

## 1. Executive Summary

**NeuroMet-SLM** is a modular Mixture-of-SLMs (MoE-SLM) architecture replacing monolithic, black-box generalist LLMs with an ensemble of lightweight, specialized micro-brains (30M–124M parameters each) coordinated by a neuro-symbolic Orchestrator.

```
                           [ USER RESEARCH QUERY ]
                                      │
                                      ▼
             [ Neuro-Symbolic Orchestrator Router (30M) ]
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   [ Metabolic Brain ]       [ Neuro Brain ]           [ Structural Brain ]
   (Flux & Stoichiometry)    (BBB & Neurochemistry)    (SMILES & 2D/3D RDKit)
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                 [ Deterministic Verification Engine ]
                     • Mass Conservation (S·v = 0)
                     • RDKit Bond & Valence Checks
                                      │
                                      ▼
            [ Actionable Metabolic Engineering & Drug Protocol ]
```

---

## 2. Core Micro-Brains

1. **`MetabolicEngineerBrain` (`124M`)**:
   - Pre-trained on: Carbohydrate metabolism (Glycolysis, Gluconeogenesis, TCA Cycle, PPP), Amino acid metabolism (Shikimate, Urea Cycle, Transamination), Lipid metabolism ($\beta$-oxidation, Lipogenesis).
   - Solves: Gene knockout targets, enzyme overexpression, cofactor balance ($\text{NAD}^+ / \text{NADH}$, $\text{ATP}$).
2. **`NeurochemistBrain` (`124M`)**:
   - Pre-trained on: Synaptic transmission, neurotransmitter cascades (Catecholamines, Serotonin, GABA/Glutamate shunt), blood-brain barrier transport mechanisms (LAT1, GLUT1).
   - Solves: Receptor binding kinetics, BBB permeability, neurotoxicity risk.
3. **`StructuralChemistBrain` (`124M`)**:
   - Pre-trained on: SMILES molecular grammar, PubChem, ChEMBL chemical representations.
   - Solves: 100% syntactically valid SMILES strings, RDKit 2D molecular rendering and 3D steric conformers.

---

## 3. Benchmark Objectives for Publication

1. **Stoichiometric Mass Conservation**: $\mathbf{S} \cdot \mathbf{v} = 0$ compliance rate $> 99\%$.
2. **SMILES Validity Percentage**: RDKit parseable rate of generated intermediates $> 98\%$.
3. **Hardware Profile**: Working RAM $< 1.0\text{ GB}$, execution time $< 2.0\text{s}$ on standard CPU / \$4 VPS.

---

## 4. Systems Architecture: Selective Wakefulness & $4 VPS Deployment

Replicating biological cortical efficiency, 98% of the parameter weights remain dormant at resting state:
* **The Thalamic Orchestrator (30M)**: Always-on lightweight router (120 MB RAM). Evaluates task dependencies and awakens only the 1–2 micro-brains required.
* **Micro-Brains (124M each)**: Memory-mapped to disk. Loaded into active RAM only when invoked (248 MB active), avoiding idle resource burn.
* **Peak Concurrent RAM**: Strictly bounded at ~616 MB (easily hosted on €3.79/mo Hetzner or $5/mo DigitalOcean droplets).
* **Zero Catastrophic Forgetting**: Domain adapters update independently without destabilizing adjacent specialist networks.

---

## 5. Web Interface & Multi-Domain Chatbot Suite

The front-facing application allows students and researchers to toggle between domain experts seamlessly via 3ms LoRA hot-swapping:
1. **`Tab 1: Metabolic Engineer`**: Pathway fluxes, enzyme cofactors, and microbial engineering.
2. **`Tab 2: Neuroscientist`**: Synaptic signaling, BBB penetration, and neurochemistry.
3. **`Tab 3: Structural Chemist`**: SMILES string generation, RDKit 2D/3D structure rendering.
4. **`Tab 4: Full Team Mode`**: Multi-expert consensus orchestration for complex cross-disciplinary cases.
5. **`Tab 5: BMGpt`**: Campus personality clone simulating conversational humor and tone mimicry.

---

## 6. BioPath-Embedding & Reaction Sequence Encodings

To prevent stoichiometric and cofactor hallucinations, the Transformer replaces flat text tokenization with structured multi-aspect biochemical embeddings:

$$h_0 = E_{\text{metabolite}} + E_{\text{role}} + E_{\text{enzyme}} + E_{\text{step\_pos}} + E_{\text{compartment}}$$

* **Metabolite Token ($E_m$)**: Chemical identity representation.
* **Biochemical Role ($E_r$)**: Functional tag (`[Substrate]`, `[Product]`, `[Cofactor]`).
* **Enzyme Catalyst ($E_e$)**: EC class and catalytic machinery identity.
* **Pathway Step Positional Encoding ($E_p$)**:
  - **Linear Cascades** (e.g. Glycolysis Steps 1–10): Multi-frequency sinusoidal curves ($\sin / \cos$).
  - **Closed Thermodynamic Cycles** (e.g. Krebs / TCA Cycle Steps 1–8): Periodic harmonic Fourier encodings $\theta(pos) = \frac{2\pi \cdot pos}{N_{cycle}}$ ensuring Step 8 and Step 1 have maximal geometric proximity ($\langle PE(Step_8), PE(Step_1) \rangle \approx 1.0$).
* **Subcellular Compartment ($E_c$)**: Spatial boundary tag (`[Cytosol]`, `[Mitochondrial Matrix]`, `[Intermembrane Space]`).

Self-attention heads operating over these encodings naturally capture allosteric feedback inhibition, rate-limiting bottlenecks, and organelle transport boundaries without requiring prompt hints.

---

## 7. Adversarial Defenses & Robust Engineering

1. **Orthogonal Subspace Partitioning**: Hidden states $\mathbb{R}^{768}$ are allocated into 4 non-overlapping orthogonal blocks ($\mathcal{S}_{\text{chem}} \oplus \mathcal{S}_{\text{enzyme}} \oplus \mathcal{S}_{\text{GPS}} \oplus \mathcal{S}_{\text{harmonic}}$), proving that cross-term dot-product noise collapses strictly to zero ($\langle u_A, v_B \rangle = 0$).
2. **Biological Attention Masking**: Matrix $\mathbf{M}_{\text{tissue}}$ sets unexpressed enzymes/organelles to $-\infty$ (e.g. Erythrocytes lack mitochondria; Muscle lacks Glucose-6-Phosphatase).
3. **Dynamic Physiological State Conditioning**: Input sequences prepend a global state prefix (`[STATE: FASTED] + [INSULIN/GLUCAGON: LOW]`) that flips bifunctional enzymes (e.g. PFK-2 $\to$ FBPase-2) via attention key-value modulation.
4. **Synthetic Data Engine**: 500,000 automated `COBRApy` Flux Balance Analysis simulations generate 100M+ physically validated, mass-conserved ($\mathbf{S} \cdot \mathbf{v} = 0$) reaction sequences to bypass biological data scarcity.

---

## 8. 2D Tree-DAG Branch Coordinates & 4-Point Boundary Stress Test

To model divergent metabolic junctions (e.g. G6P branching into Glycolysis, Pentose Phosphate Pathway, and Glycogen; Pyruvate branching into Acetyl-CoA, Lactate, Alanine, and Oxaloacetate), positions are encoded as 2D tuples:

$$E_{\text{branch\_pos}} = PE_{\text{depth}}(\text{cascade\_depth}) + PE_{\text{branch}}(\text{branch\_id})$$

* Parallel branches share identical depth coordinates, preserving equal step distance from the primary carbon precursor.
* Cross-domain boundary integrity is verified across 4 adversarial stress tests:
  1. **Nitrogen Segregation**: Deamination $NH_4^+$ strictly routes to the Urea Cycle ($0.0\%$ leakage into glycolysis).
  2. **Even-Chain Lipid Barrier**: Strictly enforces the irreversibility of PDH, preventing fatty-acid-derived Acetyl-CoA from yielding net glucose via gluconeogenesis.
  3. **Anaplerotic Verification**: Glucogenic amino acids correctly enter the TCA cycle at valid anaplerotic nodes ($\alpha$-KG, Succinyl-CoA, Oxaloacetate).
  4. **RBC Mitochondrial Ban**: All mitochondrial pathways ($\beta$-oxidation, Krebs cycle, Urea cycle) are strictly inhibited in erythrocytes via biological attention masking.

---

## 9. Head-to-Head Comparative Protocol: Scratch BioPath-SLM vs. Pre-Trained Base + LoRA SFT

A formal comparative trial evaluating parameter efficiency, topological boundary integrity, and reasoning fidelity across the 3 core metabolic sectors:

1. **Model 1: BioPath-SLM (Scratch-Built)**:
   - Grounded in 4-way orthogonal subspace encodings, 2D Tree-DAG branch coordinates, and harmonic $2\pi$ cyclic encodings.
   - Dual-stage Socratic scratchpad thinking.
2. **Model 2: Downloaded SLM + LoRA SFT (Baseline)**:
   - Pre-trained generalist SLM (SmolLM-135M/360M or Qwen-2.5-0.5B).
   - Standard flat token embeddings ($W_e + W_p$) fine-tuned via LoRA ($\Delta W = B \times A$) on the exact same dataset.

**Evaluation Suite**:
* 100-question multi-domain clinical benchmark evaluating Nitrogen segregation, PDH irreversibility, organelle restriction, and Socratic diagnostic explanations.

---

## 10. Hybrid 1D-Text & Topological Pathway Mixing (`Topo-LoRA`) and 5-Tuple Enzyme-Branch Coordinate Binding

To bridge natural language Socratic reasoning (1D text order) with branched and cyclic metabolic graphs (isomorphic to Compiler Abstract Syntax Trees / Control-Flow Graphs):

1. **The 5-Tuple Biochemical Coordinate Address**:
   $$\text{Coord}(w_t) = \big(\text{Pathway\_ID}, \;\text{Branch\_ID}, \;\text{Step\_Depth}, \;\text{Role}, \;\text{Cycle\_Phase }\theta\big)$$
   Within a reaction step $(b, s)$, the Substrate, Enzyme, Cofactor, and Product share the exact same $(\text{Branch\_ID}=b, \text{Step\_Depth}=s)$ vector, enabling $O(1)$ geometric binding between an enzyme and its exact branch/step without overloading MLP weights.
2. **Zero-Initialized Gated Topological Mixing (`Topo-LoRA`)**:
   $$\mathbf{h}_0(t) = \mathbf{E}_{\text{pretrained}}(w_t) + g_t \cdot \mathbf{W}_{\text{up}} \Big[ \mathbf{e}_{\text{branch}}(b_t) \;\oplus\; \mathbf{e}_{\text{step}}(s_t) \;\oplus\; \mathbf{e}_{\text{role}}(r_t) \;\oplus\; \mathbf{e}_{\text{cycle}}(\theta_t) \Big]$$
   Initializing $\mathbf{W}_{\text{up}} = \mathbf{0}$ at Step 0 allows fine-tuning an existing pre-trained LLM with structural pathway coordinates without representation shock, while $g_t \in \{0, 1\}$ toggles between normal 1D text mode and topological pathway mode.
3. **Dual-Position Attention Logit Bias**:
   $$\text{Score}(i, j) = \frac{(R_i \mathbf{q}_i)(R_j \mathbf{k}_j)^T}{\sqrt{d_k}} + \gamma_1 \cdot \mathbf{\Phi}_{\text{step}}(s_i - s_j) + \gamma_2 \cdot \mathbf{\Psi}_{\text{branch}}(b_i, b_j)$$
4. **End-to-End Training & Tensor Pipeline**:
   * **Data Preparation**: ~140 reactions across Carbohydrate, Amino Acid, and Lipid sectors compiled into **10,000–15,000 branch-permuted graph walks** + **1,500 LIMA Socratic CoT traces**, with span-consistent BPE coordinate stamping.
   * **Parallel `DataLoader` Batch**: Sends aligned `(B, T)` tensors (`input_ids`, `bio_gate`, `pathway_ids`, `branch_ids`, `step_depths`, `role_ids`, `cycle_phases`).
   * **Dual-Head Training Loss**: Jointly optimizes next-token prediction and auxiliary coordinate prediction ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{token}} + \lambda_{\text{coord}} \mathcal{L}_{\text{coord}}$) so the model learns a true internal biochemical coordinate map and can self-tag coordinates during plain-text inference.
5. **Vector Chemistry in $\mathcal{S}_{\text{chem}} \oplus \mathcal{S}_{\text{enzyme}}$**:
   * Grounds $\mathcal{S}_{\text{chem}}$ in deterministic RDKit atom, charge, functional-group, and SMILES features $\mathbf{f}_{\text{chem}}(m)$, and constrains $\mathcal{S}_{\text{enzyme}}$ as a linear reaction operator ($\mathbf{z}_{\text{chem}}(\text{Sub}) + \mathbf{z}_{\text{op}}(\text{Enz}) \approx \mathbf{z}_{\text{chem}}(\text{Prod})$) with Tri-Modal Name $\leftrightarrow$ IUPAC $\leftrightarrow$ SMILES binding.
6. **2-Stage Systems Pharmacology & Side-Effect Engine**:
   * **Stage 1 (Target Resolution)**: Identifies inhibited `(Pathway, Branch, Step)` coordinates via the **"Decoy Key" Law** (high scaffold match in $\mathcal{S}_{\text{chem}}$ + missing reactive bond in $\mathcal{S}_{\text{enzyme}}$, e.g., `2-DG`, `Malonate`, `Statins`, `5-FU`) and **Cofactor Chemical Traps** (`Isoniazid -> PLP`, `Valproate -> CoA-SH/Carnitine`).
   * **Stage 2 (Pathway Ripple Propagation)**: Predicts organ-specific side effects and rescues across 4 topological laws (Collateral Branch Starvation, Upstream Dam & Sibling Shunt, Cross-Sector Bridge Cascade, and Shared Cofactor Depletion).
7. **Power-of-2 `d_model = 1024` (`16 Heads × 64d`) & Parallel Head Grafting (`Flamingo` / `LLaMA-Adapter`)**:
   * Upgrades `BioPath-SLM` to `d_model = 1024` (`4 Subspaces × 256d = 16 Heads × 64d`, or `d_model = 512, 16 Heads × 32d` on CPU) with **Hard-Wired Subspace Routing** so each of the 14 biochemical heads (`Heads 1–14`: 2D Structure, 3D Conformation, Chirality, Name-SMILES, EC Operator, 3D Pocket, Decoy-Key Gate, Cofactor Trap, Intra-Step Glue, Branch Ripple, $2\pi$ Cycle, Cross-Sector Bridge, Cell Type, Organelle Barrier) is physically wired to its assigned slice.
   * Grafts `BioPath14Heads` (trained 100% from scratch) in parallel onto a frozen pre-trained English backbone via a zero-initialized $\tanh(\alpha_\ell)\mathbf{W}_{\text{bridge}}$ gate:
     $$\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell-1)} + \text{FrozenEnglishHeads}\big(\mathbf{h}^{(\ell-1)}\big) + \tanh(\alpha_\ell) \cdot \mathbf{W}_{\text{bridge}}\,\text{BioPath14Heads}(\mathbf{x}_{\text{bio}})$$
   * **Downloaded Reference Papers**:
     * `papers/2022_alayrac_deepmind_flamingo_gated_xattn.pdf` (*Flamingo*, Google DeepMind, NeurIPS 2022)
     * `papers/2023_zhang_llama_adapter_zero_init_attention_grafting.pdf` (*LLaMA-Adapter*, ICLR 2024)
8. **Whole-Body Physiological Journeys & The 4 Textbook-to-`BioPath` Connection Protocols**:
   * Unifies **Chemical Reaction Steps** ($\Delta\mathcal{S}_{\text{chem}} \neq 0, \Delta\mathcal{S}_{\text{GPS}} = 0$) and **Spatial Absorption/Transport Steps** ($\Delta\mathcal{S}_{\text{chem}} = 0, \Delta\mathcal{S}_{\text{GPS}} \neq 0$, e.g., `Intestinal Lumen -> Enterocyte (SGLT1/GLUT2 vs Chylomicron) -> Portal Vein / Lymph -> Liver / Muscle` + Cori & Glucose-Alanine cycles).
   * Connects textbook prose to `BioPath14Heads` via: **(1) Coordinate Carry-Forward** on inline *"because..."* clauses, **(2) *Flamingo* Gated Cross-Attention** to a persistent Pathway Blackboard $\mathbf{M}_{\text{bio}}$, **(3) Two-Beat `[STEP]` $\leftrightarrow$ `[TEXTBOOK_REASONING]` Interleaving**, and **(4) Bidirectional Step $\leftrightarrow$ Story Training**.












