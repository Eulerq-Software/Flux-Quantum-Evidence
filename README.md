# EulerQ Flux — Quantum and Quantum-Inspired Optimization Evidence

This repository contains the experimental code, generated results, and lightweight analysis notebooks supporting the technical report:

**“Technical Rationale for the Use of Quantum and Quantum-Inspired Methods for EulerQ Flux.”**

EulerQ Flux is a real-time rider–order assignment and batching system. After classical feasibility checks and candidate generation, Flux must select a high-quality mutually compatible subset from a large set of competing assignment candidates. This candidate-selection kernel is NP-hard and can be represented as a QUBO/Ising optimization problem.

The purpose of this repository is to provide reproducible evidence for the structural properties of hard Flux QUBOs discussed in the report and to demonstrate why those properties are relevant to quantum and quantum-inspired optimization.

---

## Evidence Covered

The repository focuses on the following experimentally observed properties.

| Property / Claim | Evidence |
|---|---|
| Classical optimization becomes increasingly costly as the raw Flux candidate set grows | Raw Flux scaling experiment |
| Useful improvements can require coordinated changes across several candidate variables and cross high intermediate barriers | Landscape-structure experiment |
| Quantum coherence materially affects useful transitions between competing Flux configurations | Coherence/dephasing experiment |
| The exchange structure of Flux can be exploited using problem-aware quantum dynamics | Structured-exchange experiment |

The experiments are intended to support the technical rationale for investigating
quantum and quantum-inspired optimization methods for the Flux candidate-selection
kernel. They should not be interpreted as a claim of universal or end-to-end
quantum computational advantage.

---

## Repository Structure

```text
Flux-Quantum-Evidence/
│
├── README.md
├── requirements.txt
│
├── raw_scaling/
│   ├── experiment.py
│   ├── results/
│   └── analysis.ipynb
│
├── landscape_structure/
│   ├── experiment.py
│   ├── results/
│   └── analysis.ipynb
│
├── coherence/
│   ├── experiment.py
│   ├── results/
│   └── analysis.ipynb
│
└── structured_exchange/
    ├── experiment.py
    ├── results/
    └── analysis.ipynb
```

Each experiment directory contains:

- `experiment.py` — code used to run the experiment;
- `results/` — the corresponding generated CSV/JSON result files;
- `analysis.ipynb` — a lightweight notebook that loads the saved results and reproduces the summary statistics reported in the technical note.

The analysis notebooks do not need to rerun the full experiments unless explicitly stated.

---

## 1. Raw Flux Scaling

**Directory:** `raw_scaling/`

This experiment studies how the classical optimization stage scales as the raw Flux candidate-selection problem grows.

Representative problem scales include approximately:

- `10 x 25` riders/orders;
- `15 x 40` riders/orders;
- `20 x 50` riders/orders.

These correspond to candidate-selection problems ranging from a few thousand to approximately twenty thousand binary variables.

The experiment records quantities including:

- number of candidate variables;
- model-construction time;
- optimization time;
- total runtime;
- solution objective.

The analysis notebook reproduces the scaling statistics quoted in the report.

---

## 2. Landscape Structure

**Directory:** `landscape_structure/`

This experiment studies the structure of difficult improvements in Flux.

For an incumbent configuration \(x\) and an improved configuration \(y\), the experiment measures the coordinated change width

\[
k = d_H(x,y),
\]

where \(d_H\) denotes Hamming distance.

It also studies the minimum intermediate barrier \(B\) encountered along local one-variable transition paths and compares it with the final improvement

\[
\Delta E = E(x)-E(y).
\]

The report highlights representative Flux-derived transitions with:

\[
k \approx 8-22
\]

and barrier-to-gain ratios on the order of

\[
\frac{B}{\Delta E} \sim 10^2-10^3.
\]

These measurements support the observation that difficult Flux refinements can require coordinated rearrangements separated by unfavorable intermediate configurations.

---

## 3. Coherence Control

**Directory:** `coherence/`

This experiment studies whether quantum phase coherence materially affects transitions between competing Flux configurations.

The same Flux-derived local optimization problem is evolved while progressively increasing computational-basis dephasing. The primary quantity measured is the probability of reaching the useful improved configuration.

The analysis reproduces the reported observation that strong dephasing suppresses useful-state probability substantially, with a median suppression of approximately:

\[
94\%.
\]

This experiment is intended as a control demonstrating that the observed transition dynamics are not reproduced simply by introducing generic stochastic perturbations.

---

## 4. Structured Exchange Dynamics

**Directory:** `structured_exchange/`

Many useful Flux refinements naturally involve replacing one competing assignment candidate with another. At the binary-variable level, the basic operation resembles

\[
|10\rangle \leftrightarrow |01\rangle.
\]

This experiment compares generic quantum driving with dynamics constructed around the actual Flux exchange/conflict structure.

The analysis reproduces the key observations that:

\[
d_X = k
\]

for generic one-variable transitions, whereas the Flux-aware exchange structure reduces the driver-graph distance to

\[
d_{\mathrm{exchange}} = \frac{k}{2}
\]

in the tested cases.

Across the eight tested Flux-derived local subproblems, the structured exchange dynamics produced higher target-state probability than the generic transverse-field dynamics in all cases, with a median improvement of approximately:

\[
18.1\times.
\]

Randomized two-variable couplings are also included as a control to distinguish the effect of Flux-specific structure from the effect of merely adding additional interactions.

---

## Reproducing the Results

Create a Python environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each experiment can then be run independently from its corresponding directory. For example:

```bash
cd structured_exchange
python experiment.py
```

The generated result files are written to the corresponding `results/` directory.

To inspect the reported statistics without rerunning the complete experiment, open:

```text
analysis.ipynb
```

inside the relevant experiment directory.

---

## Interpretation

The repository is organized around the following evidence chain:

```text
Hard Flux candidate-selection problem
            ↓
Coordinated and high-barrier rearrangements
            ↓
Strongly interacting / frustrated QUBO structure
            ↓
Coherence-sensitive transitions
            ↓
Problem-aware exchange dynamics
            ↓
Motivation for quantum and quantum-inspired optimization
```

The central hypothesis is that quantum and quantum-inspired hardware may improve the **solution-quality versus latency trade-off** of the Flux candidate-selection kernel because the difficult instances exhibit structural properties aligned with collective Ising dynamics and coherent, problem-aware quantum transitions.

The surrounding Flux workflow—including routing, feasibility checking, candidate generation, decomposition, and final validation—remains classical.

---

## Reproducibility Notes

The result folders included in this repository correspond to the outputs generated by the accompanying experiment scripts. Fixed problem instances and random seeds are retained wherever required for comparison with the values reported in the technical note.

Some quantum-mechanism experiments operate on small Flux-derived local subproblems extracted from much larger raw Flux instances. These smaller systems are used because exact coherent quantum simulation scales exponentially with the number of simulated qubits. They are intended to study the underlying transition mechanism, while the larger raw Flux experiments establish the operational problem scale.

---

## Contact

**EulerQ**

For questions regarding the experiments or the technical report, please contact the EulerQ team.
