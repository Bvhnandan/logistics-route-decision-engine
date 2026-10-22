# Multimodal Logistics Route Optimisation Engine
### Multi-Criteria Decision Pipeline Under Cost, Delivery Time & Risk Constraints

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](#prerequisites)
[![Test Suite](https://img.shields.io/badge/tests-29%20passed-success.svg)](#7-test-suite--validation)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## Executive Summary

In international multimodal supply chains, selecting an optimal freight route requires resolving conflicting trade-offs between direct freight expense, delivery timeliness, and geopolitical/operational disruption risks. 

This repository implements a **pure-Python, production-grade logistics optimisation engine** designed to evaluate complex multimodal corridors (combining Air, Sea, Road, and Rail legs across intercontinental transshipment hubs). Hard operational constraints (maximum budget and delivery deadline with a safety buffer) act as non-negotiable feasibility gates, followed by Pareto-dominance pruning and profile-weighted multi-criteria penalty scoring.

---

## Table of Contents

1. [Problem Statement & Decision Framework](#1-problem-statement--decision-framework)
2. [Mathematical Formulations](#2-mathematical-formulations)
3. [Repository Blueprint & File Connections](#3-repository-blueprint--file-connections)
4. [Inter-File Data Flow & Architecture](#4-inter-file-data-flow--architecture)
5. [Setup & Quickstart](#5-setup--quickstart)
6. [Interactive CLI & Demo Capabilities](#6-interactive-cli--demo-capabilities)
7. [Test Suite & Validation](#7-test-suite--validation)
8. [Benchmark Evaluation & Comparative Results](#8-benchmark-evaluation--comparative-results)
9. [Key Engineering & Design Decisions](#9-key-engineering--design-decisions)

---

## 1. Problem Statement & Decision Framework

### Definition of "Optimal Route"
An optimal route is one that:
1. **Satisfies Hard Gates**: Strict adherence to the shipper's maximum budget and delivery deadline (including a mandatory $1.0$-day safety buffer for customs clearance/handling).
2. **Survives Pareto Pruning**: Is not strictly dominated in cost, delivery time, and disruption risk by any competitor route.
3. **Minimises Weighted Penalty**: Achieves the lowest multi-criteria penalty score tailored to the shipment's profile (**Urgent**, **Standard**, or **Cost-Sensitive**).

### Core Pipeline Stages

```
                        Raw Shipment Input (JSON)
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Risk-Adjusted Metrics Computation                            │
│  - Expected Delay E[D] = p_d * d                                       │
│  - Expected Transit Time E[T] = T_base + E[D]                          │
│  - Landed Cost E[C] = Freight + min(E[D] * α * V, β * Freight)         │
│  - Risk Index = (Congestion + Geopolitical) / 2.0                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Hard Operational Feasibility Gating                           │
│  - Budget Gate: C_route <= Max Budget                                  │
│  - Deadline Gate: E[T] + 1.0d Buffer <= Delivery Deadline              │
│                                                                        │
│  [IF 0 ROUTES FEASIBLE] ──> Structured Actionable Fallback Advisory    │
│                             (quantified budget / deadline gap)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Feasible Candidates]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Objective Pareto Dominance Filter                       │
│  - Evaluates (E[C], E[T], Risk)                                        │
│  - Prunes all routes that are strictly inferior in all 3 criteria      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Pareto Non-Dominated Set]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Profile-Weighted Multi-Criteria Scoring                       │
│  Mode A: Reference Scoring (Default)                                   │
│    cost_pen = E[C]/budget, time_pen = E[T]/deadline, risk_pen = risk   │
│    Score = w_c * cost_pen + w_t * time_pen + w_r * risk_pen            │
│  Mode B: Min-Max Normalised Scoring                                    │
│    Normalises surviving set to [0, 1] bounds                           │
│                                                                        │
│  - Low-Confidence Alert: Raised when Top-2 scores differ by < 0.05    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
                  Deterministic Operator Report (explain.py)
                    * Optimal Recommendation & Routing
                    * Quantitative Delta Trade-offs vs Baselines
                    * Disqualification & Dominated Reason Audit
```

---

## 2. Mathematical Formulations

### Stage 3: Expected Delay, Time, Landed Cost & Risk
* **Expected Delay**:
  $$\mathbb{E}[D] = p_d \cdot d$$
  where $p_d \in [0, 1]$ is the route-level delay probability and $d$ is delay duration in days.
* **Expected Transit Time**:
  $$\mathbb{E}[T] = T_{\text{base}} + \mathbb{E}[D]$$
* **Delay Disruption Cost**:
  $$C_{\text{delay}} = \min\left(\mathbb{E}[D] \cdot \alpha \cdot V, \; \beta \cdot C_{\text{route}}\right)$$
  where $\alpha = 0.0005$ (0.05% of cargo value $V$ per delayed day) and $\beta = 0.20$ (cost cap at 20% of base freight).
* **Expected Total Landed Cost**:
  $$\mathbb{E}[C] = C_{\text{route}} + C_{\text{delay}}$$
* **Composite Risk Index**:
  $$\text{Risk Index} = \frac{\text{Congestion Score} + \text{Geopolitical Risk Score}}{2.0}$$
  *(Note: $p_d$ is deliberately excluded from the Risk Index to prevent double-counting disruption impact already captured in $\mathbb{E}[C]$ and $\mathbb{E}[T]$).*

---

### Stage 1: Hard Operational Gates & Structured Fallback
* A route is disqualified if:
  $$C_{\text{route}} > \text{Max Budget} \quad \text{OR} \quad \mathbb{E}[T] + 1.0\text{d} > \text{Deadline}$$
* **Structured Fallback (Zero Feasible Routes)**:
  Instead of throwing exceptions or returning empty outputs, the pipeline identifies the best compromise route and provides actionable advisory values:
  1. If routes exist within budget but miss deadline: recommend the fastest route within budget and calculate the exact deadline extension required:
     $$\Delta_{\text{deadline}} = \mathbb{E}[T] + 1.0 - \text{Deadline}$$
  2. If all routes exceed budget: recommend the cheapest overall route and compute both the required budget increase and deadline extension:
     $$\Delta_{\text{budget}} = C_{\text{route}} - \text{Max Budget}$$

---

### Stage 2: Pareto Dominance Filtering
Route $R_1$ is **dominated** by $R_2$ if:
$$\mathbb{E}[C]_{R_2} \le \mathbb{E}[C]_{R_1} \quad \wedge \quad \mathbb{E}[T]_{R_2} \le \mathbb{E}[T]_{R_1} \quad \wedge \quad \text{Risk}_{R_2} \le \text{Risk}_{R_1}$$
with at least one strict inequality. All dominated routes are pruned and logged with explicit attribution.

---

### Stage 4: Profile-Weighted Scoring Modes

#### Profile Weight Configurations
| Shipment Profile | Cost Weight ($w_c$) | Time Weight ($w_t$) | Risk Weight ($w_r$) |
|:---|:---:|:---:|:---:|
| **Urgent** | $0.15$ | $0.50$ | $0.35$ |
| **Standard** | $0.35$ | $0.30$ | $0.35$ |
| **Cost-Sensitive** | $0.60$ | $0.20$ | $0.20$ |

#### Mode 1: Reference Scoring (`scoring="reference"`, Default)
Route scores are evaluated independently of competitor routes against shipment constraints:
$$\text{Cost Penalty} = \frac{\mathbb{E}[C]}{\text{Max Budget}}$$
$$\text{Time Penalty} = \frac{\mathbb{E}[T]}{\text{Deadline}}$$
$$\text{Risk Penalty} = \text{Risk Index}$$
$$\text{Score} = w_c \cdot \text{Cost Penalty} + w_t \cdot \text{Time Penalty} + w_r \cdot \text{Risk Penalty}$$

*(In Python code: `cost_penalty`, `time_penalty`, `risk_penalty`)*

Lowest score wins. This formulation ensures that introducing or removing a competing route cannot artificially distort an existing candidate's evaluation score.

#### Mode 2: Min-Max Normalised Scoring (`scoring="minmax"`)
Evaluates relative performance within the surviving Pareto set:
$$\text{norm}(x) = \frac{x - \min(X)}{\max(X) - \min(X)} \quad (\text{yields } 0.0 \text{ when } \max(X) = \min(X))$$
$$\text{Score} = w_c \cdot \text{norm}(\mathbb{E}[C]) + w_t \cdot \text{norm}(\mathbb{E}[T]) + w_r \cdot \text{norm}(\text{Risk})$$

#### Low-Confidence Flag
If $|\text{Score}_{\text{winner}} - \text{Score}_{\text{runner-up}}| < 0.05$, the decision is flagged as **Low-Confidence** for mandatory human operator review.

---

## 3. Repository Blueprint & File Connections

```
Logistics planner/
├── data/
│   ├── shipments.json              # Primary evaluation dataset (35 shipments, seed 42)
│   └── shipments_before.json       # Regression snapshot baseline (numeric verification)
├── src/
│   ├── __init__.py                 # Package declaration and public exports
│   ├── generator.py                # Synthetic dataset generation & realistic corridor routing
│   ├── engine.py                   # Core 4-stage optimisation pipeline, metrics & fallback logic
│   ├── explain.py                  # Plain-language explanation generator & delta trade-off reporter
│   └── evaluate.py                 # Benchmarking engine, CRN simulations & cost-weight sweeps
├── tests/
│   ├── __init__.py                 # Test package initialization
│   └── test_pipeline.py            # Comprehensive 29-test pytest suite (6 test categories)
├── artifacts/
│   ├── decision_flip_reference.png # Reference scoring decision-flip plot (SHP-001)
│   ├── decision_flip.png           # Min-max scoring decision-flip plot
│   └── .gitkeep                    # Directory preservation
├── check_numerics.py               # Dataset invariant validator (ensures 0 numeric drift)
├── run_demo.py                     # Interactive CLI demonstration tool
├── requirements.txt                # Production dependencies (numpy, matplotlib, pytest)
├── .gitignore                      # Git configuration excluding caches, environments, and logs
└── README.md                       # Architectural documentation & project overview
```

### Detailed File Roles & Responsibilities

| File | What It Does | Why It Exists | How It Connects to Other Files |
|:---|:---|:---|:---|
| [`src/engine.py`](file:///c:/Users/91837/Videos/Logistics%20planner/src/engine.py) | Executes Stages 1–4, metrics calculations, Pareto filtering, fallback calculations, and scoring. | **The algorithmic core of the repository.** Contains all business logic and decision formulas. | Imported by `src/explain.py`, `src/evaluate.py`, `run_demo.py`, and `tests/test_pipeline.py`. Operates directly on shipment dicts from `data/shipments.json`. |
| [`src/explain.py`](file:///c:/Users/91837/Videos/Logistics%20planner/src/explain.py) | Formats `PipelineResult` dataclasses into formatted terminal reports with delta blocks and disqualification tables. | Translates raw optimization numbers into **interpretable, audit-ready operator insights**. | Takes `PipelineResult` objects produced by `src/engine.py`. Called by `run_demo.py` to print recommendations. |
| [`src/generator.py`](file:///c:/Users/91837/Videos/Logistics%20planner/src/generator.py) | Generates synthetic multimodal logistics networks across realistic intercontinental maritime/air corridors using random seed `42`. | Creates reproducible, realistic test scenarios with genuine trade-offs and calibrated infeasibilities. | Writes output directly to `data/shipments.json`. Re-uses mathematical metrics from `src/engine.py`. |
| [`src/evaluate.py`](file:///c:/Users/91837/Videos/Logistics%20planner/src/evaluate.py) | Benchmarks the engine against Cheapest-First and Fastest-First policies using 1,000 CRN draws per shipment; executes cost-weight sweeps. | Quantitatively proves that the multi-criteria engine out-performs naive single-objective baselines. | Imports `src/engine.py` functions; reads `data/shipments.json`; generates visual plots in `artifacts/`. |
| [`run_demo.py`](file:///c:/Users/91837/Videos/Logistics%20planner/run_demo.py) | Unified command-line CLI tool to run specific shipments, catalog tables, or default showcase scenarios. | Provides an intuitive interface for live demos, operator reviews, and manual inspection. | Calls `run_pipeline` in `src/engine.py` and `print_explanation` in `src/explain.py`. |
| [`check_numerics.py`](file:///c:/Users/91837/Videos/Logistics%20planner/check_numerics.py) | Invariant verification script comparing `shipments.json` with `shipments_before.json`. | Guarantees that adding geographic realism and hub corridors caused zero numeric drift in benchmarks. | Compares `data/shipments.json` and `data/shipments_before.json`. |
| [`tests/test_pipeline.py`](file:///c:/Users/91837/Videos/Logistics%20planner/tests/test_pipeline.py) | 29 unit and integration tests covering hard constraints, zero-variance, Pareto pruning, profile reversals, and scoring independence. | Ensures regression stability and verifies all operational edge cases. | Imports `src/engine.py` and `src/explain.py`. |
| [`requirements.txt`](file:///c:/Users/91837/Videos/Logistics%20planner/requirements.txt) | Minimal dependency specification (`numpy`, `matplotlib`, `pytest`). | Ensures zero dependency bloat and clean reproduction across platforms. | Installs environment libraries used by `src/` and `tests/`. |

---

## 4. Inter-File Data Flow & Architecture

```
                  ┌────────────────────┐
                  │  src/generator.py  │
                  └─────────┬──────────┘
                            │ writes dataset (seed 42)
                            ▼
                  ┌────────────────────┐
                  │data/shipments.json │
                  └─────────┬──────────┘
                            │ reads
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐   ┌─────────────────┐   ┌────────────────┐
│ run_demo.py  │   │ src/evaluate.py │   │check_numerics  │
└──────┬───────┘   └────────┬────────┘   └────────────────┘
       │                    │ (CRN & Sweep)
       │ calls              │ calls
       ▼                    ▼
┌────────────────────────────────────┐
│           src/engine.py            │
│  * Metrics Calculation             │
│  * Hard Constraint Gating          │
│  * Pareto Dominance Pruning        │
│  * Reference / Min-Max Scoring     │
└─────────────────┬──────────────────┘
                  │ returns PipelineResult
                  ▼
┌────────────────────────────────────┐       ┌────────────────────────┐
│           src/explain.py           │ ----> │ Terminal Human Report  │
└────────────────────────────────────┘       └────────────────────────┘
```

---

## 5. Setup & Quickstart

### Prerequisites
- Python 3.9+ (tested through 3.12)
- Virtual environment (recommended)

### Installation & Environment Setup
```powershell
# 1. Clone repository
git clone <repo-url>
cd "Logistics planner"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate     # On Linux / macOS

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Generate the synthetic dataset (seed 42)
python -m src.generator
```

> **Note on Dataset Generation**: Running `python -m src.generator` populates `data/shipments.json` with 35 intercontinental multimodal shipments across calibrated urgent, standard, and cost-sensitive profiles, including realistic maritime/air transshipment hubs and forced-infeasible test scenarios (`SHP-011`, `SHP-021`, `SHP-031`).

---

### Command Cheat Sheet

| Task | Command | Purpose |
|:---|:---|:---|
| **Generate Dataset** | `python -m src.generator` | Generates `data/shipments.json` (35 shipments, seed 42) |
| **Run All Tests** | `pytest tests/ -v` | Runs all 29 unit & integration tests |
| **Run Evaluation Benchmarks** | `python -m src.evaluate` | Runs Table A/B benchmarks (1,000 CRN draws) & generates plots in `artifacts/` |
| **Verify Dataset Numerics** | `python check_numerics.py` | Asserts 0 numeric drift against baseline snapshot |
| **Interactive CLI (3 Showcase)** | `python run_demo.py` | Runs default showcase (1 trade-off, 2 fallbacks) |
| **Interactive CLI (Catalog Table)** | `python run_demo.py --list` | Displays clean 1-line summary catalog of all 35 shipments |
| **Run Specific Shipment** | `python run_demo.py SHP-001` | Evaluates single shipment report (case-insensitive) |
| **Run Multiple Shipments** | `python run_demo.py SHP-003 SHP-031` | Evaluates specified list of shipment IDs |

---

---

## 6. Interactive CLI & Demo Capabilities

The demonstration CLI ([`run_demo.py`](file:///c:/Users/91837/Videos/Logistics%20planner/run_demo.py)) provides full operational control:

### 1. View Complete 35-Shipment Catalog Table
Displays a clean, aligned 1-line summary showing origin/destination corridors, profiles, budgets, deadlines, and surviving route counts:
```powershell
python run_demo.py --list
```

### 2. Inspect Single Specific Shipments (Case-Insensitive)
```powershell
python run_demo.py SHP-001
python run_demo.py shp-001
```

### 3. Evaluate Multiple Specific Shipments
```powershell
python run_demo.py SHP-003 SHP-031
```

### 4. Run Default Showcase Scenarios
Runs the three core operational case studies (1 feasible trade-off, 1 budget/dual fallback, 1 deadline fallback):
```powershell
python run_demo.py
```

### 5. Switch Scoring Mode
```powershell
python run_demo.py --scoring minmax SHP-001
```

---

## 7. Test Suite & Validation

The test suite in [`tests/test_pipeline.py`](file:///c:/Users/91837/Videos/Logistics%20planner/tests/test_pipeline.py) contains **29 targeted pytest cases** covering all critical operational and algorithmic requirements:

```powershell
# Run the complete test suite (all 29 tests verbose):
pytest tests/ -v

# Run with concise summary:
pytest tests/ -q

# Run specific edge-case test categories:
pytest tests/test_pipeline.py -k TestReferenceScoring -v
pytest tests/test_pipeline.py -k TestInfeasibleFallback -v
pytest tests/test_pipeline.py -k TestProfilePreferenceReversal -v
```

### Breakdown of Test Groups

1. **`TestHardBudgetExclusion`** (2 tests): Verifies over-budget routes are pruned and disqualification logs state exact cost vs. budget.
2. **`TestInfeasibleFallback`** (4 tests): Tests fallback selection and exact gap calculation for budget-short, deadline-short, and dual-short scenarios.
3. **`TestParetoDominanceFilter`** (3 tests): Tests strict 3-dimensional dominance pruning and attribution logs.
4. **`TestZeroVarianceNormalisation`** (3 tests): Tests zero-variance metric handling ($0.0$ assignment) without dividing by zero.
5. **`TestLowConfidenceAlert`** (3 tests): Tests that decision margin $< 0.05$ triggers alert, while wide margins do not.
6. **`TestProfilePreferenceReversal`** (4 tests): Proves Urgent prefers Fast routes, Cost-Sensitive prefers Cheap routes, and weights sum to $1.0$.
7. **`TestDualGapAdvisory`** (4 tests): Validates actionable text explanations when routes fail on both budget and deadline.
8. **`TestGeneratorOutputCounts`** (2 tests): Guarantees exactly 3 infeasible shipments and $\ge 24$ shipments with $\ge 2$ Pareto survivors.
9. **`TestBaselineFallbackParity`** (1 test): Verifies fallback route selection parity across strategies.
10. **`TestDominatedNeverSelected`** (1 test): Confirms dominated routes are never recommended.
11. **`TestReferenceScoring`** (2 tests):
    - **Independence**: Confirms a candidate route's score in reference mode is identical whether or not an additional non-dominated route is present.
    - **Sensitivity**: Proves a 1% cost difference and a 100% cost difference produce proportionally scaled score gaps.

---

## 8. Benchmark Evaluation & Comparative Results

Run the full comparative evaluation benchmark:
```powershell
# Run full evaluation benchmark across all 35 shipments (Table A, Table B & sweeps):
python -m src.evaluate

# Run evaluation with an explicit dataset path:
python -m src.evaluate --data data/shipments.json
```

All 3 strategies are evaluated across **1,000 Common Random Numbers (CRN) simulation draws** (seed 2024) per shipment. A route experiences delay in trial $k$ if $U_k < p_d$. A deadline miss occurs if $\text{Arrival}_k > \text{Deadline}$.

### Table A: Feasible Shipments Only ($N=32$)
| Strategy | Avg Landed Cost $\mathbb{E}[C]$ | Deadline Miss Rate | Avg Transit Time $\mathbb{E}[T]$ | Avg Risk Index |
|:---|:---:|:---:|:---:|:---:|
| **Cheapest-First** | $\$3,289.69$ | $19.6\%$ | $22.83\text{d}$ | $0.4252$ |
| **Fastest-First** | $\$6,459.97$ | $0.0\%$ | $7.62\text{d}$ | $0.2043$ |
| **Engine (Min-Max)** | $\$5,327.41$ | $6.7\%$ | $13.08\text{d}$ | $0.2837$ |
| **Engine (Reference)** | $\$5,441.12$ | **$3.9\%$** | **$11.20\text{d}$** | **$0.2624$** |

### Table B: All Shipments ($N=35$, including fallbacks)
| Strategy | Avg Landed Cost $\mathbb{E}[C]$ | Deadline Miss Rate | Avg Transit Time $\mathbb{E}[T]$ | Avg Risk Index |
|:---|:---:|:---:|:---:|:---:|
| **Cheapest-First** | $\$3,475.58$ | $23.7\%$ | $22.01\text{d}$ | $0.4156$ |
| **Fastest-First** | $\$6,374.13$ | $5.7\%$ | $8.10\text{d}$ | $0.2136$ |
| **Engine (Min-Max)** | $\$5,338.63$ | $11.8\%$ | $13.10\text{d}$ | $0.2862$ |
| **Engine (Reference)** | $\$5,442.60$ | **$9.2\%$** | **$11.38\text{d}$** | **$0.2667$** |

### Key Findings
1. **Superior Reliability**: The Proposed Engine reduces the deadline miss rate of Cheapest-First from **$23.7\%$ down to $9.2\%$** (and down to **$3.9\%$** for feasible shipments in reference mode) while avoiding the excessive freight premium of Fastest-First ($+\$6,374$ vs $\$5,442$).
2. **Divergence from Naive Policies**: In reference mode, the engine diverges from Cheapest-First in **$75.0\%$** of shipments and from Fastest-First in **$28.1\%$** of shipments. In $6.2\%$ of cases, it selects an interior balanced route that is neither the cheapest nor fastest option.
3. **Low-Confidence Awareness**: In reference mode, $7$ shipments trigger low-confidence alerts ($20.0\%$), warning operators when trade-offs between competing routes are within razor-thin margins ($< 0.05$).

### Cost-Weight Decision Flip Analysis (SHP-001)

The cost-weight sweep plot for `SHP-001` under reference scoring is saved in [`artifacts/decision_flip_reference.png`](file:///c:/Users/91837/Videos/Logistics%20planner/artifacts/decision_flip_reference.png):

- **Shipment**: `SHP-001` (Dubai $\rightarrow$ Tokyo, Urgent profile: $w_c=0.15, w_t=0.50, w_r=0.35$).
- **Candidates**:
  - `R0 Air ($11,805, 5.9 d)`
  - `R3 Sea+Rail ($6,003, 9.3 d)`
- **Exact Analytical Decision Flip**:
  $$w_{\text{cost}}^* = \mathbf{0.2062} \quad (\approx 0.206)$$
- At the shipment's true operating weight ($w_c = 0.15 \le 0.206$), `R0 Air` is selected. If the shipper increases cost sensitivity above $0.206$, the engine flips recommendations to `R3 Sea+Rail`.

---

## 9. Key Engineering & Design Decisions

| Architectural Decision | Technical Rationale |
|:---|:---|
| **Deterministic Seed Isolation** | All synthetic data was drawn with `np.random.default_rng(42)`. Geographic corridors were populated with an isolated `Random(999)`. CRN trials use seed `2024`. This guarantees 100% reproducibility across environments. |
| **$p_d$ Excluded from Risk Index** | Delay probability $p_d$ directly drives expected transit time $\mathbb{E}[T]$ and delay cost $C_{\text{delay}}$. Incorporating it into the Risk Index would double-penalise route volatility. |
| **Strict Operational Gate Precedence** | Expected metrics (Stage 3) are evaluated before feasibility (Stage 1) so that the $1.0$-day customs buffer is verified against the route's risk-adjusted arrival date $\mathbb{E}[T]$. |
| **Non-Throwing Fallback Advisories** | Supply chain operators cannot act on unhandled exceptions. If zero routes satisfy budget/deadline gates, the engine returns quantitative recommendations detailing the exact budget increase or deadline extension needed to restore feasibility. |
| **Independent Reference Scoring** | By normalizing costs and times against the shipment's own budget and deadline bounds, route scores reflect true resource consumption and remain unaffected by the presence or absence of irrelevant competitor routes. |

---

## Verification & Integrity Check

you can verify dataset and numeric identity:

```powershell
python check_numerics.py
```

Expected confirmation:
```
OK: all 6 route numeric fields and 3 shipment numeric fields are IDENTICAL across all 35 shipments.
OK: all routes satisfy geographic realism rules (corridor hubs, overland-only Road/Rail to destination, and de-duplicated mode).
```
