# Quick Start Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### Option 1: From Source (Recommended)

```bash
git clone https://github.com/1958126580/LC-Bayes-R2.git
cd LC-Bayes-R2
pip install -r requirements.txt
```

### Option 2: Editable Install with Dev Tools

```bash
pip install -e ".[dev,docs]"
```

## Running the Pipeline

### Full Reproduction

```bash
python main.py --all
```

This executes the complete pipeline:

1. **Data Synthesis** — generates the 83,884-cow digital twin cohort (~4s)
2. **Bayesian Modeling** — LOEUF correction, HMM inference, Gibbs sampling (~1s)
3. **External Tools** — Relate/CLUES and AlphaSimR with resilient fallbacks (~0.1s)
4. **Visualization** — all 8 publication-grade figures at 300 DPI (~10s)

Output structure:

```
results/
├── figures/
│   ├── Fig1_LOEUF_correction.png
│   ├── Fig2_Manhattan_PPA.png
│   ├── Fig3_Fetal_Maternal_Tensor_Fixed.png
│   ├── Fig4_Allele_Age_ARG.png
│   ├── Fig5_Return_Service.png
│   ├── Fig6_Power_Comparison.png
│   ├── Fig7_Prediction_Accuracy.png
│   └── Fig8_Forward_Simulation.png
├── tables/
│   ├── Table1_Targets.md
│   └── Table1_Targets.html
└── logs/
    └── pipeline_YYYYMMDD_HHMMSS.log
```

### Selective Execution

```bash
# Run modeling pipeline without figures
python main.py --pipeline

# Regenerate figures only
python main.py --plot

# Pipeline with figures disabled
python main.py --all --no-plot
```

## Python API

```python
from src.data_engine import LCBayesDataSynthesizer
from src.models import (
    SelectionAwareLOEUF,
    FetalMaternalHMM,
    BayesianVarianceComponent,
    SSgBLUPWeights,
)
from src.visualizer import PaperVisualizer

# Step 1: Synthesize data
synth = LCBayesDataSynthesizer(n_cows=83_884, seed=42)
data = synth.synthesize_all()

# Step 2: Run LOEUF correction
loeuf_df = data.loeuf_correction_data
glm = SelectionAwareLOEUF()
result = glm.fit(
    loeuf_naive=loeuf_df["loeuf_naive"].values,
    mean_abs_ihs=loeuf_df["mean_abs_ihs"].values,
    fst=loeuf_df["fst"].values,
    recomb_rate=loeuf_df["recomb_rate"].values,
)
print(f"γ coefficients: {result.gamma_coefficients}")
print(f"Reclassified: {result.n_constrained_to_moderate} C→M")

# Step 3: Run HMM
import numpy as np
hmm = FetalMaternalHMM()
sire = np.random.binomial(1, 0.5, (100, 20))
dam = np.random.binomial(1, 0.5, (100, 20))
hmm_res = hmm.infer_fetal_genotype(sire, dam)
print(f"Maternal fraction: {hmm_res.maternal_fraction:.1%}")

# Step 4: Generate figures
vis = PaperVisualizer(data, output_dir="results/figures")
vis.plot_all()
```

## Running Tests

```bash
# Standard run
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'seaborn'"

Install visualization dependencies:

```bash
pip install seaborn matplotlib tabulate
```

### ANSI warning boxes in terminal

The pipeline prints colored warning boxes when external tools are missing. This is **expected behavior** — the fallback mechanism is working correctly.
