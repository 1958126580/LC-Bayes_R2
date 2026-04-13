# LC-Bayes R2 Framework

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)

> **Selection-aware, fetal-maternal variance components improve dairy cattle genomic prediction.**

## What is LC-Bayes R2?

LC-Bayes R2 is a Bayesian hierarchical framework that integrates:

- **Evolutionary constraint scores** (Livestock-LOEUF) corrected for selective sweeps
- **Fetal-maternal variance decomposition** via Hidden Markov Model inference on phased haplotypes
- **Spike-and-slab constraint priors** with Pólya–Gamma augmentation for gene-level variance estimation
- **Theoretically grounded ssGBLUP weights** that bridge the gap between full Bayesian and industrial-scale prediction

The result: **5 novel recessive lethal genes** discovered, including the first maternal-effect lethal (*PROK1*) identified in livestock.

## Quick Start

```bash
# Install
git clone https://github.com/1958126580/LC-Bayes-R2.git
cd LC-Bayes-R2
pip install -r requirements.txt

# Run everything
python main.py --all

# Run tests
python -m pytest tests/ -v
```

## Documentation

| Page | Description |
|------|-------------|
| [Quick Start](quickstart.md) | Installation, first run, and basic usage |
| [Architecture](architecture.md) | System design and pipeline flow |
| [Mathematical Foundations](mathematical_foundations.md) | Full equation derivations |
| [API Reference](api_reference.md) | Complete class and method documentation |
| [Results](results.md) | Generated figures and tables |

## Project Status

- ✅ 29 / 29 tests passing
- ✅ 8 / 8 publication figures generated
- ✅ Full pipeline runs in ~15 seconds
- ✅ Zero external tool dependencies required (resilient fallbacks)
