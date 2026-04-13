"""
Persistent Data Generation Utility
===================================
Runs the Digital Twin Data Engine and writes artifacts to ``results/data/``
for offline analysis.

Usage::

    python scripts/01_generate_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_engine import LCBayesDataSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Synthesize all data artifacts and persist to disk."""
    logger.info("Initializing Data Synthesis to persistent storage.")

    outdir = Path("results/data")
    outdir.mkdir(parents=True, exist_ok=True)

    synth = LCBayesDataSynthesizer(n_cows=83_884)

    logger.info("Synthesizing all data artifacts...")
    data = synth.synthesize_all()

    logger.info("Saving gene catalog...")
    data.gene_catalog.to_csv(outdir / "catalog.csv", index=False)

    logger.info("Saving genotype matrix (this simulates the 83k cohort)...")
    matrix = pd.DataFrame(data.genotypes_matrix)
    matrix.to_pickle(outdir / "genotype_matrix.pkl")

    logger.info("Saving phenotypes...")
    data.phenotypes_df.to_csv(outdir / "phenotypes.csv", index=False)

    logger.info("Saving LOEUF correction data...")
    data.loeuf_correction_data.to_csv(outdir / "loeuf_correction.csv", index=False)

    logger.info("Data generated and saved to results/data/")


if __name__ == "__main__":
    main()
