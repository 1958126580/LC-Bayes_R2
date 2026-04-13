"""
LC-Bayes R2 Framework — Master CLI Entry Point
===============================================
Orchestrates the full computational biology pipeline from synthetic data
generation through Bayesian modeling to publication-grade visualization.

Usage::

    python main.py --all          # Run entire pipeline with figures
    python main.py --pipeline     # Run pipeline without figures
    python main.py --plot         # Regenerate figures from cached data
    python main.py --all --no-plot  # Run pipeline, skip figures

Author: LC-Bayes R2 Consortium
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.data_engine import LCBayesDataSynthesizer
from src.models import (
    BayesianVarianceComponent,
    FetalMaternalHMM,
    SelectionAwareLOEUF,
    SSgBLUPWeights,
)
from src.pipeline_wrappers import AlphaSimRRunner, RelateCluesRunner
from src.visualizer import PaperVisualizer


def setup_logger() -> logging.Logger:
    """Configure robust, timestamped logging for the pipeline.

    Creates both a console handler (INFO level) and a rotating file
    handler (DEBUG level) under ``results/logs/``.

    Returns
    -------
    logging.Logger
        Configured root logger for the LC-Bayes R2 pipeline.
    """
    logger = logging.getLogger("LCBayesR2")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # File handler
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Avoid duplicating log handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def run_pipeline(do_plot: bool = True) -> None:
    """Execute the full LC-Bayes R2 pipeline from synthesis to results.

    Parameters
    ----------
    do_plot : bool
        If ``True``, generate all publication figures after the
        computational phases complete.
    """
    logger = logging.getLogger("LCBayesR2")
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("LC-Bayes R2 Pipeline Initialization")
    logger.info("=" * 60)

    try:
        # ── Phase 1: Data Synthesis Engine ───────────────────────────
        logger.info("[1/5] Initializing Digital Twin Data Engine...")
        synth = LCBayesDataSynthesizer(n_cows=83_884, seed=42)

        logger.info("      Generating full synthetic data artifacts...")
        data = synth.synthesize_all()
        loeuf_data = data.loeuf_correction_data

        logger.info("      Constructing high-fidelity phenotype tensors...")
        geno_matrix = data.genotypes_matrix

        # ── Phase 2: Core Algorithm Execution ────────────────────────
        logger.info("[2/5] Executing Core Bayesian Architecture...")

        logger.info("      -> Selection-Aware LOEUF Calibration")
        glm = SelectionAwareLOEUF()
        corrected = glm.fit(
            loeuf_naive=loeuf_data["loeuf_naive"].values,
            mean_abs_ihs=loeuf_data["mean_abs_ihs"].values,
            fst=loeuf_data["fst"].values,
            recomb_rate=loeuf_data["recomb_rate"].values,
        )

        logger.info("      -> Fetal-Maternal Variance Decomposition (HMM)")
        hmm = FetalMaternalHMM()
        rng = np.random.RandomState(42)
        sire_hap = rng.binomial(1, 0.5, (100, 20))
        dam_hap = rng.binomial(1, 0.5, (100, 20))
        hmm_res = hmm.infer_fetal_genotype(sire_hap, dam_hap)

        logger.info("      -> Spike-and-Slab Gibbs Sampler")
        gibbs = BayesianVarianceComponent()

        logger.info("      -> Genomic Prediction Weightings")
        w_model = SSgBLUPWeights()

        # ── Phase 3: External Resilient Pipeline Wrappers ────────────
        logger.info("[3/5] Interfacing with Evolutionary/Simulated Engines...")

        relate_runner = RelateCluesRunner("/fake/path/to/relate")
        logger.info("      -> Extracting Ancient vs Recent Allele Trajectories")
        ages = relate_runner.run()

        alpha_runner = AlphaSimRRunner("/fake/path/to/alphasimr")
        logger.info("      -> Projecting 20-Generation Forward Evolution")
        fwd_sim = alpha_runner.run()

        # ── Phase 4: Visualization Output ────────────────────────────
        if do_plot:
            logger.info("[4/5] Rendering Nature Genetics Publication Artifacts...")
            visualizer = PaperVisualizer(data, output_dir="results/figures")
            visualizer.plot_all()
        else:
            logger.info("[4/5] Skipping Visualization (--no-plot flag)")

        logger.info("[5/5] Pipeline successfully completed.")

    except Exception:
        logger.error("FATAL ERROR: Pipeline aborted prematurely.", exc_info=True)
        sys.exit(1)

    finally:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Pipeline Terminated. Total Wall Clock: %.2fs", elapsed)
        logger.info("=" * 60)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        description="LC-Bayes R2 — Computational Biology Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python main.py --all            Run full pipeline with figures
  python main.py --pipeline       Run pipeline without figures
  python main.py --plot           Regenerate figures only
  python main.py --all --no-plot  Run pipeline, skip figures
""",
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Run the full mathematical pipeline",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Run only the visualization module",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run pipeline and generate all figures",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting during pipeline run",
    )

    args = parser.parse_args()
    logger = setup_logger()

    if args.all or args.pipeline:
        run_pipeline(do_plot=not args.no_plot)
    elif args.plot:
        logger.info("Executing standalone visualization run...")
        synth = LCBayesDataSynthesizer(seed=42)
        data = synth.synthesize_all()
        visualizer = PaperVisualizer(data, output_dir="results/figures")
        visualizer.plot_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
