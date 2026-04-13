"""
Comprehensive Test Suite for LC-Bayes R2 Framework
===================================================
Covers data integrity, mathematical correctness, pipeline resilience,
and visualization output. All random generators use fixed seeds for
full deterministic reproducibility.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_engine import LCBayesDataSynthesizer, SynthesizedData
from src.models import (
    BayesianVarianceComponent,
    FetalMaternalHMM,
    GibbsResult,
    LOEUFCorrectionResult,
    SelectionAwareLOEUF,
    SSgBLUPWeights,
)
from src.pipeline_wrappers import AlphaSimRRunner, RelateCluesRunner
from src.visualizer import PaperVisualizer


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def synth():
    """Shared lightweight synthesizer for fast tests."""
    return LCBayesDataSynthesizer(n_cows=500, seed=42)


@pytest.fixture(scope="module")
def full_data(synth):
    """Shared SynthesizedData object for tests that need the full pipeline."""
    return synth.synthesize_all()


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Engine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDataEngine:
    """Tests for the Digital Twin Data Engine."""

    def test_gene_catalog_injection(self, synth):
        """Known lethal and novel target genes must be injected."""
        catalog = synth.generate_gene_catalog()
        target_genes = catalog[catalog["is_target"]]["gene"].tolist()

        for gene in ["RFC5", "PROK1", "DOCK8", "ITGB7", "APAF1", "SMC2"]:
            assert gene in target_genes, f"{gene} must be injected"

    def test_gene_catalog_loeuf_nonneg(self, synth):
        """All LOEUF scores must be non-negative."""
        catalog = synth.generate_gene_catalog()
        assert (catalog["loeuf_naive"] >= 0).all()

    def test_gene_catalog_chromosome_range(self, synth):
        """Chromosomes must span 1–29 (Bos taurus autosomes)."""
        catalog = synth.generate_gene_catalog()
        assert catalog["chr"].min() >= 1
        assert catalog["chr"].max() <= 29

    def test_loeuf_correction_reclassification_counts(self, synth):
        """Data engine must produce exactly 76 C→M and 23 M→C."""
        catalog = synth.generate_gene_catalog()
        df = synth.generate_loeuf_correction_data(catalog)
        c2m = (df["reclass_direction"] == "constrained_to_moderate").sum()
        m2c = (df["reclass_direction"] == "moderate_to_constrained").sum()
        assert c2m == 76, f"Expected 76 C→M, got {c2m}"
        assert m2c == 23, f"Expected 23 M→C, got {m2c}"

    def test_manhattan_data_has_cumulative_position(self, full_data):
        """Manhattan data must have cumulative positions for plotting."""
        mdata = full_data.manhattan_data
        assert "cumul_pos" in mdata.columns
        assert mdata["cumul_pos"].min() >= 0
        assert mdata["cumul_pos"].max() > 0

    def test_phenotype_count(self, full_data):
        """Phenotypes table must have one row per cow."""
        assert len(full_data.phenotypes_df) == 500  # n_cows from fixture

    def test_genotype_matrix_shape(self, full_data):
        """Genotype matrix must be (n_cows, n_variants)."""
        assert full_data.genotypes_matrix.shape[0] == 500
        assert full_data.genotypes_matrix.shape[1] == 500  # default n_variants

    def test_allele_ages_all_positive(self, full_data):
        """Allele age estimates must be positive."""
        assert (full_data.allele_ages["allele_age"] > 0).all()

    def test_return_to_service_range(self, full_data):
        """RTS intervals must be within [10, 95] days."""
        for arr in [
            full_data.rts_rfc5_control,
            full_data.rts_rfc5_carrier,
            full_data.rts_prok1_control,
            full_data.rts_prok1_carrier,
        ]:
            assert arr.min() >= 10
            assert arr.max() <= 95

    def test_forward_simulation_21_generations(self, full_data):
        """Forward sim must span exactly 21 rows (generations 0–20)."""
        assert len(full_data.forward_sim) == 21
        assert full_data.forward_sim["generation"].iloc[0] == 0
        assert full_data.forward_sim["generation"].iloc[-1] == 20

    def test_frequency_trajectories_populated(self, full_data):
        """Frequency trajectory arrays must be non-empty after synthesis."""
        assert len(full_data.apaf1_traj_gens) > 0
        assert len(full_data.prok1_traj_freq) > 0

    def test_synthesize_all_returns_dataclass(self, full_data):
        """synthesize_all must return a SynthesizedData instance."""
        assert isinstance(full_data, SynthesizedData)

    def test_deterministic_reproducibility(self):
        """Two synthesizers with the same seed must produce identical catalogs."""
        s1 = LCBayesDataSynthesizer(n_cows=200, seed=99)
        s2 = LCBayesDataSynthesizer(n_cows=200, seed=99)
        c1 = s1.generate_gene_catalog()
        c2 = s2.generate_gene_catalog()
        pd.testing.assert_frame_equal(c1, c2)


# ═══════════════════════════════════════════════════════════════════════
# 2. Statistical Model Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSelectionAwareLOEUF:
    """Tests for the Poisson GLM LOEUF correction model."""

    def test_fit_returns_correction_result(self):
        rng = np.random.RandomState(42)
        n = 200
        result = SelectionAwareLOEUF().fit(
            loeuf_naive=np.clip(rng.normal(0.6, 0.3, n), 0.02, 1.8),
            mean_abs_ihs=np.clip(rng.exponential(0.6, n), 0.01, 4.0),
            fst=np.clip(rng.beta(2, 8, n), 0.01, 0.5),
            recomb_rate=np.clip(rng.exponential(1.2, n), 0.1, 5.0),
        )
        assert isinstance(result, LOEUFCorrectionResult)
        assert result.corrected_loeuf.shape == (n,)
        assert (result.corrected_loeuf >= 0).all()

    def test_gamma_coefficients_shape(self):
        rng = np.random.RandomState(7)
        n = 100
        result = SelectionAwareLOEUF().fit(
            loeuf_naive=np.clip(rng.normal(0.5, 0.2, n), 0.05, 1.5),
            mean_abs_ihs=rng.exponential(0.5, n),
            fst=rng.beta(2, 8, n),
            recomb_rate=rng.exponential(1.0, n),
        )
        assert result.gamma_coefficients.shape == (3,)


class TestFetalMaternalHMM:
    """Tests for the Forward-Backward HMM genotype inference."""

    def test_posterior_sums_to_one(self):
        hmm = FetalMaternalHMM()
        rng = np.random.RandomState(42)
        sire = rng.binomial(1, 0.5, (5, 20))
        dam = rng.binomial(1, 0.5, (5, 20))
        res = hmm.infer_fetal_genotype(sire, dam)
        for row in res.fetal_posteriors:
            assert np.isclose(row.sum(), 1.0), "Posteriors must sum to 1"

    def test_variance_decomposition_positive(self):
        hmm = FetalMaternalHMM()
        rng = np.random.RandomState(7)
        sire = rng.binomial(1, 0.3, (10, 20))
        dam = rng.binomial(1, 0.7, (10, 20))
        res = hmm.infer_fetal_genotype(sire, dam)
        assert res.tau_sq_mat > 0
        assert res.tau_sq_fet > 0
        assert res.tau_sq_total == pytest.approx(
            res.tau_sq_mat + res.tau_sq_fet, rel=0.01
        )

    def test_maternal_fraction_bounded(self):
        hmm = FetalMaternalHMM()
        rng = np.random.RandomState(13)
        res = hmm.infer_fetal_genotype(
            rng.binomial(1, 0.5, (3, 20)),
            rng.binomial(1, 0.5, (3, 20)),
        )
        assert 0 <= res.maternal_fraction <= 1


class TestBayesianVarianceComponent:
    """Tests for the Gibbs sampler with spike-and-slab prior."""

    def test_ppa_in_unit_interval(self):
        rng = np.random.RandomState(42)
        gibbs = BayesianVarianceComponent(n_burnin=10, n_samples=20)
        y = rng.normal(0, 1, 50)
        X = rng.binomial(2, 0.1, (50, 5)).astype(float)
        w = np.ones(5)
        res = gibbs.fit(y, X, w, loeuf_score=0.2, functional_annotation=0.5)
        assert 0.0 <= res.ppa <= 1.0

    def test_tau_sq_nonnegative(self):
        rng = np.random.RandomState(99)
        gibbs = BayesianVarianceComponent(n_burnin=5, n_samples=10)
        y = rng.normal(0, 1, 30)
        X = rng.binomial(2, 0.05, (30, 4)).astype(float)
        w = np.ones(4)
        res = gibbs.fit(y, X, w, loeuf_score=0.3)
        assert (res.tau_sq_samples >= 0).all()

    def test_gibbs_result_fields(self):
        rng = np.random.RandomState(42)
        gibbs = BayesianVarianceComponent(n_burnin=5, n_samples=10)
        y = rng.normal(0, 1, 20)
        X = rng.binomial(2, 0.1, (20, 3)).astype(float)
        w = np.ones(3)
        res = gibbs.fit(y, X, w, loeuf_score=0.2)
        assert isinstance(res, GibbsResult)
        assert res.tau_sq_samples.shape == (10,)
        assert res.beta_samples.shape == (10, 3)
        assert res.effective_sample_size > 0


class TestSSgBLUPWeights:
    """Tests for ssGBLUP weight derivation (Equation 9)."""

    def test_weights_centered_at_one(self):
        rng = np.random.RandomState(42)
        gibbs = GibbsResult(
            tau_sq_samples=rng.uniform(0, 1, 100),
            pi_samples=np.zeros(100),
            beta_samples=rng.normal(0, 1, (100, 10)),
            ppa=0.5,
            tau_sq_posterior_mean=0.2,
            tau_sq_posterior_var=0.01,
            effective_sample_size=100.0,
            acceptance_rate=0.4,
        )
        w_res = SSgBLUPWeights().compute_weights(gibbs, rng.uniform(0.1, 1.5, 10))
        assert np.isclose(np.mean(w_res.weights), 1.0)
        assert (w_res.weights >= 0).all()


# ═══════════════════════════════════════════════════════════════════════
# 3. Pipeline Resilience Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineResilience:
    """Tests for fallback mechanisms when external tools are missing."""

    def test_alphasimr_fallback(self):
        """AlphaSimR fallback must produce a valid 21-row DataFrame."""
        runner = AlphaSimRRunner("/path/to/nonexistent_binary")
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert "generation" in df.columns
        assert "ogm_risk_pct" in df.columns
        assert len(df) == 21

    def test_relate_clues_fallback(self):
        """Relate/CLUES fallback must produce allele ages for all targets."""
        runner = RelateCluesRunner("/nonexistent/relate")
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "allele_age_generations" in df.columns
        assert len(df) >= 11  # at least 11 target genes

    def test_fallback_deterministic(self):
        """Two fallback calls must produce identical results."""
        r1 = AlphaSimRRunner("/nope").run()
        r2 = AlphaSimRRunner("/nope").run()
        pd.testing.assert_frame_equal(r1, r2)


# ═══════════════════════════════════════════════════════════════════════
# 4. Visualization Tests
# ═══════════════════════════════════════════════════════════════════════


class TestVisualization:
    """Tests for publication-grade figure generation."""

    def test_table_generation(self, full_data):
        """generate_tables must create a Markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vis = PaperVisualizer(full_data, output_dir=tmpdir)
            vis.generate_tables(table_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "Table1_Targets.md"))
            assert os.path.exists(os.path.join(tmpdir, "Table1_Targets.html"))

    def test_fig1_creation(self, full_data):
        """plot_fig1 must create a PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vis = PaperVisualizer(full_data, output_dir=tmpdir)
            vis.plot_fig1()
            assert os.path.exists(os.path.join(tmpdir, "Fig1_LOEUF_correction.png"))

    def test_fig2_creation(self, full_data):
        """plot_fig2 must create the Manhattan plot PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vis = PaperVisualizer(full_data, output_dir=tmpdir)
            vis.plot_fig2()
            assert os.path.exists(os.path.join(tmpdir, "Fig2_Manhattan_PPA.png"))

    def test_all_figures_creation(self, full_data):
        """plot_all must create all 8 figures plus tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vis = PaperVisualizer(full_data, output_dir=tmpdir)
            vis.plot_all()
            expected = [
                "Fig1_LOEUF_correction.png",
                "Fig2_Manhattan_PPA.png",
                "Fig3_Fetal_Maternal_Tensor_Fixed.png",
                "Fig4_Allele_Age_ARG.png",
                "Fig5_Return_Service.png",
                "Fig6_Power_Comparison.png",
                "Fig7_Prediction_Accuracy.png",
                "Fig8_Forward_Simulation.png",
            ]
            for fname in expected:
                assert os.path.exists(os.path.join(tmpdir, fname)), (
                    f"Missing: {fname}"
                )
