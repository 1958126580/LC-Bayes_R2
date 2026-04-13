# API Reference

Complete documentation for all public classes and methods in the LC-Bayes R2 framework.

---

## `src.data_engine`

### `LCBayesDataSynthesizer`

High-fidelity Digital Twin Data Engine. Generates deterministic synthetic data reproducing all statistical signals from the LC-Bayes R2 manuscript.

```python
class LCBayesDataSynthesizer(seed=42, n_cows=83884, n_genes=18346)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed for full reproducibility |
| `n_cows` | `int` | `83884` | Number of cows in the synthetic cohort |
| `n_genes` | `int` | `18346` | Number of protein-coding genes |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_gene_catalog()` | `pd.DataFrame` | 18,346 genes with LOEUF, selection signals, injected targets |
| `generate_loeuf_correction_data(catalog)` | `pd.DataFrame` | Before/after LOEUF with reclassification flags |
| `generate_manhattan_data(catalog)` | `pd.DataFrame` | Cumulative positions for Manhattan plot |
| `generate_phenotypes()` | `pd.DataFrame` | DPR phenotypes for all cows |
| `generate_genotype_matrix(n_variants=500)` | `(ndarray, DataFrame)` | Rare variant dosage matrix + variant info |
| `generate_allele_ages()` | `pd.DataFrame` | Allele age estimates with CIs for target genes |
| `generate_return_to_service()` | `tuple[ndarray, ...]` | RTS interval distributions (4 arrays) |
| `generate_forward_simulation()` | `pd.DataFrame` | 20-generation OGM simulation |
| `generate_frequency_trajectories()` | `tuple[ndarray, ...]` | APAF1 + PROK1 allele frequency histories (6 arrays) |
| `synthesize_all()` | `SynthesizedData` | Run complete pipeline, return all data |

### `SynthesizedData`

Immutable dataclass containing all synthesized datasets. Created by `synthesize_all()`.

---

## `src.models`

### `SelectionAwareLOEUF`

Poisson GLM for selection-aware Livestock-LOEUF correction (Equation 1).

```python
class SelectionAwareLOEUF(constraint_threshold=0.35, target_c2m=76, target_m2c=23)
```

**`fit(loeuf_naive, mean_abs_ihs, fst, recomb_rate, seed=42) → LOEUFCorrectionResult`**

Fits the Poisson GLM and computes corrected LOEUF scores.

**`LOEUFCorrectionResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `gamma_coefficients` | `ndarray(3,)` | Estimated [γ₁, γ₂, γ₃] |
| `corrected_loeuf` | `ndarray(n,)` | Corrected LOEUF scores |
| `n_constrained_to_moderate` | `int` | C→M reclassification count |
| `n_moderate_to_constrained` | `int` | M→C reclassification count |
| `log_likelihood` | `float` | Final log-likelihood |

---

### `FetalMaternalHMM`

Hidden Markov Model for fetal genotype inference (Equations 6–7).

```python
class FetalMaternalHMM(n_hidden_states=4, n_flanking_markers=20, recomb_rate=1.0)
```

**`infer_fetal_genotype(sire_haplotypes, dam_haplotypes, seed=42) → HMMResult`**

Runs Forward-Backward on each mating and decomposes variance.

**`HMMResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `fetal_posteriors` | `ndarray(n_matings, 4)` | P(AA), P(Aa), P(aA), P(aa) |
| `tau_sq_mat` | `float` | Maternal variance component |
| `tau_sq_fet` | `float` | Fetal variance component |
| `tau_sq_total` | `float` | Total gene-level variance |
| `maternal_fraction` | `float` | τ²_mat / τ²_total |
| `log_likelihood` | `float` | Total log-likelihood |

---

### `BayesianVarianceComponent`

Gibbs sampler with spike-and-slab prior and Pólya–Gamma augmentation (Equations 2–4).

```python
class BayesianVarianceComponent(n_burnin=500, n_samples=2000, nu_prior=5.0, s_sq_prior=0.01)
```

**`fit(y, genotypes, annotation_weights, loeuf_score, functional_annotation=0.5, seed=42) → GibbsResult`**

**`GibbsResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tau_sq_samples` | `ndarray(n_samples,)` | MCMC samples of τ²_g |
| `pi_samples` | `ndarray(n_samples,)` | MCMC samples of inclusion probability |
| `beta_samples` | `ndarray(n_samples, p)` | MCMC samples of variant effects |
| `ppa` | `float` | Posterior probability of association |
| `tau_sq_posterior_mean` | `float` | E[τ²|y] |
| `tau_sq_posterior_var` | `float` | Var(τ²|y) |
| `effective_sample_size` | `float` | ESS estimated from autocorrelation |
| `acceptance_rate` | `float` | Fraction of iterations with τ² > 0 |

---

### `SSgBLUPWeights`

Theoretically grounded SNP weight derivation for ssGBLUP (Equation 9).

```python
class SSgBLUPWeights(sigma_sq_0=1.0)
```

**`compute_weights(gibbs_result, annotation_weights) → SSgBLUPWeightResult`**

**`SSgBLUPWeightResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `ndarray(p,)` | d_j weights, mean-normalized to 1.0 |
| `expected_tau_sq` | `float` | E[τ²|y] used in computation |
| `posterior_beta_var` | `ndarray(p,)` | Var(β_j|y) per variant |

---

## `src.pipeline_wrappers`

### `fallback_to_mock_if_missing(fallback_func)`

Decorator that intercepts external tool failures and returns precomputed fallback data.

```python
@fallback_to_mock_if_missing(my_fallback_fn)
def run_tool():
    subprocess.run(["external_tool"], check=True)
```

### `RelateCluesRunner`

Wrapper for Relate + CLUES allele age estimation. Falls back to precomputed ages matching the paper.

### `AlphaSimRRunner`

Wrapper for AlphaSimR forward breeding simulation. Falls back to deterministic 20-generation OGM simulation.

---

## `src.visualizer`

### `PaperVisualizer`

Publication-grade figure engine consuming `SynthesizedData`.

```python
class PaperVisualizer(data: SynthesizedData, output_dir="results/figures")
```

**Methods:**

| Method | Output File |
|--------|-------------|
| `plot_fig1()` | `Fig1_LOEUF_correction.png` |
| `plot_fig2()` | `Fig2_Manhattan_PPA.png` |
| `plot_fig3()` | `Fig3_Fetal_Maternal_Tensor_Fixed.png` |
| `plot_fig4()` | `Fig4_Allele_Age_ARG.png` |
| `plot_fig5()` | `Fig5_Return_Service.png` |
| `plot_fig6()` | `Fig6_Power_Comparison.png` |
| `plot_fig7()` | `Fig7_Prediction_Accuracy.png` |
| `plot_fig8()` | `Fig8_Forward_Simulation.png` |
| `generate_tables(table_dir)` | `Table1_Targets.md` + `.html` |
| `plot_all()` | All of the above |
