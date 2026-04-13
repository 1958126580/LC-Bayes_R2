# Mathematical Foundations

This page provides the complete mathematical derivation of the LC-Bayes R2 framework, directly corresponding to the equations in the manuscript.

## 1. Selection-Aware Livestock-LOEUF (Equation 1)

### Problem

Naive LOEUF scores underestimate constraint for genes near selective sweeps. Sweep-proximal genes show artificially depleted observed pLoF counts because linked deleterious variants are purged alongside the sweep target.

### Model

We model the expected number of pLoF variants per gene using a Poisson GLM with exponential link:

$$E_g = 2n \sum_v \mu_v \cdot c_v \cdot \exp(\gamma_1 |\text{iHS}|_g + \gamma_2 F_{\text{ST},g} + \gamma_3 \rho_g)$$

where:

- $|\text{iHS}|_g$ is the mean absolute integrated haplotype score across the gene body
- $F_{\text{ST},g}$ is the cross-breed fixation index
- $\rho_g$ is the local recombination rate (cM/Mb)
- $\gamma_1, \gamma_2, \gamma_3$ are regression coefficients estimated by maximum likelihood

### Estimation

The negative log-likelihood is:

$$-\ell(\boldsymbol{\gamma}) = -\sum_{g=1}^{G} \left[ y_g \log \mu_g - \mu_g - \log(y_g!) \right]$$

where $\mu_g = E_g^{\text{neutral}} \cdot \exp(\mathbf{x}_g^T \boldsymbol{\gamma})$. We optimize via L-BFGS-B with box constraints $\gamma_k \in [-1, 1]$.

### Correction

The selection-aware LOEUF is computed as:

$$\text{LOEUF}_g^{\text{sel}} = \text{LOEUF}_g^{\text{naive}} \cdot \frac{\exp(\mathbf{x}_g^T \hat{\boldsymbol{\gamma}})}{\text{median}_g \exp(\mathbf{x}_g^T \hat{\boldsymbol{\gamma}})}$$

This reclassifies exactly **76 genes** from Constrained → Moderate (sweep-proximal) and **23 genes** from Moderate → Constrained.

---

## 2–4. Bayesian Variance-Component Model

### Gene-Level Variance (Equation 2)

For each gene $g$ with $p$ rare variants, the variant effects follow:

$$\beta_v \mid \tau_g^2 \sim \mathcal{N}(0, \, w_v \tau_g^2)$$

where $w_v$ are annotation-derived weights (CADD, PhyloP, FarmGTEx).

### Spike-and-Slab Prior (Equation 3)

The gene-level variance uses a mixture prior:

$$\tau_g^2 \sim \pi_g \cdot \text{Inv-}\chi^2(\nu, s^2) + (1 - \pi_g) \cdot \delta_0$$

The spike component $\delta_0$ assigns exactly zero variance to non-causal genes.

### Constraint-Informed Inclusion (Equation 4)

The inclusion probability $\pi_g$ is linked to selection constraint:

$$\text{logit}(\pi_g) = \alpha_0 + \alpha_1 f(\text{LOEUF}_g^{\text{sel}}) + \alpha_2 a_g$$

where $a_g$ is a functional annotation evidence score. Lower LOEUF (more constrained) → higher prior probability of association.

### Gibbs Sampler

The Gibbs sampler iterates:

1. **Sample inclusion** $I_g$ from $\text{Bernoulli}(\pi_g^*)$ using the Bayes factor
2. **Sample** $\boldsymbol{\beta} \mid \tau_g^2, \sigma^2, \mathbf{y}$ from conjugate normal posterior
3. **Sample** $\tau_g^2 \mid \boldsymbol{\beta}$ from inverse chi-squared posterior
4. **Sample** $\sigma^2$ from inverse gamma posterior
5. **Update** $\pi_g$ via Pólya–Gamma augmentation

---

## 5. Two-Kernel Mixed Model (Equation 5)

The full phenotype model is:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{G}\boldsymbol{\beta}_{\text{gene}} + \mathbf{Z}_1 \mathbf{u}_1 + \mathbf{Z}_2 \mathbf{u}_2 + \boldsymbol{\varepsilon}$$

where $\mathbf{u}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{G}_1 \sigma_1^2)$ and $\mathbf{u}_2 \sim \mathcal{N}(\mathbf{0}, \mathbf{G}_2 \sigma_2^2)$ are polygenic random effects from two genomic relationship matrices.

---

## 6–7. Fetal-Maternal HMM

### Fetal Genotype Inference (Equation 6)

For service sire $i$ and dam $j$, the fetal diploid genotype is a latent variable with posterior:

$$P(g_{ij}^{\text{fetus}} = k \mid \mathbf{h}_i^{\text{sire}}, \mathbf{h}_j^{\text{dam}}, \mathbf{M}_{\text{flanking}})$$

for $k \in \{AA, Aa, aA, aa\}$.

**Transition matrix** uses the Haldane mapping function:

$$r = \frac{1}{2}(1 - e^{-2d})$$

where $d$ is the inter-marker distance in Morgans.

**Emission probabilities** incorporate Minimac4 imputation uncertainty:

$$P(\text{obs} \mid \text{state } k) \propto \phi(\text{obs}; \mu_k, \sigma_{\text{dosage}}^2)$$

The Forward-Backward algorithm computes exact posteriors with numerical stability via scaling.

### Variance Decomposition (Equation 7)

The gene-level variance decomposes into:

$$\tau_g^2 = \tau_{g,\text{mat}}^2 + \tau_{g,\text{fet}}^2$$

**Identifiability**: $\tau_{\text{fet}}^2$ depends on the sire × dam mating combination; $\tau_{\text{mat}}^2$ depends only on the dam genotype.

---

## 9. ssGBLUP Weight Derivation (Equation 9)

The theoretically optimal SNP weight for single-step GBLUP is:

$$d_j = 1 + \frac{E(\tau^2 \mid \mathbf{y}) \cdot w_j + \text{Var}(\beta_j \mid \mathbf{y})}{\sigma_0^2}$$

These weights are normalized to have mean 1.0 and achieve **98.5%** of the full Bayesian model's accuracy within the computationally efficient ssGBLUP framework.
