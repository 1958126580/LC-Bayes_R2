# Analysis Results

## Generated Figures

All figures are rendered at **300 DPI** with Nature/Science publication aesthetics and saved to `results/figures/`.

### Figure 1: Selection-Aware LOEUF Correction

Three panels showing:

- **(a)** Naive LOEUF vs |iHS| — sweep-proximal genes cluster at low LOEUF despite high |iHS|
- **(b)** Corrected LOEUF — 76 reclassified genes now correctly identified as moderate constraint
- **(c)** Reclassification bar chart with BTA breakdown (BTA14: 52 genes near DGAT1 sweep)

### Figure 2: Manhattan Plot of PPA

Genome-wide posterior probability of association across 29 Bos taurus autosomes. Stars mark 7 known lethal haplotype genes; triangles mark 5 novel discoveries. Threshold at PPA = 0.90 identifies 59 significant genes.

### Figure 3: Fetal-Maternal Tensor Architecture

Flowchart illustrating the HMM-based decomposition:

- Service sire and dam haplotypes feed into the HMM
- HMM outputs P(g^fetus = k) for k ∈ {AA, Aa, aA, aa}
- Variance splits into maternal endometrial (PROK1) and fetal recessive (RFC5) components

### Figure 4: Evolutionary Architecture

- **(a)** Allele age estimates from Relate, log-scaled. DOCK8 (~3,200 gen) and ITGB7 (~4,800 gen) predate domestication.
- **(b)** Frequency trajectories from CLUES. APAF1 shows recent hitchhiking on the DGAT1 sweep; PROK1 shows ancient oscillating balancing selection.

### Figure 5: Return-to-Service Intervals

KDE plots comparing control vs carrier × carrier matings:

- **(a)** RFC5: excess mortality at 35–50 days (late embryonic death window)
- **(b)** PROK1: sharpened peak at 21–28 days (implantation failure window)

### Figure 6: Statistical Power Comparison

Grouped bar charts across three constraint categories (High / Moderate / Low LOEUF). LC-Bayes R2 (VC+FM) achieves 81% power for high-constraint genes, compared to 42% for SAIGE-GENE+ and 15% for single-variant LMM.

### Figure 7: Prediction Accuracy

- **(a)** Within-breed Holstein: LC-Bayes R2 achieves +17 pts over standard GBLUP
- **(b)** Cross-breed (trained on Holstein → tested on Jersey, Brown Swiss): +23 pts improvement

### Figure 8: Forward Simulation

20-generation projection comparing three mating strategies:

- Random mating: carrier × carrier risk stable at ~1.5%
- Standard avoidance (7 known lethals): modest reduction to ~1.35%
- LC-Bayes R2 OGM (47 loci): **eliminates risk by generation 12** while retaining 99.6% of genetic gain

## Generated Tables

### Table 1: Novel and Confirmed Recessive Lethal Targets

Markdown and HTML versions saved to `results/tables/`. Contains gene name, chromosome, position, naive LOEUF, PPA, effect class (fetal/maternal), and estimated allele age for all 13 target genes.
