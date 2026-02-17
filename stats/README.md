## Illumination Robustness Statistical Tests

We evaluate illumination robustness of KAN-AINet against the baseline using per-image Dice scores.

**1️⃣ Brightness-Stratified Paired Comparison**

Images are divided into brightness tertiles: dark, medium, bright. Wilcoxon signed-rank test is applied per tertile

Reports include mean Dice (KAN vs Baseline), mean difference, Cohen’s d, and p-value.

**2️⃣ Variance Ratio Test (Brown–Forsythe)**

Evaluates segmentation variability under different illumination, reporting dice variance per brightness bin and variance ratio.

Variance ratio computes as `Var(KAN) / Var(Baseline)`.

### To run the statistical tests

```bash
python illumination_stat_tests.py
```
Requires inputs

```bash
./illumination_robustness_results/combined_per_image.csv
./baseline_illumination_robustness_results/combined_per_image.csv
```
