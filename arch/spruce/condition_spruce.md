# Condition number report — spruce

**Tree:** spruce  
**Source directory:** `/home/vstein/Documents/PhD/Code/coupledOpt/arch/spruce/drutes_run`  
**Norm used:** 2  
**Number of matrices:** 3  

> **Interpretation:** κ ≈ 10^k means roughly *k* decimal digits of accuracy are lost in a direct linear solve.  Double precision provides ~16 significant digits, so κ < 10^8 is generally acceptable.

## spruce_dry

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 3.695487e+01 |
| log₁₀(κ) — digits of precision lost | 1.57 |
| Largest singular value σ_max | 1.640851e+00 |
| Smallest singular value σ_min | 4.440149e-02 |
| Assessment | well-conditioned |

## spruce_first-rain

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 6.335673e+01 |
| log₁₀(κ) — digits of precision lost | 1.80 |
| Largest singular value σ_max | 1.346130e+00 |
| Smallest singular value σ_min | 2.124683e-02 |
| Assessment | well-conditioned |

## spruce_second-rain

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 5.741970e+00 |
| log₁₀(κ) — digits of precision lost | 0.76 |
| Largest singular value σ_max | 1.115408e+00 |
| Smallest singular value σ_min | 1.942554e-01 |
| Assessment | well-conditioned |

## Summary

| Matrix | κ | log₁₀(κ) | Assessment |
|---|---|---|---|
| spruce_dry | 3.695e+01 | 1.57 | well-conditioned |
| spruce_first-rain | 6.336e+01 | 1.80 | well-conditioned |
| spruce_second-rain | 5.742e+00 | 0.76 | well-conditioned |
