# Condition number report — beech

**Tree:** beech  
**Source directory:** `/home/vstein/Documents/PhD/Code/coupledOpt/arch/beech/drutes_run`  
**Norm used:** 2  
**Number of matrices:** 3  

> **Interpretation:** κ ≈ 10^k means roughly *k* decimal digits of accuracy are lost in a direct linear solve.  Double precision provides ~16 significant digits, so κ < 10^8 is generally acceptable.

## beech_dry

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 6.355970e+01 |
| log₁₀(κ) — digits of precision lost | 1.80 |
| Largest singular value σ_max | 1.399495e+00 |
| Smallest singular value σ_min | 2.201860e-02 |
| Assessment | well-conditioned |

## beech_first-rain

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 3.190535e+02 |
| log₁₀(κ) — digits of precision lost | 2.50 |
| Largest singular value σ_max | 1.426006e+00 |
| Smallest singular value σ_min | 4.469489e-03 |
| Assessment | well-conditioned |

## beech_second-rain

| Property | Value |
|---|---|
| Matrix size | 96 × 96 |
| Numerical rank | 96 |
| Condition number κ (norm=2) | 2.739880e+02 |
| log₁₀(κ) — digits of precision lost | 2.44 |
| Largest singular value σ_max | 1.422119e+00 |
| Smallest singular value σ_min | 5.190443e-03 |
| Assessment | well-conditioned |

## Summary

| Matrix | κ | log₁₀(κ) | Assessment |
|---|---|---|---|
| beech_dry | 6.356e+01 | 1.80 | well-conditioned |
| beech_first-rain | 3.191e+02 | 2.50 | well-conditioned |
| beech_second-rain | 2.740e+02 | 2.44 | well-conditioned |
