"""
condition.py
------------
Reads all .mtx sparse matrix files from a given DRUtES run directory,
assembles each matrix as a dense NumPy array, and computes its condition
number.  Results are printed to the terminal and written to a Markdown file.

.mtx file format used by DRUtES (NOT the standard Matrix Market format):
  - Line 1 : <nrows>  <ncols>  <nnz>   (matrix dimensions and non-zero count)
  - Lines 2…nnz+1 : <row>  <col>  <value>   (1-based COO triplets)

Condition number κ(A) = ||A|| · ||A⁻¹||
  A perfectly conditioned matrix has κ = 1.
  If κ ≈ 10^k, you lose roughly k decimal digits of accuracy in the solution.
  Rule of thumb: κ > 10^8 is considered severely ill-conditioned for
  double-precision arithmetic (machine epsilon ≈ 2.2e-16).

Usage:
  python3 condition.py               # defaults to tree = "spruce"
  python3 condition.py beech         # switch tree; reads arch/beech/drutes_run/
  python3 condition.py spruce 2-norm # override the norm used (default: 2)

Supported norm strings (passed straight to numpy.linalg.cond):
  2        – largest / smallest singular value (default, most informative)
  -2       – smallest / largest singular value
  1        – max absolute column sum
  -1       – min absolute column sum
  inf      – max absolute row sum
  -inf     – min absolute row sum
  fro      – Frobenius norm ratio (cheap but less tight)
"""

import sys
import os
import glob
import numpy as np


# ---------------------------------------------------------------------------
# Configuration – change `tree` here or pass it as the first CLI argument
# ---------------------------------------------------------------------------
TREE = sys.argv[1] if len(sys.argv) > 1 else "spruce"

# Optional second argument: which matrix norm to use for numpy.linalg.cond.
# "2" (the spectral norm) is the most informative but also the most expensive
# because it requires a full SVD.  For large matrices you might prefer "fro".
NORM_ARG = sys.argv[2] if len(sys.argv) > 2 else 2   # int 2 = spectral norm

# Resolve to a numeric type when the user passes "2", "-2", "1", etc., so
# numpy gets the right type (int vs. str like "inf"/"-inf"/"fro").
try:
    NORM_ARG = int(NORM_ARG)
except (ValueError, TypeError):
    pass  # keep as string for "inf", "-inf", "fro"

# Base directory of this repo (the directory where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the run directory that holds the .mtx files
MTX_DIR = os.path.join(BASE_DIR, "arch", TREE, "drutes_run")

# Output Markdown file – named after the tree so results don't overwrite each other
OUTPUT_MD = os.path.join(BASE_DIR, f"condition_{TREE}.md")


# ---------------------------------------------------------------------------
# Helper: parse one .mtx file into a dense NumPy matrix
# ---------------------------------------------------------------------------
def load_mtx(filepath):
    """
    Parse a DRUtES-format .mtx file and return a dense (nrows × ncols) array.

    Parameters
    ----------
    filepath : str
        Absolute path to the .mtx file.

    Returns
    -------
    A : np.ndarray, shape (nrows, ncols), dtype float64
        Dense matrix assembled from the COO triplets.
    """
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # --- Header line: matrix dimensions and number of stored non-zeros ----
    header = lines[0].split()
    nrows = int(header[0])
    ncols = int(header[1])
    nnz   = int(header[2])   # expected number of non-zero entries

    # Allocate a zero matrix; we will fill in only the stored entries.
    # Using float64 matches DRUtES' double-precision Fortran arithmetic.
    A = np.zeros((nrows, ncols), dtype=np.float64)

    # --- Fill in COO triplets (1-based → convert to 0-based Python indices) -
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue   # skip blank or malformed lines
        row = int(parts[0]) - 1    # convert from 1-based to 0-based
        col = int(parts[1]) - 1
        val = float(parts[2])
        A[row, col] = val

    # Sanity check: warn if the file had fewer entries than the header promised
    actual_nnz = np.count_nonzero(A)
    if actual_nnz != nnz:
        print(f"  [WARNING] {os.path.basename(filepath)}: header says {nnz} "
              f"non-zeros but found {actual_nnz} after assembly.")

    return A


# ---------------------------------------------------------------------------
# Helper: compute and format the condition number diagnostics for one matrix
# ---------------------------------------------------------------------------
def analyse_matrix(A, name, norm):
    """
    Compute the condition number of matrix A and return a result dict.

    numpy.linalg.cond uses the definition:
        κ(A, p) = ||A||_p · ||A⁻¹||_p
    For the default spectral norm (p=2) this equals σ_max / σ_min where
    σ are the singular values – i.e. how much the matrix stretches vs.
    compresses vectors in the worst direction.

    Parameters
    ----------
    A    : np.ndarray  – the dense matrix
    name : str         – human-readable label (filename without extension)
    norm : int or str  – norm identifier passed to numpy.linalg.cond

    Returns
    -------
    dict with keys: name, shape, rank, cond, log10_cond, singular_max,
                    singular_min, assessment
    """
    nrows, ncols = A.shape

    # --- Condition number via NumPy (uses LAPACK dgesvd for norm=2) --------
    kappa = np.linalg.cond(A, p=norm)

    # log₁₀ of the condition number gives the number of digits of precision
    # that are potentially lost in a linear solve with this matrix.
    log10_kappa = np.log10(kappa) if kappa > 0 else float("inf")

    # Compute singular values explicitly so we can report σ_max and σ_min.
    # This is redundant for norm=2 (cond already does SVD internally) but
    # makes the diagnostics richer and norm-independent.
    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max = singular_values[0]           # largest singular value
    sigma_min = singular_values[-1]          # smallest singular value

    # Numerical rank: count singular values above machine-epsilon threshold
    eps = np.finfo(np.float64).eps
    rank = int(np.sum(singular_values > sigma_max * max(nrows, ncols) * eps))

    # Human-readable assessment of how ill-conditioned the matrix is
    if kappa < 1e4:
        assessment = "well-conditioned"
    elif kappa < 1e8:
        assessment = "mildly ill-conditioned"
    elif kappa < 1e12:
        assessment = "severely ill-conditioned (significant precision loss)"
    else:
        assessment = "EXTREMELY ill-conditioned (results likely unreliable)"

    return {
        "name":        name,
        "shape":       (nrows, ncols),
        "rank":        rank,
        "cond":        kappa,
        "log10_cond":  log10_kappa,
        "sigma_max":   sigma_max,
        "sigma_min":   sigma_min,
        "assessment":  assessment,
    }


# ---------------------------------------------------------------------------
# Helper: format one result as a Markdown section string
# ---------------------------------------------------------------------------
def format_result_md(r, norm_label):
    lines = [
        f"## {r['name']}",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Matrix size | {r['shape'][0]} × {r['shape'][1]} |",
        f"| Numerical rank | {r['rank']} |",
        f"| Condition number κ (norm={norm_label}) | {r['cond']:.6e} |",
        f"| log₁₀(κ) — digits of precision lost | {r['log10_cond']:.2f} |",
        f"| Largest singular value σ_max | {r['sigma_max']:.6e} |",
        f"| Smallest singular value σ_min | {r['sigma_min']:.6e} |",
        f"| Assessment | {r['assessment']} |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: format one result for terminal output (plain text)
# ---------------------------------------------------------------------------
def format_result_terminal(r, norm_label):
    sep = "-" * 60
    lines = [
        sep,
        f"  Matrix : {r['name']}",
        f"  Size   : {r['shape'][0]} × {r['shape'][1]}   rank = {r['rank']}",
        f"  κ (norm={norm_label}) = {r['cond']:.6e}   "
        f"log₁₀(κ) = {r['log10_cond']:.2f}",
        f"  σ_max  = {r['sigma_max']:.6e}",
        f"  σ_min  = {r['sigma_min']:.6e}",
        f"  → {r['assessment']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\nCondition number analysis  |  tree = {TREE}")
    print(f"Reading .mtx files from : {MTX_DIR}")
    print(f"Norm used               : {NORM_ARG}")
    print(f"Output Markdown file    : {OUTPUT_MD}")

    # Collect all .mtx files in the run directory, sorted alphabetically
    mtx_files = sorted(glob.glob(os.path.join(MTX_DIR, "*.mtx")))

    if not mtx_files:
        print(f"\n[ERROR] No .mtx files found in {MTX_DIR}")
        sys.exit(1)

    print(f"Found {len(mtx_files)} .mtx file(s).\n")

    results = []   # accumulate result dicts for Markdown output

    for filepath in mtx_files:
        # Strip path and extension to get a clean matrix name for labelling
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Processing: {name} …", flush=True)

        # Load sparse triplets → dense NumPy array
        A = load_mtx(filepath)

        # Compute condition number and related diagnostics
        r = analyse_matrix(A, name, NORM_ARG)

        # Print to terminal immediately so the user can watch progress
        print(format_result_terminal(r, norm_label=str(NORM_ARG)))
        print()

        results.append(r)

    # -----------------------------------------------------------------------
    # Write Markdown report
    # -----------------------------------------------------------------------
    md_lines = [
        f"# Condition number report — {TREE}",
        "",
        f"**Tree:** {TREE}  ",
        f"**Source directory:** `{MTX_DIR}`  ",
        f"**Norm used:** {NORM_ARG}  ",
        f"**Number of matrices:** {len(results)}  ",
        "",
        "> **Interpretation:** κ ≈ 10^k means roughly *k* decimal digits of "
        "accuracy are lost in a direct linear solve.  Double precision provides "
        "~16 significant digits, so κ < 10^8 is generally acceptable.",
        "",
    ]
    for r in results:
        md_lines.append(format_result_md(r, norm_label=str(NORM_ARG)))

    # Summary table at the end for quick comparison across all matrices
    md_lines += [
        "## Summary",
        "",
        "| Matrix | κ | log₁₀(κ) | Assessment |",
        "|---|---|---|---|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['name']} | {r['cond']:.3e} | {r['log10_cond']:.2f} "
            f"| {r['assessment']} |"
        )
    md_lines.append("")

    with open(OUTPUT_MD, "w") as fh:
        fh.write("\n".join(md_lines))

    print(f"\nResults written to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
