"""One Python analysis step against one step of the OpenFOAM implementation.

The reference file is produced as described in docs/theory.md, Section 7.
The test is skipped until that file exists.
"""
from pathlib import Path
import numpy as np
import pytest
from pr_enkf import analysis

REF = Path(__file__).parent / "data" / "cpp_reference_step.npz"


@pytest.mark.skipif(not REF.exists(), reason="C++ reference step not yet generated")
def test_matches_cpp_analysis_step():
    r = np.load(REF)
    alpha_a = analysis(r["alpha"], r["q"], r["d_pert"], float(r["sigma_d"]),
                       p_pert=r["p_pert"], gamma=r["gamma"],
                       inflation=float(r["inflation"]))
    np.testing.assert_allclose(alpha_a, r["alpha_analysis"], rtol=1e-10, atol=1e-12)
