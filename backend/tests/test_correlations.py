"""FOM_PROOF Sec. 7: correlation blocks and the complete-case rule."""

from __future__ import annotations

import pytest

from cnms_fom.db.enums import CorrelationBlock, Transform
from cnms_fom.fom_engine.correlations import (
    complete_case_mask,
    correlation_block,
    correlation_cell,
    spearman_from_ranks,
)


def test_complete_case_is_pairwise():
    """Eq. (33): the mask is computed per pair, not once for the matrix."""
    x = [1.0, None, 3.0, 4.0]
    y = [1.0, 2.0, None, 4.0]
    assert list(complete_case_mask(x, y)) == [True, False, False, True]


def test_cell_reports_its_own_n_and_members():
    cell = correlation_cell(
        [1.0, 2.0, None, 4.0, 5.0],
        [2.0, 4.1, 6.0, None, 10.2],
        block=CorrelationBlock.SP,
        x_key="V_fu",
        y_key="k",
        material_ids=["a", "b", "c", "d", "e"],
        permutations=199,
        seed=0,
    )
    assert cell.n_complete == 3
    assert cell.material_ids == ["a", "b", "e"]


def test_spearman_is_pearson_on_ranks():
    """Eq. (32), verified against the definition rather than a library default."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 8.0, 27.0, 64.0, 125.0]
    cell = correlation_cell(
        x, y, block=CorrelationBlock.SP, x_key="x", y_key="y", permutations=199, seed=0
    )
    assert cell.spearman_rho == pytest.approx(spearman_from_ranks(x, y))
    assert cell.spearman_rho == pytest.approx(1.0)


def test_too_few_points_yields_no_coefficient():
    cell = correlation_cell(
        [1.0, 2.0], [1.0, 2.0], block=CorrelationBlock.SP, x_key="x", y_key="y"
    )
    assert cell.pearson_r is None
    assert cell.outcome == "inconclusive"


def test_zero_variance_yields_no_coefficient():
    cell = correlation_cell(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        block=CorrelationBlock.SP,
        x_key="x",
        y_key="y",
    )
    assert cell.pearson_r is None


def test_block_is_fdr_corrected_within_itself():
    columns_x = {"s1": [1.0, 2, 3, 4, 5, 6], "s2": [6.0, 5, 4, 3, 2, 1]}
    columns_y = {"p1": [1.0, 2, 3, 4, 5, 6], "p2": [1.0, 1, 2, 2, 3, 3]}
    cells = correlation_block(
        columns_x,
        columns_y,
        block=CorrelationBlock.SP,
        material_ids=list("abcdef"),
        permutations=199,
        seed=0,
    )
    assert len(cells) == 4
    assert all(cell.q_fdr is not None for cell in cells)
    assert all(cell.q_fdr >= cell.p_permutation - 1e-12 for cell in cells)


def test_symmetric_block_skips_the_diagonal():
    """R_SS is symmetric; only the upper triangle is a distinct hypothesis."""
    columns = {"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10], "c": [5.0, 3, 1, 4, 2]}
    cells = correlation_block(
        columns, columns, block=CorrelationBlock.SS, permutations=99, seed=0
    )
    assert len(cells) == 3  # ab, ac, bc
    assert all(cell.x_key != cell.y_key for cell in cells)


def test_preregistered_sign_flows_into_the_result():
    cells = correlation_block(
        {"omega_TO_min": [100.0, 200, 300, 400, 500, 600]},
        {"eps_ionic": [60.0, 30, 20, 15, 12, 10]},
        block=CorrelationBlock.SP,
        transforms={"omega_TO_min": Transform.LOG10, "eps_ionic": Transform.NONE},
        permutations=999,
        seed=0,
        hypotheses={("omega_TO_min", "eps_ionic"): ("-", "Softer polar modes raise eps_ionic.")},
    )
    cell = cells[0]
    assert cell.predicted_sign == "-"
    assert cell.pearson_r < 0
    assert cell.outcome == "supports"
