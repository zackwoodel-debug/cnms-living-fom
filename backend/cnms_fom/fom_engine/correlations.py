"""Correlation calculations (FOM_PROOF Sec. 7).

Two rules shape this module:

  * **Complete-case is per pair** (Eq. 33, Sec. 7.3).  There is no run-level n
    anywhere in the API, because a correlation matrix computed over different
    complete-case sets has no single sample size.  Every cell carries its own
    ``n_complete`` and the exact material identifiers that entered it.
  * **Report Pearson and Spearman together** (Sec. 7.2).  Disagreement between
    them is information: it points at nonlinearity, monotone-but-curved
    relationships, or outlier sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from cnms_fom.db.enums import CorrelationBlock, Transform

from .inference import benjamini_hochberg, classify_outcome, permutation_p_value
from .normalization import apply_transform


@dataclass
class CorrelationCell:
    """One cell of a correlation block — FOM_PROOF Table 6 in full."""

    block: CorrelationBlock
    x_key: str
    y_key: str
    x_transform: Transform
    y_transform: Transform
    n_complete: int
    pearson_r: float | None = None
    spearman_rho: float | None = None
    p_permutation: float | None = None
    q_fdr: float | None = None
    predicted_sign: str | None = None
    mechanism: str | None = None
    outcome: str | None = None
    material_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "block": self.block.value,
            "x_key": self.x_key,
            "y_key": self.y_key,
            "x_transform": self.x_transform.value,
            "y_transform": self.y_transform.value,
            "n_complete": self.n_complete,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_rho,
            "p_permutation": self.p_permutation,
            "q_fdr": self.q_fdr,
            "predicted_sign": self.predicted_sign,
            "mechanism": self.mechanism,
            "outcome": self.outcome,
            "material_ids": self.material_ids,
        }


def complete_case_mask(x, y) -> np.ndarray:
    """Eq. (33): the indicator I[X_i != NA and Y_i != NA] for this pair only."""
    xa = np.asarray([np.nan if v is None else float(v) for v in x], dtype=float)
    ya = np.asarray([np.nan if v is None else float(v) for v in y], dtype=float)
    if xa.shape != ya.shape:
        raise ValueError(f"x and y must align; got {xa.shape} and {ya.shape}.")
    return np.isfinite(xa) & np.isfinite(ya)


def _prepare(values, transform: Transform, mask: np.ndarray) -> np.ndarray:
    selected = [float(v) for v, keep in zip(values, mask, strict=True) if keep]
    if transform is Transform.NONE:
        return np.asarray(selected, dtype=float)
    return np.asarray([apply_transform(v, transform) for v in selected], dtype=float)


def correlation_cell(
    x,
    y,
    *,
    block: CorrelationBlock,
    x_key: str,
    y_key: str,
    x_transform: Transform = Transform.NONE,
    y_transform: Transform = Transform.NONE,
    material_ids: list[str] | None = None,
    permutations: int = 10_000,
    seed: int | None = None,
    predicted_sign: str | None = None,
    mechanism: str | None = None,
) -> CorrelationCell:
    """Compute one correlation cell with everything Table 6 requires.

    ``x`` and ``y`` are full-length sequences aligned with ``material_ids``;
    ``None`` entries are NA and are dropped pairwise, never imputed.
    """
    mask = complete_case_mask(x, y)
    n = int(mask.sum())
    ids = (
        [mid for mid, keep in zip(material_ids, mask, strict=True) if keep]
        if material_ids is not None
        else []
    )

    cell = CorrelationCell(
        block=block,
        x_key=x_key,
        y_key=y_key,
        x_transform=x_transform,
        y_transform=y_transform,
        n_complete=n,
        predicted_sign=predicted_sign,
        mechanism=mechanism,
        material_ids=ids,
    )
    if n < 3:
        # Sec. 15.2: a two-material comparison does not establish a population
        # correlation, so we do not produce a coefficient at all.
        cell.outcome = "inconclusive"
        return cell

    xs = _prepare(x, x_transform, mask)
    ys = _prepare(y, y_transform, mask)
    if xs.std() == 0.0 or ys.std() == 0.0:
        cell.outcome = "inconclusive"
        return cell

    cell.pearson_r = float(pearsonr(xs, ys)[0])
    cell.spearman_rho = float(spearmanr(xs, ys)[0])
    _, cell.p_permutation = permutation_p_value(
        xs, ys, b=permutations, method="pearson", seed=seed
    )
    return cell


def correlation_block(
    columns_x: dict[str, list],
    columns_y: dict[str, list],
    *,
    block: CorrelationBlock,
    transforms: dict[str, Transform] | None = None,
    material_ids: list[str] | None = None,
    permutations: int = 10_000,
    seed: int | None = None,
    hypotheses: dict[tuple[str, str], tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> list[CorrelationCell]:
    """Compute a whole block (Eqs. 35-38) and FDR-correct within it.

    The family for the BH correction is the block: that is the set of hypotheses
    actually being tested together.  Correcting across unrelated blocks would
    dilute power; not correcting at all would violate Sec. 8.2.

    ``hypotheses`` maps ``(x_key, y_key)`` to ``(predicted_sign, mechanism)``
    and must be pre-registered before this is called (Sec. 4.2).
    """
    transforms = transforms or {}
    hypotheses = hypotheses or {}
    symmetric = columns_x is columns_y

    cells: list[CorrelationCell] = []
    for xi, (x_key, x_values) in enumerate(columns_x.items()):
        for yi, (y_key, y_values) in enumerate(columns_y.items()):
            if symmetric and yi <= xi:
                continue  # upper triangle only; the diagonal is trivially 1
            sign, mechanism = hypotheses.get((x_key, y_key), (None, None))
            cells.append(
                correlation_cell(
                    x_values,
                    y_values,
                    block=block,
                    x_key=x_key,
                    y_key=y_key,
                    x_transform=transforms.get(x_key, Transform.NONE),
                    y_transform=transforms.get(y_key, Transform.NONE),
                    material_ids=material_ids,
                    permutations=permutations,
                    seed=seed,
                    predicted_sign=sign,
                    mechanism=mechanism,
                )
            )

    q_values = benjamini_hochberg([c.p_permutation for c in cells])
    for cell, q in zip(cells, q_values, strict=True):
        cell.q_fdr = None if np.isnan(q) else float(q)
        cell.outcome = classify_outcome(
            cell.pearson_r, cell.q_fdr, cell.predicted_sign, alpha=alpha
        )
    return cells


def spearman_from_ranks(x, y) -> float:
    """Eq. (32): Spearman as Pearson on ranks, spelled out.

    Used by the tests to confirm the library result is the protocol's definition
    and not some other convention.
    """
    xr, yr = rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float))
    return float(pearsonr(xr, yr)[0])
