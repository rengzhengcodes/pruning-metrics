"""Classical MDS (PCoA) and distance-matrix diagnostics.

These are the shared preprocessing utilities of the sublibrary: half of the
reducers (see each module's ``SPEC.needs_coords``) have no
``metric="precomputed"`` mode and must be fed Euclidean coordinates, which
:func:`classical_mds_coords` derives from a distance matrix by Torgerson
double-centering.

That derivation is lossy whenever the input is not Euclidean: the negative
eigenvalues of the double-centered Gram matrix are discarded outright.  For
the v2 KLD matrices this throws away ~25% of the eigenvalue mass and over
half the dimensions — enough to visibly distort the resulting picture.  Use
:func:`mds_spectrum` to quantify the loss and report it alongside any
coordinate-based figure, rather than letting a reader assume all reducers
saw the same object.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "classical_mds_coords",
    "complete_submatrix_indices",
    "mds_spectrum",
]

#: Relative eigenvalue cutoff shared by :func:`classical_mds_coords` and
#: :func:`mds_spectrum`.  Sharing the constant is what keeps the spectrum's
#: ``n_pos`` equal to the number of dimensions the coordinates actually get.
_TOL_REL: float = 1e-9


def _double_center(D: np.ndarray) -> np.ndarray:
    """Torgerson double-centering: B = -1/2 * J D^2 J.

    The Gram matrix of a centered configuration whose pairwise Euclidean
    distances reproduce ``D`` (exactly when ``D`` is Euclidean).  Shared by
    :func:`classical_mds_coords` and :func:`mds_spectrum` so the two are
    guaranteed to analyse the same matrix.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.

    Returns
    -------
    np.ndarray
        The ``(n, n)`` double-centered Gram matrix.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D**2) @ J


def classical_mds_coords(D: np.ndarray) -> np.ndarray:
    """Embeds a distance matrix into Euclidean coordinates via classical MDS.

    Torgerson double-centering: B = -1/2 * J D^2 J is the Gram matrix of a
    centered configuration whose pairwise Euclidean distances reproduce D
    (exactly when D is Euclidean).  Dimensions with non-positive
    eigenvalues — the non-Euclidean part of D — are dropped.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.

    Returns
    -------
    np.ndarray
        ``(n, k)`` coordinates, where ``k`` is the number of positive
        eigenvalues (padded to >= 2).
    """
    n = D.shape[0]
    evals, evecs = np.linalg.eigh(_double_center(D))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > max(evals.max(), 0.0) * _TOL_REL
    X = evecs[:, keep] * np.sqrt(evals[keep])
    if X.shape[1] < 2:  # degenerate D — pad so 2-D reducers still run
        X = np.hstack([X, np.zeros((n, 2 - X.shape[1]))])
    return X


def mds_spectrum(D: np.ndarray) -> dict[str, float]:
    """Measures how far a distance matrix is from being Euclidean.

    Runs the same double-centering as :func:`classical_mds_coords` but
    reports the eigenvalue spectrum instead of the coordinates.
    ``neg_ratio`` is the headline number: the total magnitude of the
    *discarded* negative eigenvalues relative to the retained positive
    ones.  It is 0 for a perfectly Euclidean ``D`` and grows as ``D``
    violates the triangle inequality.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.

    Returns
    -------
    dict[str, float]
        A dict with the keys:

        * ``n_pos`` — number of positive eigenvalues, i.e. how many
          dimensions survive into :func:`classical_mds_coords`' output
          (out of ``n``).
        * ``pos_mass``, ``neg_mass`` — summed magnitudes of the positive
          and negative eigenvalues.
        * ``neg_ratio`` — ``neg_mass / pos_mass``; ``0.0`` when
          ``pos_mass`` is 0.
        * ``var_2d`` — fraction of ``pos_mass`` captured by the top 2
          dimensions: an upper bound on how much of the (Euclidean part
          of the) structure any 2-D picture can possibly show.
    """
    evals = np.linalg.eigvalsh(_double_center(D))

    # Same relative cutoff classical_mds_coords uses, so n_pos really is the
    # number of dimensions its output will have. An absolute `> 0` test would
    # instead count every ~1e-14 rounding artefact as a genuine dimension
    # (e.g. 24 "dimensions" for data that is exactly 4-dimensional).
    tol = max(float(evals.max()), 0.0) * _TOL_REL
    pos = evals[evals > tol]
    pos_mass = float(pos.sum())
    # abs(), not negation: summing an empty selection yields -0.0, which would
    # print as a nonsensical "-0.0000" neg_ratio for a perfectly Euclidean D.
    neg_mass = abs(float(evals[evals < -tol].sum()))
    top2 = float(np.sort(pos)[::-1][:2].sum()) if pos.size else 0.0
    return {
        "n_pos": int(pos.size),
        "pos_mass": pos_mass,
        "neg_mass": neg_mass,
        "neg_ratio": (neg_mass / pos_mass) if pos_mass > 0 else 0.0,
        "var_2d": (top2 / pos_mass) if pos_mass > 0 else 0.0,
    }


def complete_submatrix_indices(
    D: np.ndarray, *, counts: np.ndarray | None = None
) -> np.ndarray:
    """Finds a large principal submatrix with no unobserved pairs.

    The v2 distance matrices are accumulated over evaluation tasks, and a
    pair of variants that never appeared on a common task is left at
    ``0.0`` — indistinguishable, numerically, from two identical models.
    That is far from harmless: a row of zeros makes one point sit at
    distance 0 from everything, which collapses Isomap's geodesic graph
    entirely and silently corrupts every neighbourhood-based measure.

    This greedily drops the row with the most unobserved partners until
    none remain, which is a good approximation to the maximum complete
    submatrix (exactly solving it is the NP-hard maximum-clique problem)
    and is exact in the common case where a few variants are missing
    outright.

    Parameters
    ----------
    D:
        Symmetric ``(n, n)`` distance matrix.
    counts:
        Optional ``(n, n)`` matrix of how many tasks contributed to
        each entry.  When given, "unobserved" means ``counts == 0``,
        which is exact.  Without it, a zero off-diagonal distance is
        treated as unobserved.

    Returns
    -------
    np.ndarray
        Sorted indices of the retained rows/columns.  May be empty.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    missing = (np.asarray(counts) == 0) if counts is not None else (D == 0.0)
    missing = missing.copy()
    np.fill_diagonal(missing, False)

    keep = np.ones(n, dtype=bool)
    while keep.any():
        sub = missing[np.ix_(keep, keep)]
        per_row = sub.sum(axis=1)
        if per_row.max(initial=0) == 0:
            break
        kept_idx = np.flatnonzero(keep)
        keep[kept_idx[int(np.argmax(per_row))]] = False
    return np.flatnonzero(keep)
