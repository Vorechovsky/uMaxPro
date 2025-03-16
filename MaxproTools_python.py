import numpy as np
import math

"""
This file contains three functions, each in two different versions:
- maxPro
  For calculationg the MaxPro
- delta
  For calculating, what would be the change in MaxPro in case of a swap
- swap
  For actually performing the swap
"""


# --------------------------------
# -- WITHOUT PRECOMPUTED ARRAYS --
# --------------------------------
# Not optimized

def maxPro_np(x: np.ndarray, ns: int, nv: int, periodic = False) -> float:  # single loop
    """
    Compute the (u)MaxPro criterion for a given design matrix.

    This function calculates the MaxPro or uMaxPro criterion for a given 2D design matrix `x`.
    The MaxPro criterion is used in experimental design to ensure good space-filling properties.
    If `periodic` is set to True, the function computes the uMaxPro criterion, which accounts
    for periodic boundary conditions.

    Args:
        x (np.ndarray): A 2D array of shape (ns, nv) representing the design points.
        ns (int): The number of samples (design points).
        nv (int): The number of variables (dimensions).
        periodic (bool, optional): If True, computes the uMaxPro criterion
            (periodic case). If False, computes the MaxPro criterion
            (non-periodic case). Default is False.

    Returns:
        float: The computed (u)MaxPro criterion value.

    Notes:
        - The MaxPro criterion favors designs that are space-filling by maximizing the
          minimum product of squared distances between points.
        - The uMaxPro variant uses periodic distance calculations.
    """

    # Ensure ns and nv are consistent with the shape of x
    ns, nv = x.shape

    maxpro = 0  # Initialize the criterion accumulator

    # Iterate over each design point
    for i in range(ns):
        # Compute the absolute differences between point i and all previous points
        deltas = np.abs(x[i, :] - x[0:i, :])

        if periodic is True:
            # Apply periodic boundary conditions by wrapping distances
            deltas = np.minimum(deltas, 1 - deltas)

        # Square the differences to get squared distances
        dsq = deltas ** 2

        # Compute the reciprocal of the product of squared distances for each pair
        # Sum them up and add to the maxpro accumulator
        maxpro += np.sum(1. / np.prod(dsq, axis=1))

    return maxpro

def delta_np(x, ns, nv, pt1, pt2, var, periodic=False):
    """
    Compute the delta change in (u)MaxPro criterion due to swapping two points on a specific coordinate.

    This function efficiently evaluates the change in the (u)MaxPro criterion when swapping
    the `var`-th coordinate between two design points (`pt1` and `pt2`). It avoids recalculating
    the entire criterion, making it useful for fast local search optimizations.

    Args:
        x (np.ndarray): A 2D array of shape (ns, nv) representing the design points.
        ns (int): The number of design points (samples).
        nv (int): The number of variables (dimensions).
        pt1 (int): Index of the first point selected for swapping.
        pt2 (int): Index of the second point selected for swapping.
        var (int): The coordinate (dimension index) along which the swap is performed.
        periodic (bool, optional): If True, computes using periodic distances (uMaxPro criterion).
            If False, computes the standard MaxPro criterion. Default is False.

    Returns:
        float: The difference (delta) in the (u)MaxPro criterion due to the swap.
               Positive values indicate an increase, negative values a decrease.

    Notes:
        - Only the contributions involving `pt1` and `pt2` are updated.
        - Periodic mode applies wrap-around distance calculations.
        - Useful for optimization without full recomputation of the criterion.
    """

    old = 0  # Accumulator for the old criterion contribution
    new = 0  # Accumulator for the new criterion contribution after swap

    # Create a copy of the selected coordinate column and perform the swap
    x_new_v = x[:, var].copy()
    x_new_v[pt1] = x[pt2, var]  # Swap pt1 and pt2 coordinates along dimension var
    x_new_v[pt2] = x[pt1, var]

    # --------------------
    # First cycle: evaluate contributions for pt1
    # --------------------

    # Compute differences between pt1 and all other points (ns * nv)
    deltas_old = x[:, :] - x[pt1, :]

    # Compute the new differences for var coordinate with swapped values (ns * 1)
    deltas_new_v = x_new_v - x[pt2, var]

    if periodic:
        # Apply periodic wrapping if necessary (absolute differences + wrap-around)
        deltas_old = np.abs(deltas_old)
        deltas_old = np.minimum(deltas_old, 1 - deltas_old)
        deltas_new_v = np.abs(deltas_new_v)
        deltas_new_v = np.minimum(deltas_new_v, 1 - deltas_new_v)

    # Avoid self-distance (pt1 to pt1) by replacing zeros with ones artificially
    deltas_old[pt1, :] = 1.0
    deltas_new_v[pt1] = 1.0

    # Squared distances
    dsq_old = deltas_old ** 2
    dsq_new = dsq_old.copy()

    # Replace the var-th coordinate squared distances with the swapped ones
    dsq_new[:, var] = deltas_new_v ** 2

    # Compute and accumulate the old and new contributions (subtract 1 to remove self-pair)
    old += np.sum(1 / np.prod(dsq_old, axis=1)) - 1
    new += np.sum(1 / np.prod(dsq_new, axis=1)) - 1

    # --------------------
    # Second cycle: evaluate contributions for pt2
    # --------------------

    # Compute differences between pt2 and all other points
    deltas_old = x[:, :] - x[pt2, :]

    # Compute the new differences for var coordinate with swapped values
    deltas_new_v = x_new_v - x[pt1, var]

    if periodic:
        # Apply periodic wrapping if necessary
        deltas_old = np.abs(deltas_old)
        deltas_old = np.minimum(deltas_old, 1 - deltas_old)
        deltas_new_v = np.abs(deltas_new_v)
        deltas_new_v = np.minimum(deltas_new_v, 1 - deltas_new_v)

    # Avoid self-distance (pt2 to pt2)
    deltas_old[pt2, :] = 1.0
    deltas_new_v[pt2] = 1.0

    # Squared distances
    dsq_old = deltas_old ** 2
    dsq_new = dsq_old.copy()

    # Replace the var-th coordinate squared distances with the swapped ones
    dsq_new[:, var] = deltas_new_v ** 2

    # Compute and accumulate the old and new contributions (subtract 1 to remove self-pair)
    old += np.sum(1 / np.prod(dsq_old, axis=1)) - 1
    new += np.sum(1 / np.prod(dsq_new, axis=1)) - 1

    # Return the difference in criterion after swap (positive if criterion increased)
    return new - old


def swap_np(x, pt1, pt2, var):
    """
    Swap the value of a single coordinate (variable) between two points in the design matrix.

    Args:
        x (np.ndarray): Design matrix of shape (ns, nv).
        pt1 (int): Index of the first point to swap.
        pt2 (int): Index of the second point to swap.
        var (int): The specific coordinate (variable/dimension) to swap.

    Example:
        # Swaps x[pt1, var] and x[pt2, var]
        swap_np(x, 0, 1, 2)
    """
    tmp = x[pt2, var]
    x[pt2, var] = x[pt1, var]
    x[pt1, var] = tmp



# -----------------------------
# -- WITH PRECOMPUTED ARRAYS --
# -----------------------------
# Optimized, but sometimes is slower, as it maintains additional arrays


def maxPro_np_arr(x, row_sum, inv_prod, periodic=False):
    """
    Compute the full (u)MaxPro criterion with precomputed structures.

    This function calculates the (u)MaxPro criterion for a given design `x` and stores the
    intermediate computations in `inv_prod` and `row_sum` arrays. It is optimized for use
    with incremental updates where partial recalculations are performed later.

    Args:
        x (np.ndarray): Design matrix of shape (ns, nv), where `ns` is the number of points
            and `nv` is the number of variables (dimensions).
        row_sum (np.ndarray): Preallocated array of shape (ns,). Will be populated with the
            sum of squared inverse products for each point.
        inv_prod (np.ndarray): Preallocated symmetric array of shape (ns, ns). Will be
            populated with the inverse of the products of deltas between points.
        periodic (bool, optional): If True, uses periodic distance calculation (uMaxPro criterion).
            If False, uses standard Euclidean-based MaxPro criterion. Default is False.

    Returns:
        float: The computed (u)MaxPro criterion value.

    Notes:
        - Only the lower triangle of `inv_prod` is directly computed; the upper triangle is filled
          by symmetry. This saves computation.
        - The returned criterion is `0.5 * sum(row_sum)` because the off-diagonal elements are
          counted twice in the sum.
    """

    ns, nv = x.shape

    # Clear previous contents of inv_prod and row_sum if necessary
    # (Assumes inv_prod and row_sum are correctly sized/preallocated)

    # -------------------------
    # Step 1: Compute pairwise inverse products for each unique pair (i, j)
    # -------------------------
    for i in range(ns):
        # Compute deltas between point i and all previous points (0 to i-1)
        deltas = np.abs(x[i, :] - x[0:i, :])

        if periodic:
            # Apply periodic boundary conditions if necessary
            deltas = np.minimum(deltas, 1 - deltas)

        # Compute inverse product of deltas for each pair (i, j), j < i
        # 1. Take product of deltas over all variables (axis=1)
        # 2. Take reciprocal to get inverse product
        inv_prod[i, 0:i] = 1.0 / np.prod(deltas, axis=1)

        # Symmetric assignment to the upper triangle
        inv_prod[0:i, i] = inv_prod[i, 0:i]

    # -------------------------
    # Step 2: Compute row sums of squared inverse products
    # -------------------------
    # Each row_sum[i] sums over inv_prod[:, i] squared values
    row_sum[:] = np.sum(inv_prod ** 2, axis=0)

    # -------------------------
    # Step 3: Compute the final criterion value
    # -------------------------
    # We multiply by 0.5 to account for the symmetric structure
    # (each pair is counted twice in row sums)
    return 0.5 * np.sum(row_sum)

def delta_np_arr(
    x,
    row_sum,
    inv_prod,
    pt1,
    pt2,
    var,
    row1,
    row2,
    new_sums,
    periodic=False
):
    """
    Compute the delta change in (u)MaxPro criterion for a swap using precomputed arrays.

    This optimized version of `delta_np` evaluates the change in the (u)MaxPro criterion
    when swapping two points (`pt1` and `pt2`) along a specific coordinate (`var`). It uses
    precomputed matrices and vectors for fast updates without recalculating pairwise distances.

    Args:
        x (np.ndarray): Design matrix of shape (ns, nv) representing the current points.
        row_sum (np.ndarray): Vector of length ns. Sum of squared inverse products for each row.
        inv_prod (np.ndarray): Matrix of shape (ns, ns). Each entry holds the inverse product of
            squared distances between points (excluding var coordinate swaps).
        pt1 (int): Index of the first point to swap.
        pt2 (int): Index of the second point to swap.
        var (int): Coordinate (dimension index) along which to swap the values.
        row1 (np.ndarray): Preallocated vector of length ns. Will store updated inverse products for pt1.
        row2 (np.ndarray): Preallocated vector of length ns. Will store updated inverse products for pt2.
        new_sums (np.ndarray): Preallocated vector of length ns. Will store updated row sums.
        periodic (bool, optional): If True, applies periodic boundary conditions (uMaxPro criterion).
            If False, standard MaxPro criterion is used. Default is False.

    Returns:
        float: The difference (delta) in the (u)MaxPro criterion after the swap.
               Positive values indicate an increase, negative values a decrease.

    Notes:
        - This function operates in-place on `row1`, `row2`, and `new_sums`.
        - The key optimization is working only on the impacted rows/columns of the inverse product matrix.
    """

    # --------------------
    # Step 1: Copy relevant rows from inverse product matrix
    # --------------------
    row1[:] = inv_prod[pt1]  # Copy pt1's current row of inverse products
    row2[:] = inv_prod[pt2]  # Copy pt2's current row of inverse products

    # --------------------
    # Step 2: Compute old deltas (differences) for var coordinate between pt1/pt2 and all other points
    # --------------------
    deltas1_old = np.abs(x[:, var] - x[pt1, var])
    deltas2_old = np.abs(x[:, var] - x[pt2, var])

    if periodic:
        # Apply periodic boundary adjustments (wrap-around distances)
        deltas1_old = np.minimum(deltas1_old, 1 - deltas1_old)
        deltas2_old = np.minimum(deltas2_old, 1 - deltas2_old)

    # Prevent zero distances (pt1-pt1 and pt2-pt2), avoid division by zero
    deltas1_old[pt1] = 1
    deltas2_old[pt2] = 1

    # --------------------
    # Step 3: Compute new deltas after swapping the coordinates
    # --------------------
    deltas1_new = deltas2_old.copy()  # pt1 now has pt2's value
    deltas2_new = deltas1_old.copy()  # pt2 now has pt1's value

    # Correct self-distance terms again after swap
    deltas1_new[pt2] = deltas2_new[pt2]
    deltas2_new[pt1] = deltas1_new[pt1]
    deltas1_new[pt1] = 1
    deltas2_new[pt2] = 1

    # --------------------
    # Step 4: Update inverse product rows for pt1 and pt2
    # --------------------
    # Element-wise scaling of inverse products by ratio of old/new deltas for pt1
    row1[:] *= deltas1_old / deltas1_new

    # Same for pt2
    row2[:] *= deltas2_old / deltas2_new

    # --------------------
    # Step 5: Compute new row sums
    # --------------------
    # Subtract the old squared row contributions from pt1 and pt2
    # Add the new squared row contributions after swap
    new_sums[:] = row_sum - (inv_prod[pt1]**2 + inv_prod[pt2]**2) + (row1**2 + row2**2)

    # Recompute self-contributions (diagonal terms)
    new_sums[pt1] = np.sum(row1**2)
    new_sums[pt2] = np.sum(row2**2)

    # --------------------
    # Step 6: Return the change in the objective function (delta)
    # --------------------
    return (new_sums[pt1] + new_sums[pt2]) - (row_sum[pt1] + row_sum[pt2])



def swap_np_arr(x, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums):
    """
    Swap two points in a design matrix `x` along a specific coordinate (variable), and
    update the associated arrays (`inv_prod` and `row_sum`) to reflect this change.

    Args:
        x (np.ndarray): Design matrix of shape (ns, nv). This function swaps two entries in `x`.
        row_sum (np.ndarray): Vector of shape (ns,) holding the sum of squared inverse products
            for each point. Will be updated after the swap.
        inv_prod (np.ndarray): Symmetric matrix of shape (ns, ns) holding inverse products of deltas
            between all points. Rows/columns corresponding to swapped points will be updated.
        pt1 (int): Index of the first point to swap.
        pt2 (int): Index of the second point to swap.
        var (int): The specific coordinate (variable) to swap between pt1 and pt2.
        row1 (np.ndarray): Precomputed vector of shape (ns,) with updated inverse products for pt1 after swap.
        row2 (np.ndarray): Precomputed vector of shape (ns,) with updated inverse products for pt2 after swap.
        new_sums (np.ndarray): Precomputed vector of shape (ns,) with updated row sums after swap.

    Notes:
        - This function assumes `row1`, `row2`, and `new_sums` have already been computed
          (e.g., by `delta_np_arr`).
        - `inv_prod` is symmetric, so both the row and column for pt1 and pt2 are updated.
        - `swap_np` is an auxiliary function that performs the swap on `x`. Make sure it's defined.
    """

    # ---- Step 1: Perform the swap in x ----
    # Swap the `var`-th coordinate of pt1 and pt2
    swap_np(x, pt1, pt2, var)

    # ---- Step 2: Update inv_prod matrix ----
    # ns*ns matrix of inverse dist products
    # Replace row and column entries for pt1 with precomputed row1
    inv_prod[pt1, :] = row1
    inv_prod[:, pt1] = row1

    # Replace row and column entries for pt2 with precomputed row2
    inv_prod[pt2, :] = row2
    inv_prod[:, pt2] = row2

    # ---- Step 3: Update row_sum ----
    # Update row sums with precomputed new_sums
    row_sum[:] = new_sums

