#-a
# cython: boundscheck=False, wraparound=False, cdivision=True
#import numpy as np
#cimport numpy as cnp


# in notebook preceding cell write:     %load_ext cython
# in notebook cell write (at the top):  %%cython --compile-args=-fopenmp --link-args=-fopenmp

from libc.math cimport fabs, fmin, pow
from cython cimport boundscheck, wraparound

from cython.parallel import prange


#cpdef double maxPro_c(cnp.ndarray[cnp.float64_t, ndim=2] x, int ns, int nv, bint periodic=False):
cpdef double maxPro_c(double[:, :] x, int ns, int nv,
                      bint periodic=False):
    """
    Complete computation of the (u)MaxPro criterion for a given design matrix `x` without auxiliary arrays.
    This function evaluates the MaxPro (or uMaxPro if `periodic=True`) criterion by computing the inverse
    product of squared distances between all pairs of points in the design.

    Args:
        x (2D double array): Design matrix of shape (ns, nv), where ns = number of points and nv = number of variables.
        ns (int): Number of points in the design.
        nv (int): Number of variables (dimensions).
        periodic (bool, optional): If True, computes the uMaxPro criterion with periodic boundary conditions.
                                   If False, computes the standard MaxPro criterion.

    Returns:
        double: Value of the MaxPro (or uMaxPro) criterion.
    """
    cdef:
        int i, j, v
        double maxpro = 0.0
        double prod, dij_val

    for i in range(ns):
        for j in range(i):
            prod = 1.0
            for v in range(nv):
                dij_val = fabs(x[i, v] - x[j, v])
                if periodic:
                    dij_val = fmin(dij_val, 1 - dij_val)
                prod *= dij_val ** 2
            if prod > 0:
                maxpro += 1.0 / prod
    return maxpro

@boundscheck(False)
@wraparound(False)
cpdef double delta_c(double[:, :] x, int ns, int nv,
                     int pt1, int pt2, int var,
                     bint periodic=False):
    """
    Compute the change (delta) in the (u)MaxPro criterion associated with swapping the coordinate `var`
    between two points `pt1` and `pt2` in the design matrix `x`. This function performs an efficient
    local update without recalculating the entire criterion from scratch.

    Args:
        x (2D double array): Design matrix of shape (ns, nv), where ns = number of points and nv = number of variables.
        ns (int): Number of points in the design.
        nv (int): Number of variables (dimensions).
        pt1 (int): Index of the first point to swap.
        pt2 (int): Index of the second point to swap.
        var (int): The index of the coordinate (variable) being swapped.
        periodic (bool, optional): If True, uses periodic distances (uMaxPro); otherwise, standard MaxPro.

    Returns:
        double: The difference between the new and old contributions to the criterion (new - old),
                resulting from the proposed swap of `var` between `pt1` and `pt2`.
    """
    # ---- Local variable declarations ----
    cdef:
        int i, v
        double old = 0.0, new = 0.0
        double delta_old, delta_old_var, product_old
        double delta_new, delta_new_var, product_new

    # ---- First loop: update contributions for point pt1 ----
    for i in range(ns):

        if i != pt1:
            product_old = 1.0

            for v in range(nv):
                delta_old = fabs(x[i, v] - x[pt1, v])
                if periodic:
                    delta_old = fmin(delta_old, 1.0 - delta_old)

                product_old *= delta_old

            # For new, consider the swap only if j is the selected column
            product_new = product_old
            delta_old_var = fabs(x[i, var] - x[pt1, var]) #original point
            if i == pt2:
                delta_new_var = fabs(x[pt1, var] - x[pt2, var]) #swapped point
            else:
                delta_new_var = fabs(x[i, var] - x[pt2, var]) #swapped point
            if periodic:
                delta_old_var = fmin(delta_old_var, 1.0 - delta_old_var)
                delta_new_var = fmin(delta_new_var, 1.0 - delta_new_var)
            # Remove old contribution and introduce new distance for var
            product_new /= delta_old_var
            product_new *= delta_new_var

            # Add contributions to old and new criterion sums
            old += 1.0 / (product_old*product_old)
            new += 1.0 / (product_new*product_new)

    # ---- Second loop: update contributions for point pt2 ----
    for i in range(ns):

        if i != pt2:
            product_old = 1.0

            for v in range(nv):
                delta_old = fabs(x[i, v] - x[pt2, v])
                if periodic:
                    delta_old = fmin(delta_old, 1.0 - delta_old)

                product_old *= delta_old

            # For new, consider the swap only if j is the selected column
            product_new = product_old
            delta_old_var = fabs(x[i, var] - x[pt2, var]) #original point
            if i == pt1:
                delta_new_var = fabs(x[pt2, var] - x[pt1, var]) #swapped point
            else:
                delta_new_var = fabs(x[i, var] - x[pt1, var]) #swapped point
            if periodic:
                delta_old_var = fmin(delta_old_var, 1.0 - delta_old_var)
                delta_new_var = fmin(delta_new_var, 1.0 - delta_new_var)
            # Remove old contribution and introduce new distance for var
            product_new /= delta_old_var
            product_new *= delta_new_var

            # Add contributions to old and new criterion sums
            old += 1.0 / (product_old*product_old)
            new += 1.0 / (product_new*product_new)

    return new - old


cpdef void swap_c(double[:, :] x, int i, int j, int v):
    '''
    Swap the `v`-th coordinate (column) between two points `i` and `j` (rows) in the design matrix `x`.
    This operation is **in-place**, meaning it directly modifies the input array `x`.

    Args:
        x (2D double array): The design matrix of shape (ns, nv).
        i (int): Row index of the first point.
        j (int): Row index of the second point.
        v (int): Column index (coordinate) to swap between the two points.

    Returns:
        None: The function performs an in-place update of `x`.
    '''
    cdef double tmp

    tmp = x[j, v]
    x[j, v] = x[i, v]
    x[i, v] = tmp



cpdef double maxPro_c_arr(double[:, :] x, int ns, int nv, 
                          double[:]    row_sum,  # .. radkova suma
                          double[:, :] inv_prod, # .. ctvercova matice inverznich productu delt
                          bint periodic=False):
    '''
    Complete recalculation of the (u)MaxPro criterion, along with its auxiliary arrays.

    This function fully updates:
        - `inv_prod`: a symmetric (ns x ns) matrix holding the inverse of the product of distances between points.
        - `row_sum`: a vector (size ns) holding the sum of squared inverse products for each row.

    Args:
        x        (2D double array): Design matrix of shape (ns, nv), where `ns` is the number of points and `nv` the number of variables.
        ns       (int): Number of points in the design.
        nv       (int): Number of variables (dimensions).
        row_sum  (1D double array): Output array, size ns, to store row-wise sums of squared inverse products.
        inv_prod (2D double array): Output symmetric matrix (ns x ns), storing inverse products of distances.
        periodic (bool): If True, distances are computed in periodic space (uMaxPro criterion); otherwise, standard space (MaxPro criterion).

    Returns:
        double: The computed (u)MaxPro criterion value.
    '''
    cdef:
        int i, j
        double prod, maxpro=0

    for i in range(ns):
        for j in range(i):
            prod = 1.0
            for v in range(nv):
                delta = fabs(x[i,v] - x[j,v])
                if periodic:
                    delta = fmin(delta, 1.0-delta)
                prod *= delta

            prod = 1./prod
            inv_prod[i,j] = prod
            inv_prod[j,i] = prod

    for i in range(ns):
        row_sum[i] = 0
        for j in range(ns):
            row_sum[i] += inv_prod[i,j]**2
        maxpro += row_sum[i]

    return 0.5*maxpro


cpdef double delta_c_arr(double[:, :] x, int ns,
                         double[:]    row_sum,  # .. row sum
                         double[:, :] inv_prod, # .. square matrix of inverted products of deltas
                         int pt1, int pt2, int var, 
                         double[:] row1, # .. new row and column of inv_prod corresponding to pt1
                         double[:] row2, # .. new row and column of inv_prod corresponding to pt1
                         double[:] new_sums,
                         bint periodic=False):
    '''
    Complete recalculation of the (u)MaxPro criterion and the auxiliary arrays (row_sum, inv_prod),
    and precomputation of the new (row1, row2, new_sums) arrays for use in the 'swap' function.

    This function calculates the delta in the (u)MaxPro criterion when swapping two points in the design,
    without recomputing the entire criterion from scratch.

    Args:
        x        (2D double array): Design matrix of shape (ns, nv), where `ns` is the number of points and `nv` the number of variables.
        ns       (int): Number of points.
        row_sum  (1D double array): Current row sums (squared inverse products of deltas).
        inv_prod (2D double array): Current matrix of inverse products of distances.
        pt1, pt2 (int): Points to swap in the design matrix `x`.
        var      (int): The selected coordinate (variable) to swap.
        row1     (1D double array): New row for pt1 in `inv_prod` after the swap.
        row2     (1D double array): New row for pt2 in `inv_prod` after the swap.
        new_sums (1D double array): New row sums for each point after the swap.
        periodic (bool): If True, use periodic boundary conditions (uMaxPro criterion). Otherwise, use standard distance (MaxPro criterion).

    Returns:
        double: The change in the (u)MaxPro criterion caused by swapping `pt1` and `pt2`.
    '''
    cdef:
        int i, j
        double delta1_old, delta2_old
        double delta1_new, delta2_new
        double prod_old, prod_new
        double row1_sum = 0
        double row2_sum = 0

    for i in range(ns):  #prange (uses OMP) https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html
    #for i in prange(n, nogil=True, use_threads_if=n>1000):
        prod_old = 1.0
        prod_new = 1.0
        delta1_old = fabs(x[i,var] - x[pt1,var])
        delta2_old = fabs(x[i,var] - x[pt2,var])
        if periodic:
            delta1_old = fmin(delta1_old, 1.0 - delta1_old)
            delta2_old = fmin(delta2_old, 1.0 - delta2_old)

        delta1_new = delta2_old
        delta2_new = delta1_old

        if i == pt1:
            delta2_new = delta1_new
            delta1_new = 1.0
        if i == pt2:
            delta1_new = delta2_new
            delta2_new = 1.0

        row1[i] = inv_prod[pt1,i] * delta1_old / delta1_new #introduce changes for pt1
        row2[i] = inv_prod[pt2,i] * delta2_old / delta2_new #introduce changes for pt2

        row1_sum += row1[i]**2
        row2_sum += row2[i]**2

        new_sums[i] = row_sum[i] - (inv_prod[i,pt1]**2 + inv_prod[i,pt2]**2) + (row1[i]**2 + row2[i]**2) # move to swap

    new_sums[pt1] = row1_sum # move to swap
    new_sums[pt2] = row2_sum # move to swap

    # ........ new ...........   -  ... old ...................
    return (row1_sum + row2_sum) - (row_sum[pt1] + row_sum[pt2])


cpdef void swap_c_arr(double[:, :] x, int ns,
                        double[:] row_sum,      # Row sum of squared inverse products of deltas
                        double[:, :] inv_prod,  # Square matrix of inverted products of deltas
                        int pt1, int pt2, int var,
                        double[:] row1,         # New row for pt1 in inv_prod (after swap)
                        double[:] row2,         # New row for pt2 in inv_prod (after swap)
                        double[:] new_sums):    # New row sums for each point after swap
    '''
    Perform the physical swap of two points in the design matrix `x` and update the auxiliary arrays
    (row_sum, inv_prod) using the precomputed entries (row1, row2, new_sums).

    Args:
        x        (2D double array): Design matrix of shape (ns, nv), where `ns` is the number of points and `nv` is the number of variables.
        ns       (int): Number of points.
        row_sum  (1D double array): Current row sums (squared inverse products of deltas).
        inv_prod (2D double array): Current matrix of inverse products of distances.
        pt1, pt2 (int): Points to swap in the design matrix `x`.
        var      (int): The selected coordinate (variable) to swap.
        row1     (1D double array): New row for pt1 in `inv_prod` after the swap.
        row2     (1D double array): New row for pt2 in `inv_prod` after the swap.
        new_sums (1D double array): New row sums for each point after the swap.

    Returns:
        void: This function modifies `x`, `inv_prod`, `row_sum`, and `new_sums` in-place.
    '''
    cdef:
        int i, j
        double tmp
        double row1_sum = 0
        double row2_sum = 0

    # perform the swap
    tmp = x[pt2,var]
    x[pt2,var] = x[pt1,var]
    x[pt1,var] = tmp

    for i in range(ns):
        inv_prod[pt1, i] = row1[i]# update the ns*ns matrix of inverse dist products
        inv_prod[i, pt1] = row1[i]
        inv_prod[pt2, i] = row2[i]
        inv_prod[i, pt2] = row2[i]



        #new_sums[i] = row_sum[i] - (inv_prod[i,pt1]**2 + inv_prod[i,pt2]**2) + (row1[i]**2 + row2[i]**2) # move to swap
        row_sum[i] = new_sums[i] # update the row sums of the above matrix




##### Parallel (OMP versions) of the delta function (without and with arrays)


@boundscheck(False)
@wraparound(False)
cpdef double delta_c_p(double[:, :] x,
                       double[:] old1, # Row vector for pt1 (old), working array with length ns
                       double[:] old2, # Row vector for pt2 (old), working array with length ns
                       double[:] new1, # Row vector for pt1 (new), working array with length ns
                       double[:] new2, # Row vector for pt2 (new), working array with length ns
                       int ns, int nv,  # ns = number of points, nv = number of variables
                       int pt1, int pt2, int var,  # pt1 and pt2 are the points to swap, var is the coordinate to swap
                       int num_threads,  # Number of threads for parallelization
                       bint periodic=False):  # If periodic boundary conditions should be considered
    '''
    Parallelized version of delta computation (delta), using four arrays (old1, old2, new1, new2).
    This function calculates the difference in the criterion before and after swapping two points,
    and uses parallelization for faster computation over large datasets.
    '''

    cdef:
        int i, v  # Loop variables
        double old = 0.0, new = 0.0  # Variables to accumulate old and new values
        double delta_old1, delta_new1  # Distance values for pt1 before and after the swap
        double delta_old2, delta_new2  # Distance values for pt2 before and after the swap
        double delta_old_var1, delta_old_var2  # Distance for the selected variable (var) for pt1 and pt2 before the swap
        double delta_new_var1, delta_new_var2  # Distance for the selected variable (var) for pt1 and pt2 after the swap

    for i in prange(ns, nogil=True, num_threads = num_threads):

        if i != pt1:

            # Compute old1
            old1[i] = 1.0
            for v in range(nv):
                delta_old1 = fabs(x[i, v] - x[pt1, v])
                if periodic:
                    delta_old1 = fmin(delta_old1, 1.0 - delta_old1)
                old1[i] *= delta_old1

            # For new1, consider the swap only for the the selected column 'var'
            delta_old_var1 = fabs(x[i, var] - x[pt1, var]) #original point
            if i == pt2:
                delta_new_var1 = fabs(x[pt1, var] - x[pt2, var]) #swapped point
            else:
                delta_new_var1 = fabs(x[i, var] - x[pt2, var]) #swapped point
            if periodic:
                delta_old_var1 = fmin(delta_old_var1, 1.0 - delta_old_var1)
                delta_new_var1 = fmin(delta_new_var1, 1.0 - delta_new_var1)
            new1[i] = old1[i] / delta_old_var1 * delta_new_var1

            old1[i] = 1.0 / (old1[i]*old1[i])
            new1[i] = 1.0 / (new1[i]*new1[i])
        else:
            old1[i] = 0.
            new1[i] = 0.

        if i != pt2:

            # Compute old1
            old2[i] = 1.0
            for v in range(nv):
                delta_old2 = fabs(x[i, v] - x[pt2, v])
                if periodic:
                    delta_old2 = fmin(delta_old2, 1.0 - delta_old2)
                old2[i] *= delta_old2

            # For new2, consider the swap only for the the selected column 'var'
            delta_old_var2 = fabs(x[i, var] - x[pt2, var]) #original point
            if i == pt1:
                delta_new_var2 = fabs(x[pt2, var] - x[pt1, var]) #swapped point
            else:
                delta_new_var2 = fabs(x[i, var] - x[pt1, var]) #swapped point
            if periodic:
                delta_old_var2 = fmin(delta_old_var2, 1.0 - delta_old_var2)
                delta_new_var2 = fmin(delta_new_var2, 1.0 - delta_new_var2)
            new2[i] = old2[i] / delta_old_var2 * delta_new_var2

            old2[i] = 1.0 / (old2[i]*old2[i])
            new2[i] = 1.0 / (new2[i]*new2[i])
        else:
            old2[i] = 0.
            new2[i] = 0.

    for i in prange(ns, nogil=True, num_threads = num_threads):
        old += old1[i] + old2[i]
        new += new1[i] + new2[i]

    return new - old




@boundscheck(False)
@wraparound(False)
cpdef double delta_c_arr_p(double[:, :] x, int ns,
                            double[:] row_sum,  # .. Row sum (previous sums of distances)
                            double[:, :] inv_prod,  # .. Square matrix of inverted products of deltas
                            int pt1, int pt2, int var,  # pt1, pt2: the two points to swap; var: the coordinate/variable
                            double[:] row1,  # .. New row and column for pt1 after swap
                            double[:] row2,  # .. New row and column for pt2 after swap
                            double[:] new_sums,  # .. New row sums after swap (precomputed)
                            int num_threads,  # Number of threads for parallel execution
                            bint periodic=False):  # If periodic boundary conditions are applied
    '''
    Calculates the change (delta) using precomputed arrays (row_sum, inv_prod).
    It returns the updated (row1, row2, new_sums) arrays that will be used in 'swap'.
    '''
    cdef:
        int i, j  # Loop variables
        double delta1_old, delta2_old  # Distance differences for pt1 and pt2 before the swap
        double delta1_new, delta2_new  # Distance differences for pt1 and pt2 after the swap
        double prod_old, prod_new  # Temporary variables to hold old and new product values
        double row1_sum = 0  # Sum of the new row1 values (for pt1)
        double row2_sum = 0  # Sum of the new row2 values (for pt2)

    for i in prange(ns, nogil=True, num_threads = num_threads): #use_threads_if=n>1000:

        prod_old = 1.0
        prod_new = 1.0
        delta1_old = fabs(x[i,var] - x[pt1,var])
        delta2_old = fabs(x[i,var] - x[pt2,var])
        if periodic:
            delta1_old = fmin(delta1_old, 1.0 - delta1_old)
            delta2_old = fmin(delta2_old, 1.0 - delta2_old)

        delta1_new = delta2_old
        delta2_new = delta1_old

        if i == pt1:
            delta2_new = delta1_new
            delta1_new = 1.0
        if i == pt2:
            delta1_new = delta2_new
            delta2_new = 1.0

        row1[i] = inv_prod[pt1,i] * delta1_old / delta1_new #introduce changes for pt1
        row2[i] = inv_prod[pt2,i] * delta2_old / delta2_new #introduce changes for pt2

        row1_sum += row1[i]**2
        row2_sum += row2[i]**2

        new_sums[i] = row_sum[i] - (inv_prod[i,pt1]**2 + inv_prod[i,pt2]**2) + (row1[i]**2 + row2[i]**2) # move to swap

    new_sums[pt1] = row1_sum # move to swap
    new_sums[pt2] = row2_sum # move to swap

    # ........ new ...........   -  ... old ...................
    return (row1_sum + row2_sum) - (row_sum[pt1] + row_sum[pt2])
