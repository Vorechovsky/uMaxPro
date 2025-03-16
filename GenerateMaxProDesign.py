import sys
import os
import time
import math
import random
from random import random
import numpy as np
import multiprocessing
from multiprocessing import Value

import subprocess
import glob

from tqdm import tqdm  # Optional, if you want pretty progress bars


from MaxproTools_python import (maxPro_np,     delta_np,     swap_np,
                                maxPro_np_arr, delta_np_arr, swap_np_arr)
# def maxPro_np(x,periodic=False)
# def delta_np(x, ns, nv, pt1, pt2, var, periodic=False)
# def delta_arrays_np(x, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, periodic=False)
# def maxPro_arrays(x, row_sum, inv_prod, periodic=False)
# def swap_arrays_np(x, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums)
# def swap_np(x, pt1, pt2, var)





#import cythonfn
from MaxproTools_cython import (maxPro_c,     delta_c,      swap_c,
                                              delta_c_p,
                                maxPro_c_arr, delta_c_arr,  swap_c_arr,
                                              delta_c_arr_p
                                )
# the following functions are available from MaxproTools_cython.pyx:
#
# single-threaded versions
#  maxPro_c (x, ns, nv, periodic)                        complete computation of maxpro without auxiliary arrays
#  delta_c  (x, ns, nv, pt1, pt2, var, periodic)
#  maxPro_c_arr(x, ns, nv, row_sum, inv_prod, periodic)
#  delta_c_arr (x, ns,     row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, periodic)
#  swap_c(x, i, j,  v)
#  swap_c_arr(x, ns, row_sum,  inv_prod, pt1, pt2, var, row1, row2, new_sums)

# parallel versions
#  delta_c_arr_p(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic)
#  delta_c_p(x, old1, old2, new1, new2, ns, nv, pt1, pt2,  var, num_threads, periodic)
#
# for parallel versions, one can set the following:
#import os
#os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
#os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6







# select a triple of functions depending on
#    ns (parallel or serial)
#    nv (with/out aux arrays)
'''



if (nv<4): # no auxiliary arrays
     def maxPro(x, ns, nv, row_sum, inv_prod, periodic):
        return  maxPro_c(x, ns, nv, periodic)
     def swap(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums):
        return swap_c(x, pt1, pt2, var)
    if(ns<500):  # not parallel
        def deltafn(x,ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c(x, ns, nv, pt1, pt2, var, periodic)
    else:  # parallel with optimal 'numthreads'
        def deltafn(x,ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c_p(x, old1, old2, new1, new2, ns, nv, pt1, pt2,  var, num_threads, periodic)
else:   # with auxiliary arrays
    def maxPro(x, ns, nv, row_sum, inv_prod, periodic):
        return  return maxPro_c_arr(x, ns, nv, row_sum, inv_prod, periodic)
    def swap(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums):
        return swap_c_arr(x, ns, row_sum,  inv_prod, pt1, pt2, var, row1, row2, new_sums)
    if(ns<500): # not parallel
         def deltafn(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c_arr (x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, periodic)
    else: # parallel with optimal 'numthreads'
        def deltafn             (x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c_arr_p(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic)

'''


##################################################################
# generate randomly permuted LHS table of sampling probabilities #
##################################################################
def get_LHS_design(ns,nv, seed=None):
    ranks = np.empty((ns, nv))
    arr = np.arange(ns)
    if seed != None:
        np.random.seed(seed)
    for v in range (nv): #each column separately
        ranks[:,v] = np.random.permutation(arr)

    return (ranks + 0.5)/ns


def generate_design(nv, ns, periodic, ntrials, nrecalc, T_ini, T_fact, tsk_name, dir_name, des, seed):

    num_threads = 5
    if (ns>3000):
        num_threads = 10

    if (nv<4): # no auxiliary arrays
        def maxPro(x, ns, nv, row_sum, inv_prod, periodic):
            return  maxPro_c(x, ns, nv, periodic)
        def swap(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums):
            return swap_c(x, pt1, pt2, var)
        def deltafn(x,ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c(x, ns, nv, pt1, pt2, var, periodic)
    else:  # with aux arrays (row_sum, inv_prod)
        def maxPro(x, ns, nv, row_sum, inv_prod, periodic):
            return  maxPro_c_arr(x, ns, nv, row_sum, inv_prod, periodic)
        def swap(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums):
            return swap_c_arr(x, ns, row_sum,  inv_prod, pt1, pt2, var, row1, row2, new_sums)
        def deltafn(x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic):
            return delta_c_arr (x, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, periodic)

    x_ini = get_LHS_design(ns, nv, seed)
    x_opt = x_ini.copy() # final (optimized) design
    x_wrk = x_ini.copy() # working version

    # working arrays for array versions
    row_sum  = np.zeros(ns)       #arrays with numpy
    inv_prod = np.zeros((ns,ns))
    row1     = np.zeros(ns)
    row2     = np.zeros(ns)
    new_sums = np.zeros(ns)

    old1 = np.zeros(ns) #row vector for pt1 (old)
    old2 = np.zeros(ns) #row vector for pt2 (old)
    new1 = np.zeros(ns) #row vector for pt1 (new)
    new2 = np.zeros(ns) #row vector for pt2 (new)

    T_hist = np.zeros(ntrials+nrecalc)
    maxpro_hist = np.zeros(ntrials+nrecalc)
    maxpro_opt_hist = np.zeros(ntrials+nrecalc)

    T = T_ini


    maxpro = maxPro(x_wrk, ns, nv, row_sum, inv_prod, periodic) #generic version
    maxpro_ini = maxpro
    maxpro_opt = maxpro

    # Simulated Annealing loops
    for trial in range (ntrials):

        T *= T_fact
        pt1, pt2 = np.random.choice(ns, size=2, replace=False) #select two different points for swap
        var = np.random.randint(nv, size=1)[0] #select a dimension for swap
        delta = deltafn(x_wrk,ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic)

        crit = np.exp(-delta/T)

        # accept?
        # negative deltas accepted always, positive deltas depending on uniform random variable
        if crit > np.random.rand():
            swap(x_wrk, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums)
            maxpro += delta

        if  maxpro<maxpro_opt: # archive the best
            maxpro_opt = maxpro
            np.copyto(x_opt, x_wrk)


        T_hist[trial] = T
        maxpro_hist[trial] = maxpro
        maxpro_opt_hist[trial] = maxpro_opt

        if trial%nrecalc == 0: #prevent accumulation of rounding errors (recalculate from scratch)
            maxpro = maxPro(x_wrk, ns, nv, row_sum, inv_prod, periodic) #generic version
            #print(maxpro_ini, maxpro, maxpro_opt, end='\r')

        #progress.update(task_trials, advance=1) #update the progress bar with MetropHast trials


    # Hungry optimization loops (work directly on x_opt)
    maxpro_opt = maxPro(x_opt, ns, nv, row_sum, inv_prod, periodic) #generic version
    for trial in range (ntrials,ntrials+nrecalc):

        var_i,var_j = np.random.choice(ns, size=2, replace=False) #select two different points for swap
        v = np.random.randint(nv, size=1) #select a dimension for swap
        delta = deltafn(x_opt,ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums, num_threads, periodic)

        crit = np.exp(-delta/T)

        if delta < 0: #accept negative deltas only
            swap(x_opt, ns, row_sum, inv_prod, pt1, pt2, var, row1, row2, new_sums)

            maxpro_opt += delta
            #maxpro_opt = maxPro_c(x_opt,ns,nv,periodic)
            #print(maxpro_opt, end='\r')

        T_hist[trial] = T
        maxpro_opt_hist[trial] = maxpro_opt
        maxpro_hist[trial] = maxpro_opt
        #progress.update(task_recalc, advance=1) #update the progress bar with hungry algorithm trials


    #maxpros[des] = maxpro_opt
    #print(maxpro_opt)
    #

    #progress.update(task_dsgns, advance=1)
    #progress.reset(task_trials)
    #progress.reset(task_recalc)

    #save file with the final design
    des_name = tsk_name + "_des=" + str(des).zfill(6)
    des_fname = dir_name + '/' + des_name + '.npy'
    np.save(des_fname,x_opt)

    return x_opt, x_ini, maxpro_opt, T_hist, maxpro_hist, maxpro_opt_hist


def get_tskname_dirname(nv, ns, periodic, n_des):
    '''construct the task name and the directory path
    '''
    tsk_name = "nv=" + str(nv).zfill(4) + "_ns=" + str(ns).zfill(5) + "_per=" + str(periodic) + "_ndes=" + str(n_des).zfill(6)
    dir_name = os.path.join('data', tsk_name)
    return tsk_name, dir_name




if __name__ == '__main__':
    #cdir = os.getcwd()
    import multiprocessing
    import time
    CPU_NUM = multiprocessing.cpu_count()-1

    #python GenerateMaxProDesign.py 2 11 True 100 1000 12354566.123 0.6555 13
    print(sys.argv[1:])
    nv, ns, periodic, ntrials, nrecalc, T_ini, T_fact, n_des = sys.argv[1:]

    nv = int(nv)
    ns = int(ns)
    if periodic == 'False':
        periodic = False
    else:
        periodic = True
    n_des = int(n_des)
    ntrials = int(ntrials)
    nrecalc = int(nrecalc)
    T_ini = float(T_ini)
    T_fact = float(T_fact)

    tsk_name, dir_name = get_tskname_dirname(nv, ns, periodic, n_des)

    dir_Exist = os.path.exists(dir_name)
    if not dir_Exist:
        os.mkdir(dir_name)
        print("New directory created:", dir_name)


    print('Task settings:', nv, ns, ntrials, nrecalc, T_ini, T_fact, tsk_name, dir_name)
    #input('Press enter to continue')

    # test generate_design
    #print(generate_design(nv, ns, periodic, ntrials, nrecalc, T_ini, T_fact, tsk_name, dir_name, 1))
    #print(generate_design(nv, ns, periodic, ntrials, nrecalc, T_ini, T_fact, tsk_name, dir_name, 1))
    #exit()

    seeds = np.random.choice(1000000, size=n_des, replace=False) #select different seeds

    try:
        pool = multiprocessing.Pool(processes=CPU_NUM)
        for des in range(n_des):
            results = pool.apply_async(generate_design, args=[nv, ns, periodic, ntrials, nrecalc, T_ini, T_fact, tsk_name, dir_name, des, seeds[des]], kwds={})
        print('pool apply complete')
    except (KeyboardInterrupt, SystemExit):
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')
    finally:
        print('joining pool processes')
        pool.close()
        pool.join()
        print('join complete')
    print('the end' )
    #print(results.get())


