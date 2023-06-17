
"""
Modified based on the DEQ repo.

Note that the convergence error isn't based on ||x_prev - x_curr|| but on || f(x_curr) - x_curr ||.
These 2 are only equivalent for the fpi solver

"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random
import pickle
import sys
import os
from scipy.optimize import root
import time
import torch

from utils.normalize import *

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """
    see https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py
    Minimize over alpha, the function phi(alpha). Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57. alpha > 0 is assumed to be a descent direction.

    phi = callable function phi(alpha)
    phi0 = value of phi(alpha) for original estimate
    derphi = callable function phi'(alpha).

    In our case phi(alpha) = torch.norm(g(x0 + alpha * update))**2 ?

    """
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite

def line_search(update, x0, g0, g, on=True):
    """
    Instead of solving for the best step size to use exactly, we use a fast line search algorithm
    to find an okay step size, so that compute can be spent on computing the update itself rather
    than the step size.

    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    # s_norm = torch.norm(x0) / torch.norm(update) #for wolfe search only

    def phi(s, store=True):
        """takes in step size alpha being tried, produces the next x_est with it,
        and returns what we want to minimize, i.e. norm of g(x_est)"""
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)

    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))

def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)

def broyden(f, x0, max_iters, eps=1e-3, stop_mode="rel", ls=False, verbose=False, save_trajectory=False):
    # print(f'broyden input size: {x0.size()}')
    bsz, total_hsize, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    trajectory = []

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, max_iters).to(dev)  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, max_iters, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)  # Formally should be -torch.matmul(inv_jacobian, (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len

    trace_dict = {'abs': [], 'rel': []}
    lowest_dict = {'abs': 1e8, 'rel': 1e8}
    lowest_step_dict = {'abs': 0, 'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < max_iters:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, on=ls)  # returns x_est, gx_new, x_est_new - x_est_prev, gx_new - gx_prev, ite
        nstep += 1
        tnstep += (ite + 1)

        abs_diffs = gx.norm(dim=1)
        rel_diffs = (abs_diffs / (1e-5+(gx + x_est).norm(dim=1)))
        abs_diff, rel_diff = abs_diffs.mean(), rel_diffs.mean() #rel diff correctly calculated is ~5% different from official implementation

        if verbose: print(f'abs diff {abs_diff:.2E} \t rel diff: {rel_diff:.2E} \t z scale: {torch.mean(x_est):.0E} +/- {torch.std(x_est):.0E}')
        diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        # print(f'broyden step {nstep} --- abs diff {abs_diff} --- rel diff {rel_diff}')
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode] or nstep==1:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        if save_trajectory:
            trajectory.append(x_est.view_as(x0).clone().detach())

        ## Added by Paul to measure stability of solver
        if nstep == 1:
            stability = 1
            prev_rel_diff = rel_diff
        else:
            if rel_diff > prev_rel_diff:  # error is jumping around
                stability = 0
            prev_rel_diff = rel_diff

        new_objective = diff_dict[stop_mode]
        if new_objective < eps:  # stop even if haven't reached max_iters steps
            if verbose: print(f'STOPPING BROYDEN SPECIAL CASE: met tolerance')
            break
        if new_objective < 3 * eps and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            if verbose: print('STOPPING BROYDEN SPECIAL CASE: no progress in last 30 steps')
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            if verbose: print('STOPPING BROYDEN SPECIAL CASE: protect thresh')
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:, None, None]
        vT[vT != vT] = 0 #replace nans with zeros
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)
        # print(update.device)

    # Fill everything up to the max_iters length
    for _ in range(max_iters + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    # print(f'{name} total broyden steps: {nstep} --- rel diff {rel_diff:02.5f}')

    out =  {"result": lowest_xest,
            "lowest_abs_diff": lowest_dict['abs'].item(),
            "lowest_rel_diff": lowest_dict['rel'].item(),
            "nstep_best": lowest_step_dict[stop_mode],  # which step was the best in hindsight
            "nstep": nstep,
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "eps": eps,
            "trajectory": trajectory,
            "stability": stability}

    return out

def anderson(f, x0, m=6, lam=1e-4, max_iters=50, eps=1e-3, stop_mode='rel', beta=1.0, verbose=False, save_trajectory=False):
    """ Anderson acceleration for fixed point iteration. """
    # print('stop mode ', stop_mode)
    bsz, d, L = x0.shape
    m = int(m)
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, int(d*L), dtype=x0.dtype, device=x0.device) #keep track of all previous estimates x_i s
    F = torch.zeros(bsz, m, int(d*L), dtype=x0.dtype, device=x0.device) #keep track of all previous f(x_i) s
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1) #first estimate x0 is given as input
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1) #second estimate in X is just f(x0) as in fpi because we don't have any previous estimates to lookback to

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {'abs': [], 'rel': []}
    lowest_dict = {'abs': 1e8, 'rel': 1e8}
    lowest_step_dict = {'abs': 0, 'rel': 0}
    trajectory = []

    # if verbose: print('Original tensors ')
    # if verbose: debug_print([X, F, H, y])

    for k in range(2, max_iters+2):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]

        # alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0] #beta=1.0 in normal anderson formulation. beta<1 is damped anderson acceleration, while beta>1 is overprojected
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m])#.view_as(x0)

        abs_diffs = gx.norm(dim=1)
        rel_diffs = (abs_diffs / (1e-5 + F[:,k%m]).norm(dim=1))
        abs_diff, rel_diff = abs_diffs.mean(), rel_diffs.mean() #rel diff correctly calculated is ~5% different from official implementation
        if verbose: print(f'abs diff {abs_diff:.2E} \t rel diff: {rel_diff:.2E} \t z scale: {torch.mean(X[:, k % m]):.0E} +/- {torch.std(X[:,k % m]):.0E}')

        diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        for mode in ['rel', 'abs']:
            # print(diff_dict[mode], lowest_dict[mode])
            if (diff_dict[mode] < lowest_dict[mode]) or k==2:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = X[:, k % m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if save_trajectory:
            trajectory.append(X[:,k%m].view_as(x0).clone().detach())
            # print('------ ', float(torch.sum(X[:,k%m].view_as(x0).clone().detach())))

        ##--------------- Added by Paul to measure stability of solver
        if k == 2:
            stability = 1
            abs_error_prev = abs_diff
        else:
            if abs_diff > abs_error_prev:  # error is jumping around
                stability = 0
            abs_error_prev = abs_diff
        ##---------------

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(max_iters + 1 - k): #paul changed -1 to +1
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {"result": lowest_xest, #not necessarily the last z of trajectory. It's the z with lowest error
           "lowest_abs_diff": lowest_dict['abs'].item(),
           "lowest_rel_diff": lowest_dict['rel'].item(),
           "nstep_best": lowest_step_dict[stop_mode],  # which step was the best in hindsight
           "nstep": k-1,
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "trajectory": trajectory,
           "stability": stability}

    return out

def fpi(f, x0, max_iters, eps=1e-3, stop_mode='rel', verbose=False, save_trajectory=False):
    """fast and cheap in memory but no guarantees to return stable FP, contrary to other solvers"""
    trajectory = []
    bsz = x0.shape[0]
    x_prev = x0
    iter_idx = 0

    while iter_idx < max_iters:
        x_new = f(x_prev)
        abs_diffs = (x_new - x_prev).view(bsz, -1).norm(dim=1)
        rel_diffs = abs_diffs / (1e-5 + x_new.view(bsz, -1).norm(dim=1))
        abs_diff, rel_diff = abs_diffs.mean(), rel_diffs.mean()
        if verbose: print(f'abs diff {abs_diff:.3E} \t rel diff: {rel_diff:.3E} \t z scale: {torch.mean(x_new):.0E} +/- {torch.std(x_new):.0E}')

        if save_trajectory:
            trajectory.append(x_new.clone().detach())

        ##--------------- Added by Paul to measure stability of solver
        if iter_idx == 0:
            stability = 1
            abs_error_prev = abs_diff
        else:
            if abs_diff > abs_error_prev:  # error is jumping around
                stability = 0
            abs_error_prev = abs_diff
        ##---------------

        iter_idx += 1

        if (stop_mode=='abs' and abs_diff < eps) or (stop_mode=='rel' and rel_diff < eps):
            break

        x_prev = x_new

    return x_new, iter_idx, abs_diff.item(), rel_diff.item(), stability, trajectory

def root_solver(f, x0, max_iters, solver_args, stochastic_max_iters=False, save_trajectory=False, name='forward'):
    """
    There are many solvers that all return different metrics and take different arguments.
    This is a wrapping function that evaluates each solver.
    solver_args must contain the solver specific arguments like:
    solver_args.anderson_m = 6 etc.

    returns: n_iters, final_rel_error
    """

    max_iters = random.randint(1, max_iters) if stochastic_max_iters else max_iters
    if solver_args.verbose_solver: print(f'----- SOLVER: {solver_args.solver} {name} mi={max_iters}')

    if solver_args.solver == 'broyden':
        results_dict = broyden(f=f,
                               x0=x0,
                               max_iters=max_iters,
                               eps=solver_args.abs_diff_target if solver_args.stop_mode=='abs' else solver_args.rel_diff_target,
                               stop_mode=solver_args.stop_mode,
                               ls=False,
                               verbose=solver_args.verbose_solver,
                               save_trajectory=save_trajectory)
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = results_dict['result'], results_dict['nstep'], results_dict['lowest_abs_diff'], results_dict['lowest_rel_diff'], results_dict['stability'], results_dict['trajectory']

    elif solver_args.solver == 'anderson':
        results_dict = anderson(f=f,
                                x0=x0,
                                m=solver_args.anderson_m,
                                lam=solver_args.anderson_lam,
                                max_iters=max_iters,
                                eps=solver_args.abs_diff_target if solver_args.stop_mode=='abs' else solver_args.rel_diff_target,
                                stop_mode=solver_args.stop_mode,
                                beta=solver_args.anderson_beta,
                                verbose=solver_args.verbose_solver,
                                save_trajectory=save_trajectory)
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = results_dict['result'], results_dict['nstep'], results_dict['lowest_abs_diff'], results_dict['lowest_rel_diff'], results_dict['stability'], results_dict['trajectory']

    elif solver_args.solver == 'fpi':
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = fpi(f,
                                                                                       x0=x0,
                                                                                       max_iters=max_iters,
                                                                                       eps=solver_args.abs_diff_target if solver_args.stop_mode == 'abs' else solver_args.rel_diff_target,
                                                                                       stop_mode=solver_args.stop_mode,
                                                                                       verbose=solver_args.verbose_solver,
                                                                                       save_trajectory=save_trajectory)

    else:
        raise NotImplementedError(f'solver {solver_args.solver} unknown')

    # print('stability ', stability)

    solver_logs = {'n_iters': n_iters, 'final_abs_diff': final_abs_diff, 'final_rel_diff': final_rel_diff, 'stability': stability, 'trajectory': trajectory, 'max_iters':max_iters}

    return solution, solver_logs


if __name__ == '__main__':
    pass


