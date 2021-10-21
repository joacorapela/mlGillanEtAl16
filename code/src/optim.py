
import warnings
import time
import itertools
import multiprocessing
import numpy as np
import scipy.optimize
import torch

import utils

def estimate_map_params(logPosteriorModel, likelihood_params0,
                        likelihood_params_bounds, prior_params,
                        max_iter, MAP_LBFGSB_ftol):
    def nLogPosterior(likelihood_params):
        logPosteriorModel.likelihood_params = likelihood_params
        value = -logPosteriorModel.logPosterior(prior_params=prior_params)
        return value
    eval_func_wrapped = utils.wrap_torch_to_numpy_func(torch_func=
                                                       nLogPosterior)
    optim_params = {'maxiter': max_iter, 'ftol': MAP_LBFGSB_ftol}
    min_res = minimize_scipy_LBFGSB(func=eval_func_wrapped, 
                                    x0=likelihood_params0,
                                    bounds=likelihood_params_bounds,
                                    optim_params=optim_params)
    max_x = torch.from_numpy(min_res["minimum_x"])
    return max_x

def estimate_multiple_models_map_params(log_posterior_models,
                                        likelihood_params0,
                                        likelihood_params_bounds,
                                        prior_params, max_iter, LBFGSB_ftol):
    pool_argument = zip(log_posterior_models,
                        itertools.repeat(likelihood_params0),
                        itertools.repeat(likelihood_params_bounds),
                        itertools.repeat(prior_params),
                        itertools.repeat(max_iter),
                        itertools.repeat(LBFGSB_ftol))
    with multiprocessing.Pool() as pool:
        pool_res = pool.starmap(estimate_map_params, pool_argument)
    N = len(pool_res)
    models_params = np.empty((N, len(pool_res[0])))
    for i in range(N):
        models_params[i,:] = pool_res[i]
    return models_params

def approximatePosteriorWithIIDNormal(logPosteriorModel,
                                      likelihood_params0,
                                      likelihood_params_bounds,
                                      prior_params,
                                      max_iter, MAP_LBFGSB_ftol):
    max_x = estimate_map_params(logPosteriorModel=logPosteriorModel,
                                likelihood_params0=likelihood_params0,
                                likelihood_params_bounds=
                                 likelihood_params_bounds,
                                prior_params=prior_params, max_iter=max_iter,
                                MAP_LBFGSB_ftol=MAP_LBFGSB_ftol)
    # compute 2nd order derivatives of posterior
    max_x.requires_grad = True
    logPosteriorModel.likelihood_params = max_x
    maximum = logPosteriorModel.logPosterior(prior_params=prior_params)
    grad = torch.autograd.grad(maximum, max_x , create_graph=True, retain_graph=True)[0]
    ones = torch.ones_like(max_x)
    grad2 = torch.autograd.grad(grad, max_x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    max_x.requires_grad = False
    #
    mean = max_x
    var = -1.0/grad2
    negative_var_indices = torch.where(var<0)[0]
    if len(negative_var_indices)>0:
        var_as_string = np.array2string(var.detach().numpy())
        warnings.warn("Negative var elements in " + var_as_string)
        var[negative_var_indices] = 0.0
    mean = mean.detach()
    var = var.detach()
    return mean, var

def estimate_hbi_prior_params(log_posterior_models,
                              prior_params0,  
                              likelihood_params0,
                              likelihood_params_bounds,
                              max_em_iter=100, 
                              max_approxPosterior_iter=50,
                              MAP_LBFGSB_ftol=2.220446049250313e-09,
                              EM_tol=1e-04):
    def eStep(likelihood_params0, log_posterior_models, prior_params, max_iter):
        N = len(log_posterior_models)
        m = torch.empty((len(prior_params[0]), N), dtype=torch.double)
        Sigma = torch.empty((len(prior_params[0]), N), dtype=torch.double)
        logPosteriorModel_args = [logPosteriorModel["model"] for
                                  logPosteriorModel in log_posterior_models]
        pool_argument = zip(logPosteriorModel_args,
                            itertools.repeat(likelihood_params0),
                            itertools.repeat(likelihood_params_bounds),
                            itertools.repeat(prior_params),
                            itertools.repeat(max_iter),
                            itertools.repeat(MAP_LBFGSB_ftol))
        with multiprocessing.Pool() as pool:
            poolRes = pool.starmap(approximatePosteriorWithIIDNormal, pool_argument)
        for i in range(N):
            m[:,i] = poolRes[i][0]
            Sigma[:,i] = poolRes[i][1]
        return m, Sigma

    def mStep(m, Sigma):
        mu = torch.mean(m, dim=1)
        nu2 = 2*torch.mean(m**2+Sigma, dim=1)-mu**2
        return mu, nu2

    prior_params = prior_params0
    old_prior_params = []
    for i in range(len(prior_params)):
        old_prior_params.append(torch.tensor([float("inf")]*len(prior_params[i])))
    exit = False
    iterations_runtimes = []
    i = 0
    while not exit and i<max_em_iter:
        print("EM iteration {:d}".format(i))
        for prior_param in prior_params:
            print(prior_param)
        start_time = time.time()
        m, Sigma = eStep(likelihood_params0=likelihood_params0,
                         log_posterior_models=log_posterior_models,
                         prior_params=prior_params,
                         max_iter=max_approxPosterior_iter)
        old_prior_params = prior_params
        prior_params = mStep(m=m, Sigma=Sigma)
        exit = True
        index = 0
        while exit and index<len(prior_params):
            exit = torch.mean((prior_params[index]-old_prior_params[index])**2)<EM_tol
            index += 1
        iterations_runtimes.append(time.time()-start_time)
        i += 1
    return prior_params, iterations_runtimes

def minimize_scipy_LBFGSB(func, x0, bounds, optim_params):
    optim_res = scipy.optimize.minimize(fun=func, x0=x0, method="L-BFGS-B",
                                        jac=True, bounds=bounds,
                                        options=optim_params)
    minimum = optim_res.fun
    minimum_x = optim_res.x
    nfeval = optim_res.nfev
    niter = optim_res.nit
    return {"minimum": minimum, "minimum_x": minimum_x,
            "nfeval": nfeval, "niter": niter}


def minimize_torch_LBFGS(x, eval_func, max_iter=1000,
                         params_change_tol=1e-6, LBFGS_max_iter=1,
                         LBFGS_max_eval=1,
                         LBFGS_line_search_fn="strong_wolfe"):
    for i in range(len(x)):
        x[i].requires_grad = True
    optimizer = torch.optim.LBFGS(x, max_iter=LBFGS_max_iter,
                                  max_eval=LBFGS_max_eval,
                                  line_search_fn=LBFGS_line_search_fn)

    def closure():
        optimizer.zero_grad()
        curEval = eval_func()
        curEval.backward()
        return curEval

    params_change = torch.tensor(float("inf"))
    i = 0
    while i < max_iter and params_change > params_change_tol:
        old_x = x[0].clone()
        optimizer.step(closure)
        print("params_change:", params_change.item())
        params_change = torch.mean((old_x-x[0])**2)
        i += 1

    for i in range(len(x)):
        x[i].requires_grad = False
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    maximum = eval_func()
    maximum_x = x
    nfeval = stateOneEpoch["func_evals"]
    niter = stateOneEpoch["n_iter"]
    return {"maximum": maximum, "maximum_x": maximum_x, "nfeval": nfeval, "niter": niter}

