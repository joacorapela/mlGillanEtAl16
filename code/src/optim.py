
import torch
import scipy.optimize


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


def maximize_torch_LBFGS(x, eval_func, max_iter=1000,
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
        curEval = -eval_func()
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

