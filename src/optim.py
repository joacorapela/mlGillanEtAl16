
import torch
import autograd
import scipy.optimize

def minimize_scipy_LBFGSB(func, x0, bounds, optim_params):
    optim_res = scipy.optimize.minimize(fun=func, x0=x0, method="L-BFGS-B",
                                        jac=True, bounds=bounds,
                                        options=optim_params)
    minimum = optim_res.fun
    minimum_x = optim_res.x
    nfeval = optim_res.nfev
    niter = optim_res.nit
    return {"minimum": minimum, "minimum_x": minimum_x, "nfeval": nfeval, "niter": niter}

