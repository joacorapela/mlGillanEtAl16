
import torch
import autograd
import scipy.optimize

def maximize_LBFGSB(func, x0, bounds, optim_params):
    func_grad = autograd.grad(func)
    # being debug
    print("func(x0)={:f}".format(func(x0)))
    print("func_grad(x0)=", func_grad(x0))
    import pdb; pdb.set_trace()
    # end debug
    optim_res = scipy.optimize.minimize(fun=func, x0=x0, method='L-BFGS-B',
                                        jac=func_grad, options=optim_params)
#     optim_res = scipy.optimize.fmin_l_bfgs_b(func=func, x0=x0,
#                                              jac=func_grad,
#                                              # approx_grad=1,
#                                              bounds=bounds,
#                                              maxiter=optim_params["max_iter"]
#                                             )
    maximum = optim_res[1]
    maximum_x = optim_res[0]
    nfeval = optim_res[2]["funcalls"]
    niter = optim_res[2]["nit"]
    return {"maximum": maximum, "maximum_x": maximum_x, "nfeval": nfeval, "niter": niter}

# def maximize_LBFGS_torch(x, eval_func, bounds, n_iter=1000, params_change_tol=1e-6, max_LBFGS_iter=1):
def maximize_LBFGS_torch(x, eval_func, bounds, n_iter=1,
                         params_change_tol=1e-6, max_LBFGS_iter=1000):
    # note: when eval_func should use params
    for i in range(len(x)):
        x[i].requires_grad = True
    optimizer = torch.optim.LBFGS(x, max_iter=max_LBFGS_iter)

    def closure():
        optimizer.zero_grad()
        curEval = -eval_func()
        curEval.backward()
        return  curEval

    params_change = torch.tensor(float("inf"))
    i = 0
    while i<n_iter and params_change>params_change_tol:
        old_x = x[0].clone()
        optimizer.step(closure)
        print("params_change:", params_change.item())
#         with torch.no_grad():
#             for i, xi in enumerate(x[0]):
#                 if bounds[i] is not None:
#                     xi.clamp_(bounds[i][0], bounds[i][1])
        params_change = torch.mean((old_x-x[0])**2)
        i += 1

    for i in range(len(x)):
        x[i].requires_grad = False
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    maximum = -eval_func()
    maximum_x = x[0].clone()
    nfeval = stateOneEpoch["func_evals"]
    niter = stateOneEpoch["n_iter"]
    return {"maximum": maximum, "maximum_x": maximum_x, "nfeval": nfeval, "niter": niter}

