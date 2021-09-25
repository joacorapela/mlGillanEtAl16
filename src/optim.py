
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

def maximize_LBFGS(x, eval_func, optim_params):
    # note: when eval_func should use params
    for i in range(len(x)):
        x[i].requires_grad = True
    optimizer = torch.optim.LBFGS(x, **optim_params)
    def closure():
        optimizer.zero_grad()
        curEval = -eval_func()
        print(curEval)
        # curEval.backward(retain_graph=True)
        curEval.backward()
        return curEval
    optimizer.step(closure)
    for i in range(len(x)):
        x[i].requires_grad = False
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    maximum = -eval_func()
    maximum_x = x.clone()
    nfeval = stateOneEpoch["func_evals"]
    niter = stateOneEpoch["n_iter"]
    return {"maximum": maximum, "maximum_x": maximum_x, "nfeval": nfeval, "niter": niter}


