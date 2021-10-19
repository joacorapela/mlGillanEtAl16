
import torch
import scipy.optimize


def approximatePosteriorWithIIDNormal(logPosteriorModel, prior_params, max_iter):
    def logPosteriorWrapper():
        value = logPosteriorModel.logPosterior(prior_params=prior_params)
        return value
    x = list(logPosteriorModel.parameters())
    max_res = maximize_torch_LBFGS(x=x, eval_func=logPosteriorWrapper, max_iter=max_iter)
    max_x = max_res["maximum_x"]
    # compute 2nd order derivatives of posterior
    for i in range(len(max_x)):
        max_x[i].requires_grad = True
    maximum = logPosteriorWrapper()
    grad = torch.autograd.grad(maximum, max_x[0] , create_graph=True)[0]
    ones = torch.ones_like(max_x[0])
    grad2 = torch.autograd.grad(grad, max_x, grad_outputs=ones, create_graph=True)[0]
    for i in range(len(max_x)):
        max_x[i].requires_grad = False
    #
    mean = max_x
    var = -1.0/grad2
    return mean, var

def huysEtAl11EM(logPosteriorModels, mu0, nu20, max_em_iter=100,
                 max_approxPosterior_iter=50):
    def eStep(muOld, nu2Old, logPosteriorModels, max_iter):
        N = len(individualPosteriors)
        m = torch.empty((len(mu0), N), dtype=torch.double)
        Sigma = torch.empty((len(mu0), N), dtype=torch.double)
        for i in range(N):
            m[:,i], Sigma[:,i] = \
                approximatePosteriorWithNormal(logPosteriorModel=
                                                logPosteriorModels[i],
                                               muOld=muOld, nu2Old=nu2Old,
                                               max_iter=max_iter)
        return m, Sigma

    def mStep(m, Sigma):
        mu = torch.mean(m, axis=0)
        nu2 = 2*(torch.mean(m**2+Sigma)-mu**2)
        return mu, nu2

    mu = mu0
    nu2 = nu20
    for i in range(max_iter):
        m, Sigma = eStep(muOld=mu, nu2Old=nu2, logPosteriorModels=logPosteriorModels,
                         max_iter=max_approxPosterior_iter)
        mu, nu2 = mStep(m=m, Sigma=Sigma)
    return mu, nu2

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

