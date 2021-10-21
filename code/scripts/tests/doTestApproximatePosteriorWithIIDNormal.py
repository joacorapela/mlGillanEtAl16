
import sys
import numpy as np
import torch

import logPosteriorModels
sys.path.append("../../src")
import optim

def main(argv):
    max_iter = 100
    mu = torch.tensor([4.0, 2.0], dtype=torch.double)
    sigma2 = torch.tensor([0.2, 0.1], dtype=torch.double)
    likelihood_params0 = np.array([3.7, 1.6], dtype=np.double)
    likelihood_params_bounds = [(3.0, 5.0), (1.0, 3.0)]

    logPosteriorModel = logPosteriorModels.NormalLogPosteriorModel(mu=mu, sigma2=sigma2)
    mu, nu2 = optim.approximatePosteriorWithIIDNormal(logPosteriorModel=logPosteriorModel, likelihood_params0=likelihood_params0, likelihood_params_bounds=likelihood_params_bounds, prior_params=None, max_iter=max_iter)

if __name__=="__main__":
    main(sys.argv)
