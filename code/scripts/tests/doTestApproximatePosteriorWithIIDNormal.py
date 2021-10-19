
import sys
import torch

import logPosteriorModels
sys.path.append("../../src")
import optim

def main(argv):
    max_iter = 100
    mu = torch.tensor([4.0, 2.0], dtype=torch.double)
    sigma2 = torch.tensor([0.2, 0.3], dtype=torch.double)
    theta0 = torch.tensor([3.7, 1.6], dtype=torch.double)

    logPosteriorModel = logPosteriorModels.NormalLogPosteriorModel(mu=mu, sigma2=sigma2, theta=theta0)
    mu, nu2 = optim.approximatePosteriorWithIIDNormal(logPosteriorModel=logPosteriorModel, prior_params=None, max_iter=max_iter)

if __name__=="__main__":
    main(sys.argv)
