
import torch


class NormalLogPosteriorModel(torch.nn.Module):

    def __init__(self, mu, sigma2, theta):
        super(NormalLogPosteriorModel, self).__init__()
        self._normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=torch.diag(sigma2))
        self.register_parameter("theta", torch.nn.Parameter(theta))

    def logPosterior(self, prior_params):
        value = torch.exp(self._normal_distribution.log_prob(self.theta))
        return value
