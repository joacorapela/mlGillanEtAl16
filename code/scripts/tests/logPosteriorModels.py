
import torch


class NormalLogPosteriorModel:

    def __init__(self, mu, sigma2):
        super(NormalLogPosteriorModel, self).__init__()
        self._normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=torch.diag(sigma2))

    @property
    def likelihood_params(self):
        return self._likelihood_params

    @likelihood_params.setter
    def likelihood_params(self, likelihood_params):
        self._likelihood_params = likelihood_params

    def logPosterior(self, prior_params):
        value = torch.exp(self._normal_distribution.log_prob(self.likelihood_params))
        return value
