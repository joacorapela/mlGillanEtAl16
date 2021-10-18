
import numpy as np
import torch
import gpytorch


class DawRLmodel(gpytorch.Module):
    def __init__(self,
                 alpha_0, alpha_bounds,
                 beta_stage2_0, beta_stage2_bounds,
                 beta_MB_0, beta_MB_bounds,
                 beta_MF0_0, beta_MF0_bounds,
                 beta_MF1_0, beta_MF1_bounds,
                 beta_stick_0, beta_stick_bounds,
                 subject_data, init_value_function, state_colname,
                 r1_colname, r2_colname, reward_colname):
        super(DawRLmodel, self).__init__()

        # alpha_0
        self.alpha_constraint = gpytorch.constraints.Interval(lower_bound=alpha_bounds[0], upper_bound=alpha_bounds[1])
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(self.alpha_constraint.inverse_transform(alpha_0)))
        self.register_constraint("raw_alpha", self.alpha_constraint)

        # beta_stage2_0
        self.beta_stage2_constraint = gpytorch.constraints.Interval(lower_bound=beta_stage2_bounds[0], upper_bound=beta_stage2_bounds[1])
        self.register_parameter(
            name="raw_beta_stage2",
            parameter=torch.nn.Parameter(self.beta_stage2_constraint.inverse_transform(beta_stage2_0)))
        self.register_constraint("raw_beta_stage2", self.beta_stage2_constraint)

        # beta_MB_0
        self.beta_MB_constraint = gpytorch.constraints.Interval(lower_bound=beta_MB_bounds[0], upper_bound=beta_MB_bounds[1])
        self.register_parameter(
            name="raw_beta_MB",
            parameter=torch.nn.Parameter(self.beta_MB_constraint.inverse_transform(beta_MB_0)))
        self.register_constraint("raw_beta_MB", self.beta_MB_constraint)

        # beta_MF0_0
        self.beta_MF0_constraint = gpytorch.constraints.Interval(lower_bound=beta_MF0_bounds[0], upper_bound=beta_MF0_bounds[1])
        self.register_parameter(
            name="raw_beta_MF0",
            parameter=torch.nn.Parameter(self.beta_MF0_constraint.inverse_transform(beta_MF0_0)))
        self.register_constraint("raw_beta_MF0", self.beta_MF0_constraint)

        # beta_MF1_0
        self.beta_MF1_constraint = gpytorch.constraints.Interval(lower_bound=beta_MF1_bounds[0], upper_bound=beta_MF1_bounds[1])
        self.register_parameter(
            name="raw_beta_MF1",
            parameter=torch.nn.Parameter(self.beta_MF1_constraint.inverse_transform(beta_MF1_0)))
        self.register_constraint("raw_beta_MF1", self.beta_MF1_constraint)

        # beta_stick_0
        self.beta_stick_constraint = gpytorch.constraints.Interval(lower_bound=beta_stick_bounds[0], upper_bound=beta_stick_bounds[1])
        self.register_parameter(
            name="raw_beta_stick",
            parameter=torch.nn.Parameter(self.beta_stick_constraint.inverse_transform(beta_stick_0)))
        self.register_constraint("raw_beta_stick", self.beta_stick_constraint)

        self._subject_data = subject_data
        self._init_value_function = init_value_function
        self._state_colname=state_colname
        self._r1_colname=r1_colname
        self._r2_colname=r2_colname
        self._reward_colname=reward_colname
        self._trials = self._subject_data.index
        self._states = np.sort(self._subject_data[self._state_colname].unique())
        self._r2s = self._subject_data[self._r2_colname].unique()
        self._r1s = self._subject_data[self._r1_colname].unique()


        self._selected_r2_given_s_indicator = \
            self._buildSelectedR2GivenSIndicator(trials=self._trials, 
                                                 r2s=self._r2s,
                                                 states=self._states,
                                                 state_colname=state_colname,
                                                 r2_colname=r2_colname,
                                                 subject_data=subject_data)
        r1s = self._subject_data[self._r1_colname].unique()
        self._selected_r1_indicator = \
            self._buildSelectedR1Indicator(trials=self._trials,
                                           r1s=self._r1s,
                                           r1_colname=r1_colname,
                                           subject_data=subject_data)
    @property
    def alpha(self):
        value = self.alpha_constraint.transform(self.raw_alpha)
        return value

    @alpha.setter
    def alpha(self, value):
        self.initialize(raw_alpha=self.alpha_constraint.inverse_transform(value))

    @property
    def beta_stage2(self):
        value = self.beta_stage2_constraint.transform(self.raw_beta_stage2)
        return value

    @beta_stage2.setter
    def beta_stage2(self, value):
        self.initialize(raw_beta_stage2=self.beta_stage2_constraint.inverse_transform(value))

    @property
    def beta_MB(self):
        value = self.beta_MB_constraint.transform(self.raw_beta_MB)
        return value

    @beta_MB.setter
    def beta_MB(self, value):
        self.initialize(raw_beta_MB=self.beta_MB_constraint.inverse_transform(value))

    @property
    def beta_MF0(self):
        value = self.beta_MF0_constraint.transform(self.raw_beta_MF0)
        return value

    @beta_MF0.setter
    def beta_MF0(self, value):
        self.initialize(raw_beta_MF0=self.beta_MF0_constraint.inverse_transform(value))

    @property
    def beta_MF1(self):
        value = self.beta_MF1_constraint.transform(self.raw_beta_MF1)
        return value

    @beta_MF1.setter
    def beta_MF1(self, value):
        self.initialize(raw_beta_MF1=self.beta_MF1_constraint.inverse_transform(value))

    @property
    def beta_stick(self):
        value = self.beta_stick_constraint.transform(self.raw_beta_stick)
        return value

    @beta_stick.setter
    def beta_stick(self, value):
        self.initialize(raw_beta_stick=self.beta_stick_constraint.inverse_transform(value))

    def _buildSelectedR2GivenSIndicator(self, trials, r2s, states,
                                        state_colname, r2_colname,
                                        subject_data):
        # indicator \in states x r2s x n_trials

        n_trials = len(trials)
        indicator = torch.zeros((len(states), len(r2s), n_trials),
                                 dtype=torch.double)
        for t in range(1, n_trials):
            trial_state = subject_data.iloc[t][state_colname]
            state_index = np.where(states==trial_state)[0]
            trial_r2 = subject_data.iloc[t][r2_colname]
            r2_index = np.where(r2s==trial_r2)[0]
            indicator[state_index, r2_index, t] = 1
        return indicator

    def _buildSelectedR1Indicator(self, trials, r1s, r1_colname, subject_data):
        n_trials = len(trials)
        indicator = torch.zeros((len(r1s), n_trials), dtype=torch.uint8)
        for t in range(n_trials):
            r1_t = subject_data.iloc[t][r1_colname]
            r1_t_index = np.where(r1s==r1_t)[0]
            indicator[r1_t_index, t] = 1
        return indicator

    def buidStage2ValueFunction(self, alpha, initial_value, trials,
                                r2s, states, state_colname,
                                r2_colname, reward_colname,
                                subject_data):
        # Q_stage2 \in states x r2s x n_trials
        n_trials = len(trials)
        Q_stage2 = torch.empty((len(states), len(r2s), len(trials)),
                               dtype=torch.double)
        Q_stage2[:,:,0] = initial_value
        for tp1 in range(1, n_trials):
            Q_stage2[:,:,tp1] = (1-alpha)*Q_stage2[:,:,tp1-1].clone()

            trial_state = subject_data.iloc[tp1-1][state_colname]
            state_index = np.where(states==trial_state)[0]
            trial_r2 = subject_data.iloc[tp1-1][r2_colname]
            r2_index = np.where(r2s==trial_r2)[0]
            Q_stage2[state_index, r2_index, tp1] = \
                Q_stage2[state_index, r2_index, tp1].clone() + \
                subject_data.iloc[tp1-1][reward_colname]
        return Q_stage2

    def mostProbableStateForR1(self, r1):
        if r1 == "left":
            state = 2
        elif r1 == "right":
            state = 3
        else:
            raise ValueError("r1 should be 2 or 3")
        return state

    def buildModelBasedValueFunction(self, Q_stage2, r1s, states):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(r1s)
        Q_MB = torch.empty((n_r1, n_trials), dtype=torch.double)

        for r1_index in range(n_r1):
            state = self.mostProbableStateForR1(r1=r1s[r1_index])
            state_index = np.where(states==state)[0]
            for t in range(n_trials):
                Q_MB[r1_index, t] = torch.max(Q_stage2[state_index, :, t])
        return Q_MB

    def buildModelFree0ValueFunction(self, alpha, initial_value, Q_stage2,
                                     r1s, r2s, states,
                                     state_colname, r1_colname,
                                     r2_colname, subject_data):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(r1s)
        Q_MF0 = torch.empty((n_r1, n_trials), dtype=torch.double)

        Q_MF0[:,0] = initial_value
        for tp1 in range(1, n_trials):
            Q_MF0[:,tp1] = (1-alpha)*Q_MF0[:,tp1-1].clone()

            trial_state = subject_data.iloc[tp1-1][state_colname]
            state_index = np.where(states==trial_state)[0]
            trial_r1 = subject_data.iloc[tp1-1][r1_colname]
            r1_index = np.where(r1s==trial_r1)[0]
            trial_r2 = subject_data.iloc[tp1-1][r2_colname]
            r2_index = np.where(r2s==trial_r2)[0]
            Q_MF0[r1_index, tp1] = \
                Q_MF0[r1_index, tp1].clone() + \
                Q_stage2[state_index, r2_index, tp1-1]
        return Q_MF0

    def buildModelFree1ValueFunction(self, alpha, initial_value, Q_stage2,
                                     r1s, r2s, states, 
                                     r1_colname, reward_colname,
                                     subject_data):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(r1s)
        Q_MF1 = torch.empty((n_r1, n_trials), dtype=torch.double)

        Q_MF1[:,0] = initial_value
        for tp1 in range(1, n_trials):
            Q_MF1[:,tp1] = (1-alpha)*Q_MF1[:,tp1-1].clone()

            trial_r1 = subject_data.iloc[tp1-1][r1_colname]
            r1_index = np.where(r1s==trial_r1)[0]
            trial_reward = subject_data.iloc[tp1-1][reward_colname]
            Q_MF1[r1_index, tp1] = Q_MF1[r1_index, tp1].clone() + \
                                            trial_reward
        return Q_MF1

    def buildLogPR2GivenState(self, Q_stage2, beta_stage2):
        uLogP_r2_given_s = beta_stage2*Q_stage2
        uP_r2_given_s = torch.exp(uLogP_r2_given_s)
        normalization_factor = torch.sum(uP_r2_given_s, axis=1)
        normalization_factor = torch.unsqueeze(normalization_factor, dim=1)
        p_r2_given_s =  uP_r2_given_s/normalization_factor
        logP_r2_given_s = torch.log(p_r2_given_s)
        return logP_r2_given_s

    def buildLogPR1(self, Q_MB, Q_MF0, Q_MF1, beta_MB, beta_MF0, beta_MF1,
                    beta_stick, r1s, r1_colname, subject_data):
        logP_r1 = beta_MB*Q_MB+beta_MF0*Q_MF0+beta_MF1*Q_MF1
        n_trials = Q_MB.shape[1]
        for t in range(1, n_trials):
            r1_prev_t = subject_data.iloc[t-1][r1_colname]
            r1_prev_t_index = np.where(r1s==r1_prev_t)[0]
            logP_r1[r1_prev_t_index, t] = \
                logP_r1[r1_prev_t_index, t].clone() + beta_stick
            # assert(not logP_r1[r1_prev_t_index, t].isnan())
            # import pdb; pdb.set_trace()
        p_r1 = torch.exp(logP_r1)
        normalization_factor = torch.sum(p_r1, axis=0)
        p_r1 = p_r1/normalization_factor
        nLogP_r1 = torch.log(p_r1)
        assert(not any(torch.reshape(nLogP_r1, (-1,)).isnan()))
        return nLogP_r1

    def logLikelihood(self):
        alpha = self.alpha
        beta_stage2 = self.beta_stage2
        beta_MB = self.beta_MB
        beta_MF0 = self.beta_MF0
        beta_MF1 = self.beta_MF1
        beta_stick = self.beta_stick

        Q_stage2 = self.buidStage2ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            trials=self._trials, states=self._states, r2s=self._r2s,
            state_colname=self._state_colname,
            r2_colname=self._r2_colname,
            reward_colname=self._reward_colname,
            subject_data=self._subject_data)
        Q_MB = self.buildModelBasedValueFunction(Q_stage2=Q_stage2,
                                                 r1s=self._r1s,
                                                 states=self._states)
        Q_MF0 = self.buildModelFree0ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            Q_stage2=Q_stage2, r1s=self._r1s,
            r2s=self._r2s, states=self._states,
            state_colname=self._state_colname,
            r1_colname=self._r1_colname,
            r2_colname=self._r2_colname,
            subject_data=self._subject_data)
        Q_MF1 = self.buildModelFree1ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            Q_stage2=Q_stage2, r1s=self._r1s,
            r2s=self._r2s, states=self._states,
            r1_colname=self._r1_colname,
            reward_colname=self._reward_colname,
            subject_data=self._subject_data)

        logP_r2_given_s = self.buildLogPR2GivenState(Q_stage2=Q_stage2,
                                                     beta_stage2=beta_stage2)
        logP_r1 = self.buildLogPR1(Q_MB=Q_MB, Q_MF0=Q_MF0, Q_MF1=Q_MF1,
                                   beta_MB=beta_MB, beta_MF0=beta_MF0,
                                   beta_MF1=beta_MF1, r1s=self._r1s,
                                   beta_stick=beta_stick,
                                   r1_colname=self._r1_colname,
                                   subject_data=self._subject_data)
        sum_selected_logP_r2_given_s = \
            torch.sum(logP_r2_given_s*self._selected_r2_given_s_indicator)
        sum_selected_logP_r1 = torch.sum(logP_r1*self._selected_r1_indicator)
        log_likelihood = sum_selected_logP_r2_given_s+sum_selected_logP_r1
        print("LL = {:f}".format(log_likelihood)); print("params:"); print(list(self.parameters()))
        # import pdb; pdb.set_trace()
        return log_likelihood

    def posterior(self, mu, nu2):
        ll = self.logLikelihood()

