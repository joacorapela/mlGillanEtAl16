
import numpy as np
import autograd.numpy as anp

class DawRLmodel:
    def __init__(self, params, subject_data, init_value_function, state_colname,
                 s1_response_colname, s2_response_colname, reward_colname):
        # params = [alpha, beta_stage2, beta_MB, beta_MF0, beta_MF1, beta_stick]
        self._params = params
        self._subject_data = subject_data
        self._init_value_function = init_value_function
        self._state_colname=state_colname
        self._s1_response_colname=s1_response_colname
        self._s2_response_colname=s2_response_colname
        self._reward_colname=reward_colname

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    def buidStage2ValueFunction(self, alpha, initial_value, trials,
                                s2_responses, states, state_colname,
                                s2_response_colname, reward_colname,
                                subject_data):
        # Q_stage2 \in states x s2_responses x n_trials
        n_trials = len(trials)
        Q_stage2 = anp.empty((len(states), len(s2_responses), len(trials)),
                               dtype=anp.double)
        Q_stage2[:,:,0] = initial_value
        for t in range(1, n_trials):
            Q_stage2[:,:,t] = (1-alpha)*Q_stage2[:,:,t-1].copy()

            trial_state = subject_data.iloc[t][state_colname]
            state_index = np.where(states==trial_state)[0]
            trial_s2_response = subject_data.iloc[t][s2_response_colname]
            s2_response_index = np.where(s2_responses==trial_s2_response)[0]
            Q_stage2[state_index, s2_response_index, t] = \
                Q_stage2[state_index, s2_response_index, t].copy() + \
                subject_data.iloc[t][reward_colname]
        return Q_stage2

    def mostProbableStateForR1(self, r1):
        if r1 == "left":
            state = 2
        elif r1 == "right":
            state = 3
        else:
            raise ValueError("r1 should be 2 or 3")
        return state

    def buildModelBasedValueFunction(self, Q_stage2, s1_responses, states):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(s1_responses)
        Q_MB = anp.empty((n_r1, n_trials), dtype=anp.double)

        for r1_index in range(n_r1):
            for t in range(n_trials):
                state = self.mostProbableStateForR1(r1=s1_responses[r1_index])
                state_index = np.where(states==state)[0]
                Q_MB[r1_index, t] = anp.max(Q_stage2[state_index, :, t])
        return Q_MB

    def buildModelFree0ValueFunction(self, alpha, initial_value, Q_stage2,
                                     s1_responses, s2_responses, states,
                                     state_colname, s1_response_colname,
                                     s2_response_colname, subject_data):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(s1_responses)
        Q_MF0 = anp.empty((n_r1, n_trials), dtype=anp.double)

        Q_MF0[:,0] = initial_value
        for t in range(1, n_trials):
            Q_MF0[:,t] = (1-alpha)*Q_MF0[:,t-1].copy()

            trial_state = subject_data.iloc[t][state_colname]
            state_index = np.where(states==trial_state)[0]
            trial_s1_response = subject_data.iloc[t][s1_response_colname]
            s1_response_index = np.where(s1_responses==trial_s1_response)[0]
            trial_s2_response = subject_data.iloc[t][s2_response_colname]
            s2_response_index = np.where(s2_responses==trial_s2_response)[0]
            Q_MF0[s1_response_index, t] = Q_MF0[s1_response_index, t].copy() + \
                                          Q_stage2[state_index,
                                                   s2_response_index, t]
        return Q_MF0

    def buildModelFree1ValueFunction(self, alpha, initial_value, Q_stage2,
                                     s1_responses, s2_responses, states, 
                                     s1_response_colname, reward_colname,
                                     subject_data):
        n_trials = Q_stage2.shape[2]
        n_r1 = len(s1_responses)
        Q_MF1 = anp.empty((n_r1, n_trials), dtype=anp.double)

        Q_MF1[:,0] = initial_value
        for t in range(1, n_trials):
            Q_MF1[:,t] = (1-alpha)*Q_MF1[:,t-1].copy()

            trial_s1_response = subject_data.iloc[t][s1_response_colname]
            s1_response_index = np.where(s1_responses==trial_s1_response)[0]
            trial_reward = subject_data.iloc[t][reward_colname]
            Q_MF1[s1_response_index, t] = Q_MF1[s1_response_index, t].copy() + \
                                          trial_reward
        return Q_MF1

    def buildPStateGivenS1Response(self, states, s1_responses):
        # p_s_given_r1 \in states x s1_responses
        p_s_given_r1 = anp.empty((2,2), dtype=anp.double)
        state_2_index = np.where(states==2)[0]
        state_3_index = np.where(states==3)[0]
        s1_response_left_index = np.where(s1_responses=="left")[0]
        s1_response_right_index = np.where(s1_responses=="right")[0]
        p_s_given_r1[state_2_index, s1_response_left_index] = 0.7
        p_s_given_r1[state_2_index, s1_response_right_index] = 0.3
        p_s_given_r1[state_3_index, s1_response_left_index] = 0.3
        p_s_given_r1[state_3_index, s1_response_right_index] = 0.7
        return p_s_given_r1

    def buildPStateGivenResponses(self, beta_stage2, Q_stage2, p_s_given_r1):
        # p_s_given_rs \in states x s1_responses x s2_responses x n_trials
        n_states = p_s_given_r1.shape[0]
        n_r1 = p_s_given_r1.shape[1]
        n_r2 = Q_stage2.shape[1]
        n_trials = Q_stage2.shape[2]
        p_s_given_rs = anp.empty((n_states, n_r1, n_r2, n_trials),
                                    dtype=anp.double)
        for state_index in range(n_states):
            for r1_index in range(n_r1):
                p_s_given_rs[state_index, r1_index, :, :] = \
                    anp.exp(beta_stage2*Q_stage2[state_index, :, :])* \
                    p_s_given_r1[state_index, r1_index]
        normalization_factor = anp.sum(p_s_given_rs, axis=0)
        p_s_given_rs = p_s_given_rs/normalization_factor
        return p_s_given_rs

    def buildPR2GivenState(self, Q_stage2, beta_stage2):
        logP_r2_given_s = beta_stage2*Q_stage2
        normalization_factor = anp.sum(logP_r2_given_s, axis=1)
        normalization_factor = anp.expand_dims(normalization_factor, axis=1)
        logP_r2_given_s =  logP_r2_given_s/normalization_factor
        return logP_r2_given_s

    def buildLogPR1(self, Q_MB, Q_MF0, Q_MF1, beta_MB, beta_MF0, beta_MF1,
                    beta_stick, s1_responses, subject_data):
        logP_r1 = beta_MB*Q_MB+beta_MF0*Q_MF0+beta_MF1*Q_MF1
        n_trials = Q_MB.shape[1]
        for t in range(1, n_trials):
            r1_prev_t = subject_data.iloc[t-1].s1_response
            r1_prev_t_index = np.where(s1_responses==r1_prev_t)[0]
            logP_r1[r1_prev_t_index, t] = logP_r1[r1_prev_t_index, t].copy() + \
                                          beta_stick
        p_r1 = anp.exp(logP_r1)
        normalization_factor = anp.sum(p_r1, axis=0)
        p_r1 = p_r1/normalization_factor
        logP_r1 = anp.log(p_r1)
        return logP_r1

    def buildLogPStateResponses(self, logP_r2_given_s, p_s_given_r1, logP_r1,
                                states, s1_responses):
        n_states = len(states)
        n_r1 = p_s_given_r1.shape[1]
        n_r2 = logP_r2_given_s.shape[1]
        n_trials = logP_r2_given_s.shape[2]
        logP_states_rs = anp.empty((n_states, n_r1, n_r2, n_trials),
                                    dtype=anp.double)
        for s_index in range(n_states):
            for r1_index in range(n_r1):
                logP_states_rs[s_index, r1_index, :, :] = \
                        logP_r2_given_s[s_index, :, :] + \
                        anp.log(p_s_given_r1[s_index, r1_index]) + \
                        logP_r1[r1_index, :]
        return logP_states_rs

    def complete_ll_expectation(self):
        alpha = self._params[0]
        beta_stage2 = self._params[1]
        beta_MB = self._params[2]
        beta_MF0 = self._params[3]
        beta_MF1 = self._params[4]
        beta_stick = self._params[5]

        # alpha = 0.5
        # beta_stage2 = self._params[0]
        # beta_MB = self._params[1]
        # beta_MF0 = self._params[2]
        # beta_MF1 = self._params[3]
        # beta_stick = self._params[4]

        trials = self._subject_data.index
        states = np.sort(self._subject_data[self._state_colname].unique())
        s2_responses = self._subject_data[self._s2_response_colname].unique()
        s1_responses = self._subject_data[self._s1_response_colname].unique()

        Q_stage2 = self.buidStage2ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            trials=trials, states=states, s2_responses=s2_responses,
            state_colname=self._state_colname,
            s2_response_colname=self._s2_response_colname,
            reward_colname=self._reward_colname,
            subject_data=self._subject_data)
        Q_MB = self.buildModelBasedValueFunction(Q_stage2=Q_stage2,
                                                 s1_responses=s1_responses,
                                                 states=states)
        Q_MF0 = self.buildModelFree0ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            Q_stage2=Q_stage2, s1_responses=s1_responses,
            s2_responses=s2_responses, states=states,
            state_colname=self._state_colname,
            s1_response_colname=self._s1_response_colname,
            s2_response_colname=self._s2_response_colname,
            subject_data=self._subject_data)
        Q_MF1 = self.buildModelFree1ValueFunction(
            alpha=alpha, initial_value=self._init_value_function,
            Q_stage2=Q_stage2, s1_responses=s1_responses,
            s2_responses=s2_responses, states=states,
            s1_response_colname=self._s1_response_colname,
            reward_colname=self._reward_colname,
            subject_data=self._subject_data)

        p_s_given_r1 = self.buildPStateGivenS1Response(
            states=states, s1_responses=s1_responses)
        p_s_given_rs = self.buildPStateGivenResponses(beta_stage2=beta_stage2,
                                                      Q_stage2=Q_stage2,
                                                      p_s_given_r1=p_s_given_r1)
        logP_r2_given_s = self.buildPR2GivenState(Q_stage2=Q_stage2,
                                                  beta_stage2=beta_stage2)
        logP_r1 = self.buildLogPR1(Q_MB=Q_MB, Q_MF0=Q_MF0, Q_MF1=Q_MF1,
                                   beta_MB=beta_MB, beta_MF0=beta_MF0,
                                   beta_MF1=beta_MF1, s1_responses=s1_responses,
                                   beta_stick=beta_stick,
                                   subject_data=self._subject_data)
        logP_s_rs = self.buildLogPStateResponses(
            logP_r2_given_s=logP_r2_given_s, p_s_given_r1=p_s_given_r1,
            logP_r1=logP_r1, states=states, s1_responses=s1_responses)
        answer = anp.sum(p_s_given_rs*logP_s_rs)/anp.size(p_s_given_rs)
        # print("eLL={:f}".format(answer)); print("params:"); print(self._params); import pdb; pdb.set_trace()
        print("eLL={:f}".format(answer)); print("params:"); print(self._params)
        return answer

