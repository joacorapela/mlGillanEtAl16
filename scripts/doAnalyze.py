
import sys
import pdb
import argparse
import pandas as pd
import numpy as np
import torch

def getSubjectData(subjectFilename, columns_names, index_colname, prev_line_string):
    f = open(subjectFilename, "rt")
    found = False
    line = f.readline()
    while line is not None and not found:
        print(line)
        if prev_line_string in line:
            found = True
        else:
            line = f.readline()
    subject_data = pd.read_csv(f, names=columns_names)
    subject_data = subject_data.set_index(index_colname)
    return subject_data

def buidStage2ValueFunction(alpha, initial_value, trials, s2_responses,
                            states, state_colname, s2_response_colname,
                            reward_colname, subject_data):
    # Q_stage2 \in states x s2_responses x n_trials
    n_trials = len(trials)
    Q_stage2 = torch.empty((len(states), len(s2_responses), len(trials)),
                           dtype=torch.double)
    Q_stage2[:,:,0] = initial_value
    for t in range(1, n_trials):
        Q_stage2[:,:,t] = (1-alpha)*Q_stage2[:,:,t-1]

        trial_s2_response = subject_data.iloc[t][s2_response_colname]
        s2_response_index = np.where(s2_responses==trial_s2_response)[0]
        trial_state = subject_data.iloc[t][state_colname]
        state_index = np.where(states==trial_state)[0]
        Q_stage2[state_index, s2_response_index, t] += \
            subject_data.iloc[t][reward_colname]
    return Q_stage2

def mostProbableStateForR1(r1):
    if r1 == "left":
        state = 2
    elif r1 == "right":
        state = 3
    else:
        raise ValueError("r1 should be 2 or 3")
    return state

def buildModelBasedValueFunction(Q_stage2, s1_responses):
    n_trials = Q_stage2.shape[2]
    n_r1 = len(s1_responses)
    Q_MB = torch.empty((n_r1, n_trials), dtype=torch.double)

    for r1_index in range(n_r1):
        for t in range(n_trials):
            mpState_for_r1 = mostProbableStateForR1(r1=s1_responses[r1_index])
            Q_MB[r1_index, t] = torch.max(Q_stage2[mpState_for_r1, :, t])
    return Q_MB

def buildProbStateGivenS1Response(states, s1_responses):
    # p_state_given_r1 \in states x s1_responses
    p_state_given_r1 = torch.empty((2,2), dtype=torch.double)
    state_2_index = np.where(states==2)[0]
    state_3_index = np.where(states==3)[0]
    s1_response_left_index = np.where(states=="left")[0]
    s1_response_right_index = np.where(states=="right")[0]
    p_state_given_r1[state_2_index, s1_response_left_index] = 0.7
    p_state_given_r1[state_2_index, s1_response_right_index] = 0.3
    p_state_given_r1[state_3_index, s1_response_left_index] = 0.3
    p_state_given_r1[state_3_index, s1_response_right_index] = 0.7
    return p_state_given_r1

def buildProbStateGivenResponses(beta_stage2, Q_stage2, p_state_given_r1):
    # p_state_given_r1 \in states x s1_responses x s2_responses x n_trials
    n_states = p_state_given_r1.shape[0]
    n_r1 = p_state_given_r1.shape[1]
    n_trials = Q_stage2.shape[2]
    p_state_given_rs = torch.empty((n_states, nS1Responses,
                                          nS2responses, n_trials), dtype=double)
    for state_index in range(n_states):
        for r1_index in range(n_r1):
            p_state_given_rs[state_index, r1_index, :, :] = \
                torch.exp(beta_stage2*Q_stage2[state_index, :, :])* \
                p_state_given_r1[state_index, r1_index]
    return p_state_given_rs

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_filename", help="subject filenane",
                        default="../../gillan_model_assignment/individual_participant_data/3018Q3ZVOIQGSXZ9L4SKHNWVPJAART.csv")
    parser.add_argument("--columns_names", 
                        help="columns names in subject filenane",
                        default="trial#,s2_prob_reward_1,s2_prob_reward_2,s2_prob_reward_3,s2_prob_reward_4,s1_response,s1_selected_stim,s1_rt,transition,s2_response,s2_selected_stim,state,s2_rt,reward,redundant")
    parser.add_argument("--index_colname", 
                        help="column name of the index",
                        default="trial#")
    parser.add_argument("--state_colname", 
                        help="s2 state column name",
                        default="state")
    parser.add_argument("--s2_response_colname", 
                        help="s2 response column name",
                        default="s2_response")
    parser.add_argument("--s1_response_colname", 
                        help="s1 response column name",
                        default="s1_response")
    parser.add_argument("--reward_colname", 
                        help="reward column name",
                        default="reward")
    parser.add_argument("--prev_line_string", 
                        help="text in line preceding the data", 
                        default="twostep_instruct_9")

    args = parser.parse_args()

    alpha = 0.3
    beta_stage2 = 0.5

    # get arguments
    subjectFilename = args.subject_filename
    columns_names = args.columns_names.split(",")
    index_colname = args.index_colname
    state_colname = args.state_colname
    s1_response_colname = args.s1_response_colname
    s2_response_colname = args.s2_response_colname
    reward_colname = args.reward_colname
    prev_line_string = args.prev_line_string

    subject_data = getSubjectData(subjectFilename=subjectFilename,
                                  columns_names=columns_names,
                                  index_colname=index_colname,
                                  prev_line_string=prev_line_string)
    trials = subject_data.index
    states = subject_data[state_colname].unique()
    s2_responses = subject_data[s2_response_colname].unique()
    s1_responses = subject_data[s1_response_colname].unique()

    Q_stage2 = buidStage2ValueFunction(alpha=alpha, initial_value=0,
                                      trials=trials, states=states,
                                      s2_responses=s2_responses,
                                      state_colname=state_colname,
                                      s2_response_colname=s2_response_colname,
                                      reward_colname=reward_colname,
                                      subject_data=subject_data)
    Q_MB = buildModelBasedValueFunction(Q_stage2=Q_stage2,
                                        s1_responses=s1_responses)

    p_state_given_r1 = buildProbStateGivenS1Response(states=states,
                                                     s1_responses=s1_responses)
    p_state_given_rs = buildProbStateGivenResponses(beta_stage2=beta_stage2,
                                                     Q_stage2=Q_stage2,
                                                     p_state_given_r1=
                                                      p_state_given_r1)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
