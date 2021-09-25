
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
    parser.add_argument("--init_value_function", 
                        help="initial value for value functions",
                        type=int, default=0)

    args = parser.parse_args()

    alpha = 0.3
    beta_stage2 = 0.5
    beta_MB = 0.5
    beta_MF0 = 0.5
    beta_MF1 = 0.5
    beta_stick = 1.0

    # get arguments
    subjectFilename = args.subject_filename
    columns_names = args.columns_names.split(",")
    index_colname = args.index_colname
    state_colname = args.state_colname
    s1_response_colname = args.s1_response_colname
    s2_response_colname = args.s2_response_colname
    reward_colname = args.reward_colname
    prev_line_string = args.prev_line_string
    init_value_function = args.init_value_function

    subject_data = getSubjectData(subjectFilename=subjectFilename,
                                  columns_names=columns_names,
                                  index_colname=index_colname,
                                  prev_line_string=prev_line_string)
    trials = subject_data.index
    states = np.sort(subject_data[state_colname].unique())
    s2_responses = subject_data[s2_response_colname].unique()
    s1_responses = subject_data[s1_response_colname].unique()

    Q_stage2 = buidStage2ValueFunction(alpha=alpha,
                                       initial_value=init_value_function,
                                       trials=trials, states=states,
                                       s2_responses=s2_responses,
                                       state_colname=state_colname,
                                       s2_response_colname=s2_response_colname,
                                       reward_colname=reward_colname,
                                       subject_data=subject_data)
    Q_MB = buildModelBasedValueFunction(Q_stage2=Q_stage2,
                                        s1_responses=s1_responses,
                                        states=states)
    Q_MF0 = buildModelFree0ValueFunction(alpha=alpha,
                                         initial_value=init_value_function,
                                         Q_stage2=Q_stage2,
                                         s1_responses=s1_responses,
                                         s2_responses=s2_responses,
                                         states=states,
                                         state_colname=state_colname,
                                         s1_response_colname=
                                          s1_response_colname,
                                         s2_response_colname=
                                          s2_response_colname,
                                         subject_data=subject_data)
    Q_MF1 = buildModelFree1ValueFunction(alpha=alpha, 
                                         initial_value=init_value_function,
                                         Q_stage2=Q_stage2,
                                         s1_responses=s1_responses,
                                         s2_responses=s2_responses,
                                         states=states, 
                                         s1_response_colname=
                                          s1_response_colname,
                                         reward_colname=reward_colname,
                                         subject_data=subject_data)

    p_s_given_r1 = buildPStateGivenS1Response(states=states,
                                              s1_responses=s1_responses)
    p_s_given_rs = buildPStateGivenResponses(beta_stage2=beta_stage2,
                                             Q_stage2=Q_stage2,
                                             p_s_given_r1=p_s_given_r1)
    logP_r2_given_s = beta_stage2*Q_stage2
    logP_r1 = buildLogPR1(Q_MB=Q_MB, Q_MF0=Q_MF0, Q_MF1=Q_MF1, beta_MB=beta_MB,
                          beta_MF0=beta_MF0, beta_MF1=beta_MF1,
                          s1_responses=s1_responses, beta_stick=beta_stick,
                          subject_data=subject_data)
    logP_s_rs = buildLogPStateResponses(logP_r2_given_s=logP_r2_given_s,
                                        p_s_given_r1=p_s_given_r1,
                                        logP_r1=logP_r1, states=states,
                                        s1_responses=s1_responses)
    e_logP_s_rs = torch.sum(p_s_given_rs*logP_s_rs)/torch.numel(p_s_given_rs)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
