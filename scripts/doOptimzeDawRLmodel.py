
import sys
import pdb
import argparse
import torch

sys.path.append("../src")
import models
import optim
import utils

def main(argv):
    torch.autograd.set_detect_anomaly(True)

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

    subject_data = utils.getSubjectData(subjectFilename=subjectFilename,
                                        columns_names=columns_names,
                                        index_colname=index_colname,
                                        prev_line_string=prev_line_string)

    # params = [alpha, beta_stage2, beta_MB, beta_MF0, beta_MF1, beta_stick]
    # params = [0.9, 0.5, 0.5, 0.5, 0.5, 1.0]
    params = [0.5, 0.5, 0.5, 0.5, 1.0]
    x = [torch.tensor(params, dtype=torch.double)]
    optim_params = dict(max_iter=100)

    dawRLmodel = models.DawRLmodel(params=x[0], subject_data=subject_data,
                                   init_value_function=init_value_function,
                                   state_colname=state_colname,
                                   s1_response_colname=s1_response_colname,
                                   s2_response_colname=s2_response_colname,
                                   reward_colname=reward_colname)
    max_res = optim.maximize_LBFGS(x=x,
                                   eval_func=dawRLmodel.complete_ll_expectation,
                                   optim_params=optim_params)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
