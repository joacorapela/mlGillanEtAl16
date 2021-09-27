
import sys
import pdb
import argparse
import torch

sys.path.append("../src")
import models_pytorch as models
import optim
import utils

def main(argv):
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_filename", help="subject filenane",
                        default="../../gillan_model_assignment/individual_participant_data/3018Q3ZVOIQGSXZ9L4SKHNWVPJAART.csv")
    parser.add_argument("--alpha_0", help="initial value for parameter alpha",
                        type=float, default=0.3)
    parser.add_argument("--beta_stage2_0", 
                        help="initial value for parameter beta_stage2",
                        type=float, default=0.7)
    parser.add_argument("--beta_MB_0", 
                        help="initial value for parameter beta_MB",
                        type=float, default=0.5)
    parser.add_argument("--beta_MF0_0", 
                        help="initial value for parameter beta_MF0",
                        type=float, default=0.5)
    parser.add_argument("--beta_MF1_0", 
                        help="initial value for parameter beta_MF1",
                        type=float, default=0.5)
    parser.add_argument("--beta_stick_0", 
                        help="initial value for parameter beta_stick",
                        type=float, default=0.5)
    parser.add_argument("--columns_names", 
                        help="columns names in subject filenane",
                        default="trial#,s2_prob_reward_1,s2_prob_reward_2,s2_prob_reward_3,s2_prob_reward_4,s1_response,s1_selected_stim,s1_rt,transition,s2_response,s2_selected_stim,state,s2_rt,reward,redundant")
    parser.add_argument("--index_colname", 
                        help="column name of the index",
                        default="trial#")
    parser.add_argument("--state_colname", 
                        help="s2 state column name",
                        default="state")
    parser.add_argument("--r1_colname", 
                        help="s1 response column name",
                        default="s1_response")
    parser.add_argument("--r2_colname", 
                        help="s2 response column name",
                        default="s2_response")
    parser.add_argument("--reward_colname", 
                        help="reward column name",
                        default="reward")
    parser.add_argument("--prev_line_string", 
                        help="text in line preceding the data", 
                        default="twostep_instruct_9")
    parser.add_argument("--init_value_function", 
                        help="initial value for value functions",
                        type=int, default=0.1)

    args = parser.parse_args()

    # get arguments
    subjectFilename = args.subject_filename
    alpha_0 = args.alpha_0
    beta_stage2_0 = args.beta_stage2_0
    beta_MB_0 = args.beta_MB_0
    beta_MF0_0 = args.beta_MF0_0
    beta_MF1_0 = args.beta_MF1_0
    beta_stick_0 = args.beta_stick_0
    columns_names = args.columns_names.split(",")
    index_colname = args.index_colname
    state_colname = args.state_colname
    r1_colname = args.r1_colname
    r2_colname = args.r2_colname
    reward_colname = args.reward_colname
    prev_line_string = args.prev_line_string
    init_value_function = args.init_value_function

    subject_data = utils.getSubjectData(subjectFilename=subjectFilename,
                                        columns_names=columns_names,
                                        index_colname=index_colname,
                                        prev_line_string=prev_line_string)

    params = [alpha_0, beta_stage2_0, beta_MB_0, beta_MF0_0, beta_MF1_0,
              beta_stick_0]
    bounds = [(0.01, .99), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
              (-5.0, 5.0)]
    x = [torch.tensor(params, dtype=torch.double)]

    dawRLmodel = models.DawRLmodel(params=x[0], subject_data=subject_data,
                                   init_value_function=init_value_function,
                                   state_colname=state_colname,
                                   r1_colname=r1_colname,
                                   r2_colname=r2_colname,
                                   reward_colname=reward_colname)
    max_res = optim.maximize_torch_LBFGS(x=x,
                                         eval_func=dawRLmodel.logLikelihood,
                                         bounds=bounds)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
