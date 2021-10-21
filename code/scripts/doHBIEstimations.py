
import sys
import pdb
import os
import time
import argparse
import numpy as np
import torch

sys.path.append("../src")
import models_pytorch as models
import optim
import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_dirname", help="directory of subjects filenanes",
                        default="../../../gillan_model_assignment/individual_participant_data")
    parser.add_argument("--subjects_filenames_filename",
                        help="filename of subjects filenanes",
                        default="../../../gillan_model_assignment/subjects_filenames.txt")
    parser.add_argument("--alpha_prior_mean_0",
                        help="initial prior mean for parameter alpha",
                        type=float, default=0.5)
    parser.add_argument("--alpha_prior_var_0",
                        help="initial prior variance for parameter alpha",
                        type=float, default=0.01)
    parser.add_argument("--alpha_0", help="initial value for parameter alpha",
                        type=float, default=0.3)
    parser.add_argument("--alpha_bounds", help="bounds for parameter alpha",
                        default="(0.1, 1.0)")
    parser.add_argument("--beta_stage2_prior_mean_0",
                        help="initial prior mean for parameter beta_stage2",
                        type=float, default=0.0)
    parser.add_argument("--beta_stage2_prior_var_0",
                        help="initial prior variance for parameter beta_stage2",
                        type=float, default=2.0)
    parser.add_argument("--beta_stage2_0", 
                        help="initial value for parameter beta_stage2",
                        type=float, default=0.7)
    parser.add_argument("--beta_stage2_bounds",
                        help="bounds for parameter beta_stage2",
                        default="(-5.0, 5.0)")
    parser.add_argument("--beta_MB_prior_mean_0",
                        help="initial prior mean for parameter beta_MB",
                        type=float, default=0.0)
    parser.add_argument("--beta_MB_prior_var_0",
                        help="initial prior variance for parameter beta_MB",
                        type=float, default=2.0)
    parser.add_argument("--beta_MB_0", 
                        help="initial value for parameter beta_MB",
                        type=float, default=0.5)
    parser.add_argument("--beta_MB_bounds",
                        help="bounds for parameter beta_MB",
                        default="(-5.0, 5.0)")
    parser.add_argument("--beta_MF0_prior_mean_0",
                        help="initial prior mean for parameter beta_MF0",
                        type=float, default=0.0)
    parser.add_argument("--beta_MF0_prior_var_0",
                        help="initial prior variance for parameter beta_MF0",
                        type=float, default=2.0)
    parser.add_argument("--beta_MF0_0", 
                        help="initial value for parameter beta_MF0",
                        type=float, default=0.5)
    parser.add_argument("--beta_MF0_bounds",
                        help="bounds for parameter beta_MF0",
                        default="(-5.0, 5.0)")
    parser.add_argument("--beta_MF1_prior_mean_0",
                        help="initial prior mean for parameter beta_MF1",
                        type=float, default=0.0)
    parser.add_argument("--beta_MF1_prior_var_0",
                        help="initial prior variance for parameter beta_MF1",
                        type=float, default=2.0)
    parser.add_argument("--beta_MF1_0", 
                        help="initial value for parameter beta_MF1",
                        type=float, default=0.5)
    parser.add_argument("--beta_MF1_bounds",
                        help="bounds for parameter beta_MF1",
                        default="(-5.0, 5.0)")
    parser.add_argument("--beta_stick_prior_mean_0",
                        help="initial prior mean for parameter beta_stick",
                        type=float, default=0.0)
    parser.add_argument("--beta_stick_prior_var_0",
                        help="initial prior variance for parameter beta_stick",
                        type=float, default=2.0)
    parser.add_argument("--beta_stick_0", 
                        help="initial value for parameter beta_stick",
                        type=float, default=0.5)
    parser.add_argument("--beta_stick_bounds",
                        help="bounds for parameter beta_stick",
                        default="(-5.0, 5.0)")
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
    parser.add_argument("--max_approxPosterior_iter", 
                        help="maximum number of iterations for posterior approximation",
                        type=int, default=50)
    parser.add_argument("--max_em_iter", 
                        help="maximum number of EM iterations",
                        type=int, default=1000)
    parser.add_argument("--max_map_iter", 
                        help="maximum number of MAP iterations",
                        type=int, default=50)
    parser.add_argument("--MAP_LBFGSB_ftol", 
                        help="MAP L-BFGS-B convergence tolereance",
                        type=float, default=2.220446049250313e-09)
    parser.add_argument("--hbi_EM_tol", 
                        help="HuysEtAl11 EM convergence tolereance",
                        type=float, default=1e-04)
    parser.add_argument("--results_filename", 
                        help="results filename",
                        default="../../results/hbiResults.npz")

    args = parser.parse_args()

    # get arguments
    subjects_dirname = args.subjects_dirname
    subjects_filenames_filename = args.subjects_filenames_filename
    alpha_prior_mean_0 = args.alpha_prior_mean_0
    alpha_prior_var_0 = args.alpha_prior_var_0
    alpha_0 = args.alpha_0
    alpha_bounds = eval(args.alpha_bounds)
    beta_stage2_prior_mean_0 = args.beta_stage2_prior_mean_0
    beta_stage2_prior_var_0 = args.beta_stage2_prior_var_0
    beta_stage2_0 = args.beta_stage2_0
    beta_stage2_bounds = eval(args.beta_stage2_bounds)
    beta_MB_prior_mean_0 = args.beta_MB_prior_mean_0
    beta_MB_prior_var_0 = args.beta_MB_prior_var_0
    beta_MB_0 = args.beta_MB_0
    beta_MB_bounds = eval(args.beta_MB_bounds)
    beta_MF0_prior_mean_0 = args.beta_MF0_prior_mean_0
    beta_MF0_prior_var_0 = args.beta_MF0_prior_var_0
    beta_MF0_0 = args.beta_MF0_0
    beta_MF0_bounds = eval(args.beta_MF0_bounds)
    beta_MF1_prior_mean_0 = args.beta_MF1_prior_mean_0
    beta_MF1_prior_var_0 = args.beta_MF1_prior_var_0
    beta_MF1_0 = args.beta_MF1_0
    beta_MF1_bounds = eval(args.beta_MF1_bounds)
    beta_stick_prior_mean_0 = args.beta_stick_prior_mean_0
    beta_stick_prior_var_0 = args.beta_stick_prior_var_0
    beta_stick_0 = args.beta_stick_0
    beta_stick_bounds = eval(args.beta_stick_bounds)
    columns_names = args.columns_names.split(",")
    index_colname = args.index_colname
    state_colname = args.state_colname
    r1_colname = args.r1_colname
    r2_colname = args.r2_colname
    reward_colname = args.reward_colname
    prev_line_string = args.prev_line_string
    init_value_function = args.init_value_function
    max_approxPosterior_iter = args.max_approxPosterior_iter
    max_em_iter = args.max_em_iter
    max_map_iter = args.max_map_iter
    MAP_LBFGSB_ftol = args.MAP_LBFGSB_ftol
    hbi_EM_tol = args.hbi_EM_tol
    results_filename = args.results_filename

    with open(subjects_filenames_filename, "r") as f:
        subjects_filenames = f.read().splitlines()

    named_models = []
    for subject_filename in subjects_filenames:
        subject_data = utils.getSubjectData(subject_dirname=subjects_dirname,
                                            subject_filename=subject_filename,
                                            columns_names=columns_names,
                                            index_colname=index_colname,
                                            prev_line_string=prev_line_string)
        model = models.DawRLmodel(subject_data=subject_data,
                                   init_value_function=init_value_function,
                                   state_colname=state_colname,
                                   r1_colname=r1_colname,
                                   r2_colname=r2_colname,
                                   reward_colname=reward_colname)
        named_models.append({"label": subject_filename, "model": model})

    likelihood_params0 = [alpha_0, beta_stage2_0, beta_MB_0, beta_MF0_0,
                          beta_MF1_0, beta_stick_0]
    likelihood_params_bounds = [alpha_bounds, beta_stage2_bounds,
                                beta_MB_bounds, beta_MF0_bounds,
                                beta_MF1_bounds, beta_stick_bounds]
    prior_means0 = torch.tensor([alpha_prior_mean_0, 
                                 beta_stage2_prior_mean_0,
                                 beta_MB_prior_mean_0,
                                 beta_MF0_prior_mean_0,
                                 beta_MF1_prior_mean_0,
                                 beta_stick_prior_mean_0])
    prior_vars0 = torch.tensor([alpha_prior_var_0, 
                                 beta_stage2_prior_var_0,
                                 beta_MB_prior_var_0,
                                 beta_MF0_prior_var_0,
                                 beta_MF1_prior_var_0,
                                 beta_stick_prior_var_0])
    prior_params0 = [prior_means0, prior_vars0]

    prior_params, prior_iteration_runtimes = optim.estimate_hbi_prior_params(
        log_posterior_models=named_models,
        likelihood_params0=likelihood_params0,
        prior_params0=prior_params0,
        likelihood_params_bounds=likelihood_params_bounds,
        max_approxPosterior_iter=max_approxPosterior_iter,
        max_em_iter=max_em_iter,
        MAP_LBFGSB_ftol=MAP_LBFGSB_ftol,
        EM_tol=hbi_EM_tol,
    )

    # begin debug
    # prior_params = prior_params0
    # end debug

    log_posterior_models = [named_model["model"] for named_model in named_models]
    start_time = time.time()
    subjects_params = optim.estimate_multiple_models_map_params(
        log_posterior_models=log_posterior_models,
        likelihood_params0=likelihood_params0,
        likelihood_params_bounds=likelihood_params_bounds,
        prior_params=prior_params,
        max_iter=max_map_iter,
        LBFGSB_ftol=MAP_LBFGSB_ftol)
    estimation_subjects_params_elapsed_time = time.time()-start_time

    subjects_labels = [named_model["label"] for named_model in named_models]
    prior_params_np = [prior_param.detach().numpy() for prior_param in prior_params]
    subjects_params_colnames = ["alpha", "beta_stage2", "beta_MB", "beta_MF0", "beta_MF1", "beta_stick"]
    np.savez(results_filename, prior_params=prior_params_np,
             subjects_labels=subjects_labels, subjects_params=subjects_params,
             subjects_params_colnames=subjects_params_colnames,
             prior_iteration_runtimes=prior_iteration_runtimes,
             estimation_subjects_params_elapsed_time=estimation_subjects_params_elapsed_time)

    # pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
