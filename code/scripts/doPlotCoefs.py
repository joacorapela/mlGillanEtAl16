
import sys
import pdb
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append("../src")
import plotting

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_filename", help="results filenane",
                        default="../../results/hbiResults_v1.0.npz")
    # parser.add_argument("--results_filename", help="results filenane",
    #                     default="../../results/allResults_v1.0.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filenane_pattern",
                        default="../../figures/hbi_coefs_v1.0.{:s}")

    args = parser.parse_args()
    results_filename = args.results_filename
    fig_filename_pattern = args.fig_filename_pattern

    root, ext = os.path.splitext(results_filename)
    if ext==".csv":
        results = pd.read_csv(results_filename)
        results = results.drop(columns=['subject_filename', 'll_wo_constants', 'func_evals', 'iterations'])
    elif ext==".npz":
        np_data = np.load(results_filename)
        results = pd.DataFrame(data=np_data["subjects_params"],
                               columns=np_data["subjects_params_colnames"])
    else:
        raise ValueError("Extension must be csv or npz")
    results_dict = results.to_dict(orient="list")

    fig = plotting.getBoxPlots(data=results_dict, ylab="Value")
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
