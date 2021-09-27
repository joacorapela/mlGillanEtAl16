
import sys
import pdb
import argparse
import pandas as pd

sys.path.append("../src")
import plotting

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_filename", help="results filenane",
                        default="../../results/allResults_v1.0.csv")
    parser.add_argument("--summary_table_filename",
                        help="summary table filenane",
                        default="../../results/summary_table_v1.0.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filenane_pattern",
                        default="../../figures/coefs_v1.0.{:s}")

    args = parser.parse_args()
    results_filename = args.results_filename
    summary_table_filename = args.summary_table_filename
    fig_filename_pattern = args.fig_filename_pattern

    results = pd.read_csv(results_filename)
    results = results.drop(columns=['subject_filename', 'll_wo_constants', 'func_evals', 'iterations'])
    results_dict = results.to_dict(orient="list")

    median = results.median()
    percentile_25p = results.quantile(q=.25)
    percentile_75p = results.quantile(q=.75)

    summary_dict = {"median": median, "25th percentile": percentile_25p, "75th percentile": percentile_75p}
    summary_table = pd.DataFrame(summary_dict).transpose()

    fig = plotting.getBoxPlots(data=results_dict, ylab="Value")
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    fig.show()

    summary_table.to_csv(summary_table_filename)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
