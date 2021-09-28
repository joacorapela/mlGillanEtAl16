# Python code for the estimation of the parameters of the combined-controller model in [Gillan et al., 2016](https://elifesciences.org/articles/11305)

A Python script to estimate the parameters of this model is provided in [code/scripts/doMLforDawRLmodel_scipy_LBFGSB.py](code/scripts/doMLforDawRLmodel_scipy_LBFGSB.py). Use the parameter *--subject_filename* to specify the location of the subject data filename.

A SLURM script to simultaneously estimate the parameters of multiple subjects appears at [code/slurm/scripts/doMLforDawRLmodel.csh](code/slurm/scripts/doMLforDawRLmodel.csh). Please specify the location of the subjects data filenames in the file *code/slurm/scripts/subjectsFilename.txt*.

A report summarizing the results of using of this code to characterize behavioral data from 253 subjects is given [doc/dawRLmodel.pdf](doc/dawRLmode.pdf). The script I used to generate Figure 1 in this report is [code/scripts/doPlotCoefs.py](code/scripts/doPlotCoefs.py).

