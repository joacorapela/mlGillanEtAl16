# Python code for the estimation of the parameters of the combined-controller model in [Gillan et al., 2016](https://elifesciences.org/articles/11305)

## Maximum likelihood estimation

A Python script for maximum likelihood estimation of the parameters of this model is provided in [code/scripts/doMLforDawRLmodel_scipy_LBFGSB.py](code/scripts/doMLforDawRLmodel_scipy_LBFGSB.py). Use the parameter *--subject_filename* to specify the location of the subject data filename.

A SLURM script to simultaneously maximum likelihood estimation of the parameters of multiple subjects appears at [code/slurm/scripts/doMLforDawRLmodel.csh](code/slurm/scripts/doMLforDawRLmodel.csh). Please specify the location of the subjects data filenames in the file *code/slurm/scripts/subjectsFilename.txt*.

A report summarizing the results of using of this maximum likelhood estimation to characterize behavioral data from 253 subjects is given [doc/dawRLmodel.pdf](doc/dawRLmode.pdf).

## Bayesian hierarchical random effect model

A Python script to estimate the parameters of this model using a Bayesian hierarchical model in [Huys et al., 2011](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002028) is given in [code/scripts/doHBIEstimation.py](code/scripts/doHBIEstimation.py).

A report summarizing the results of using of this Bayesian hierarchical estimation to characterize behavioral data from 253 subjects is given [doc/hbiDawRLmodel.pdf](doc/hbiDawRLmodel.pdf).

