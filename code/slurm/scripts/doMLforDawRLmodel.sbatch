#!/bin/csh

set subjects_filenames=./subjects_filenames.txt

set subject_filename=(`awk -v myNR=$SLURM_ARRAY_TASK_ID 'NR==myNR {print $1}' $subjects_filenames`)

echo "Starting task " $SLURM_ARRAY_TASK_ID " on " $SLURMD_NODENAME

cd ../../scripts

python -u doMLforDawRLmodel_scipy_LBFGSB.py --subject_filename $subject_filename

cd -

