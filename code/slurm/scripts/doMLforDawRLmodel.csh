#!/bin/csh

set arrayOpt=1-253
    
sbatch \
--job-name=DawRLmodel \
--output=../outputs/doMLforDawRLmodel_%A_%a.out \
--error=../outputs/doMLforDawRLmodel_%A_%a.err \
--time=1:00:00 \
--mem=1G \
--array=$arrayOpt \
./doMLforDawRLmodel.sbatch

