#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

# split 16 is to train on all the data (CV folds + test set)
# Note that the number of models for the ensemble if defined in 
# the configuration file
python do_experiment.py --configfile $CONFIGFILE --split 16 --repeat 0

fi