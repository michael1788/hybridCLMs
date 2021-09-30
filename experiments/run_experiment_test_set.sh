#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

# split 4 is for the test set fold
python do_experiment.py --configfile $CONFIGFILE --split 4 --repeat 0

fi