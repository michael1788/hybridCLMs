#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

for i in $(seq 0 1 3);
    do python do_experiment.py --configfile $CONFIGFILE --split $i --repeat 0
done

fi