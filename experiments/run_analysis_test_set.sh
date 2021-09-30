#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

python do_analysis.py --configfile $CONFIGFILE --repeat 0 --test_set

fi