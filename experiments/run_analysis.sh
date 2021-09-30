#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

python do_analysis.py --configfile $CONFIGFILE --repeat 0 --not_test_set

fi