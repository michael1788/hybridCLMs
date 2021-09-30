#!/bin/bash

CONFIGFILE=$1

if [ $# -eq 0 ] ; then
    echo "Configfile path not supplied."
else

python do_novo.py --configfile $CONFIGFILE --repeat 0

fi