#!/usr/bin/env bash
STARTTIME=$(date +%s)
time python3 main.py --image "Tsukuba"
time python3 main.py --image "Teddy"
time python3 main.py --image "Venus"
time python3 main.py --image "Cones"; 
ENDTIME=$(date +%s)
echo "Time elpased $(($ENDTIME - $STARTTIME)) seconds"