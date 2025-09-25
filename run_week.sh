#!/bin/bash

echo "Running experiment for 2019-10-08..."
./run.sh --processing-date=2019-10-06 --vehicle-type yellow --boroughs Manhattan --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 45

echo "Running experiment for 2019-10-09..."
./run.sh --processing-date=2019-10-10 --vehicle-type yellow --boroughs Manhattan --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 45


echo "Running experiment for 2019-10-08..."
./run.sh --processing-date=2019-10-06 --vehicle-type fhv --boroughs Manhattan --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 30

echo "Running experiment for 2019-10-09..."
./run.sh --processing-date=2019-10-10 --vehicle-type fhv --boroughs Manhattan --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 30 --log-level INFO --num-workers 30 --time-window-size 30

echo "Running experiment for 2019-10-08..."
./run.sh --processing-date=2019-10-06 --vehicle-type fhv --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120

echo "Running experiment for 2019-10-09..."
./run.sh --processing-date=2019-10-10 --vehicle-type fhv --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120


echo "Running experiment for 2019-10-08..."
./run.sh --processing-date=2019-10-06 --vehicle-type yellow --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120

echo "Running experiment for 2019-10-09..."
./run.sh --processing-date=2019-10-10 --vehicle-type yellow --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120


echo "Running experiment for 2019-10-08..."
./run.sh --processing-date=2019-10-06 --vehicle-type green --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120

echo "Running experiment for 2019-10-09..."
./run.sh --processing-date=2019-10-10 --vehicle-type green --boroughs EWR --methods LP MinMaxCostFlow MAPS LinUCB --start-hour 00 --end-hour 23 --time-delta 15 --num-iter 60 --log-level INFO --num-workers 30 --time-window-size 120

