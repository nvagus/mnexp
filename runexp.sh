#!/usr/bin/env bash

./run.sh dkn 7 20 5 "--use-vertical" Bin
./run.sh cnn 7 20 5 "--use-vertical" Bin
./run.sh dssm 7 20 5 "--use-vertical" FT
./run.sh deepfm 7 20 5 "--use-vertical" FT
./run.sh wnd 7 20 5 "--use-vertical" FT

#./run.sh inigru 7 20 5 "--id-keep 0.5 --use-vertical"
#./run.sh inagru 7 20 5 "--id-keep 0.5 --use-vertical"
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh igru 7 20 5 "--id-keep $rate --use-vertical"
#done
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh agru 7 20 5 "--id-keep $rate --use-vertical"
#done
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh ingru 7 20 5 "--id-keep $rate --use-vertical"
#done
#
#./run.sh avg 30 10 5
#./run.sh gru 30 10 5
#./run.sh vo 30 10 5
#./run.sh avg 30 10 5 "--use-vertical"
#./run.sh gru 30 10 5 "--use-vertical"
#./run.sh vo 30 10 5 "--use-vertical"
#
#./run.sh inigru 30 10 5 "--id-keep 0.5 --use-vertical"
#./run.sh inagru 30 10 5 "--id-keep 0.5 --use-vertical"
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh igru 30 10 5 "--id-keep $rate --use-vertical"
#done
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh agru 30 10 5 "--id-keep $rate --use-vertical"
#done
#
#for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#    ./run.sh ingru 30 10 5 "--id-keep $rate --use-vertical"
#done
