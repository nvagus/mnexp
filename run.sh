#!/usr/bin/env bash

for ((i=0; i<$4; i++)) do
    pueue add "python3 main.py --input-training-data-path ../Dataset1223 cook --debug --background -e 6 -b 400 -v 20000 -t Cook$6 -a $1 -y $2 -w $3 $5 --name $1.$2.$3.$i"
done

# ./run.sh avg 7 20 5
# ./run.sh gru 7 20 5
# ./run.sh vo 7 20 5
# ./run.sh igru 7 20 5 '--id-keep 0.25'
# ./run.sh igru 7 20 5 '--id-keep 0.5'
# ./run.sh igru 7 20 5 '--id-keep 0.75'
# ./run.sh igru 7 20 5 '--id-keep 1'
# ./run.sh agru 7 20 5 '--id-keep 0.25'
# ./run.sh agru 7 20 5 '--id-keep 0.5'
# ./run.sh agru 7 20 5 '--id-keep 0.75'
# ./run.sh agru 7 20 5 '--id-keep 1'
# ./run.sh ingru 7 20 5 '--id-keep 0.25'
# ./run.sh ingru 7 20 5 '--id-keep 0.5'
# ./run.sh ingru 7 20 5 '--id-keep 0.75'
# ./run.sh ingru 7 20 5 '--id-keep 1'

# ./run.sh avg 7 20 5
# ./run.sh gru 7 20 5
# ./run.sh igru 7 20 5
# ./run.sh ingru 7 20 5
# ./run.sh vo 7 20 5
#
# ./run.sh avg 7 20 5 --use-vert
# ./run.sh gru 7 20 5 --use-vert
# ./run.sh igru 7 20 5 --use-vert
# ./run.sh ingru 7 20 5 --use-vert
# ./run.sh vo 7 20 5 --use-vert
#
# ./run.sh avg 30 10 5
# ./run.sh gru 30 10 5
# ./run.sh igru 30 10 5
# ./run.sh ingru 30 10 5
# ./run.sh vo 30 10 5
#
# ./run.sh avg 30 10 5 --use-vert
# ./run.sh gru 30 10 5 --use-vert
# ./run.sh igru 30 10 5 --use-vert
# ./run.sh ingru 30 10 5 --use-vert
# ./run.sh vo 30 10 5 --use-vert
