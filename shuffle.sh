#!/usr/bin/env bash

awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' ../data/${1}.tsv | sort -n | cut -c8- > ../data/${1}Shuffled.tsv
