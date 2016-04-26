#!/usr/bin/env bash

mv data.csv data.csv.bak

# 0.2.9
echo "==== previous version ===="
source /opt/jubatus/profile
for hashnum in 64 128 256 512 1024 2048
do
    cat nn.json | jq ".parameter.parameter.hash_num=${hashnum}" > nn_bench.json
    ./classifier --config nn_bench.json iris.csv dorothea_train.csv 20news.csv
done

# pr-260
echo "==== pr-260 ===="
source ../setenv_configure.sh
for hashnum in 64 128 256 512 1024 2048
do
    for threads in 1 2 4 8 12
    do
	cat nn_template.json | jq ".parameter.parameter.hash_num=${hashnum}" > nn_bench_tmp.json
	cat nn_bench_tmp.json | jq ".parameter.parameter.threads=${threads}" > nn_bench.json
	./classifier --config nn_bench.json iris.csv dorothea_train.csv 20news.csv
    done
done

rm nn_bench_tmp.json
