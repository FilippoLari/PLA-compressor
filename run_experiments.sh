#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [benchmark|entropy]"
    exit 1
fi

RUN="$1"

if [[ "$RUN" == "benchmark" ]]; then
    echo "dataset,mode,epsilon,compressor,space,bps,predict"
elif [[ "$RUN" == "entropy" ]]; then
    echo "dataset,mode,epsilon,sign,exponent,mantissa,opt_sign,opt_exponent,opt_mantissae"
else
    echo "Invalid run: $RUN (expected 'benchmark' or 'entropy')"
    exit 1
fi

PROGRAM="./build/benchmark"

$PROGRAM "datasets/DNA_1" 64 "compression" "$RUN"
$PROGRAM "datasets/DNA_1" 128 "compression" "$RUN"
$PROGRAM "datasets/DNA_1" 256 "compression" "$RUN"

$PROGRAM "datasets/5GRAM_1" 64 "compression" "$RUN"
$PROGRAM "datasets/5GRAM_1" 128 "compression" "$RUN"
$PROGRAM "datasets/5GRAM_1" 256 "compression" "$RUN"

$PROGRAM "datasets/URL_1" 64 "compression" "$RUN"
$PROGRAM "datasets/URL_1" 128 "compression" "$RUN"
$PROGRAM "datasets/URL_1" 256 "compression" "$RUN"

$PROGRAM "datasets/wiki_ts_200M_uint64" 64 "indexing" "$RUN"
$PROGRAM "datasets/wiki_ts_200M_uint64" 128 "indexing" "$RUN"
$PROGRAM "datasets/wiki_ts_200M_uint64" 256 "indexing" "$RUN"

$PROGRAM "datasets/books_800M_uint64" 64 "indexing" "$RUN"
$PROGRAM "datasets/books_800M_uint64" 128 "indexing" "$RUN"
$PROGRAM "datasets/books_800M_uint64" 256 "indexing" "$RUN"

$PROGRAM "datasets/osm_cellids_800M_uint64" 64 "indexing" "$RUN"
$PROGRAM "datasets/osm_cellids_800M_uint64" 128 "indexing" "$RUN"
$PROGRAM "datasets/osm_cellids_800M_uint64" 256 "indexing" "$RUN"