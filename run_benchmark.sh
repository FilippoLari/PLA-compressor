#!/bin/bash

echo "dataset,epsilon,compressor,space,bps,predict"

PROGRAM="./build/benchmark"

$PROGRAM "datasets/DNA_1" 64 "32"
$PROGRAM "datasets/DNA_1" 128 "32"
$PROGRAM "datasets/DNA_1" 256 "32"

$PROGRAM "datasets/5GRAM_1" 64 "32"
$PROGRAM "datasets/5GRAM_1" 128 "32"
$PROGRAM "datasets/5GRAM_1" 256 "32"

$PROGRAM "datasets/URL_1" 64 "32"
$PROGRAM "datasets/URL_1" 128 "32"
$PROGRAM "datasets/URL_1" 256 "32"

$PROGRAM "datasets/books_200M_uint32" 64 "32"
$PROGRAM "datasets/books_200M_uint32" 128 "32"
$PROGRAM "datasets/books_200M_uint32" 256 "32"

$PROGRAM "datasets/wiki_ts_200M_uint64" 64 "64"
$PROGRAM "datasets/wiki_ts_200M_uint64" 128 "64"
$PROGRAM "datasets/wiki_ts_200M_uint64" 256 "64"


