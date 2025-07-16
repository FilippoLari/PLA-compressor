#!/bin/bash

echo "dataset,epsilon,compressor,space,bps,predict"

PROGRAM="./build/benchmark"

$PROGRAM "datasets/DNA_1" 64
$PROGRAM "datasets/DNA_1" 128
$PROGRAM "datasets/DNA_1" 256

$PROGRAM "datasets/5GRAM_1" 64
$PROGRAM "datasets/5GRAM_1" 128
$PROGRAM "datasets/5GRAM_1" 256

$PROGRAM "datasets/URL_1" 64
$PROGRAM "datasets/URL_1" 128
$PROGRAM "datasets/URL_1" 256

