#!/bin/bash

echo "dataset,mode,epsilon,compressor,space,bps,predict"

PROGRAM="./build/benchmark"

$PROGRAM "datasets/DNA_1" 64 "compression"
$PROGRAM "datasets/DNA_1" 128 "compression"
$PROGRAM "datasets/DNA_1" 256 "compression"

$PROGRAM "datasets/5GRAM_1" 64 "compression"
$PROGRAM "datasets/5GRAM_1" 128 "compression"
$PROGRAM "datasets/5GRAM_1" 256 "compression"

$PROGRAM "datasets/URL_1" 64 "compression"
$PROGRAM "datasets/URL_1" 128 "compression"
$PROGRAM "datasets/URL_1" 256 "compression"

$PROGRAM "datasets/wiki_ts_200M_uint64" 32 "indexing"
$PROGRAM "datasets/wiki_ts_200M_uint64" 64 "indexing"
$PROGRAM "datasets/wiki_ts_200M_uint64" 128 "indexing"

$PROGRAM "datasets/books_800M_uint64" 32 "indexing"
$PROGRAM "datasets/books_800M_uint64" 64 "indexing"
$PROGRAM "datasets/books_800M_uint64" 128 "indexing"

$PROGRAM "datasets/osm_cellids_800M_uint64" 32 "indexing"
$PROGRAM "datasets/osm_cellids_800M_uint64" 64 "indexing"
$PROGRAM "datasets/osm_cellids_800M_uint64" 128 "indexing"