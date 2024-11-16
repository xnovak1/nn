#!/bin/bash
## change this file to your needs

echo "Adding some modules"

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -O3 main.cpp nn.cpp matrix.cpp -o network

## tests
## g++ -Wall -O3 tests.cpp matrix.cpp -o tests

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./network
