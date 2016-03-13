#!/bin/bash
for i in 1 2 3 
do
    #echo $i
    java -cp ABAGAIL.jar opt.test.myFlipFlop 3 1000000
done
