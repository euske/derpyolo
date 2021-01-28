#!/bin/sh
exec >>train.log
exec 2>&1
exec </dev/null

echo
date +'*** START %Y-%m-%d %H:%M:%S ***'
renice -n +20 -p $$
exec python train.py -n10 -o./yolo_net.pt ./COCO/train2017.zip ./COCO/annotations/instances_train2017.json
