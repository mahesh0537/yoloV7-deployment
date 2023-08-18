#!/bin/bash

python3 detect.py --weights best.pt --conf 0.5 --img-size 640 --source test/images --no-trace --device cpu