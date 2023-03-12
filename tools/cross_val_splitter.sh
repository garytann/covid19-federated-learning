#! /bin/sh

python3 cross_val_splitter.py -d ./data/splitted/dataset_clientA -o ./data/cross_validation_A -v ./data/convex
python3 cross_val_splitter.py -d ./data/splitted/dataset_clientB -o ./data/cross_validation_B -v ./data/convex
python3 cross_val_splitter.py -d ./data/splitted/dataset_clientC -o ./data/cross_validation_C -v ./data/convex