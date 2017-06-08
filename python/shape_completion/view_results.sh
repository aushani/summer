BIN=/home/aushani/dascar/build/bin/dascar-sc-viewer

OBJ=$1
IT=$2

$BIN --df results/shape_$OBJ\_$IT\.df --sdf results/shape_$OBJ\_$IT\.sdf --dfgen results/shape_$OBJ\_$IT\.dfgen
