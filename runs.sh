#!/bin/sh

set -e

py='exec caffeinate -i time python3.10 train.py'

verify() {
	awk -v dim=$1 '/^n_parameters=/ {split($0, res, "="); if (res[2] != dim) exit 1}'
}

P='python3.10 train.py'
D='-epochs 1 -patience 1 -lr 1 -dry' # dummy arguments
# It takes some time to verify all of them. Mostly because tarfile is very slow.
# $P -data mnist -model fc               $D | verify 199210 >/dev/null 2&>1
# $P -data mnist -model lenet            $D | verify 44426  >/dev/null 2&>1
# $P -data cifar -model fc               $D | verify 656810 >/dev/null 2&>1
# $P -data cifar -model lenet            $D | verify 62006  >/dev/null 2&>1
# $P -data mnist -model fc    -intr 750  $D | verify 750    >/dev/null 2&>1
# $P -data mnist -model lenet -intr 300  $D | verify 300    >/dev/null 2&>1 # This should have been 290.
# $P -data cifar -model fc    -intr 9000 $D | verify 9000   >/dev/null 2&>1
# $P -data cifar -model lenet -intr 2900 $D | verify 2900   >/dev/null 2&>1
# $P -data mnist -model fc    -small     $D | verify 2395   >/dev/null 2&>1
# $P -data mnist -model lenet -small     $D | verify 222    >/dev/null 2&>1
# $P -data cifar -model fc    -small     $D | verify 46495  >/dev/null 2&>1
# $P -data cifar -model lenet -small     $D | verify 2186   >/dev/null 2&>1

# This group should reach 90% accuracy -> 81%
# $py -data mnist -model fc -epochs 5 -patience 2 -lr 1
# $py -data mnist -model fc -intr 750 -epochs 1000 -patience 1000 -lr 0.04 # SLOW!
# $py -data mnist -model lenet -epochs 5 -patience 2 -lr 0.01
# $py -data mnist -model lenet -intr 300 -epochs 500 -patience 30 -lr 0.004
# $py -data mnist -model fc -small -epochs 250 -patience 20 -lr 0.5
# $py -data mnist -model lenet -small -epochs 10 -patience 2 -lr 0.004
# $py -data mnist -model fc -proj sparse -intr 750 -epochs 2 -patience 1 -lr 0.0004
# $py -data mnist -model lenet -proj sparse -intr 300 -epochs 5 -patience 3 -lr 0.00004

# This group should only go above 50% accuracy. -> 45%
# $py -data cifar -model fc -epochs 600 -patience 30 -lr 1 # 0.02
# $py -data cifar -model fc -intr 9000 -epochs 1000 -patience 1000 -lr 0.04 # TODO: TOO SLOW!
# $py -data cifar -model lenet -epochs 15 -patience 10 -lr 0.01
# $py -data cifar -model lenet -intr 2900 -epochs 2000 -patience 1000 -lr 0.0002 # SLOW!
# $py -data cifar -model fc -small -epochs 500 -patience 50 -lr 0.05
# $py -data cifar -model lenet -small -epochs 200 -patience 50 -lr 0.005
# $py -data cifar -model fc -proj sparse -intr 9000 -epochs 1000 -patience 1000 -lr 0.04 # TODO: TOO SLOW!
# $py -data cifar -model lenet -proj sparse -intr 2900 -epochs 10 -patience 3 -lr 0.00002
