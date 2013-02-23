#!/bin/bash

mc="../mcpercol.py"
model="--V1 0.003 --V2 0.001 --common 1"
steps="--N-MC 2000 --N-equi 1000 --N-samp 1000"
# struc="POSCAR.fcc-conv.2x2x2 --supercell 2 2 2"
struc="POSCAR.gLiFeO2 --supercell 2 2 2 -c 0.5"

T_range="1000 1200 1400 1600"
# T_range="1300 1350 1400 1450 1500 1550"
# T_range="1300"

export PYTHONUNBUFFERED=1

for T in $T_range
do

  echo -n "now running T = ${T} ... "
  $mc $struc $model $steps -T $T > out.$T
  mv mcpercol.out mcsteps.$T
  echo "done."
  p=$(awk '/percolation probability:/{print $3}' out.$T)
  echo "${T} ${P}"
  echo "${T} ${P}" >> percol.dat

done

exit 0
