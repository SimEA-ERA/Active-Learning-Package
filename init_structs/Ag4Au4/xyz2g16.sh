#!/bin/bash

 for file_name in *.xyz; do
 cp $file_name ${file_name%.*}.bck
 done
 for file_name in *.bck; do
 sed -i '1,2d' $file_name
 xyz="$(grep '[0-9]\.[0-9][0-9]' $file_name)"

 echo "%mem=180000MB
%nprocshared=32
#p wb97xd/def2TZVP scf(xqc,Conver=6) scfcyc=999

######Geom=Single Point

0 1
$xyz
    " > ${file_name%.*}.com

 done

rm *.bck
