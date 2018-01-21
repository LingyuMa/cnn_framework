#!/usr/bin/env bash
mkdir images
mkdir labels
ROOT_DIR=$PWD
cd stage1_train
i=0
for entry in *; do
    cp $entry/images/$entry'.png' ../images/$i.png
    printf '%s\n' $entry/images/$entry'.png'
    cp $entry/images/'label_'$entry'.png' ../labels/label_$i.png
    printf '%s\n' $i.png $ROOT_DIR/labels/label_$i.png | paste -sd ',' >> ../label.csv
    i=$((i + 1))
done