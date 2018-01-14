#!/usr/bin/env bash
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xzf 101_ObjectCategories.tar.gz
rm 101_ObjectCategories.tar.gz
mkdir images
cd 101_ObjectCategories
i=0
for entry in *; do
    for image in $entry/*; do
        cp $image ../images/$i.jpg
	printf '%s\n' $i.jpg $entry | paste -sd ',' >> ../label.csv
        i=$((i + 1))
    done
done
rm -r ../101_ObjectCategories
