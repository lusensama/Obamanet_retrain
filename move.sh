#!/bin/bash
# mkdir extracted_images_debug
# cd extracted_images
# rm *square*
for i in `seq -f '%05g' 1 98`;
do
    mkdir $i || true
    mv $i-* $i || true
    mv $i images
done

# python un_trim_process.py
