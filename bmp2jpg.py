#!/usr/bin/env python
from glob import glob
import cv2
import os

pngs = glob('images/*.bmp')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)
