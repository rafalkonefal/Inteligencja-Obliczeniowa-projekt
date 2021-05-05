import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'D:\Z\IO\proj\Inteligencja-Obliczeniowa-projekt\inputs\lena64.png'
img =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
cv2.imshow('test',img)
cv2.waitKey()
cv2.destroyAllWindows()