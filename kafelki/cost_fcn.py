import numpy as np
import cv2
import settings as s
from kafelki import decompose_img, compose_img

def inverse_dct(dct, params):
    width = len(s.img[0])  # one row of image
    params = params.tolist()
    # inverse
    invList = []
    for ipart in dct:
        ipart[0][0] = int(params.pop(0))
        ipart[0][1] = int(params.pop(0))
        ipart[0][2] = int(params.pop(0))
        ipart[1][0] = int(params.pop(0))
        ipart[1][1] = int(params.pop(0))
        ipart[2][0] = int(params.pop(0))
        curriDCT = cv2.dct(ipart,flags=cv2.DCT_INVERSE)+128
        invList.append(curriDCT)
    row = 0
    rowNcol = []
    for j in range(int(width / s.block), len(invList) + 1, int(width / s.block)):
        rowNcol.append(np.hstack((invList[row:j])))
        row = j
    res = np.vstack((rowNcol))
    return np.round(res)

def MSE(params):
    res = inverse_dct(s.dct, params)
    err = np.sum((res.astype("float") - s.img.astype("float")) ** 2)
    err /= float(res.shape[0] * res.shape[1])
    return err
def dif(params):
    dct = compose_img(s.dct, s.img_size, s.tile_size, s.shift)
    dct2 = s.dct
    res = inverse_dct(dct, params)
    # print(res)
    # print(s.img)
    return sum(sum(abs(res-s.img)))
def fun1(params):
    res = inverse_dct(s.dct, params)
    return sum(sum(abs(s.mask*res-s.mask*s.img)))






