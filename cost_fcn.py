import numpy as np
import cv2
import settings as s

def inverse_dct(dct, params):
    width = len(s.img[0])  # one row of image
    params = params.tolist()
    # inverse
    invList = []
    for ipart in s.dct:
        ipart[0][0] = int(params.pop(0))
        ipart[0][1] = int(params.pop(0))
        ipart[0][2] = int(params.pop(0))
        ipart[1][0] = int(params.pop(0))
        ipart[1][1] = int(params.pop(0))
        ipart[2][0] = int(params.pop(0))
        curriDCT = cv2.idct(ipart)
        invList.append(curriDCT)
    row = 0
    rowNcol = []
    for j in range(int(width / s.block), len(invList) + 1, int(width / s.block)):
        rowNcol.append(np.hstack((invList[row:j])))
        row = j
    res = np.vstack((rowNcol))
    return res

def fun1(params):
    res = inverse_dct(s.dct, params)
    return sum(sum(abs(res-s.img)))




