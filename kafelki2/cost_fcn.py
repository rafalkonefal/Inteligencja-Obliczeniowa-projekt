import numpy as np
import cv2
import settings as s
import math

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

def create_right_vector(actual, right):
    return [abs(actual[i,s.block-1]-right[i,0]) for i in range(s.block)]

def create_down_vector(actual, down):
    return [abs(actual[s.block-1,i]-down[0,i]) for i in range(s.block)]

def l1_norm(vec):
    return sum(vec)

def l2_norm(vec):
    return math.sqrt(sum([ x**3 for x in vec ]))

def create_array_of_blocks(img):
    height = len(img)  # one column of image
    width = len(img[0])  # one row of image
    sliced = [[0]*round(width/s.block)]*round(height/s.block)
    # print(len(sliced),len(sliced[0]))
    currY = 0  # current Y index
    for i in range(s.block, height + 1, s.block):
        currX = 0  # current X index
        for j in range(s.block, width + 1, s.block):
            sliced[round(currY/s.block)][round(currX/s.block)] = img[currY:i, currX:j]
            currX = j
        currY = i
    return sliced

def MSE(params):
    res = inverse_dct(s.dct, params)
    err = np.sum((res.astype("float") - s.img.astype("float")) ** 2)
    err /= float(res.shape[0] * res.shape[1])
    return err

def dif(params):
    res = inverse_dct(s.dct, params)
    return sum(sum(abs(res-s.img)))

def fun1(params):
    res = inverse_dct(s.dct, params)
    return sum(sum(abs(s.mask*res-s.mask*s.img)))

def fun2(params):
    result = 0
    res = inverse_dct(s.dct, params)
    sliced = create_array_of_blocks(res)
    height = len(sliced)
    width = len(sliced[0])
    # print("rozmiary",width,height)
    for i in range(height):
        for j in range(width):
            # print(i,j)
            if(j < width - 1):
                result += l2_norm(create_right_vector(sliced[i][j], sliced[i][j+1]))
            if (i < height - 1):
                result += l2_norm(create_down_vector(sliced[i][j], sliced[i+1][j]))
    return result

def fun3(params):
    alpha = 0.1
    betha = 100
    return alpha*dif(params)+betha*fun2(params)







