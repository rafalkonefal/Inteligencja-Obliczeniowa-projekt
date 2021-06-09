import cv2
import numpy as np
from kafelki import decompose_img
import json
import cost_fcn as c
import os

import show_fcn as show

# Quantization Arrays

def selectQMatrix(qName):
    Q10 = np.array([[80, 60, 50, 80, 120, 200, 255, 255],
                     [55, 60, 70, 95, 130, 255, 255, 255],
                     [70, 65, 80, 120, 200, 255, 255, 255],
                     [70, 85, 110, 145, 255, 255, 255, 255],
                     [90, 110, 185, 255, 255, 255, 255, 255],
                     [120, 175, 255, 255, 255, 255, 255, 255],
                     [245, 255, 255, 255, 255, 255, 255, 255],
                     [255, 255, 255, 255, 255, 255, 255, 255]])

    Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])

    Q90 = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
                       [2, 2, 3, 4, 5, 12, 12, 11],
                       [3, 3, 3, 5, 8, 11, 14, 11],
                       [3, 3, 4, 6, 10, 17, 16, 12],
                       [4, 4, 7, 11, 14, 22, 21, 15],
                       [5, 7, 11, 13, 16, 12, 23, 18],
                       [10, 13, 16, 17, 21, 24, 24, 21],
                       [14, 18, 19, 20, 22, 20, 20, 20]])
    if qName == "Q10":
        return Q10
    elif qName == "Q50":
        return Q50
    elif qName == "Q90":
        return Q90
    else:    # it suppose to return original image back
        return np.ones((block, block))


def read_params():
    with open('config.json','r') as file:
        config = json.load(file)
    
    global img_name
    global block
    global proceed_with_tiles
    global tile_size
    global shift
    global cost_fcn
    global ga_maxit
    global ga_npop
    global ga_beta
    global ga_mu
    global ga_sigma
    global ga_fraction
    img_name = config['img_name']
    if not img_name in os.listdir("inputs"):
        raise ValueError("File not exists")
    block = config['dct_block_size']
    ga_maxit = config['ga_maxit']
    ga_npop = config['ga_npop']
    ga_beta = config['ga_beta']
    ga_mu = config['ga_mu']
    ga_sigma = config['ga_sigma']
    ga_fraction = config['ga_fraction']
    proceed_with_tiles = config['proceed_with_tiles']
    tile_size = config['tile_size']
    shift = config['tile_shift']
    cost_fcn_name = config["cost_fcn"]
    if cost_fcn_name in ["MSE", "dif", "fun1"]:
        cost_fcn = {"MSE": c.MSE, "dif": c.dif, "fun1": c.fun1}[cost_fcn_name]
    else:
        raise TypeError("read _comment aobout cost_fcn in config.json")
  
                    
def preinit_tile():
    img = cv2.imread('inputs/' + img_name, 0)
    global img_size
    img_size = len(img)
    tiles = decompose_img(img, img_size, tile_size, shift)
    return tiles


def init(tile = None):
    global dct
    global img
    global N
    global params
    global mask
    if tile is None:
        img = cv2.imread('inputs/' + img_name, 0)
    else:
        img = tile
    
    height = len(img)  # one column of image
    width = len(img[0])  # one row of image
    sliced = []  # new list for 8x8 sliced image
    print("The image heigh is " + str(height) + ", and image width is " + str(width) + " pixels")
    # dividing 8x8 parts
    currY = 0  # current Y index
    for i in range(block, height + 1, block):
        currX = 0  # current X index
        for j in range(block, width + 1, block):
            sliced.append(img[currY:i, currX:j] - np.ones((block, block)) * 128)  # Extracting 128 from all pixels
            currX = j
        currY = i

    print("Size of the sliced image: " + str(len(sliced)))
    print("Each elemend of sliced list contains a " + str(sliced[0].shape) + " element.")

    # float32
    imf = [np.float32(img) for img in sliced]

    # DCT
    dct = []
    for part in imf:
        currDCT = cv2.dct(part)
        dct.append(currDCT)

    # kwantyzacja
    selectedQMatrix = selectQMatrix("Q90")
    for ndct in dct:
        for i in range(block):
            for j in range(block):
                ndct[i, j] = np.around(ndct[i, j] / selectedQMatrix[i, j])
    print("kwant")
    print(dct[0])

    # 6 params
    params = []
    for part in dct:
        params.append(np.int(part[0][0]))
        params.append(np.int(part[0][1]))
        params.append(np.int(part[0][2]))
        params.append(np.int(part[1][0]))
        params.append(np.int(part[1][1]))
        params.append(np.int(part[2][0]))
    print(len(params), params)

    N=len(params)

    mask = np.zeros((height,height),bool)
    for x in range(height):
        for y in range(height):
            if (x%block == 0 or x%block == 7 or y%block == 0 or y%block == 7):
                mask[x,y]=True

            