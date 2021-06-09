import numpy as np
import math


def decompose_img(img, img_size, tile_size = 24, shift = 8):   #img to obraz albo dct
    n_tiles = math.ceil( (img_size-tile_size) / shift) +1   #w jednym wymiarze   
    output_tiles = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            output_tiles.append( img[ i*shift : i*shift+tile_size, j*shift : j*shift+tile_size] )
    return output_tiles #zwraca listę arrayów (kafelków) zliczanych rzędami


def compose_img(decomposed_img, img_size, tile_size = 24, shift = 8):
    n_tiles = math.ceil( (img_size-tile_size) / shift) +1   #w jednym wymiarze
    output_list = [[ [] for _ in range(img_size)] for _ in range(img_size)]
    for t,tile in enumerate(decomposed_img):
        ytile = int(t/n_tiles)
        xtile = t%n_tiles
        for i in range(tile_size): # y
            for j in range(tile_size): # x
                try:
                    output_list[i+shift*ytile][j+shift*xtile].append(tile[i,j])
                except IndexError:
                    pass
    output = np.ndarray((img_size,img_size))
    for i in range(img_size):
        for j in range(img_size):
            output[i,j] = np.mean(output_list[i][j])
    return output