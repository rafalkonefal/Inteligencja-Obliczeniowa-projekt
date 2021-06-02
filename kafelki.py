import numpy as np
import math

def decompose_img(img, img_size = 256, tile_size = 16, shift = 8):   #img to obraz albo dct
    n_tiles = math.ceil( (img_size-tile_size) / shift) +1   #w jednym wymiarze
    
    output_img = [np.zeros((tile_size,tile_size))] * (n_tiles**2)
    #output_dct = output_img.copy()
    
    
    for i in range(n_tiles):
        for j in range(n_tiles):
            cut = img[ i*shift : i*shift+tile_size, j*shift : j*shift+tile_size]
            #if koniec wiersza, kolumny
            #output_img[i*n_tiles+j] = np.pad(cut, (0,),(0,))
            
    return output_img #zwraca listę arrayów (kafelków) zliczanych rzędami

img_size = 21
xd = np.ndarray((img_size,img_size))
for i in range(img_size):
        for j in range(img_size):
            xd[i,j] = i + j + 1

xdd = decompose_img(xd,img_size,10,5)

def compose_img(decomposed_img, img_size = 256, tile_size = 16, shift = 8):
    
    n_tiles = math.ceil( (img_size-tile_size) / shift) +1   #w jednym wymiarze
    
    #srednia? mediana? rozne opcje?
    img = []
    return img