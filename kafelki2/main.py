import numpy as np
import cv2
from matplotlib import pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from cost_fcn import MSE, dif, inverse_dct, fun1
import settings as s
import os
from ypstruct import structure
import ga
print(os.listdir("inputs"))

def showImage(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

def proceed():
    showImage(s.img)
    print(s.img)
    print(inverse_dct(s.dct, np.asarray(s.params)))
    showImage(inverse_dct(s.dct, np.asarray(s.params)))

    # Problem Definition
    problem = structure()
    problem.costfunc = MSE
    problem.nvar = s.N
    problem.varmin = [-255]*s.N
    problem.varmax = [255]*s.N

    # GA Parameters
    params = structure()
    params.maxit = 500
    params.npop = 40
    params.beta = 1
    params.pc = 1
    params.fraction = 0.8
    params.mu = 0.2
    params.sigma = 10
    params.init_individual = s.params

    # Run GA
    out = ga.run(problem, params)

    # Results
    plt.plot(out.bestcost)
    # plt.semilogy(out.bestcost)
    plt.xlim(0, params.maxit)
    plt.xlabel('Iterations')
    plt.ylabel('Best Cost')
    plt.title('Genetic Algorithm (GA)')
    plt.grid(True)
    plt.show()
    print(out.bestsol.position)
    showImage(inverse_dct(s.dct, out.bestsol.position))

# ==========================================

proceed_with_tiles = False

if proceed_with_tiles:
    tiles = s.preinit()
    for t, tile in enumerate(tiles):
       s.init_tile(tile)
       proceed()
else:
    s.init()
    proceed()