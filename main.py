import numpy as np
import cv2
from matplotlib import pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from cost_fcn import MSE, dif, inverse_dct, fun1, fun2, fun3
import settings as s
import os
from ypstruct import structure
import ga
print(os.listdir("inputs"))

s.init()

def showImage(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

showImage(s.img)
# print(s.img)
# print(inverse_dct(s.dct, np.asarray(s.params)))
showImage(inverse_dct(s.dct, np.asarray(s.params)))

# Problem Definition
problem = structure()
problem.costfunc = fun2
problem.nvar = s.N
problem.varmin = [-255]*s.N
problem.varmax = [255]*s.N

# GA Parameters
params = structure()
params.maxit = 400
params.npop = 50
params.beta = 1
params.fraction = 0.8
params.mu = 0.6
params.sigma = 15
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