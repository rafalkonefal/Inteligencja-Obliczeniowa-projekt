import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from PIL import Image
from geneticalgorithm import geneticalgorithm as ga
from cost_fcn import fun1, inverse_dct
import settings as s

print(os.listdir("inputs"))

s.init()



def showImage(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.show()
    
def showImage2(img):
    out = Image.fromarray(img, 'L')
    #out.save('outputlena.png')
    out.show()


showImage(s.img)
showImage2(s.img)

f=open('dct.txt','w')
for part in s.dct:
    for row in part:
        for element in row:
            f.write(str(element)+',')
        f.write(';')
    f.write('\n')
f.close()




algorithm_param = {'max_num_iteration': 100,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
varbound=np.array([[1,255]]*s.N)
model=ga(function=fun1,dimension=s.N,variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param)

model.run()
report = model.report
min_iter = report.index(min(report))
print(f"Rozwiązanie znalezione w {min_iter} iteracji")
opt_solution = model.best_variable
cost_val = fun1(opt_solution)
opt_solution = opt_solution.astype(int)

print(f"Rozwiązanie:\n {opt_solution}\n")

output_img = inverse_dct(s.dct, opt_solution)


showImage(output_img)
showImage2(output_img)

# cv2.imshow('window', output_img)
# cv2.waitKey()
# cv2.destroyAllWindows()