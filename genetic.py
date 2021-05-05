from geneticalgorithm import geneticalgorithm as ga
import numpy as np
from cost_fcn import fun1
import settings as s

N=384 #to sie potem zminei zeby samo liczyli ile jest params
print(s.N)

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
