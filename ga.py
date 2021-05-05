import numpy as np
from ypstruct import structure

def run(problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    gamma = params.fraction
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    first_individual = structure()
    first_individual.position = np.array(params.init_individual)

    first_individual.cost = costfunc(first_individual.position)
    print (first_individual.cost)
    pop = empty_individual.repeat(npop)
    pop[0] = first_individual
    for i in range(1,npop):
        if i%2:
            pop[i].position = mutate(first_individual, 0.1, sigma).position
        else:
            pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    # Main Loop
    for it in range(maxit):

        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)

        popc = []
        for _ in range(5):

            # Select Parents
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]

            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            
            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Perform Mutation
            c3 = mutate(p1, mu, sigma)
            c4 = mutate(p2, mu, sigma)

            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)
            apply_bound(c3, varmin, varmax)
            apply_bound(c4, varmin, varmax)
            # Evaluate First Offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()
            c3.cost = costfunc(c3.position)
            if c3.cost < bestsol.cost:
                bestsol = c3.deepcopy()
            c4.cost = costfunc(c4.position)
            if c4.cost < bestsol.cost:
                bestsol = c4.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)
            popc.append(c3)
            popc.append(c4)
        

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

def crossover(p1, p2, fraction=0.8):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    n = len(c1.position)
    indexes = [np.random.randint(0,n-1) for i in range(round(fraction*n))]
    for idx in range(n):
        c1.position[idx] = p1.position[idx] if idx in indexes else p2.position[idx]
        c2.position[idx] = p2.position[idx] if idx in indexes else p1.position[idx]
    return c1,c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += round(np.random.normal(0,sigma))
    return y

def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
