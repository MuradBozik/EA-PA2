from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
import numpy as np
import sys
import heapq

budget = 100

np.random.seed(123)

# hyperparameters
npop = 50 # population size
beta = np.pi/36 # 5 degrees


def ES_4(problem):
    """
    This function uses:
        - Correlated Mutation
        - Intermediate Recombination
        - (mu + lambda) selection
    """
    n = problem.number_of_variables
    n_q = int(n * (n - 1) / 2)

    tao_prime = 1 / np.sqrt(2 * n)  # global learning rate
    tao_0 = 1 / np.sqrt(2 * np.sqrt(n))  # local learning rate

    lamda = npop * 7  # offspring population size

    fopt = -sys.maxsize - 1

    # Initial Population samples uniformly distributed over the interval (boundaries)
    P = np.random.uniform(low=problem.lowerbound[0], high=problem.upperbound[0], size=(npop, n))

    alphas = np.random.uniform(low=-np.pi, high=np.pi, size=(lamda, n_q))  # rotation angles initialized

    OOB_correction = lambda x: x - 2 * np.pi * (x / np.abs(x)) if (np.abs(x) > np.pi).any() else x  # out of boundary correction function

    sigmas = [[(problem.upperbound[0] - problem.lowerbound[0]) / 6] * n] * lamda  # Individual sigma initialization with feasible range

    # Fitness evaluation of the population
    fitness = -1 * np.apply_along_axis(problem, 1, P)

    if np.max(fitness) >= fopt:
        x_prime = P[np.argmax(fitness)]
        fopt = np.max(fitness)

    ## !! final_target_hit returns True if the optimum has been found.
    ## !! evaluations returns the number of function evaluations has been done on the problem.
    while not problem.final_target_hit and problem.evaluations < budget * n:
        OP = []  # offspring population
        n_q = int(n * (n - 1) / 2)
        ### Intermediate Recombination
        for i in range(lamda):
            parent1 = P[np.random.choice(npop)]
            parent2 = P[np.random.choice(npop)]
            offspring = np.mean((parent1, parent2), axis=0)
            OP.append(offspring)

        OP = np.array(OP)  # Offspring population

        N_tao_prime = np.random.normal(0, tao_prime)

        ### Mutation
        # Step size update
        sigmas_prime = [[sigma_i * np.exp(N_tao_prime + np.random.normal(0, tao_0)) for sigma_i in sigma] for sigma in sigmas]

        # Rotation angles update
        alphas_prime = alphas + np.random.normal(0, beta, size=(lamda, n_q))
        for alpha_prime in alphas_prime:
            alpha_prime = np.apply_along_axis(OOB_correction, 0, alpha_prime)

        # Correlation of steps sizes
        OP_prime = []
        for j, x, sigma_prime in zip(range(lamda), OP, sigmas_prime):
            s = np.zeros(n)
            n_q = int(n * (n - 1) / 2)
            for n_i in range(n):
                s[n_i] = sigma_prime[n_i] * np.random.normal(0, 1)  # uncorrelated mutation vector initialization

            for k in range(n - 1):
                n_i = n - 1 - k
                n_ii = n - 1
                for i in range(k):
                    d1, d2 = s[n_i], s[n_ii]
                    s[n_ii] = d1 * np.sin(alphas_prime[j, n_q - 1]) + d2 * np.cos(alphas_prime[j, n_q - 1])
                    s[n_i] = d1 * np.cos(alphas_prime[j, n_q - 1]) - d2 * np.sin(alphas_prime[j, n_q - 1])
                    n_ii = n_ii - 1
                    n_q = n_q - 1

            x_prime = x + s
            OP_prime.append(x_prime)
        OP_prime = np.array(OP_prime)

        ### Evaluation of (OP_prime + P)
        total_pop = np.append(P, OP_prime, axis=0)
        fitness = -1 * np.apply_along_axis(problem, 1, total_pop)
        if np.max(fitness) >= fopt:
            x_prime = total_pop[np.argmax(fitness)]
            fopt = np.max(fitness)

        ### Selection
        top_fits = heapq.nlargest(npop, fitness)
        sorter = np.argsort(fitness)
        indices = sorter[np.searchsorted(fitness, top_fits, sorter=sorter)]
        P = total_pop[indices]

    return x_prime, fopt

def ES_5(problem):
    """
    This function uses:
        - Global Sigma Mutation
        - Intermediate Recombination
        - (mu, lambda) selection
    """
    n = problem.number_of_variables

    tao_0 = 1 / np.sqrt(n)  # learning rate (tao 0)
    lamda = npop * 7  # offspring population size

    fopt = -sys.maxsize - 1

    # Initial Population samples uniformly distributed over the interval (boundaries)
    P = np.random.uniform(low=problem.lowerbound[0], high=problem.upperbound[0], size=(npop, n))

    sigmas = [(problem.upperbound[0] - problem.lowerbound[0]) / 6] * lamda  # Global Sigma initialization with feasible range

    # Fitness evaluation of the population
    fitness = -1 * np.apply_along_axis(problem, 1, P)

    if np.max(fitness) >= fopt:
        x_prime = P[np.argmax(fitness)]
        fopt = np.max(fitness)

    ## !! final_target_hit returns True if the optimum has been found.
    ## !! evaluations returns the number of function evaluations has been done on the problem.
    while not problem.final_target_hit and problem.evaluations < budget * n:
        OP = []  # offspring population

        ### Intermediate Recombination
        for i in range(lamda):
            parent1 = P[np.random.choice(npop)]
            parent2 = P[np.random.choice(npop)]
            offspring = np.mean((parent1, parent2), axis=0)
            OP.append(offspring)

        OP = np.array(OP)  # Offspring population
        OP_prime = []

        ### Mutation
        for j, x in enumerate(OP):
            sigmas[j] = sigmas[j] * np.exp(np.random.normal(0, tao_0))
            x_prime = x + np.random.normal(0, sigmas[j], size=n)
            OP_prime.append(x_prime)

        OP_prime = np.array(OP_prime)

        ### Evaluation of OP_prime
        fitness = -1 * np.apply_along_axis(problem, 1, OP_prime)
        if np.max(fitness) >= fopt:
            x_prime = OP_prime[np.argmax(fitness)]
            fopt = np.max(fitness)

        ### Selection
        top_fits = heapq.nlargest(npop, fitness)
        sorter = np.argsort(fitness)
        indices = sorter[np.searchsorted(fitness, top_fits, sorter=sorter)]
        P = OP_prime[indices]

    return x_prime, fopt

def ES_6(problem):
    """
    This function uses:
        - Individual Sigma Mutation
        - Intermediate Recombination
        - (mu, lambda) selection
    """
    n = problem.number_of_variables

    tao_prime = 1 / np.sqrt(2 * n)  # global learning rate
    tao_0 = 1 / np.sqrt(2 * np.sqrt(n))  # local learning rate

    lamda = npop * 7  # offspring population size

    fopt = -sys.maxsize - 1

    # Initial Population samples uniformly distributed over the interval (boundaries)
    P = np.random.uniform(low=problem.lowerbound[0], high=problem.upperbound[0], size=(npop, n))

    sigmas = [[(problem.upperbound[0] - problem.lowerbound[0]) / 6] * n] * lamda  # Individual sigma initialization with feasible range

    # Fitness evaluation of the population
    fitness = -1 * np.apply_along_axis(problem, 1, P)

    if np.max(fitness) >= fopt:
        x_prime = P[np.argmax(fitness)]
        fopt = np.max(fitness)

    ## !! final_target_hit returns True if the optimum has been found.
    ## !! evaluations returns the number of function evaluations has been done on the problem.
    while not problem.final_target_hit and problem.evaluations < budget * n:
        OP = []  # offspring population

        ### Intermediate Recombination
        for i in range(lamda):
            parent1 = P[np.random.choice(npop)]
            parent2 = P[np.random.choice(npop)]
            offspring = np.mean((parent1, parent2), axis=0)
            OP.append(offspring)

        OP = np.array(OP)  # Offspring population
        OP_prime = []

        N_tao_prime = np.random.normal(0, tao_prime)

        ### Mutation
        for x, sigma in zip(OP, sigmas):
            for x_i, sigma_i in zip(x, sigma):
                sigma_i = sigma_i * np.exp(N_tao_prime + np.random.normal(0, tao_0))
                x_i = x_i + np.random.normal(0, sigma_i)
            OP_prime.append(x)

        OP_prime = np.array(OP_prime)

        ### Evaluation of OP_prime
        fitness = -1 * np.apply_along_axis(problem, 1, OP_prime)
        if np.max(fitness) >= fopt:
            x_prime = OP_prime[np.argmax(fitness)]
            fopt = np.max(fitness)

        ### Selection
        top_fits = heapq.nlargest(npop, fitness)
        sorter = np.argsort(fitness)
        indices = sorter[np.searchsorted(fitness, top_fits, sorter=sorter)]
        P = OP_prime[indices]

    return x_prime, fopt

if __name__ == '__main__':

    function_list = [ES_4, ES_5, ES_6]
    function_names = ["ES_4", "ES_5", "ES_6"]

    for function, name in zip(function_list, function_names):
        ## Declarations of Ids, instances, and dimensions that the problems to be tested.
        problem_id = range(1, 25)
        instance_id = range(1, 26)
        dimension = [2, 5, 20]

        ## Declariation of IOHprofiler_csv_logger.
        ## 'result' is the name of output folder.
        ## 'studentname1_studentname2' represents algorithm name and algorithm info, which will be caption of the algorithm in IOHanalyzer.
        logger = IOH_logger("./", "FinalResult-"+name, name, name)

        for p_id in problem_id :
            for d in dimension :
                for i_id in instance_id:
                    ## Getting the problem with corresponding id,dimension, and instance.
                    f = IOH_function(p_id, d, i_id, suite="BBOB")
                    f.add_logger(logger)
                    xopt, fopt = function(f)
            print("Problem ", p_id, " completed!")
        logger.clear_logger()
