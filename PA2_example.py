from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
import numpy as np
import sys
import heapq

budget = 100

np.random.seed(123)

# hyperparameters
npop = 50 # population size
beta = np.pi/36 # 5 degrees


def ES_3(problem):
    """
    This function uses:
        - Correlated Mutation
        - Intermediate Recombination
        - (mu, lambda) selection
    """
    n = problem.number_of_variables
    n_q = int(n * (n - 1) / 2)

    tao_prime = 1 / np.sqrt(2 * n)  # global learning rate
    tao_0 = 1 / np.sqrt(2 * np.sqrt(n))  # local learning rate

    lamda = npop * 7  # offspring population size

    fopt = -sys.maxsize - 1

    # Initial Population samples uniformly distributed over the interval (boundaries)
    P = np.random.uniform(low=problem.lowerbound[0], high=problem.upperbound[0], size=(npop, n))

    alphas = np.random.uniform(low=-np.pi, high=np.pi, size=n_q)  # rotation angles initialized

    OOB_correction = lambda x: x - 2 * np.pi * (x / np.abs(x)) if (np.abs(x) > np.pi).any() else x  # out of boundary correction function

    sigmas = [(problem.upperbound[0] - problem.lowerbound[0]) / 6] * npop  # Individual sigma initialization with feasible range

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
        sigmas_prime = [sigma * np.exp(N_tao_prime + np.random.normal(0, tao_0)) for sigma in sigmas]

        # Rotation angles update

        alphas_prime = np.apply_along_axis(OOB_correction, 0, alphas + np.random.normal(0, beta, size=n_q))

        # Correlation of steps sizes
        s = np.zeros(n)
        for n_i in range(n):
            s[n_i] = sigmas_prime[n_i] * np.random.normal(0, 1)  # uncorrelated mutation vector initialization

        for k in range(1, n):
            n_i = n - 1 - k
            n_ii = n - 1
            for i in range(1, k + 1):
                d1, d2 = s[n_i], s[n_ii]
                s[n_ii] = d1 * np.sin(alphas_prime[n_q - 1]) + d2 * np.cos(alphas_prime[n_q - 1])
                s[n_i] = d1 * np.cos(alphas_prime[n_q - 1]) - d2 * np.sin(alphas_prime[n_q - 1])
                n_ii = n_ii - 1
                n_q = n_q - 1

        OP_prime = []
        for x_i, sigma_i in zip(OP, sigmas_prime):
            x_i_prime = x_i + sigma_i * s
            OP_prime.append(x_i_prime)
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

    ## Declarations of Ids, instances, and dimensions that the problems to be tested.
    problem_id = range(1,2)
    instance_id = range(1,3)
    dimension = [5,7]

    ## Declariation of IOHprofiler_csv_logger.
    ## 'result' is the name of output folder.
    ## 'studentname1_studentname2' represents algorithm name and algorithm info, which will be caption of the algorithm in IOHanalyzer.
    logger = IOH_logger("./", "result-ES_3", "ES_3", "ES_3")

    for p_id in problem_id :
        for d in dimension :
            for i_id in instance_id:
                ## Getting the problem with corresponding id,dimension, and instance.
                f = IOH_function(p_id, d, i_id, suite="BBOB")
                f.add_logger(logger)
                xopt, fopt = ES_3(f)
    logger.clear_logger()
