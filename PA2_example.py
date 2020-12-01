from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
import numpy as np
import sys

budget = 10000

## !! This is where your algorithm locates, please replace the function name by your names and remain arguments the same.
def studentname1_studentname2_ES(problem):
    n = problem.number_of_variables
    
    fopt = -sys.maxsize-1

    ## !! final_target_hit returns True if the optimum has been found.
    ## !! evaluations returns the number of function evaluations has been done on the problem. 
    while not problem.final_target_hit and problem.evaluations < budget * n:
       
        x = np.random.rand(n) * 10 - 5
        
        ## !! problem(x) returns the fitness of x on the problem
        f = problem(x)

        if f >= fopt:
            x_prime = x
            fopt = f

    return x_prime, fopt

if __name__ == '__main__':

    ## Declarations of Ids, instances, and dimensions that the problems to be tested.
    problem_id = range(1,25)
    instance_id = range(1,26)
    dimension = [2,5,20]

    ## Declariation of IOHprofiler_csv_logger.
    ## 'result' is the name of output folder.
    ## 'studentname1_studentname2' represents algorithm name and algorithm info, which will be caption of the algorithm in IOHanalyzer.
    logger = IOH_logger("./", "result", "studentname1_studentname2", "studentname1_studentname2")

    for p_id in problem_id :
        for d in dimension :
            for i_id in instance_id:
                ## Getting the problem with corresponding id,dimension, and instance.
                f = IOH_function(p_id, d, i_id, suite="BBOB")
                f.add_logger(logger)
                xopt, fopt = studentname1_studentname2_ES(f)
    logger.clear_logger()
