
#include "IOHprofiler_BBOB_suite.hpp"
#include "IOHprofiler_csv_logger.h"

#define EVALUATION_BUDGET 10000

// !! This is where your algorithm locates, please replace the function name by your names and remain arguments the same.
void studentname1_studentname2_ES(std::shared_ptr<IOHprofiler_problem<double> > problem, std::shared_ptr<IOHprofiler_csv_logger> logger) {

  int n = problem->IOHprofiler_get_number_of_variables();
  // !! IOHprofiler_get_evaluations() returns the number of function evaluations has been done on the problem.
  // !! IOHprofiler_hit_optimal() returns True if the optimum has been found.
  while (problem->IOHprofiler_get_evaluations() <= EVALUATION_BUDGET * n  && !problem->IOHprofiler_hit_optimal()) {
    std::vector<double> x(n);
    for (int i = 0; i != problem->IOHprofiler_get_number_of_variables(); ++i) {
      x[i] = ((double) rand() / (RAND_MAX))  * 10 - 5;
    }
    
    // !! evaluate(x) returns the fitness of x on the problem, and the number of function evaluations plus one.
    // !! do_log(problem->loggerInfo()) will output the evaluation info. Please make sure you call this function after every evaluate(x). 
    double y = problem->evaluate(x);
    logger->do_log(problem->loggerInfo());
  }
}

int main() {

  // Declariation of a suite consists of all problems to be tested.
  // Here we declare 24 functions in bbob suite on three dimensions. 
  // Each problem will be tested with 25 independent instances, and each instance will be tested only once.
  std::vector<int> problem_id;
  for (int i = 1; i != 25; i++) {
    problem_id.push_back(i);
  }
  std::vector<int> instance_id;
  for (int i = 1; i != 26; i++) {
    instance_id.push_back(i);
  }
  std::vector<int> dimension = {2,5,20};
  BBOB_suite bbob(problem_id,instance_id,dimension);

  // Declariation of IOHprofiler_csv_logger.
  // 'result' is the name of output folder.
  // 'studentname1_studentname2' represents algorithm name and algorithm info, which will be caption of the algorithm in IOHanalyzer.
  std::shared_ptr<IOHprofiler_csv_logger > csv_logger (new IOHprofiler_csv_logger("./", "result","studentname1_studentname2_cpp","studentname1_studentname2_cpp"));
  csv_logger->activate_logger();
  csv_logger->track_suite(bbob.IOHprofiler_suite_get_suite_name());

  std::shared_ptr<IOHprofiler_problem<double>> problem;

  /// Problems are tested one by one until 'get_next_problem' returns NULL.
  while ((problem = bbob.get_next_problem()) != nullptr) {
    csv_logger->track_problem(problem->IOHprofiler_get_problem_id(), 
                          problem->IOHprofiler_get_number_of_variables(), 
                          problem->IOHprofiler_get_instance_id(),
                          problem->IOHprofiler_get_problem_name(),
                          problem->IOHprofiler_get_optimization_type());
    studentname1_studentname2_ES(problem,csv_logger);
  }

  return 0;
}