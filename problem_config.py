from     pymoo.problems.single import Ackley,Zakharov, G1, G2, Schwefel, Rosenbrock,Rastrigin, Himmelblau, Sphere, Griewank
import numpy as np


#from my_custom_problem import MyCustomProblem  # Import your custom problem class
def get_problem(problem_name, lb, ub):
    if problem_name == "Ackley":
        problem = Ackley()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub
    elif problem_name == "Sphere":
        problem = Sphere()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub

    elif problem_name == "Himmelblau":
        problem = Himmelblau()
        problem.n_var = 2
        problem.xl = lb
        problem.xu = ub
    elif problem_name == "Rastrigin":
        problem = Rastrigin()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub

    elif problem_name == "Rosenbrock":
        problem = Rosenbrock()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub

    elif problem_name == "Zakharov":
        problem = Zakharov()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub
    elif problem_name == "griewank":
        problem = Griewank()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub
    elif problem_name == "Schwefel":
        problem = Schwefel()
        problem.n_var = 10
        problem.xl = lb
        problem.xu = ub
    elif problem_name == "G1":
        problem = G1()
    elif problem_name == "G2":
        problem = G2()
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    return problem
