from pymoo.problems.single import Ackley, G1, G2
import numpy as np

#from my_custom_problem import MyCustomProblem  # Import your custom problem class
def get_problem(problem_name, lb, ub):
    if problem_name == "Ackley":
        problem = Ackley()
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
