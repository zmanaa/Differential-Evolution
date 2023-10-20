import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from numpy.linalg import eig
import pandas as pd
import os

# Plotting params
plt.style.use(['science', 'muted'])
plt.rcParams['figure.figsize'] = (19/2,6/2) 
plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2.0


def cost_fn_1(x):
    """
    Calculates the cost function for the given individual.
    
    Parameters:
        individual: List containing the gains and time constants
        
    Returns:
        cost: Value of the cost function
    """
    func = x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + x[0]*x[1] + x[1]*x[2] - 8*x[0] - 16*x[1] - 32*x[2] + 110
    return func

def cost_fn_2(individual):
    """
    Calculates the cost function for the given individual for control systems problem.
    
    Parameters:
        individual: List containing the gains and time constants
        
    Returns:
        cost: Value of the cost function
    """
     
    # Extract the gains
    K = individual[0]
    T1 = individual[1]
    T2 = individual[2]
    
    eigenvalues = []
    real_parts = []

    A = np.array([[0, 377, 0, 0],
                  [-0.0587, 0, -0.1303, 0],
                  [-0.0899, 0, -0.1956, 0.1289],
                  [95.605, 0, -816.0862, -20]])
    
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1000]]).T

    KCL = np.array([[-0.0587, 0, -0.1303, 0],
                    [-0.0587*K*T1/T2, 0, -0.1303*K*T1/T2, 0]])

    BCL = np.array([[-0.333, 0],
                    [K/T2*(1-T1/3), -1/T2]])

    Ac = np.block([[A, B], [KCL, BCL]])
    eigenvalues_closed_loop = np.linalg.eigvals(Ac)
    
    # Find indices of eigenvalues with non-zero imaginary parts
    non_zero_imaginary_indices = np.where(np.imag(eigenvalues_closed_loop) != 0)[0]
    
    # Store the real parts of the eigenvalues in sigmas
    sigma = np.real(eigenvalues_closed_loop)
    
    # Store the real parts corresponding to real_idxs in osci_modes_sigmas
    non_zero_imaginary_sigmas = sigma[non_zero_imaginary_indices]
    
    non_negative_real_indices = np.where(sigma >= 0)[0]
    
    if non_negative_real_indices.size != 0:
        func = float('inf')
    else:
        func = np.max(non_zero_imaginary_sigmas)
    
    return func


def plot_results(data, generation_count, iteration, filename):
    """Plot the cost function against the generations"""
    plt.plot(range(0, generation_count), data['best_cost'], label=f"Iteration No. {iteration+1}")
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.legend()


def tabulate_data(data, settings, generation_count):
    """Tabulating the data"""    
    df = pd.DataFrame({
        'Best Cost': [data['best_cost'][-1]],
        'Best Solution': [data['best_solution'][-1]],
        'Population Size': [settings['population_size']],
        'Maximum No. of Generations': [settings['number_of_generations']],
        'No. of Generations': generation_count,
        'Mutation Constant': [settings['mutation_constant']],
        'Crossover Constant': [settings['crossover_constant']]
    })

    return df

