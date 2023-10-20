"""
Differential Evolution optimizer implementation

Uses the differential evolution algorithm to minimize a cost function. This optimizer 
represents a population of candidate solutions that evolve towards better solutions over 
multiple generations using mutation and crossover genetic operators.

Author: Zeyad M. Manaa
Date: October 20, 2023
"""


import numpy as np
import random
from typing import List, Callable
from collections import deque
import math


class DifferentialEvolution:
    """
    Differential Evolution Optimizer Class
    
    Performs differential evolution optimization to minimize
    a given cost function over multiple generations.
    """


    def __init__(self, opts, ):
        """Initialization for the optmization params""" 
        self.var_min_list = opts['var_min_list']
        self.var_max_list = opts['var_max_list']
        self.chromosome_size = opts['chromosome_size']
        self.population_size = opts['population_size']
        self.mutation_constant = opts['mutation_constant']
        self.cost_func = opts['cost_func']
        self.crossover_constant = opts['crossover_constant']
        self.number_of_generations = opts['number_of_generations']
        self.term_counter = 20
        self.data = {'best_solution': [],
                'cost': deque([]),
                'best_cost': deque([])}  # cost means avg_cost

    def get_opts(self):
        opts  = {
            'var_min_list': self.var_min_list,
            'var_max_list': self.var_max_list,
            'chromosome_size': self.chromosome_size,
            'population_size': self.population_size,
            'mutation_constant': self.mutation_constant,
            'crossover_constant': self.crossover_constant,
            'number_of_generations': self.number_of_generations
            }
        return opts


    def ensure_bounds(self, individual):
        """Ensure individual remains within variable bounds"""

        # Get problem options
        var_min_list = self.var_min_list
        var_max_list = self.var_max_list

        # Check and enforce limits for each mutated gene
        for i in range(len(individual)):
            individual[i] = max(var_min_list[i], min(var_max_list[i], individual[i]))
        
        return individual


    def initialize_population(self):
        """
        Initialize a random population within given bounds.

        Generates a random population matrix where each gene is uniformly
        distributed between its respective var_min and var_max.

        Returns:
            Random population matrix of shape (self.population_size, self.chromosome_size)
        """
        population = np.empty((self.population_size, self.chromosome_size))

        for i in range(self.population_size):
            for j in range(self.chromosome_size):
                population[i, j] = random.uniform(self.var_min_list[j], self.var_max_list[j])
        return population


    def evaluate_costs(self,
        population: List,
        population_size: int, 
        cost_fn: Callable
    ) -> List[float]:
        """
        Evaluate the costs/fitness of each chromosome in a population.

        Applies the given cost function to each chromosome and returns
        the list of costs.

        Args:
            population: Population 
            population_size: Size of population
            cost_fn: Function that takes a chromosome and 
                                returns the corresponding cost

        Returns:
            List of costs corresponding to each chromosome
        """

        costs = []

        for individual in population:
            costs.append(cost_fn(individual))

        return costs


    def mutate(self, population):
        """Generate trial vectors using mutation operator""" 

        trial_vectors = list()

        for individual in range(0, self.population_size):
            random_ints = [random.randint(0, self.population_size-1) for ـ in range(len(population[0]))]

            x_r1 = population[random_ints[0]]
            x_r2 = population[random_ints[1]]
            x_r3 = population[random_ints[2]]
            x_target = population[individual]

            mutant_vector = x_r1 + self.mutation_constant * (x_r2 - x_r3)

            # Making sure the mutant vector is within the limits
            mutant_vector = self.ensure_bounds(mutant_vector)

            # Update trial vectors
            trial_vectors.append(mutant_vector)

        return trial_vectors

    
    def crossover(self, population, trial_vector):
        """Generate next generation using crossover"""

        offspring = np.array(population.copy())

        for i in range(0, self.population_size - 1):
            for j in range(0, len(population[0])):
                random_number = [random.uniform(0, 1) for ـ in range(len(population[0]))]
                if random_number[j] < self.crossover_constant:
                    offspring[i][j] = trial_vector[i][j]
                else:
                    offspring[i][j] = population[i][j]
            # Making sure the mutant vector is within the limits
            offspring[i] = self.ensure_bounds(offspring[i])

        return offspring


    def solve_optimization(self):
        """Run DE algorithm until termination criteria met"""

        # Initialize a population
        population = self.initialize_population()

        # Set initial params
        best_sol, best_cost, avg_cost = (population[0]), (math.inf), (0)  
        termination_counter = 0
        data = self.data

        generation_count = 0

        # Initialize the next population temporarily
        next_population = np.array(population.copy())

        while True:
            # Create the trial vectors, by applying the mutation operator
            trial_vectors = self.mutate(population)

            # Create an offspring, by applying the crossover operator
            offspring = self.crossover(population,
                                       trial_vectors)

            # Cost evaluation for both offspring and parents
            parents_costs = self.evaluate_costs(population,
                                                self.population_size,
                                                self.cost_func)
            offspring_costs = self.evaluate_costs(offspring,
                                                self.population_size,
                                                self.cost_func)
            
            for j in range(self.population_size):
                if offspring_costs[j] < parents_costs[j]:
                    next_population[j] = offspring[j]
                else:
                    next_population[j] = population[j]
            
            population = next_population

            generation_count +=1

            new_population_costs = self.evaluate_costs(population,
                                                self.population_size,
                                                self.cost_func)
            
            mean_cost = np.mean(new_population_costs)
            data['best_solution'].append(best_sol)
            data['best_cost'].append(best_cost)
            data['cost'].append(mean_cost)

            last_best_cost = best_cost
            for indx in range(self.population_size):
                if new_population_costs[indx] < best_cost:
                    best_sol = population[indx]
                    best_cost = new_population_costs[indx]
            
            if last_best_cost == best_cost:
                termination_counter +=1
            else:
                termination_counter = 0
            
            if termination_counter > self.term_counter:
                return data, generation_count

            # Terminate if maximum number of generation is reached
            if generation_count == self.number_of_generations:
                return data, generation_count
