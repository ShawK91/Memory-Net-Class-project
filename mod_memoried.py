from random import randint
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import LSTM, GRU, SimpleRNN
#from keras.layers.advanced_activations import SReLU
#from keras.regularizers import l2, activity_l2
#from keras.optimizers import SGD
from deap import base
from deap import creator
from deap import tools
import math, copy
import MultiNEAT as NEAT
import numpy as np, time
import random
#import pickle
#import neat as py_neat
#from neat import nn
import sys,os

class normal_net:
    def __init__(self, num_input, num_hnodes, num_output, mean = 0, std = 1):
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.net_output = [] * num_output
        #W_01 matrix contains the weigh going from input (0) to the 1st hidden layer(1)
        self.w_01 = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1))) #Initilize weight using Gaussian distribution
        self.w_01 = np.mat(np.reshape(self.w_01, (num_hnodes, (num_input + 1)))) #Reshape the array to the weight matrix
        self.w_12 = np.mat(np.random.normal(mean, std, num_output*(num_hnodes + 1)))
        self.w_12 = np.mat(np.reshape(self.w_12, (num_output, (num_hnodes + 1)))) #Reshape the array to the weight matrix

    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def sigmoid(self, layer_input): #Sigmoid transform

        #Just make sure the sigmoid does not explode and cause a math error
        p = layer_input.A[0][0]
        if p > 700:
            ans = 0.999999
        elif p < -700:
            ans = 0.000001
        else:
            ans =  1 / (1 + math.exp(-p))

        return ans

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        for i in layer_input: i = i / (1 + math.fabs(i))
        return layer_input
    #TODO TEST BENCHMARK


    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def format_input(self, input, add_bias = True): #Formats and adds bias term to given input at the end
        if add_bias:
            input = np.concatenate((input, [1.0]))
        return np.mat(input)

    def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
        #First hidden layer
        self.input = self.format_input(input) #Format and add bias term at the end
        self.z1 = self.linear_combination(self.w_01, self.input.transpose()) #Forward pass linear
        self.z1 = self.format_input(self.z1, False)#Format
        self.h1 = self.fast_sigmoid(self.z1) #Use Relu transform

        #Output layer
        self.h1 = np.vstack((self.h1,[1])) #Add bias term
        self.z2 = self.w_12 * self.h1 #Forward pass linear
        self.net_output = (self.fast_sigmoid((self.z2))) #Use sigmoid transform
        return np.array(self.net_output).tolist()

    def get_weights(self):
        w1 = np.array(self.w_01).flatten().copy()
        w2 = np.array(self.w_12).flatten().copy()
        weights = np.concatenate((w1, w2 ))
        return weights

    def set_weights(self, weights):
        w1 = weights[:self.num_hnodes*(self.num_input + 1)]
        w2 = weights[self.num_hnodes*(self.num_input + 1):]
        self.w_01 = np.mat(np.reshape(w1, (self.num_hnodes, (self.num_input + 1)))) #Reshape the array to the weight matrix
        self.w_12 = np.mat(np.reshape(w2, (self.num_output, (self.num_hnodes + 1)))) #Reshape the array to the weight matrix

class memory_net:
    def __init__(self, num_input, num_hnodes, num_output, mean = 0, std = 1):
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.net_output = []
        self.last_output = np.mat(np.zeros(num_output)).transpose()
        self.memory_cell = np.mat(np.random.normal(mean, std, num_hnodes)).transpose() #Memory Cell

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inpgate = np.mat(np.reshape(self.w_inpgate, (num_hnodes, (num_input + 1))))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inpgate = np.mat(np.reshape(self.w_rec_inpgate, (num_hnodes, (num_output + 1))))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_inpgate = np.mat(np.reshape(self.w_mem_inpgate, (num_hnodes, (num_hnodes + 1))))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_inp = np.mat(np.reshape(self.w_inp, (num_hnodes, (num_input + 1))))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_inp = np.mat(np.reshape(self.w_rec_inp, (num_hnodes, (num_output + 1))))

        #Forget gate
        self.w_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_input + 1)))
        self.w_forgetgate = np.mat(np.reshape(self.w_forgetgate, (num_hnodes, (num_input + 1))))
        self.w_rec_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_output + 1)))
        self.w_rec_forgetgate = np.mat(np.reshape(self.w_rec_forgetgate, (num_hnodes, (num_output + 1))))
        self.w_mem_forgetgate = np.mat(np.random.normal(mean, std, num_hnodes*(num_hnodes + 1)))
        self.w_mem_forgetgate = np.mat(np.reshape(self.w_mem_forgetgate, (num_hnodes, (num_hnodes + 1))))

        #Output weights
        self.w_output = np.mat(np.random.normal(mean, std, num_output*(num_hnodes + 1)))
        self.w_output = np.mat(np.reshape(self.w_output, (num_output, (num_hnodes + 1)))) #Reshape the array to the weight matrix



    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def sigmoid(self, layer_input): #Sigmoid transform

        #Just make sure the sigmoid does not explode and cause a math error
        p = layer_input.A[0][0]
        if p > 700:
            ans = 0.999999
        elif p < -700:
            ans = 0.000001
        else:
            ans =  1 / (1 + math.exp(-p))

        return ans

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        for i in layer_input: i[0] = i / (1 + math.fabs(i))
        return layer_input

    def gated_fast_sigmoid(self, layer_input): #Sigmoid transform between 0 and 1
        for i in layer_input:
            i[0] = 0.5 * (i / (1 + math.fabs(i))) + 0.5
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def format_input(self, input, add_bias = True): #Formats and adds bias term to given input at the end
        if add_bias:
            input = np.concatenate((input, [1.0]))
        return np.mat(input)

    def format_memory(self, memory):
        ig = np.mat([1])
        return np.concatenate((memory, ig))


    def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
        self.input = self.format_input(input).transpose()  # Format and add bias term at the end
        last_memory = self.format_memory(self.memory_cell)
        last_output = self.format_memory(self.last_output)

        #Input gate
        ig_1 = self.linear_combination(self.w_inpgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_inpgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_inpgate, last_memory)
        input_gate_out = ig_1 + ig_2 + ig_3
        input_gate_out = self.gated_fast_sigmoid(input_gate_out)

        #Input processing
        ig_1 = self.linear_combination(self.w_inp, self.input)
        ig_2 = self.linear_combination(self.w_rec_inp, last_output)
        block_input_out = ig_1 + ig_2
        block_input_out = self.fast_sigmoid(block_input_out)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Forget Gate
        ig_1 = self.linear_combination(self.w_forgetgate, self.input)
        ig_2 = self.linear_combination(self.w_rec_forgetgate, last_output)
        ig_3 = self.linear_combination(self.w_mem_forgetgate, last_memory)
        forget_gate_out = ig_1 + ig_2 + ig_3
        forget_gate_out = self.gated_fast_sigmoid(forget_gate_out)

        #Memory Output
        memory_output = np.multiply(forget_gate_out, self.memory_cell)

        #Update memory Cell
        self.memory_cell = memory_output + input_out

        #Compute final output
        new_mem = self.format_memory(self.memory_cell)
        self.net_output = self.linear_combination(self.w_output, new_mem)
        self.net_output = self.fast_sigmoid(self.net_output)
        return np.array(self.net_output).tolist()

    def get_weights(self):
        #TODO NOT OPERATIONAL
        w1 = np.array(self.w_01).flatten().copy()
        w2 = np.array(self.w_12).flatten().copy()
        weights = np.concatenate((w1, w2 ))
        return weights

    def set_weights(self, weights):
        #Input gates
        start = 0; end = self.num_hnodes*(self.num_input + 1)
        w_inpgate = weights[start:end]
        self.w_inpgate = np.mat(np.reshape(w_inpgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inpgate = weights[start:end]
        self.w_rec_inpgate = np.mat(np.reshape(w_rec_inpgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_inpgate = weights[start:end]
        self.w_mem_inpgate = np.mat(np.reshape(w_mem_inpgate, (self.num_hnodes, (self.num_hnodes + 1))))

        # Block Inputs
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_inp = weights[start:end]
        self.w_inp = np.mat(np.reshape(w_inp, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_inp = weights[start:end]
        self.w_rec_inp = np.mat(np.reshape(w_rec_inp, (self.num_hnodes, (self.num_output + 1))))

        #Forget Gates
        start = end; end += self.num_hnodes*(self.num_input + 1)
        w_forgetgate = weights[start:end]
        self.w_forgetgate = np.mat(np.reshape(w_forgetgate, (self.num_hnodes, (self.num_input + 1))))

        start = end; end += self.num_hnodes*(self.num_output + 1)
        w_rec_forgetgate = weights[start:end]
        self.w_rec_forgetgate = np.mat(np.reshape(w_rec_forgetgate, (self.num_hnodes, (self.num_output + 1))))

        start = end; end += self.num_hnodes*(self.num_hnodes + 1)
        w_mem_forgetgate = weights[start:end]
        self.w_mem_forgetgate = np.mat(np.reshape(w_mem_forgetgate, (self.num_hnodes, (self.num_hnodes + 1))))

        #Output weights
        start = end; end += self.num_output*(self.num_hnodes + 1)
        w_output= weights[start:end]
        self.w_output = np.mat(np.reshape(w_output, (self.num_output, (self.num_hnodes + 1))))

        #Memory Cell (prior)
        start = end; end += self.num_hnodes
        memory_cell= weights[start:end]
        self.memory_cell = np.mat(memory_cell).transpose()

class Deap_evo:

    def __init__(self, parameters):
        self.num_weights = parameters.deap_param.total_num_weights; self.population_size = parameters.population_size; self.paramaters = parameters
        self.num_elitists = int(parameters.deap_param.elite_fraction * parameters.population_size)

        #Numpy based optimization module
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("weight_init", random.uniform, -0.25, 0.25)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.weight_init, self.num_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Operators
        #toolbox.register("evaluate", assign_fitness)  # Fitness assignment function
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.01)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        self.pop = self.toolbox.population(n=self.population_size)

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def epoch(self, fitnesses):
        for i, individual in enumerate(self.pop):
            individual.fitness.values = fitnesses[i] #Assign fitness

        #Elitist reserve
        elitist_reserve = []
        if self.num_elitists != 0:
            elitist_index = self.list_argsort(fitnesses)[-self.num_elitists:]
            for ind in elitist_index: elitist_reserve.append(copy.deepcopy(self.pop[ind]))

        #Selection step
        offspring = self.toolbox.select(self.pop, len(self.pop)-self.num_elitists)
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.paramaters.crossover_prob:
                self.toolbox.mate(child1, child2)

        #Apply mutation
        for mutant in offspring:
            if random.random() < self.paramaters.mutation_prob:
                self.toolbox.mutate(mutant)

        #Put elitist individuals back into population
        offspring = offspring + elitist_reserve

        #Replace the old population
        self.pop[:] = offspring

    def save_population(self):
        for index, individual in self.pop:
            np.savetxt('ind_'+str(index), individual, delimeter = ',')

class PyNeat_Config_object(object):
    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']

    def __init__(self, parameters):
        from neat.reproduction import DefaultReproduction
        from neat.stagnation import DefaultStagnation
        from neat.genes import NodeGene, ConnectionGene
        from neat.genome import Genome, FFGenome
        from neat import activation_functions

        self.registry = {'DefaultStagnation': DefaultStagnation,
                         'DefaultReproduction': DefaultReproduction}
        self.type_config = {}

        # Phenotype configuration
        self.input_nodes = parameters.evo_input_size
        self.output_nodes = 5
        self.hidden_nodes = parameters.py_neat_config.hidden_nodes
        self.initial_connection = parameters.py_neat_config.initial_connection
        self.connection_fraction = None
        self.max_weight = parameters.py_neat_config.max_weight
        self.min_weight = parameters.py_neat_config.min_weight
        self.feedforward = parameters.py_neat_config.feedforward
        self.weight_stdev = parameters.py_neat_config.weight_stdev
        self.activation_functions = parameters.py_neat_config.activation_functions.strip().split()

        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify that specified activation functions are valid.
        for fn in self.activation_functions:
            if not activation_functions.is_valid(fn):
                raise Exception("Invalid activation function name: {0!r}".format(fn))

        # Select a genotype class.
        if self.feedforward:
            self.genotype = FFGenome
        else:
            self.genotype = Genome

        # Genetic algorithm configuration
        self.pop_size = parameters.population_size
        self.max_fitness_threshold = parameters.py_neat_config.max_fitness_threshold
        self.prob_add_conn = parameters.py_neat_config.prob_add_conn
        self.prob_add_node = parameters.py_neat_config.prob_add_node
        self.prob_delete_conn = parameters.py_neat_config.prob_delete_conn
        self.prob_delete_node = parameters.py_neat_config.prob_delete_node
        self.prob_mutate_bias = parameters.py_neat_config.prob_mutate_bias
        self.bias_mutation_power = parameters.py_neat_config.bias_mutation_power
        self.prob_mutate_response = parameters.py_neat_config.prob_mutate_response
        self.response_mutation_power = parameters.py_neat_config.response_mutation_power
        self.prob_mutate_weight = parameters.py_neat_config.prob_mutate_weight
        self.prob_replace_weight = parameters.py_neat_config.prob_replace_weight
        self.weight_mutation_power = parameters.py_neat_config.weight_mutation_power
        self.prob_mutate_activation = parameters.py_neat_config.prob_mutate_activation
        self.prob_toggle_link = parameters.py_neat_config.prob_toggle_link
        self.reset_on_extinction = bool(parameters.py_neat_config.reset_on_extinction)

        # genotype compatibility
        self.compatibility_threshold = parameters.py_neat_config.compatibility_threshold
        self.excess_coefficient = parameters.py_neat_config.excess_coefficient
        self.disjoint_coefficient = parameters.py_neat_config.disjoint_coefficient
        self.weight_coefficient = parameters.py_neat_config.weight_coefficient

        # Gene types
        self.node_gene_type = NodeGene
        self.conn_gene_type = ConnectionGene

        # Default stagnation
        self.species_fitness_func = parameters.py_neat_config.species_fitness_func
        self.max_stagnation = parameters.py_neat_config.max_stagnation

        # Default Reporoduction
        self.elitism = parameters.py_neat_config.elitism
        self.survival_threshold = parameters.py_neat_config.survival_threshold

        stagnation_type_name = parameters.py_neat_config.stagnation_type
        reproduction_type_name = parameters.py_neat_config.reproduction_type

        if stagnation_type_name not in self.registry:
            raise Exception('Unknown stagnation type: {!r}'.format(stagnation_type_name))
        self.stagnation_type = self.registry[stagnation_type_name]

        self.type_config[stagnation_type_name] = [
            ('species_fitness_func', parameters.py_neat_config.species_fitness_func),
            ('max_stagnation ', parameters.py_neat_config.max_stagnation)]

        if reproduction_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.reproduction_type = self.registry[reproduction_type_name]
        self.type_config[reproduction_type_name] = [('elitism', parameters.py_neat_config.elitism), (
        'survival_threshold', parameters.py_neat_config.survival_threshold)]

        # Gather statistics for each generation.
        self.collect_statistics = True
        # Show stats after each generation.
        self.report = True
        # Save the best genome from each generation.
        self.save_best = True
        # Time in minutes between saving checkpoints, None for no timed checkpoints.
        self.checkpoint_time_interval = 30
        # Time in generations between saving checkpoints, None for no generational checkpoints.
        self.checkpoint_gen_interval = 100

    def register(self, typeName, typeDef):
        """
        User-defined classes mentioned in the config file must be provided to the
        configuration object before the load() method is called.
        """
        self.registry[typeName] = typeDef

    def get_type_config(self, typeInstance):
        return dict(self.type_config[typeInstance.__class__.__name__])

class PyNeat_handler():
    def __init__(self, parameters):
        self.pyNeat_config_object = PyNeat_Config_object(parameters)


    def get_genomes(self, pop):
        genomes = []
        for s in pop.species.species:
            genomes.extend(s.members)
        return genomes

    def epoch(self, pop, genomes):
        sys.stdout = open(os.devnull, "w")
        """
        The user-provided fitness_function should take one argument, a list of all genomes in the population,
        and its return value is ignored.  This function is free to maintain external state, perform evaluations
        in parallel, and probably any other thing you want.  The only requirement is that each individual's
        fitness member must be set to a floating point value after this function returns.
        It is assumed that fitness_function does not modify the list of genomes, or the genomes themselves, apart
        from updating the fitness member.
        """
        pop.generation += 1
        pop.reporters.start_generation(pop.generation)
        pop.total_evaluations += len(genomes)

        # Gather and report statistics.
        best = max(genomes)
        pop.reporters.post_evaluate(genomes, pop.species.species, best)

        # Save the best genome from the current generation if requested.
        if pop.config.save_best:
            with open('best_genome', 'wb') as f:
                pickle.dump(best, f)

        # Save if the fitness threshold is reached.
        if best.fitness >= pop.config.max_fitness_threshold:
            pop.reporters.found_solution(pop.generation, best)
            with open('solution_genome', 'wb') as f:
                pickle.dump(best, f)

        # Create the next generation from the current generation.
        new_population = pop.reproduction.reproduce(pop.species, pop.config.pop_size)

        # Check for complete extinction
        if not pop.species.species:
            pop.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if pop.config.reset_on_extinction:
                new_population = pop.reproduction.create_new(pop.config.pop_size)
            else:
                print 'Extinction'

        # Update species age.
        for s in pop.species.species:
            s.age += 1

        # Divide the new population into species.
        pop.species.speciate(new_population)

        if pop.config.checkpoint_gen_interval is not None and pop.generation % pop.config.checkpoint_gen_interval == 0:
            pop.save_checkpoint(checkpoint_type="generation")

        pop.reporters.end_generation()
        sys.stdout = sys.__stdout__

class Evo_net():
    def __init__(self, parameters):
        self.parameters = parameters

        if parameters.use_deap:
            self.num_weights = parameters.deap_param.total_num_weights;
            self.population_size = parameters.population_size;
            self.paramaters = parameters
            self.num_elitists = int(parameters.deap_param.elite_fraction * parameters.population_size)

            # Numpy based optimization module
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
            self.toolbox = base.Toolbox()
            # Attribute generator
            self.toolbox.register("weight_init", random.uniform, -0.25, 0.25)
            # Structure initializers
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.weight_init,
                                  self.num_weights)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            # Operators
            # toolbox.register("evaluate", assign_fitness)  # Fitness assignment function
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.01)
            self.toolbox.register("select", tools.selTournament, tournsize=5)
            self.pop = self.toolbox.population(n=self.population_size)
            self.fitness_evals = [[] for x in xrange(parameters.population_size)]
            if self.parameters.is_memoried:
                self.net = memory_net(self.parameters.deap_param.num_input, self.parameters.deap_param.num_hnodes, self.parameters.deap_param.num_output)
            else:
                self.net = normal_net(self.parameters.deap_param.num_input, self.parameters.deap_param.num_hnodes,
                                      self.parameters.deap_param.num_output)



        elif parameters.use_neat:
            if parameters.use_py_neat: #Python implementation of NEAT


                from neat import population, nn#, statistics, visualize, config
                self.pyneat_handler = PyNeat_handler(self.parameters) #Make the pyNeat_handler object
                pyneat_config = self.pyneat_handler.pyNeat_config_object #Import the configurations
                self.pop = population.Population(pyneat_config, use_config_override=True)
                self.genome_list = self.pyneat_handler.get_genomes(self.pop)
                self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations through an iteration
                self.net_list = [[] for x in xrange(len(self.genome_list))] #Stores the networks for the genomes
                self.hof_fitness_evals = [[] for x in xrange(len(self.genome_list))]

            else: #C++ NEAT

                seed = 0 if (parameters.params.evo_hidden == 0) else 1  # Controls sees based on genome initialization
                g = NEAT.Genome(0, parameters.evo_input_size, parameters.params.evo_hidden, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                                NEAT.ActivationFunction.UNSIGNED_SIGMOID, seed, parameters.params)  # Constructs genome
                g.Save('initial')
                self.pop = NEAT.Population(g, parameters.params, True, 1.0, 0)  # Constructs population of genome
                self.pop.RNG.Seed(0)
                self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
                self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations through an iteration
                self.net_list = [[] for x in xrange(len(self.genome_list))] #Stores the networks for the genomes
                self.base_mpc = self.pop.GetBaseMPC()
                self.current_mpc = self.pop.GetCurrentMPC()
                self.delta_mpc = self.current_mpc - self.base_mpc
                self.oldest_genome_id = 0
                self.youngest_genome_id = 0
                self.delta_age = self.oldest_genome_id - self.youngest_genome_id
                self.test_net = NEAT.NeuralNetwork()

        else: #keras
            self.pop = Population(parameters.evo_input_size, parameters.keras_evonet_hnodes, 5, parameters.population_size)
            self.fitness_evals = [[] for x in xrange(parameters.population_size)] #Controls fitnesses calculations through an iteration
            self.net_list = [[] for x in xrange(parameters.population_size)] #Stores the networks for the genomes

    def save_population(self):
        if not os.path.exists('Deap_pop'):
            os.makedirs('Deap_pop')
        for index, individual in enumerate(self.pop):
            np.savetxt('Deap_pop/ind_'+str(index), individual)

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def epoch(self): #Method to complete epoch after fitness has been assigned to the genomes
        if self.parameters.use_deap:
            for i, individual in enumerate(self.pop):
                individual.fitness.values = self.fitness_evals[i]  # Assign fitness

            # Elitist reserve
            elitist_reserve = []
            if self.num_elitists != 0:
                elitist_index = self.list_argsort(self.fitness_evals)[-self.num_elitists:]
                for ind in elitist_index: elitist_reserve.append(copy.deepcopy(self.pop[ind]))

            # Selection step
            offspring = self.toolbox.select(self.pop, len(self.pop) - self.num_elitists)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.paramaters.deap_param.crossover_prob:
                    self.toolbox.mate(child1, child2)

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.paramaters.deap_param.mutation_prob:
                    self.toolbox.mutate(mutant)

            # Put elitist individuals back into population
            offspring = offspring + elitist_reserve

            # Replace the old population
            self.pop[:] = offspring
            return

        elif self.parameters.use_py_neat: #Python based NEAT use Epoch method written outside
            self.pyneat_handler.epoch(self.pop, self.genome_list)
            return
        else: #For C++ NEAT and Keras based Evonet, use inbuilt method
            self.pop.Epoch()  # Epoch update method inside NEAT and Keras
            return

    def referesh_genome_list(self):

        if self.parameters.use_deap:
            self.fitness_evals = [[] for x in xrange(self.parameters.population_size)]
            return


        # elif self.parameters.use_neat:
        #     if self.parameters.use_py_neat: #Python implementation of NEAT
        #         self.genome_list = self.pyneat_handler.get_genomes(self.pop)
        #     else:
        #         self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
        #     self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations throug an iteration
        #     self.net_list = [[] for x in xrange(len(self.genome_list))]  # Stores the networks for the genomes
        #     return
        # else: #Keras Evo-net
        #     self.fitness_evals = [[] for x in xrange(self.parameters.population_size)]  # Controls fitnesses calculations throug an iteration
        #     self.net_list = [[] for x in xrange(self.parameters.population_size)]  # Stores the networks for the genomes
        #     return

    def build_net(self, index):
        if self.parameters.use_deap:
            self.net.set_weights(self.pop[index])
            return


            # if self.parameters.use_neat:
            #     if self.parameters.use_py_neat: #Python NEAT
            #         self.net_list[index] = nn.create_feed_forward_phenotype(self.genome_list[index])
            #     else: #C++ NEAT
            #         self.net_list[index] = NEAT.NeuralNetwork();
            #         self.genome_list[index].BuildPhenotype(self.net_list[index]);
            #         self.net_list[index].Flush()  # Build net from genome
            #     #self.genome_list[index].Save('test')
            # else:
            #     self.net_list[index] = self.pop.net_pop[int(self.pop.pop_handle[index][0])]


    # Get action choice from Evo-net
    def run_evo_net(self, state):
        if self.parameters.use_deap:
            output = self.net.feedforward(state)
            return np.argmax(output)



        scores = [] #Probability output for five action choices
        if self.parameters.use_neat:
            if self.parameters.use_py_neat:  # Python NEAT
                if net_id != None: scores = self.net_list[net_id].serial_activate(state) #Not HOF
                else:
                    scores = self.hof_net.serial_activate(state) #HOF


            else: #C++ NEAT
                self.net_list[net_id].Flush()
                self.net_list[net_id].Input(state)  # can input numpy arrays, too for some reason only np.float64 is supported
                self.net_list[net_id].Activate()
                for i in range(5):
                    if not math.isnan(1 * self.net_list[net_id].Output()[i]):
                        scores.append(1 * self.net_list[net_id].Output()[i])
                    else:
                        scores.append(0)
        else: #Use keras Evo-net
            state = np.reshape(state, (1, len(state)))
            scores = self.net_list[net_id].predict(state)[0]
        if self.parameters.wheel_action and sum(scores) != 0: action = roulette_wheel(scores)
        elif sum(scores) != 0 and len(set(scores)) != 1: action = np.argmax(scores)
        else: action = randint(0,4)

        return action

    def update_fitness(self): #Update the fitnesses of the genome and also encode the best one for the generation

        if self.parameters.use_deap:
            best = 0;
            for i in range(self.parameters.population_size):
                if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                    if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i]) #Use lenient learner
                    else: avg_fitness = sum(self.fitness_evals[i])/len(self.fitness_evals[i])
                    if avg_fitness > best:
                        best = avg_fitness
                    self.pop[i].fitness.values = avg_fitness,



        elif self.parameters.use_neat:
            if self.parameters.use_py_neat: #Python NEAT
                best = 0; best_sim_index = 0
                for i, g in enumerate(self.genome_list):
                    if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                        if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i])  # Use lenient learner
                        else: avg_fitness = sum(self.fitness_evals[i]) / len(self.fitness_evals[i])
                        if self.parameters.use_hall_of_fame and len(self.hof_fitness_evals[0]) != 0:  # Hall of fame fitness adjustments (minus first time
                            avg_fitness = (1.0 - self.parameters.hof_weight) * avg_fitness + self.parameters.hof_weight * self.hof_fitness_evals[i][0]

                        if avg_fitness > best:
                            best = avg_fitness;
                            best_sim_index = i
                        g.fitness = avg_fitness #Update fitness

            else: #C++ NEAT
                youngest = 0; oldest = 10000000 #Magic intitalization numbers to find the oldest and youngest survuving genome
                best = 0; best_sim_index = 0
                for i, g in enumerate(self.genome_list):
                    if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                        if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i])  # Use lenient learner
                        else: avg_fitness = sum(self.fitness_evals[i]) / len(self.fitness_evals[i])
                        if avg_fitness > best:
                            best = avg_fitness;
                            best_sim_index = i
                        g.SetFitness(avg_fitness) #Update fitness
                        g.SetEvaluated() #Set as evaluated
                        if g.GetID() > youngest: youngest = g.GetID();
                        if g.GetID() < oldest: oldest = g.GetID();
                self.oldest_genome_id = oldest
                self.youngest_genome_id = youngest
                self.delta_age = self.youngest_genome_id - self.oldest_genome_id
                self.current_mpc = self.pop.GetCurrentMPC(); self.delta_mpc = self.current_mpc - self.base_mpc  # Update MPC's as well

        else: #Using keras Evo-net
            best = 0; best_sim_index = 0
            for i in range(self.parameters.population_size):
                if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                    if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i]) #Use lenient learner
                    else: avg_fitness = sum(self.fitness_evals[i])/len(self.fitness_evals[i])
                    if avg_fitness > best:
                        best = avg_fitness; best_sim_index = i
                    self.pop.pop_handle[i][1] = 1-avg_fitness #Update fitness
        #print best

class Agent:
    def __init__(self, grid, parameters, team_role_index):
        self.parameters = parameters
        self.team_role_index = team_role_index
        self.spawn_position = self.init_agent(grid)
        self.position = self.spawn_position[:]


        self.action = []
        self.evo_net = Evo_net(parameters)
        self.perceived_state = None #State of the gridworld as perceived by the agent
        self.split_learner_state = None #Useful for split learner
        self.fuel = 0



    def init_agent(self, grid, is_new_epoch=True):
        if not is_new_epoch: #If not a new epoch and intra epoch (random already initialized)
            x = self.spawn_position[0]; y = self.spawn_position[1]
            grid.state[x][y] = 1
            return [x,y]

        if self.parameters.aamas_domain == 1:  # AAMAS test domain
            if self.team_role_index == 0:
                x = 9; y = 9
            elif self.team_role_index == 1:
                x = 9; y = 10
            elif self.team_role_index == 2:
                x = 10; y = 9
            elif self.team_role_index == 3:
                x = 10; y = 10

            grid.state[x][y] = 1
            return [x, y]



        if self.parameters.domain_setup != 0:  # Known domain testing
            if self.parameters.domain_setup == 1:
                x = 4; y = 4


            elif self.parameters.domain_setup == 2:
                if self.team_role_index == 0:
                    x = 14; y = 7
                elif self.team_role_index == 1:
                    x = 14; y = 8

            elif self.parameters.domain_setup == 3:
                if self.team_role_index == 0:
                    x = 5; y = 5


            grid.state[x][y] = 1
            return [x, y]




        start = grid.observe;  end = grid.state.shape[0] - grid.observe - 1
        rad = int(grid.dim_row / math.sqrt(3) / 3)
        center = int((start + end) / 2)
        if grid.agent_rand:
            while True:
                x = randint(center - rad, center + rad)
                y = randint(center - rad, center + rad)
                if grid.state[x][y] != 1: #position not already occupied
                    break
        else:  # Not random
            trial = 0
            while True:
                while True:
                    x = center - rad + (trial % (rad*2))
                    if x <= center + rad: break #If within limits
                while True:
                    y = center - rad + (trial / (rad*2))
                    if y <= center + rad: break #If within limits

                if grid.state[x][y] != 1 and grid.state[x][y] != 4: #position not already occupied
                    break
                trial+=1

        grid.state[x][y] = 1  # Agent Code
        return [x, y]

    def reset(self, grid, is_new_epoch=False):
        self.spawn_position = self.init_agent(grid, is_new_epoch)
        self.position = self.spawn_position[:]
        self.fuel = 0

    def take_action_test(self):
        #Modify state input to required input format
        if self.parameters.baldwin:
            if self.parameters.split_learner: padded_state = self.pad_state(self.split_learner_state)
            else: padded_state = self.pad_state(self.perceived_state)
            evo_input = self.evo_net.bald.get_evo_input(padded_state)  # Hidden nodes from simulator
            if self.parameters.augmented_input:
                #if self.parameters.split_learner: evo_input = np.append(evo_input, self.split_learner_state.flatten())  # Augment input with state info
                evo_input = np.append(evo_input, self.perceived_state.flatten())  # Augment input with state info
        else: #Darwin
            if self.parameters.split_learner:
                evo_input = np.append(self.perceived_state, self.split_learner_state)
                #evo_input = np.reshape(self.perceived_state, (self.perceived_state.shape[1]))
            else:
                evo_input = np.reshape(self.perceived_state, (self.perceived_state.shape[1]))  # State input only (Strictly Darwinian approach)

        #Run test evonet and return action
        scores = [] #Probability output for five action choices
        if self.parameters.use_neat:
            if self.parameters.use_py_neat:  # Python NEAT
                scores = self.evo_net.test_net.serial_activate(evo_input)

            else: #C++ NEAT
                self.evo_net.test_net.Flush()
                self.evo_net.test_net.Input(evo_input)  # can input numpy arrays, too for some reason only np.float64 is supported
                self.evo_net.test_net.Activate()
                for i in range(5):
                    if not math.isnan(1 * self.evo_net.test_net.Output()[i]):
                        scores.append(1 * self.evo_net.test_net.Output()[i])
                    else:
                        scores.append(0)
        else: #Use keras Evo-net
            state = np.reshape(evo_input, (1, len(evo_input)))
            scores = self.evo_net.test_net.predict(state)[0]
        #if self.parameters.wheel_action and sum(scores) != 0: action = roulette_wheel(scores)
        if sum(scores) != 0 and len(set(scores)) != 1: action = np.argmax(scores)
        else: action = randint(0,4)
        #action = np.argmax(scores)

        self.action = action


    def take_action(self):
        #Modify state input to required input format
        evo_input = np.reshape(self.perceived_state, (self.perceived_state.shape[1]))  # State input only (Strictly Darwinian approach)
        self.action = self.evo_net.run_evo_net(evo_input) #Take action

    def ready_for_simulation(self, net_id):
        if self.parameters.online_learning and self.parameters.baldwin:  # Update interim model belonging to the teams[i] indexed individual in the ith sub-population
            self.evo_net.bald.update_interim_model(net_id)

    def pad_state(self, state):
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.reshape(state, (1, len(state)))
        return state

class POI:
    def __init__(self, grid, parameters, team_role_index):
        self.parameters = parameters
        self.team_role_index = team_role_index
        self.spawn_position = self.init_poi(grid)
        self.position = self.spawn_position[:]
        self.spawn_position = self.position[:]
        self.previous_actions = [0,0]
        self.observation_log = []
        self.is_observed = False

        #Periodicity
        self.visible = True
        self.period = parameters.period



    def do_periodic(self, step):
        ig = step + self.team_role_index
        ig = int(ig / self.period)
        ig = ig % 2
        self.visible = ig


    def take_action(self):
        if self.parameters.periodic_poi:
            action_choice = self.previous_actions[0]
            action_choice = (action_choice % 4) + 1
            self.previous_actions[0] = self.previous_actions[1]
            self.previous_actions[1] = action_choice

            # self.previous_actions[0] = self.previous_actions[1]
            # self.previous_actions[1] = self.previous_actions[2]
            # self.previous_actions[2] = action_choice
            #print self.previous_actions, action_choice
            return action_choice


        else:
            rand_choice = randint(1,4)
            return rand_choice


    def init_poi(self, grid, is_new_epoch=True):
        if not is_new_epoch: #If not a new epoch and intra epoch (random already initialized)
            x = self.spawn_position[0]; y = self.spawn_position[1]
            grid.state[x][y] = 2
            return [x,y]

        start = grid.observe; end = grid.state.shape[0] - grid.observe - 1
        rad = int(grid.dim_row / math.sqrt(3) / 2)
        center = int((start + end) / 2)

        if self.parameters.domain_setup != 0: #Known domain testing
            if self.parameters.domain_setup == 1:
                x = 4; y = 1 + 6 * self.team_role_index


            elif self.parameters.domain_setup == 2:
                if self.team_role_index == 0:
                    x = 14; y = 4
                elif self.team_role_index == 1:
                    x = 14; y = 11
                elif self.team_role_index == 2:
                    x = 12; y = 4
                elif self.team_role_index == 3:
                    x = 12; y = 11

            elif self.parameters.domain_setup == 3:
                if self.team_role_index == 0:
                    if random.random() < 0.5:
                        x = 2; y = 3
                    else:
                        x = 2; y = 7

                if self.team_role_index == 1:
                    if grid.poi_list[0].spawn_position[1] == 3:
                        x = 8; y = 7
                    else:
                        x = 8; y = 3












            grid.state[x][y] = 2
            return [x, y]

        if grid.poi_rand:
            while True:
                rand = random.random()
                if rand < 0.25:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif rand < 0.5:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif rand < 0.75:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad - 1)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad + 1, end)
                if grid.state[x][y] != 2: #Position not already occupied
                    break

        else: #Pre-defined starting positions
            trial = 0
            while True: #unoccuped
                k = len(grid.poi_list)
                region = trial % 4 #4 distinct regions of distribution
                access = trial / 4 #Access number
                if region == 0:
                    x = start + (access * 2) % (center - rad - 1 - start)
                    y = start + (access * 2) % (end - start)
                elif region == 1:
                    x = start + (access * 2) % (end - center - rad - 1)
                    y = end - (access * 2) % (end - start)
                elif region == 2:
                    x = end - (access * 2) % (2 * rad)
                    y = start + (access * 2) % (center - rad - 1 - start)
                else:
                    x = end - (access * 2) % (2 * rad)
                    y = end - (access * 2) % (end - center - rad - 1)
                if grid.state[x][y] != 2: #Position not already occupied
                    break
                trial += 1


        grid.state[x][y] = 2
        return [x,y]



    def reset(self, grid, is_new_epoch=False):
        self.spawn_position = self.init_poi(grid, is_new_epoch)
        self.position = self.spawn_position[:]
        self.is_observed = False
        self.observation_log = []

class Gridworld:
    def __init__(self, parameters):
        self.parameters = parameters
        self.observe = 1; self.dim_row = parameters.grid_row; self.dim_col = parameters.grid_col; self.poi_rand = parameters.poi_random; self.agent_rand = parameters.agent_random
        self.num_agents = parameters.num_agents;  self.num_poi = parameters.num_poi; self.angle_res = parameters.angle_res #Angle resolution
        self.coupling = parameters.coupling #coupling requirement
        self.obs_dist = parameters.obs_dist #Observation radius requirements

        #Resettable stuff
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, #POI = 2, #WALL = 3, AGENT_Scout = 1, AGENT_service_bot = 4
        self.init_wall() #initialize wall
        self.epoch_best_team = None

        self.poi_list = [] #List of POI objects
        for i in range(self.num_poi):
            self.poi_list.append(POI(self, parameters, i))

        self.agent_list = []
        for i in range(self.num_agents): self.agent_list.append(Agent(self, parameters, i))

    def init_wall(self):
        for i in range(self.observe):
            for x in range(self.state.shape[0]):
                self.state[x][i] = 3
                self.state[x][self.state.shape[1] - 1-i] = 3
            for y in range(self.state.shape[1]):
                self.state[i][y] = 3
                self.state[self.state.shape[0] - 1-i][y] = 3

    def new_epoch_reset(self):
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        self.epoch_best_team = None
        for poi in self.poi_list: poi.reset(self, is_new_epoch=True)
        for agent_id, agent in enumerate(self.agent_list):
            agent.reset(self, is_new_epoch=True)


    def reset(self, genome_index):
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()

        for poi in self.poi_list: poi.reset(self)
        for agent_id, agent in enumerate(self.agent_list):
            agent.reset(self)
            agent.evo_net.build_net(genome_index)


    def move(self):
        for agent in self.agent_list: #Move and agent
            action = agent.action
            #print action
            next_pos = np.copy(agent.position)
            if action == 1: next_pos[1] += 1  # Right
            elif action == 2: next_pos[0] += 1  # Down
            elif action == 3: next_pos[1] -= 1  # Left
            elif action == 4: next_pos[0] -= 1  # Up

            # Computer reward and check illegal moves
            x = next_pos[0]; y = next_pos[1]
            #TODO Make this better (Wall out of bounds due to increased speed
            if x < 0: x = 0;
            if x > self.dim_row + 1: x = self.dim_row + 1
            if y < 0: y = 0;
            if y > self.dim_col + 1: y = self.dim_col + 1

            if self.state[x][y] == 3: next_pos = np.copy(agent.position) #Reset if hit wall
            if self.state[x][y] == 1 and action != 0: next_pos = np.copy(agent.position) #Reset if other Scout Agent and action != 0

            # Update gridworld and agent position
            self.state[agent.position[0]][agent.position[1]] = 0 #Encode newly freed position in the state template
            self.state[next_pos[0]][next_pos[1]] = 1 #Encode newly occupied position in the state template
            agent.position[0] = next_pos[0]; agent.position[1] = next_pos[1] #Update new positions for the agent object



    def get_state(self, agent, state_representation = None):  # Returns a flattened array around the agent position
        if state_representation == None: state_representation = self.parameters.state_representation #If no override use choice in parameters
        if state_representation == 1:  # Angle brackets
            state = np.zeros(((360 / self.angle_res), 2))
            if self.parameters.sensor_avg:  # Average distance
                dist_poi_list = [[] for x in xrange(360 / self.angle_res)]
                dist_agent_list = [[] for x in xrange(360 / self.angle_res)]


            for poi in self.poi_list:
                if poi.visible:  # For all POI's that are still active
                    x1 = poi.position[0] - agent.position[0];
                    x2 = -1
                    y1 = poi.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    bracket = int(angle / self.angle_res)
                    dist_poi_list[bracket].append(dist / (2.0 * self.dim_col))


            for other_agent in self.agent_list:
                if other_agent != agent:  # FOR ALL AGENTS MINUS MYSELF
                    x1 = other_agent.position[0] - agent.position[0];
                    x2 = -1
                    y1 = other_agent.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    bracket = int(angle / self.angle_res)
                    dist_agent_list[bracket].append(dist / (2.0 * self.dim_col))



            if self.parameters.sensor_avg:
                for bracket in range(len(dist_agent_list)):
                    try: state[bracket][0] = sum(dist_poi_list[bracket]) / len(dist_poi_list[bracket])  # Encode average POI distance
                    except: None
                    try: state[bracket][1] = sum(dist_agent_list[bracket]) / len(dist_agent_list[bracket])  # Encode average agent distance
                    except: None


            # if self.parameters.sensor_noise != 0:
            #     for bracket in range(len(dist_agent_list)):
            #         for i in range(6):
            #             state[bracket][i] += (random.random() - 0.5) * 2 * self.parameters.sensor_noise * state[bracket][i]


            state = np.reshape(state, (1, 360 / self.angle_res * 2))  # Flatten array

        if state_representation == 2:  # List agent/POI representation fully obserbavle
            state = np.zeros(self.num_agents * 2 + self.num_poi * 2)
            for id, poi in enumerate(self.poi_list):
                if True: #not poi.is_observed:  # For all POI's that are still active
                    x1 = poi.position[0] - agent.position[0];
                    x2 = -1
                    y1 = poi.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    state[2 * id] = angle / 360.0
                    state[2 * id + 1] = dist / (2.0 * self.dim_col)

            for id, other_agent in enumerate(self.agent_list):
                if other_agent != agent:  # FOR ALL AGENTS MINUS MYSELF
                    x1 = other_agent.position[0] - agent.position[0];
                    x2 = -1
                    y1 = other_agent.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    state[2 * self.num_poi + 2 * id] = angle /360.0
                    state[2 * self.num_poi + 2 * id + 1] = dist / (2.0 * self.dim_col)
            state = np.reshape(state, (1, self.num_agents * 2 + self.num_poi * 2))  # Flatten array


        return state

    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two agents relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  #dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    def update_poi_observations(self):
        # Check for credit assignment
        for poi in self.poi_list:  # POI COUPLED
            test_gate = True
            if self.parameters.test_observation and not poi.visible: test_gate = False



            if test_gate:
                #Agents
                soft_stat = []
                for agent_id, agent in enumerate(self.agent_list): #Find all scouts within range
                    if abs(poi.position[0] - agent.position[0]) <= self.obs_dist*2 and abs(poi.position[1] - agent.position[1]) <= self.obs_dist*2:  # and self.goal_complete[poi_id] == False:
                        soft_stat.append(agent_id)
                if len(soft_stat) >= self.coupling:  # If coupling requirement is met
                    for ag_id in soft_stat: poi.observation_log.append(ag_id)
                    poi.is_observed = True


    def save_pop(self):
        if self.parameters.use_deap:
            self.agent_list[0].evo_net.save_population()

    def check_goal_complete(self):
        is_complete = True
        for poi in self.poi_list:
            is_complete *= poi.is_observed
        return is_complete

    def get_reward(self):
        global_reward = 0.0 #Global reward obtained
        for poi in self.poi_list: global_reward += poi.is_observed
        global_reward /= self.parameters.num_poi
        rewards = np.zeros(self.parameters.num_agents)


        if self.parameters.D_reward: #Difference reward scheme
            for poi in self.poi_list:
                if poi.is_observed: #If POI observed
                    no_reward = False

                    # Check if over-observed (Scout)
                    no_reward = False
                    all_agents = [a for a in poi.observation_log]
                    unique_agents = set(all_agents)
                    if len(unique_agents) > self.parameters.coupling:  # Only if it's observed by exactly the numbers needed
                        no_reward = True;

                    # Disburse rewards
                    if not no_reward:
                        for agent_id in unique_agents:
                            rewards[agent_id] += 1.0 / self.parameters.num_poi  # Reward the first group of agents to get there

        else: #G reward
            rewards = np.zeros(self.parameters.num_agents)
            rewards += global_reward  # Global reward scheme
        return rewards, global_reward

    def poi_move(self):
        for poi in self.poi_list:
            if random.random() < self.parameters.poi_motion_probability:
                action = poi.take_action()
                next_pos = np.copy(poi.position)
                if action == 1:
                    next_pos[1] += 1  # Right
                elif action == 2:
                    next_pos[0] += 1  # Down
                elif action == 3:
                    next_pos[1] -= 1  # Left
                elif action == 4:
                    next_pos[0] -= 1  # Up

                # Computer reward and check illegal moves
                x = next_pos[0];
                y = next_pos[1]
                if self.state[x][y] == 3: next_pos = np.copy(poi.position)  # Reset if hit wall

                # Update gridworld and agent position
                self.state[poi.position[0]][poi.position[1]] = 0  # Encode newly freed position in the state template
                self.state[next_pos[0]][next_pos[1]] = 2  # Encode newly occupied position in the state template
                poi.position[0] = next_pos[0];
                poi.position[1] = next_pos[1]  # Update new positions for the agent object

    def save_best_team(self, generation):
        if not os.path.exists('Best_team'):
            os.makedirs('Best_team')
        for i, member_id in enumerate(self.epoch_best_team):
            if i < self.num_agents_scout: #For Scout agents
                 self.agent_list_scout[i].evo_net.net_list[member_id].Save('Best_team/' + 'Scout_' + str(i))
            else:
                index = i - self.num_agents_scout
                self.agent_list_service_bot[index].evo_net.net_list[member_id].Save('Best_team/' + 'Service_bot_' + str(index))
        saved_gen = np.zeros(1) + generation
        np.savetxt('Best_team/save_gen', saved_gen)

    def load_test_policies(self):
        for i, agent in enumerate(self.agent_list_scout):
            is_success = agent.evo_net.test_net.Load('Best_team/' + 'Scout_' + str(i))
            if is_success != True:
                print 'Trained Netword Loading failed'
                sys.exit()
        for i, agent in enumerate(self.agent_list_service_bot):
            is_success = agent.evo_net.test_net.Load('Best_team/' + 'Service_bot_' + str(i))
            if is_success != True:
                print 'Trained Netword Loading failed'
                sys.exit()

class statistics(): #Tracker
    def __init__(self, parameters):
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.avg_mpc = 0; self.tr_avg_mpc = []; self.mpc_std = []; self.tr_mpc_std = []
        if parameters.is_memoried:
            self.file_save = 'Memory_net.csv'
        else:
            self.file_save = 'Normal_net.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 200:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            self.save_csv(generation)

    def add_mpc(self, gridworld, parameters):
        all_mpc = np.zeros(parameters.num_agents_scout + parameters.num_agents_service_bot)
        for id, agent in enumerate(gridworld.agent_list): all_mpc[id] = agent.evo_net.delta_mpc
        self.avg_mpc = np.average(all_mpc) #Average mpc
        self.mpc_std = np.std(all_mpc)


    def save_csv(self, generation):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(self.file_save, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class keras_Population(): #Keras population
    def __init__(self, input_size, hidden_nodes, output, population_size, elite_fraction = 0.2):
        self.population_size = population_size
        self.elite_fraction = int(elite_fraction * population_size)
        self.net_pop = [] #List of networks
        for i in range(population_size): self.net_pop.append(self.init_net(input_size, hidden_nodes, output))
        self.pop_handle = np.zeros(population_size * 2, dtype=np.float64) #COntrols the indexing to the net population (net_pop) and fitness values
        self.pop_handle = np.reshape(self.pop_handle, (population_size, 2))
        for x in range(population_size):  self.pop_handle[x][0] = x #Initializing our net population indexing
        self.longest_survivor = 0
        self.best_net_index = 0 #Current index of the best net


    def init_net(self, input_size, hidden_nodes, output):
        model = Sequential()
        model.add(Dense(hidden_nodes, input_dim=input_size, init='he_uniform'))
        model.add(Activation('sigmoid'))
        model.add(Dense(output))
        model.add(Activation('softmax'))
        model.compile(loss='mean_absolute_error', optimizer='Nadam')
        return model

    def Epoch(self):
        self.pop_handle = self.pop_handle[self.pop_handle[:, 1].argsort()]  ##Ranked on fitness (reverse of weakness) s.t. 0 index is the best
        if int(self.pop_handle[0][0]) == self.best_net_index: self.longest_survivor += 1 #Check if the leader candidate is the same one
        else:
            self.longest_survivor = 0; self.best_net_index = int(self.pop_handle[0][0])
        self.best_net_index = self.pop_handle[0][0] #Update the leader candidate
        for x in range(self.elite_fraction, self.population_size): #Mutate to renew population
            many = randint(1,5); much = randint(1,10)
            if (randint(1,100) == 91):
                many = randint(1,10); much = randint(1,100)
            self.mutate(self.net_pop[int(self.pop_handle[x][0])], self.net_pop[int(self.pop_handle[x][0])], many, much) #Mutate same model in and out

    def mutate(self, model_in, model_out, many_strength=1, much_strength=1):
        # NOTE: Takes in_num file, mutates it and saves as out_num file, many_strength denotes how many mutation while
        # much_strength controls how strong each mutation is

        w = model_in.get_weights()
        for many in range(many_strength):  # Number of mutations
            i = randint(0, len(w) - 1)
            if len(w[i].shape) == 1:  # Bias
                j = randint(0, len(w[i]) - 1)
                w[i][j] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
                # if (randint(1, 100) == 5): #SUPER MUTATE
                #     w[i][j] += np.random.normal(-1 * much_strength, 1 * much_strength)
            else:  # Bias
                j = randint(0, len(w[i]) - 1)
                k = randint(0, len(w[i][j]) - 1)
                w[i][j][k] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
                # if (randint(1, 100) == 5):  # SUPER MUTATE
                #     w[i][j][k] += np.random.normal(-1 * much_strength, 1 * much_strength)
        model_out.set_weights(w)  # Save weights

def vizualize_trajectory(filename = 'trajectory.csv'):
    print
    import Tkinter as tk
    # from visualizer_math import *
    import csv

    class Path:

        def __init__(
                self, position_arr=[],
                color="black",
                style="solid"  # or "dashed" #or "circle"
        ):
            self.position_arr = position_arr
            self.color = color
            self.style = style

        def get_path_slice(self, start=0, end=1):  # inclusive
            return self.position_arr[start:end + 1]

        def get_path_position(self, time):
            return self.position_arr[time]

    class Visualizer:

        def __init__(self):
            self.path_arr = []
            self.time_index = 0

            self.root = None
            self.canvas = None
            self.max_time_index = 0

            # options (all applied after drawing transform)
            self.circle_radius = 10
            self.line_width = 2
            self.line_dash = (5, 5)
            self.arrow_shape = (8, 10, 3)
            self.transform = [[0, 20, 0],
                              [20, 0, 0],
                              [0, 0, 1]]  # stored in rows
            self.max_x = 200  # auto adjust as paths are entered
            self.max_y = 200  # auto adjust as paths are entered
            self.make_grid = True
            self.grid_x_spacing = 20
            self.grid_y_spacing = 20
            self.grid_color = "white"
            self.grid_dash = ()  # use () for no dash

        def transform_points(self, point_arr, transform):
            transformed_point_arr = []

            for point in point_arr:
                transformed_point_arr.append([
                    point[0] * transform[0][0] + point[1] * transform[0][1] + transform[0][2],
                    point[0] * transform[1][0] + point[1] * transform[1][1] + transform[1][2],
                ])
            return transformed_point_arr

        def create_path(self, position_arr, color="black", style="solid"):
            position_arr = self.transform_points(position_arr, self.transform)
            for point in position_arr:
                self.max_x = max(point[0], self.max_x)
                self.max_y = max(point[1], self.max_y)
            self.path_arr.append(Path(position_arr, color, style))
            self.max_time_index = len(position_arr) - 1

        def run(self):
            self.root = tk.Tk()
            self.root.bind("<Return>", self.increment_time)
            self.root.bind("<BackSpace>", self.decrement_time)

            self.canvas = tk.Canvas(self.root, width=self.max_x, height=self.max_y)
            self.canvas.pack()
            self.update_canvas()

            self.root.mainloop()

        def draw_path(self, path, start=0, end=1):  # inclusive
            if path.style == "circle":
                path_position = path.get_path_position(end)
                self.canvas.create_oval(
                    path_position[0] - self.circle_radius,
                    path_position[1] - self.circle_radius,
                    path_position[0] + self.circle_radius,
                    path_position[1] + self.circle_radius,
                    fill=path.color,
                    outline=path.color
                )
            elif end != 0:
                path_slice = path.get_path_slice(start, end)

                if path.style == "solid":
                    self.canvas.create_line(
                        path_slice,
                        fill=path.color,
                        width=self.line_width,
                        arrowshape=self.arrow_shape,
                        arrow=tk.LAST
                    )
                elif path.style == "dashed":
                    self.canvas.create_line(
                        path_slice,
                        fill=path.color,
                        dash=self.line_dash,
                        width=self.line_width,
                        arrowshape=self.arrow_shape,
                        arrow=tk.LAST
                    )

        def update_canvas(self):
            self.canvas.delete(tk.ALL)

            if self.make_grid:
                self.draw_grid()

            for path in self.path_arr:
                self.draw_path(path, start=0, end=self.time_index)

            self.root.update_idletasks()

        def draw_grid(self):
            num_horizontal = int(self.max_y / self.grid_y_spacing)
            num_vertical = int(self.max_x / self.grid_x_spacing)

            for horizontal_index in range(num_horizontal):
                self.canvas.create_line(
                    0,
                    horizontal_index * self.grid_y_spacing,
                    self.max_x,
                    horizontal_index * self.grid_y_spacing,
                    fill=self.grid_color,
                    dash=self.grid_dash
                )

            for vertical_index in range(num_vertical):
                self.canvas.create_line(
                    vertical_index * self.grid_x_spacing,
                    0,
                    vertical_index * self.grid_x_spacing,
                    self.max_y,
                    fill=self.grid_color,
                    dash=self.grid_dash
                )

        def increment_time(self, event):
            if self.time_index < self.max_time_index:
                self.time_index += 1
                self.update_canvas()

        def decrement_time(self, event):
            if self.time_index > 0:
                self.time_index -= 1
                self.update_canvas()

    v = Visualizer()

    #import pandas as pd
    datafile = filename
    data = list(csv.reader(open(datafile)))

    macros = data[0] #Comments about number of scouts, service bots and POIs
    macros = [float(a) for a in macros]
    print 'Scouts: ', macros[0]
    print 'Service-bots: ', macros[1]
    print 'POIs: ', macros[2]
    data.pop(0)

    data = np.array(data)
    for agent_index in range(len(data[0]) / 2):
        position_arr = []
        for time in range(len(data)):
            position_arr.append([
                float(data[time][agent_index * 2]),
                float(data[time][agent_index * 2 + 1])
            ])
        if agent_index < int(macros[0]): #Scouts
            v.create_path(position_arr, 'blue', style='dashed')
            v.create_path(position_arr, 'blue', 'circle')
        elif agent_index < int(macros[1])+int(macros[0]): #Service-bots
            v.create_path(position_arr, 'green', 'solid')
            v.create_path(position_arr, 'green', 'circle')
        else:
            v.create_path(position_arr, 'red', 'circle')

    # v.create_path([[10.,10.],[30.,30.],[21.,45.]],'blue',style ='dashed')
    # v.create_path([[10.,10.],[10.,30.],[40.,40.]],'red', 'solid')
    # v.create_path([[10.,10.],[10.,30.],[40.,40.]],'red','circle')
    v.run()

def dispGrid(gridworld, state = None, full=True, agent_id = None):


    if state == None: #Given agentq
        if full:
            st = np.copy(gridworld.state)
        else:
            x_beg = gridworld.agent_pos[agent_id][0] - gridworld.observe
            y_beg = gridworld.agent_pos[agent_id][1] - gridworld.observe
            x_end = gridworld.agent_pos[agent_id][0] + gridworld.observe + 1
            y_end = gridworld.agent_pos[agent_id][1] + gridworld.observe + 1
            st = np.copy(gridworld.state)
            st = st[x_beg:x_end,:]
            st = st[:,y_beg:y_end]
    else:
        st = []
        print len(state)
        row_leng = int(math.sqrt(len(state)))
        for i in range(row_leng):
            ig = []
            for j in range(row_leng):
                ig.append(state[i*row_leng + j])
            st.append(ig)

    grid = [["-" for i in range(len(st))] for i in range(len(st))]
    grid[0][0] = "o"
    for i in range(len(st)):
        for j in range(len(st)):
            if st[i][j] == 2:
                grid[i][j] = '$'
            if st[i][j] == 1:
                grid[i][j] = '*'
            if st[i][j] == 3:
                grid[i][j] = '#'
            if st[i][j] == 4:
                grid[i][j] = '@'
    for row in grid:
        for e in row:
            print e,
        print '\t'

def roulette_wheel(scores):
    scores = scores / np.sum(scores)  # Normalize
    rand = random.random()
    counter = 0
    for i in range(len(scores)):
        counter += scores[i]
        if rand < counter:
            return i




#BACKUPS

def init_nn(input_size, hidden_nodes, middle_layer = False, weights = 0):
    model = Sequential()


    if middle_layer:
        model.add(Dense(hidden_nodes, input_dim=input_size, weights=weights, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    else:
        model.add(Dense(hidden_nodes, input_dim=input_size, init='he_uniform', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(LeakyReLU(alpha=.2))
    #model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.1))
    #model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if not middle_layer:
        model.add(Dense(input_size, init= 'he_uniform')) #Output of the prediction module
    model.compile(loss='mse', optimizer=sgd)

    # if pretrain: #Autoencoder pretraining
    #     model.fit(train_x, train_x, nb_epoch=50, batch_size=32, shuffle=True, validation_data=(valid_x, valid_x),
    #                     verbose=1)
    return model

def init_rnn(gridworld, hidden_nodes, angled_repr, angle_res, sim_all, hist_len = 3, design = 1):
    model = Sequential()
    if angled_repr:
        sa_sp = (360/angle_res) * 4
    else:
        sa_sp = (pow(gridworld.observe * 2 + 1,2)*4) #BIT ENCODING
    if design == 1:
        model.add(LSTM(hidden_nodes, init= 'he_uniform', return_sequences=False, input_shape=(hist_len, sa_sp), inner_init='orthogonal', forget_bias_init='one', inner_activation='sigmoid'))#, activation='sigmoid', inner_activation='hard_sigmoid'))
    elif design == 2:
        model.add(SimpleRNN(hidden_nodes, init='he_uniform', input_shape=(hist_len, sa_sp), inner_init='orthogonal'))
    elif design == 3:
        model.add(GRU(hidden_nodes, init='he_uniform', consume_less= 'cpu',  input_shape=(hist_len, sa_sp),inner_init='orthogonal'))
    #model.add(Dropout(0.1))
    #model.add(LeakyReLU(alpha=.2))
    model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.add(Activation('sigmoid'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(1, init= 'he_uniform'))
    model.compile(loss='mse', optimizer='Nadam')
    return model

def save_model(model):
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    model.save_weights('shaw_2/model_weights.h5',overwrite=True)

def load_model_architecture(seed='Models/architecture.json'):  # Get model architecture
    import yaml
    from keras.models import model_from_yaml
    with open('Models/architecture.yaml', 'r') as f:
        yaml_string = yaml.load(f)
    model_arch = model_from_yaml(yaml_string)
    return model_arch

def save_model_architecture(qmodel, foldername = '/Models/'):
    import os, yaml
    import json
    from keras.models import model_from_json
    from keras.models import model_from_yaml
    #Create folder to store all networks if not present
    filename = os.getcwd() + foldername
    if not os.path.exists(os.path.dirname(filename)):
        try: os.makedirs(os.path.dirname(filename))
        except: 1+1
    yaml_string = qmodel.to_yaml()
    output_stream = open("Models/architecture.yaml", "w")
    yaml.dump(yaml_string, output_stream)#, default_flow_style=False)

def load_model(foldername = 'Models/'):
    import copy
    q_model = []
    model_arch = load_model_architecture()
    for i in range(5):
        ig = copy.deepcopy(model_arch)
        ig.load_weights(foldername + 'model_weights_' + str(i) + '.h5')
        q_model.append(ig)
        #q_model[i].compile(loss='mse', optimizer='rmsprop')
    return q_model

def save_qmodel(q_model, foldername = '/Models/'):
    import os
    #Create folder to store all networks if not present
    filename = os.getcwd() + foldername
    if not os.path.exists(os.path.dirname(filename)):
        try: os.makedirs(os.path.dirname(filename))
        except: 1+1
    #Save weights
    for i in range(len(q_model)):
        q_model[i].save_weights('Models/model_weights_' + str(i) + '.h5', overwrite=True)

def test_nets():
    from fann2 import libfann
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import LSTM, GRU, SimpleRNN
    from keras.layers.advanced_activations import LeakyReLU, PReLU, SReLU
    from keras.models import model_from_json
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD, Nadam
    test_x = np.arange(50)
    model = Sequential()
    model.add(Dense(50, input_dim=50, init='he_uniform'))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(1, init= 'he_uniform'))
    model.compile(loss='mse', optimizer='Nadam')
    ann = libfann.neural_net()
    ann.create_standard_array([3, 50, 50, 1])
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

    curtime = time.time()
    for i in range (1000000):
        ann.run(test_x)
    elapsed = time.time() - curtime
    print elapsed

    test_x = np.reshape(test_x, (1,50))
    curtime = time.time()
    for i in range (1000000):
        model.predict(test_x)
    elapsed = time.time() - curtime
    print elapsed

def bck_angled_state(self, agent_id, sensor_avg):
    state = np.zeros(((360 / self.angle_res), 4))
    if sensor_avg:  # Average distance
        dist_poi_list = [[] for x in xrange(360 / self.angle_res)]
        dist_agent_list = [[] for x in xrange(360 / self.angle_res)]

    for id in range(self.num_poi):
        if self.goal_complete[id] == False:  # For all POI's that are still active
            x1 = self.poi_pos[id][0] - self.agent_pos[agent_id][0];
            x2 = 1
            y1 = self.poi_pos[id][1] - self.agent_pos[agent_id][1];
            y2 = 0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            bracket = int(angle / self.angle_res)
            state[bracket][0] += 1.0 / self.num_poi  # Add POIs
            if sensor_avg:
                dist_poi_list[bracket].append(dist / (2.0 * self.dim_col))
            else:  # Min distance
                if state[bracket][1] > dist / (2.0 * self.dim_col) or state[bracket][
                    1] == 0:  # Update min distance from POI
                    state[bracket][1] = dist / (2.0 * self.dim_col)

    for id in range(self.num_agents):
        if id != agent_id:  # FOR ALL AGENTS MINUS MYSELF
            x1 = self.agent_pos[id][0] - self.agent_pos[agent_id][0];
            x2 = 1
            y1 = self.agent_pos[id][1] - self.agent_pos[agent_id][1];
            y2 = 0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            bracket = int(angle / self.angle_res)
            state[bracket][2] += 1.0 / (self.num_agents - 1)  # Add agent
            if sensor_avg:
                dist_agent_list[bracket].append(dist / (2.0 * self.dim_col))
            else:  # Min distance
                if state[bracket][3] > dist / (2.0 * self.dim_col) or state[bracket][
                    3] == 0:  # Update min distance from other agent
                    state[bracket][3] = dist / (2.0 * self.dim_col)

    if sensor_avg:
        for bracket in range(len(dist_agent_list)):

            try:
                state[bracket][1] = sum(dist_poi_list[bracket]) / len(
                    dist_poi_list[bracket])  # Encode average POI distance
            except:
                None
            try:
                state[bracket][3] = sum(dist_agent_list[bracket]) / len(
                    dist_agent_list[bracket])  # Encode average POI distance
            except:
                None

    state = np.reshape(state, (1, 360 / self.angle_res * 4))  # Flatten array
    return state

def novelty(weak_matrix, archive, k = 10):
    import bottleneck
    #Handle early gens with archive size less that 10
    if (len(archive) < k):
        k = len(archive)

    novel_matrix = np.zeros(len(archive))
    for i in range(len(archive)):
        novel_matrix[i] = np.sum(np.square(weak_matrix - archive[i]))

    #k-nearest neighbour algorithm
    k_neigh = bottleneck.partsort(novel_matrix, k)[:k] #Returns a subarray of k smallest novelty scores

    #Return novelty score as the average Euclidean distance (behavior space) between its k-nearest neighbours
    return np.sum(k_neigh)/k

def import_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    from keras.models import model_from_json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def rec_weakness(setpoints, initial_state, model, n_prev=7, novelty = False, test = False): #Calculates weakness (anti fitness) of RECCURRENT models
    weakness = np.zeros(19)
    input = np.reshape(train_data[0:n_prev], (1, n_prev, 21))  #First training example in its entirety

    for example in range(len(train_data)-n_prev):#For all training examples
        model_out = model.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(19):
            weakness[index] += math.fabs(model_out[0][index] - train_data[example+n_prev][index])#Time variant simulation
        #Fill in new input data
        for k in range(len(model_out)): #Modify the last slot
            input[0][0][k] = model_out[0][k]
            input[0][0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][0][19] = train_data[example+n_prev][19]
        input[0][0][20] = train_data[example+n_prev][20]
        input = np.roll(input, -1, axis=1)  # Track back everything one step and move last one to the last row
    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(train_data)-n_prev)
    else:
        return np.sum(np.square(weakness))

def ff_weakness(setpoints, initial_state, simulator, model, novelty = False, test = False, actuator_noise = 0, sensor_noise = 0, sensor_failure = None, actuator_failure = None): #Calculates weakness (anti fitness) of FEED-FORWARD models

    indices = [11,15]
    weakness = np.zeros(len(indices))
    # input = np.append(initial_state, setpoints[0])
    # input = np.reshape(input, (1,23))
    input = np.copy(initial_state)
    input[0][19] = setpoints[0][0]
    input[0][20] = setpoints[0][1]


    for example in range(len(setpoints)-1):#For all training examples
        #Add noise to the state input to the controller
        noise_input = np.copy(input)
        if sensor_noise != 0: #Add sensor noise
            for i in range(19):
                std = sensor_noise * abs(noise_input[0][i]) / 100.0
                if std != 0:
                    noise_input[0][i] +=  np.random.normal(0, std/2.0)

        if sensor_failure != None: #Failed sensor outputs 0 regardless
            for i in sensor_failure:
                noise_input[0][i] = 0

        # Get the controller output
        control_out = model.predict(noise_input)

        #Add actuator noise (controls)
        if actuator_noise != 0:
            for i in range(len(control_out[0])):
                std = actuator_noise * abs(control_out[0][i]) / 100.0
                if std != 0:
                    control_out[0][i] +=  np.random.normal(0, std/2.0)

        if actuator_failure != None: #Failed actuator outputs 0 regardless
            for i in actuator_failure:
                control_out[0][i] = 0

        # Fill in the controls
        input[0][19] = control_out[0][0]
        input[0][20] = control_out[0][1]

        #Use the simulator to get the next state
        model_out = simulator.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(len(indices)):
            weakness[index] += math.fabs(model_out[0][indices[index]] - setpoints[example][index])#Time variant simulation

        #Fill in new input data
        for k in range(len(model_out[0])):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]

        #Fill in next setpoints
        input[0][19] = setpoints[example+1][0]
        input[0][20] = setpoints[example+1][1]

    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(setpoints)-1)
    else:
        return np.sum(np.square(weakness))

def get_first_state(self, agent_id, use_rnn, sensor_avg,
                    state_representation):  # Get first state, action input to the q_net
    if not use_rnn:  # Normal NN
        st = self.get_state(agent_id, sensor_avg, state_representation)
        return st

    rnn_state = []
    st = self.get_state(agent_id)
    for time in range(3):
        rnn_state.append(st)
    rnn_state = np.array(rnn_state)
    rnn_state = np.reshape(rnn_state, (1, rnn_state.shape[0], rnn_state.shape[2]))
    return rnn_state


def referesh_state(self, current_state, agent_id, use_rnn):
    st = self.get_state(agent_id)
    if use_rnn:
        new_state = np.roll(current_state, -1, axis=1)
        new_state[0][2] = st
        return new_state
    else:
        return st

def pstats():
    import pstats
    p = pstats.Stats('output.file')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('cumulative').print_stats(50)
