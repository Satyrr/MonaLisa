import numpy as np
import numpy.random as rnd
import torch
import signal
import multiprocessing as mp
import time

class ImageEvaluator(object):

    ## This class is used to evaluate images.
    ## It holds target image which is being reconstructed. 
    ##
    ## fixed_img is an image constructed
    ## with already constructed shapes that are not mutable anymore. 
    ## It allows to save computing time - if image has 200 shapes, and
    ## we mutate only last onem, then we don't have to
    ## redraw leading 199 shapes each time.
    ##
    ## shape_drawer is a an object with draw function which takes chromosome
    ## and returns constructed numpy matrix (image).

    def __init__(self, target_image, shape_drawer, cuda=False):

        self.cuda = cuda
        if self.cuda:
            self.target_img = torch.from_numpy(target_image).cuda().float()
        else:
            self.target_img = target_image

        self.shape_drawer = shape_drawer

        # initial empty fixed image 
        self.fixed_img = shape_drawer.draw(np.array([]))

        self.pool = mp.Pool(initializer=self.__pool_initializer)
    
    def __pool_initializer(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def objective(self, population):
        ppl_size = population.genotypes.shape[0]
        population.objectives = np.zeros(ppl_size)

        target_img = self.target_img
        fixed_img = self.fixed_img

        # draw shapes
        args = [(population.genotypes[i], fixed_img.copy(), self.shape_drawer.draw) 
                                        for i in range(ppl_size)]
        ims = self.pool.map(shape_drawer_wrapper, args)

        # evaluate drawings of shapes
        if self.cuda:
            ims = torch.from_numpy(np.asarray(ims)).cuda().float()
            diff = (ims - target_img).view(ims.size(0),-1)
            population.objectives = (diff * diff).sum(dim=1).sqrt().cpu().numpy()
        else:
            ims = np.asarray(ims).astype(float)
            diff = (ims - target_img).reshape(ims.shape[0],-1)
            population.objectives = np.sqrt((diff * diff).sum(axis=1))
        
        return population.objectives

    def update_fixed_chromosome(self, chromosome):
        ## new fixed chrosome
        self.fixed_img = self.shape_drawer.draw(chromosome)

def shape_drawer_wrapper(args):
    chromosome, fixed_img, draw_fun = args
    return draw_fun(chromosome, fixed_img)

class Population(object):
    
    ## Population has one fixed genotype which
    ## encode leading shapes.
    ## Genotypes encode trailing shapes 
    ## for each individual.

    def __init__(self):
        self.fixed_genotype = None
        self.genotypes = None
        self.sigmas = None
        self.objectives = None
    
    def best(self):
        return np.hstack([
            self.fixed_genotype,
            self.genotypes[0]
        ])

    def get_chromosome(self, idx):
        return np.hstack([
            self.fixed_genotype,
            self.genotypes[idx]
        ])

    def get_copy(self):
        copy = Population()
        copy.fixed_genotype = self.fixed_genotype.copy()
        copy.genotypes = self.genotypes.copy()
        copy.sigmas = self.sigmas.copy()
        copy.objectives = self.objectives.copy()

        return copy

class MonaLisaES(object):

    def __init__(self, target_img, img_drawer, shape_encoding_len, cuda=False, 
                max_shapes=200, population=30, offspring=30, parents=1, 
                sigma=1.0, tau=1.0, tau_bias=1.0, passive_iters_limit=15,
                starting_population=None, custom_mutate_operator=None,
                log_objects=None):

        self.evaluator = ImageEvaluator(target_img, img_drawer, cuda=True)

        # Shapes config
        self.shape_encoding_len = shape_encoding_len
        self.max_shapes = max_shapes
        self.current_shapes = 1

        # Algorithm config
        self.population = population
        self.offspring = offspring
        self.parents = parents
        self.sigma = sigma
        self.tau = tau
        self.tau_bias = tau_bias
        self.passive_iters_limit = passive_iters_limit
        self.current_population = starting_population
        self.custom_mutate_operator = custom_mutate_operator

        # Misc. config
        self.log_objects = log_objects

    def evolve(self, iters):
        
        if not self.current_population:
            # init new population
            self.current_population = self.init_population()
            self.best_solution = np.array([])
        else:
            # continue evolving old population
            self.best_solution = self.current_population.best()
            self.current_shapes = \
                self.best_solution.shape[0]/self.shape_encoding_len
            
        mutable_len = self.shape_encoding_len
        fixed_chromosome = self.best_solution[:-mutable_len]
        self.evaluator.update_fixed_chromosome(fixed_chromosome)
        self.evaluator.objective(self.current_population)

        self.best_solution_objective = self.current_population.objectives[0]

        self.passive_iters = 0
        best_updated = False
        try:
            for iter_ in range(iters):

                # selection
                parent_indices = self.selection()
                
                # crossover
                children_population = self.crossover(parent_indices)

                # mutation
                self.mutation(children_population)
                if self.custom_mutate_operator:
                    self.custom_mutate_operator(children_population)

                # clip to domains
                self.normalize(children_population)
                
                # evaluate children
                self.evaluator.objective(children_population)
                
                # reproduction
                self.reproduction(children_population)
                
                self.passive_iters += 1

                # best update
                if self.current_population.objectives[0] < self.best_solution_objective:
                    self.best_update()
                    
                    self.passive_iters = 0
                    best_updated = True

                # new shapes
                if self.passive_iters > self.passive_iters_limit and \
                    self.current_shapes < self.max_shapes and \
                    best_updated == True: 

                    self.current_shapes += 1
                    self.next_shape()
                    self.evaluator.objective(self.current_population)

                    self.passive_iters = 0
                    best_updated = False
                
                if self.passive_iters > 400:
                    if self.current_shapes == self.max_shapes:
                        break

                    self.reinitalize_mutable_shape()
                    
                    self.passive_iters = 0
                    best_updated = False

                # logging
                for log_obj in self.log_objects:
                    log_obj.log(iter_, self)

        except KeyboardInterrupt:
            pass
        finally:
            self.evaluator.pool.terminate()
            self.evaluator.pool.join()

    def best_update(self):
        self.best_solution = self.current_population.best()
        self.best_solution_objective = self.current_population.objectives[0]

    def reinitalize_mutable_shape(self):
        N = self.population
        d = self.shape_encoding_len

        ppl = self.current_population
        ppl.genotypes = rnd.uniform(0.0, 10.0, size=(N, d)).astype(float)
        ppl.sigmas = self.sigma * np.ones((N, d))

    def next_shape(self):
        fixed_chromosome = self.best_solution[:]
        self.evaluator.update_fixed_chromosome(fixed_chromosome)
        self.current_population.fixed_genotype = fixed_chromosome

        N = self.current_population.genotypes.shape[0]
        d = self.shape_encoding_len

        self.current_population.genotypes = \
            rnd.uniform(0, 10.0, size=(N, d)).astype(float)

        self.current_population.sigmas = \
            self.sigma*np.ones((N, d))

    def init_population(self):
        N = self.population
        d = self.shape_encoding_len

        ppl = Population()
        ppl.fixed_genotype = np.array([])
        ppl.genotypes = rnd.uniform(0.0, 10.0, size=(N, d)).astype(float)
        ppl.sigmas = self.sigma * np.ones((N, d))

        return ppl

    def selection(self):
        objective_values = self.current_population.objectives
        ppl_size = objective_values.shape[0]

        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = 1.0 / ppl_size * np.ones(ppl_size)
        
        size = (self.offspring, self.parents)
        parent_indices = np.random.choice(ppl_size, size, True, fitness_values)

        return parent_indices.astype(np.int64)

    def crossover(self, parents):
        children = Population()
        current_population = self.current_population
        d = current_population.genotypes.shape[1]

        children.genotypes = np.empty((self.offspring, d))
        children.sigmas = np.empty((self.offspring, d))

        for i in range(self.offspring):
            children.genotypes[i, :] = \
                current_population.genotypes[parents[i]].mean(axis=0)
        
        for i in range(self.offspring):
            children.sigmas[i, :] = \
                current_population.sigmas[parents[i]].mean(axis=0)

        return children

    def mutation(self, population):
        d = population.genotypes.shape[1]

        population.sigmas = population.sigmas * np.exp(
            self.tau * np.random.randn(self.offspring, d) + \
            self.tau_bias * np.random.randn(self.offspring, 1))
            
        for i in range(self.offspring):
            population.genotypes[i] += population.sigmas[i] * np.random.randn(d)

    def normalize(self, population):
        population.genotypes[population.genotypes > 10.0] = 10.0
        population.genotypes[population.genotypes < 0.0] = 0.0

    def reproduction(self, new_population):
        current_population = self.current_population

        current_population.objectives = \
            np.hstack([current_population.objectives, new_population.objectives])
        current_population.genotypes = \
            np.vstack([current_population.genotypes, new_population.genotypes])
        current_population.sigmas = \
            np.vstack([current_population.sigmas, new_population.sigmas])

        I = np.argsort(current_population.objectives)

        current_population.genotypes = \
            current_population.genotypes[I[:self.population]]
        current_population.objectives = \
            current_population.objectives[I[:self.population]]
        current_population.sigmas = \
            current_population.sigmas[I[:self.population]]