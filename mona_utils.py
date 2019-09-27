import numpy as np
import matplotlib.pyplot as plt

import pickle
import datetime
import time

import cv2

    
def load_file(file_name):
    f = pickle.load(open(file_name, 'rb'))
    return f

class BasicLogger(object):
    
    def __init__(self, iters, log_freq, max_chrom_len, img_drawer):
        self.log_freq = log_freq
        self.log_objective_stats = np.zeros((iters, 4))
        self.log_best_solutions = []
        self.log_best_sigmas = np.zeros((iters, max_chrom_len))
        self.img_drawer = img_drawer
        self.log_time = 0.0
        self.evolving_start_time = 0.0

    def start_timer(self):
        self.evolving_start_time = time.time()
    
    def log(self, iter_, evolver):

        population = evolver.current_population
        self.log_objective_stats[iter_] = [population.objectives.min(),
                                          population.objectives.max(),
                                          population.objectives.mean(),
                                          population.objectives.std()]
        
        d = evolver.current_population.sigmas[0].shape[0]
        self.log_best_sigmas[iter_,:d] = evolver.current_population.sigmas[0]
        
        
        if iter_ % self.log_freq == 0:
            print("Shape count = %d, Iteration %04d, Time = %0.1f sec. " % 
                (evolver.current_shapes,
                 iter_,
                 time.time() - self.evolving_start_time)
                )
            print("Best score = %0.1f, Mean score = %0.1f. Std = %0.2f." % 
                (self.log_objective_stats[iter_, 0],
                 self.log_objective_stats[iter_, 2],
                 self.log_objective_stats[iter_, 3])
                )

            self.img_drawer.show_img(evolver.best_solution)
            self.log_best_solutions.append(evolver.best_solution)
            
    def save(self):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        pickle.dump(self, open("wyniki/ppl{:s}".format(timestamp), 'wb'))
    
    def show_results(self):
    
        plt.figure(figsize=(18, 4))
        plt.plot(self.log_objective_stats[:, 0], 'r-', label='minimum')
        plt.plot(self.log_objective_stats[:, 1], 'g-', label='maximum')
        plt.plot(self.log_objective_stats[:, 2], 'b-', label='mean')
        plt.xlabel('iteration')
        plt.ylabel('objective function value')
        plt.title('min/avg/max objective function values')
        plt.legend()
        plt.show()

        plt.figure(figsize=(18, 4))
        plt.plot(self.log_best_sigmas, 'r-')
        plt.xlabel('iteration')
        plt.ylabel('sigma value')
        plt.title('best sigmas')
        plt.show()

######
## Base class for shape drawers
######
class ShapeDrawer(object):

    def __init__(self, HEIGHT, WIDTH, SHAPE_LEN):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.SHAPE_LEN = SHAPE_LEN

    def show_img(self, chromosome):
        img = self.draw(chromosome)

        plt.figure()
        plt.imshow(img)
        plt.show()

class CircleDrawer(ShapeDrawer):

    def draw(self, chromosome, starting_img=None):

        if starting_img is not None:
            img = starting_img
        else:
            img = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)

        shapes = self.__decode_chromosome(chromosome) 
        shapes_num = shapes.shape[0]

        ## draw each shape encoded in chromosome onto starting_img
        for i in range(shapes_num):
            shape_description = shapes[i]

            shape_img, alpha = self.__shape_img(img.copy(), shape_description) 
            beta = 1.0 - alpha
            img = cv2.addWeighted(img, beta, shape_img, alpha, 0)

        return img

    def __decode_chromosome(self, chromosome):
        # Each circle is encoded as [0,10]^SHAPE_LEN
        # This function decdodes it to valid circle description:
        # (x, y, radius, r, g, b, alpha)
        shapes = chromosome.reshape(-1, self.SHAPE_LEN)
        shapes = shapes/10.0
        
        # x, y, radius, r, g, b, alpha
        max_r = 0.5*max([self.WIDTH, self.HEIGHT])
        shapes *= np.array([[self.WIDTH, self.HEIGHT, max_r, 255, 255, 255, 110]])
        shapes += np.array([0, 0, 2, 0, 0, 0, 15])
        
        return shapes.astype(np.int16)

    def __shape_img(self, starting_img, shape):
        x, y, radius, r, g, b, alpha = tuple(shape)
        r, g, b = float(r), float(g), float(b)
        return cv2.circle(
            starting_img, 
            (x,y),
            radius, 
            (r, g, b),
            -1
            ), alpha/255.0


############
# other mutation/crossover operators
############
def single_circle_mutation(circle):
    rnd = np.random
    
    ## HARD MUTATION
    if rnd.rand() < 0.01:
        circle[0:3] = 10.0*rnd.rand(3)
        circle[3:] = 10.0*rnd.rand(4) + 0.5
    
    ## SOFT MUTATION
    if rnd.rand() < 0.33:
        circle[0:2] += rnd.randn(2)
    if rnd.rand() < 0.33:
        circle[2] += rnd.randn()
    if rnd.rand() < 0.33:
        circle[3:] += rnd.randn(4)
    
def single_circle_mutation_operator(population):
    population = population.genotypes
    ppl_size = population.shape[0]
    ppl = population.reshape(ppl_size, -1, 7)
    for i in range(ppl_size):
        circle_idx = -1#np.random.randint(ppl.shape[1])
        single_circle_mutation(ppl[i, circle_idx])