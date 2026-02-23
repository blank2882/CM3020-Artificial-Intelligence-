# If you on a Windows machine with any Python version 
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# the multi-threaded version does not work
# so instead, you can use this version. 

import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np
import pandas as pd

""" START OF CODE THAT I WROTE BY MYSELF """
# change this to the data folder that you want to store the csv files in
data_filenum = 1
# variables to change for each run
pop_size = 20
num_gen = 100
iterations = 2400

# define mutation rates
point_mutate_rates = {
    creature.PartType.TORSO: 0.05,
    creature.PartType.LEG: 0.20
}
# range from 0.1 to 0.25
# higher values means more mutation in the torso and legs of the creature
point_mutate_amt = 0.5
shrink_mutate_rate = 0.15
# high shrink_mutate_rate and low grow_mutate_rate will favour smaller creatures 
# high grow_mutate_rate and low shrink_mutate_rate will favour creatures to evolve more legs
grow_mutate_rate = 0.15 
""" END OF CODE THAT I WROTE BY MYSELF """

""" START OF CODE THAT I PARTIALLY WROTE BY MYSELF """
class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size, 
                                    gene_count=None)
        #sim = simulation.ThreadedSim(pool_size=1)
        sim = simulation.Simulation(iterations=iterations)

        for iteration in range(num_gen):
            # this is a non-threaded version 
            # where we just call run_creature instead
            # of eval_population
            for cr in pop.creatures:
                # sim.run_creature(cr, 2400)            
                sim.run_creature(cr)            
            #sim.eval_population(pop, 2400)
            fits = [cr.calculate_fitness(iterations) 
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) 
                    for cr in pop.creatures]
            # save the fitnesses in a dataframe
            fits_df = pd.DataFrame({'iteration': [iteration], 
                                    'fitness': [np.round(np.max(fits), 3)], 
                                    'mean fitness': [np.round(np.mean(fits), 3)]})
            fits_df.to_csv(f"advanced_cw_data/run{data_filenum}/fits.csv", columns=['iteration', 'fitness', 'mean fitness'], mode='a', header=True if iteration == 0 else False, index=False)
            
            print(iteration, "fittest:", np.round(np.max(fits), 3), 
                  "mean:", np.round(np.mean(fits), 3), "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))       
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                
                # target mutate and crossover
                dna = genome.Genome.targeted_crossover(p1.dna, p2.dna)
                dna = genome.Genome.targeted_point_mutate(dna, rates=point_mutate_rates, amount=point_mutate_amt)
                dna = genome.Genome.targeted_shrink_mutate(dna, rate=shrink_mutate_rate)
                dna = genome.Genome.targeted_grow_mutate(dna, rate=grow_mutate_rate)
                
                cr = creature.Creature(num_legs=0)
                cr.update_dna(dna)
                new_creatures.append(cr)
            # elitism
            max_fit = np.max(fits)
            for cr in pop.creatures:
                if cr.calculate_fitness(iterations) == max_fit:
                    best_cr_dna = cr.dna
                    filename = f"advanced_cw_data/run{data_filenum}/elite_" + str(iteration)+".csv"
                    # save the raw gene arrays for playback
                    genome.Genome.to_csv([p['gene'] for p in cr.dna], filename)
                    break
            
            # apply elitism 
            if best_cr_dna:
                elite_cr = creature.Creature(num_legs=0)
                elite_cr.update_dna(best_cr_dna)
                new_creatures[0] = elite_cr
            
            pop.creatures = new_creatures
        """ END OF CODE THAT I PARTIALLY WROTE BY MYSELF """
                            
        self.assertNotEqual(fits[0], 0)

unittest.main()
