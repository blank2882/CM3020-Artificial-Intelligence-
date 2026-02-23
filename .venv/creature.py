import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np
import pybullet as p

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0
    

    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1
            
        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)
        
        return output 

class PartType(Enum):
    # define different body parts for a targeted evolution
    TORSO = 1
    LEG = 2

class Creature:
    def __init__(self, gene_count = None, num_legs=4):
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        
        """ START OF CODE THAT I WROTE BY MYSELF """
        # initialize a creature with a structured genome
        self.spec = genome.Genome.get_gene_spec()
        gene_length = len(self.spec)
        
        # create a structured DNA with tagged parts
        # initialize an empty DNA
        self.dna = []
        # add one torso gene
        self.dna.append({
            "type": PartType.TORSO,
            "gene": genome.Genome.get_random_gene(gene_length)
        })
        
        # add a specific number of LEG genes
        for _ in range(num_legs):
            self.dna.append({
                "type": PartType.LEG,
                "gene": genome.Genome.get_random_gene(gene_length)
            })

        # track distance closest to mountain
        self.distance_to_mountain_top = None
        self.mountain_contact = 0
        self.cid = None
        self.size = None
        self.failed = False
        """ END OF CODE THAT I WROTE BY MYSELF """

    def get_flat_links(self):
        if self.flat_links == None:
            # extract the raw gene arrays from the structured DNA
            gene_arrays = [d['gene'] for d in self.dna]
            gdicts = genome.Genome.get_genome_dicts(gene_arrays, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], 
                                self.flat_links[0].name, 
                                self.flat_links, 
                                exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node! 
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp,  l.control_freq)
                motors.append(m)
            self.motors = motors 
        return self.motors 
    
    def update_position(self, pos):        
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos

    """ START OF CODE THAT I WROTE BY MYSELF """
    # update distance to the mountain top
    def update_dist_to_peak(self, mountain_top):
        if self.start_position is None or self.last_position is None:
            # if the creature has not moved yet, return 100
            return 100
        dist = np.linalg.norm(np.subtract(mountain_top, self.last_position))
        # calculate the Euclidean distance between the current position
        # and the mountain top
        if self.distance_to_mountain_top is None or dist < self.distance_to_mountain_top:
            # if the distance is smaller than the current distance or
            # the distance has not been set yet, update the distance
            self.distance_to_mountain_top = dist
        
    def check_mountain_contact(self, mid):
        point_contact = p.getContactPoints(bodyA=self.cid, bodyB=mid)
        
        if len(point_contact) > 0:
            self.mountain_contact += 1

    def num_contact_mountain(self):
        return self.mountain_contact
        
    # get the distance to mountain top
    def get_distance_to_mountain(self):
        return self.distance_to_mountain_top
    
    # fail the creature and set failed to true when the creature fails the simulation
    def cr_failed(self):
        self.failed = True
    
    # set the creature id
    def get_cid(self, cid):
        self.cid = cid
    
    # set the creature size
    def update_size(self):
        # get the bounding box of the creature
        min_dim,max_dim = p.getAABB(self.cid)
        # calculate the size
        cr_dim = max(np.array(max_dim) - np.array(min_dim))
        # The size of the creature is the maximum of the length, width, and height of the bounding box.
        self.size = cr_dim
        
    # get the creature size
    def get_size(self):
        return self.size
    
#   Calculate the fitness of the creature.
    def calculate_fitness(self, sim_time):
        # If the creature has failed the simulation, the fitness is 0.
        if self.failed:
            return 0
        # the final fitness is the sum of all components.
        return max(
            # dist_reward: the distance to the mountain top
            self.dist_reward() -
            # size_penalty: penalize the size of the creature if it gets too big
            self.size_penalty() +
            # vertD_reward: the vertical displacement of the creature
            self.vertD_reward() +
            # dist_travelled_reward: the distance travelled by the creature
            self.dist_travelled_reward() +
            # mountain_contact_reward: the reward for mountain contact
            self.mountain_contact_reward(sim_time),
            0
        )
    
    def dist_reward(self):
        if self.failed:
            # if the creature has failed the simulation, set its fitness to 0
            return 0
        dist = self.distance_to_mountain_top
        if dist == 0:
            # if the distance is 0, the creature is right at the mountain top. set its fitness to 0.
            return 0
        # calculate the fitness as 100 divided by the distance
        fitness = 100 / dist
        # return the minimum of this value and 100, give a  reward for closer creatures
        return min(fitness, 100)
            
    def size_penalty(self):
        # if the creature's start and last pos is unknown, then the creature fails
        if self.size == None or self.failed:
            return 50
        best_size = 0.5
        # calculate size
        size_diff = abs(self.size - best_size) / best_size
        # normalize to a value between 0 and 80, give a larger penalty for bigger creatures
        return min(80 * size_diff, 80)
    
    def mountain_contact_reward(self, sim_time):
        if self.failed:
            # if the creature has failed the simulation, set its fitness to 0
            return 0
        # normalize to a value between 0 and 40, give a smaller reward for more contact with the mountain
        return min(50 * (self.mountain_contact / sim_time), 50)
    
    def vertD_reward(self):
        # if the creature's start and last pos is unknown, set the default distance to 0
        if self.start_position == None or self.last_position == None or self.failed:
            return 0
        # calculate vertical displacement
        vert_progress = self.last_position[2] - self.start_position[2]
        # normalize to a value between 0 and 50,
        return 50 * np.tanh(vert_progress/ 5)
    
    def dist_travelled_reward(self):
        # Check if the start or last position is not set, or if the creature has failed.
        # If any of these conditions are true, return a distance of 0.
        if self.start_position is None or self.last_position is None or self.failed:
            return 0
        
        # calculate the distance travelled using get_distance_travelled function.
        # apply the hyperbolic tangent function to the distance travelled, divided by 10.
        # normalize the result to a value between 0 and 50.
        return 50 * np.tanh(self.get_distance_travelled() / 10)
    """ END OF CODE THAT I WROTE BY MYSELF """
    
    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1 - p2)
        return dist

    # update the creature's DNA
    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None