import os 
import genome
import sys
import creature
import pybullet as p
import time 
import random
import numpy as np
import envt

## ... usual starter code to create a sim and floor
def main(csv_file):
    assert os.path.exists(csv_file), "Tried to load " + csv_file + " but it does not exists"

    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    

    """ START OF CODE THAT I PARTIALLY WROTE BY MYSELF """
    # add the mountain
    mountain_height, mid = envt.create_landscape()
    mountain_top = (0, 0, mountain_height)
    p.setGravity(0, 0, -10)
    
    # initialize a creature
    cr = creature.Creature(num_legs=0)
    
    # load the raw DNA from the CSV file
    raw_dna = genome.Genome.from_csv(csv_file)
    # reconstruct the structured dna from the raw data
    structured_dna = []
    if raw_dna:
        # assume first gene in the csv is a torso
        structured_dna.append({
            'type': creature.PartType.TORSO,
            'gene': raw_dna[0]
        })
        # assume the rest are legs
        for gene_array in raw_dna[1:]:
            structured_dna.append({
                "type": creature.PartType.LEG,
                "gene": gene_array
            })

    # update the creature with the structured DNA
    cr.update_dna(structured_dna)
    
    # save it to XML
    with open('test.urdf', 'w') as f:
        f.write(cr.to_xml())
    # load it into the sim
    rob1 = p.loadURDF('test.urdf')
    # air drop it
    p.resetBasePositionAndOrientation(rob1, [7, 0, 2], [0, 0, 0, 1])
    start_pos, orn = p.getBasePositionAndOrientation(rob1)
    """ END OF CODE THAT I PARTIALLY WROTE BY MYSELF """

    # iterate 
    elapsed_time = 0
    wait_time = 1.0/240 # seconds
    total_time = 30 # seconds
    step = 0
    while True:
        p.stepSimulation()
        step += 1
        if step % 24 == 0:
            motors = cr.get_motors()
            assert len(motors) == p.getNumJoints(rob1), "Something went wrong"
            for jid in range(p.getNumJoints(rob1)):
                mode = p.VELOCITY_CONTROL
                vel = motors[jid].get_output()
                p.setJointMotorControl2(rob1, 
                            jid,  
                            controlMode=mode, 
                            targetVelocity=vel)
            new_pos, orn = p.getBasePositionAndOrientation(rob1)
            #print(new_pos)
            dist_moved = np.linalg.norm(np.asarray(start_pos) - np.asarray(new_pos))
            print(dist_moved)
        time.sleep(wait_time)
        elapsed_time += wait_time
        if elapsed_time > total_time:
            break

    print("TOTAL DISTANCE MOVED:", dist_moved)



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python playback_test.py csv_filename"
    main(sys.argv[1])

