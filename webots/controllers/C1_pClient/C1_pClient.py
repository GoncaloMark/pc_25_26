"""my_controller controller."""

import numpy
import xml.etree.ElementTree as ET
import numpy as np


# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

CELLROWS=7
CELLCOLS=14
CELL_SIZE = 0.15
KP = 10.0
MAX_SPEED = 6.27

class Map():
    def __init__(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        
        self.labMap = [[' '] * (CELLCOLS*2-1) for i in range(CELLROWS*2-1) ]
        i=1
        for child in root.iter('Row'):
           line=child.attrib['Pattern']
           row =int(child.attrib['Pos'])
           if row % 2 == 0:  # this line defines vertical lines
               for c in range(len(line)):
                   if (c+1) % 3 == 0:
                       if line[c] == '|':
                           self.labMap[row][(c+1)//3*2-1]='|'
                       else:
                           None
           else:  # this line defines horizontal lines
               for c in range(len(line)):
                   if c % 3 == 0:
                       if line[c] == '-':
                           self.labMap[row][c//3*2]='-'
                       else:
                           None
               
           i=i+1

class MyRob:
    def __init__(self):
        # create the Robot instance.
        self.robot = Robot()

        # Get simulation step length.
        self.timeStep = int(self.robot.getBasicTimeStep())

        # Constants of the e-puck motors and distance sensors.
        self.cruiseVelocity = 4.0
        self.num_dist_sensors = 8

        # Get left and right wheel motors.
        self.leftMotor = self.robot.getDevice("left wheel motor")
        self.rightMotor = self.robot.getDevice("right wheel motor")

        # Get frontal distance sensors.
        self.dist_sensors = [self.robot.getDevice('ps' + str(x)) for x in range(self.num_dist_sensors)]  # distance sensors
        list(map((lambda s: s.enable(self.timeStep)), self.dist_sensors))  # Enable all distance sensors

        # Disable motor PID control mode.
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))


        # Set the initial velocity of the left and right wheel motors.
        self.driveMotors(0.0, 0.0)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timeStep)

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timeStep)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timeStep)

        self.direction = None
        self.labMap = None
        self.prob_matrix = np.full((CELLROWS, CELLCOLS), 1.0 / (CELLROWS * CELLCOLS))

        self.sensor_model = {
            'left':  {'mean': 146.22, 'std': 5.15},
            'right': {'mean': 146.42, 'std': 4.70},
            'front': {'mean': 133.43, 'std': 3.53}
        }

        self.no_wall = {
            'left':  {'mean': 67.13, 'std': 3.38},
            'right': {'mean': 67.27, 'std': 3.38},
            'front': {'mean': 67.11, 'std': 2.35}
        }

        # Step simulation to initialize sensors.
        self.step()

        self.abs_ref_pos = numpy.array(self.gps.getValues())
        print("Initial position:", self.abs_ref_pos)

    def step(self):
        self.robot.step(self.timeStep)
        self.cur_pos = numpy.array(self.gps.getValues())
        self.cur_dir = (-numpy.arctan2(self.compass.getValues()[1], self.compass.getValues()[0]) + numpy.pi/2) % (2 * numpy.pi)
        self.cur_dir = (self.cur_dir + numpy.pi) % (2 * numpy.pi) - numpy.pi
        
        # Left / Right Sensors: No wall when < 80 | Wall when > 140
        # Front Sensors (7 + 0 / 2): No wall when < 80 | Wall when > 110
        self.ds = [s.getValue() for s in self.dist_sensors]
        self.sensor_data = {
            'front': (self.ds[0] + self.ds[7]) / 2,
            'left': self.ds[5],
            'right': self.ds[2]
        }

        # self.get_direction()

        # print(self.sensor_map)
        # if self.labMap:
            # print(self.check_walls(0,0))

    def set_direction(self, dir):
        # degs = np.rad2deg(self.cur_dir) % 360

        # if 45 <= degs < 135:
        #     self.direction = 'north'
        # elif 135 <= degs < 225:
        #     self.direction = 'west'
        # elif 225 <= degs < 315:
        #     self.direction = 'south'
        # else:
        #     self.direction = 'east'
        self.direction = dir

    def motion_update(self):
        new_prob_matrix = np.zeros_like(self.prob_matrix)
        
        movement = {
            'north': (1, 0),   
            'south': (-1, 0),  
            'east': (0, 1),   
            'west': (0, -1)    
        }

        di, dj = movement[self.direction]
        
        for i in range(CELLROWS):
            for j in range(CELLCOLS):
                prob = self.prob_matrix[i, j]
                
                if prob > 0: 
                    new_i = i + di
                    new_j = j + dj
                    
                    if 0 <= new_i < CELLROWS and 0 <= new_j < CELLCOLS:
                        walls = self.check_walls(i, j)
                        direction_blocked = walls[self.direction]
                        
                        if direction_blocked:
                            new_prob_matrix[i, j] += prob
                        else:
                            new_prob_matrix[new_i, new_j] += prob
                    else:
                        new_prob_matrix[i, j] += prob
        
        total = np.sum(new_prob_matrix)
        if total > 0:
            self.prob_matrix = new_prob_matrix / total

    def sense_update(self):
        new_prob_matrix = np.zeros_like(self.prob_matrix)

        sensor_to_world = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        mapping = sensor_to_world[self.direction]

        for i in range(CELLROWS):
            for j in range(CELLCOLS):
                prob = self.prob_matrix[i, j]
                
                walls = self.check_walls(i, j)
                
                likelihood = 1.0
                
                for sensor_dir, measured_value in self.sensor_data.items():
                    world_dir = mapping[sensor_dir]
                    has_wall = walls[world_dir]
                    
                    if has_wall:
                        mu = self.sensor_model[sensor_dir]['mean']
                        sigma = self.sensor_model[sensor_dir]['std']
                    else:
                        mu = self.no_wall[sensor_dir]['mean']
                        sigma = self.no_wall[sensor_dir]['std']
                    
                    variance = sigma ** 2
                    
                    gauss = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(
                        -((measured_value - mu) ** 2) / (2 * variance)
                    )
                    
                    likelihood *= gauss

                new_prob_matrix[i, j] = prob * likelihood

        total_sum = np.sum(new_prob_matrix)
        if total_sum > 0:
            self.prob_matrix = new_prob_matrix / total_sum

    def driveMotors(self, leftSpeed, rightSpeed):
        self.leftMotor.setVelocity(max(min(leftSpeed,MAX_SPEED),-MAX_SPEED))
        self.rightMotor.setVelocity(max(min(rightSpeed,MAX_SPEED),-MAX_SPEED))

    def rotate(self, target_dir):
        #print("Rotating to direction:", target_dir)
        while(numpy.abs(target_dir - self.cur_dir) > 0.01):
            angle_error = target_dir - self.cur_dir
            angle_error = (angle_error + numpy.pi) % (2 * numpy.pi) - numpy.pi
            #print("Current direction:", self.cur_dir*180/numpy.pi, " Target direction    :", target_dir*180/numpy.pi, " Angle error:", angle_error*180/numpy.pi)

            self.driveMotors(-KP * angle_error, +KP * angle_error)

            self.step()
        self.driveMotors(0.0, 0.0)

    def move_to(self, target_pos):
        #print("Moving to position:", target_pos, " from ", self.cur_pos)

        target_dir = numpy.arctan2(target_pos[1] - self.cur_pos[1], target_pos[0] - self.cur_pos[0])
        target_dir = (target_dir + numpy.pi/4)//(numpy.pi/2) * numpy.pi/2  # snap to 90 degrees

        #print("Target direction:", target_dir*180/numpy.pi, " from ", self.cur_dir*180/numpy.pi)

        if(numpy.abs(target_dir - self.cur_dir) > 0.1):
            self.rotate(target_dir)

        dist = numpy.linalg.norm(target_pos - self.cur_pos)
        #print("Distance to target:", dist)
        while(dist > 0.005):

            target_pos_dir = target_pos + CELL_SIZE * numpy.array([numpy.cos(target_dir), numpy.sin(target_dir), 0.0])
            angle_error = numpy.arctan2(target_pos_dir[1] - self.cur_pos[1],
                                        target_pos_dir[0] - self.cur_pos[0]) - self.cur_dir
            angle_error = (angle_error + numpy.pi) % (2 * numpy.pi) - numpy.pi

            #print("Current norm position:", (self.cur_pos - self.abs_ref_pos)/CELL_SIZE, " Target norm position:", (target_pos-self.abs_ref_pos)/CELL_SIZE, 
            #    " Distance:", dist, " Target Angle Pos:", (target_pos_dir-self.abs_ref_pos)/CELL_SIZE, " Current Angle:", self.cur_dir*180/numpy.pi)

            #print("Angle error:", angle_error*180/numpy.pi)

            self.driveMotors( self.cruiseVelocity - KP * angle_error,
                             self.cruiseVelocity + KP * angle_error)   

            self.step()
            dist = numpy.linalg.norm(target_pos - self.cur_pos)
        self.driveMotors(0.0, 0.0)

    def check_walls(self, i, j):
        walls = {'north': False, 'south': False, 'west': False, 'east': False}

        if i == CELLROWS - 1:  # Top row
            walls['north'] = True
        elif (i * 2 + 1) < len(self.labMap):
            if self.labMap[i * 2 + 1][j * 2] == '-':
                walls['north'] = True

        # Check south wall
        if i == 0:  # Bottom row
            walls['south'] = True
        elif (i * 2 - 1) >= 0:
            if self.labMap[i * 2 - 1][j * 2] == '-':
                walls['south'] = True

        # Check west wall
        if j == 0:  # Leftmost column
            walls['west'] = True
        elif (j * 2 - 1) >= 0:
            if self.labMap[i * 2][j * 2 - 1] == '|':
                walls['west'] = True

        # Check east wall
        if j == CELLCOLS - 1:  # Rightmost column
            walls['east'] = True
        elif (j * 2 + 1) < len(self.labMap[0]):
            if self.labMap[i * 2][j * 2 + 1] == '|':
                walls['east'] = True

        return walls

    def save_probability_matrix(self, filename="localization.out"):
        with open(filename, "a") as file:
            for i in range(CELLROWS-1,-1,-1):
                row = " ".join(f"{self.prob_matrix[i, j]:.3f}" for j in range(CELLCOLS))
                file.write(row + "\n")
            file.write("\n")

    # In this map the center of cell (i,j), (i in 0..6, j in 0..13) is mapped to labMap[i*2][j*2].
    # to know if there is a wall on top of cell(i,j) (i in 0..5), check if the value of labMap[i*2+1][j*2] is space or not
    def setMap(self, labMap):
        self.labMap = labMap

    def printMap(self):
        for l in reversed(self.labMap):
            print(''.join([str(l) for l in l]))


# main 
if __name__ == '__main__':

    # open commands file
    commands_file = open("commands.txt", "r")

    myrob = MyRob()

    mapc = Map("../C1_supervisor/C1-lab.xml")
    myrob.setMap(mapc.labMap)
    myrob.printMap()

    target_pos = myrob.abs_ref_pos.copy()

    open("localization.out", "w").close()
    myrob.save_probability_matrix()

    while myrob.step() != -1:
        # Read next movement from file
        command = commands_file.readline().strip()
        if command == "N":
            print("Moving North")
            target_pos[1] += CELL_SIZE
            myrob.set_direction('north')
        elif command == "S":
            print("Moving South")
            target_pos[1] -= CELL_SIZE
            myrob.set_direction('south')
        elif command == "E":
            print("Moving East")
            target_pos[0] += CELL_SIZE
            myrob.set_direction('east')
        elif command == "W":
            print("Moving West")
            target_pos[0] -= CELL_SIZE
            myrob.set_direction('west')
        elif command == "exit":
            break

        myrob.motion_update()
        myrob.move_to(target_pos)
        myrob.sense_update()
        myrob.save_probability_matrix()

        max_prob_idx = np.unravel_index(np.argmax(myrob.prob_matrix), myrob.prob_matrix.shape)
        entropy = -np.sum(myrob.prob_matrix * np.log(myrob.prob_matrix + 1e-10))

        print(f"Most likely position: ({max_prob_idx[0]}, {max_prob_idx[1]}) with prob {myrob.prob_matrix[max_prob_idx]:.4f}")
        print(f"Uncertainty (entropy): {entropy:.4f}\n")

    commands_file.close()
