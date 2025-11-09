"""my_controller controller."""

import numpy
import xml.etree.ElementTree as ET
import numpy as np
from enum import Enum


# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

CELLROWS=7
CELLCOLS=14
CELL_SIZE = 0.15
KP = 10.0
MAX_SPEED = 6.27

class Orientation(Enum):
    NORTH = 'north'
    SOUTH = 'south'
    EAST = 'east'
    WEST = 'west'

class Direction(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    FRONT = 'front'
    BACK = 'back'


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

        self.orientation = None
        self.labMap = None
        self.prob_matrix = np.full((CELLROWS, CELLCOLS), 1.0 / (CELLROWS * CELLCOLS))

        # Wall
        self.sensor_model_wall = {
            Direction.LEFT:  {'mean': 146.21, 'std': 4.89},
            Direction.RIGHT: {'mean': 146.34, 'std': 4.70},
            Direction.FRONT: {'mean': 133.30, 'std': 3.57},
            Direction.BACK: {'mean': 119.13, 'std': 3.45}
        }

        # No wall
        self.sensor_model_no_wall = {
            Direction.LEFT:  {'mean': 67.23, 'std': 3.29},
            Direction.RIGHT: {'mean': 67.23, 'std': 3.19},
            Direction.FRONT: {'mean': 67.13, 'std': 2.35},
            Direction.BACK: {'mean': 67.15, 'std': 2.28}
        }

        ###* MEASUREMENT CODE START *###

        # self.measures_right = []
        # self.measures_left = []
        # self.measures_back = []
        # self.measures_front = []

        # self.measure_count = 0

        ###* MEASUREMENT CODE END *###

        # Step simulation to initialize sensors.
        self.step()

        self.abs_ref_pos = numpy.array(self.gps.getValues())
        print("Initial position:", self.abs_ref_pos)

    def step(self):
        self.robot.step(self.timeStep)
        self.cur_pos = numpy.array(self.gps.getValues())
        self.cur_dir = (-numpy.arctan2(self.compass.getValues()[1], self.compass.getValues()[0]) + numpy.pi/2) % (2 * numpy.pi)
        self.cur_dir = (self.cur_dir + numpy.pi) % (2 * numpy.pi) - numpy.pi
        
        self.ds = [s.getValue() for s in self.dist_sensors]
        self.sensor_data = {
            Direction.FRONT: (self.ds[0] + self.ds[7]) / 2, #* Avg of both front facing sensors
            Direction.LEFT: self.ds[5],
            Direction.RIGHT: self.ds[2],
            Direction.BACK: (self.ds[3] + self.ds[4]) / 2 #* Avg of both back facing sensors
        }

        ###* MEASUREMENT CODE START *###

        # self.measures_front.append((self.ds[0] + self.ds[7]) / 2)
        # self.measures_right.append(self.ds[2])
        # self.measures_left.append(self.ds[5])
        # self.measures_back.append((self.ds[3] + self.ds[4]) / 2)

        # self.measure_count += 1
        # print(self.measure_count)

        ###* MEASUREMENT CODE END *###

    def set_orientation(self, dir):
        self.orientation = dir

    def motion_update(self):
        new_prob_matrix = np.zeros_like(self.prob_matrix)
        
        movement = {
            Orientation.NORTH: (1, 0),   
            Orientation.SOUTH: (-1, 0),  
            Orientation.EAST: (0, 1),   
            Orientation.WEST: (0, -1)    
        }

        di, dj = movement[self.orientation]
        
        for i in range(CELLROWS):
            for j in range(CELLCOLS):
                prob = self.prob_matrix[i, j]
                
                if prob > 0: 
                    new_i = i + di
                    new_j = j + dj
                    
                    if 0 <= new_i < CELLROWS and 0 <= new_j < CELLCOLS:
                        walls = self.check_walls(i, j)
                        direction_blocked = walls[self.orientation]
                        
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
            Orientation.NORTH: {Direction.FRONT: Orientation.NORTH, Direction.BACK: Orientation.SOUTH, Direction.LEFT: Orientation.WEST, Direction.RIGHT: Orientation.EAST},
            Orientation.SOUTH: {Direction.FRONT: Orientation.SOUTH, Direction.BACK: Orientation.NORTH, Direction.LEFT: Orientation.EAST, Direction.RIGHT: Orientation.WEST},
            Orientation.EAST: {Direction.FRONT: Orientation.EAST, Direction.BACK: Orientation.WEST, Direction.LEFT: Orientation.NORTH, Direction.RIGHT: Orientation.SOUTH},
            Orientation.WEST: {Direction.FRONT: Orientation.WEST, Direction.BACK: Orientation.EAST, Direction.LEFT: Orientation.SOUTH, Direction.RIGHT: Orientation.NORTH}
        }
        
        mapping = sensor_to_world[self.orientation]

        for i in range(CELLROWS):
            for j in range(CELLCOLS):
                prob = self.prob_matrix[i, j]
                
                walls = self.check_walls(i, j)
                
                likelihood = 1.0
                
                for sensor_dir, measured_value in self.sensor_data.items():
                    world_dir = mapping[sensor_dir]
                    has_wall = walls[world_dir]
                    
                    if has_wall:
                        mu = self.sensor_model_wall[sensor_dir]['mean']
                        sigma = self.sensor_model_wall[sensor_dir]['std']
                    else:
                        mu = self.sensor_model_no_wall[sensor_dir]['mean']
                        sigma = self.sensor_model_no_wall[sensor_dir]['std']
                    
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
        walls = {Orientation.NORTH: False, Orientation.SOUTH: False, Orientation.WEST: False, Orientation.EAST: False}

        if i == CELLROWS - 1:  # Top row
            walls[Orientation.NORTH] = True
        elif (i * 2 + 1) < len(self.labMap):
            if self.labMap[i * 2 + 1][j * 2] == '-':
                walls[Orientation.NORTH] = True

        # Check south wall
        if i == 0:  # Bottom row
            walls[Orientation.SOUTH] = True
        elif (i * 2 - 1) >= 0:
            if self.labMap[i * 2 - 1][j * 2] == '-':
                walls[Orientation.SOUTH] = True

        # Check west wall
        if j == 0:  # Leftmost column
            walls[Orientation.WEST] = True
        elif (j * 2 - 1) >= 0:
            if self.labMap[i * 2][j * 2 - 1] == '|':
                walls[Orientation.WEST] = True

        # Check east wall
        if j == CELLCOLS - 1:  # Rightmost column
            walls[Orientation.EAST] = True
        elif (j * 2 + 1) < len(self.labMap[0]):
            if self.labMap[i * 2][j * 2 + 1] == '|':
                walls[Orientation.EAST] = True

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
            myrob.set_orientation(Orientation.NORTH)
        elif command == "S":
            print("Moving South")
            target_pos[1] -= CELL_SIZE
            myrob.set_orientation(Orientation.SOUTH)
        elif command == "E":
            print("Moving East")
            target_pos[0] += CELL_SIZE
            myrob.set_orientation(Orientation.EAST)
        elif command == "W":
            print("Moving West")
            target_pos[0] -= CELL_SIZE
            myrob.set_orientation(Orientation.WEST)
        ###* MEASUREMENT CODE START *###
        # elif command == "R":
        #     print("Resetting")
        #     myrob.measure_count = 0
        #     myrob.measures_right.clear()
        #     myrob.measures_left.clear()
        #     myrob.measures_back.clear()
        #     myrob.measures_front.clear()
        #     print("Reset Done")
        ###* MEASUREMENT CODE END *###
        elif command == "":
            break

        ###* MEASUREMENT CODE START *###
            # pass

        # if myrob.measure_count == 1000:
        #     mean_right = np.mean(myrob.measures_right)
        #     std_right = np.std(myrob.measures_right)

        #     mean_left = np.mean(myrob.measures_left)
        #     std_left = np.std(myrob.measures_left)

        #     mean_front = np.mean(myrob.measures_front)
        #     std_front = np.std(myrob.measures_front)

        #     mean_back = np.mean(myrob.measures_back)
        #     std_back = np.std(myrob.measures_back)

        #     print(f"MEAN_RIGHT: {mean_right} | STD_RIGHT: {std_right}")
        #     print(f"MEAN_LEFT: {mean_left} | STD_LEFT: {std_left}")
        #     print(f"MEAN_FRONT: {mean_front} | STD_FRONT: {std_front}")
        #     print(f"MEAN_BACK: {mean_back} | STD_BACK: {std_back}")
        #     break
        ###* MEASUREMENT CODE END *###

        myrob.sense_update()
        myrob.move_to(target_pos)
        myrob.motion_update()
        myrob.sense_update()
        myrob.save_probability_matrix()

        max_prob_idx = np.unravel_index(np.argmax(myrob.prob_matrix), myrob.prob_matrix.shape)
        entropy = -np.sum(myrob.prob_matrix * np.log(myrob.prob_matrix + 1e-10))

        print(f"Most likely position: ({max_prob_idx[0]}, {max_prob_idx[1]}) with prob {myrob.prob_matrix[max_prob_idx]:.4f}")
        print(f"Uncertainty (entropy): {entropy:.4f}\n")

    commands_file.close()
