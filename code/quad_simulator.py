import os
import numpy as np
import pickle
import imageio
import argparse
import matplotlib.pyplot as plt
from math import cos, sin
from mpl_toolkits.mplot3d import Axes3D
import time

# Patch to 3d axis to remove margins around x, y and z limits.
# Taken from here: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new



class QuadrotorSimulator():
    """
    Class for plotting a quadrotor
    Original author: Daniel Ingram (daniel-s-ingram)
    https://github.com/AtsushiSakai/PythonRobotics
    """
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, reward=0, title=None, filepath=None, size=3.0):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.reward_data = []

        # start
        self.sx = x
        self.sy = y
        self.sz = z

        # target
        self.tx = 0
        self.ty = 0
        self.tz = 0

        #fig = plt.figure(figsize=(16,12), dpi=72)
        #fig = plt.figure(dpi=120)
        fig = plt.figure()
        self.ax = plt.subplot2grid((32, 24), (0, 0), colspan=20, rowspan=20, projection='3d')
        #self.ax5 = plt.subplot2grid((32, 24), (24, 0), colspan=20, rowspan=8)
        self.update_pose(0, x, y, z, roll, pitch, yaw, reward, title, filepath)

    def set_target(self, x, y, z):
        self.tx = x
        self.ty = y
        self.tz = z

    def update_pose(self, iter_, x, y, z, roll, pitch, yaw, reward, title, filepath):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.reward_data.append(reward)
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        self.plot(title, filepath, iter_)

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
              sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])

    def clear(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.reeard_data = []

    def close(self):
        plt.close()
        
    def plot(self, title, filepath, iter_):
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        #plt.cla()
        self.ax.cla()
        
        if title:
            plt.suptitle(title, fontsize=10)

        # plot start
        self.ax.scatter(self.sx, self.sy, self.sz, zdir='z', c='g')

        # plot target
        self.ax.scatter(self.tx, self.ty, self.tz, zdir='z', c='b')

        # plot rotors
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.', zdir='z')

        # plot frame
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-', zdir='z')
        
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-', zdir='z')

        # plot track
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:', zdir='z')

        x_bounds = 155#10 #7.5
        y_bounds = 155#10 #7.5
        z_bounds = 100 #15
        self.ax.set_xlim(-x_bounds, x_bounds)
        self.ax.set_ylim(-y_bounds, y_bounds)
        self.ax.set_zlim(0, z_bounds)

        '''
        # Plot reward
        self.ax5.plot(self.reward_data, label='Reward', c=[0,0,0,0.7], linewidth=1.0)
        #self.ax5.set_xlim(0, max(30, len(self.reward_data)))
        #self.ax5.set_ylim(-1, 1)
        self.ax5.set_xlabel('Time')
        self.ax5.set_ylabel('Reward')
        '''

        if filepath:
            if iter_ % 100 == 0:
                plt.savefig(filepath)
        else:
            plt.pause(0.000001)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def load_states():
    dbfile = open('state_results_traj.pickle', 'rb')      
    states = pickle.load(dbfile)
    dbfile.close()
    return states


if __name__ == "__main__":

    exportPath = './video_traj/'       
    if not os.path.exists(exportPath):
        os.makedirs(exportPath)


    states = load_states()

    results = Memory()
    rew = 1
    for i in range(20):
        state = np.random.randint(-5, 5, (9))
        rew = np.random.randint(rew, rew+100)
        results.rewards.append(rew)
        results.states.append(state)
    results.states.append(np.zeros(9))
    results.rewards.append(2000)

    states = np.array(states)
    print(states.shape)
    title = 'Hovering'
    init_state = [6,6,6,0,0,0,0,0,0]

    filepath = "{}frame_{:04}.png".format(exportPath, 0)
    print("Processing: {}".format(filepath))
    images = []

    quad_sim = QuadrotorSimulator(
                    x=init_state[0], 
                    y=init_state[1], 
                    z=init_state[2],  
                    roll=init_state[3], 
                    pitch=init_state[4], 
                    yaw=init_state[5], 
                    title=title,
                    filepath=filepath)
    quad_sim.set_target(0.0, 0.0, 0.0)

    images.append(imageio.imread(filepath))

    for i, state in enumerate(states[-3000:]):
        filepath = "{}frame_{:04}.png".format(exportPath, i+1)
        quad_sim.update_pose(i,
                        x=state[0], 
                        y=state[1], 
                        z=state[2], 
                        roll=state[3],
                        pitch=state[4], 
                        yaw=state[5],
                        reward=0,
                        title=title,
                        filepath=filepath)
        
        if i % 100 == 0:
            images.append(imageio.imread(filepath))
    
    #time.sleep(200)
    # Save all frames to animated gif
    imageio.mimsave("movie_traj.gif", images)
    


    

    



