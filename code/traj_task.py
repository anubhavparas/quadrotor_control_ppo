import numpy as np
import math
from quad_dynamics import QuadDynamics

class TrajTaskEnv:
    def __init__(self):
        self.quad = QuadDynamics()
        self.target_position = np.array([0,0,0])   # circle center
        self.observation_space = self.quad.state_size
        self.action_space = 4 # or 8 for mu and sigma
        self.target_thresh = 2 #0.5
        self.thresh_red_factor = 0.1 #0.01
        self.bonus_reward = 200
        self.penalty = -1
        self.done_count = 0
        self.des_rad = 1 #1m
        self.des_vel = 1  # 1 m/s
        self.z_target = 5


    def reset(self):
        self.quad.reset()
        return self.quad.state
    
    def resize_threshold(self):
      self.target_thresh *= self.thresh_red_factor
      

    def set_target_position(self, x,y,z):
        self.target_position = np.array([x,y,z])


    def step(self, action):
        new_state, done, penalty = self.quad.update(action, self.penalty)
        rewards, done = self.get_rewards(new_state, action, done)
        rewards += penalty
        if math.isnan(rewards) or math.isinf(rewards):
          print('in step() new_state d penalty a', new_state, done, penalty, action)
        return new_state, rewards, done


    def get_rewards(self, state, action, done):
        rewards = self.calc_rewards(state, action)
        is_goal_reached, bonus_goal_reward = self.is_target_reached(state)

        if is_goal_reached:
            done = True
            self.done_count +=1 
            rewards += bonus_goal_reward * self.done_count * 0.05
        if math.isnan(rewards) or math.isinf(rewards):
          print('in get_rewards() s done bon_rew a', state, done, bonus_goal_reward, action)
        return rewards, done


    def calc_rewards(self, state, action):
        w1 = 0.004 #10 #20 #10 
        w2 = 0.008 #5
        w3 = 0.0 #0.01
        curr_position = state[:2] # only x,y
        curr_vel = state[6:8]
        dist_err = abs(np.linalg.norm(self.target_position[:2] - curr_position) - self.des_rad)  # distance from the circle
        z_err = abs(self.z_target - state[2])


        x,y = state[0], state[2]
        vel_x, vel_y = state[6], state[7]
        vel_err = abs(x*vel_y - y*vel_x - self.des_vel*self.des_rad)

        vel_z_err = abs(state[8])

        
        action_err = np.linalg.norm(action)
        reward = -(w1*dist_err + w1*z_err + w2*vel_err + w2*vel_z_err)# + w3*action_err)
        if math.isnan(reward) or math.isinf(reward):
          print('in calc_rewards() rewards', reward)
        return reward

    def is_target_reached(self, state):
        curr_pos = state[:3]
        dist = abs(np.linalg.norm(curr_pos-self.target_position) - self.des_rad)

        x,y = state[0], state[2]
        vel_x, vel_y = state[6], state[7]
        vel_err = abs(x*vel_y - y*vel_x - self.des_vel*self.des_rad)


        z_err = abs(self.z_target - state[2])
        if (dist <= self.target_thresh) and vel_err <= self.target_thresh:
            print('Inside the threshold limit')
            return True, self.bonus_reward
        
        return False, 0


    def render(self):
        ### render
        pass


    


    