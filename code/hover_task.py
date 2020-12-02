import numpy as np
import math
from quad_dynamics import QuadDynamics

class HoverTaskEnv:
    def __init__(self):
        self.quad = QuadDynamics()
        self.target_position = np.array([0,0,0])
        self.observation_space = self.quad.state_size
        self.action_space = 4 # or 8 for mu and sigma
        self.target_thresh = 5 #0.5
        self.thresh_red_factor = 0.1 #0.01
        self.bonus_reward = 200
        self.penalty = -1
        self.done_count = 0


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
            rewards += bonus_goal_reward * self.done_count * 0.85
        if math.isnan(rewards) or math.isinf(rewards):
          print('in get_rewards() s done bon_rew a', state, done, bonus_goal_reward, action)
        return rewards, done


    def calc_rewards(self, state, action):
        w1 = 0.004 #10 #20 #10 
        w2 = 0.008 #5
        w3 = 0.0 #0.01
        curr_position = state[:3]
        curr_angle = state[3:6]
        pos_err = np.linalg.norm(self.target_position - curr_position)

        angle_err = np.linalg.norm(curr_angle)
        action_err = np.linalg.norm(action)
        reward = -(w1*pos_err + w2*angle_err)# + w3*action_err)
        if math.isnan(reward) or math.isinf(reward):
          print('in calc_rewards() curr_pos cuur_angle', curr_position, curr_angle)
          print('in calc_rewards() pos_err angle_err action_err', pos_err, angle_err, action_err)
          print('in calc_rewards() state action', state, action)
          print('in calc_rewards() rewards', reward)
        return reward

    def is_target_reached(self, state):
        curr_pos = state[:3]
        dist = np.linalg.norm(curr_pos-self.target_position)
        if (dist <= self.target_thresh):
            print('Inside the threshold limit')
            return True, self.bonus_reward
        
        return False, 0


    def render(self):
        ### render
        pass


    


    