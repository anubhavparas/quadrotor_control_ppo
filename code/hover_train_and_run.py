import argparse
import logging
import pathlib

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import pickle

from ppo import PPO
from config import *
from memory import Memory


from quad_dynamics import QuadDynamics
from hover_task import HoverTaskEnv

from plot_results import plot_hover_result


ENV = "hover_task-v0"
RENDER = False
SOLVED_REWARD = 300
BONUS_REWARD = 1_000
PENALTY = -1
PRINT_EVERY =20
MAX_EPISODES = 3_000 #10_000 #50000 #10000
MAX_TIMESTEPS = 350 #1500  # Batch size
UPDATE_TIMESTEP = 350 #4000  # Batch size
MINI_BATCH__SIZE = 64 #500 mini-batch size
ACTION_STD = 0.0  # 0.5
K_EPOCHS = 10 #10
GAMMA = 0.99
lr = 0.0001
betas = (.9, 0.999)
LOG_PATH = ''
MODEL_PATH = ''









log_path = "{LOG_PATH}/{EXPERIMENT_NAME}"

pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

summary_writer = SummaryWriter(log_dir=LOG_PATH)
logging.info(f"Tensorboard path: {LOG_PATH}")

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creating environment
env = HoverTaskEnv()  # can be made from the the env_factory
env.bonus_reward = BONUS_REWARD
env.penalty = PENALTY
state_dim = env.observation_space
action_dim = env.action_space
random_seed = 10
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    #env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(device, state_dim, action_dim, ACTION_STD, lr, betas, GAMMA, K_EPOCHS, 0.2)
#print(args.lr, betas)

# logging variables
running_reward = 0
avg_length = 0
time_step = 0








################################################
### TRAINING                                   #
################################################


# training loop
num_transitions = 0
poses = []
rewards_plot = []
sum_rewards = 0
sum_rewards_plot = []
state = env.reset()
force_break = False
start_time = time.clock()
for i_episode in tqdm(range(1, MAX_EPISODES + 1), desc='Training'):
    
    state = env.reset()
    state[0] = np.random.randint(20, 30)
    state[1] = np.random.randint(20, 30)
    state[2] = np.random.randint(20, 30)

    if i_episode % 10 == 0:
      env.target_thresh = max(0.5, env.target_thresh*0.9)
    

    for t in tqdm(range(MAX_TIMESTEPS), desc='Running environment'):
        time_step += 1
        num_transitions += 1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        act_count = 0
        while np.any(np.isnan(action)) and not force_break:
          print('in between episode: action is nan',  action)
          action = ppo.select_action(state, memory)
          state = env.reset()
          act_count += 1
          if act_count > 5:
            force_break = True


        state, reward, done= env.step(action)
        if math.isnan(reward) or math.isnan(state[0]):
          force_break = True
          break

        poses.append(state[:9])

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        #if time_step % UPDATE_TIMESTEP == 0:
        #    print('\ngoing to train the model....')
        #    ppo.update(memory)
        #    memory.clear_memory()
        #    time_step = 0
        running_reward += reward
        sum_rewards += reward
        sum_rewards_plot.append(sum_rewards)
        rewards_plot.append(reward)
        if RENDER:
            env.render()
        if done:
            print('\nDone returned!\n')
            state = env.reset()
            state[0] = np.random.randint(20, 30)
            state[1] = np.random.randint(20, 30)
            state[2] = np.random.randint(20, 30)
            break

        if num_transitions % 500 == 0:
          plot_hover_result(num_transitions, rewards_plot, sum_rewards_plot, poses, True)
    
    
    if force_break:
      break
      
        
    print("going to train the model")
    ppo.update(memory)
    memory.clear_memory()

    avg_length += t
    #rewards_plot.append(running_reward)

    # stop training if avg_reward > solved_reward
    '''
    if running_reward > (PRINT_EVERY * SOLVED_REWARD):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(args.env_name))
        break
    '''

    # save every 500 episodes
    if i_episode % 50 == 0:
        torch.save(ppo.policy.state_dict(), '{}PPO_{}.pth'.format(MODEL_PATH, ENV))

    # logging
    summary_writer.add_scalar('avg episode length', avg_length, i_episode)
    summary_writer.add_scalar('reward', running_reward, i_episode)
    summary_writer.close()

    if i_episode % PRINT_EVERY == 0:
        avg_length = int(avg_length / PRINT_EVERY)
        running_reward = int((running_reward / PRINT_EVERY))

        time_elapsed = time.clock() - start_time
        print('Time elapsed: {}min: {}sec \nEpisode {} \t Avg length: {} \t Avg reward: {}'.format(time_elapsed//60, time_elapsed % 60, i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
        
      


#plot(num_transitions, rewards_plot, poses)
time_elapsed = time.clock() - start_time
print('Time elapsed: {}min: {}sec'.format(time_elapsed//60, time_elapsed % 60))





################################################
### TESTING                                    #
################################################

state = env.reset()
env.target_position = np.array([0,0,0])
state[0] = np.random.randint(20, 30)
state[1] = np.random.randint(20, 30)
state[2] = np.random.randint(20, 30)

sim_memory = Memory()
force_break = False
sim_poses = []
sim_sum_rewards = 0
sim_run_rew_plot = []

transitions = 0
reached = False
iter = 0
MAX_ITER = 40_000

while not reached and iter < MAX_ITER:

  action = ppo.select_action(state, memory)
  transitions += 1

  if iter % 300 == 0:
    state = env.reset()
    state[0] = np.random.randint(0, 5)
    state[1] = np.random.randint(0, 5)
    state[2] = np.random.randint(0, 5)
  
  
  act_count = 0
  while np.any(np.isnan(action)) and not force_break:
    print('in between episode: action is nan',  action)
    action = ppo.select_action(state, memory)
    state = env.reset()
    state[0] = np.random.randint(0, 5)
    state[1] = np.random.randint(0, 5)
    state[2] = np.random.randint(0, 5)
    act_count += 1
    if act_count > 5:
      force_break = True


  state, reward, done= env.step(action)
  sim_memory.states.append(state[:9])
  sim_memory.rewards.append(reward)

  sim_sum_rewards += reward
  sim_run_rew_plot.append(sim_sum_rewards)

  if iter % 1000 == 0:
    plot_hover_result(iter, sim_memory.rewards, sim_run_rew_plot, sim_memory.states, True)


  if done:
    reached = True

  iter += 1


#memory.clear_memory()
plot_hover_result(iter, sim_memory.rewards, sim_run_rew_plot, sim_memory.states, True)  



#saving the params for quadsimulation
file_ = open('state_results_hover.pickle', 'ab')      
pickle.dump(sim_memory.states, file_)                      
file_.close()

