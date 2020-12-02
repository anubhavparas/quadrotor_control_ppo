
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def plot_hover_result(frame_idx, rewards, running_rew, position, plot_running=False):
    clear_output(True)
    plt.figure(figsize=(20,5), tight_layout=2)
    plt.subplot(241)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)

    position = np.array(position)
    plt.subplot(242)
    plt.title('frame %s. x-position: %s' % (frame_idx, position[:, 0].mean()))
    plt.plot(position[:, 0])


    plt.subplot(243)
    plt.title('frame %s. y-position: %s' % (frame_idx, position[-1, 1].mean()))
    plt.plot(position[:, 1])


    plt.subplot(244)
    plt.title('frame %s. z-position: %s' % (frame_idx, position[-1, 2].mean()))
    plt.plot(position[:, 2])


    if plot_running:
      plt.subplot(245)
      plt.title('frame %s. running reward: %s' % (frame_idx, running_rew[-1]))
      plt.plot(running_rew)



    plt.show()



def plot_traj_result(frame_idx, rewards, running_rew, position, plot_running=False):
    clear_output(True)
    plt.figure(figsize=(20,5), tight_layout=2)
    plt.subplot(241)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)

    

    position = np.array(position)
    x_vel = position[:, 6]
    y_vel = position[:, 7]

    plt.subplot(242)
    plt.title('frame %s. x:%s,  y:%s' % (frame_idx, position[-1, 0], position[-1, 1]))
    plt.plot(position[:, 0], position[:, 1])


    plt.subplot(243)
    plt.title('xy vel')
    plt.plot(x_vel)


    plt.subplot(243)
    plt.title('xy vel')
    plt.plot(y_vel)


    if plot_running:
      plt.subplot(245)
      plt.title('frame %s. running reward: %s' % (frame_idx, running_rew[-1]))
      plt.plot(running_rew)



    plt.show()