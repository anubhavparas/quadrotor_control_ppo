## Low-level autonomous control and tracking of quadrotor using reinforcement learning

#

## Instructions to run the code:
- Make sure the following libraries are installed:
    - Python 3.4x
    - Pytorch
    - Numpy
- go to the /code/ folder
- to train and run the hover task:
    - run *$ python hover_train_and_run.py*
- to train and run the trajectory following task:
    - run *$ python traj_train_and_run.py*
- The code can be run in collab if GPU environment to be used
    - [Hover task](https://colab.research.google.com/drive/15vZwsICSzSOU_38TRTV2KuEi7A5ZIt25?authuser=3#scrollTo=fClNsLY6DFvl)
    - [Trajectory following Task](https://colab.research.google.com/drive/1988BaVXe4V81RoQO3oeOvO7hvUTCN6BS?authuser=3#scrollTo=kmx_8l29LWJK)


#
## Results:

Hovering Task:

![alt text](./results/hovering_q.gif?raw=true "Hovering quad reaching a point in space")




Trajectory following Task:

![alt text](./results/traj_follow_q.gif?raw=true "Hovering quad reaching a point in space")


