import numpy as np
import csv


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def body_to_earth_frame(ii, jj, kk):
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.transpose(R)


class QuadDynamics():
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5.):
        self.init_pose = init_pose
        self.init_velocities = init_velocities
        self.init_angle_velocities = init_angle_velocities
        self.runtime = runtime

        self.gravity = -9.81  # m/s
        
        self.dt = 1 / 50.0  # Timestep
    
        

        # mass, len_to_rotor, moments of inertia - as per the paper
        self.mass = 0.665  # 0.665 kg 
        self.len_to_rotor = 0.105 # motor to motor length = 21cm

        I_x = 0.0023   # kg-m^2
        I_y = 0.0025
        I_z = 0.0037
        
        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        env_bounds = 300.0  # 300 m / 300 m / 300 m
        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds/2])

        self.reset()

    def reset(self):
        self.time = 0.0
        self.pose = np.array([6.0, 6.0, 6.0, 0.0, 0.0, 0.0]) if self.init_pose is None else np.copy(self.init_pose)
        self.v = np.array([0.0, 0.0, 0.0]) if self.init_velocities is None else np.copy(self.init_velocities)
        self.angular_v = np.array([0.0, 0.0, 0.0]) if self.init_angle_velocities is None else np.copy(self.init_angle_velocities)
        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.done = False
        self.state = np.array(list(self.pose) + list(self.v) + list(self.angular_v))
        self.state_size = self.state.shape[0]

    
    def get_linear_forces(self, thrusts):
        # Gravity
        gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        # Thrust
        thrust_body_force = np.array([0, 0, sum(thrusts)])

        R = body_to_earth_frame(*list(self.pose[3:]))
        linear_forces = R @ thrust_body_force
        linear_forces += gravity_force
        return linear_forces

    def get_moments(self, thrusts):
        thrust_moment = np.array([(thrusts[3] - thrusts[2]) * self.len_to_rotor,
                            (thrusts[1] - thrusts[0]) * self.len_to_rotor,
                            0])
        return thrust_moment

    
    
    def update(self, rotor_thrusts, penalty_rew=-1):
        if np.any(np.isnan(rotor_thrusts)):
          print('NAN thrusts', rotor_thrusts)

        thrusts = rotor_thrusts
        self.linear_accel = self.get_linear_forces(thrusts) / self.mass

        position = self.pose[:3] + self.v * self.dt + 0.5 * self.linear_accel * self.dt**2
        self.v += self.linear_accel * self.dt

        moments = self.get_moments(thrusts)

        self.angular_accels = moments / self.moments_of_inertia
        angles = self.pose[3:] + self.angular_v * self.dt + 0.5 * self.angular_accels * self.angular_accels * self.dt ** 2
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        self.angular_v = self.angular_v + self.angular_accels * self.dt

        new_positions = []
        penalty = 0
        for ii in range(3):
            #new_positions.append(position[ii])
            
            if position[ii] <= self.lower_bounds[ii]:
                new_positions.append(self.lower_bounds[ii])
                #self.done = True
                penalty = penalty_rew
            elif position[ii] > self.upper_bounds[ii]:
                new_positions.append(self.upper_bounds[ii])
                #self.done = True
                penalty = penalty_rew
            else:
                new_positions.append(position[ii])
            

        self.pose = np.array(new_positions + list(angles))
        self.state = np.array(list(self.pose) + list(self.v) + list(self.angular_v))
        self.time += self.dt
        #if self.time > self.runtime:
        #    self.done = True
        if np.any(np.isnan(self.state)):
          print('NAN State', self.state, rotor_thrusts)
        
        return self.state, self.done, penalty