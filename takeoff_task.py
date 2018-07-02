from task import Task
import numpy as np

class TakeOffTask(Task):
    
    def __init__(self, init_pose, target_pos=None, runtime=5., action_repeat=1):
        super().__init__(init_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], init_velocities=None, init_angle_velocities=None
                         , runtime=runtime, target_pos=target_pos, action_repeat=action_repeat)
        self.z_bonus = 10
        self.take_off_hight = 10
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        reward = 0
        z_diff = abs(self.sim.pose[2]- self.target_pos[2]) #negative means current lower than target
        x_diff = abs(self.sim.pose[0]- self.target_pos[0])
        y_diff = abs(self.sim.pose[1]- self.target_pos[1])
        # z_factor = self.z_bonus if z_diff >= 0 else 5.0 #z_bonus is 5.0

        
        # reward = reward - x_diff - y_diff - z_diff
        reward = reward + 1/(x_diff + 0.01) + 1/(y_diff + 0.01) +1/(z_diff + 0.001)
        
        reward = reward - 2*abs(self.sim.v[0]) - 2*abs(self.sim.v[1])
        vz = self.sim.v[2]
        if self.sim.v[2] <= 0:
            reward-=1
        else: #the faster this is, the higher the reward
            reward += vz
       
        #toReturn = np.tanh(reward) #normalize return to [-1,1]
        
        
        return reward
