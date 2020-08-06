# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3.8.2 64-bit
#     language: python
#     name: python38264bitf0d5344a704f4947979a10b7b3742980
# ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# %matplotlib inline


# D2Q9 channels of Lattice Bolzman Method
C_SET = np.array([ 
    [0,0],
    [1,0],
    [0,1],
    [-1,0],
    [0,-1],
    [1,1],
    [-1,1],
    [-1,-1],
    [1,-1],
])
C_FREEDOM, C_DIMENSION = np.shape(C_SET)
C_WEIGHT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
C_ALL = slice(None)
C_UP, C_DOWN = [2, 5, 6], [4, 7, 8] ## one to one matched as inverse directions
C_LEFT, C_RIGHT = [3, 6, 7], [1, 8, 5] ## one to one matched as inverse directions


class BaseFlow(object):
    "make it possilbe to slice the flow, flow property"
    
    def __init__(self, population):
        self.population = population
    
    @property
    def mass(self):
        return np.einsum("f...->...", self.population)
    
    @property
    def velocity(self):
        momentum = np.einsum("fd,f...->d...", C_SET, self.population)
        divisible_mass = np.copy(self.mass)
        divisible_mass[divisible_mass == 0] = np.inf ## divide by infinity gives back 0
        return momentum / divisible_mass[np.newaxis,...]
    
    @property
    def equilibrium(self):
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, self.velocity)
        u_square = np.einsum("d...,d...->...", self.velocity, self.velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, self.mass, in_brackets)
        
    @staticmethod
    def compute_equilibrium(mass, velocity):
        assert np.shape(mass) == np.shape(velocity)[1:]
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, velocity)
        u_square = np.einsum("d...,d...->...", velocity, velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, mass, in_brackets)
    
    def __getitem__(self, index):
        assert len(index) == 2
        return BaseFlow(self.population[slice(None), index[0], index[1]])
    
    def flowing(self, time):
        assert isinstance(time, int) and time > 0
        for count in range(time):
            self.onestep_flow()
    
    def onestep_flow(self):
        pass


# +
class KarmanVortex(BaseFlow):
    "flow data, operator and plot"
        
    def __init__(self, mass, velocity, *, omega=1.0, blockage_ratio):
#         assert in compute_equilibrium
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
        length, width = np.shape(self.mass)
        assert length % 4 == 0 and width % 2 == 0
        assert blockage_ratio > 0 and blockage_ratio < 1
        self.obstacle_height = round(width*blockage_ratio)
    
    def onestep_flow(self):
#         collision
        self.population += self.omega*(self.equilibrium - self.population)
        
#         rigid plate at (x=first quartile, y=middle)
#         decompose obstacle interaction area into six parts
#             left_top_edge    2|5  right_top_edge
#                              1|4
#             left_side        1|4  right_side
#                              1|4
#             left_bottom_edge 3|6  right_bottom_edge
        length, width = np.shape(self.mass)
        half_obstacle = int(np.ceil(self.obstacle_height/2))
        left_side = self[length//4 - 1, width//2 - half_obstacle + 1 : width//2 + half_obstacle - 1]
        right_side = self[length//4, width//2 - half_obstacle + 1 : width//2 + half_obstacle - 1]
        left_top_edge = self[length//4 - 1, width//2 + half_obstacle - 1]
        left_bottom_edge = self[length//4 - 1, width//2 - half_obstacle]
        right_top_edge = self[length//4, width//2 + half_obstacle - 1]
        right_bottom_edge = self[length//4, width//2 - half_obstacle]
        
#         copy the relative population before streaming
        obstacle_is_odd = (self.obstacle_height % 2 == 1)
        left_sketch = left_side.population[C_RIGHT].copy()
        BLOCK = [8] if obstacle_is_odd else [1,8]
        left_top_sketch = left_top_edge.population[BLOCK].copy()
        BLOCK = [5] if obstacle_is_odd else [1,5]
        left_bottom_sketch = left_bottom_edge.population[BLOCK].copy()
        right_sketch = right_side.population[C_LEFT].copy()
        BLOCK = [7] if obstacle_is_odd else [3,7]
        right_top_sketch = right_top_edge.population[BLOCK].copy()
        BLOCK = [6] if obstacle_is_odd else [3,6]
        right_bottom_sketch = right_bottom_edge.population[BLOCK].copy()
        
#         pure streaming
        for count in range(1, C_FREEDOM):
            self.population[count] = np.roll(self.population[count], C_SET[count,:], axis=(0,1))        
        
#         recover the population after streaming
        left_side.population[C_LEFT] = left_sketch
        BOUNCE = [6] if obstacle_is_odd else [3,6]
        left_top_edge.population[BOUNCE] = left_top_sketch
        BOUNCE = [7] if obstacle_is_odd else [3,7]
        left_bottom_edge.population[BOUNCE] = left_bottom_sketch
        right_side.population[C_RIGHT] = right_sketch
        BOUNCE = [5] if obstacle_is_odd else [1,5]
        right_top_edge.population[BOUNCE] = right_top_sketch
        BOUNCE = [8] if obstacle_is_odd else [1,8]
        right_bottom_edge.population[BOUNCE] = right_bottom_sketch
                
#         left inlet
        MASS_IN = np.ones_like(self[0,:].mass)
        VELOCITY_IN = np.zeros_like(self[0,:].velocity)
        VELOCITY_IN[0] = 0.1
        self[0,:].population[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        
#         right outlet
        self[-1,:].population[C_LEFT] = self[-2,:].population[C_LEFT].copy()


# -

# ## test ideas

# +
LENGTH, WIDTH = 420, 180
mass = np.ones((LENGTH, WIDTH))
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
velocity[0] = 0.1 ## u_x
velocity[:,:,WIDTH//2:] += 10**(-6) ## perturbation on upper half velocity
VISCOSITY = 0.04
omega = 2 / (6*VISCOSITY + 1)
karman_vortex = KarmanVortex(mass, velocity, omega=omega, blockage_ratio=2/9)

time_axis = []
velocity_axis = []
for time in range(100000):
    karman_vortex.flowing(1)
    time_axis.append(time)
    u_norm = np.linalg.norm(karman_vortex[3*LENGTH//4, WIDTH//2].velocity)
    velocity_axis.append(u_norm)
    
    if time % 100 == 0:
        fig, ax = plt.subplots()
        ax.set_title("velocity magnitude at {} timestep".format(time))
        colornorm = mcolors.TwoSlopeNorm(vcenter=0.1, vmin=0.0)
        velocity_norm = np.linalg.norm(karman_vortex.velocity, axis=0).T
        colormap = ax.imshow(velocity_norm, norm=colornorm)
        fig.colorbar(colormap)
        plt.show()

# plot velocity magnitude evolution
plt.figure(figsize=(6.4*2, 4.8))
plt.title("velocity evolution at [{}, {}]".format(3*LENGTH//4, WIDTH//2))
plt.xlabel("time [step]")
plt.ylabel("velocity magnitude [unit?]")
plt.plot(time_axis, velocity_axis, "b-", label="numerical")
plt.legend()
plt.show()
# -

# ## good sample

# +
LENGTH, WIDTH = 420, 180
mass = np.ones((LENGTH, WIDTH))
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
velocity[0] = 0.1 ## u_x
velocity[:,:,WIDTH//2:] += 10**(-6) ## perturbation on upper half velocity
VISCOSITY = 0.04
omega = 2 / (6*VISCOSITY + 1)
karman_vortex = Flow(mass, velocity, omega=omega, boundary="karman vortex")

time_axis = []
velocity_axis = []
for count in range(100000):  
    time = karman_vortex.time + 1
    karman_vortex.flow_to_time(time)
    time_axis.append(karman_vortex.time)
    u_norm = np.linalg.norm(karman_vortex[3*LENGTH//4, WIDTH//2].velocity)
    velocity_axis.append(u_norm)
    
    if count % 1000 == 0:
        fig, ax = plt.subplots()
        ax.set_title("velocity magnitude at {} timestep".format(karman_vortex.time))
        colormap = ax.imshow(np.linalg.norm(karman_vortex.velocity, axis=0).T)
        fig.colorbar(colormap)
        plt.show()

# plot velocity magnitude evolution
plt.figure(figsize=(6.4*2, 4.8))
plt.title("velocity evolution at [{}, {}]".format(3*LENGTH//4, WIDTH//2))
plt.xlabel("time [step]")
plt.ylabel("velocity magnitude [unit?]")
plt.plot(time_axis, velocity_axis, "b-", label="numerical")
plt.legend()
plt.show()
# -

plt.figure(figsize=(6.4*2, 4.8))
plt.title("velocity evolution at [{}, {}]".format(3*LENGTH//4, WIDTH//2))
plt.xlabel("time [step]")
plt.ylabel("velocity magnitude [unit?]")
plt.plot(time_axis, velocity_axis, "b-", label="numerical")
plt.plot(time_axis, np.ones_like(time_axis)/10, label="0.1")
plt.legend()
plt.show()


