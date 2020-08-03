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

# ## import

import matplotlib
matplotlib.get_backend()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
# %matplotlib inline


# ## global constants

# D2Q9 channels of Lattice Bolzman Method
C_SET = np.array([ 
    (0,0),
    (1,0),
    (0,1),
    (-1,0),
    (0,-1),
    (1,1),
    (-1,1),
    (-1,-1),
    (1,-1),
])
C_WEIGHT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
C_FREEDOM, C_DIMENSION = np.shape(C_SET)
C_UP, C_DOWN = [2, 5, 6], [4, 7, 8] ## one to one matched as inverse directions
C_LEFT, C_RIGHT = [3, 6, 7], [1, 8, 5] ## one to one matched as inverse directions


# ## base class

class BaseFlow(object):
    "Flow data, property, slice, flowing"
    
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
        assert np.shape(mass) == np.shape(velocity[0,...])
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, velocity)
        u_square = np.einsum("d...,d...->...", velocity, velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, mass, in_brackets)
    
    def __getitem__(self, index):
        assert len(index) == 2
        return BaseFlow(self.population[slice(None), index[0], index[1]])
    
    def flowing(self, num_time):
        assert isinstance(num_time, int) and num_time > 0
        for count in range(num_time):
            self.onestep_flow()
            assert np.all(self.mass > 0), np.where(self.mass <= 0)
    
    def onestep_flow(self):
        pass
    
    def plot_flow(self):
#         plot mass
        plt.figure()
        plt.title("mass colormap")
        plt.imshow(self.mass.T, origin="lower") ## transpose due to NumPy mechanics, the same as below
        plt.colorbar()
        plt.show()
#         plot velocity
        plt.figure()
        plt.title("velocity streamline")
        x_length, y_length = np.shape(self.mass)
        plt.streamplot(np.arange(x_length), np.arange(y_length), self.velocity[0].T, self.velocity[1].T)
        plt.show()


# ## Milestone 1&2

class Scatter(BaseFlow):
            
    def __init__(self, mass, velocity, omega=1.0):
        assert np.shape(mass) == np.shape(velocity[0])
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         streaming
        for count in range(1, C_FREEDOM):
            self.population[count] = np.roll(self.population[count], C_SET[count,:], axis=(0,1))
    
    def plot_flow(self):
#         plot mass
        plt.figure()
        plt.title("mass colormap at time = {} step".format(self.time))
        plt.imshow(self.mass.T, origin="lower") ## transpose due to NumPy mechanics, the same as above
        plt.colorbar()
        plt.show()
#         plot velocity
        plt.figure()
        plt.title("velocity streamline at time = {} step".format(self.time))
        x_length, y_length = np.shape(self.mass)
        plt.streamplot(np.arange(x_length), np.arange(y_length), self.velocity[0].T, self.velocity[1].T)
        plt.show()


LENGTH, WIDTH = 70, 30
mass = np.ones([LENGTH, WIDTH]) / 2
mass[LENGTH//2, WIDTH//2] += 1/20
velocity = np.zeros([C_DIMENSION, LENGTH, WIDTH])
middle_heavier = Scatter(mass, velocity)
for time in range(1000+1):
    middle_heavier.flowing(1)
    if time in [0, 10, 100, 1000]:
        fig, ax = plt.subplots()
        ax.set_title("mass colormap at time = {} step".format(time))
        colormap = ax.imshow(middle_heavier.mass.T, origin="lower")
        fig.colorbar(colormap)
        plt.show()


# ## Milestone 3 case 1

class ShearWave(BaseFlow):
        
    def __init__(self, mass, velocity, omega=1.0):
        assert np.shape(mass) == np.shape(velocity[0])
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         streaming
        for count in range(1, C_FREEDOM):
            self.population[count] = np.roll(self.population[count], C_SET[count,:], axis=(0,1))
    
    def plot_flow(self):
#         plot mass
        plt.figure()
        plt.title("mass colormap at time = {} step".format(self.time))
        plt.imshow(self.mass.T, origin="lower") ## transpose due to NumPy mechanics, the same as above
        plt.colorbar()
        plt.show()
#         plot velocity
        plt.figure()
        plt.title("velocity streamline at time = {} step".format(self.time))
        x_length, y_length = np.shape(self.mass)
        plt.streamplot(np.arange(x_length), np.arange(y_length), self.velocity[0].T, self.velocity[1].T)
        plt.show()


# +
LENGTH, WIDTH = 70, 30
MASS_STEADY = 0.5
mass = np.ones((LENGTH, WIDTH)) * MASS_STEADY
for y in range(WIDTH):
    EPSILON = 0.1
    mass[:,y] += EPSILON * np.sin(2*np.pi*y / WIDTH)
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
mass_shear_wave = ShearWave(mass, velocity)
mass_shear_wave.plot_flow()

ZETA_SQUARE = (2*np.pi/WIDTH)**2
omega_axis = np.linspace(0.1, 1.9, num=19)
viscosity_analytic = 1/3 * (1/omega_axis - 1/2)
viscosity_numeric = []
for count in range(omega_axis.size):
    omega = round(omega_axis[count], 1)
    mass_shear_wave = Flow(mass, velocity, omega=omega)
    TOTAL_TIME = 1000
    time_axis = range(TOTAL_TIME+1) ## 0 to total_time
    amplitude = []
    viscosity = []
    for time in time_axis:
        mass_shear_wave.flow_to_time(time)
        amplitude.append(np.max(np.abs(mass_shear_wave.mass - MASS_STEADY)))
        if time != 0: ## cannot divide 0
            viscosity.append(-np.log(amplitude[time] / amplitude[0]) / (ZETA_SQUARE * time))
    viscosity_numeric.append(sum(viscosity) / len(viscosity))
    
#     draw decay curve
    plt.figure()
    plt.title("decay of mass oscillation at omega = {}".format(omega))
    plt.xlabel("timesteps [unit time]")
    plt.ylabel("maximal mass [unit mass]")
    plt.plot(time_axis, amplitude, "r-", label="numeric")
    decay = amplitude[0] * np.exp(-viscosity_analytic[count] * ZETA_SQUARE * time_axis)
    plt.plot(time_axis, decay, "b--", label="analytic")
    plt.legend()
    plt.show()

# draw relation curve
plt.figure()
plt.title("viscosity - omega relation")
plt.xlabel("omega [none]")
plt.ylabel("viscosity [unit]")
plt.plot(omega_axis, viscosity_numeric, "ro-", label="numeric")
plt.plot(omega_axis, viscosity_analytic, "b-", label="analytic")
plt.legend() 
plt.show()

# -

# ## Milestone 3 case 2

# +
LENGTH, WIDTH = 70, 30
mass = np.ones((LENGTH, WIDTH))
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
EPSILON = 0.1
for y in range(WIDTH):
    velocity[0,:,y] = EPSILON * np.sin(2*np.pi*y / WIDTH) #  u_x
velocity_shear_wave = ShearWave(mass, velocity)
velocity_shear_wave.plot_flow()

ZETA_SQUARE = (2*np.pi/WIDTH)**2
omega_axis = np.linspace(0.1, 1.9, num=19)
viscosity_analytic = 1/3 * (1/omega_axis - 1/2)
viscosity_numeric = []
for count in range(omega_axis.size):
    omega = round(omega_axis[count], 1)
    velocity_shear_wave = Flow(mass, velocity, omega=omega)
    TOTAL_TIME = 100
    time_axis = range(TOTAL_TIME+1) ## 0 to total_time
    amplitude = []
    viscosity = []
    for time in time_axis:
        velocity_shear_wave.flow_to_time(time)
        amplitude.append(np.max(velocity_shear_wave.velocity[0,:,:]))
        if time != 0: ## cannot divide 0
            viscosity.append(-np.log(amplitude[time] / amplitude[0]) / (ZETA_SQUARE * time))
    viscosity_numeric.append(sum(viscosity) / len(viscosity))

#     draw decay of big deviation
    plt.figure()
    plt.title("decay of velocity oscillation at omega = {}".format(omega))
    plt.xlabel("time [step]")
    plt.ylabel("maximal velocity [unit velocity]")
    plt.plot(time_axis, amplitude, "r-", label="numeric")
    decay = amplitude[0] * np.exp(-viscosity_analytic[count] * ZETA_SQUARE * time_axis)
    plt.plot(time_axis, decay, "b--", label="analytic")
    plt.legend()
    plt.show()

# draw relation curve
plt.figure()
plt.title("viscosity - omega relation")
plt.xlabel("omega [none]")
plt.ylabel("viscosity [unit]")
plt.plot(omega_axis, viscosity_numeric, "ro-", label="numeric")
plt.plot(omega_axis, viscosity_analytic, "b-", label="analytic")
plt.legend() 
plt.show()


# -

# ## Milestone 4

class CouetteFlow(BaseFlow):
        
    def __init__(self, mass, velocity, omega=1.0):
        assert np.shape(mass) == np.shape(velocity[0])
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         before streaming
        top_sketch = np.copy(self[:,-1].population[C_UP])
        bottom_sketch = np.copy(self[:,0].population[C_DOWN])
#         pure streaming
        for count in range(1, C_FREEDOM):
            self.population[count] = np.roll(self.population[count], C_SET[count,:], axis=(0,1))
#         after streaming
        WALL_VELOCITY = (0.1, 0.)
        loss = 6 * np.einsum("f,x,fd,d->fx", C_WEIGHT[C_UP], self[:,-1].mass, C_SET[C_UP], WALL_VELOCITY)
        self[:,-1].population[C_DOWN] = top_sketch - loss
        self[:,0].population[C_UP] = bottom_sketch    
    
    def plot_flow(self):
#         plot mass
        plt.figure()
        plt.title("mass colormap")
        plt.imshow(self.mass.T, origin="lower") ## transpose due to NumPy mechanics, the same as above
        plt.colorbar()
        plt.show()
#         plot velocity
        plt.figure()
        plt.title("velocity streamline")
        x_length, y_length = np.shape(self.mass)
        plt.streamplot(np.arange(x_length), np.arange(y_length), self.velocity[0].T, self.velocity[1].T)
        plt.show()


# +
LENGTH, WIDTH = 70, 30
mass = np.ones((LENGTH, WIDTH))
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
couette_flow = CouetteFlow(mass, velocity)
couette_flow.plot_flow()

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(6.4*2,4.8)
for time in range(3000+1):
    couette_flow.flowing(1)
#     plot u_x, u_y profile
    if time in [100, 500, 1500, 3000]:
        centerline = couette_flow[LENGTH//2,:]
        y_axis = range(WIDTH)
        ax1.plot(centerline.velocity[0], y_axis, "-", label="{} step".format(time))
        ax1.legend()
        ax1.grid(True)
        ax1.set_title("u_x profile at centerline")
        ax1.set_xlabel("u_x [unit]")
        ax1.set_ylabel("y [unit]")
        ax2.plot(centerline.velocity[1], y_axis, "-", label="{} step".format(time))
        ax2.legend()    
        ax2.set_title("u_y profile at centerline".format(time))
        ax2.set_xlabel("u_y [unit]")
        ax2.set_ylabel("y [unit]")
plt.show()
    
# -

# ## Milestone 5

class PoiseuilleFlow(BaseFlow):
        
    def __init__(self, mass, velocity, *, omega=1.0, delta_mass):
        assert np.shape(mass) == np.shape(velocity[0])
        self.population = self.compute_equilibrium(mass, velocity)
#         ghost cells
        length, width = np.shape(mass)
        self.population = np.insert(self.population, [0,length], 0, axis=1)
        assert omega > 0 and omega < 2
        self.omega = omega
        self.delta_mass = delta_mass
        
    def onestep_flow(self):
#         collision
        region = slice(1,-1), slice(None)
        self[region].population += self.omega * (self[region].equilibrium - self[region].population)
#         before streaming
#         left inlet
        MASS_IN = np.full(np.shape(self[0,:].mass), 1+self.delta_mass)
        self[0,:].population[C_RIGHT] = self.compute_equilibrium(MASS_IN, self[-2,:].velocity)[C_RIGHT]
        self[0,:].population[C_RIGHT] += self[-2,:].population[C_RIGHT] - self[-2,:].equilibrium[C_RIGHT]
#         right outlet
        MASS_OUT = np.full(np.shape(self[-1,:].mass), 1-self.delta_mass)
        self[-1,:].population[C_LEFT] = self.compute_equilibrium(MASS_OUT, self[1,:].velocity)[C_LEFT]
        self[-1,:].population[C_LEFT] += self[1,:].population[C_LEFT] - self[1,:].equilibrium[C_LEFT]
#         top wall
        top_sketch = np.copy(self[:,-1].population[C_UP])
#         bottom wall
        bottom_sketch = np.copy(self[:,0].population[C_DOWN])
#         pure streaming
        for count in range(1, C_FREEDOM):
            self.population[count] = np.roll(self.population[count], C_SET[count,:], axis=(0,1))
#         after streaming
#         top wall
        self[:,-1].population[C_DOWN] = top_sketch
#         bottom wall
        self[:,0].population[C_UP] = bottom_sketch
    
    def plot_flow(self):
#         plot mass
        plt.figure()
        plt.title("mass colormap")
        plt.imshow(self.mass.T, origin="lower") ## transpose due to NumPy mechanics, the same as above
        plt.colorbar()
        plt.show()
#         plot velocity
        plt.figure()
        plt.title("velocity streamline")
        x_length, y_length = np.shape(self.mass)
        plt.streamplot(np.arange(x_length), np.arange(y_length), self.velocity[0].T, self.velocity[1].T)
        plt.show()    


# +
LENGTH, WIDTH = 200, 30
mass = np.ones((LENGTH, WIDTH))
velocity = np.zeros((C_DIMENSION, LENGTH, WIDTH))
DELTA_MASS = 0.04
poiseuille_flow = PoiseuilleFlow(mass, velocity, delta_mass=DELTA_MASS)
poiseuille_flow.plot_flow()

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(6.4*2,4.8)
for time in range(2000+1):
    poiseuille_flow.flowing(1)
#     plot u_x, u_x profiile
    if time % 400 == 0:
        y_axis = range(WIDTH)
        length, width = np.shape(poiseuille_flow.mass)
        centerline = poiseuille_flow[length//2,:]
        ax1.plot(centerline.velocity[0], y_axis, "-", label=" {} step".format(time))
        ax1.legend()
        ax1.grid(True)
        ax1.set_title("u_x profile at centerline")
        ax1.set_xlabel("u_x [unit]")
        ax1.set_ylabel("y [unit]")
        ax2.plot(centerline.velocity[1], y_axis, "-", label=" {} step".format(time))
        ax2.legend(loc='lower right')
        ax2.set_title("u_y profile at centerline")
        ax2.set_xlabel("u_y [unit]")
        ax2.set_ylabel("y [unit]")

length, width = np.shape(poiseuille_flow.mass)
centerline = poiseuille_flow[length//2,:]
y_analytic = np.linspace(0-0.5, WIDTH-0.5)
niu = 1/3*(1/poiseuille_flow.omega - 1/2)
miu = np.mean(centerline.mass) * niu
# mass drop is 0.05, pressure drop is 0.05/3
u_x = 1/(2*miu) * (2*DELTA_MASS/3/length) * (y_analytic+0.5) * (width-0.5 - y_analytic)
ax1.plot(u_x, y_analytic, "--", label="analytic")
ax1.legend(loc='lower right')

plt.figure()
plt.title("mass profile along x-direction")
x_axis = np.arange(length)
flow_slice = poiseuille_flow[:,WIDTH//2]
plt.plot(x_axis[1:-1], flow_slice.mass[1:-1], label=" {} step".format(time))
mass_analytic = 1 + DELTA_MASS - 2*DELTA_MASS/length*x_axis
plt.plot(x_axis, mass_analytic,"--", label="analytic")
plt.legend(loc='lower right')
# -



