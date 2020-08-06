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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
C_UP = np.array([2, 5, 6], dtype="intp")
C_DOWN = np.array([4, 7, 8], dtype="intp")
C_LEFT = np.array([3, 6, 7], dtype="intp")
C_RIGHT = np.array([1, 8, 5], dtype="intp")
C_ALL = slice(None)


# ## base class

class BaseFlow(object):
    "Flow data, property, slice, flowing"
    
    def __init__(self, population):
        assert population.shape[0] == C_FREEDOM
        self.population = population
        
    # only for the ease of getting attribute for a slice
    def __getitem__(self, index):
        assert len(index) == 2
        return BaseFlow(self.population[:, index[0], index[1]])
    
    @property
    def mass(self):
        return np.einsum("f...->...", self.population)
    
    @property
    def velocity(self):
        momentum = np.einsum("fd,f...->d...", C_SET, self.population)
        divisible_mass = np.copy(self.mass)
        # divide by infinity gives back 0
        divisible_mass[divisible_mass == 0] = np.inf
        return momentum / divisible_mass[np.newaxis,...]
    
    @property
    def equilibrium(self):
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, self.velocity)
        u_square = np.einsum("d...,d...->...", self.velocity, self.velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, self.mass, in_brackets)
        
    @staticmethod
    def compute_equilibrium(mass, velocity):
        assert mass.shape == velocity.shape[1:]
        assert velocity.shape[0] == C_DIMENSION
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, velocity)
        u_square = np.einsum("d...,d...->...", velocity, velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, mass, in_brackets)
    
    def flowing(self, num_time):
        assert isinstance(num_time, int) and num_time >= 0
        for count in range(num_time):
            self.onestep_flow()
            assert np.all(self.mass > 0), "empty at {}".format(np.where(self.mass <= 0))
            self.record_data()
    
    def onestep_flow(self):
        pass
    
    def record_data(self):
        pass


# ## Milestone 1&2

class MidddleHeavier(BaseFlow):
        
    def __init__(self, shape, *, omega=1.0):
        assert len(shape) == C_DIMENSION
        mass = np.full(shape, 0.5)
        mass[shape[0]//2, shape[1]//2] *= 1.2
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))


# +
shape = 70, 30
middle_heavier = MidddleHeavier(shape)

fig, ax = plt.subplots()
ax.set_title("initial mass density")
colormap = ax.imshow(middle_heavier.mass.T, origin="lower")
fig.colorbar(colormap)
plt.show()

num_time = 100
middle_heavier.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards mass density")
colormap = ax.imshow(middle_heavier.mass.T, origin="lower")
fig.colorbar(colormap)


# -


# ## Milestone 3 case 1

class MassShearWave(BaseFlow):
        
    def __init__(self, shape, *, omega=1.0): 
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        sin_2pi = lambda t: np.sin(2*np.pi*t)
        x = np.arange(self.length)
        mass_pattern = 0.5 + 0.05*sin_2pi(x/self.length)
        mass = np.repeat(mass_pattern, self.width).reshape(shape)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
        self.time = 0
        self.marks = [self.time]
        self.rho_profiles = [self[:,0].mass]
        self.ux_profiles = [self[:,0].velocity[0]]
        self.uy_profiles = [self[:,0].velocity[1]]
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))
    
    def record_data(self):
        self.time += 1
        self.marks.append(self.time)
        self.rho_profiles.append(self[:, 0].mass)
        self.ux_profiles.append(self[:, 0].velocity[0])
        self.uy_profiles.append(self[:, 0].velocity[1])


# +
shape = 70, 30
mass_shear_wave = MassShearWave(shape)

fig, ax = plt.subplots()
ax.set_title("initial rho")
colormap = ax.imshow(mass_shear_wave.mass.T, origin="lower")
fig.colorbar(colormap)

num_time = 2000
mass_shear_wave.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards rho")
colormap = ax.imshow(mass_shear_wave.mass.T, origin="lower")
fig.colorbar(colormap)
# -

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(6.4, 4.8*3)
for time in mass_shear_wave.marks:
    if time % (num_time // 5) == 0:
        x_axis = range(mass_shear_wave.length)
        ax1.plot(x_axis, mass_shear_wave.rho_profiles[time], label="t={}".format(time))
        ax2.plot(x_axis, mass_shear_wave.ux_profiles[time], label="t={}".format(time))
        ax3.plot(x_axis, mass_shear_wave.uy_profiles[time], label="t={}".format(time))
ax1.plot(x_axis, np.ones_like(x_axis)*0.5, "--", label="0.5")
ax1.legend(loc="best")
ax1.set_title("rho profile evolution")
ax2.legend(loc="best")
ax2.set_title("ux profile evolution")
ax3.legend(loc="best")
ax3.set_title("uy profile evolution")
ax3.set_ylim(-1e-6, 1e-6)

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(6.4, 4.8*2)
t = np.array(mass_shear_wave.marks)
rho_amplitude = []
for rho_profile in mass_shear_wave.rho_profiles:
    rho_amplitude.append(rho_profile.max() - 0.5)
ax1.plot(t, rho_amplitude, "b+", label="numeric")
ux_amplitude = []
for ux_profile in mass_shear_wave.ux_profiles:
    ux_amplitude.append(ux_profile.max())
ax2.plot(t, ux_amplitude, "b+", label="numeric")
niu = 1/3 * (1/mass_shear_wave.omega - 1/2)
zeta = 2*np.pi/mass_shear_wave.length
rho_amplitude = rho_amplitude[0] * np.exp(-niu * zeta**2 * t)
ax1.plot(t, rho_amplitude, "r-", label="analytic")
ax1.plot(t, np.zeros_like(t), "c-.", label="0")
ax1.legend(loc="best")
ax1.set_title("rho amplitude evolution")
ux_max = np.max(ux_amplitude)
index_max = np.argmax(ux_amplitude)
ux_amplitude = ux_max * np.exp(-niu * zeta**2 * (t - t[index_max]))
ax2.plot(t, ux_amplitude, "r-", label="analytic")
ax2.plot(t, np.zeros_like(t), "c-.", label="0")
ax2.legend(loc="best")
ax2.set_title("ux amplitude evolution")


# +
# omega_axis = np.linspace(0.1, 1.9, num=19).round(decimals=1)
# viscositx_axis = np.empty(19)
# for omega in omega_axis:
#     mass_shear_wave = MassShearWave(shape, omega=omega)
#     num_time = 500
#     mass_shear_wave.flowing(num_time)
#     amplitude = np.empty(num_time+1)
#     for time in mass_shear_wave.mark:
#         amplitude[time] = mass_shear_wave.mass_profile[time].max()
#     drho_dt = np.diff(amplitude).mean()
#     np.append(viscositx_axis, viscosity)
# ax.plot(omega_axis, viscositx_axis, "bo", label="numeric")
# ax.legend(loc="best")
# plt.show()

# # draw relation curve
# plt.figure()
# plt.title("viscosity - omega relation")
# plt.xlabel("omega [none]")
# plt.ylabel("viscosity [unit]")
# plt.plot(omega_axis, viscosity_numeric, "bo-", label="numeric")
# viscosity_analytic = 1/3 * (1/omega_axis - 1/2)
# plt.plot(omega_axis, viscosity_analytic, "r--", label="analytic")
# plt.legend() 
# plt.show()
# -

# ## Milestone 3 case 2

class VelocityShearWave(BaseFlow):
        
    def __init__(self, shape, *, omega=1.0): 
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        mass = np.full(shape, 0.5)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        sin_2pi = lambda t: np.sin(2*np.pi*t)
        y = np.arange(self.width)
        velicty_pattern = 0.1*sin_2pi(y/self.width)
        velocity[0] = np.broadcast_to(velicty_pattern, shape)
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
        self.time = 0
        self.marks = [self.time]
        self.rho_profiles = [self[0,:].mass]
        self.ux_profiles = [self[0,:].velocity[0]]
        self.uy_profiles = [self[0,:].velocity[1]]
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))
    
    def record_data(self):
        self.time += 1
        self.marks.append(self.time)
        self.rho_profiles.append(self[0,:].mass)
        self.ux_profiles.append(self[0,:].velocity[0])
        self.uy_profiles.append(self[0,:].velocity[1])


# +
shape = 70, 30
velocity_shear_wave = VelocityShearWave(shape)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(velocity_shear_wave.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_time = 500
velocity_shear_wave.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(velocity_shear_wave.velocity[0].T, origin="lower")
fig.colorbar(colormap)
# -

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(6.4, 4.8*3)
for time in velocity_shear_wave.marks:
    if time % (num_time // 5) == 0:
        y_axis = range(velocity_shear_wave.width)
        ax1.plot(velocity_shear_wave.rho_profiles[time], y_axis, label="t={}".format(time))
        ax2.plot(velocity_shear_wave.ux_profiles[time], y_axis, label="t={}".format(time))
        ax3.plot(velocity_shear_wave.uy_profiles[time], y_axis, label="t={}".format(time))
ax1.legend(loc="best")
ax1.set_xlim(0.5-1e-6, 0.5+1e-6)
ax1.set_title("rho profile evolution")
ax2.plot(np.zeros_like(y_axis), y_axis, "--", label="0")
ax2.legend(loc="best")
ax2.set_title("ux profile evolution")
ax3.legend(loc="best")
ax3.set_xlim(-1e-6, 1e-6)
ax3.set_title("uy profile evolution")

fig, ax = plt.subplots()
# numerical
t = np.array(velocity_shear_wave.marks)
ux_amplitude = []
for ux_profile in velocity_shear_wave.ux_profiles:
    ux_amplitude.append(ux_profile.max())
ax.plot(t, ux_amplitude, "y+", label="numeric")
# analytic
niu = 1/3 * (1/velocity_shear_wave.omega - 1/2)
zeta = 2*np.pi/velocity_shear_wave.width
ux_max = np.max(ux_amplitude)
index_max = np.argmax(ux_amplitude)
ux_amplitude = ux_max * np.exp(-niu * zeta**2 * (t - t[index_max]))
ax.plot(t, ux_amplitude, "r-", label="analytic")
ax.legend(loc="best")
ax.set_title("ux magnitude evolution")
plt.show()


# ## Milestone 4

class CouetteFlow(BaseFlow):
        
    def __init__(self, shape, *, omega=1.0, wall_velocity):
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        mass = np.full(shape, 1)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        self.population = self.compute_equilibrium(mass, velocity)
        assert omega > 0 and omega < 2
        self.omega = omega
        self.wall_velocity = wall_velocity
        self.time = 0
        self.marks = [self.time]
        centerline = self[self.length//2,:]
        self.rho_profiles = [centerline.mass]
        self.ux_profiles = [centerline.velocity[0]]
        self.uy_profiles = [centerline.velocity[1]]
    
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         before streaming
        top_sketch = self.population[C_UP,:,-1].copy()
        bottom_sketch = self.population[C_DOWN,:,0].copy()
#         pure streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))
#         after streaming
        loss = 6 * np.einsum("f,x,fd,d->fx", C_WEIGHT[C_UP], self[:,-1].mass, C_SET[C_UP], self.wall_velocity)
        self.population[C_DOWN,:,-1] = top_sketch - loss
        self.population[C_UP,:,0] = bottom_sketch    
    
    def record_data(self):
        self.time += 1
        self.marks.append(self.time)
        centerline = self[self.length//2,:]
        self.rho_profiles.append(centerline.mass)
        self.ux_profiles.append(centerline.velocity[0])
        self.uy_profiles.append(centerline.velocity[1])


# +
shape = 70, 30
u_w = (0.1, 0.)
couette_flow = CouetteFlow(shape, wall_velocity=u_w)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(couette_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_time = 3000
couette_flow.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(couette_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)
# -

fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(6.4, 4.8*2)
for time in couette_flow.marks:
    if time in [0, 20, 150, 500, 1000, 3000]:
        y_axis = np.arange(couette_flow.width)
        ax1.plot(couette_flow.ux_profiles[time], y_axis, "-", label="{} step".format(time))
        ax2.plot(couette_flow.uy_profiles[time], y_axis, "-", label="{} step".format(time))
y = np.linspace(-0.5, couette_flow.width-0.5)
ux_analytic = u_w[0] / couette_flow.width * (y+0.5)
x = np.linspace(0, u_w[0])
ax1.plot(x, np.ones_like(x)*(-0.5), "-.", label="top wall")
ax1.plot(x, np.ones_like(x)*(couette_flow.width-0.5), "-.", label="bottom wall")
ax1.plot(ux_analytic, y, "k--", label="analytic")
ax1.legend(loc="lower right")
ax1.set_title("u_x profile at centerline")
ax1.set_xlabel("u_x [unit]")
ax1.set_ylabel("y [unit]")
ax2.legend(loc="best")
ax2.set_xlim(-1e-6, 1e-6)
ax2.set_title("u_y profile at centerline".format(time))
ax2.set_xlabel("u_y [unit]")
ax2.set_ylabel("y [unit]")


# ## Milestone 5

class PoiseuilleFlow(BaseFlow):
        
    def __init__(self, shape, *, omega=1.0, pressure_gradient):
        assert len(shape) == C_DIMENSION
        mass = np.full(shape, 1)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        self.population = self.compute_equilibrium(mass, velocity)
        # add ghost cells
        self.population = np.insert(self.population, [0,shape[0]], 0, axis=1)
        self.length, self.width = self.mass.shape
        assert pressure_gradient >= 0
        self.delta_mass = pressure_gradient * self.length/2 * 3 
        assert omega > 0 and omega < 2
        self.omega = omega
        self.time = 0
        self.marks = [self.time]
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles = [another_centerline.mass]
        self.ux_profiles = [centerline.velocity[0]]
        self.uy_profiles = [centerline.velocity[1]]
        
    def onestep_flow(self):
#         collision
        self.population += self.omega * (self.equilibrium - self.population)
#         before streaming
#         top wall
        top_sketch = self[:,-1].population[C_UP].copy()
#         bottom wall
        bottom_sketch = self[:,0].population[C_DOWN].copy()
#         left inlet
        MASS_IN = np.full(np.shape(self[0,:].mass), 1+self.delta_mass)
        self[0,:].population[C_RIGHT] = self.compute_equilibrium(MASS_IN, self[-2,:].velocity)[C_RIGHT]
        self[0,:].population[C_RIGHT] += self[-2,:].population[C_RIGHT] - self[-2,:].equilibrium[C_RIGHT]
#         right outlet
        MASS_OUT = np.full(np.shape(self[-1,:].mass), 1-self.delta_mass)
        self[-1,:].population[C_LEFT] = self.compute_equilibrium(MASS_OUT, self[1,:].velocity)[C_LEFT]
        self[-1,:].population[C_LEFT] += self[1,:].population[C_LEFT] - self[1,:].equilibrium[C_LEFT]
#         pure streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))
#         after streaming
#         top wall
        self[:,-1].population[C_DOWN] = top_sketch
#         bottom wall
        self[:,0].population[C_UP] = bottom_sketch
    
    def record_data(self):
        self.time += 1
        self.marks.append(self.time)
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles.append(another_centerline.mass)
        self.ux_profiles.append(centerline.velocity[0])
        self.uy_profiles.append(centerline.velocity[1])


# +
shape = 70, 30
dp_dx = 1e-4
poiseuille_flow = PoiseuilleFlow(shape, pressure_gradient=dp_dx)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(poiseuille_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_time = 3000
poiseuille_flow.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(poiseuille_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)


# +
fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(6.4, 4.8*2)
for time in poiseuille_flow.marks:
    if time in [0, 100, 300, 500, 1000, 2000, 3000]:
        y_axis = range(poiseuille_flow.width)
        ax1.plot(poiseuille_flow.ux_profiles[time], y_axis, label="{} step".format(time))
        ax2.plot(poiseuille_flow.uy_profiles[time], y_axis, label="{} step".format(time))

y = np.linspace(-0.5, poiseuille_flow.width-0.5)
niu = 1/3*(1/poiseuille_flow.omega - 1/2)
miu = poiseuille_flow.rho_profiles[-1].mean() * niu
ux_analytic = 1/(2*miu) * dp_dx * (y + 0.5) * (poiseuille_flow.width-0.5 - y)
ax1.plot(ux_analytic, y, "k--", label="analytic")
x = np.linspace(0, ux_analytic.max())
ax1.plot(x, np.ones_like(x)*(-0.5), "-.", label="top wall")
ax1.plot(x, np.ones_like(x)*(poiseuille_flow.width-0.5), "-.", label="bottom wall")
ax1.legend(loc="center left")
ax1.set_title("u_x profile at centerline")
ax1.set_xlabel("u_x [unit]")
ax1.set_ylabel("y [unit]")
ax2.legend(loc="best")
ax2.set_title("u_y profile at centerline")
ax2.set_xlabel("u_y [unit]")
ax2.set_ylabel("y [unit]")
# -

fig, ax = plt.subplots()
x_axis = range(poiseuille_flow.length)
ax.plot(x_axis, poiseuille_flow.rho_profiles[-1], label="numerical".format(time))
x = np.linspace(-1, poiseuille_flow.length)
rho_analytic = -(3*dp_dx) * (x - poiseuille_flow.length//2) + 1
ax.plot(x, rho_analytic,"k--", label="analytic")
ax.legend(loc='best')
ax.set_title("rho profile")


# ## Milestone 6

# +
class KarmanVortex(BaseFlow):
    "flow data, operator and plot"
        
    def __init__(self, shape, *, viscosity, blockage_ratio):
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        mass = np.full(shape, 1)
        velocity = np.full(shape, (0.1, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)
        # perturbation
        velocity[:,:,self.width//2:] += 10**(-6)
        self.population = self.compute_equilibrium(mass, velocity)
        
        assert viscosity > 0
        self.omega = 2 / (6*viscosity + 1)
        
        length, width = np.shape(self.mass)
        assert length % 4 == 0 and width % 2 == 0
        assert blockage_ratio > 0 and blockage_ratio < 1
        self.obstacle_size = round(width*blockage_ratio)
        self.obstacle_is_odd = (self.obstacle_size % 2 == 1)
#         obstacle at (x=first quartile, y=middle)
#         decompose obstacle interaction area into six parts
#             left_top_edge    2|5  right_top_edge
#                              1|4
#             left_side        1|4  right_side
#                              1|4
#             left_bottom_edge 3|6  right_bottom_edge
        obstacle_half_size = np.ceil(self.obstacle_size/2).astype("int")
        self.left_side = self[length//4-1, width//2-obstacle_half_size+1 : width//2+obstacle_half_size- 1]
        self.left_top_edge = self[length//4-1, width//2+obstacle_half_size-1]
        self.left_bottom_edge = self[length//4-1, width//2-obstacle_half_size]
        self.right_side = self[length//4, width//2-obstacle_half_size+1 : width//2+obstacle_half_size-1]
        self.right_top_edge = self[length//4, width//2+obstacle_half_size-1]
        self.right_bottom_edge = self[length//4, width//2-obstacle_half_size]
        
        self.time = 0
        self.marks = [self.time]
        key_point = self[self.length*3//4, self.width//2]
        self.key_point_u_norm = [np.sqrt(np.sum(key_point.velocity**2))]
    
    def onestep_flow(self):
#         collision
        self.population += self.omega*(self.equilibrium - self.population)
        
#         copy the relative population before streaming
        left_sketch = self.left_side.population[C_RIGHT].copy()
        C_BLOCK = [8] if self.obstacle_is_odd else [1,8]
        left_top_sketch = self.left_top_edge.population[C_BLOCK].copy()
        C_BLOCK = [5] if self.obstacle_is_odd else [1,5]
        left_bottom_sketch = self.left_bottom_edge.population[C_BLOCK].copy()
        right_sketch = self.right_side.population[C_LEFT].copy()
        C_BLOCK = [7] if self.obstacle_is_odd else [3,7]
        right_top_sketch = self.right_top_edge.population[C_BLOCK].copy()
        C_BLOCK = [6] if self.obstacle_is_odd else [3,6]
        right_bottom_sketch = self.right_bottom_edge.population[C_BLOCK].copy()
        
#         pure streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))
        
#         recover the population after streaming
        self.left_side.population[C_LEFT] = left_sketch
        C_BOUNCE = [6] if self.obstacle_is_odd else [3,6]
        self.left_top_edge.population[C_BOUNCE] = left_top_sketch
        C_BOUNCE = [7] if self.obstacle_is_odd else [3,7]
        self.left_bottom_edge.population[C_BOUNCE] = left_bottom_sketch
        self.right_side.population[C_RIGHT] = right_sketch
        C_BOUNCE = [5] if self.obstacle_is_odd else [1,5]
        self.right_top_edge.population[C_BOUNCE] = right_top_sketch
        C_BOUNCE = [8] if self.obstacle_is_odd else [1,8]
        self.right_bottom_edge.population[C_BOUNCE] = right_bottom_sketch
                
#         left inlet
        MASS_IN = np.ones_like(self[0,:].mass)
        VELOCITY_IN = np.zeros_like(self[0,:].velocity)
        VELOCITY_IN[0] = 0.1
        self[0,:].population[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        
#         right outlet
        self[-1,:].population[C_LEFT] = self[-2,:].population[C_LEFT].copy()
    
    def record_data(self):
        self.time += 1
        self.marks.append(self.time)
        key_point = self[self.length*3//4, self.width//2]
        self.key_point_u_norm.append(np.sqrt(np.sum(key_point.velocity**2)))
        if self.time % 1000 == 0:
            print("t={}".format(self.time))


# +
shape = 200, 90
niu = 0.04
karman_vortex = KarmanVortex(shape, viscosity=niu, blockage_ratio=2/9)

fig, ax = plt.subplots()
ax.set_title("initial velocity magnitude")
u_magnitude = np.linalg.norm(karman_vortex.velocity, axis=0).T
colormap = ax.imshow(u_magnitude)
fig.colorbar(colormap)

num_time = 30000
karman_vortex.flowing(num_time)

fig, ax = plt.subplots()
ax.set_title("afterwards velocity magnitude")
u_magnitude = np.linalg.norm(karman_vortex.velocity, axis=0).T
colormap = ax.imshow(u_magnitude)
fig.colorbar(colormap)
# -

# plot velocity magnitude evolution
fig, ax = plt.subplots()
fig.set_size_inches(6.4*2, 4.8)
ax.plot(karman_vortex.marks, karman_vortex.key_point_u_norm, "b-", label="numerical")
ax.set_title("velocity evolution at [{}, {}]".format(3*LENGTH//4, WIDTH//2))
ax.set_xlabel("time [step]")
ax.set_ylabel("velocity magnitude [unit?]")
ax.legend(loc="best")
fig.savefig("velocity_evolution_serial")

# +
steady_time = np.array(karman_vortex.marks[20000:])
fft_freq = np.fft.fftfreq(steady_time.size)

steady_magnitude = np.array(karman_vortex.key_point_u_norm[20000:])
fft_magnitude = np.abs(np.fft.fft(steady_magnitude - steady_magnitude.mean()))

fig, ax = plt.subplots()
fig.set_size_inches(6.4*2, 4.8)
ax.set_xlim(-1/100, 1/100)
ax.set_title("velocity spectrum")
ax.plot(fft_freq, fft_magnitude, label="numeric")
ax.legend(loc="best")
# -

main_freq = np.abs(fft_freq[np.argmax(fft_magnitude)])
1/main_freq

characteristic_length = karman_vortex.width - karman_vortex.obstacle_size
strouhal_number = main_freq * characteristic_length / steady_magnitude.mean()
strouhal_number

reynolds_number = characteristic_length * steady_magnitude.mean() / niu
reynolds_number


