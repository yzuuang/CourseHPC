# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## import

# %%
from matplotlib import pyplot as plt
import numpy as np

# %% [markdown]
# ## global constants

# %%
# particle velocity of D2Q9 Lattice Bolzmann Method
C_SET = np.array([ 
    (0,0), (1,0), (0,1), (-1,0), (0,-1), 
    (1,1), (-1,1), (-1,-1), (1,-1)])
C_FREEDOM, C_DIMENSION = np.shape(C_SET)

# weight function of particle velocity
C_WEIGHT = np.array([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# group indices together by its direction
C_UP = np.array([2, 5, 6], dtype="intp")
C_DOWN = np.array([4, 7, 8], dtype="intp")
C_LEFT = np.array([3, 6, 7], dtype="intp")
C_RIGHT = np.array([1, 8, 5], dtype="intp")
C_ALL = slice(None)

# %% [markdown]
# ## base class

# %%
class BaseFlow(object):
    """Common features related to population.

    Important properties such as mass density, velocity, equilibrium and population.
    Support of getting and setting(if settable) properties for only a slice of flow.
    A method includes a framework of simulation, with details to be implemented in child class.

    Attributes:
        population: the particle population  about the whole flow or a slice of the flow,
            with the first axis always be the freedom axis.
    """
    
    def __init__(self, population):
        """Save the population data."""
        assert population.shape[0] == C_FREEDOM
        self.population = population
        
    def __getitem__(self, index):
        """Slice the flow.
        Note: to set all indexs of the population of a flow slice, one must use 
        a slicing operator. The reason remains unclear.
        For example:
        some_flow[x_slice, y_slice].population[C_ALL] = some_values  # it works
        some_flow[x_slice, y_slice].population = some_values  # it does not work
        """
        assert len(index) == 2
        return BaseFlow(self.population[:, index[0], index[1]])
    
    @property
    def mass(self):
        """Get mass density from the population of the instance. Read-only property."""
        return np.einsum("f...->...", self.population)
    
    @property
    def velocity(self):
        """Get velocity from the population of the instance. Read-only property."""
        momentum = np.einsum("fd,f...->d...", C_SET, self.population)
        divisible_mass = self.mass.copy()
        # handle divide by 0 through divide by infinity
        if isinstance(divisible_mass, np.ndarray):
            divisible_mass[divisible_mass == 0] = np.inf
        else:  # one element rather than a slice
            assert np.size(divisible_mass) == 1
            divisible_mass = np.array(divisible_mass or np.inf)
        return momentum / divisible_mass[np.newaxis,...]
    
    @property
    def equilibrium(self):
        """Get equilibrium from the mass density and velocity of the instance. Read-only property."""
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, self.velocity)
        u_square = np.einsum("d...,d...->...", self.velocity, self.velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, self.mass, in_brackets)
        
    @staticmethod
    def compute_equilibrium(mass, velocity):
        """The same as equilibrium(self) except using the given mass density and velocity data."""
        assert mass.shape == velocity.shape[1:]
        assert velocity.shape[0] == C_DIMENSION
        c_scalar_u = np.einsum("fd,d...->f...", C_SET, velocity)
        u_square = np.einsum("d...,d...->...", velocity, velocity)
        in_brackets = 1 + 3*c_scalar_u + 9/2*(c_scalar_u**2) - 3/2*u_square[np.newaxis,...]
        return np.einsum("f,...,f...->f...", C_WEIGHT, mass, in_brackets)
    
    def simulate(self, num_step):
        """Simulate the flow for a given number of timesteps."""
        assert isinstance(num_step, int) and num_step >= 0
        for _ in range(num_step):
            self._flow_dynamics()
            assert np.all(self.mass >= 0), "problems at {}".format(np.where(self.mass <= 0))
            self._record_data()
    
    def _flow_dynamics(self):
        """To be implemented in child class."""
        pass
    
    def _record_data(self):
        """To be implemented in child class."""
        pass


# %% [markdown]
## Milestone 1&2

# # %%
# class MidddleHeavier(BaseFlow):

#     def __init__(self, shape, *, omega=1.0):
#         assert len(shape) == C_DIMENSION
#         mass = np.full(shape, 0.5)
#         mass[shape[0]//2, shape[1]//2] *= 1.2
#         velocity = np.full(shape, (0, 0), dtype=(float, 2))
#         velocity = np.moveaxis(velocity, -1, 0)
#         self.population = self.compute_equilibrium(mass, velocity)
#         assert omega > 0 and omega < 2
#         self.omega = omega
#         self.time = 0
#         self.marks=[self.time]
    
#     def _flow_dynamics(self):
#         self.population -= self.omega * (self.population - self.equilibrium)
#         for index in range(1, C_FREEDOM):
#             self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))

#     def _record_data(self):
#         self.time += 1
#         self.marks.append(self.time)


# # %%
# shape = 70, 30
# middle_heavier = MidddleHeavier(shape)

# fig, ax = plt.subplots()
# ax.set_title("initial mass density")
# colormap = ax.imshow(middle_heavier.mass.T, origin="lower")
# fig.colorbar(colormap)

# num_step = 500
# middle_heavier.simulate(num_step)

# fig, ax = plt.subplots()
# ax.set_title("afterwards mass density")
# colormap = ax.imshow(middle_heavier.mass.T, origin="lower")
# fig.colorbar(colormap)

# %% [markdown]
# ## Milestone 3 case 1

# %%
class MassShearWave(BaseFlow):
    """Shear wave decay with sinous perturbation in mass density in x direction.

    Overload __init__ method.
    Implement _flow_dynamics and _record_data method.

    Attributes:
        population: the particle population  about the whole flow or a slice
            of the flow, with the first axis always be the freedom axis.
        omega: collision coefficient.
        time: private time axis. Automatically add 1 (dt*=1) after each step.
        marks: time records.
        rho_profiles: mass density profiles along x direction. 
        ux_profiles: x component of velocity profiles along x direction.
        uy_profiils: y component of velocity profiles along x direction.
    """

    def __init__(self, shape, *, omega=1.0): 
        """Initilize population by computing equilibrium 
        from mass density and velocity of special flow kind.
        Prepare varibles to be used in _flow_dynamics and _record_data."""
        # Initialize population
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        sin_2pi = lambda t: np.sin(2*np.pi*t)
        x = np.arange(self.length)
        mass_pattern = 0.5 + 5e-6*sin_2pi(x/self.length)
        mass = np.repeat(mass_pattern, self.width).reshape(shape)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)  # move the axis of dimension to the 0 axis
        self.population = self.compute_equilibrium(mass, velocity)
        
        # Varibles for flow dynamics
        assert omega > 0 and omega < 2
        self.omega = omega
        
        # Varibles for recording data
        self.time = 0
        self.marks = [self.time]
        self.rho_profiles = [self[:,0].mass]
        self.ux_profiles = [self[:,0].velocity[0]]
        self.uy_profiles = [self[:,0].velocity[1]]
    
    def _flow_dynamics(self):
        """Simple flow dynamics with periodic boundary."""
        # Collision
        self.population -= self.omega * (self.population - self.equilibrium)
        
        # Streaming
        for index in range(1, C_FREEDOM):
            self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))
    
    def _record_data(self):
        """Save interested data for future processing."""
        self.time += 1
        self.marks.append(self.time)
        self.rho_profiles.append(self[:,0].mass)
        self.ux_profiles.append(self[:,0].velocity[0])
        self.uy_profiles.append(self[:,0].velocity[1])


# %%
shape = 70, 30
mass_shear_wave = MassShearWave(shape)

fig, ax = plt.subplots()
ax.set_title("initial rho")
colormap = ax.imshow(mass_shear_wave.mass.T, origin="lower")
fig.colorbar(colormap)

num_step = 2000
mass_shear_wave.simulate(num_step)

fig, ax = plt.subplots()
ax.set_title("afterwards rho")
colormap = ax.imshow(mass_shear_wave.mass.T, origin="lower")
fig.colorbar(colormap)


# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(6.4, 4.8*3)

# Plot profile evolution.
for mark, rho_profile, ux_profile, uy_profile in zip(
        mass_shear_wave.marks, mass_shear_wave.rho_profiles,
        mass_shear_wave.ux_profiles, mass_shear_wave.uy_profiles):
    if mark % (num_step // 5) == 0:
        x_axis = range(mass_shear_wave.length)
        ax1.plot(x_axis, rho_profile, label="t={}".format(mark))
        ax2.plot(x_axis, ux_profile, label="t={}".format(mark))
        ax3.plot(x_axis, uy_profile, label="t={}".format(mark))

# Configuration.
ax1.plot(x_axis, np.ones_like(x_axis)*0.5, "--", label="0.5")
ax1.legend(loc="best")
ax1.set_title("rho profile evolution")
ax1.set_xlabel("x")
ax1.set_ylabel("rho")

ax2.legend(loc="best")
ax2.set_title("ux profile evolution")
ax2.set_xlabel("x")
ax2.set_ylabel("u_x")

ax3.legend(loc="best")
ax3.set_title("uy profile evolution")
ax3.set_xlabel("x")
ax3.set_ylabel("u_y")
ax3.set_ylim(-1e-6, 1e-6)

fig.savefig("ms3c1_profiles")

# %%
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(6.4, 4.8*2)

# Plot numerical oscillation decay.
t = np.array(mass_shear_wave.marks)
rho_amplitude = np.maximum.reduce(mass_shear_wave.rho_profiles, axis=1) - 0.5
ax1.plot(t, rho_amplitude, "b+", label="numeric")

ux_amplitude = np.maximum.reduce(mass_shear_wave.ux_profiles, axis=1)
ax2.plot(t, ux_amplitude, "b+", label="numeric")

# Plot analytic oscillation decay.
niu = 1/3 * (1/mass_shear_wave.omega - 1/2)
zeta = 2*np.pi / mass_shear_wave.length
rho_amplitude = rho_amplitude[0] * np.exp(-niu * zeta**2 * t)
ax1.plot(t, rho_amplitude, "r-", label="analytic")
ax1.plot(t, np.zeros_like(t), "c-.", label="0")

ux_max = np.max(ux_amplitude)
index_max = np.argmax(ux_amplitude)
ux_amplitude = ux_max * np.exp(-niu * zeta**2 * (t - t[index_max]))
ax2.plot(t, ux_amplitude, "r-", label="analytic")
ax2.plot(t, np.zeros_like(t), "c-.", label="0")

# Configuration
ax1.legend(loc="best")
ax1.set_title("rho oscillation decay")
ax1.set_xlabel("time")
ax1.set_ylabel("rho amplitude")

ax2.legend(loc="best")
ax2.set_title("u_x oscillation decay")
ax2.set_xlabel("time")
ax2.set_ylabel("u_x amplitude")

fig.savefig("ms3c1_decay")

# %%
fig, ax = plt.subplots()

# Plot numeric viscosity.
omega_axis = np.linspace(0.1, 1.9, num=19).round(decimals=1)
viscosity_axis = []
for omega in omega_axis:
    print("omega={}".format(omega))
    shape = 200, 50
    mass_shear_wave = MassShearWave(shape, omega=omega)
    num_step = 2000
    mass_shear_wave.simulate(num_step)
    amplitude = np.maximum.reduce(mass_shear_wave.ux_profiles, axis=1)
    reference_step = np.argmax(amplitude)
    amplitude = amplitude[reference_step:]
    zeta = 2*np.pi / mass_shear_wave.length
    viscosity = np.log(amplitude[0]/amplitude[-1]) / zeta**2 / (num_step - reference_step)
    viscosity_axis.append(viscosity)
ax.plot(omega_axis, viscosity_axis, "b.", label="numeric")

# Plot anaylitic viscosity.
omg = np.linspace(0.05, 1.95)
niu = 1/3 * (1/omg - 1/2)
ax.plot(omg, niu, "r-", label="analytic")

# Configurations
ax.legend(loc="best")
ax.set_title("viscosity - omega relation")
ax.set_xlabel("omega")
ax.set_ylabel("viscosity")

fig.savefig("ms3c1_viscosity")

# %% [markdown]
# ## Milestone 3 case 2

# %%
class VelocityShearWave(BaseFlow):
    """Shear wave decay with sinous perturbation in x component of velocity in y direction.

    Overload __init__ method.
    Implement _flow_dynamics and _record_data method.

    Attributes:
        population: the particle population  about the whole flow or a slice
            of the flow, with the first axis always be the freedom axis.
        omega: collision coefficient.
        time: private time axis. Automatically add 1 (dt*=1) after each step.
        marks: time records.
        rho_profiles: mass density profiles along y direction. 
        ux_profiles: x component of velocity profiles along y direction.
        uy_profiils: y component of velocity profiles along y direction.
    """
        
    def __init__(self, shape, *, omega=1.0): 
        """Initilize population by computing equilibrium 
        from mass density and velocity of special flow kind.
        Prepare varibles to be used in _flow_dynamics and _record_data."""
        # Initialize population
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        mass = np.full(shape, 0.5)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)  # move the axis of dimension to the 0 axis
        sin_2pi = lambda t: np.sin(2*np.pi*t)
        y = np.arange(self.width)
        velicty_pattern = 1e-5*sin_2pi(y/self.width)
        velocity[0] = np.broadcast_to(velicty_pattern, shape)  # only x component
        self.population = self.compute_equilibrium(mass, velocity)
        
        # Varibles for flow dynamics
        assert omega > 0 and omega < 2
        self.omega = omega
        
        # Varibles for recording data
        self.time = 0
        self.marks = [self.time]
        self.rho_profiles = [self[0,:].mass]
        self.ux_profiles = [self[0,:].velocity[0]]
        self.uy_profiles = [self[0,:].velocity[1]]
    
    def _flow_dynamics(self):
        """Simple flow dynamics with periodic boundary."""
        # Collision
        self.population -= self.omega * (self.population - self.equilibrium)

        # Streaming
        for index in range(1, C_FREEDOM):
            self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))
    
    def _record_data(self):
        """Save interested data for future processing."""
        self.time += 1
        self.marks.append(self.time)
        self.rho_profiles.append(self[0,:].mass)
        self.ux_profiles.append(self[0,:].velocity[0])
        self.uy_profiles.append(self[0,:].velocity[1])


# %%
shape = 70, 30
velocity_shear_wave = VelocityShearWave(shape)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(velocity_shear_wave.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_step = 500
velocity_shear_wave.simulate(num_step)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(velocity_shear_wave.velocity[0].T, origin="lower")
fig.colorbar(colormap)


# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(6.4, 4.8*3)

# Plot profile evolution.
for mark, rho_profile, ux_profile, uy_profile in zip(
        velocity_shear_wave.marks, velocity_shear_wave.rho_profiles,
        velocity_shear_wave.ux_profiles, velocity_shear_wave.uy_profiles):
    if mark % (num_step // 5) == 0:
        y_axis = range(velocity_shear_wave.width)
        ax1.plot(rho_profile, y_axis, label="t={}".format(mark))
        ax2.plot(ux_profile, y_axis, label="t={}".format(mark))
        ax3.plot(uy_profile, y_axis, label="t={}".format(mark))

# Configurations.
ax1.legend(loc="best")
ax1.set_xlim(0.5-1e-6, 0.5+1e-6)
ax1.set_title("rho profile evolution")
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2.plot(np.zeros_like(y_axis), y_axis, "--", label="0")
ax2.legend(loc="best")
ax2.set_title("ux profile evolution")

ax3.legend(loc="best")
ax3.set_xlim(-1e-6, 1e-6)
ax3.set_title("uy profile evolution")

fig.savefig("ms3c2_profiles")

# %%
fig, ax = plt.subplots()

# Plot numerical oscillation decay.
t = np.array(velocity_shear_wave.marks)
ux_amplitude = np.maximum.reduce(velocity_shear_wave.ux_profiles, axis=1)
ax.plot(t, ux_amplitude, "y+", label="numeric2")

# Plot analytic oscillation decay.
niu = 1/3 * (1/velocity_shear_wave.omega - 1/2)
zeta = 2*np.pi / velocity_shear_wave.width
ux_max = np.max(ux_amplitude)
index_max = np.argmax(ux_amplitude)
ux_amplitude = ux_max * np.exp(-niu * zeta**2 * (t - t[index_max]))
ax.plot(t, ux_amplitude, "r-", label="analytic")

# Configurations.
ax.legend(loc="best")
ax.set_title("ux magnitude evolution")

fig.savefig("ms3c2_decay")

# %%
fig, ax = plt.subplots()

# Plot numeric viscosity.
omega_axis = np.linspace(0.1, 1.9, num=19).round(decimals=1)
viscosity_axis = []
for omega in omega_axis:
    print("omega={}".format(omega))
    shape = 50, 200
    velocity_shear_wave = VelocityShearWave(shape, omega=omega)
    num_step = 500
    velocity_shear_wave.simulate(num_step)
    amplitude = np.maximum.reduce(velocity_shear_wave.ux_profiles, axis=1)
    zeta = 2*np.pi / velocity_shear_wave.width
    viscosity = np.log(amplitude[0]/amplitude[-1]) / zeta**2 / num_step
    viscosity_axis.append(viscosity)
ax.plot(omega_axis, viscosity_axis, "b.", label="numeric")

# Plot anaylitic viscosity.
omg = np.linspace(0.1, 1.9)
niu = 1/3 * (1/omg - 1/2)
ax.plot(omg, niu, "r-", label="analytic")

# Configurations.
ax.legend(loc="best")
ax.set_title("viscosity - omega relation")
ax.set_xlabel("omega [none]")
ax.set_ylabel("viscosity")

fig.savefig("ms3c2_viscosity")

# %% [markdown]
# ## Milestone 4

# %%
class CouetteFlow(BaseFlow):
    """2D Couette flow
    
    Overload __init__ method.
    Implement _flow_dynamics and _record_data method.
    
    Attributes:
        population: the particle population  about the whole flow or a slice
            of the flow, with the first axis always be the freedom axis.
        omega: collision coefficient.
        time: private time axis. Automatically add 1 (dt*=1) after each step.
        marks: time records.
        ux_profiles: x component of velocity profiles along y-direction centerline.
        uy_profiils: y component of velocity profiles along y-direction centerline.
    """
        
    def __init__(self, shape, *, omega=1.0, wall_velocity):
        """Initilize population by computing equilibrium from mass density and velocity 
        of special flow kind. Prepare varibles to be used in _flow_dynamics and _record_data.
        """
        # Initialize population
        assert len(shape) == C_DIMENSION
        self.length, self.width = shape
        mass = np.full(shape, 1)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)  # move the axis of dimension to the 0 axis
        self.population = self.compute_equilibrium(mass, velocity)
        
        # Varibles for flow dynamics
        assert omega > 0 and omega < 2
        self.omega = omega
        self.wall_velocity = wall_velocity
        
        # Varibles for record data
        self.time = 0
        self.marks = [self.time]
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles = [another_centerline.mass]
        self.ux_profiles = [centerline.velocity[0]]
        self.uy_profiles = [centerline.velocity[1]]
    
    def _flow_dynamics(self):
        """Based on simple flow dynamics, adding walls by copy relative indexs before streaming
        and recover bounce-back indexs after streaming"""
        # Collision
        self.population -= self.omega * (self.population - self.equilibrium)
        
        # After collision
        top_sketch = self[:,-1].population[C_UP].copy()
        bottom_sketch = self[:,0].population[C_DOWN].copy()
        
        # Streaming
        for index in range(1, C_FREEDOM):
            self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))
        
        # After streaming
        loss = 6 * np.einsum("f,x,fd,d->fx", C_WEIGHT[C_UP], self[:,-1].mass, C_SET[C_UP], self.wall_velocity)
        self[:,-1].population[C_DOWN] = top_sketch - loss
        self[:,0].population[C_UP] = bottom_sketch
    
    def _record_data(self):
        """Save interested data for future processing."""
        self.time += 1
        self.marks.append(self.time)
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles.append(another_centerline.mass)
        self.ux_profiles.append(centerline.velocity[0])
        self.uy_profiles.append(centerline.velocity[1])

# %%
shape = 70, 30
u_w = (0.1, 0.)
couette_flow = CouetteFlow(shape, wall_velocity=u_w)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(couette_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_step = 3000
couette_flow.simulate(num_step)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(couette_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)

# %%
fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(6.4, 4.8*2)

# Plot profile evolution.
for mark, ux_profile, uy_profile in zip(
        couette_flow.marks, 
        couette_flow.ux_profiles, couette_flow.uy_profiles):
    if mark in [0, 20, 150, 500, 1000, 3000]:
        y_axis = np.arange(couette_flow.width)
        ax1.plot(ux_profile, y_axis, "-", label="{} step".format(mark))
        ax2.plot(uy_profile, y_axis, "-", label="{} step".format(mark))

# Plot analytic steady profile and walls.
y = np.linspace(-0.5, couette_flow.width-0.5)
ux_analytic = u_w[0] / couette_flow.width * (y+0.5)
ax1.plot(ux_analytic, y, "k--", label="analytic")

x = np.linspace(0, u_w[0])
ax1.plot(x, np.ones_like(x)*(-0.5), "-.", label="top wall")
ax1.plot(x, np.ones_like(x)*(couette_flow.width-0.5), "-.", label="bottom wall")

# Configurations.
ax1.legend(loc="lower right")
ax1.set_title("u_x profile at centerline")
ax1.set_xlabel("u_x")
ax1.set_ylabel("y")

ax2.legend(loc="best")
ax2.set_xlim(-1e-6, 1e-6)
ax2.set_title("u_y profile at centerline")
ax2.set_xlabel("u_y")
ax2.set_ylabel("y")

fig.savefig("ms4_profiles")

# %% [markdown]
# ## Milestone 5

# %%
class PoiseuilleFlow(BaseFlow):
    """2D Poiseuille Flow
    
    Overload __init__ method.
    Implement _flow_dynamics and _record_data method.
    
    Attributes:
        population: the particle population  about the whole flow or a slice
            of the flow, with the first axis always be the freedom axis.
        omega: collision coefficient.
        time: private time axis. Automatically add 1 (dt*=1) after each step.
        marks: time records.
        rho_profiles: mass density profiles along x-direction centerline.
        ux_profiles: x component of velocity profiles along y-direction centerline.
        uy_profiils: y component of velocity profiles along y-direction centerline.
    """

    def __init__(self, shape, *, omega=1.0, pressure_gradient):
        """Initilize population by computing equilibrium 
        from mass density and velocity of special flow kind.
        Prepare varibles to be used in _flow_dynamics and _record_data."""
        # Initialize population
        assert len(shape) == C_DIMENSION
        mass = np.full(shape, 1)
        velocity = np.full(shape, (0, 0), dtype=(float, 2))
        velocity = np.moveaxis(velocity, -1, 0)  # move the axis of dimension to the 0 axis
        self.population = self.compute_equilibrium(mass, velocity)
        
        # Add ghost cells
        self.population = np.insert(self.population, [0,shape[0]], 0, axis=1)
        self.length, self.width = self.mass.shape
        
        # Varibles for flow dynamics
        assert omega > 0 and omega < 2
        self.omega = omega
        assert pressure_gradient >= 0
        self.delta_mass = pressure_gradient * self.length/2 * 3

        # Varibles for recording data
        self.time = 0
        self.marks = [self.time]
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles = [another_centerline.mass]
        self.ux_profiles = [centerline.velocity[0]]
        self.uy_profiles = [centerline.velocity[1]]
        
    def _flow_dynamics(self):
        """Based on flow dynamics with walls, adding pressure to periodic boundary
        by 'Pre-streaming' exactly between 'Streaming' and 'Before streaming'."""
        # Collision
        self.population -= self.omega * (self.population - self.equilibrium)

        # After collision
        top_sketch = self[:,-1].population[C_UP].copy()
        bottom_sketch = self[:,0].population[C_DOWN].copy()

        # Pre-streaming
        MASS_IN = np.full(np.shape(self[0,:].mass), 1+self.delta_mass)
        self[0,:].population[C_RIGHT] = self.compute_equilibrium(MASS_IN, self[-2,:].velocity)[C_RIGHT]
        self[0,:].population[C_RIGHT] += self[-2,:].population[C_RIGHT] - self[-2,:].equilibrium[C_RIGHT]
        
        MASS_OUT = np.full(np.shape(self[-1,:].mass), 1-self.delta_mass)
        self[-1,:].population[C_LEFT] = self.compute_equilibrium(MASS_OUT, self[1,:].velocity)[C_LEFT]
        self[-1,:].population[C_LEFT] += self[1,:].population[C_LEFT] - self[1,:].equilibrium[C_LEFT]

        # Normal streaming
        for index in range(1, C_FREEDOM):
            self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))
        
        # After streaming
        self[:,-1].population[C_DOWN] = top_sketch
        self[:,0].population[C_UP] = bottom_sketch
    
    def _record_data(self):
        """Save interested data for future processing."""
        self.time += 1
        self.marks.append(self.time)
        centerline = self[self.length//2,:]
        another_centerline = self[:,self.width//2]
        self.rho_profiles.append(another_centerline.mass)
        self.ux_profiles.append(centerline.velocity[0])
        self.uy_profiles.append(centerline.velocity[1])


# %%
shape = 70, 30
dp_dx = 1e-4
poiseuille_flow = PoiseuilleFlow(shape, pressure_gradient=dp_dx)

fig, ax = plt.subplots()
ax.set_title("initial u_x")
colormap = ax.imshow(poiseuille_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)

num_step = 3000
poiseuille_flow.simulate(num_step)

fig, ax = plt.subplots()
ax.set_title("afterwards u_x")
colormap = ax.imshow(poiseuille_flow.velocity[0].T, origin="lower")
fig.colorbar(colormap)


# %%
fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(6.4, 4.8*2)

# Plot profile evolution.
for mark, ux_profile, uy_profile in zip(
    poiseuille_flow.marks, 
    poiseuille_flow.ux_profiles, poiseuille_flow.uy_profiles):
    if mark in [0, 100, 300, 500, 1000, 2000, 3000]:
        y_axis = range(poiseuille_flow.width)
        ax1.plot(ux_profile, y_axis, label="{} step".format(mark))
        ax2.plot(uy_profile, y_axis, label="{} step".format(mark))

# Plot analytic steady profile and walls.
y = np.linspace(-0.5, poiseuille_flow.width-0.5)
niu = 1/3 * (1/poiseuille_flow.omega - 1/2)
miu = poiseuille_flow.rho_profiles[-1].mean() * niu
ux_analytic = 1/(2*miu) * dp_dx * (y + 0.5) * (poiseuille_flow.width-0.5 - y)
ax1.plot(ux_analytic, y, "k--", label="analytic")

x = np.linspace(0, ux_analytic.max())
ax1.plot(x, np.ones_like(x)*(-0.5), "-.", label="top wall")
ax1.plot(x, np.ones_like(x)*(poiseuille_flow.width-0.5), "-.", label="bottom wall")

# Configurations.
ax1.legend(loc="center left")
ax1.set_title("u_x profile at centerline")
ax1.set_xlabel("u_x")
ax1.set_ylabel("y")

ax2.legend(loc="best")
ax2.set_title("u_y profile at centerline")
ax2.set_xlabel("u_y")
ax2.set_ylabel("y ")

fig.savefig("ms5_profiles")

# %%
fig, ax = plt.subplots()

# Plot numeric mass profile
x_axis = range(poiseuille_flow.length)
for mark, rho_profile in zip(
        poiseuille_flow.marks, poiseuille_flow.rho_profiles):
    if mark in [100, 800, 3000]:
        ax.plot(x_axis, rho_profile, label="t={}".format(mark))

# Plot analytic mass profile
x = np.linspace(-1, poiseuille_flow.length)
rho_analytic = -(3*dp_dx) * (x - poiseuille_flow.length//2) + 1
ax.plot(x, rho_analytic,"k--", label="pressure drop")

# Configurations.
ax.legend(loc='best')
ax.set_title("rho profile")
ax.set_xlabel("x")
ax.set_ylabel("rho")

fig.savefig("ms5_mass_density")

# %% [markdown]
## Milestone 6

# # %%
# class KarmanVortex(BaseFlow):
        
#     def __init__(self, shape, *, viscosity, obstacle_size):
#         assert len(shape) == C_DIMENSION
#         self.length, self.width = shape
#         mass = np.full(shape, 1)
#         velocity = np.full(shape, (0.1, 0), dtype=(float, 2))
#         velocity = np.moveaxis(velocity, -1, 0)  # move the axis of dimension to the 0 axis
#         velocity[:,:,self.width//2:] += 10**(-6)  # perturbation in upper half
#         self.population = self.compute_equilibrium(mass, velocity)
        
#         assert viscosity > 0
#         self.omega = 2 / (6*viscosity + 1)
        
#         length, width = np.shape(self.mass)
#         assert length % 4 == 0 and width % 2 == 0
#         assert obstacle_size > 0 and obstacle_size < self.width
#         self.obstacle_size = obstacle_size
#         self.obstacle_is_odd = (self.obstacle_size % 2 == 1)
#         obstacle_half_size = np.ceil(self.obstacle_size/2).astype("int")
#         # Decompose obstacle interaction area into six parts
#         #     part_left_top     2|5  part_right_top
#         #                       1|4
#         #     part_left_side    1|4  part_right_side
#         #                       1|4
#         #     part_left_bottom  3|6  part_right_bottom
#         self.part_left_side = self[length//4-1, width//2-obstacle_half_size+1 : width//2+obstacle_half_size- 1]
#         self.part_left_top = self[length//4-1, width//2+obstacle_half_size-1]
#         self.part_left_bottom = self[length//4-1, width//2-obstacle_half_size]
#         self.part_right_side = self[length//4, width//2-obstacle_half_size+1 : width//2+obstacle_half_size-1]
#         self.part_right_top = self[length//4, width//2+obstacle_half_size-1]
#         self.part_right_bottom = self[length//4, width//2-obstacle_half_size]
        
#         self.time = 0
#         self.marks = [self.time]
#         key_point = self[self.length*3//4, self.width//2]
#         norm = lambda v: np.sqrt(np.sum(v**2))
#         self.evolution = [norm(key_point.velocity)]
    
#     def _flow_dynamics(self):
#         self.population += self.omega*(self.equilibrium - self.population)
        
#         left_sketch = self.part_left_side.population[C_RIGHT].copy()
#         C_BLOCK = [8] if self.obstacle_is_odd else [1,8]
#         left_top_sketch = self.part_left_top.population[C_BLOCK].copy()
#         C_BLOCK = [5] if self.obstacle_is_odd else [1,5]
#         left_bottom_sketch = self.part_left_bottom.population[C_BLOCK].copy()
#         right_sketch = self.part_right_side.population[C_LEFT].copy()
#         C_BLOCK = [7] if self.obstacle_is_odd else [3,7]
#         right_top_sketch = self.part_right_top.population[C_BLOCK].copy()
#         C_BLOCK = [6] if self.obstacle_is_odd else [3,6]
#         right_bottom_sketch = self.part_right_bottom.population[C_BLOCK].copy()
        
#         for index in range(1, C_FREEDOM):
#             self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))
        
#         self.part_left_side.population[C_LEFT] = left_sketch
#         C_BOUNCE = [6] if self.obstacle_is_odd else [3,6]
#         self.part_left_top.population[C_BOUNCE] = left_top_sketch
#         C_BOUNCE = [7] if self.obstacle_is_odd else [3,7]
#         self.part_left_bottom.population[C_BOUNCE] = left_bottom_sketch
#         self.part_right_side.population[C_RIGHT] = right_sketch
#         C_BOUNCE = [5] if self.obstacle_is_odd else [1,5]
#         self.part_right_top.population[C_BOUNCE] = right_top_sketch
#         C_BOUNCE = [8] if self.obstacle_is_odd else [1,8]
#         self.part_right_bottom.population[C_BOUNCE] = right_bottom_sketch
        
#         MASS_IN = np.ones_like(self[0,:].mass)
#         VELOCITY_IN = np.zeros_like(self[0,:].velocity)
#         VELOCITY_IN[0] = 0.1
#         self[0,:].population[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        
#         self[-1,:].population[C_LEFT] = self[-2,:].population[C_LEFT].copy()
    
#     def _record_data(self):
#         self.time += 1
#         self.marks.append(self.time)
#         key_point = self[self.length*3//4, self.width//2]
#         norm = lambda v: np.sqrt(np.sum(v**2))
#         self.evolution.append(norm(key_point.velocity))
#         if self.time % 1000 == 0:
#             print("t={}".format(self.time))


# # %%
# shape = 200, 90
# niu = 0.04
# karman_vortex = KarmanVortex(shape, viscosity=niu, obstacle_size=20)

# fig, ax = plt.subplots()
# ax.set_title("initial velocity magnitude")
# u_magnitude = np.linalg.norm(karman_vortex.velocity, axis=0).T
# colormap = ax.imshow(u_magnitude, origin="lower")
# fig.colorbar(colormap)

# num_step = 30000
# karman_vortex.simulate(num_step)

# fig, ax = plt.subplots()
# ax.set_title("afterwards velocity magnitude")
# u_magnitude = np.linalg.norm(karman_vortex.velocity, axis=0).T
# colormap = ax.imshow(u_magnitude, origin="lower")
# fig.colorbar(colormap)


# # %%
# # plot velocity magnitude evolution
# fig, ax = plt.subplots()
# fig.set_size_inches(6.4*2, 4.8)
# ax.plot(karman_vortex.marks, karman_vortex.evolution, "b-", label="numerical")
# ax.set_title("velocity evolution at [{}, {}]".format(3*shape[0]//4, shape[1]//2))
# ax.set_xlabel("time [step]")
# ax.set_ylabel("velocity magnitude")
# ax.legend(loc="best")


# # %%
# steady_time = np.array(karman_vortex.marks[20000:])
# fft_freq = np.fft.fftfreq(steady_time.size)

# steady_magnitude = np.array(karman_vortex.evolution[20000:])
# fft_magnitude = np.abs(np.fft.fft(steady_magnitude - steady_magnitude.mean()))

# fig, ax = plt.subplots()
# fig.set_size_inches(6.4*2, 4.8)
# ax.set_xlim(-1/100, 1/100)
# ax.set_title("velocity spectrum")
# ax.plot(fft_freq, fft_magnitude, label="numeric")
# ax.legend(loc="best")


# # %%
# main_freq = np.abs(fft_freq[np.argmax(fft_magnitude)])
# print("The period is {}".format(1/main_freq))

# reynolds_number = karman_vortex.obstacle_size * steady_magnitude.mean() / niu
# print("Reynolds = {}".format(reynolds_number))

# strouhal_number = main_freq * karman_vortex.obstacle_size / steady_magnitude.mean()
# print("Strouhal = {}".format(strouhal_number))

