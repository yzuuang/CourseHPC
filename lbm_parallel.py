# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import timeit
import sys
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np


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


# %%
# Basic core information
comm = MPI.COMM_WORLD
core_world_size = comm.Get_size()
# Get a proper divisor so that x size and y size are close values.
for divisor in range(np.sqrt(core_world_size).astype("int"), 0, -1):
    if core_world_size % divisor == 0:
        break
core_x_size = core_world_size // divisor
core_y_size = divisor
del divisor

core_world_rank = comm.Get_rank()
# Arrange cores in x direction first
# an example for 8 cores:
#     |4|5|6|7|
#     |0|1|2|3|
core_x_rank = core_world_rank % core_x_size
core_y_rank = core_world_rank // core_x_size

if core_world_rank == 0:
    print("======= {} cores =======".format(core_world_size))
    print("Arrange: {} x {}".format(core_x_size, core_y_size))


# %%
class BaseFlow(object):
    """Common features related to population.

    Important properties such as mass, velocity, equilibrium and population.
    Support of getting and setting(if settable) properties for only a slice of flow.
    Public method for simulation, with a framework of recording data after flow dynamics.

    Attributes:
        population: the probability density about the whole flow or a slice of the flow,
            with the first axis always be the freedom axis.
    """
    
    def __init__(self, population):
        """Save the population data."""
        assert population.shape[0] == C_FREEDOM
        self.population = population
        
    def __getitem__(self, index):
        """Slice the flow.
        Note: to set all channels of the population of a flow slice, one must use 
        a slicing operator. The reason remains unclear.
        e.g., some_flow[x_slice, y_slice].population[C_ALL] = some_values
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
        """The same as equilibrium except using the given mass density and velocity data."""
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
            assert np.all(self.mass > 0), "problems at {}".format(np.where(self.mass <= 0))
            self._record_data()
    
    def _flow_dynamics(self):
        """To be implemented in child class."""
        pass
    
    def _record_data(self):
        """To be implemented in child class."""
        pass


# %%
class KarmanVortex(BaseFlow):
    """2D Karman vortex street
    
    Overload __init__ method.
    Implement _flow_dynamics and _record_data method.
    
    Attributes:
        population: the rticle population about the whole flow or a slice of the flow,
            with the first axis always be the freedom axis.
        omega: collision coefficient.
        obstacle_size: an integer value representing the length of obstacle.
        obstacle_is_odd: a bool value telling whether the obstacle length is an odd number.
        part_left_middle: If the local region contains the respective region of obstacle,
            it is the flow slice of that region. Otherwise, it is None.
        part_left_top: similar to part_left_size.
        part_left_bottom: similar to part_left_size.
        part_right_middle: similar to part_left_size.
        part_right_top: similar to part_left_size.
        part_right_bottom: similar to part_left_size.
        time: private time axis. Automatically add 1 (dt*=1) after each step.
        marks: time records.
        key_point: If the local region contains the key point to be tracked, it is the
            flow slice of that point. Otherwise, it is None.
        evolution: The data tracking the norm of velocity at the key point. Will
            only be created when key_point is not None.
    """

    def __init__(self, global_shape, *, initial_velocity, viscosity, obstacle_size):
        """Initilize population by computing equilibrium from the mass density and 
        velocity of local flow. Prepare varibles to be used in _flow_dynamics and 
        _record_data.
        """
        # Initialize the population
        ## Get the global shape
        global_length, global_width = global_shape
        
        ## Analyze the local shape
        local_length = np.ceil(global_length / core_x_size).astype("int")
        while(global_length <= (core_x_size - 1)*local_length):
            local_length -= 1
            if core_world_rank == 0:
                print("{} cores are not cooperative in x direction.".format(core_world_size))
        if core_x_rank == core_x_size - 1:
            local_length = global_length - (core_x_size - 1)*local_length
        
        local_width = np.ceil(global_width / core_y_size).astype("int")
        while(global_width <= (core_y_size - 1)*local_width):
            local_width -= 1
            if core_world_rank == 0:
                print("{} cores are not cooperative in y direction.".format(core_world_size))
        if core_y_rank == core_y_size - 1:
            local_width = global_width - (core_y_size - 1)*local_width

        ## Corresponding global indices
        if core_x_rank == core_x_size - 1:
            global_x_range = range(global_length-local_length, global_length)
        else:
            global_x_range = range(core_x_rank*local_length, (core_x_rank + 1)*local_length)
        
        if core_y_rank == core_y_size - 1:
            global_y_range = range(global_width-local_width, global_width)
        else:
            global_y_range = range(core_y_rank*local_width, (core_y_rank + 1)*local_width)

        ## Initialize with respective local shape
        local_mass = np.full((local_length, local_width), 1)
        assert np.abs(initial_velocity) <= 1/3
        local_velocity = np.full((local_length, local_width), (initial_velocity, 0), dtype=(float, 2))
        local_velocity = np.moveaxis(local_velocity, -1, 0)  # move the axis of dimension to the 0 axis
        ## perturbation in upper half
        if global_y_range[0] > global_width//2:
            local_velocity += 10e-6
        elif global_width//2 in global_y_range:
            y_half =  global_y_range.index(global_width//2)
            local_velocity[:,:,y_half:] += 10e-6
        self.population = self.compute_equilibrium(local_mass, local_velocity)
        
        ## Add ghost cells
        self.population = np.insert(self.population, [0,local_length], 0, axis=1)
        self.population = np.insert(self.population, [0,local_width], 0, axis=2)
        ## From now on, local_index = global_range.index(some_value) + 1
        
        # Varibles for flow dynamics
        ## Coefficient omega
        self.omega = 2 / (6*viscosity + 1)        
        
        ## Obstacle position in global indices
        assert isinstance(obstacle_size, int) 
        assert obstacle_size > 0 and obstacle_size < global_width
        obstacle_x_left = global_length//4 - 1
        obstacle_x_right = global_length//4
        if global_width % 2 == 0:
            obstacle_half_size = np.ceil(obstacle_size/2).astype("int")
            obstacle_y_top = global_width//2 + obstacle_half_size - 1
            obstacle_y_bottom = global_width//2 - obstacle_half_size
            self.is_solely_blocked = (obstacle_size % 2 == 1)
        else:
            obstacle_half_size = np.floor(obstacle_size/2).astype("int")
            obstacle_y_top = global_width//2 + obstacle_half_size
            obstacle_y_bottom = global_width//2 - obstacle_half_size
            self.is_solely_blocked = (obstacle_size % 2 == 0)
        
        ## Decompose obstacle interaction region into six parts
        ##     part_left_top     2|5  part_right_top
        ##                       1|4
        ##     part_left_middle  1|4  part_right_middle
        ##                       1|4
        ##     part_left_bottom  3|6  part_right_bottom

        ## Left side of obstacle
        if obstacle_x_left in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_left)
            # contains 1,2,3
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top) + 1
                self.part_left_top = self[local_x_index, local_y_top]
                local_y_bottom = global_y_range.index(obstacle_y_bottom) + 1
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom+1 < local_y_top:
                    self.part_left_middle = self[local_x_index, local_y_bottom + 1:local_y_top]
                else:
                    self.part_left_middle = None
            # contains 1,2
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top) + 1
                self.part_left_top = self[local_x_index, local_y_top]
                self.part_left_bottom = None
                if local_y_top > 1: 
                    self.part_left_middle = self[local_x_index, 1:local_y_top]
                else:
                    self.part_left_middle = None
            # contains 1,3
            elif obstacle_y_bottom in global_y_range:
                self.part_left_top = None
                local_y_bottom = global_y_range.index(obstacle_y_bottom) + 1
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_left_middle = self[local_x_index, local_y_bottom + 1:local_width + 1]
                else:
                    self.part_left_middle = None
            # contains 1
            elif obstacle_y_bottom < global_y_range[0] and global_y_range[-1] < obstacle_y_top:
                self.part_left_top = None
                self.part_left_bottom = None
                self.part_left_middle = self[local_x_index, 1:-1]
            # contains none of 1,2,3
            else:
                self.part_left_top = None
                self.part_left_bottom = None
                self.part_left_middle = None
        else:  # not even a possibility
            self.part_left_top = None
            self.part_left_bottom = None
            self.part_left_middle = None
            
        ## Right side of obstacle
        if obstacle_x_right in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_right)
            # contains 4,5,6
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top) + 1
                self.part_right_top = self[local_x_index, local_y_top]
                local_y_bottom = global_y_range.index(obstacle_y_bottom) + 1
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom + 1 < local_y_top:
                    self.part_right_middle = self[local_x_index, local_y_bottom+1 : local_y_top]
                else:
                    self.part_right_middle = None
            # contains 4,5
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top) + 1
                self.part_right_top = self[local_x_index, local_y_top]
                self.part_right_bottom = None
                if local_y_top > 1: 
                    self.part_right_middle = self[local_x_index, 1 : local_y_top]
                else:
                    self.part_right_middle = None
            # contains 4,6
            elif obstacle_y_bottom in global_y_range:
                self.part_right_top = None
                local_y_bottom = global_y_range.index(obstacle_y_bottom) + 1
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_right_middle = self[local_x_index, local_y_bottom+1 : local_width+1]
                else:
                    self.part_right_middle = None
            # contains 4
            elif global_y_range[0] > obstacle_y_bottom and global_y_range[-1] < obstacle_y_top:
                self.part_right_top = None
                self.part_right_bottom = None
                self.part_right_middle = self[local_x_index, 1:-1]
            # contains none of 4,5,6
            else:
                self.part_right_top = None
                self.part_right_bottom = None
                self.part_right_middle = None
        else:  # not even a possibility
            self.part_right_top = None
            self.part_right_bottom = None
            self.part_right_middle = None


        # Varibles for data recording
        self.time = 0
        self.marks = [self.time]
        if global_length*3//4 in global_x_range and global_width//2 in global_y_range:
            key_point_x = global_x_range.index(global_length*3//4) + 1
            key_point_y = global_y_range.index(global_width//2) + 1
            # The key point for tracking velocity magnitude
            self.key_point = self[key_point_x, key_point_y]
            self.evolution = [np.linalg.norm(self.key_point.velocity)]
        else:
            self.key_point = None
                    
    def _flow_dynamics(self):
        """Based on simple flow with collision and streaming. Add periodic boundary by 
        the communication between different cores in 'Pre-streaming'. The rigid obstacle, 
        inlet and outlet are implemented almost the same as serial implementation in 
        'Before streaming' and 'After streaming', except with a judgement beforehand on 
        whether the local region contains the relative flow slices.
        """
        # Collision
        self[1:-1,1:-1].population += self.omega * (self[1:-1,1:-1].equilibrium - self[1:-1,1:-1].population)
        
        # After collision
        ## Rigid obstacle
        if self.part_left_middle is not None:
            sketch_left_middle = self.part_left_middle.population[C_RIGHT].copy()
        if self.part_left_top is not None:
            C_BLOCK = [8] if self.is_solely_blocked else [1,8]
            sketch_left_top = self.part_left_top.population[C_BLOCK].copy()
        if self.part_left_bottom is not None:
            C_BLOCK = [5] if self.is_solely_blocked else [1,5]
            sketch_left_bottom = self.part_left_bottom.population[C_BLOCK].copy()
        if self.part_right_middle is not None:
            sketch_right_middle = self.part_right_middle.population[C_LEFT].copy()
        if self.part_right_top is not None:
            C_BLOCK = [7] if self.is_solely_blocked else [3,7]
            sketch_right_top = self.part_right_top.population[C_BLOCK].copy()
        if self.part_right_bottom is not None:
            C_BLOCK = [6] if self.is_solely_blocked else [3,6]
            sketch_right_bottom = self.part_right_bottom.population[C_BLOCK].copy()
            
        # Pre-streaming
        ## Periodic right boundary
        sendbuf = self[-2,:].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(
            sendbuf, (core_world_rank + 1)%core_world_size, sendtag=1,
            recvbuf=recvbuf, source=(core_world_rank - 1)%core_world_size, recvtag=1)
        self[0,:].population[C_ALL] = recvbuf
        ## Periodic left boundary
        sendbuf = self[1,:].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(
            sendbuf, (core_world_rank - 1)%core_world_size, sendtag=2,
            recvbuf=recvbuf, source=(core_world_rank + 1)%core_world_size, recvtag=2)
        self[-1,:].population[C_ALL] = recvbuf
        ## Periodic top boundary
        sendbuf = self[:,-2].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(
            sendbuf, (core_world_rank + core_x_size)%core_world_size, sendtag=3,
            recvbuf=recvbuf, source=(core_world_rank - core_x_size)%core_world_size, recvtag=3)
        self[:,0].population[C_ALL] = recvbuf
        ## Periodic bottom boundary
        sendbuf = self[:,1].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(
            sendbuf, (core_world_rank - core_x_size)%core_world_size, sendtag=4,
            recvbuf=recvbuf, source=(core_world_rank + core_x_size)%core_world_size, recvtag=4)
        self[:,-1].population[C_ALL] = recvbuf
        
        # Normal streaming
        for index in range(1, C_FREEDOM):
            self.population[index] = np.roll(self.population[index], C_SET[index], axis=(0,1))    

        # After streaming
        ## Rigid obstacle
        if self.part_left_middle is not None:
            self.part_left_middle.population[C_LEFT] = sketch_left_middle
        if self.part_left_top is not None:
            C_BOUNCE = [6] if self.is_solely_blocked else [3,6]
            self.part_left_top.population[C_BOUNCE] = sketch_left_top
        if self.part_left_bottom is not None:
            C_BOUNCE = [7] if self.is_solely_blocked else [3,7]
            self.part_left_bottom.population[C_BOUNCE] = sketch_left_bottom
        if self.part_right_middle is not None:
            self.part_right_middle.population[C_RIGHT] = sketch_right_middle
        if self.part_right_top is not None:
            C_BOUNCE = [5] if self.is_solely_blocked else [1,5]
            self.part_right_top.population[C_BOUNCE] = sketch_right_top
        if self.part_right_bottom is not None:
            C_BOUNCE = [8] if self.is_solely_blocked else [1,8]
            self.part_right_bottom.population[C_BOUNCE] = sketch_right_bottom
        ## The very left inlet
        if core_x_rank == 0:
            MASS_IN = np.ones_like(self[1,1:-1].mass)
            VELOCITY_IN = np.zeros_like(self[1,1:-1].velocity)
            VELOCITY_IN[0] = 0.1
            self[1,1:-1].population[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        ## The very right outlet
        if core_x_rank == core_x_size - 1:
            self[-2,1:-1].population[C_LEFT] = self[-3,1:-1].population[C_LEFT].copy()
    
    def _record_data(self):
        """Only record data if this core contains the key point."""
        self.time += 1
        self.marks.append(self.time)
        if self.key_point is not None:
            self.evolution.append(np.linalg.norm(self.key_point.velocity))


# %%
def run_scale(num_step):
    """Simulate the evolution at different Reynolds number."""
    if core_world_rank == 0:
        print("---> scale --->")
    global_shape = 420, 180
    niu = 0.04
    u = 0.1
    L = 40
    karman_vortex = KarmanVortex(global_shape, initial_velocity=u,
                                 viscosity=niu, obstacle_size=L)
    
    # assert num_step % 10000 == 0
    # num_feedback = num_step // 10000
    # for count in range(num_feedback):
    #     karman_vortex.simulate(10000)
    #     if core_world_rank == 0:
    #         if count == num_feedback - 1:
    #             print("Finish at t={}!".format(karman_vortex.time))
    #         else:
    #             print("Evovle to t={}...".format(karman_vortex.time))

    karman_vortex.simulate(num_step)
    if karman_vortex.key_point is not None:
        with open("job0_evolution_c{}r{}.npy".format(
                core_world_size, core_world_rank), "wb") as f:
            np.save(f, karman_vortex.evolution)
        print("{} core has saved data!".format(core_world_rank))


# %%
def run_job1(num_step):
    """Simulate the evolution at different Reynolds number."""
    if core_world_rank == 0:
        print("---> job1 --->")
    global_shape = 420, 180
    niu = 0.04
    u = 0.1
    reynolds_axis = np.linspace(40, 200, num=20)
    evolution_axis = []

    for reynolds in reynolds_axis:
        L = int(round(reynolds * niu / u))
        karman_vortex = KarmanVortex(global_shape, initial_velocity=u,
                                     viscosity=niu, obstacle_size=L)
        karman_vortex.simulate(num_step)
        if karman_vortex.key_point is not None:
            evolution_axis.append(karman_vortex.evolution)
            print("Re={} done".format(reynolds))

    if evolution_axis != []:
        with open("job1_evolution_c{}r{}.npy".format(
                core_world_size, core_world_rank), "wb") as f:
            np.save(f, reynolds_axis)
            np.save(f, evolution_axis)
        print("{} core has saved data!".format(core_world_rank))


# %%
def run_job2(num_step):
    """Simulate the evolution at different length of flow."""
    if core_world_rank == 0:
        print("---> job2 --->")
    L = 40
    niu = 0.04
    u = 0.1
    # reynolds = L*u/niu = 100
    nx_axis = np.linspace(400, 600, num=20, dtype=int)
    evolution_axis = []

    for nx in nx_axis:
        global_shape = nx, 180
        karman_vortex = KarmanVortex(global_shape, initial_velocity=u,
                                     viscosity=niu, obstacle_size=L)
        karman_vortex.simulate(num_step)
        if karman_vortex.key_point is not None:
            evolution_axis.append(karman_vortex.evolution)
            print("nx={} done".format(nx))
    
    if evolution_axis != []:
        with open("job2_evolution_c{}r{}.npy".format(
                core_world_size, core_world_rank), "wb") as f:
            np.save(f, nx_axis)
            np.save(f, evolution_axis)
        print("{} core has saved data!".format(core_world_rank))


# %%
def run_job3(num_step):
    """Simulate the evolution at different blockage ratio."""
    if core_world_rank == 0:
        print("---> job3 --->")
    global_shape = 420, 180
    reynolds = 100
    u = 0.1
    B_axis = np.linspace(0.05, 0.95, num=20).round(decimals=2)
    evolution_axis = []

    for B in B_axis:
        L = int(round(global_shape[1] * B))
        niu = L * u / reynolds
        karman_vortex = KarmanVortex(global_shape, initial_velocity=u, 
                                     viscosity=niu, obstacle_size=L)
        karman_vortex.simulate(num_step)
        if karman_vortex.key_point is not None:
            evolution_axis.append(karman_vortex.evolution)
            print("B={} done".format(B))

    if evolution_axis != []:
        with open("job3_evolution_c{}r{}.npy".format(
                core_world_size, core_world_rank), "wb") as f:
            np.save(f, B_axis)
            np.save(f, evolution_axis)
        print("{} core has saved data!".format(core_world_rank))


# %%
if __name__ == "__main__":
    start_time = timeit.default_timer()
    num_step = 50000
    if len(sys.argv) == 1:
        print("No job is assigned！Possible choice of jobs includes:")
        print("||\tscale: an evolution of {} steps".format(num_step))
        print("||\tjob1: Re-St relation")
        print("||\tjob2: nx-St relation")
        print("||\tjob3: B-St relation")
    elif sys.argv[1] == "scale":
        run_scale(num_step)
    elif sys.argv[1] == "job1":
        run_job1(num_step)
    elif sys.argv[1] == "job2":
        run_job2(num_step)
    elif sys.argv[1] == "job3":
        run_job3(num_step)
    else:
        print("Undefined job. Possible choice of jobs includes:")
        print("||\tscale: an evolution of {} steps".format(num_step))
        print("||\tjob1: Re-St relation")
        print("||\tjob2: nx-St relation")
        print("||\tjob3: B-St relation")
    end_time = timeit.default_timer()
    execution_time = np.array(end_time - start_time)
    reduced_execution_time = np.empty_like(execution_time)
    comm.Reduce(execution_time, reduced_execution_time, op=MPI.SUM, root=0)
    if core_world_rank == 0:
        reduced_execution_time /= core_world_size
        print("Execution time = {} seconds".format(reduced_execution_time))


# %%
