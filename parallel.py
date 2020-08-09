# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np


# %%
# channels of D2Q9 Lattice Bolzman Method
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
# weight of channels
C_WEIGHT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
# D2Q9, 9 is freedom, 2 is dimension
C_FREEDOM, C_DIMENSION = np.shape(C_SET)
# group special indices of channels together according to their directions
C_UP = np.array([2, 5, 6], dtype="intp")
C_DOWN = np.array([4, 7, 8], dtype="intp")
C_LEFT = np.array([3, 6, 7], dtype="intp")
C_RIGHT = np.array([1, 8, 5], dtype="intp")
C_ALL = slice(None)


# %%
# Basic core information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Get two divisors. The bigger will be the size in x direction.
for divisor in range(np.sqrt(size).astype("int"), 0, -1):
    if size % divisor == 0:
        break
core_x_size = size // divisor
core_y_size = divisor
del divisor

# Arrange cores in x direction first
# an example for 8 cores:
#     |4|5|6|7|
#     |0|1|2|3|
core_x_rank = rank % core_x_size
core_y_rank = rank // core_x_size
print("rank = {} / {}; arrange = {},{} / {}x{}".format(rank, size, core_x_rank, core_y_rank, core_x_size, core_y_size))


# %%
class BaseFlow(object):
    """Common features related to distribution.

    Important properties such as mass, velocity, equilibrium and distribution.
    Support of getting and setting(if settable) properties for only a slice of flow.
    Public method for simulation, with a framework of recording data after flow mechanics.

    Attributes:
        distribution: the probability density about the whole flow or a slice of the flow,
                    with the first axis always be the freedom axis.
    """
    
    def __init__(self, distribution):
        """Save the distribution data."""
        assert distribution.shape[0] == C_FREEDOM
        self.distribution = distribution
        
    def __getitem__(self, index):
        """Slice the flow.
        Note: to set all channels of the distribution of a flow slice, one must use 
        a slicing operator. The reason remains unclear.
        e.g., some_flow[x_slice, y_slice].distribution[C_ALL] = some_values
        """
        assert len(index) == 2
        return BaseFlow(self.distribution[:, index[0], index[1]])
    
    @property
    def mass(self):
        """Get mass density from the distribution of the instance. Read-only property."""
        return np.einsum("f...->...", self.distribution)
    
    @property
    def velocity(self):
        """Get velocity from the distribution of the instance. Read-only property."""
        momentum = np.einsum("fd,f...->d...", C_SET, self.distribution)
        divisible_mass = self.mass.copy()
        divisible_mass[divisible_mass == 0] = np.inf  # divide by infinity gives back 0
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
    
    def flowing(self, num_time):
        """Simulate the flow for a given number of timesteps."""
        assert isinstance(num_time, int) and num_time >= 0
        for _ in range(num_time):
            self._flow_mechanics()
            assert np.all(self.mass > 0), "problems at {}".format(np.where(self.mass <= 0))
            self._record_data()
    
    def _flow_mechanics(self):
        """To be implemented in child class."""
        pass
    
    def _record_data(self):
        """To be implemented in child class."""
        pass


# %%
class KarmanVortex(BaseFlow):
    """2D Karman vortex street
    
    Overload __init__ method.
    Implement _flow_mechanics and _record_data method.
    
    Attributes:
        distribution: the probability density about the whole flow or a slice of the flow,
            with the first axis always be the freedom axis.
        omega: collision coefficient.
        obstacle_size: an integer value representing the size of obstacle.
        obstacle_is_odd: a bool value telling whether the obstacle size is an odd number.
        part_left_side: If the local region contains the respective region of obstacle,
            it is the flow slice of that region. Otherwise, it is None.
        part_left_top: similar to part_left_size.
        part_left_bottom: similar to part_left_size.
        part_right_side: similar to part_left_size.
        part_right_top: similar to part_left_size.
        part_right_bottom: similar to part_left_size.
        __time: private time axis. Automatically add 1 (dt=1) after each step.
        key_point: If the local region contains the key point to be tracked, it is the
            flow slice of that point. Otherwise, it is None.
        marks: time records. Will only be created when key_point is not None.
        key_point_u_norm: The data tracking the norm of velocity at the key point. Will
            only be created when key_point is not None.
    """

    def __init__(self, global_shape, *, viscosity, obstacle_size):
        """Initilize distribution by computing equilibrium from the mass density and 
        velocity of local flow. Prepare varibles to be used in _flow_mechanics and 
        _record_data.
        """
        # Initialize the distribution
        ## Get the global shape
        global_length, global_width = global_shape
        assert global_length % 4 == 0 and global_width % 2 == 0
        
        ## Analyze the local shape
        local_length = np.ceil(global_length / core_x_size).astype("int")
        if core_x_rank == core_x_size - 1:
            local_length = global_length - (core_x_size - 1)*local_length
        
        local_width = np.ceil(global_width / core_y_size).astype("int")
        if core_y_rank == core_y_size - 1:
            local_width = global_width - (core_y_size - 1)*local_width

        ## Initialize with respective local shape
        local_mass = np.full((local_length, local_width), 1)
        local_velocity = np.full((local_length, local_width), (0.1, 0), dtype=(float, 2))
        local_velocity = np.moveaxis(local_velocity, -1, 0)  # move the axis of dimension to the 0 axis
        local_velocity[:,:,local_width//2:] += 10**(-6)  # perturbation in upper half
        self.distribution = self.compute_equilibrium(local_mass, local_velocity)
        
        ## Add ghost cells
        self.distribution = np.insert(self.distribution, [0,local_length], 0, axis=1)
        self.distribution = np.insert(self.distribution, [0,local_width], 0, axis=2)        
        
        # Varibles for flow mechanics
        ## Coefficient omega
        self.omega = 2 / (6*viscosity + 1)        

        ## Corresponding global indices
        if core_x_rank == core_x_size - 1:
            global_x_range = range(global_length-local_length, global_length)
        else:
            global_x_range = range(core_x_rank*local_length, (core_x_rank + 1)*local_length)
        
        if core_y_rank == core_y_size - 1:
            global_y_range = range(global_width-local_width, global_width)
        else:
            global_y_range = range(core_y_rank*local_width, (core_y_rank + 1)*local_width)
        
        ## Obstacle position in global indices
        assert isinstance(obstacle_size, int) and obstacle_size > 0 and obstacle_size < global_width
        self.obstacle_size = obstacle_size
        self.obstacle_is_odd = (self.obstacle_size % 2 == 1)
        obstacle_x_left = global_length//4 - 1
        obstacle_x_right = global_length//4
        obstacle_half_size = np.ceil(self.obstacle_size/2).astype("int")
        obstacle_y_top = global_width//2 + obstacle_half_size - 1
        obstacle_y_bottom = global_width//2 - obstacle_half_size
        
        ## Decompose obstacle interaction area into six parts
        ##     part_left_top     2|5  part_right_top
        ##                       1|4
        ##     part_left_side    1|4  part_right_side
        ##                       1|4
        ##     part_left_bottom  3|6  part_right_bottom

        ## Left of obstacle
        if obstacle_x_left in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_left)
            # contains 1,2,3
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_left_top = self[local_x_index, local_y_top]
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom+1 < local_y_top:
                    self.part_left_side = self[local_x_index, local_y_bottom + 1:local_y_top]
                else:
                    self.part_left_side = None
            # contains 1,2
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_left_top = self[local_x_index, local_y_top]
                self.part_left_bottom = None
                if local_y_top > 1: 
                    self.part_left_side = self[local_x_index, 1:local_y_top]
                else:
                    self.part_left_side = None
            # contains 1,3
            elif obstacle_y_bottom in global_y_range:
                self.part_left_top = None
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_left_side = self[local_x_index, local_y_bottom + 1:local_width + 1]
                else:
                    self.part_left_side = None
            # contains 1
            elif obstacle_y_bottom < global_y_range[0] and global_y_range[-1] < obstacle_y_top:
                self.part_left_top = None
                self.part_left_bottom = None
                self.part_left_side = self[local_x_index, 1:-1]
            # contains none of 1,2,3
            else:
                self.part_left_top = None
                self.part_left_bottom = None
                self.part_left_side = None
            
        ## Right of obstacle
        if obstacle_x_right in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_right)
            # contains 4,5,6
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_right_top = self[local_x_index, local_y_top]
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom + 1 < local_y_top:
                    self.part_right_side = self[local_x_index, local_y_bottom+1 : local_y_top]
                else:
                    self.part_right_side = None
            # contains 4,5
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_right_top = self[local_x_index, local_y_top]
                self.part_right_bottom = None
                if local_y_top > 1: 
                    self.part_right_side = self[local_x_index, 1 : local_y_top]
                else:
                    self.part_right_side = None
            # contains 4,6
            elif obstacle_y_bottom in global_y_range:
                self.part_right_top = None
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_right_side = self[local_x_index, local_y_bottom+1 : local_width+1]
                else:
                    self.part_right_side = None
            # contains 4
            elif global_y_range[0] > obstacle_y_bottom and global_y_range[-1] < obstacle_y_top:
                self.part_right_top = None
                self.part_right_bottom = None
                self.part_right_side = self[local_x_index, 1:-1]
            # contains none of 4,5,6
            else:
                self.part_right_top = None
                self.part_right_bottom = None
                self.part_right_side = None
            
        # Varibles for data recording
        self.__time = 0
        if global_length*3//4 in global_x_range and global_width//2 in global_y_range:
            key_point_x = global_x_range.index(global_length*3//4)
            key_point_y = global_y_range.index(global_width//2)
            # The key point for tracking velocity magnitude
            self.key_point = self[key_point_x, key_point_y]
            # Time and velocity magnitude data varibles
            self.marks = [self.__time]
            norm = lambda v: np.sqrt(np.sum(v**2))
            self.key_point_u_norm = [norm(self.key_point.velocity**2)]
        else:
            self.key_point = None
                    
    def _flow_mechanics(self):
        """Based on simple flow with collision and streaming. Add periodic boundary by 
        the communication between different cores in 'Pre-streaming'. The rigid obstacle, 
        inlet and outlet are implemented almost the same as serial implementation in 
        'Before streaming' and 'After streaming', except with a judgement beforehand on 
        whether the local region contains the relative flow slices.
        """
        # Collision
        self[1:-1,1:-1].distribution += self.omega * (self[1:-1,1:-1].equilibrium - self[1:-1,1:-1].distribution)
        
        # Before streaming
        #    Rigid obstacle
        if self.part_left_side is not None:
            sketch_left_side = self.part_left_side.distribution[C_RIGHT].copy()
        if self.part_left_top is not None:
            C_BLOCK = [8] if self.obstacle_is_odd else [1,8]
            sketch_left_top = self.part_left_top.distribution[C_BLOCK].copy()
        if self.part_left_bottom is not None:
            C_BLOCK = [5] if self.obstacle_is_odd else [1,5]
            sketch_left_bottom = self.part_left_bottom.distribution[C_BLOCK].copy()
        if self.part_right_side is not None:
            sketch_right_side = self.part_right_side.distribution[C_LEFT].copy()
        if self.part_right_top is not None:
            C_BLOCK = [7] if self.obstacle_is_odd else [3,7]
            sketch_right_top = self.part_right_top.distribution[C_BLOCK].copy()
        if self.part_right_bottom is not None:
            C_BLOCK = [6] if self.obstacle_is_odd else [3,6]
            sketch_right_bottom = self.part_right_bottom.distribution[C_BLOCK].copy()
            
        # Pre-streaming
        #   Periodic right boundary
        sendbuf = self[-2,:].distribution[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank + 1)%size, recvbuf=recvbuf, source=(rank - 1)%size)
        self[0,:].distribution[C_ALL] = recvbuf
        #   Periodic left boundary
        sendbuf = self[1,:].distribution[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank - 1)%size, recvbuf=recvbuf, source=(rank + 1)%size)
        self[-1,:].distribution[C_ALL] = recvbuf
        #   Periodic top boundary
        sendbuf = self[:,-2].distribution[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank + core_x_size)%size, recvbuf=recvbuf, source=(rank - core_x_size)%size)
        self[:,0].distribution[C_ALL] = recvbuf
        #   Periodic bottom boundary
        sendbuf = self[:,1].distribution[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank - core_x_size)%size, recvbuf=recvbuf, source=(rank + core_x_size)%size)
        self[:,-1].distribution[C_ALL] = recvbuf
        
        # Streaming
        for channel in range(1, C_FREEDOM):
            self.distribution[channel] = np.roll(self.distribution[channel], C_SET[channel], axis=(0,1))    

        # After streaming
        #   Rigid obstacle
        if self.part_left_side is not None:
            self.part_left_side.distribution[C_LEFT] = sketch_left_side
        if self.part_left_top is not None:
            C_BOUNCE = [6] if self.obstacle_is_odd else [3,6]
            self.part_left_top.distribution[C_BOUNCE] = sketch_left_top
        if self.part_left_bottom is not None:
            C_BOUNCE = [7] if self.obstacle_is_odd else [3,7]
            self.part_left_bottom.distribution[C_BOUNCE] = sketch_left_bottom
        if self.part_right_side is not None:
            self.part_right_side.distribution[C_RIGHT] = sketch_right_side
        if self.part_right_top is not None:
            C_BOUNCE = [5] if self.obstacle_is_odd else [1,5]
            self.part_right_top.distribution[C_BOUNCE] = sketch_right_top
        if self.part_right_bottom is not None:
            C_BOUNCE = [8] if self.obstacle_is_odd else [1,8]
            self.part_right_bottom.distribution[C_BOUNCE] = sketch_right_bottom
        #   The very left inlet
        if core_x_rank == 0:
            MASS_IN = np.ones_like(self[1,1:-1].mass)
            VELOCITY_IN = np.zeros_like(self[1,1:-1].velocity)
            VELOCITY_IN[0] = 0.1
            self[1,1:-1].distribution[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        #   The very right outlet
        if core_x_rank == core_x_size - 1:
            self[-2,1:-1].distribution[C_LEFT] = self[-3,1:-1].distribution[C_LEFT].copy()
    
    def _record_data(self):
        """Only record data if this core contains the key point."""
        self.__time += 1
        if self.key_point is not None:
            self.marks.append(self.__time)
            norm = lambda v: np.sqrt(np.sum(v**2))
            self.key_point_u_norm.append(norm(self.key_point.velocity))


# %%
def simulate_compute_strouhal(the_flow, L, u):
    """Simulate the flow for a certian timesteps. Analyze the main frequency of 
    vortex awake, and compute Strouhal number afterwards from the main frequency 
    and the given characteristic length L and flow velocity u.
    """
    num_time = 30000
    the_flow.flowing(num_time)
    if the_flow.key_point is None:
        return None
    else:
        steady_time = np.array(the_flow.marks[20000:])
        fft_freq = np.fft.fftfreq(steady_time.size)
        steady_magnitude = np.array(the_flow.key_point_u_norm[20000:])
        fft_magnitude = np.abs(np.fft.fft(steady_magnitude - steady_magnitude.mean()))
        main_freq = np.abs(fft_freq[np.argmax(fft_magnitude)])
        strouhal = main_freq * L / u
        return strouhal


# %%
def run_job1():
    """Simulate and analyze the relation between Strouhal number and Reynolds number."""
    global_shape = 420, 180
    niu = 0.04
    u = 0.1
    reynolds_axis = np.linspace(40, 200, num=17)
    strouhal_axis = []
    for reynolds in reynolds_axis:
        L = round(reynolds * niu / u)
        karman_vortex = KarmanVortex(global_shape, viscosity=niu, obstacle_size=L)
        strouhal = simulate_compute_strouhal(karman_vortex, L, u)
        if strouhal is not None:
            strouhal_axis.append(strouhal)
            print("when Re = {}, St={}".format(reynolds, strouhal))
    
    if strouhal_axis != []:
        print("the {}({},{}) core is saving data...".format(rank, core_x_rank, core_y_rank))
        with open("job1_Re_St_relation.npy", "wb") as f:
            np.save(f, reynolds_axis)
            np.save(f, strouhal_axis)
        print("job1 done!")


# %%
def run_job2():
    """Simulate and analyze the relation between Strouhal number and the length of flow."""
    # reynolds = 100 = L*u/niu
    L = 40
    u = 0.1
    niu = 0.04
    nx_axis = np.linspace(400, 600, num=11).astype("int")
    strouhal_axis = []
    for nx in nx_axis:
        global_shape = nx, 180
        karman_vortex = KarmanVortex(global_shape, viscosity=niu, obstacle_size=L)
        strouhal = simulate_compute_strouhal(karman_vortex, L, u)
        if strouhal is not None:
            strouhal_axis.append(strouhal)
            print("when nx = {}, St={}".format(nx, strouhal))
    
    if strouhal_axis != []:
        print("the {}({},{}) core is saving data...".format(rank, core_x_rank, core_y_rank))
        with open("job2_St_nx_relation.npy", "wb") as f:
            np.save(f, nx_axis)
            np.save(f, strouhal_axis)
        print("job2 done!")


# %%
def run_job3():
    """Simulate and analyze the relation between Strouhal number and blockage ratio."""
    global_shape = 420, 180
    reynolds = 100
    u = 0.1
    B_axis = np.linspace(0.1, 0.9, num=9).round(decimals=1)
    strouhal_axis = []
    for B in B_axis:
        L = round(global_shape[1] * B)
        niu = L * u / reynolds
        karman_vortex = KarmanVortex(global_shape, viscosity=niu, obstacle_size=L)
        strouhal = simulate_compute_strouhal(karman_vortex, L, u)
        if strouhal is not None:
            strouhal_axis.append(strouhal)
            print("when B = {}, St={}".format(B, strouhal))
    
    if strouhal_axis != []:
        print("the {}({},{}) core is saving data...".format(rank, core_x_rank, core_y_rank))
        with open("job3_St_B_relation.npy", "wb") as f:
            np.save(f, B_axis)
            np.save(f, strouhal_axis)
        print("job3 done!")


# %%
if __name__ == "__main__":   
    run_job1()
    run_job2()
    run_job3()

