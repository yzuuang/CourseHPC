# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

from mpi4py import MPI
import numpy as np
# import matplotlib
# matplotlib.use("TKagg")
import matplotlib.pyplot as plt
# import matplotlib.animation as anime


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# get two divisors, bigger one for x direction(length direction)
for divisor in range(np.sqrt(size).astype("int"), 0, -1):
    if size % divisor == 0:
        break
core_x_size = size // divisor
core_y_size = divisor
# arrange cores in x direction first
core_x_rank = rank % core_x_size
core_y_rank = rank // core_x_size
# an example for 6 cores:
#     |3|4|5|
#     |0|1|2|
print("rank = {} / {}; arrange = {},{} / {}x{}".format(rank, size, core_x_rank, core_y_rank, core_x_size, core_y_size))


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
# group special indices together as a direction
C_UP = np.array([2, 5, 6], dtype="intp")
C_DOWN = np.array([4, 7, 8], dtype="intp")
C_LEFT = np.array([3, 6, 7], dtype="intp")
C_RIGHT = np.array([1, 8, 5], dtype="intp")
C_ALL = slice(None)


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
            self.record_data()
#             assert np.all(self.population > 0), np.where(self.population <= 0)
    
    def onestep_flow(self):
        pass
    
    def record_data(self):
        pass


# +
class KarmanVortex(BaseFlow):
    
    def __init__(self, global_shape, *, viscosity, blockage_ratio):
#         get the global shape
        global_length, global_width = global_shape
        assert global_length % 4 == 0 and global_width % 2 == 0
        
#         analyze the local shape
        local_length = np.ceil(global_length / core_x_size).astype("int")
        if core_x_rank == core_x_size - 1:
            local_length = global_length - (core_x_size - 1)*local_length
        
        local_width = np.ceil(global_width / core_y_size).astype("int")
        if core_y_rank == core_y_size -1:
            local_width = global_width - (core_y_size - 1)*local_width

#         initialize with respective slice of flow
        local_mass = np.full((local_length, local_width), 1)
        local_velocity = np.full((local_length, local_width), (0.1, 0), dtype=(float, 2))
        local_velocity = np.moveaxis(local_velocity, -1, 0)
        local_velocity[:,:,local_width//2:] += 10**(-6)
        self.population = self.compute_equilibrium(local_mass, local_velocity)
        
#         create ghost cells
        self.population = np.insert(self.population, [0,local_length], 0, axis=1)
        self.population = np.insert(self.population, [0,local_width], 0, axis=2)        
        
#         coefficient omega
        self.omega = 2 / (6*viscosity + 1)        

#         corresponding global indices
        if core_x_rank == core_x_size - 1:
            global_x_range = range(global_length - local_length, global_length)
        else:
            global_x_range = range(core_x_rank*local_length, (core_x_rank + 1)*local_length)
        
        if core_y_rank == core_y_size - 1:
            global_y_range = range(global_width - local_width, global_width)
        else:
            global_y_range = range(core_y_rank*local_width, (core_y_rank + 1)*local_width)
        
#         obstacle position in global indices
        assert blockage_ratio > 0 and blockage_ratio < 1
        self.obstacle_size = round(global_width * blockage_ratio)
        self.obstacle_is_odd = (self.obstacle_size % 2 == 1)
        obstacle_x_left = global_length//4 - 1
        obstacle_x_right = global_length//4
        obstacle_half_size = np.ceil(self.obstacle_size/2).astype("int")
        obstacle_y_top = global_width//2 + obstacle_half_size - 1
        obstacle_y_bottom = global_width//2 - obstacle_half_size
        
#         decompose obstacle interaction area into six parts
#             part_left_top     2|5  part_right_top
#                               1|4
#             part_left_side    1|4  part_right_side
#                               1|4
#             part_left_bottom  3|6  part_right_bottom

#         left of obstacle
        if obstacle_x_left in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_left)
#             contains 1,2,3
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_left_top = self[local_x_index, local_y_top]
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom+1 < local_y_top:
                    self.part_left_side = self[local_x_index, local_y_bottom + 1:local_y_top]
#             contains 1,2
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_left_top = self[local_x_index, local_y_top]
                if local_y_top > 1: 
                    self.part_left_side = self[local_x_index, 1:local_y_top]
#             contains 1,3
            elif obstacle_y_bottom in global_y_range:
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_left_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_left_side = self[local_x_index, local_y_bottom + 1:local_width + 1]
#             contains 1
            elif obstacle_y_bottom < global_y_range[0] and global_y_range[-1] < obstacle_y_top:
                self.part_left_side = self[local_x_index, 1:-1]
            
#         right of obstacle
        if obstacle_x_right in global_x_range:
            local_x_index = global_x_range.index(obstacle_x_right)
#             contains 4,5,6
            if obstacle_y_top in global_y_range and obstacle_y_bottom in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_right_top = self[local_x_index, local_y_top]
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom + 1 < local_y_top:
                    self.part_right_side = self[local_x_index, local_y_bottom+1 : local_y_top]
#             contains 4,5
            elif obstacle_y_top in global_y_range:
                local_y_top = global_y_range.index(obstacle_y_top)
                self.part_right_top = self[local_x_index, local_y_top]
                if local_y_top > 1: 
                    self.part_right_side = self[local_x_index, 1 : local_y_top]
#            contains 4,6
            elif obstacle_y_bottom in global_y_range:
                local_y_bottom = global_y_range.index(obstacle_y_bottom)
                self.part_right_bottom = self[local_x_index, local_y_bottom]
                if local_y_bottom < local_width:
                    self.part_right_side = self[local_x_index, local_y_bottom+1 : local_width+1]
#             contains 4
            elif global_y_range[0] > obstacle_y_bottom and global_y_range[-1] < obstacle_y_top:
                self.part_right_side = self[local_x_index, 1 : -1]
            
#         key point to track velocity magnitude, get local indices
        if global_length*3//4 in global_x_range and global_width//2 in global_y_range:
            self.time = 0
            self.marks = [self.time]
            key_point_x = global_x_range.index(global_length*3//4)
            key_point_y = global_y_range.index(global_width//2)
            self.key_point = self[key_point_x, key_point_y]
            self.key_point_u_norm = [np.sqrt(np.sum(self.key_point.velocity**2))]
        else:
            self.key_point = None
                    
    def onestep_flow(self):
#         collision
        self[1:-1,1:-1].population += self.omega * (self[1:-1,1:-1].equilibrium - self[1:-1,1:-1].population)
        
#         copy the block state before streaming
        if "part_left_side" in self.__dict__.keys():
            sketch_left_side = self.part_left_side.population[C_RIGHT].copy()
        if "part_left_top" in self.__dict__.keys():
            C_BLOCK = [8] if self.obstacle_is_odd else [1,8]
            sketch_left_top = self.part_left_top.population[C_BLOCK].copy()
        if "part_left_bottom" in self.__dict__.keys():
            C_BLOCK = [5] if self.obstacle_is_odd else [1,5]
            sketch_left_bottom = self.part_left_bottom.population[C_BLOCK].copy()
        if "part_right_side" in self.__dict__.keys():
            sketch_right_side = self.part_right_side.population[C_LEFT].copy()
        if "part_right_top" in self.__dict__.keys():
            C_BLOCK = [7] if self.obstacle_is_odd else [3,7]
            sketch_right_top = self.part_right_top.population[C_BLOCK].copy()
        if "part_right_bottom" in self.__dict__.keys():
            C_BLOCK = [6] if self.obstacle_is_odd else [3,6]
            sketch_right_bottom = self.part_right_bottom.population[C_BLOCK].copy()
            
#         periodic by ghost cell
#         send to the right
        sendbuf = self[-2, :].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank + 1)%size, recvbuf=recvbuf, source=(rank - 1)%size)
        self[0, :].population[C_ALL] = recvbuf
#         send to the left
        sendbuf = self[1, :].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank - 1)%size, recvbuf=recvbuf, source=(rank + 1)%size)
        self[-1, :].population[C_ALL] = recvbuf
#         send to the above
        sendbuf = self[:, -2].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank + core_x_size)%size, recvbuf=recvbuf, source=(rank - core_x_size)%size)
        self[:, 0].population[C_ALL] = recvbuf
#         send to the below
        sendbuf = self[:, 1].population[C_ALL].copy()
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf, (rank - core_x_size)%size, recvbuf=recvbuf, source=(rank + core_x_size)%size)
        self[:, -1].population[C_ALL] = recvbuf
        
#         streaming
        for channel in range(1, C_FREEDOM):
            self.population[channel] = np.roll(self.population[channel], C_SET[channel], axis=(0,1))    

#         recover the bounce back after streaming
        if "part_left_side" in self.__dict__.keys():
            self.part_left_side.population[C_LEFT] = sketch_left_side
        if "part_left_top" in self.__dict__.keys():
            C_BOUNCE = [6] if self.obstacle_is_odd else [3,6]
            self.part_left_top.population[C_BOUNCE] = sketch_left_top
        if "part_left_bottom" in self.__dict__.keys():
            C_BOUNCE = [7] if self.obstacle_is_odd else [3,7]
            self.part_left_bottom.population[C_BOUNCE] = sketch_left_bottom
        if "part_right_side" in self.__dict__.keys():
            self.part_right_side.population[C_RIGHT] = sketch_right_side
        if "part_right_top" in self.__dict__.keys():
            C_BOUNCE = [5] if self.obstacle_is_odd else [1,5]
            self.part_right_top.population[C_BOUNCE] = sketch_right_top
        if "part_right_bottom" in self.__dict__.keys():
            C_BOUNCE = [8] if self.obstacle_is_odd else [1,8]
            self.part_right_bottom.population[C_BOUNCE] = sketch_right_bottom

#         left inlet
        if core_x_rank == 0:
            MASS_IN = np.ones_like(self[1, 1:-1].mass)
            VELOCITY_IN = np.zeros_like(self[1, 1:-1].velocity)
            VELOCITY_IN[0] = 0.1
            self[1, 1:-1].population[C_ALL] = self.compute_equilibrium(MASS_IN, VELOCITY_IN)
        
#         right outlet
        if core_x_rank == core_x_size - 1:
            self[-2, 1:-1].population[C_LEFT] = self[-3, 1:-1].population[C_LEFT].copy()
    
    def record_data(self):
        if self.key_point is not None:
            self.time += 1
            self.marks.append(self.time)
            self.key_point_u_norm.append(np.sqrt(np.sum(self.key_point.velocity**2)))


# +
global_shape = 200, 90
niu = 0.04
karman_vortex = KarmanVortex(global_shape, viscosity=niu, blockage_ratio=2/9)

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
if karman_vortex.key_point is not None:
    print("the {}({},{}) core is saving data...".format(rank, core_x_rank, core_y_rank))
    with open("{}cores_data.npy".format(size), "wb") as f:
        np.save(f, karman_vortex.marks)
        np.save(f, karman_vortex.key_point_u_norm)
        np.save(f, global_shape[1] - karman_vortex.obstacle_size)
        np.save(f, niu)
    print("done!")




