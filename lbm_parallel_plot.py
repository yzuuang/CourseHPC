# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# %%
def analyze_frequency(evolution, *, time_slice=slice(25000, None)):
    """Analyze the frequency of flow oscillation"""
    steady_time = np.arange(len(evolution))[time_slice]
    fft_freq = np.fft.fftfreq(steady_time.size)

    steady_magnitude = np.array(evolution)[time_slice]
    fft_magnitude = np.abs(np.fft.fft(steady_magnitude - steady_magnitude.mean()))
    
    main_freq_index = np.argmax(fft_magnitude)
    main_freq = np.abs(fft_freq[main_freq_index])
    return main_freq

# %%
with open("job0_evolution_c35r19.npy", "rb") as f:
    evolution = np.load(f)

fig, ax = plt.subplots()
ax.plot(range(len(evolution)), evolution, "-", label="numeric")
ax.legend(loc="best")
ax.set_title("Evolution of velocity magnitude")
ax.set_xlabel("time")
ax.set_ylabel("velocity magnitude")
fig.savefig("ms6_job0_evolution")

steady_time = np.arange(len(evolution))[25000:]
fft_freq = np.fft.fftfreq(steady_time.size)
steady_magnitude = np.array(evolution)[25000:]
fft_magnitude = np.abs(np.fft.fft(steady_magnitude))

fig, ax = plt.subplots()
ax.plot(fft_freq, fft_magnitude, "-", label="numeric")
ax.legend(loc="best")
ax.set_title("Frequency spectrum")
ax.set_xlabel("frequency")
ax.set_ylabel("magnitude")
ax.set_xlim(-5e-3, 5e-3)
fig.savefig("ms6_job0_spectrum")

f = analyze_frequency(evolution, time_slice=slice(25000,None))
print("Oscillation period = {} steps".format(1/f))


# %%
# The relation between Reynolds number and Strouhal number
with open("job1_evolution_c35r19.npy", "rb") as f:
    reynolds_axis = np.load(f)
    evolution_axis = np.load(f)

for reynolds, evolution in zip(reynolds_axis, evolution_axis):
    fig, ax = plt.subplots()
    ax.plot(range(len(evolution)), evolution, "-", label="numeric")
    ax.legend(loc="best")
    ax.set_title("Evolution at reynolds = {}".format(reynolds))
    ax.set_xlabel("time")
    ax.set_ylabel("velocity magnitude")
    plt.show()

strouhal_axis = []
for reynolds, evolution in zip(reynolds_axis, evolution_axis):
    f = analyze_frequency(evolution, time_slice=slice(25000, None))
    niu = 0.04
    u = 0.1
    L = int(round(reynolds * niu / u))
    strouhal = f * L / u
    strouhal_axis.append(strouhal)

fig, ax = plt.subplots()
ax.plot(reynolds_axis, strouhal_axis, "o-")
ax.set_title("St-Re relation")
ax.set_xlabel("Reynolds number")
ax.set_ylabel("Strouhal number")
fig.savefig("ms6_job1_relation.png")


# %%
# The relation between number of x-nodes(nx) and Strouhal number
with open("job2_evolution_c35r19.npy", "rb") as f:
    nx_axis = np.load(f)
    evolution_axis = np.load(f)

for nx, evolution in zip(nx_axis, evolution_axis):
    fig, ax = plt.subplots()
    ax.plot(evolution, "-", label="numeric")
    ax.legend(loc="best")
    ax.set_title("Evolution at nx = {}".format(nx))
    ax.set_xlabel("time")
    ax.set_ylabel("velocity magnitude")
    plt.show()


strouhal_axis = []
for evolution in evolution_axis:
    f = analyze_frequency(evolution, time_slice=slice(25000, None))
    L = 180
    u = 0.1
    strouhal = f * L / u
    strouhal_axis.append(strouhal)

fig, ax = plt.subplots()
ax.plot(nx_axis, strouhal_axis, "o-")
ax.set_title("St-nx relation")
ax.set_xlabel("Number of x-nodes")
ax.set_ylabel("Strouhal number")
fig.savefig("ms6_job2_relation.png")


# %%
# The relation between blockage ratio(B) and Strouhal number(St)
with open("job3_evolution_c35r19.npy", "rb") as f:
    B_axis = np.load(f)
    evolution_axis = np.load(f)

for B, evolution in zip(B_axis, evolution_axis):
    fig, ax = plt.subplots()
    ax.plot(range(len(evolution)), evolution, "-", label="numeric")
    ax.legend(loc="best")
    ax.set_title("Evolution at B = {}".format(B))
    ax.set_xlabel("time")
    ax.set_ylabel("velocity magnitude")
    plt.show()

f_axis = []
strouhal_axis = []
for B, evolution in zip(B_axis, evolution_axis):
    if B == 0.05:
        f = analyze_frequency(evolution, time_slice=slice(40000, None))
    else:
        f = analyze_frequency(evolution, time_slice=slice(25000, None))
    f_axis.append(f)
    L = int(round(180 * B))
    u = 0.1
    strouhal = f * L / u
    strouhal_axis.append(strouhal)

fig, ax = plt.subplots()
ax.plot(B_axis, strouhal_axis, "o-")
ax.set_title("St-B relation")
ax.set_xlabel("Blockage ratio")
ax.set_ylabel("Strouhal number")
fig.savefig("ms6_job3_relation.png")

fig, ax = plt.subplots()
fig.set_size_inches(7,4.8)
ax.plot(B_axis, f_axis, "o-")
ax.set_title("f-B relation")
ax.set_xlabel("Blockage ratio")
ax.set_ylabel("Oscillatory frequency")
fig.savefig("ms6_job3_extra_relation.png")


# %%
num_core = np.linspace(0, 40, num=41)
evolution_scale = np.array([np.inf,
    1595.7335580093786, 780.9275345443748, 324.3205841335778, 251.34814920113422,
    209.29658202584832, 183.63359038221338, 154.19885186811112, 142.87747383886017,
    130.34697859703252, 122.58874471792952, 113.45377862444994, 110.51394767430611,
    104.73191700617855, 101.42920218634286, 97.83206058784077, 97.54155630397145,
    89.89228224485893, 90.69682091712538, 87.19001546399177, 81.26731539755129,
    78.25869757761913, 75.32991076053374, 74.54551274066225, 74.33905981729428,
    74.91589717704802, 72.53645115306314, 70.86289127202083, 71.20009807058211,
    99.13334526754274, 71.0991087219057, 99.65489305129215, 70.42604681069497,
    66.64408853842002, 64.74177052801036, 65.31958073074264, 64.68137938653429,
    94.11572991253657, 63.9664621262468, 60.68734258597191, 61.4930693322327])
speed_up = evolution_scale[1] / evolution_scale
fig, ax = plt.subplots()
ax.plot(num_core, speed_up, "o-")
ax.set_title("Original scaling data")
ax.set_xlabel("Num of cores")
ax.set_ylabel("Speed up")
fig.savefig("ms7_scale_unfit")

outlier_index = [29, 31, 37]
p = np.delete(num_core, outlier_index)
s = np.delete(speed_up, outlier_index)
curve_function = lambda p, f_p: p / (f_p + (1-f_p)*p)
para, _ = optimize.curve_fit(curve_function, p, s)
f_p = np.round(*para, decimals=8)
s_fitted = curve_function(p, f_p)
fig, ax = plt.subplots()
ax.plot(num_core, speed_up, "o", label="original")
ax.plot(p, s_fitted, "-", label="fitted f_p={}".format(f_p))
ax.legend(loc="best")
ax.set_title("Fitted scaling data")
ax.set_xlabel("Num of cores")
ax.set_ylabel("Speed up")
fig.savefig("ms7_scale_fit")

for core in num_core:
    print("S({} cores) = {}".format(core, curve_function(core, f_p)))

# %%
job_scale = np.array([
    [10, 2483.545240123663, 2870.1459991855545, 2576.3996335709467],
    [15, 2033.999211067582, 2320.2585236075024, 2032.5753148471936],
    [20, 1616.0246238368563, 1929.7891496453435, 1618.2222148197238],
    [25, 1614.3967219237238, 1877.4600554962083, 1566.5037669100984],
    [30, 1452.2048919669974, 1650.367087360844, 1466.0402827610262],
    [35, 1421.047909810953, 1558.5834222358785, 1429.8266937026488],
    [40, 1245.3991287395124, 1435.7739326781825, 1304.095231380849]])

time_data = job_scale[:,1:].copy()
print(time_data / time_data.min())
# %%
