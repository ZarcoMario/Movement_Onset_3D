import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from findiff import FinDiff

from minimum_jerk_3d import minimum_jerk_3d
from motion_onset_detection_3d import onset_detection

# np.random.seed(123456789)

step = 1/90
duration = np.random.uniform(1, 1.5)  # Random duration of the movement phase
t_mp = np.arange(0, 1 + step, step)  # Time of the movement phase

# Let's test the method using synthetic data without noise
# If interedted, use the Smooth Noise-Robust Differentiator in derivative.py
# to test the synthetic data with noise as FinDiff does not handle it properly

x_init = np.array([0, 0, 0])
y_init = np.array([0, 0, 0])
z_init = np.array([0, 0, 0])

# Assuming that final velocities are not zero to create a curved trajectory

x_fin = np.array([1, -0.5, 0])
y_fin = np.array([1, 2, 0])
z_fin = np.array([1, -0.25, 0])

# Movement phase
x_tr, y_tr, z_tr = minimum_jerk_3d(x_init, y_init, z_init, x_fin, y_fin, z_fin, duration, t_mp)

# Add Static phase
to_real = np.random.uniform(0.3, 0.9)  # Random movement onset time
t_tr = np.arange(0, 1 + to_real + step, step)
x_tr = np.append(np.full(t_tr.size - x_tr.size, x_init[0]), x_tr)
y_tr = np.append(np.full(t_tr.size - y_tr.size, y_init[0]), y_tr)
z_tr = np.append(np.full(t_tr.size - z_tr.size, z_init[0]), z_tr)

# print(t_tr.size, x_tr.size)

# Findiff can be used to calculate velocities because synthetic data is not noisy
d_dt = FinDiff(0, step, 1, acc=6)
vx = d_dt(x_tr)
vy = d_dt(y_tr)
vz = d_dt(z_tr)

# Estimate onset time
delta_T = 0.1  # 100 ms.
Ts = step
m = int(delta_T / Ts) - 1
# print("m:", m)
tm = m * Ts
# print(tm)

res = onset_detection(m, x_tr, y_tr, z_tr, t_tr, vx, vy, vz, debug=False)
to = res[0]
print("Real Time Onset", to_real)
print("Estimated Time Onset", to)


fig = plt.figure(figsize=(16, 8))
gs = GridSpec(1, 3)
ax = fig.add_subplot(gs[0, 0], projection='3d')
ax.plot(x_tr, y_tr, z_tr, '.')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax = fig.add_subplot(gs[0, 1])
ax.grid(True)
ax.plot(t_tr, x_tr, '.', label='x')
ax.plot(t_tr, y_tr, '.', label='y')
ax.plot(t_tr, z_tr, '.', label='z')
ax.axvline(to_real, ls='-', color='g', label='to_real')
ax.axvline(to, ls='--', color='r', label='to_estimated')
ax.legend()
ax.set_xlabel("t")

ax = fig.add_subplot(gs[0, 2])
ax.grid(True)
ax.plot(t_tr, vx, '.', label='vx')
ax.plot(t_tr, vy, '.', label='vy')
ax.plot(t_tr, vz, '.', label='vz')
ax.axvline(to_real, ls='-', color='g', label='to_real')
ax.axvline(to, ls='--', color='r', label='to_estimated')
ax.legend()
ax.set_xlabel("t")

plt.tight_layout()
plt.show()