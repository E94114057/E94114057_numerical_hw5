import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt  

def f(t, y):
    return 1 + (y / t) + (y / t) ** 2

def f_t(t, y):
    return (-y / t**2) + (-2 * y**2 / t**3)

def f_y(t, y):
    return (1 / t) + (2 * y / t**2)

def y_exact(t):
    return t * math.tan(math.log(t))

pd.set_option('display.float_format', '{:>12.6f}'.format)
pd.set_option('display.expand_frame_repr', False)

t0 = 1.0
y0 = 0.0
h = 0.1
n_steps = int((2.0 - 1.0) / h)

# 1.a Euler method

t_values = [t0]
y_euler = [y0]
y_real = [y_exact(t0)]

t = t0
y = y0

for _ in range(n_steps):
    t_new = t + h
    y_new = y + h * f(t, y)

    t_values.append(t_new)
    y_euler.append(y_new)
    y_real.append(y_exact(t_new))

    t = t_new
    y = y_new

df_euler = pd.DataFrame({
    't': t_values,
    'Euler y': y_euler,
    'Exact y': y_real,
    'Error': np.abs(np.array(y_real) - np.array(y_euler))
})

print("\n(1.a) Euler Method Results:\n")
print(df_euler.to_string(index=False))

# 1.b Taylor method of order 2

t_values_taylor = [t0]
y_taylor2 = [y0]
y_real_taylor = [y_exact(t0)]

t = t0
y = y0

for _ in range(n_steps):
    t_new = t + h
    fy = f(t, y)
    ft = f_t(t, y)
    fy_prime = f_y(t, y)
    y_new = y + h * fy + (h**2 / 2) * (ft + fy_prime * fy)

    t_values_taylor.append(t_new)
    y_taylor2.append(y_new)
    y_real_taylor.append(y_exact(t_new))

    t = t_new
    y = y_new

df_taylor2 = pd.DataFrame({
    't': t_values_taylor,
    'Taylor2 y': y_taylor2,
    'Exact y': y_real_taylor,
    'Error': np.abs(np.array(y_real_taylor) - np.array(y_taylor2))
})

print("\n(1.b) Taylor Method Results:\n")
print(df_taylor2.to_string(index=False))

plt.figure(figsize=(8,5))
plt.plot(t_values, y_euler, marker='o', linestyle='-', label='Euler Approximation')
plt.plot(t_values, y_real, marker='x', linestyle='--', label='Exact Solution')
plt.title('(1.a) Euler Method vs Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(8,5))
plt.plot(t_values_taylor, y_taylor2, marker='s', linestyle='-', color='orange', label='Taylor2 Approximation')
plt.plot(t_values_taylor, y_real_taylor, marker='x', linestyle='--', color='green', label='Exact Solution')
plt.title('(1.b) Taylor2 Method vs Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

