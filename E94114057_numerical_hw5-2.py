import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def f(t, u1, u2):
    du1_dt = 9 * u1 + 24 * u2 + 5 * np.cos(t) - (1/3) * np.sin(t)
    du2_dt = -24 * u1 - 52 * u2 - 9 * np.cos(t) + (1/3) * np.sin(t)
    return np.array([du1_dt, du2_dt])

def exact_u1(t):
    return 2 * np.exp(-3*t) - np.exp(-39*t) + (1/3) * np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2 * np.exp(-39*t) - (1/3) * np.cos(t)

def rk4_system(h, T):
    n_steps = int(T / h)
    t_values = [0]
    u1_values = [4/3]
    u2_values = [2/3]
    u1_exact = [exact_u1(0)]
    u2_exact = [exact_u2(0)]

    t = 0
    u = np.array([4/3, 2/3])  

    for _ in range(n_steps):
        k1 = f(t, u[0], u[1])
        k2 = f(t + h/2, u[0] + h/2 * k1[0], u[1] + h/2 * k1[1])
        k3 = f(t + h/2, u[0] + h/2 * k2[0], u[1] + h/2 * k2[1])
        k4 = f(t + h, u[0] + h * k3[0], u[1] + h * k3[1])

        u = u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_values.append(t)
        u1_values.append(u[0])
        u2_values.append(u[1])
        u1_exact.append(exact_u1(t))
        u2_exact.append(exact_u2(t))

    df = pd.DataFrame({
        't': t_values,
        'u1 (RK4)': u1_values,
        'u1 (Exact)': u1_exact,
        'Error u1': np.abs(np.array(u1_values) - np.array(u1_exact)),
        'u2 (RK4)': u2_values,
        'u2 (Exact)': u2_exact,
        'Error u2': np.abs(np.array(u2_values) - np.array(u2_exact)),
    })
    return df

def plot_results(df, h_value):
    plt.figure(figsize=(12, 5))
    
    # u1
    plt.subplot(1, 2, 1)
    plt.plot(df['t'], df['u1 (RK4)'], 'bo-', label='u1 RK4', markersize=4)
    plt.plot(df['t'], df['u1 (Exact)'], 'r--', label='u1 Exact')
    plt.xlabel('t')
    plt.ylabel('u1')
    plt.title(f'u1: RK4 vs Exact (h={h_value})')
    plt.legend()
    plt.grid(True)

    # u2
    plt.subplot(1, 2, 2)
    plt.plot(df['t'], df['u2 (RK4)'], 'go-', label='u2 RK4', markersize=4)
    plt.plot(df['t'], df['u2 (Exact)'], 'r--', label='u2 Exact')
    plt.xlabel('t')
    plt.ylabel('u2')
    plt.title(f'u2: RK4 vs Exact (h={h_value})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

pd.set_option('display.float_format', '{:>12.6f}'.format)
pd.set_option('display.expand_frame_repr', False)

# h = 0.1 
print("\nRunge-Kutta with h = 0.1\n")
df_h01 = rk4_system(0.1, 1.0)
print(df_h01.to_string(index=False))
plot_results(df_h01, 0.1)

# h = 0.05 
print("\nunge-Kutta with h = 0.05\n")
df_h005 = rk4_system(0.05, 1.0)
print(df_h005.to_string(index=False))
plot_results(df_h005, 0.05)
