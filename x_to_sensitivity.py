import math
import numpy as np
import matplotlib.pyplot as plt
# constants
rho_cruise = 0.9 # 10,000 ft
rho_non_cruise = 0 # fix
rho_ground = 1.225
g = 9.81
rho_climb = 0 # fix

# kinematics
v_cruise = 125
v_non_cruise = 0 # fix
v_climb = 0 # fix
x_to = 150

# geometry
radius_prop = 0.5
K_duct = 1.25
R_eff = K_duct * radius_prop
N_prop = 8
# S = 26 # m^2
b = 16
c = 1.6

# engine stuff
eta_v = 0.95
eta_prop = 0.75
turbogen_power = 600 # kW

# aerodynamics
lift_to_drag = 12
form_drag_coef = 0 # CHANGE
C_L_to = 10

# weight breakdown
W_fixed = 1360 * g # N
W_airframe = 2176 * g
W_power = 1904 * g
W_total = 5440 * g
mass = 5440
wing_loading = 140 # kg/m^2
S = mass / wing_loading
AR = (b**2) / S

def x_takeoff(C_L_to, thrust, AR, W):
    # use all other values to calculate x_to and compare
    # to "ideal"
    # thrust = thrust(gamma, C_L_to, AR)
    return (W / S) / (thrust * rho_cruise * g * C_L_to)

percent_range = 0.3
N = 20

C_L_to_nominal = 7
AR_nominal = 10
to_thrust_nominal = 6000
W_nominal = W_total

to_thrust = np.linspace(to_thrust_nominal * (1 - percent_range), to_thrust_nominal * (1 + percent_range), N)
C_L_to = np.linspace(C_L_to_nominal * (1 - percent_range), C_L_to_nominal * (1 + percent_range), N)
W_to = np.linspace(W_nominal * (1 - percent_range), W_nominal * (1 + percent_range), N)
# AR = np.linspace(AR * (1 - percent_range), AR *(1 + percent_range), N)

# --- Sensitivity calculations ---
CL_varied_xto = x_takeoff(C_L_to, to_thrust_nominal, AR_nominal, W_nominal)
CL_varied_xto = 100 * CL_varied_xto / np.mean(CL_varied_xto)

W_varied_xto = x_takeoff(C_L_to_nominal, to_thrust_nominal, AR_nominal, W_to)
W_varied_xto = 100 * W_varied_xto / np.mean(W_varied_xto)
# AR_varied_xto = x_takeoff(C_L_to_nominal, to_thrust_nominal, AR)
thrust_varied_xto = x_takeoff(C_L_to_nominal, to_thrust, AR_nominal, W_nominal)
thrust_varied_xto = 100 * thrust_varied_xto / np.mean(thrust_varied_xto)

percent = np.linspace(100 * (1 - percent_range), 100 * (1 + percent_range), N)

plt.figure(figsize=(8, 6))
plt.plot(percent, CL_varied_xto, '-', label='CL variation', linewidth=1.5, marker='^')
plt.plot(percent, W_varied_xto, '-.', label='W variation', linewidth=1.5, marker='^')
plt.plot(percent, thrust_varied_xto, '-.', label='Thrust variation', linewidth=1.5, marker='v')

plt.xlabel('% Change in design parameter')
plt.ylabel('% Change in x_to')
plt.title('takeoff distance sensitivity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.axis('square')
plt.savefig('./sensitivity.png', dpi=300)
plt.show()
