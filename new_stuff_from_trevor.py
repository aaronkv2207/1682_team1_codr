# new stuff
import numpy as np

C_L_to = 8
T_c = 3.5
rho_ground = 1.225
V_takeoff = 17 # m/s
g = 9.81
weight = 5440 * g
x_to = 45 # m

rho_cruise = 0.909
C_L_cruise = 0.25

S = np.sqrt((weight**2/x_to)*2/(T_c*rho_ground*V_takeoff**2)*1/C_L_to*1/(rho_ground*g))
print("S:", S)
T = T_c*(0.5*rho_ground*(V_takeoff**2)*S)
print("Thrust:", T)
P = (T*V_takeoff)/0.6
print("Power:", P)

V_cruise = np.sqrt((2 * weight) / (rho_cruise * S * C_L_cruise))
print("V_cruise:", V_cruise)

# Things to do
# Drag: induced drag + Cd_0
# Range
    # Consider efficiency (lower than normal plane!)
# Thrust for climb to make sure it's reasonable
# Static margin
# Maybe improve takeoff model to account for acceleration
# Climb: make sure ROC makes sense
