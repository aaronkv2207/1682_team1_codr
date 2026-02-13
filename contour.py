# Contour Plots
import matplotlib.pyplot as plt
import numpy as np
# import aerosandbox.tools.pretty_plots

def make_contour(f1, f2, var1, var2):
    '''
    Docstring for make_contour
    
    :param f1: function that takes in two variables and outputs an objective
    :param f2: function that takes in the same two objectives and outputes a second objective
    :param var1: nominal value for input 1
    :param var2: nominal value for input 2
    '''
    percent_range = 0.3
    N = 20

    var1_vals = np.linspace(var1*(1 - percent_range), var1 * (1 + percent_range), N)
    var2_vals = np.linspace(var2*(1 - percent_range), var2 * (1 + percent_range), N)

    Z1 = np.zeros([N, N])
    Z2 = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            Z1[j, i] = f1(var1_vals[i], var2_vals[j])
            Z2[j, i] = f2(var1_vals[i], var2_vals[j])
    plt.contour(var1_vals, var2_vals, Z1, label="Objective 1", alpha=0.5)
    plt.contourf(var1_vals, var2_vals, Z2, label="Objective 2", alpha=0.5)
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    C_L_to = 8
    T_c = 3.5
    rho_ground = 1.225
    V_takeoff = 17 # m/s
    g = 9.81
    weight = 5440 * g
    x_to = 45 # m


    rho_cruise = 0.909
    C_L_cruise = 0.2

    def get_S(weight, x):
        return np.sqrt((weight**2/x_to)*2/(T_c*rho_ground*V_takeoff**2)*1/C_L_to*1/(rho_ground*g))

    S = get_S(weight, 1)

    T = T_c*(0.5*rho_ground*(V_takeoff**2)*S)

    P = (T*V_takeoff)/0.6
    def get_V_cruise(weight, S):
        return np.sqrt((2 * weight) / (rho_cruise * S * C_L_cruise))

    #Design variables:
    S_nominal = 32.5 #from above
    # WS = weight / S_nominal
    weight_nominal = weight

    make_contour(get_S, get_V_cruise, weight_nominal, S_nominal)