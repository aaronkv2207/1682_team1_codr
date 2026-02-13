# Contour Plots
import matplotlib.pyplot as plt
import numpy as np
import inspect
import re
# import aerosandbox.tools.pretty_plots

def make_contour(f1, f2, f3, var1, var2, xlabel="Variable 1", ylabel="Variable 2"):
    '''
    Docstring for make_contour
    
    :param f1: function that takes in two variables and outputs an objective
    :param f2: function that takes in the same two objectives and outputes a second objective
    :param f3: function that takes in the same two objectives and outputs a third objective
    :param var1: nominal value for input 1
    :param var2: nominal value for input 2
    '''
    percent_range = 0.3
    N = 100

    var1_vals = np.linspace(var1*(1 - percent_range), var1 * (1 + percent_range), N)
    var2_vals = np.linspace(var2*(1 - percent_range), var2 * (1 + percent_range), N)

    Z1 = np.zeros([N, N])
    Z2 = np.zeros([N, N])
    Z3 = np.zeros([N, N])
    
    for i in range(N):
        for j in range(N):
            Z1[j, i] = f1(var1_vals[i], var2_vals[j])
            Z2[j, i] = f2(var1_vals[i], var2_vals[j])
            Z3[j, i] = f3(var1_vals[i], var2_vals[j])

    plt.subplot(1, 3, 1)

    plt.contourf(var1_vals, var2_vals, Z1, label="Objective 1", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("X_to")
    plt.legend()
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.contourf(var1_vals, var2_vals, Z2, label="Objective 2", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("V_cruise")
    plt.legend()
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.contourf(var1_vals, var2_vals, Z1-Z2, label="Objective 3", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Range")
    plt.legend()
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    C_L_to = 8
    T_c = 3.5
    rho_ground = 1.225
    V_takeoff = 17 # m/s
    g = 9.81
    weight = 5440 * g
    W_final = 4440 * g
    x_to = 45 # m


    rho_cruise = 0.909
    C_L_cruise = 0.2
    n_p = 0.8
    c_p = 1*10**(-7)
    c_d = 0.035

    def get_S(weight, x):
        return np.sqrt((weight**2/x_to)*2/(T_c*rho_ground*V_takeoff**2)*1/C_L_to*1/(rho_ground*g))
    
    def get_xto(weight, S):
        return (weight / S) / (T_c * rho_ground * g * C_L_to)
    
    def get_range(weight, S):
        return (1/g)*(n_p/c_p)*(C_L_cruise/c_d)*np.log(weight/W_final)

    S = get_S(weight, 1)

    T = T_c*(0.5*rho_ground*(V_takeoff**2)*S)

    P = (T*V_takeoff)/0.6

    def get_V_cruise(weight, S):
        return np.sqrt((2 * weight) / (rho_cruise * S * C_L_cruise))

    #Design variables:
    S_nominal = 32.5 #from above
    # WS = weight / S_nominal
    weight_nominal = weight

    make_contour(get_xto, get_V_cruise, get_range, weight_nominal, S_nominal, "Weight", "Wing Area")