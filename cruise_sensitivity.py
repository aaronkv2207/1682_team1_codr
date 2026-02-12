import math
import numpy as np
import matplotlib.pyplot as plt

# things to figure out:
# how to include sizing as an output? or input?
# how does tail sizing come into play
# do we need to consider takeoff constraints or benefits and involve that in cruise calcs
# add in range function when we have it

# constants
rho_cruise = 0.909 # 10,000 ft
rho_non_cruise = 1.225 # setting as rho_ground for now - we can make more specific later
rho_ground = 1.225
g = 9.81
rho_climb = 1.225 # update

# kinematics
v_cruise = 125
v_non_cruise = 0 # fix
v_climb = 0 # fix
x_to = 45

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
# eta_prop = ???
turbogen_power = 600 # kW

# aerodynamics
lift_to_drag = 12
form_drag_coef = 0 # CHANGE
C_L_to = 8
C_L_cruise = 0.19

# weight breakdown
W_fixed = 1360 * g # N
W_airframe = 2176 * g
W_power = 1904 * g
mass = 5440
W_total = mass * g
wing_loading = 135 # kg/m^2
S = mass / wing_loading
# print(S)
AR = (b**2) / S

# questions:
# ask about all estimations + specs see if they're normal
# ask about C_L, weight stuff, is weight normal (show breakdown)?
# ask about electric vs. fuel for (breguet) range

def get_C_L_cruise():
    return W_total / (0.5 * rho_cruise * (v_cruise**2) * S)

def get_v_cruise():
    return math.sqrt((2 * W_total) / (rho_cruise * S * C_L_cruise))

def takeoff_velocity():
    return math.sqrt(W_total / (0.5 * C_L_to * rho_ground * S))

def power(state, T=0): # where T is thrust
    if state == "cruise":
        C_L = C_L_cruise()
        if T == 0: # don't know the thrust
            T = thrust_cruise(C_L, AR)
        radical = 1 + T / (0.5 * rho_cruise * (v_cruise**2) * math.pi * (R_eff**2) * N_prop) # should it be R_eff? Do we multiply by 8 props?
        power = ((T * v_cruise) / eta_v) * (2 + math.sqrt(radical) - 1) / 2
        return power
    elif state == "climb":
        if T == 0:
            T = thrust_climb()
        radical = 1 + T / (0.5 * rho_climb * (v_climb**2) * math.pi * (R_eff**2) * N_prop) # should it be R_eff? Do we multiply by 8 props?
        power = ((T * v_climb) / eta_v) * (2 + math.sqrt(radical) - 1) / 2
        return power
    # else, takeoff
    if T == 0:
        T = thrust_takeoff()
    v_takeoff = takeoff_velocity()
    radical = 1 + T / (0.5 * rho_ground * (v_takeoff**2) * math.pi * (R_eff**2) * N_prop) # should it be R_eff? Do we multiply by 8 props?
    power = ((T * v_takeoff) / eta_v) * (2 + math.sqrt(radical) - 1) / 2
    return power

def thrust_climb(gamma, acceleration, C_L, AR):
    C_D = drag_coeff(C_L, AR)
    term1 = 0.5 * rho_non_cruise * (v_cruise**2) * S * C_D
    term2 = W_total * math.sin(gamma)
    term3 = mass * acceleration
    return term1 + term2 + term3

def thrust_takeoff():
    takeoff_vel = takeoff_velocity()
    return 0.5 * mass * (takeoff_vel**2) / x_to

def thrust_cruise(C_L, AR):
    C_D = drag_coeff(C_L, AR)
    return 0.5 * rho_cruise * (v_cruise**2) * S * C_D

def induced_drag_coeff(C_L, AR, e=1):
    return (C_L**2) / (math.pi * AR * e)

def drag_coeff(C_L, AR):
    return form_drag_coef + induced_drag_coeff(C_L, AR)

def get_x_takeoff(C_L_to):
    thrust = 16500
    x_takeoff = (W_total / S) / (thrust/W_total) * 1/ (rho_ground * g) * 1/C_L_to
    return x_takeoff
    # return (W_total / S) / (thrust/W_total * rho_cruise * g * C_L_to)


# def plot_velocity(var_name, min, max, n_points):
#     """
#     Sweep a variable (SI units) and plot cruise velocity vs that variable.

#     Special cases:
#     - If var_name is "mass" (kg) or "wing_loading" (kg/m^2), S is recomputed as:
#           S = mass / wing_loading   [m^2]
#     """

#     # Variables used by get_v_cruise (and their SI units)
#     used_vars_units = {
#         "W_total": "N",
#         "rho_cruise": "kg/m^3",
#         "S": "m^2",
#         "C_L_cruise": "-"
#     }

#     # Check variable exists (or is special case)
#     if var_name not in globals() and var_name not in ["mass", "wing_loading"]:
#         raise ValueError(f"Variable '{var_name}' not found in globals().")

#     # Save originals
#     originals = {
#         "S": globals().get("S", None),
#         "mass": globals().get("mass", None),
#         "wing_loading": globals().get("wing_loading", None),
#         var_name: globals().get(var_name, None)
#     }

#     sweep_vals = np.linspace(min, max, n_points)
#     velocities = []

#     for val in sweep_vals:
#         if var_name == "mass":  # kg
#             globals()["mass"] = val
#             globals()["S"] = globals()["mass"] / globals()["wing_loading"]  # m^2
#         elif var_name == "wing_loading":  # kg/m^2
#             globals()["wing_loading"] = val
#             globals()["S"] = globals()["mass"] / globals()["wing_loading"]  # m^2
#         else:
#             globals()[var_name] = val

#         velocities.append(get_v_cruise())  # m/s

#     # Restore originals
#     for k, v in originals.items():
#         if k in globals() and v is not None:
#             globals()[k] = v

#     # Build info text (baseline reference)
#     info_lines = []
#     for v, unit in used_vars_units.items():
#         info_lines.append(f"{v} = {globals()[v]:.5g} [{unit}]")
#     info_text = "\n".join(info_lines)

#     # Axis labels with units
#     x_unit_map = {
#         "mass": "kg",
#         "wing_loading": "kg/m^2",
#         "S": "m^2",
#         "W_total": "N",
#         "rho_cruise": "kg/m^3",
#         "C_L_cruise": "-"
#     }
#     x_unit = x_unit_map.get(var_name, "")

#     plt.figure()
#     plt.plot(sweep_vals, velocities)
#     plt.xlabel(f"{var_name} [{x_unit}]")
#     plt.ylabel("Cruise velocity [m/s]")
#     plt.title(f"Cruise velocity vs {var_name}")
#     plt.grid(True)

#     # Always add the baseline reference text box
#     plt.gca().text(
#         0.02, 0.98, info_text,
#         transform=plt.gca().transAxes,
#         fontsize=10,
#         verticalalignment="top",
#         horizontalalignment="left",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
#     )

#     plt.show()


def compare_sweeps(ranges, n_points=50):
    """
    Sweep multiple variables and compare their effect on cruise velocity.

    ranges: dict like
        {
          "W_total": (0.8*W_total, 1.2*W_total),
          "rho_cruise": (0.6, 1.2),
          "S": (0.5*S, 1.5*S),
          "C_L_cruise": (0.1, 0.6),
          "wing_loading": (80, 200),   # kg/m^2  -> updates S only
          "mass": (4000, 7000),        # kg      -> updates S and W_total
        }
    """

    # Compute baseline cruise velocity
    V0 = get_v_cruise()

    # Save originals for restoration AND baseline reference
    orig = {
        "S": S,
        "mass": mass,
        "wing_loading": wing_loading,
        "W_total": W_total,
        "rho_cruise": rho_cruise,
        "C_L_cruise": C_L_cruise
    }

    # Variables used by get_v_cruise (for text box)
    used_vars_units = {
        "W_total": "N",
        "rho_cruise": "kg/m^3",
        "S": "m^2",
        "C_L_cruise": "-"
    }

    # ---------- Plot 1: range-normalized x (0 -> 1) ----------
    plt.figure()

    for var_name, (min_val, max_val) in ranges.items():
        if var_name not in globals() and var_name not in ["mass", "wing_loading"]:
            raise ValueError(f"{var_name} not found.")

        # Baseline value for normalization
        if var_name in ["mass", "wing_loading"]:
            x0 = orig[var_name]
        else:
            x0 = globals()[var_name]

        sweep_vals = np.linspace(min_val, max_val, n_points)
        V_vals = []

        for val in sweep_vals:
            if var_name == "wing_loading":
                globals()["wing_loading"] = val
                globals()["S"] = globals()["mass"] / globals()["wing_loading"]
            elif var_name == "mass":
                globals()["mass"] = val
                globals()["S"] = globals()["mass"] / globals()["wing_loading"]
                globals()["W_total"] = globals()["mass"] * g
            else:
                globals()[var_name] = val

            V_vals.append(get_v_cruise())

        # Restore originals
        globals()["S"] = orig["S"]
        globals()["mass"] = orig["mass"]
        globals()["wing_loading"] = orig["wing_loading"]
        globals()["W_total"] = orig["W_total"]

        # Normalizations
        x_range_norm = (sweep_vals - sweep_vals.min()) / (sweep_vals.max() - sweep_vals.min())
        V_norm = np.array(V_vals)
        plt.plot(x_range_norm, V_norm, label=var_name)

    # ---------- Labels, grid, legend ----------
    plt.xlabel("Normalized sweep position (0 = min, 1 = max)")
    plt.ylabel("Cruise velocity / baseline velocity")
    plt.title("Range-normalized comparison (including mass and wing loading)")
    plt.grid(True)
    plt.legend(loc="upper right")  # move legend to top right

    # ---------- Baseline reference text box ----------
    info_lines = [f"Baseline cruise velocity = {V0:.3f} m/s"]  # add V0 at top
    for v, unit in used_vars_units.items():
        info_lines.append(f"{v} = {orig[v]:.5g} [{unit}]")  # use saved baseline
    info_text = "\n".join(info_lines)

    plt.gca().text(
        0.02, 0.98, info_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.show()

def plot_velocity(var_name, min, max, n_points):
    """
    Sweep a variable (SI units) and plot cruise velocity and takeoff distance vs that variable.

    Special cases:
    - If var_name is "mass" (kg) or "wing_loading" (kg/m^2), S is recomputed as:
          S = mass / wing_loading   [m^2]
    """

    # Variables used by get_v_cruise (and their SI units)
    used_vars_units = {
        "W_total": "N",
        "rho_cruise": "kg/m^3",
        "S": "m^2",
        "C_L_cruise": "-"
    }

    # Check variable exists (or is special case)
    if var_name not in globals() and var_name not in ["mass", "wing_loading"]:
        raise ValueError(f"Variable '{var_name}' not found in globals().")

    # Save originals
    originals = {
        "S": globals().get("S", None),
        "mass": globals().get("mass", None),
        "wing_loading": globals().get("wing_loading", None),
        var_name: globals().get(var_name, None)
    }

    sweep_vals = np.linspace(min, max, n_points)
    velocities = []
    takeoff_distances = []

    for val in sweep_vals:
        # Update the variable and derived S if needed
        if var_name == "mass":
            globals()["mass"] = val
            globals()["S"] = globals()["mass"] / globals()["wing_loading"]
            globals()["W_total"] = globals()["mass"] * g  # weight updates with mass
        elif var_name == "wing_loading":
            globals()["wing_loading"] = val
            globals()["S"] = globals()["mass"] / globals()["wing_loading"]
        else:
            globals()[var_name] = val

        velocities.append(get_v_cruise())

        # Calculate takeoff distance using your actual function
        takeoff_distances.append(get_x_takeoff(C_L_to))


    # Restore originals
    for k, v in originals.items():
        if k in globals() and v is not None:
            globals()[k] = v

    # Build info text
    info_lines = [f"Target cruise velocity = 125 m/s",
                  f"Target takeoff distance = 45 m"]
    for v, unit in used_vars_units.items():
        if v == var_name or (var_name in ["mass", "wing_loading"] and v == "S"):
            info_lines.append(f"{v} = (swept/derived) [{unit}]")
        else:
            info_lines.append(f"{v} = {globals()[v]:.5g} [{unit}]")
    info_text = "\n".join(info_lines)

    # -------------------- Plotting --------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cruise velocity plot
    ax1.plot(sweep_vals, velocities, color="blue")
    ax1.axhline(125, color="red", linestyle="--", label="Target 125 m/s")
    ax1.set_xlabel(f"{var_name}")
    ax1.set_ylabel("Cruise velocity [m/s]")
    ax1.set_title(f"Cruise velocity vs {var_name}")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Takeoff distance plot
    ax2.plot(sweep_vals, takeoff_distances, color="green")
    ax2.axhline(45, color="red", linestyle="--", label="Target 45 m")
    ax2.set_xlabel(f"{var_name}")
    ax2.set_ylabel("Takeoff distance [m]")
    ax2.set_title(f"Takeoff distance vs {var_name}")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    # Add info text box on the first plot
    ax1.text(
        0.02, 0.98, info_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.show()



compare_sweeps({
    "mass": (4000, 7000),
    "wing_loading": (80, 200),
    "S": (20, 60),
    "rho_cruise": (0.909, 1.225),
    "C_L_cruise": (0.1, 0.4),
})

plot_velocity("wing_loading", 100, 200, 50)

print("C_L_cruise", get_C_L_cruise())
print("v_takeoff", takeoff_velocity())
print("takeoff thrust/weight", thrust_takeoff()/W_total)
print("takeoff thrust", thrust_takeoff())
print("Takeoff distance", get_x_takeoff(C_L_to))
print("V_cruise", get_v_cruise())
print("takeoff power", (thrust_takeoff()*takeoff_velocity())/0.50)
