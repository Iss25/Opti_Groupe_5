import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
import time

temperatures_montreal = np.load("Temperatures-Montreal.npy")
eta = 0.99
max_pump_power = 1 # kW
Cp = 0.4 # °C/1kWh
T_min = 19 # °C
T_max = 21 # °C
inconfort_penality_supp = 1
inconfort_penality_inf = 3
ref_week_start_idx = 13050
computing_intervals_amount = 7*24*4

electricity_cost = [0.18 if (i % (24*4)) /4 >= 22 or (i % (24*4))/4 < 7 else 0.26 for i in range(len(temperatures_montreal))]

def COP_warming(T_ext):
    return 3 + 10 * abs(np.tanh(T_ext/100)) * np.tanh(T_ext/100)

def COP_reverse():
    return 3.2

def next_temperature(T_old, T_ext):
    return - (1-eta) * (T_old - T_ext) + T_old


def min_electricity_cost(first_interval_idx):
    last_interval_idx = first_interval_idx + computing_intervals_amount                                 # 7 days 
    temperatures_ext = temperatures_montreal[first_interval_idx:last_interval_idx]
    mid_temperature = (T_max + T_min)//2 

    p_warming = cp.Variable(computing_intervals_amount, nonneg=True) # Puissance de la pompe à l'intervalle i en réchauffement
    p_reverse = cp.Variable(computing_intervals_amount, nonneg=True) # Puissance de la pompe à l'intervalle i en reverse
    temperatures_int = cp.Variable(computing_intervals_amount) # Températures intérieures

    partial_electricity_cost = electricity_cost[first_interval_idx:last_interval_idx]

    objective = cp.Minimize(cp.sum(partial_electricity_cost @ (p_warming + p_reverse) * 4 ))

    constraints = [T_min <= temperatures_int]
    constraints += [T_max >= temperatures_int]
    constraints += [p_warming >= 0]
    constraints += [p_reverse >= 0]
    constraints += [p_warming <= max_pump_power]
    constraints += [p_reverse <= max_pump_power]
    constraints += [temperatures_int[0] == mid_temperature]
    constraints += [temperatures_int[-1] == mid_temperature]

    for i in range(computing_intervals_amount - 1):
        constraints += [
            temperatures_int[i+1] == next_temperature(temperatures_int[i], temperatures_ext[i])              # isolation loss
                                    + (COP_warming(temperatures_ext[i]) * p_warming[i] * 15 * Cp)            # warming mode
                                    - (COP_reverse() * p_reverse[i] * 15 * Cp)                               # reverse mode
                                    ] 

    problem = cp.Problem(objective, constraints)
    start_time = time.time()
    solution = problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    end_time = time.time()

    print("1:", "\n","Puissances normales = ", p_warming.value, "\n", "Puissances reverses = ", p_reverse.value,"\n", "Températures internes = ", temperatures_int.value, "\n",
      "Cout = ", problem.value,"\n", "Temps de résolution = ", end_time  - start_time)


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    x = np.linspace(0, computing_intervals_amount, computing_intervals_amount)

    axs[0][0].plot(x,temperatures_int.value)
    axs[0][0].set_title("Période 1 - Évolution des températures")
    axs[0][0].set_xlabel("Intervalle de temps")
    axs[0][0].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur

    x = np.linspace(0, computing_intervals_amount, computing_intervals_amount)

    axs[1][0].plot(x, p_warming.value, label="Fonctionnement normal")
    axs[1][0].plot(x, p_reverse.value, label="Fonctionnement reverse")
    axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
    axs[1][0].set_xlabel("Intervalle de temps")
    axs[1][0].set_ylabel("Puissance (kW)")
    axs[1][0].legend()


def task2(first_interval_idx, max_cost):
    last_interval_idx = first_interval_idx + computing_intervals_amount                                 # 7 days 
    temperatures_ext = temperatures_montreal[first_interval_idx:last_interval_idx]
    mid_temperature = (T_max + T_min)//2 

    p_warming = cp.Variable(computing_intervals_amount, nonneg=True) # Puissance de la pompe à l'intervalle i en réchauffement
    p_reverse = cp.Variable(computing_intervals_amount, nonneg=True) # Puissance de la pompe à l'intervalle i en reverse
    temperatures_int = cp.Variable(computing_intervals_amount) # Températures intérieures
    partial_electricity_cost = electricity_cost[first_interval_idx:last_interval_idx]
    inconforts_sup = cp.Variable(computing_intervals_amount, nonneg=True)
    inconforts_inf = cp.Variable(computing_intervals_amount, nonneg=True)
    
    objective = cp.sum(inconforts_sup*inconfort_penality_supp + inconfort_penality_inf*inconforts_inf)

    cost = cp.sum(partial_electricity_cost @ (p_warming + p_reverse) * 4)

    constraints = [p_warming >= 0]
    constraints += [p_reverse >= 0]
    constraints += [p_warming <= max_pump_power]
    constraints += [p_reverse <= max_pump_power]
    constraints += [temperatures_int[0] == mid_temperature]
    constraints += [temperatures_int[-1] == mid_temperature]
    constraints += [cost <= max_cost]


    for i in range(computing_intervals_amount - 1):
        constraints += [
            temperatures_int[i+1] == next_temperature(temperatures_int[i], temperatures_ext[i])               # isolation loss
                                    + (COP_warming(temperatures_ext[i]) * p_warming[i] * 15 * Cp)            # warming mode
                                    - (COP_reverse() * p_reverse[i] * 15 * Cp)                               # reverse mode
                                    ] 
        constraints += [temperatures_int[i] - T_min >= -inconforts_inf[i]]
        constraints += [temperatures_int[i] - T_max <= inconforts_sup[i]]
        
    constraints += [temperatures_ext[-1] - T_min >= -inconforts_inf[-1]]
    constraints += [temperatures_int[-1] - T_max <= inconforts_sup[-1]]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.time()
    solution = problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    end_time = time.time()

    print("2:", "\n","Puissances normales = ", p_warming.value, "\n", "Puissances reverses = ", p_reverse.value,"\n", "Températures internes = ", temperatures_int.value, "\n",
     "Inconforts supp = ", inconforts_sup.value, "\nInconforts inf = ", inconforts_inf.value, 
     "\nCout = ", cost.value,"\n","Inconfort total = ", problem.value ,"\n", "Temps de résolution = ", end_time-start_time)


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    x = np.linspace(0, computing_intervals_amount, computing_intervals_amount)

    axs[0][0].plot(x,temperatures_int.value)
    axs[0][0].set_title("Période 1 - Évolution des températures")
    axs[0][0].set_xlabel("Intervalle de temps")
    axs[0][0].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur

    x = np.linspace(0, computing_intervals_amount, computing_intervals_amount)

    axs[1][0].plot(x, p_warming.value, label="Fonctionnement normal")
    axs[1][0].plot(x, p_reverse.value, label="Fonctionnement reverse")
    axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
    axs[1][0].set_xlabel("Intervalle de temps")
    axs[1][0].set_ylabel("Puissance (kW)")
    axs[1][0].legend()
    

#min_electricity_cost(ref_week_start_idx)
task2(ref_week_start_idx, max_cost=2)
plt.show()