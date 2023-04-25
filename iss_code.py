import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import math
import time

#init
dt = 15*60 
nb_intervalles = 672 #nombres d'intervalles de 15 minutes sur 7 jours
temperatures_ext = np.load("Temperatures-Montreal.npy") # donées des températures à Montreal

eta = 0.99 # coefficient relatif à l'isolation
capacite_calorifique = 2.5 #kWh pour chauffer de 1°c le batiment qui fait 360 m³

T_min = 19 #Température minimale du batimenta
T_max = 21 #Température maximale du batiment

cout_elec1 = np.full(len(temperatures_ext),0.26) #initalisation du cout de l'élec a 0.26 partout
for i in range(366):
    for j in range(96):
        if j < 28 or j>= 88: #réajustement du cout de l'élec a 0.18 entre 22h et 7h
            cout_elec1[j+i*96] = 0.18 
            
COPT_reverse = 3.2 #COPT de la pompe quand on refroidi
def COP_normal(T_ext):
    return 3 + 10 * abs(np.tanh(T_ext/100)) * np.tanh(T_ext/100) #Fonction qui décrit le comportement du COP quand on réchauffe


def task1(strt):
    
    cout_elec = cout_elec1[strt:strt+672]
    ##Initialisation des variables :

    T_int = cp.Variable(nb_intervalles)

    #puissance qu'on va utiliser pour la pompe à chaleur
    P_chauff = cp.Variable(nb_intervalles, nonneg=True) #en mode normal
    P_refroid = cp.Variable(nb_intervalles, nonneg=True) #en mode inverse

    ## Initialisation du tableau de contraintes pour le probleme qui commence à 0:
    contraintes = []

    contraintes += [T_int[0] == 20] # Cf énoncé
    contraintes += [T_int[-1] == 20] # Cf énoncé
    contraintes += [T_int[0] >= 19]
    contraintes += [T_int[0] <= 21]

    #la température du batiment doit rester admissible: 
    contraintes += [T_min <= T_int[i] for i in range(nb_intervalles)]
    contraintes += [T_int[i] <= T_max for i in range(nb_intervalles)]

    for i in range(nb_intervalles - 1):
        contraintes += [T_int[i+1] - T_int[i] == - (1 - eta) * ( T_int[i]- temperatures_ext[i+strt]) + #perte de temp sans action
                            (COP_normal(temperatures_ext[i+strt]) * P_chauff[i] * dt / (60 *capacite_calorifique)) - #augmentation de la temp en mode normal
                            (COPT_reverse * P_refroid[i] * dt / (60*capacite_calorifique))] #diminution de la temp en mode reverse
        

    #Contrainte sur la positivité des puissances et max kW
    contraintes += [P_chauff >= 0]
    contraintes += [P_refroid >= 0]
    contraintes += [P_chauff <= 1]
    contraintes += [P_refroid <= 1]

    ## Initialisation du cout total :
    cost = cp.sum(cout_elec @ (P_chauff + P_refroid)*4) 

    start_time = time.time()

    ##Résolution 1 :
    problem = cp.Problem(cp.Minimize(cost), contraintes)
    pb = problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    temps_calcul = time.time() - start_time
    
    ##Récupération des valeurs :

    # print("1:", "\n","Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
    #     "Cout = ", problem1.value,"\n", "Temps de résolution = ", temps_calcul1)
    print("Cout =", problem.value,"\n","Temps de résolution =", temps_calcul)
    
    return T_int, P_chauff, P_refroid, problem.value

def task2(strt, budget):

    cout_elec = cout_elec1[strt:strt+672]

    penalite_inf = 3  # pénalité pour chaque degré en dessous de T_min
    penalite_sup = 1  # pénalité pour chaque degré au-dessus de T_max

    ##Initialisation des variables :

    T_int = cp.Variable(nb_intervalles)

    #puissance qu'on va utiliser pour la pompe à chaleur
    P_chauff = cp.Variable(nb_intervalles, nonneg=True) #en mode normal
    P_refroid = cp.Variable(nb_intervalles, nonneg=True) #en mode inverse


    inconfort_inf = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures inférieures à T_min
    inconfort_sup = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures supérieures à T_max

    ## Initialisation du tableau de contraintes pour le probleme qui commence à 0:
    contraintes = []

    contraintes += [T_int[0] == 20] # Cf énoncé
    contraintes += [T_int[-1] == 20] # Cf énoncé


    for i in range(nb_intervalles):
        contraintes.append(T_int[i] - T_min >= -inconfort_inf[i]) #l'inconfort correspondant à la différence avec la limite inferieure est stocké dans la variable inconfort_inf[i].
        contraintes.append(T_int[i] - T_max <= inconfort_sup[i]) #l'inconfort correspondant à la différence avec la limite supérieure est stocké dans la variable inconfort_inf[i].

    for i in range(nb_intervalles - 1):
        contraintes += [T_int[i+1] - T_int[i] == - (1 - eta) * ( T_int[i]- temperatures_ext[i+strt]) + #perte de temp sans action
                            (COP_normal(temperatures_ext[i+strt]) * P_chauff[i] * dt / (60 *capacite_calorifique)) - #augmentation de la temp en mode normal
                            (COPT_reverse * P_refroid[i] * dt / (60*capacite_calorifique))] #diminution de la temp en mode reverse
        
        
    #Contrainte sur la positivité des puissances
    contraintes += [P_chauff >= 0, P_refroid >= 0]
    contraintes += [P_chauff <=  1, P_refroid <= 1]

    cost = cp.sum(cout_elec @ (P_chauff + P_refroid)*4)

    #Contrainte sur le budget
    contraintes.append(cost <= budget)

    total_inconfort = cp.sum(penalite_inf * inconfort_inf + penalite_sup * inconfort_sup) 

    start_time = time.time()

    ##Résolution 1 :
    problem = cp.Problem(cp.Minimize(total_inconfort), contraintes)
    pb = problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    temps_calcul = time.time() - start_time

    ##Récupération des valeurs :

    # print("1:", "\n","Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
    #     "Inconfort = ", problem1.value,"\n", "Temps de résolution = ", temps_calcul1)
    print("Cout =",cost.value,"\n","Inconfort =", problem.value,"\n","Temps de résolution =", temps_calcul)

    return T_int, P_chauff, P_refroid, cost.value, problem.value

def plot_graph12(strt1,strt2):
    
    T_int1, P_chauff1, P_refroid1, Cout1 = task1(strt1)
    # Graphique de l'évolution des températures
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    x = np.linspace(strt1,strt1 + 672, 672)

    axs[0][0].plot(x,T_int1.value)
    axs[0][0].set_title("Période 1 - Évolution des températures")
    axs[0][0].set_xlabel("Intervalle de temps")
    axs[0][0].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur

    x = np.linspace(strt1,strt1 + 672, 672)

    axs[1][0].plot(x, P_chauff1.value, label="Fonctionnement normal")
    axs[1][0].plot(x, P_refroid1.value, label="Fonctionnement reverse")
    axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
    axs[1][0].set_xlabel("Intervalle de temps")
    axs[1][0].set_ylabel("Puissance (kW)")
    axs[1][0].legend()

    T_int2, P_chauff2, P_refroid2, Cout2  = task1(strt2)

    # Graphique de l'évolution des températures
    x = np.linspace(strt2, strt2+672, 672)

    axs[0][1].plot(x,T_int2.value)
    axs[0][1].set_title("Période 2 - Évolution des températures")
    axs[0][1].set_xlabel("Intervalle de temps")
    axs[0][1].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur
    x = np.linspace(strt2, strt2+672, 672)

    axs[1][1].plot(x, P_chauff2.value, label="Fonctionnement normal")
    axs[1][1].plot(x, P_refroid2.value, label="Fonctionnement reverse")
    axs[1][1].set_title("Période 2 - Utilisation de la pompe à chaleur")
    axs[1][1].set_xlabel("Intervalle de temps")
    axs[1][1].set_ylabel("Puissance (kW)")
    axs[1][1].legend()

    #Ajustement des graphs
    plt.subplots_adjust(wspace=0.5, hspace= 1)
    plt.show()

    ##task 2:
    T_int1, P_chauff1, P_refroid1, Cout1, Inconfort1 = task2(strt1,2)

    # Graphique de l'évolution des températures
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    x = np.linspace(strt1,strt1+672, 672)

    axs[0][0].plot(x,T_int1.value)
    axs[0][0].set_title("Période 1 - Évolution des températures")
    axs[0][0].set_xlabel("Intervalle de temps")
    axs[0][0].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur

    x = np.linspace(strt1,strt1+672, 672)

    axs[1][0].plot(x, P_chauff1.value, label="Fonctionnement normal")
    axs[1][0].plot(x, P_refroid1.value, label="Fonctionnement reverse")
    axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
    axs[1][0].set_xlabel("Intervalle de temps")
    axs[1][0].set_ylabel("Puissance (kW)")
    axs[1][0].legend()

    T_int2, P_chauff2, P_refroid2, Cout2, Inconfort2 = task2(strt2,8.7)

    # Graphique de l'évolution des températures
    x = np.linspace(strt2, strt2+672, 672)

    axs[0][1].plot(x,T_int2.value)
    axs[0][1].set_title("Période 2 - Évolution des températures")
    axs[0][1].set_xlabel("Intervalle de temps")
    axs[0][1].set_ylabel("Température (°C)")

    # Graphique représentant l'utilisation de la pompe à chaleur
    x = np.linspace(strt2, strt2+672, 672)

    axs[1][1].plot(x, P_chauff2.value, label="Fonctionnement normal")
    axs[1][1].plot(x, P_refroid2.value, label="Fonctionnement reverse")
    axs[1][1].set_title("Période 2 - Utilisation de la pompe à chaleur")
    axs[1][1].set_xlabel("Intervalle de temps")
    axs[1][1].set_ylabel("Puissance (kW)")
    axs[1][1].legend()

    #Ajustement des graphs
    plt.subplots_adjust(wspace=0.5, hspace= 1)
    plt.show()

def task3(strt1,strt2,pas):
    T_int11, P_chauff11, P_refroid11, Cout11 = task1(strt1)
    T_int21, P_chauff21, P_refroid21, Cout21  = task1(strt2)

    Cout_1 = np.arange(1,101,1)
    Cout_2 = np.arange(1,101,1)
    Cout_11 = Cout_1*Cout11/100
    Cout_21 = Cout_2*Cout21/100

    Inconfort_1 = np.zeros(100)
    Inconfort_2 = np.zeros(100)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    x_int1 = np.linspace(strt1, strt1+672, 672)
    x_int2 = np.linspace(strt2, strt2+672, 672)
    
    for i in range(1,101):
        if(i%pas == 0):
            T_int12, P_chauff12, P_refroid12, Cout12, Inconfort_1[i-1] = task2(strt1,Cout11*i/100)
            
            if(Cout12 is None):
                Inconfort_1[i-1] = 0
            elif(Cout12 is not None):
                axs[0][0].plot(x_int1,T_int12.value, label = str(i) + "%" + " du budget")
                axs[0][0].set_title("Période 1 - Évolution des températures")
                axs[0][0].set_xlabel("Intervalle de temps")
                axs[0][0].set_ylabel("Température (°C)")
                axs[0][0].legend(loc='center left', bbox_to_anchor=(1,0.5))

            T_int22, P_chauff22, P_refroid22, Cout22, Inconfort_2[i-1] = task2(strt2,Cout21*i/100)
            
            if(Cout22 is None):
                Inconfort_2[i-1] = 0
            elif(Cout22 is not None):
                axs[0][1].plot(x_int2,T_int22.value, label = str(i) + "%" + " du budget")
                axs[0][1].set_title("Période 1 - Évolution des températures")
                axs[0][1].set_xlabel("Intervalle de temps")
                axs[0][1].set_ylabel("Température (°C)")
                axs[0][1].legend(loc='center left', bbox_to_anchor=(1,0.5))


    axs[1][0].plot(Cout_11,Inconfort_1, label = "inconfort/cout")
    axs[1][0].set_title("Période 1 - Rapport cout/inconfort")
    axs[1][0].set_xlabel("Cout ($)")
    axs[1][0].set_ylabel("Inconfort")
    axs[1][0].legend()

    axs[1][1].plot(Cout_21,Inconfort_2, label = "inconfort/cout")
    axs[1][1].set_title("Période 1 - Rapport cout/inconfort")
    axs[1][1].set_xlabel("Cout ($)")
    axs[1][1].set_ylabel("Inconfort")
    axs[1][1].legend()
    
    plt.subplots_adjust(wspace=0.5, hspace= 1)
    plt.show()



if __name__ == "__main__":
    strt1 = 13050
    strt2 = 0
    pas = 2
    #plot_graph12(strt1,strt2)
    task3(strt1,strt2,pas)