import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import math
import time
'''
Tache 1:
On souhaite que la température du bâtiment reste
comprise dans une certaine plage admissible de températures,
et on cherche à minimiser le coût total de l'électricité consommée par la pompe à chaleur.

'''

##Initialisation des paramètres :

dt = 15*60  # durée de chaque intervalle (en s)
nb_intervalles = 672 #nombres d'intervalles de 15 minutes sur 7 jours
temperatures_ext = np.load("Temperatures-Montreal.npy") # donées des températures à Montreal
eta = 0.99 # coefficient relatif à l'isolation
capacite_calorifique = 0.4 * 360 # en kwH
cost_electricity = np.full(nb_intervalles,0.26) #initalisation du cout de l'élec a 0.26 parto72 ut
for i in range(7):
    for j in range(96):
        if j < 28 or j>= 88: #réajustement du cout de l'élec a 0.18 entre 22h et 7h
            cost_electricity [j+i*96] = 0.18  # cout selon les heures creuses et les heures pleines

#intervalle de températures admissibles
T_min = 19
T_max = 21

##Initialisation des variables :

T_int = cp.Variable(nb_intervalles)

#puissance qu'on va utiliser pour la pompe à chaleur
P_chauff = cp.Variable(nb_intervalles, nonneg=True) #en mode normal
P_refroid = cp.Variable(nb_intervalles, nonneg=True) #en mode inverse

## Initialisation du tableau de contraintes pour le probleme qui commence à 0:
contraintes_1 = []

#la température du batiment doit rester admissible: 
contraintes_1 += [T_min <= T_int[i] for i in range(nb_intervalles)]
contraintes_1 += [T_int[i] <= T_max for i in range(nb_intervalles)]

for i in range(nb_intervalles - 1):
    COP_chauff = 3 + 10 * np.abs(np.tanh(temperatures_ext[i] / 100)) * np.tanh(temperatures_ext[i] / 100)#Contrainte sur la positivité des puissances

    COP_refroid = 3.2  #ne dépend pas de la température extérieure
    
    contraintes_1 += [T_int[i+1] == T_int[i] + (dt/(3600*COP_chauff))*(P_chauff[i]*COP_chauff - (T_int[i] - temperatures_ext[i])) - #FAUT ENLEVER
                      (dt/(3600*COP_refroid ))*(P_refroid[i]*COP_refroid  - (T_int[i+1] - temperatures_ext[i]))] ## deuxième terme = perte environnementale

    
    contraintes_1 += [T_int[i + 1] - T_int[i] - (1 - eta) * ( T_int[i]- temperatures_ext[i]) <= 0]#Contrainte sur la positivité des puissances
    #AJOUTER + (COPrechauffe * Puissance * dt * ε) - (COPrefroid * Puissance * dt * ε)

#Contrainte sur la positivité des puissances et max KW
contraintes_1 += [P_chauff >= 0, P_refroid >= 0]
contraintes_1 += [P_chauff <=  1, P_refroid <= 1]

## Initialisation du tableau de contraintes pour le probleme qui commence à 672:
contraintes_2 = []

#la température du batiment doit rester admissible: 
contraintes_2 += [T_min <= T_int[i] for i in range(nb_intervalles)]
contraintes_2 += [T_int[i] <= T_max for i in range(nb_intervalles)]

for i in range(nb_intervalles - 1):
    COP_chauff = 3 + 10 * np.abs(np.tanh(temperatures_ext[i] / 100)) * np.tanh(temperatures_ext[i+672] / 100)
    COP_refroid = 3.2  #ne dépend pas de la température extérieure
    
    contraintes_2 += [T_int[i+1] == T_int[i] + (dt/(3600*COP_chauff))*(P_chauff[i]*COP_chauff - (T_int[i] - temperatures_ext[i])) -
                      (dt/(3600*COP_refroid ))*(P_refroid[i]*COP_refroid  - (T_int[i+1] - temperatures_ext[i]))] ## deuxième terme = perte environnementale

    contraintes_2 += [T_int[i + 1] - T_int[i] - (1 - eta) * (temperatures_ext[i+672] - T_int[i]) >= 0]

#Contrainte sur la positivité des puissances
contraintes_2 += [P_chauff >= 0, P_refroid >= 0]

## Initialisation du cout total :
cost = cp.sum(cost_electricity * (P_chauff + P_refroid)) 

start_time1 = time.time()

##Résolution 1 :
problem1 = cp.Problem(cp.Minimize(cost), contraintes_1)
first =problem1.solve(solver= cp.SCIPY)

temps_calcul1 = time.time() - start_time1

##Récupération des valeurs :

print("1:", "\n","Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
      "Cout = ", problem1.value,"\n", "Temps de résolution = ", temps_calcul1)


# Graphique de l'évolution des températures
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
x = np.linspace(0, 672, 672)

axs[0][0].plot(x,T_int.value)
axs[0][0].set_title("Période 1 - Évolution des températures")
axs[0][0].set_xlabel("Intervalle de temps")
axs[0][0].set_ylabel("Température (°C)")



# Graphique représentant l'utilisation de la pompe à chaleur

x = np.linspace(0, 672, 672)

axs[1][0].plot(x, P_chauff.value, label="Fonctionnement normal")
axs[1][0].plot(x, P_refroid.value, label="Fonctionnement reverse")
axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
axs[1][0].set_xlabel("Intervalle de temps")
axs[1][0].set_ylabel("Puissance (kW)")
axs[1][0].legend()




##Résolution 2 :

start_time2 = time.time()

problem2 = cp.Problem(cp.Minimize(cost), contraintes_2)
second= problem2.solve(solver= cp.SCIPY)

temps_calcul2 = time.time() - start_time2

print("Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
      "Cout = ", problem2.value,"\n", "Temps de résolution = ", temps_calcul2)


# Graphique de l'évolution des températures
x = np.linspace(672, 2*672, 672)

axs[0][1].plot(x,T_int.value)
axs[0][1].set_title("Période 2 - Évolution des températures")
axs[0][1].set_xlabel("Intervalle de temps")
axs[0][1].set_ylabel("Température (°C)")



# Graphique représentant l'utilisation de la pompe à chaleur
x = np.linspace(672, 2*672, 672)

axs[1][1].plot(x, P_chauff.value, label="Fonctionnement normal")
axs[1][1].plot(x, P_refroid.value, label="Fonctionnement reverse")
axs[1][1].set_title("Période 2 - Utilisation de la pompe à chaleur")
axs[1][1].set_xlabel("Intervalle de temps")
axs[1][1].set_ylabel("Puissance (kW)")
axs[1][1].legend()

#Ajustement des graphs
plt.subplots_adjust(wspace=0.5, hspace= 1)


plt.show()


'''
Tache 2:
0n cherche à minimiser l'inconfort total (somme des inconforts sur toute la période considérée)
tout en respectant la contrainte de budget.

'''

##Initialisation des paramètres :

dt = 15*60  # durée de chaque intervalle (en s)
nb_intervalles = 672 #nombres d'intervalles de 15 minutes sur 7 jours
temperatures_ext = np.load("Temperatures-Montreal.npy") # donées des températures à Montreal
eta = 0.99 # coefficient relatif à l'isolation
capacite_calorifique = 0.4 * 360 # en kwH
cost_electricity = np.full(nb_intervalles,0.26) #initalisation du cout de l'élec a 0.26 partout
for i in range(7):
    for j in range(96):
        if j < 28 or j>= 88: #réajustement du cout de l'élec a 0.18 entre 22h et 7h
            cost_electricity [j+i*96] = 0.18  # cout selon les heures creuses et les heures pleines

#intervalle de températures admissibles
T_min = 19
T_max = 21

penalite_inf = 3  # pénalité pour chaque degré en dessous de T_min
penalite_sup = 1  # pénalité pour chaque degré au-dessus de T_max

budget_maximal_1 = 0.5*problem1.value #par exemple

##Initialisation des variables :

T_int = cp.Variable(nb_intervalles)

#puissance qu'on va utiliser pour la pompe à chaleur
P_chauff = cp.Variable(nb_intervalles, nonneg=True) #en mode normal
P_refroid = cp.Variable(nb_intervalles, nonneg=True) #en mode inverse


inconfort_inf = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures inférieures à T_min
inconfort_sup = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures supérieures à T_max


## Initialisation du tableau de contraintes pour le probleme qui commence à 0:
contraintes_1 = []


for i in range(nb_intervalles):
    contraintes_1.append(T_int[i] - T_min >= -inconfort_inf[i]) #l'inconfort correspondant à la différence avec la limite inferieure est stocké dans la variable inconfort_inf[i].
    contraintes_1.append(T_int[i] - T_max <= inconfort_sup[i]) #l'inconfort correspondant à la différence avec la limite supérieure est stocké dans la variable inconfort_inf[i].

for i in range(nb_intervalles - 1):
    COP_chauff = 3 + 10 * np.abs(np.tanh(temperatures_ext[i] / 100)) * np.tanh(temperatures_ext[i] / 100)
    COP_refroid = 3.2  #ne dépend pas de la température extérieure

    contraintes_1 += [T_int[i+1] == T_int[i] + (dt/(3600*COP_chauff))*(P_chauff[i]*COP_chauff - (T_int[i] - temperatures_ext[i])) -
                      (dt/(3600*COP_refroid ))*(P_refroid[i]*COP_refroid  - (T_int[i+1] - temperatures_ext[i]))] ## deuxième terme = perte environnementale

    contraintes_1 += [T_int[i + 1] - T_int[i] - (1 - eta) * ( T_int[i] -temperatures_ext[i] ) >= 0]

#Contrainte sur la positivité des puissances
contraintes_1 += [P_chauff >= 0, P_refroid >= 0]

## Initialisation du cout total et de l'inconfort :

cost = cp.sum(cost_electricity * (P_chauff + P_refroid))
total_inconfort = cp.sum(penalite_inf * inconfort_inf + penalite_sup * inconfort_sup)

#Contrainte sur le budget
contraintes_1.append(cost <= budget_maximal_1) 



## Initialisation du tableau de contraintes pour le probleme qui commence à 672:
contraintes_2 = []

for i in range(nb_intervalles):
    contraintes_2.append(T_int[i] - T_min >= -inconfort_inf[i]) #l'inconfort correspondant à la différence avec la limite inferieure est stocké dans la variable inconfort_inf[i].
    contraintes_2.append(T_int[i] - T_max <= inconfort_sup[i]) #l'inconfort correspondant à la différence avec la limite supérieure est stocké dans la variable inconfort_inf[i].

for i in range(nb_intervalles - 1):
    COP_chauff = 3 + 10 * np.abs(np.tanh(temperatures_ext[i] / 100)) * np.tanh(temperatures_ext[i+672] / 100)
    COP_refroid = 3.2  #ne dépend pas de la température extérieure
    
    contraintes_2 += [T_int[i+1] == T_int[i] + (dt/(3600*COP_chauff))*(P_chauff[i]*COP_chauff - (T_int[i] - temperatures_ext[i])) -
                      (dt/(3600*COP_refroid ))*(P_refroid[i]*COP_refroid  - (T_int[i+1] - temperatures_ext[i]))] ## deuxième terme = perte environnementale

    contraintes_2 += [T_int[i + 1] - T_int[i] - (1 - eta) * (temperatures_ext[i] - T_int[i]) >= 0]

#Contrainte sur la positivité des puissances
contraintes_2 += [P_chauff >= 0, P_refroid >= 0]

## Initialisation du cout total et de l'inconfort :

cost = cp.sum(cost_electricity * (P_chauff + P_refroid))
total_inconfort = cp.sum(penalite_inf * inconfort_inf + penalite_sup * inconfort_sup)

#Contrainte sur le budget
budget_maximal_2 = 0.5*problem2.value #par exemple
contraintes_2.append(cost <= budget_maximal_2) 


##Résolution 1 :
start_time1 = time.time()

problem1 = cp.Problem(cp.Minimize(total_inconfort), contraintes_1)
first =problem1.solve()

temps_calcul1 = time.time() - start_time1

##Récupération des valeurs :

print("1:", "\n","Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
      "Cout = ", cost.value,"\n","Inconfort total = ", problem1.value ,"\n", "Temps de résolution = ", temps_calcul1)





# Graphique de l'évolution des températures
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
x = np.linspace(0, 672, 672)

axs[0][0].plot(x,T_int.value)
axs[0][0].set_title("Période 1 - Évolution des températures")
axs[0][0].set_xlabel("Intervalle de temps")
axs[0][0].set_ylabel("Température (°C)")



# Graphique représentant l'utilisation de la pompe à chaleur

x = np.linspace(0, 672, 672)

axs[1][0].plot(x, P_chauff.value, label="Fonctionnement normal")
axs[1][0].plot(x, P_refroid.value, label="Fonctionnement reverse")
axs[1][0].set_title("Période 1 - Utilisation de la pompe à chaleur")
axs[1][0].set_xlabel("Intervalle de temps")
axs[1][0].set_ylabel("Puissance (kW)")
axs[1][0].legend()


##Résolution 2 :
start_time2 = time.time()

problem2 = cp.Problem(cp.Minimize(total_inconfort), contraintes_2)
second =problem2.solve()

temps_calcul2 = time.time() - start_time2

##Récupération des valeurs :

print("2:", "\n","Puissances normales = ", P_chauff.value, "\n", "Puissances reverses = ", P_refroid.value,"\n", "Températures internes = ", T_int.value, "\n",
      "Cout = ", cost.value,"\n","Inconfort total = ", problem2.value ,"\n", "Temps de résolution = ", temps_calcul2)



# Graphique de l'évolution des températures
x = np.linspace(672, 2*672, 672)

axs[0][1].plot(x,T_int.value)
axs[0][1].set_title("Période 2 - Évolution des températures")
axs[0][1].set_xlabel("Intervalle de temps")
axs[0][1].set_ylabel("Température (°C)")



# Graphique représentant l'utilisation de la pompe à chaleur
x = np.linspace(672, 2*672, 672)

axs[1][1].plot(x, P_chauff.value, label="Fonctionnement normal")
axs[1][1].plot(x, P_refroid.value, label="Fonctionnement reverse")
axs[1][1].set_title("Période 2 - Utilisation de la pompe à chaleur")
axs[1][1].set_xlabel("Intervalle de temps")
axs[1][1].set_ylabel("Puissance (kW)")
axs[1][1].legend()

#Ajustement des graphs
plt.subplots_adjust(wspace=0.5, hspace= 1)


plt.show()












