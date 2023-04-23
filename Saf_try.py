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
 

dt = 15*60 #en seconde
nb_intervalles = 672 #nombres d'intervalles de 15 minutes sur 7 jours
temperatures_ext = np.load("Temperatures-Montreal.npy") # donées des températures à Montreal
eta = 0.99 # coefficient relatif à l'isolation
capacite_calorifique = 10/4  #kWh pour chauffer de 1°c

T_min = 19 #Température minimale du batiment
T_max = 21 #Température maximale du batiment

cout_elec = np.full(nb_intervalles,0.26) #initalisation du cout de l'élec a 0.26 partout
for i in range(7):
    for j in range(96):
        if j < 28 or j>= 88: #réajustement du cout de l'élec a 0.18 entre 22h et 7h
            cout_elec[j+i*96] = 0.18
            
COPT_reverse = 3.2 #COPT de la pompe quand on refroidi
def COP_normal(T_ext):
    return 3 + 10 * abs(np.tanh(T_ext/100)) * np.tanh(T_ext/100) #Fonction qui décrit le comportement du COP quand on réchauffe


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

    contraintes_1 += [T_int[i + 1] - T_int[i] == - (1 - eta) * ( T_int[i]- temperatures_ext[i]) + #perte de temp sans action
                        (COP_normal(temperatures_ext[i]) * P_chauff[i] * dt/(60*capacite_calorifique)) - #augmentation de la temp en mode normal
                        (COPT_reverse * P_refroid[i] * dt/(60*capacite_calorifique))] #diminution de la temp en mode reverse
    

#Contrainte sur la positivité des puissances et max KW
contraintes_1 += [P_chauff >= 0]
contraintes_1 += [P_refroid >= 0]
contraintes_1 += [P_chauff <= 1]
contraintes_1 += [P_refroid <= 1]

## Initialisation du cout total :
cost = cp.sum(cout_elec * (P_chauff + P_refroid)*4) 

start_time1 = time.time()

##Résolution 1 :
problem1 = cp.Problem(cp.Minimize(cost), contraintes_1)
first =problem1.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
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


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
"""DÉBUT TÂCHE 2"""
'''
Tache 2:
0n cherche à minimiser l'inconfort total (somme des inconforts sur toute la période considérée)
tout en respectant la contrainte de budget.
'''



 #budget_maximal #??, À TROUVER!!!!
#début de la résolution
#Ici les températures ne doivent plus appartenir à un certain intevevalle mais
#si jamais ça dépasse, des coefficients de pénalité apparaissent

 
dt = 15*60 #en seconde
nb_intervalles = 672 #nombres d'intervalles de 15 minutes sur 7 jours
temperatures_ext = np.load("Temperatures-Montreal.npy") # donées des températures à Montreal
eta = 0.99 # coefficient relatif à l'isolation
capacite_calorifique = 10/4  #kWh pour chauffer de 1°c
inconfort_inf = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures inférieures à T_min
inconfort_sup = cp.Variable(nb_intervalles, nonneg=True)  # inconfort pour les températures supérieures à T_max
penalite_inf = 3  # pénalité pour !!chaque degré!! en dessous de T_min
penalite_sup = 1  # pénalité pour !!chaque degré!! au-dessus de T_max
T_min = 19
T_max = 21
cout_tache1 = 1 #à trouver
alpha = 1 #à trouver
budget_max = alpha*cout_tache1 

cout_elec = np.full(nb_intervalles,0.26) #initalisation du cout de l'élec a 0.26 partout
for i in range(7):
    for j in range(96):
        if j < 28 or j>= 88: #réajustement du cout de l'élec a 0.18 entre 22h et 7h
            cout_elec[j+i*96] = 0.18
            
COPT_reverse = 3.2 #COPT de la pompe quand on refroidi
def COP_normal(T_ext):
    return 3 + 10 * abs(np.tanh(T_ext/100)) * np.tanh(T_ext/100) #Fonction qui décrit le comportement du COP quand on réchauffe


##Initialisation des variables :

T_int = cp.Variable(nb_intervalles)

#puissance qu'on va utiliser pour la pompe à chaleur
P_chauff = cp.Variable(nb_intervalles, nonneg=True) #en mode normal
P_refroid = cp.Variable(nb_intervalles, nonneg=True) #en mode inverse

## Initialisation du tableau de contraintes pour le probleme qui commence à 0:

contraintes_1 = []

contraintes_1 += [T_int[0] == 20] # Cf énoncé
contraintes_1 += [T_int[-1] == 20] # Cf énoncé


for i in range(nb_intervalles - 1):

    contraintes_1 += [T_int[i + 1] - T_int[i] == - (1 - eta) * ( T_int[i]- temperatures_ext[i]) + #perte de temp sans action
                        (COP_normal(temperatures_ext[i]) * P_chauff[i] * dt/(60*capacite_calorifique)) - #augmentation de la temp en mode normal
                        (COPT_reverse * P_refroid[i] * dt/(60*capacite_calorifique))] #diminution de la temp en mode reverse
     
for i in range(nb_intervalles):
    contraintes.append(T_int[i] - T_min >= -inconfort_inf[i]) #Cette contrainte s'assure que si la température intérieure T_int[i] est inférieure à T_min, alors la différence T_int[i] - T_min sera négative, et l'inconfort correspondant à cette différence sera stocké dans la variable inconfort_inf[i]. Le moins devant c'est car ça sera une valeur négative or on prend que les positives
    contraintes.append(T_int[i] - T_max <= inconfort_sup[i])

total_inconfort = cp.sum(penalite_inf * inconfort_inf + penalite_sup * inconfort_sup)
        
 
#Contrainte sur la positivité des puissances et max KW
contraintes_1 += [P_chauff >= 0]
contraintes_1 += [P_refroid >= 0]
contraintes_1 += [P_chauff <= 1]
contraintes_1 += [P_refroid <= 1]

## Initialisation du cout total :
cost = cp.sum(cout_elec * (P_chauff + P_refroid)*4) 

start_time1 = time.time()

##Résolution 1 :
problem1 = cp.Problem(cp.Minimize(total_inconfort), contraintes_1)
first =problem1.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
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








