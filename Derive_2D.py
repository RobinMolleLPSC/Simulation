# Developpement : juin 2022
# Auteur : Molle Robin
# Mise à jour : 21/07/2022


# Titre : PyCom-drift (2D)

# Simulation deterministe 2D de derives des charges pour un detecteur diamant
# utilisation de COMSOL pour les champs electriques et weighting field
# Python effectue ensuite la derive


# Imports

import time
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import Comsol_file as data
import random
from matplotlib.colors import SymLogNorm
from scipy.signal import butter,lfilter
import os
import sys
from PIL import Image
import glob


############## Variables #############
global Mobilite_electron
global Mobilite_electron_Entry
global Mobilite_trou
global Mobilite_trou_Entry
global Vitesse_saturation_electron
global Vitesse_saturation_electron_Entry
global Vitesse_saturation_trou
global Vitesse_saturation_trou_Entry
global Pas_de_temps
global Pas_de_temps_Entry
global Paquets_charges
global Paquets_charges_Entry
global Entree_particule
global Entree_particule_Entry
global Energie_perdue
global Energie_perdue_Entry
global Epaisseur_implantation
global Epaisseur_implantation_Entry
global position_particule_haut_bas
global position_particule_haut_bas_Entry
global Dispersion_angulaire
global Dispersion_angulaire_Entry
global Temperature
global Temperature_Entry
global Freq_Cividec_haut
global Freq_Cividec_haut_Entry
global Freq_Cividec_bas
global Freq_Cividec_bas_Entry
global Freq_Lecroy
global Freq_Lecroy_Entry
global Freq_Cable
global Freq_Cable_Entry
global mu_e
global mu_h
global vsat_e
global vsat_h
global delta_temps
global paquet_paires
global entree_part
global energie_perdue
global epaisseur_implantation
global entree
global disp_angle
global temp
global frequence_coupure_bas
global frequence_coupure_haut
global frequence_coupure_haut_Lecroy
global freq_rc
global Save_parameter
global Save_parameter_Entry
global save
global Modele
global Modele_Entry
global model
global Affichage_derive
global Affichage_derive_Entry
global deriv
global Affichage_intensite
global Affichage_intensite_Entry
global intensite
global Filtre
global Filtre_Entry
global fitre
global parametre


# Constantes
epsilon = 13.1 #eV  # Energie creation paires electron-trou
epaisseur = 0.15 #mm  # Epaisseur du diamant (à modifier suivant le modèle COMSOL
q = 1.6*10**-19 #C    # Charge élementaire
masse_effective = 9.1*10**-31 #kg  # Masse d'un électron
kB = 1.38 * 10 ** -23 #J/K  # Constante de Boltzmann

def Derive_charges_2D():
    # Fermeture des fenêtres graphiques
    plt.close("all")

    # Récupération des variables issues des données utilisateur rentrées sur l'interface
    Tempe = Temperature_oui_value.get()                                         # Prise en compte de la temperature
    mu_e = float(Mobilite_electron_Entry.get())                                 # Mobilité electron
    mu_h = float(Mobilite_trou_Entry.get())                                     # Mobilité trou
    vsat_e = float(Vitesse_saturation_electron_Entry.get())                     # Vitesse de saturation électron
    vsat_h = float(Vitesse_saturation_trou_Entry.get())                         # Vitesse de saturation trou
    paquet_paires = float(Paquets_charges_Entry.get())                          # Nombre de paires e-h par paquets
    entree_part = [float(Entree_particule_Entry.get())]                         # Abscisse d'entree de la particule
    energie_perdue = float(Energie_perdue_Entry.get())                          # Energie perdue par la particule
    epaisseur_implantation = float(Epaisseur_implantation_Entry.get())          # Epaisseur d'implantation de la particule dans le diamant
    entree = position_particule_haut_bas_Entry.get()                            # Arrivée de la particule par la face supérieure ou inférieure
    disp_angle = float(Dispersion_angulaire_Entry.get())                        # Dispersion angulaire possible de la trajectoire
    if Tempe == "ON":
            temp = float(Temperature_Entry.get())                               # Valeur de la température
    else :
        temp = 0.                                                               # Pas de prise en compte de la température à 0K
    frequence_coupure_bas = float(Freq_Cividec_bas_Entry.get())                 # Fréquence préamplificateur passe bas
    frequence_coupure_haut = float(Freq_Cividec_haut_Entry.get())               # Fréquence préamplificateur passe haut
    frequence_coupure_haut_Lecroy = float(Freq_Lecroy_Entry.get())              # Fréquence coupure passe bas Lecroy
    freq_rc = float(Freq_Cable_Entry.get())                                     # Fréquence coupure passe bas Cables+Diamant
    nombre_de_paire = energie_perdue / epsilon                                  # Nombre de paires électron-trous créées
    save2 = save.get()                                                          # Sauvegarde des images
    model2 = model.get()                                                        # Modele de Canali utilisé ou non
    deriv2 = deriv.get()                                                        # Affichage de la dérive des charges
    intensite2 = intensite.get()                                                # Affichage de l'intensité
    filtre2 = filtre.get()                                                      # Prise en compte du filtrage par l'électronique
    delta_temps = float(Pas_de_temps_Entry.get())                               # Pas de temps de la simulation
    temps_final = float(temps_final_Entry.get())                                # Duree de la simulation
    nom_simu_gif = nom_simu_entry.get()                                         # Nom pour la sauvegarde
    affichage_derive= deriv2            # Nouvelle variable pour raison de type
    affichage_intensite2 = intensite2   # Nouvelle variable pour raison de type
    modele= model2                      # Nouvelle variable pour raison de type

    # Sauvegarde : lieu de sauvegarde :
    if save2=="ON":
        path_png = 'C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\'   # Emplacement images .png
        path_gif = 'C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\GIF\\'     # Emplacement image  .gif
        os.chdir(path_png)

        # Suppression de toutes les images dans le dossier .png => Sinon le .gif comportera des images de simulations antérieures
        for file in os.listdir(path_png):
            if file.endswith('.png'):
                os.remove(file)

    # Start pour le temps de la simulation
    time_0 = time.time()

    # En-tête
    print("Debut de la simulation PyCom-drift (2D) : " + nom_simu_gif+"\n")
    print("Parametres choisis :")
    print("Save : "+save2+" / Affichage de la derive des charges : "+affichage_derive+" / Affichage de l'intensite : "+affichage_intensite2)
    print("Filtrage avec preamplificateur :"+ filtre2)
    if save2=="ON":
        print('\nLieu de sauvegarde des images de derive :'+path_png)
        print("Lieu de sauvegarde du gif")
    print("\n////////////////////////////\n")

    # Variables pour conditions
    if save2 == "ON":
        save3 = 1
    else :
        save3 = 0
    if filtre2 == "ON":
        filtre3 = 1
    else :
        filtre3 = 0
    if affichage_intensite2=="ON":
        affichage_intensite3=1
    else :
        affichage_intensite3=0
    if affichage_derive=="ON":
        affichage_derive2=1
    else :
        affichage_derive2=0

    # Suite En-tête
    print(entree_part)
    # Initialisation :
    print("INITIALISATIONS :")
    print("Initialisation : fenetre graphique")
    entree_particule,figure_0 = initialisation_graph(entree_part=entree_part)
    print("Initialisation : trajectoire de l'ion")
    trace = initialisation_trajectoire(entree_particule=entree_particule,entree=entree,disp_angle=disp_angle,epaisseur_implantation=epaisseur_implantation)
    print("Initialisation : paires electron-trou")
    line, line2, electron_x_time, electron_y_time, hole_x_time, hole_y_time = initialisation_paires(trace=trace,figure=figure_0,save=save,nombre_de_paire=nombre_de_paire,paquet_paires=paquet_paires)
    print("Initialisation : resolution temporelle")
    i_1, i_2, i_3, time_intensite, temps_final, time_total = initialisation_temporelle(hole_x_time,hole_y_time,electron_x_time,electron_y_time,modele=modele,delta_temps=delta_temps,paquet_paires=paquet_paires
                                                                                       ,mu_e=mu_e,mu_h=mu_h,vsat_e=vsat_e,vsat_h=vsat_h,temps_final=temps_final)

    # Etude temporelle et drift
    print("\nResolution temporelle :")
    i_1, i_2, i_3 = temporel(affichage_derive=affichage_derive2, time=time_intensite,temps_final=temps_final,i_1=i_1,i_2=i_2,i_3=i_3,figure=figure_0,
                             position_e_x_time=electron_x_time,position_e_y_time=electron_y_time,position_h_x_time=hole_x_time,position_h_y_time=hole_y_time,
                             line=line,line2=line2,save=save3,modele=modele,temperature=temp,mu_e=mu_e,mu_h=mu_h,vsat_e=vsat_e,vsat_h=vsat_h,
                             delta_temps=delta_temps,paquet_paires=paquet_paires)
    # Stop temps
    time_1 = time.time()
    print("Duree de la simulation : "+str(round(time_1-time_0,1))+"s")
    # Sauvegarde en PNG
    if save3==1:
        print("Sauvegarde du .gif en cours")
        png_gif(path_png=path_png,path_gif=path_gif,duration=40,nom_simu=nom_simu_gif)
        print("Sauvegarde du .gif terminee")
    # Affichage :
    if affichage_intensite3==1:
        print("Affichage : intensite")
        affichage_intensite(filtre=filtre3,save=save3,i_1=i_1,i_2=i_2,i_3=i_3,time_total=time_total,time=time_intensite,frequence_coupure_bas=frequence_coupure_bas,
                            frequence_coupure_haut=frequence_coupure_haut,frequence_coupure_haut_Lecroy=frequence_coupure_haut_Lecroy,freq_rc=freq_rc,nom_simu=nom_simu_gif)

##############  Fonctions  ##################

# fonctions generiques
def trajectoire(x_entree,entree,disp_angle,epaisseur_implantation):
    angle = np.random.normal(0,disp_angle)                                                                              # tirage de l'angle avec lequel l'ion passe dans le diamant
    result = np.zeros((2,2))                                                                                            # initialisation droite (2 points)
    if entree=='haut':
        result[1][1] = 0.15 - epaisseur_implantation * np.cos(np.pi * angle / 180.)                                     # coordonnée y incidence par le haut
        result[1][0] = 0.15
    if entree=='bas':
        result[1][1] = epaisseur_implantation * np.cos(np.pi * angle / 180.)                                            # coordonnée y incidence par le bas
        result[1][0] = 0
    else :
        print("erreur : Corriger le parametre entree. Choix : 'bas' ou 'haut'")
    result[0][0] = x_entree                                                                                             # coordonnée x dans tous les cas
    result[0][1] = x_entree+epaisseur_implantation*np.sin(np.pi*angle/180.)
    return(result)
def distribution_paires_init(trace,nombre_de_paire,paquet_paires):                                                      # Distribution uniforme des paires e-h
    position_x = np.linspace(trace[0][0],trace[0][1],int(nombre_de_paire/paquet_paires))                                # Abscisse des paires e-h
    position_y = np.linspace(trace[1][0],trace[1][1],int(nombre_de_paire/paquet_paires))                                # Ordonnée des paires e-h
    return(position_x,position_y)
def champ_electrique(position):
    x,y = position[0],position[1]                                                                                       # position de la paire e-h
    # Détermination du champ électrique à la position de la paire
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[0]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:                                                                                  # parcours de la grille en x
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<y and compteur_y<149:                                                                                  # parcours de la grille en y
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    # Champ électrique moyen sur les 4 noeuds du maillage autour de la position de la paire
    E_x = np.mean([data.Ex[(compteur_y-1)*150+(compteur_x-1)],data.Ex[(compteur_y)*150+(compteur_x-1)],data.Ex[(compteur_y-1)*150+(compteur_x)],data.Ex[(compteur_y)*150+(compteur_x)]]) # V/m
    E_y = np.mean([data.Ey[(compteur_y-1)*150+(compteur_x-1)],data.Ey[(compteur_y)*150+(compteur_x-1)],data.Ey[(compteur_y-1)*150+(compteur_x)],data.Ey[(compteur_y)*150+(compteur_x)]]) # V/m
    return([E_x,E_y])
def weighting_field_1(position):
    x,y = position[0],position[1]                                                                                       # Identique à la fonction champ électrique, mais pour le weighting field de l'électrode 1
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[0]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<x and compteur_y<149:
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    Ew_x = np.mean([data.Ewx1[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewx1[(compteur_y) * 150 + (compteur_x - 1)],
                   data.Ewx1[(compteur_y - 1) * 150 + (compteur_x)], data.Ewx1[(compteur_y) * 150 + (compteur_x)]])
    Ew_y = np.mean([data.Ewy1[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewy1[(compteur_y) * 150 + (compteur_x - 1)],
                   data.Ewy1[(compteur_y - 1) * 150 + (compteur_x)], data.Ewy1[(compteur_y) * 150 + (compteur_x)]])
    return ([Ew_x, Ew_y])
def weighting_field_2(position):
    x,y = position[0],position[1]                                                                                       # Identique à la fonction champ électrique, mais pour le weighting field de l'électrode 2
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[0]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<x and compteur_y<149:
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    Ew_x = np.mean(
        [data.Ewx2[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewx2[(compteur_y) * 150 + (compteur_x - 1)],
         data.Ewx2[(compteur_y - 1) * 150 + (compteur_x)], data.Ewx2[(compteur_y) * 150 + (compteur_x)]])
    Ew_y = np.mean(
        [data.Ewy2[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewy2[(compteur_y) * 150 + (compteur_x - 1)],
         data.Ewy2[(compteur_y - 1) * 150 + (compteur_x)], data.Ewy2[(compteur_y) * 150 + (compteur_x)]])
    return ([Ew_x, Ew_y])
def weighting_field_3(position):
    x,y = position[0],position[1]                                                                                       # Identique à la fonction champ électrique, mais pour le weighting field de l'électrode 3
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[0]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<x and compteur_y<149:
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    Ew_x = np.mean(
        [data.Ewx3[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewx1[(compteur_y) * 150 + (compteur_x - 1)],
         data.Ewx3[(compteur_y - 1) * 150 + (compteur_x)], data.Ewx1[(compteur_y) * 150 + (compteur_x)]])
    Ew_y = np.mean(
        [data.Ewy3[(compteur_y - 1) * 150 + (compteur_x - 1)], data.Ewy3[(compteur_y) * 150 + (compteur_x - 1)],
         data.Ewy3[(compteur_y - 1) * 150 + (compteur_x)], data.Ewy3[(compteur_y) * 150 + (compteur_x)]])
    return ([Ew_x, Ew_y])
def v_drift_electron(position_e,modele,mu_e,vsat_e):
    [Ex,Ey] = champ_electrique(position_e) # V/um                                                                       # vitesse de dérive pour les électrons
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2) # V/cm
    if modele=="Canali":
        return([-mu_e*Ex/(1+mu_e*E/vsat_e),-mu_e*Ey/(1+mu_e*E/vsat_e)]) # cm/s
    else:
        return ([-mu_e * Ex, -mu_e * Ey])  # cm/s
def v_drift_hole(position_h,modele,mu_h,vsat_h):                                                                        # vitesse de dérive pour les trous
    [Ex,Ey] = champ_electrique(position_h)
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2)  # V/cm
    if modele=="Canali":
        return([mu_h*Ex/(1+mu_h*E/vsat_h),mu_h*Ey/(1+mu_h*E/vsat_h)]) # cm/s
    else :
        return([mu_h*Ex,mu_h*Ey]) # cm/s
def intensite_1_e(position,charge,modele,mu_e,vsat_e):                                                                  # Calcul intensité électron par le thérorème de Shockley-Ramo electrode 1
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele,mu_e=mu_e,vsat_e=vsat_e),weighting_field_1(position=position)))
def intensite_2_e(position,charge,modele,mu_e,vsat_e):                                                                  # Calcul intensité électron par le thérorème de Shockley-Ramo electrode 2
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele,mu_e=mu_e,vsat_e=vsat_e),weighting_field_2(position=position)))
def intensite_3_e(position,charge,modele,mu_e,vsat_e):                                                                  # Calcul intensité électron par le thérorème de Shockley-Ramo electrode 3
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele,mu_e=mu_e,vsat_e=vsat_e),weighting_field_3(position=position)))
def intensite_1_h(position,charge,modele,mu_h,vsat_h):                                                                  # Calcul intensité trou par le thérorème de Shockley-Ramo electrode 1
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele,mu_h=mu_h,vsat_h=vsat_h),weighting_field_1(position=position)))
def intensite_2_h(position,charge,modele,mu_h,vsat_h):                                                                  # Calcul intensité trou par le thérorème de Shockley-Ramo electrode 2
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele,mu_h=mu_h,vsat_h=vsat_h),weighting_field_2(position=position)))
def intensite_3_h(position,charge,modele,mu_h,vsat_h):                                                                  # Calcul intensité trou par le thérorème de Shockley-Ramo electrode 3
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele,mu_h=mu_h,vsat_h=vsat_h),weighting_field_3(position=position)))
def nouvelle_position_electron(position_x,position_y,modele,mu_e,vsat_e,delta_temps):
    new_pos_x = position_x + v_drift_electron([position_x, position_y],modele=modele,mu_e=mu_e,vsat_e=vsat_e)[0]*10*delta_temps*10**-12 # mm    # Détermination de la nouvelle position de l'électron
    new_pos_y = position_y + v_drift_electron([position_x, position_y],modele=modele,mu_e=mu_e,vsat_e=vsat_e)[1]*10*delta_temps*10**-12 # mm
    if new_pos_y>0.15:                                                                                                                          # Si l'électron sort du diamant, on le fait totalement sortir
        new_pos_y = 0.16
    return([new_pos_x,new_pos_y])
def nouvelle_position_trou(position_x,position_y,modele,mu_h,vsat_h,delta_temps):
    new_pos_x = position_x + v_drift_hole([position_x, position_y],modele=modele,mu_h=mu_h,vsat_h=vsat_h)[0]*10*delta_temps*10**-12 # mm        # Détermination de la nouvelle position du trou
    new_pos_y = position_y + v_drift_hole([position_x, position_y],modele=modele,mu_h=mu_h,vsat_h=vsat_h)[1]*10*delta_temps*10**-12 # mm
    if new_pos_y<0.:                                                                                                                            # Si le trou sort du diamant, on le fait totalement sortir
        new_pos_y = -0.1
    return([new_pos_x,new_pos_y])
def thermique(temperature,position,delta_temps):                                                                        # Effets thermiques
    if temperature==0:
        return(position)
    else:
        vth = np.sqrt(2*kB*temperature/masse_effective)                                                                 # Vitesse themrique
        angle = np.random.randint(0,360)                                                                                # Angle aléatoire
        x = position[0] + np.cos(angle * np.pi / 180.) * vth * 10 ** 3 * 5.10**-3 * 10 ** -12 * (-np.log(random.uniform(0, 1)))     # Nouvelle position : utilisation d'une loi exponentielle
        y = position[1] + np.sin(angle * np.pi / 180.) * vth * 10 ** 3 * 5.10**-3 * 10 ** -12 * (-np.log(random.uniform(0, 1)))
        for i in range(1,int(delta_temps//(5.10**-3))):                                                                 # On considère une interaction toutes les 5 picosecondes
            angle = np.random.randint(0, 360)
            x = x + np.cos(angle * np.pi / 180.) * vth * 10 ** 3 * 5.10**-3 * 10 ** -12 * (-np.log(random.uniform(0, 1)))
            y = y + np.sin(angle * np.pi / 180.) * vth * 10 ** 3 * 5.10**-3 * 10 ** -12 * (-np.log(random.uniform(0, 1)))
        return([x,y])

## fonction calcul
def initialisation_graph(entree_part):
    entree_particule = entree_part
    fig = plt.figure()
    plt.ylim([-0.01,0.152])  # taille fenêtre graphique
    plt.gca()
    plt.pcolormesh(data.x_maillage,data.y_maillage,np.zeros((len(data.x_maillage),len(data.y_maillage))),alpha=0)       # Création du diamant en récupérant les data de comsol niveau maillage

    plt.pcolormesh([0,0,0.45,0.45],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='red',shading='auto')            # Electrode
    plt.pcolormesh([0.55,0.55,1,1],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='green',shading='auto')          # Electrode
    plt.pcolormesh([0.1,0.1,0.9,0.9],[0,0,-0.0001,-0.0001],np.ones((4,4)),edgecolors='midnightblue',shading='auto')     # Electrode
    # Map de Champ électrique en fond
    z = np.zeros(( len(data.x_maillage),len(data.y_maillage)))
    for i in range(len(data.x_maillage)):
        for j in range(len(data.y_maillage)):
            z[i][j] = np.sqrt(data.Ex[i*150+j]**2+data.Ey[i*150+j]**2)                                                  # calcul de la norme du champ electrique
            # z[i][j] = data.Ex[i*150+j]
    plot = plt.pcolormesh(data.x_maillage, data.y_maillage, z, cmap='rainbow', shading='auto',norm=SymLogNorm(linthresh=10, linscale=0.001))
    plt.title("Drift of the charge in Diamond : x="+str(entree_particule[0])+" mm")
    plt.xlabel("Length in mm")
    plt.ylabel("Width in um")
    cbar = plt.colorbar(plot)
    cbar.set_label('Electric Field in V/m', rotation=270)
    plt.plot(entree_particule,[0.15 for i in range(1)],'*')                                                             # ajout de l'abscisse d'entre de l'ion
    return(entree_particule,fig)
def initialisation_trajectoire(entree_particule,entree,disp_angle,epaisseur_implantation):
    for i in range(1):                                                                                                  # option si ajout de particules
        trace = trajectoire(entree_particule[i],entree,disp_angle,epaisseur_implantation)                               # création de la trajectoire de l'ion
        plt.plot(trace[0],trace[1])                                                                                     # Affichage de la trajectoire
    return(trace)
def initialisation_paires(trace,figure,save,nombre_de_paire,paquet_paires):
    position_e_x_time = distribution_paires_init(trace,nombre_de_paire,paquet_paires)[0]
    position_e_y_time = distribution_paires_init(trace,nombre_de_paire,paquet_paires)[1]
    position_h_x_time = distribution_paires_init(trace,nombre_de_paire,paquet_paires)[0]
    position_h_y_time = distribution_paires_init(trace,nombre_de_paire,paquet_paires)[1]
    line, = plt.plot(position_e_x_time,position_e_y_time,'+',color='blue',label="electron")                             # Affichage des paires
    line2, = plt.plot(position_h_x_time,position_h_y_time,'+',color='red',label="hole")                                 # Affichage des paires
    if save == "ON":
        dir = 'C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        figure.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\00000.png')
    return(line,line2,position_e_x_time,position_e_y_time,position_h_x_time,position_h_y_time)
def initialisation_temporelle(position_h_x_time,position_h_y_time,position_e_x_time,position_e_y_time,modele,delta_temps,paquet_paires,mu_e,mu_h,vsat_e,vsat_h,temps_final):
    time = np.linspace(0,temps_final,int(temps_final/delta_temps))                                                      # Definition du temps
    time_before = np.linspace(-temps_final/10,0,int(temps_final/(10*delta_temps)))
    time_total = np.append(time_before,time)
    i_1 = []
    i_2 = []
    i_3 = []
    i1_0 = 0
    i2_0 = 0
    i3_0 = 0
    for time_i in time_before:
        i_1.append(0)
        i_2.append(0)
        i_3.append(0)
    for i in range(len(position_h_x_time)):
        if position_e_y_time[i]<0.15 or position_e_y_time[i]>0.:                                                        # Calcul intensite
            i1_0 += paquet_paires * intensite_1_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
            i2_0 += paquet_paires * intensite_2_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
            i3_0 += paquet_paires * intensite_3_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
        if position_h_y_time[i]<0.15 or position_h_y_time[i]>0.:
            i1_0 += paquet_paires * intensite_1_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
            i2_0 += paquet_paires * intensite_2_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
            i3_0 += paquet_paires * intensite_3_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
    i_1.append(i1_0)
    i_2.append(i2_0)
    i_3.append(i3_0)
    return(i_1,i_2,i_3,time,temps_final,time_total)
def temporel(affichage_derive,time,temps_final,i_1,i_2,i_3,figure,position_e_x_time,position_e_y_time,position_h_x_time,position_h_y_time,line,line2,save,modele,
             temperature,mu_e,mu_h,vsat_e,vsat_h,delta_temps,paquet_paires):
    compteur_image = 0
    for time_i in time[1:]:
        nombre_particule_electron = 0
        nombre_particule_trou = 0
        print('Effectue : ' + str(round(time_i/temps_final*100.,1))+'%')                                                # Calcul nouvelle position
        position_e_x_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i],position_e_y_time[i],modele=modele,mu_e=mu_e,vsat_e=vsat_e,delta_temps=delta_temps),delta_temps=delta_temps)[0] for i in range(len(position_e_x_time))]
        position_e_y_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i], position_e_y_time[i],modele=modele,mu_e=mu_e,vsat_e=vsat_e,delta_temps=delta_temps),delta_temps=delta_temps)[1] for i in range(len(position_e_x_time))]
        position_h_x_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i],modele=modele,mu_h=mu_h,vsat_h=vsat_h,delta_temps=delta_temps),delta_temps=delta_temps)[0] for i in range(len(position_h_x_time))]
        position_h_y_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i],modele=modele,mu_h=mu_h,vsat_h=vsat_h,delta_temps=delta_temps),delta_temps=delta_temps)[1] for i in range(len(position_h_x_time))]
        i1_i = 0
        i2_i = 0
        i3_i = 0
        for i in range(len(position_h_x_time)):                                                                         # Calcul intensite
            if position_e_y_time[i] <= 0.15 and position_e_y_time[i] >= 0. and position_e_x_time[i] >= 0 and position_e_x_time[i] <= 1.:
                nombre_particule_electron += 1
                i1_i += paquet_paires * intensite_1_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
                i2_i += paquet_paires * intensite_2_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
                i3_i += paquet_paires * intensite_3_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele,mu_e=mu_e,vsat_e=vsat_e)*10**(46./20)*50
            if position_h_y_time[i] <= 0.15 and position_h_y_time[i] >= 0. and position_h_x_time[i] >= 0 and position_h_x_time[i] <= 1.:
                nombre_particule_trou += 1
                i1_i += paquet_paires * intensite_1_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
                i2_i += paquet_paires * intensite_2_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
                i3_i += paquet_paires * intensite_3_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele,mu_h=mu_h,vsat_h=vsat_h)*10**(46./20)*50
        i_1.append(i1_i)
        i_2.append(i2_i)
        i_3.append(i3_i)
        if affichage_derive==1:
            if(time_i//10) and [i1_i,i2_i,i3_i]!=[0,0,0]:
                compteur_image+=1
                line.set_data(position_e_x_time,position_e_y_time)
                line2.set_data(position_h_x_time, position_h_y_time)
                leg = [str(round(time_i,2))+" ps","trajectoire","electron","hole"]
                plt.legend(leg)
                image = str(compteur_image).zfill(5)
                if save == 1:
                    figure.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\'+str(image)+'.png')
                plt.pause(0.001)  # pause avec duree en secondes
    return(i_1,i_2,i_3)
def butter_bandpass(lowcut, highcut, fs, order):                                                                        # filtre passe bande
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order):                                                           # filtre passe bande
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(lowcut, fs, order):                                                                                  # filtre passe bas
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a
def butter_lowpass_filter(data, lowcut, fs, order):                                                                     # filtre passe bas
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def filtrage(i,time,frequence_coupure_bas,frequence_coupure_haut,frequence_coupure_haut_Lecroy,freq_rc):                # filtres
    fs = 1.E12/(time[1]-time[0])
    #filtrage
    cut_signal = butter_lowpass_filter(i,freq_rc,fs,order=1)
    cut_signal = butter_bandpass_filter(cut_signal,frequence_coupure_bas,frequence_coupure_haut,fs,order=1)
    cut_signal = butter_lowpass_filter(cut_signal,frequence_coupure_haut_Lecroy,fs,order=1)
    return(cut_signal)              # filtre passe bas
def affichage_intensite(filtre,save,time_total,time,i_1,i_2,i_3,frequence_coupure_bas,frequence_coupure_haut,frequence_coupure_haut_Lecroy,freq_rc,nom_simu):
    fig1 = plt.figure()
    if(filtre==1):
        plt.subplot(211)
    plt.plot(time_total,i_1,label='intensite haut gauche',color="red")
    plt.plot(time_total,i_2,label='intensite haut droite',color='green')
    plt.plot(time_total,i_3,label='intensite bas',color='midnightblue')
    if(filtre == 1):
        plt.title("Signal en V (avec gain cividec, avec et sans filtre)")
    else:
        plt.title("Signal en V (avec gain cividec, sans filtre)")
    plt.xlabel("Temps en ps")
    plt.ylabel("Tension en V")
    plt.legend()
    if(filtre==1):
        plt.subplot(212)
        plt.plot(time_total,filtrage(i_1,time,frequence_coupure_bas,frequence_coupure_haut,frequence_coupure_haut_Lecroy,freq_rc),label='intensite haut gauche _ filtre',color="orange")
        plt.plot(time_total,filtrage(i_2,time,frequence_coupure_bas,frequence_coupure_haut,frequence_coupure_haut_Lecroy,freq_rc),label='intensite haut droite _ filtre',color='chartreuse')
        plt.plot(time_total,filtrage(i_3,time,frequence_coupure_bas,frequence_coupure_haut,frequence_coupure_haut_Lecroy,freq_rc),label='intensite bas _ filtre',color='blue')
        plt.legend()
    plt.xlabel("Temps en ps")
    plt.ylabel("Tension en V")
    if save == 1:
        fig1.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\GIF\\'+ str(nom_simu) +'.png')
    plt.show()
    print("\nThe End")
def png_gif(path_png,path_gif,duration,nom_simu):
    os.chdir(path_png)
    fGIF = str(nom_simu)+".gif"
    H = 480
    W = 640
    n = 1
    # Create the frames
    frames = []
    images = glob.glob("*.png")
    for i in images:
        newImg = Image.open(i)
        if (len(sys.argv) < 2 and n > 0):
            newImg = newImg.resize((W, H))
        frames.append(newImg)
    # Save into a GIF file that loops forever: duration is in milli-second
    os.chdir(path_gif)

    frames[0].save(fGIF, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration, loop=0)
    return()


###################  Interface Graphique ########################


window = Tk()
window.title("PyCom-drift : simulation 2D de chambre d'ionisation solide diamant")
window.geometry("1000x550")


StudyType_Txt = Label(window, text = "Etude PyCom-drift : simulation 2D de chambre d'ionisation solide diamant: \n"
                                     ,font="Helvetica", fg='black')
StudyType_Txt.place (x=150, y=50)
StudyType_Txt = Label(window, text = "Parametres :", fg='black')
StudyType_Txt.place (x=480, y=80)


Mobilite_electron = Label(window, text="Mobilite des electrons : [en cm2/V.s] ", fg='blue')
Mobilite_electron.place(x=50, y=100)
Mobilite_electron_Entry = Entry(window, width=10)
Mobilite_electron_Entry.place(x=400, y=100)
Mobilite_electron_Entry.insert(0,"2200.")

Mobilite_trou = Label(window, text="Mobilite des trous : [en cm2/V.s]", fg='red')
Mobilite_trou.place(x=50,y=120)
Mobilite_trou_Entry = Entry(window, width=10)
Mobilite_trou_Entry.place(x=400,y=120)
Mobilite_trou_Entry.insert(0,"2640.")

Vitesse_saturation_electron = Label(window, text="Vitesse de saturation des electrons : [en cm/s] ", fg='blue')
Vitesse_saturation_electron.place(x=50,y=140)
Vitesse_saturation_electron_Entry = Entry(window, width=10)
Vitesse_saturation_electron_Entry.place(x=400,y=140)
Vitesse_saturation_electron_Entry.insert(0,"0.821E7")

Vitesse_saturation_trou = Label(window, text="Vitesse de saturation des trous : [en cm/s] ", fg='red')
Vitesse_saturation_trou.place(x=50,y=160)
Vitesse_saturation_trou_Entry = Entry(window, width=10)
Vitesse_saturation_trou_Entry.place(x=400,y=160)
Vitesse_saturation_trou_Entry.insert(0,"1.2E7")


Paquets_charges = Label(window, text="Paquets de charges : [nombre]", fg='black')
Paquets_charges.place(x=50,y=200)
Paquets_charges_Entry = Entry(window, width=10)
Paquets_charges_Entry.place(x=400,y=200)
Paquets_charges_Entry.insert(0,"10000.")

Entree_particule = Label(window, text="Abscisse d'entree de la particule : [en mm] ", fg='black')
Entree_particule.place(x=50,y=220)
Entree_particule_Entry = Entry(window, width=10)
Entree_particule_Entry.place(x=400,y=220)
Entree_particule_Entry.insert(0,"0.5")

Energie_perdue = Label(window, text="Energie perdue par la particule : [en eV] ", fg='black')
Energie_perdue.place(x=50,y=240)
Energie_perdue_Entry = Entry(window, width=10)
Energie_perdue_Entry.place(x=400,y=240)
Energie_perdue_Entry.insert(0,"5.4E6")

Epaisseur_implantation = Label(window, text="Parcours de la particule dans le diamant : [en mm] ", fg='black')
Epaisseur_implantation.place(x=50,y=260)
Epaisseur_implantation_Entry = Entry(window, width=10)
Epaisseur_implantation_Entry.place(x=400,y=260)
Epaisseur_implantation_Entry.insert(0,"0.015")

position_particule_haut_bas = Label(window, text="Incidence de la particule : [haut ou bas] ", fg='black')
position_particule_haut_bas.place(x=50,y=280)
position_particule_haut_bas_Entry = Entry(window, width=10)
position_particule_haut_bas_Entry.place(x=400,y=280)
position_particule_haut_bas_Entry.insert(0,"haut")

Dispersion_angulaire = Label(window, text="Angle par rapport a la normale (dispersion) : [en degre] ", fg='black')
Dispersion_angulaire.place(x=50,y=300)
Dispersion_angulaire_Entry = Entry(window, width=10)
Dispersion_angulaire_Entry.place(x=400,y=300)
Dispersion_angulaire_Entry.insert(0,"0.5")


Freq_Cividec_bas = Label(window, text="Frequence coupure passe bas Cividec : [en Hz] ", fg='green')
Freq_Cividec_bas.place(x=50,y=360)
Freq_Cividec_bas_Entry = Entry(window, width=10)
Freq_Cividec_bas_Entry.place(x=400,y=360)
Freq_Cividec_bas_Entry.insert(0,"4.E6")

Freq_Cividec_haut = Label(window, text="Frequence coupure passe haut Cividec : [en Hz] ", fg='green')
Freq_Cividec_haut.place(x=50,y=380)
Freq_Cividec_haut_Entry = Entry(window, width=10)
Freq_Cividec_haut_Entry.place(x=400,y=380)
Freq_Cividec_haut_Entry.insert(0,"2.E9")

Freq_Lecroy = Label(window, text="Frequence de coupure de l'oscilloscope : [en Hz] ", fg='green')
Freq_Lecroy.place(x=50,y=400)
Freq_Lecroy_Entry = Entry(window, width=10)
Freq_Lecroy_Entry.place(x=400,y=400)
Freq_Lecroy_Entry.insert(0,"2.5E9")

Freq_Cable = Label(window, text="Frequence de coupure des cables : [en Hz] ", fg='green')
Freq_Cable.place(x=50,y=420)
Freq_Cable_Entry = Entry(window, width=10)
Freq_Cable_Entry.place(x=400,y=420)
Freq_Cable_Entry.insert(0,"3.E9")

Save_parameter = Label(window, text="Sauvegarde : ", fg='purple4')
Save_parameter.place(x=550,y=180)
save = StringVar(value="OFF")
Save_parameter_Entry = Checkbutton(window,text="Active", variable=save, onvalue="ON", offvalue="OFF")
Save_parameter_Entry.place(x=900,y=180)

Modele = Label(window, text="Modele de vitesse de derive : [Canali ou simplifie] ", fg='purple4')
Modele.place(x=550,y=200)
model = StringVar(value="Canali")
Modele_Entry = Checkbutton(window,variable=model, onvalue="Canali", offvalue="Autre",text=model.get())
Modele_Entry.place(x=900,y=200)

Affichage_derive = Label(window, text="Affichage de la derive des charges : [Oui/Non] ", fg='purple4')
Affichage_derive.place(x=550,y=220)
deriv = StringVar(value="OFF")
Affichage_derive_Entry = Checkbutton(window,variable=deriv, onvalue="ON", offvalue="OFF",text="Oui")
Affichage_derive_Entry.place(x=900,y=220)

Affichage_intensite = Label(window, text="Affichage de l'intensite : [Oui/Non] ", fg='purple4')
Affichage_intensite.place(x=550,y=240)
intensite = StringVar(value="OFF")
Affichage_intensite_Entry = Checkbutton(window,variable=intensite, onvalue="ON", offvalue="OFF",text="Oui")
Affichage_intensite_Entry.place(x=900,y=240)

Filtre = Label(window, text="Prise en compte de l'electronique : [Oui/Non] ", fg='purple4')
Filtre.place(x=550,y=260)
filtre = StringVar(value="OFF")
Filtre_Entry = Checkbutton(window, variable=filtre, onvalue="ON", offvalue="OFF",text="Oui")
Filtre_Entry.place(x=900,y=260)

Temperature_oui = Label(window, text="Prise en compte de la temperature : ", fg='purple4')
Temperature_oui.place(x=550,y=280)
Temperature_oui_value = StringVar(value="OFF")
Temperature_oui_Entry = Checkbutton(window,text="Active", variable=Temperature_oui_value, onvalue="ON", offvalue="OFF")
Temperature_oui_Entry.place(x=900,y=280)

Temperature = Label(window, text="Temperature : [en K] ", fg='purple4')
Temperature.place(x=580,y=300)
Temperature_Entry = Entry(window, width=10)
Temperature_Entry.place(x=900,y=300)
Temperature_Entry.insert(0,"0.")


Pas_de_temps = Label(window, text="Pas de temps : [en ps] ", fg='black')
Pas_de_temps.place(x=550, y=100)
Pas_de_temps_Entry = Entry(window, width=10)
Pas_de_temps_Entry.place(x=900, y=100)
Pas_de_temps_Entry.insert(0,"40.")

Temps_final = Label(window, text="Temps estime de derive : [en ps] ", fg='black')
Temps_final.place(x=550, y=120)
temps_final_Entry = Entry(window, width=10)
temps_final_Entry.place(x=900, y=120)
temps_final_Entry.insert(0,"7500.")

nom_exp = Label(window, text="Nom de la Simulation ", fg='red3')
nom_exp.place(x=550,y=340)
nom_simu_entry = Entry(window, width=43)
nom_simu_entry.place(x=700, y=340)

Validate = Button(window, bg="chartreuse", activebackground="darkgreen", relief="raised", text="Valider",
                  command=lambda: Derive_charges_2D())
Validate.place(x=333, y=500)

QUITTER = Button(window, bg='orangered', activebackground="darkred", text="QUITTER", command=exit)
QUITTER.place(x=666, y=500)


window.mainloop()


