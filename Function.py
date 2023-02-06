import numpy as np
import matplotlib.pyplot as plt
import Comsol_file as data
import random
from matplotlib.colors import SymLogNorm
from scipy.signal import butter,lfilter
import os

epsilon = 13.1 #eV
nombre_particule = 1
epaisseur = 0.15 #mm
q = 1.6*10**-19 #C
masse_effective = 9.1*10**-31 #kg
kB = 1.38 * 10 ** -23

temperature = main.temp
epaisseur_implantation = main.epaisseur_implantation
energie_perdue = main.energie_perdue
mu_e = main.mu_e
mu_h = main.mu_h
vsat_e = main.vsat_e
vsat_h = main.vsat_h
delta_temps = main.delta_temps
paquet_paires = main.paquet_paires
frequence_coupure_bas = main.frequence_coupure_bas
frequence_coupure_haut = main.frequence_coupure_haut
frequence_coupure_haut_Lecroy = main.frequence_coupure_haut_Lecroy
freq_rc = main.freq_rc
entree_particule=main.entree_part

nombre_de_paire = energie_perdue / epsilon

# fonctions generiques
def distribution_ion_aleatoire(nombre_ion):
    return([data.x_maillage[np.random.randint(len(data.x_maillage))] for i in range(nombre_ion)])
def trajectoire(x_entree,entree):
    angle = np.random.normal(0,Interface.disp_angle)
    result = np.zeros((2,2))
    if entree=='haut':
        result[1][1] = 0.15 - epaisseur_implantation * np.cos(np.pi * angle / 180.)
        result[1][0] = 0.15
    if entree=='bas':
        result[1][1] = epaisseur_implantation * np.cos(np.pi * angle / 180.)
        result[1][0] = 0
    else :
        print("erreur : Corriger le parametre entree. Choix : 'bas' ou 'haut'")
    result[0][0] = x_entree
    result[0][1] = x_entree+epaisseur_implantation*np.sin(np.pi*angle/180.)

    return(result)
def distribution_paires_init(trace):
    position_x = np.linspace(trace[0][0],trace[0][1],int(nombre_de_paire/paquet_paires))
    position_y = np.linspace(trace[1][0],trace[1][1],int(nombre_de_paire/paquet_paires))
    return(position_x,position_y)
def champ_electrique(position):
    x,y = position[0],position[1]
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[0]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<y and compteur_y<149:
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    # E_x = np.mean([data.Ex[(compteur_x-1)*150+(compteur_y-1)],data.Ex[(compteur_x)*150+(compteur_y-1)],data.Ex[(compteur_x-1)*150+(compteur_y)],data.Ex[(compteur_x)*150+(compteur_y)]]) # V/m
    # E_y = np.mean([data.Ey[(compteur_x-1)*150+(compteur_y-1)],data.Ey[(compteur_x)*150+(compteur_y-1)],data.Ey[(compteur_x-1)*150+(compteur_y)],data.Ey[(compteur_x)*150+(compteur_y)]]) # V/m
    E_x = data.Ex[(compteur_y)*150+(compteur_x)]
    E_y = data.Ey[(compteur_y)*150+(compteur_x)]
    return([E_x,E_y])
def weighting_field_1(position):
    x,y = position[0],position[1]
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
    x,y = position[0],position[1]
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
    x,y = position[0],position[1]
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
# def weighting_field_4(position):
#     x,y = position[0],position[1]
#     x_mail = data.x_maillage2[0]
#     y_mail = data.y_maillage2[0]
#     compteur_x = 0
#     compteur_y = 0
#     while x_mail<x and compteur_x<149:
#         compteur_x+=1
#         x_mail = data.x_maillage2[compteur_x]
#     while y_mail<x and compteur_y<149:
#         compteur_y+=1
#         y_mail = data.y_maillag2[compteur_y]
#     Ew_x = np.mean(
#         [data.Ewx4[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewx4[(compteur_x) * 150 + (compteur_y - 1)],
#          data.Ewx4[(compteur_x - 1) * 150 + (compteur_y)], data.Ewx4[(compteur_x) * 150 + (compteur_y)]])
#     Ew_y = np.mean(
#         [data.Ewy4[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewy4[(compteur_x) * 150 + (compteur_y - 1)],
#          data.Ewy4[(compteur_x - 1) * 150 + (compteur_y)], data.Ewy4[(compteur_x) * 150 + (compteur_y)]])
#     return ([Ew_x, Ew_y])
# def weighting_field_down(position):
#     x,y = position[0],position[1]
#     x_mail = data.x_maillage2[0]
#     y_mail = data.y_maillage2[0]
#     compteur_x = 0
#     compteur_y = 0
#     while x_mail<x and compteur_x<149:
#         compteur_x+=1
#         x_mail = data.x_maillage[compteur_x]
#     while y_mail<x and compteur_y<149:
#         compteur_y+=1
#         y_mail = data.y_maillage[compteur_y]
#     Ew_x = np.mean(
#         [data.Ewx5[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewx5[(compteur_x) * 150 + (compteur_y - 1)],
#          data.Ewx5[(compteur_x - 1) * 150 + (compteur_y)], data.Ewx5[(compteur_x) * 150 + (compteur_y)]])
#     Ew_y = np.mean(
#         [data.Ewy5[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewy5[(compteur_x) * 150 + (compteur_y - 1)],
#          data.Ewy5[(compteur_x - 1) * 150 + (compteur_y)], data.Ewy5[(compteur_x) * 150 + (compteur_y)]])
#     return ([Ew_x, Ew_y])
def v_drift_electron(position_e,modele):
    [Ex,Ey] = champ_electrique(position_e) # V/um
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2) # V/cm
    if modele=="Canali":
        return([-mu_e*Ex/(1+mu_e*E/vsat_e),-mu_e*Ey/(1+mu_e*E/vsat_e)]) # cm/s
    else:
        return ([-mu_e * Ex, -mu_e * Ey])  # cm/s
def v_drift_hole(position_h,modele):
    [Ex,Ey] = champ_electrique(position_h)
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2)  # V/cm
    if modele=="Canali":
        return([mu_h*Ex/(1+mu_h*E/vsat_h),mu_h*Ey/(1+mu_h*E/vsat_h)]) # cm/s
    else :
        return([mu_h*Ex,mu_h*Ey]) # cm/s
def intensite_1_e(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele),weighting_field_1(position=position)))
def intensite_2_e(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele),weighting_field_2(position=position)))
def intensite_3_e(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position,modele=modele),weighting_field_3(position=position)))
def intensite_1_h(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele),weighting_field_1(position=position)))
def intensite_2_h(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele),weighting_field_2(position=position)))
def intensite_3_h(position,charge,modele):
    return(-charge*10**-2*np.dot(v_drift_hole(position_h=position,modele=modele),weighting_field_3(position=position)))
# def intensite_4(position,charge):
#     return(-charge*10**-2*np.dot(v_drift_electron(position_e=position),weighting_field_4(position=position)))
# def intensite_down(position,charge):
#     return(-charge*10**-2*np.dot(v_drift_electron(position_e=position),weighting_field_down(position=position)))
def nouvelle_position_electron(position_x,position_y,modele):
    new_pos_x = position_x + v_drift_electron([position_x, position_y],modele=modele)[0]*10*delta_temps*10**-12 # mm
    new_pos_y = position_y + v_drift_electron([position_x, position_y],modele=modele)[1]*10*delta_temps*10**-12 # mm
    if new_pos_y>0.15:
        new_pos_y = 0.16
    return([new_pos_x,new_pos_y])
def nouvelle_position_trou(position_x,position_y,modele):
    new_pos_x = position_x + v_drift_hole([position_x, position_y],modele=modele)[0]*10*delta_temps*10**-12 # mm
    new_pos_y = position_y + v_drift_hole([position_x, position_y],modele=modele)[1]*10*delta_temps*10**-12 # mm
    if new_pos_y<0.:
        new_pos_y = -0.1
    return([new_pos_x,new_pos_y])
# valeur_random=[]
def thermique(temperature,position):
    vth = np.sqrt(2*kB*temperature/masse_effective)
    angle = np.random.randint(0,360)
    ran =(-np.log(random.uniform(0, 1)))
    x = position[0] + np.cos(angle*np.pi/180.)*vth*10**3*delta_temps*10**-12*ran
    # valeur_random.append(ran)
    y = position[1] + np.sin(angle*np.pi/180.)*vth*10**3*delta_temps*10**-12*ran
    # print(x-position[0])
    return([x,y])


## fonction calcul
def initialisation_graph():
    entree_particule = Interface.entree_particule
    fig = plt.figure()
    plt.ylim([-0.01,0.152])
    # plt.xlim([entree_particule[0]-0.15,entree_particule[0]+0.15])
    ax = plt.gca()
    # ax.set_aspect(1)
    plt.pcolormesh(data.x_maillage,data.y_maillage,np.zeros((len(data.x_maillage),len(data.y_maillage))),alpha=0)

    plt.pcolormesh([0,0,0.45,0.45],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='red',shading='auto')
    plt.pcolormesh([0.55,0.55,1,1],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='green',shading='auto')
    plt.pcolormesh([0.1,0.1,0.9,0.9],[0,0,-0.0001,-0.0001],np.ones((4,4)),edgecolors='midnightblue',shading='auto')
    z = np.zeros(( len(data.x_maillage),len(data.y_maillage)))
    for i in range(len(data.x_maillage)):
        for j in range(len(data.y_maillage)):
            z[i][j] = np.sqrt(data.Ex[i*150+j]**2+data.Ey[i*150+j]**2)
            # z[i][j] = data.Ex[i*150+j]
    plot = plt.pcolormesh(data.x_maillage, data.y_maillage, z, cmap='rainbow', shading='auto',norm=SymLogNorm(linthresh=10, linscale=0.001))#,norm=SymLogNorm(linthresh=100, linscale=0.001))    #norm=SymLogNorm(linthresh=100, linscale=0.001))

    ## contour needs the centers
    # cset = plt.contour(x, y,z, cmap='gray')
    # plt.clabel(cset, inline=True)
    plt.title("Drift of the charge in Diamond : x="+str(entree_particule[0])+" mm")
    plt.xlabel("Length in mm")
    plt.ylabel("Width in um")
    cbar = plt.colorbar(plot)
    cbar.set_label('Electric Field in V/m', rotation=270)

    plt.plot(entree_particule,[0.15 for i in range(nombre_particule)],'*')
    return(entree_particule,fig)
def initialisation_trajectoire(entree_particule,entree):
    for i in range(nombre_particule):
        trace = trajectoire(entree_particule[i],entree)
        plt.plot(trace[0],trace[1])
    return(trace)
def initialisation_paires(trace,figure,save):
    position_e_x_time = distribution_paires_init(trace)[0]
    position_e_y_time = distribution_paires_init(trace)[1]
    position_h_x_time = distribution_paires_init(trace)[0]
    position_h_y_time = distribution_paires_init(trace)[1]
    line, = plt.plot(position_e_x_time,position_e_y_time,'+',color='blue',label="electron")
    line2, = plt.plot(position_h_x_time,position_h_y_time,'+',color='red',label="hole")
    if save == "ON":
        dir = 'C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        figure.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\00000.png')
    return(line,line2,position_e_x_time,position_e_y_time,position_h_x_time,position_h_y_time)
def initialisation_temporelle(position_h_x_time,position_h_y_time,position_e_x_time,position_e_y_time,modele):
    temps_final = 7500.
    time = np.linspace(0,temps_final,int(temps_final/delta_temps))
    time_before = np.linspace(-temps_final/10,0,int(temps_final/(10*delta_temps)))
    time_total = np.append(time_before,time)
    i_1 = []
    i_2 = []
    i_3 = []
    # i_4 = []
    # i_down = []
    i1_0 = 0
    i2_0 = 0
    i3_0 = 0
    # i4_0 = 0
    # idown_0 = 0
    for time_i in time_before:
        i_1.append(0)
        i_2.append(0)
        i_3.append(0)
    for i in range(len(position_h_x_time)):
        if position_e_y_time[i]<0.15 or position_e_y_time[i]>0.:
            i1_0 += paquet_paires * intensite_1_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
            i2_0 += paquet_paires * intensite_2_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
            i3_0 += paquet_paires * intensite_3_e([position_e_x_time[i],position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
            # i4_0 += paquet_paires * intensite_4([position_e_x_time[i], position_e_y_time[i]], -q)
            # idown_0 += paquet_paires * intensite_down([position_e_x_time[i], position_e_y_time[i]], -q)
        if position_h_y_time[i]<0.15 or position_h_y_time[i]>0.:
            i1_0 += paquet_paires * intensite_1_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
            i2_0 += paquet_paires * intensite_2_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
            i3_0 += paquet_paires * intensite_3_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
            # i4_0 += paquet_paires * intensite_4([position_h_x_time[i], position_h_y_time[i]], q)
            # idown_0 += paquet_paires * intensite_down([position_h_x_time[i], position_h_y_time[i]], q)
    i_1.append(i1_0)
    i_2.append(i2_0)
    i_3.append(i3_0)
    # i4_0.append(i4_0)
    # idown_0.append(idown_0)
    return(i_1,i_2,i_3,time,temps_final,time_total)
def temporel(affichage_derive,time,temps_final,i_1,i_2,i_3,figure,position_e_x_time,position_e_y_time,position_h_x_time,position_h_y_time,line,line2,save,modele):
    compteur_image = 0
    for time_i in time[1:]:
        nombre_particule_electron = 0
        nombre_particule_trou = 0
        print('Effectue : ' + str(round(time_i/temps_final*100.,1))+'%')
        position_e_x_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i],position_e_y_time[i],modele=modele))[0] for i in range(len(position_e_x_time))]
        position_e_y_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i], position_e_y_time[i],modele=modele))[1] for i in range(len(position_e_x_time))]
        position_h_x_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i],modele=modele))[0] for i in range(len(position_h_x_time))]
        position_h_y_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i],modele=modele))[1] for i in range(len(position_h_x_time))]
        i1_i = 0
        i2_i = 0
        i3_i = 0
        # i4_i = 0
        # idown_i = 0
        for i in range(len(position_h_x_time)):
            if position_e_y_time[i] <= 0.15 and position_e_y_time[i] >= 0. and position_e_x_time[i] >= 0 and position_e_x_time[i] <= 1.:
                nombre_particule_electron += 1
                i1_i += paquet_paires * intensite_1_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
                i2_i += paquet_paires * intensite_2_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
                i3_i += paquet_paires * intensite_3_e([position_e_x_time[i], position_e_y_time[i]], -q,modele=modele)*10**(46./20)*50
                # i4_i += paquet_paires * intensite_4([position_e_x_time[i], position_e_y_time[i]], -q)*10**(46./20)*50
                # idown_i += paquet_paires * intensite_down([position_e_x_time[i], position_e_y_time[i]], -q)*10**(46./20)*50
            if position_h_y_time[i] <= 0.15 and position_h_y_time[i] >= 0. and position_h_x_time[i] >= 0 and position_h_x_time[i] <= 1.:
                nombre_particule_trou += 1
                i1_i += paquet_paires * intensite_1_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
                i2_i += paquet_paires * intensite_2_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
                i3_i += paquet_paires * intensite_3_h([position_h_x_time[i], position_h_y_time[i]], q,modele=modele)*10**(46./20)*50
                # i4_i += paquet_paires * intensite_4([position_h_x_time[i], position_h_y_time[i]], q) * 10 ** (46. / 20) * 50
                # idown_i += paquet_paires * intensite_down([position_h_x_time[i], position_h_y_time[i]], q) * 10 ** (46. / 20) * 50
        i_1.append(i1_i)
        i_2.append(i2_i)
        i_3.append(i3_i)
        # i_4.append(i4_i)
        # i_down.append(idown_i)
        # print(round(time_i,0))
        # print("electron : "+str(nombre_particule_electron))
        # print("trou : "+str(nombre_particule_trou))
        if affichage_derive=="ON":
            if(time_i//10) and [i1_i,i2_i,i3_i]!=[0,0,0]:
                compteur_image+=1
                # plt.plot(position_e_x_time,position_e_y_time,'+',color='blue')
                # plt.plot(position_h_x_time, position_h_y_time, '+', color='red')
                line.set_data(position_e_x_time,position_e_y_time)
                line2.set_data(position_h_x_time, position_h_y_time)
                leg = [str(round(time_i,2))+" ps","trajectoire","electron","hole"]
                plt.legend(leg)
                image = str(compteur_image).zfill(5)
                if save == "ON":
                    figure.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\Photo\\'+str(image)+'.png')
                plt.pause(0.001)  # pause avec duree en secondes
    return(i_1,i_2,i_3)
# fig = plt.figure()
# plt.plot(time,i_1,label='intensite haut gauche',color="red")
# plt.plot(time,i_2,label='intensite haut droite',color='orange')
# plt.plot(time[:len(i_3)],i_3,label='intensite bas',color='blue')
# # plt.plot(time,i_4,label='intensite 4')
# # plt.plot(time,i_down,label='intensite bas')
# plt.legend()
# plt.title("Signal en V (avec gain cividec, sans filtre)")
# plt.xlabel("temps en ps")
# plt.ylabel("Tension en V")
# if save == 1:
#     fig.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\GIF\\intensite_x_'+str(entree_particule)+'.png')
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a
def butter_lowpass_filter(data, lowcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def filtrage(i,time):
    fs = 1.E12/(time[1]-time[0])
    #filtrage
    # cut_signal = butter_lowpass_filter(i,freq_rc,fs,order=1)
    cut_signal = butter_bandpass_filter(i,frequence_coupure_bas,frequence_coupure_haut,fs,order=1)
    cut_signal = butter_lowpass_filter(cut_signal,frequence_coupure_haut_Lecroy,fs,order=1)
    return(cut_signal)
def affichage_intensite(filtre,save,time_total,time,i_1,i_2,i_3,entree_particule):
    fig1 = plt.figure()
    # plt.plot(time_total,i_1,label='intensite haut gauche',color="red")
    # plt.plot(time_total,i_2,label='intensite haut droite',color='green')
    # plt.plot(time_total,i_3,label='intensite bas',color='midnightblue')
    if(filtre=="ON"):
        plt.plot(time_total,filtrage(i_1,time),label='intensite haut gauche _ filtre',color="orange")
        plt.plot(time_total,filtrage(i_2,time),label='intensite haut droite _ filtre',color='chartreuse')
        plt.plot(time_total,filtrage(i_3,time),label='intensite bas _ filtre',color='blue')
    # plt.plot(time,i_4,label='intensite 4')
    # plt.plot(time,i_down,label='intensite bas')
    plt.legend()
    if(filtre == "ON"):
        plt.title("Signal en V (avec gain cividec, avec et sans filtre)")
    else :
        plt.title("Signal en V (avec gain cividec, sans filtre)")
    plt.xlabel("Temps en ps")
    plt.ylabel("Tension en V")
    if save == "ON":
        fig1.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\GIF\\intensitefiltre_x_alpha'+str(entree_particule[0])+'.png')
    # plt.figure();
    # plt.hist(valeur_random,1000)
    plt.show()
    print("\nThe End")
