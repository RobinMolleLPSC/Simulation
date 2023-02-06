import numpy as np
import matplotlib.pyplot as plt
import Comsol_file as data
import matplotlib.animation as animation

epsilon = 13.1 #eV
nombre_particule = 1
epaisseur_implantation = 0.15 #mm
energie_perdue = 1700000 #eV
epaisseur = 0.15 #mm
mu_e = 2200; # cm2/V.s
mu_h = 2640; # cm2/V.s
vsat_e = 0.821*10**7; # cm/s
vsat_h = 1.200*10**7; # cm/s
q = 1.6*10**-19 #C
delta_temps = 10. #ps
masse_effective = 9.1*10**-31 #kg
kB = 1.38 * 10 ** -23
paquet_paires = 1000.
nombre_de_paire = energie_perdue / epsilon
temperature = 300.

def distribution_ion_aleatoire(nombre_ion):
    return([data.x_maillage[np.random.randint(len(data.x_maillage))] for i in range(nombre_ion)])
def distribution_ion_fixe(nombre_ion):
    return([0.50 for i in range(nombre_ion)])
def trajectoire(x_entree):
    angle = np.random.normal(0,0.5)
    result = np.zeros((2,2))
    result[0][0] = x_entree
    result[1][0] = 0.15
    result[0][1] = x_entree+epaisseur_implantation*np.sin(np.pi*angle/180.)
    result[1][1] = 0.15-epaisseur_implantation*np.cos(np.pi*angle/180.)
    return(result)
def distribution_paires_init(trace):
    position_x = np.linspace(trace[0][0],trace[0][1],int(nombre_de_paire/paquet_paires))
    position_y = np.linspace(trace[1][0],trace[1][1],int(nombre_de_paire/paquet_paires))
    return(position_x,position_y)
def champ_electrique(position):
    x,y = position[0],position[1]
    x_mail = data.x_maillage[0]
    y_mail = data.y_maillage[1]
    compteur_x = 0
    compteur_y = 0
    while x_mail<x and compteur_x<149:
        compteur_x+=1
        x_mail = data.x_maillage[compteur_x]
    while y_mail<y and compteur_y<149:
        compteur_y+=1
        y_mail = data.y_maillage[compteur_y]
    E_x = np.mean([data.Ex[(compteur_x-1)*150+(compteur_y-1)],data.Ex[(compteur_x)*150+(compteur_y-1)],data.Ex[(compteur_x-1)*150+(compteur_y)],data.Ex[(compteur_x)*150+(compteur_y)]]) # V/m
    E_y = np.mean([data.Ey[(compteur_x-1)*150+(compteur_y-1)],data.Ey[(compteur_x)*150+(compteur_y-1)],data.Ey[(compteur_x-1)*150+(compteur_y)],data.Ey[(compteur_x)*150+(compteur_y)]]) # V/m
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
    Ew_x = np.mean([data.Ewx1[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewx1[(compteur_x) * 150 + (compteur_y - 1)],
                   data.Ewx1[(compteur_x - 1) * 150 + (compteur_y)], data.Ewx1[(compteur_x) * 150 + (compteur_y)]])
    Ew_y = np.mean([data.Ewy1[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewy1[(compteur_x) * 150 + (compteur_y - 1)],
                   data.Ewy1[(compteur_x - 1) * 150 + (compteur_y)], data.Ewy1[(compteur_x) * 150 + (compteur_y)]])
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
        [data.Ewx2[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewx2[(compteur_x) * 150 + (compteur_y - 1)],
         data.Ewx2[(compteur_x - 1) * 150 + (compteur_y)], data.Ewx2[(compteur_x) * 150 + (compteur_y)]])
    Ew_y = np.mean(
        [data.Ewy2[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewy2[(compteur_x) * 150 + (compteur_y - 1)],
         data.Ewy2[(compteur_x - 1) * 150 + (compteur_y)], data.Ewy2[(compteur_x) * 150 + (compteur_y)]])
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
        [data.Ewx3[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewx1[(compteur_x) * 150 + (compteur_y - 1)],
         data.Ewx3[(compteur_x - 1) * 150 + (compteur_y)], data.Ewx1[(compteur_x) * 150 + (compteur_y)]])
    Ew_y = np.mean(
        [data.Ewy3[(compteur_x - 1) * 150 + (compteur_y - 1)], data.Ewy3[(compteur_x) * 150 + (compteur_y - 1)],
         data.Ewy3[(compteur_x - 1) * 150 + (compteur_y)], data.Ewy3[(compteur_x) * 150 + (compteur_y)]])
    return ([Ew_x, Ew_y])
def v_drift_electron(position_e):
    [Ex,Ey] = champ_electrique(position_e) # V/um
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2) # V/cm
    return([-mu_e*Ex/(1+mu_e*E/vsat_e),-mu_e*Ey/(1+mu_e*E/vsat_e)]) # cm/s
def v_drift_hole(position_h):
    [Ex,Ey] = champ_electrique(position_h)
    Ex = 10 ** -2 * Ex # V/cm
    Ey = 10 ** -2 * Ey # V/cm
    E = np.sqrt(Ex**2+Ey**2)  # V/cm

    return([mu_h*Ex/(1+mu_h*E/vsat_h),mu_h*Ey/(1+mu_h*E/vsat_h)]) # cm/s
def intensite_1(position,charge):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position),weighting_field_1(position=position)))
def intensite_2(position,charge):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position),weighting_field_2(position=position)))
def intensite_3(position,charge):
    return(-charge*10**-2*np.dot(v_drift_electron(position_e=position),weighting_field_3(position=position)))
def nouvelle_position_electron(position_x,position_y):
    new_pos_x = position_x + v_drift_electron([position_x, position_y])[0]*10*delta_temps*10**-12 # mm
    new_pos_y = position_y + v_drift_electron([position_x, position_y])[1]*10*delta_temps*10**-12 # mm
    if new_pos_y>0.15:
        new_pos_y = 0.16
    return([new_pos_x,new_pos_y])
def nouvelle_position_trou(position_x,position_y):
    new_pos_x = position_x + v_drift_hole([position_x, position_y])[0]*10*delta_temps*10**-12 # mm
    new_pos_y = position_y + v_drift_hole([position_x, position_y])[1]*10*delta_temps*10**-12 # mm
    if new_pos_y<0.:
        new_pos_y = -0.1
    return([new_pos_x,new_pos_y])
def thermique(temperature,position):
    vth = np.sqrt(3*kB*temperature/masse_effective)
    angle = np.random.randint(0,360)
    x = position[0] + np.cos(angle*np.pi/180.)*vth*10**-3*delta_temps*10**-12
    y = position[1] + np.cos(angle*np.pi/180.)*vth*10**-3*delta_temps*10**-12
    # print(x-position[0])
    return(position)

entree_particule = distribution_ion_fixe(nombre_particule)
fig = plt.figure()
plt.ylim([0.140,0.152])
plt.xlim([entree_particule[0]-0.05,entree_particule[0]+0.05])
ax = plt.gca()
# ax.set_aspect(1)
plt.pcolormesh(data.x_maillage,data.y_maillage,np.zeros((len(data.x_maillage),len(data.y_maillage))),alpha=0)
plt.pcolormesh([0,0.475,0.475,0],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='red',shading='red')
plt.pcolormesh([1,0.525,0.525,1],[0.15,0.15,0.1501,0.1501],np.ones((4,4)),edgecolors='orange',shading='orange')
plt.pcolormesh([0.05,0.05,0.95,0.95],[0,-0.0001,-0.0001,0],np.ones((4,4)),edgecolors='blue',shading='blue')
plt.plot(entree_particule,[0.15 for i in range(nombre_particule)],'*')

for i in range(nombre_particule):
    trace = trajectoire(entree_particule[i])
    plt.plot(trace[0],trace[1])

position_e_x_time = distribution_paires_init(trace)[0]
position_e_y_time = distribution_paires_init(trace)[1]
position_h_x_time = distribution_paires_init(trace)[0]
position_h_y_time = distribution_paires_init(trace)[1]
line, = plt.plot(position_e_x_time,position_e_y_time,'+',color='blue')
line2, = plt.plot(position_h_x_time,position_h_y_time,'+',color='red')

i_1 = []
i_2 = []
i_3 = []
i1_0 = 0
i2_0 = 0
i3_0 = 0

for i in range(len(position_h_x_time)):
    if position_e_y_time[i]<0.15 or position_e_y_time[i]>0.:

        i1_0 += paquet_paires * intensite_1([position_e_x_time[i],position_e_y_time[i]], -q)
        i2_0 += paquet_paires * intensite_2([position_e_x_time[i],position_e_y_time[i]], -q)
        i3_0 += paquet_paires * intensite_3([position_e_x_time[i],position_e_y_time[i]], -q)
    if position_h_y_time[i]<0.15 or position_h_y_time[i]>0.:

        i1_0 += paquet_paires * intensite_1([position_h_x_time[i], position_h_y_time[i]], q)
        i2_0 += paquet_paires * intensite_2([position_h_x_time[i], position_h_y_time[i]], q)
        i3_0 += paquet_paires * intensite_3([position_h_x_time[i], position_h_y_time[i]], q)
i_1.append(i1_0)
i_2.append(i2_0)
i_3.append(i3_0)

temps_final = 5000.
time = np.linspace(0,temps_final,temps_final/delta_temps)

nombre_particule_electron = 0
nombre_particule_trou = 0
for time_i in time[1:]:
    nombre_particule_electron = 0
    nombre_particule_trou = 0
    print('Effectue : ' + str(round(time_i/10000.*100.,2))+'%')
    position_e_x_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i],position_e_y_time[i]))[0] for i in range(len(position_e_x_time))]
    position_e_y_time = [thermique(temperature,nouvelle_position_electron(position_e_x_time[i], position_e_y_time[i]))[1] for i in range(len(position_e_x_time))]
    position_h_x_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i]))[0] for i in range(len(position_h_x_time))]
    position_h_y_time = [thermique(temperature,nouvelle_position_trou(position_h_x_time[i], position_h_y_time[i]))[1] for i in range(len(position_h_x_time))]
    i1_i = 0
    i2_i = 0
    i3_i = 0
    for i in range(len(position_h_x_time)):
        if position_e_y_time[i] <= 0.15 and position_e_y_time[i] >= 0.:
            nombre_particule_electron += 1
            i1_i += paquet_paires * intensite_1([position_e_x_time[i], position_e_y_time[i]], -q)*10**(46./20)*50
            i2_i += paquet_paires * intensite_2([position_e_x_time[i], position_e_y_time[i]], -q)*10**(46./20)*50
            i3_i += paquet_paires * intensite_3([position_e_x_time[i], position_e_y_time[i]], -q)*10**(46./20)*50
        if position_h_y_time[i] <= 0.15 and position_h_y_time[i] >= 0.:
            nombre_particule_trou += 1
            i1_i += paquet_paires * intensite_1([position_h_x_time[i], position_h_y_time[i]], q)*10**(46./20)*50
            i2_i += paquet_paires * intensite_2([position_h_x_time[i], position_h_y_time[i]], q)*10**(46./20)*50
            i3_i += paquet_paires * intensite_3([position_h_x_time[i], position_h_y_time[i]], q)*10**(46./20)*50
    i_1.append(i1_i)
    i_2.append(i2_i)
    i_3.append(i3_i)

    if(time_i//100):
        # plt.plot(position_e_x_time,position_e_y_time,'+',color='blue')
        # plt.plot(position_h_x_time, position_h_y_time, '+', color='red')
        line.set_data(position_e_x_time,position_e_y_time)
        line2.set_data(position_h_x_time, position_h_y_time)
        leg = [str(round(time_i,2))+" ps",]
        plt.legend(leg)
        plt.pause(0.01)  # pause avec duree en secondes

# plt.plot(distribution_paires_init(trace)[0],distribution_paires_init(trace)[1],'+')
# plt.title('Trajectoire des ions dans le diamant')
# plt.xlabel('Largeur en mm')
# plt.ylabel('Epaisseur en mm')

fig = plt.figure()
plt.plot(time,i_1,label='intensite 1')
plt.plot(time,i_2,label='intensite 2')
plt.plot(time,i_3,label='intensite 3')
plt.legend()
plt.xlabel("temps en ps")
plt.ylabel("Intensite en A")

plt.show()