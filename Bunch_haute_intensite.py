from pylab import *
import numpy as np
from scipy.signal import butter,lfilter
from scipy.integrate import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import *

# donnees diamants
polarisation_haut = 150. # V
polarisation_bas = 0. # V
rho_0 = 0. # C.um^-1
mu_e = 2200. * 1E-4 # um^2/V.ps
vsat_e = 0.821E7 * 1E4 # um/s
mu_h = 2640. * 1E-4 # um^2/V.ps
vsat_h = 1.2E7 * 1E4 # um/s
epaisseur_diamant = 150. # µm
tau_n = 1.E8 # ns
tau_nt = 1.E8 # ns
tau_p = 1.E8 # ns
tau_pt = 1.E8 # ns

# donnees ion
energie_perdue_particule_simple = 1700000. # eV
intensite_diamant = 10. # nA
duree_bunch_on = 4. # ns
duree_bunch_off = 29. # ns
duree_train = 10. # µs
duree_inter_train = 100 # µs

# donnees simulation
pas_de_temps = 1. # ps
temps_debut = -10. # ps
temps_fin = 10. # ps
temperature = 300.
bins = 1500
alpha_1 = 0.  # um^1.ps^-1
alpha_2 = 0.  # um^1.ps^-1
alpha_3 = 0.  # um^1.ps^-1

# constantes
eps_0 = 8.85E12 #F.m^-1
kB = 1.381E-23 # J.K^-1
q = 1.6E-19 # C
eps_r = 5.7 # sans unite
creation_paires_electron_trou = 13.6 # eV

Dn = mu_e*kB*temperature/q
Dp = mu_h*kB*temperature/q


# initialisation :
def initialisation():
    length = np.linspace(-epaisseur_diamant/(bins), epaisseur_diamant*(1+1/(bins)), num=bins + 3)
    dx = (length[1] - length[0]) # um
    dt = pas_de_temps # ps
    time_simu = np.linspace(temps_debut, temps_fin, num=int((temps_fin - temps_debut) / pas_de_temps) + 1)

    densite_par_ion = energie_perdue_particule_simple/(creation_paires_electron_trou*(bins)) # C.um^-1
    nombre_ion_par_bunch = (duree_bunch_off+duree_bunch_on)*1E-9*intensite_diamant*1E-9/q

    n_x = np.zeros(np.size(length)) # µm^-1
    p_x = np.zeros(np.size(length)) # µm^-1
    nt_x = np.zeros(np.size(length)) # µm^-1
    pt_x = np.zeros(np.size(length)) # µm^-1
    V_x = np.linspace(polarisation_bas,polarisation_haut,num=bins+1) # V
    E_x = np.ones(np.size(length))*(V_x[-1]-V_x[0])/epaisseur_diamant # V/um
    rho_x = q*(n_x+p_x+nt_x+pt_x) # C.m^-1

    return(length,dx,dt,time_simu,densite_par_ion,nombre_ion_par_bunch,n_x,p_x,nt_x,pt_x,V_x,E_x,rho_x)

length,dx,dt,time_simu,densite_par_ion,nombre_ion_par_bunch,n_x,p_x,nt_x,pt_x,V_x,E_x,rho_x = initialisation()

def Poisson_bunch(nombre_ion_par_bunch,dt):
    lambda_nombre_proton_pas_de_temps = nombre_ion_par_bunch*dt*1E-3/(duree_bunch_on)
    nb_proton_temps_bunch_on = [ int(np.random.poisson(lambda_nombre_proton_pas_de_temps,1)+0.5) for i in range(int(duree_bunch_on/(pas_de_temps*1.E-3)))]
    return(nb_proton_temps_bunch_on)

fig = plt.figure()
axes = fig.add_subplot(111)
plt.plot(length,np.zeros(np.size(length)),color="grey")
line1, = plt.plot(length,n_x, color='blue', label="electron")
line2, = plt.plot(length,p_x, color='red', label="hole")
plt.legend()


# temporel

def evol_x_potentiel(V_x,E_x,dx,rho_x):
    for i in range(np.size(E_x)-1):
        E_x[i+1] = E_x[i] + dx*rho_x[i]/(eps_0*eps_r*1.E6) # V/um
    for i in range(np.size(V_x)-1):
        V_x[i+1] = V_x[i] - dx*E_x[i] # V
    return(E_x,V_x)
def evol_tx_rho_x(n_x,nt_x,p_x,pt_x):
    return(p_x+pt_x+n_x+nt_x)
def evol_t_n_x(n_x,nt_x,p_x,pt_x,E_x):
    for i in range(1,np.size(n_x)-1):
        n_x[i] += dt * (-(n_x[i]*mu_e*E_x[i] + Dn * (n_x[i-1]+n_x[i+1]-2*n_x[i])/(dx**2)) - alpha_1*n_x[i]*pt_x[i] - alpha_2*n_x[i]*p_x[i] + n_x[i]/tau_n - nt_x[i]/tau_nt)
    n_x[0] += 0
    n_x[-1] += 0
    return(n_x)
def evol_t_p_x(n_x,nt_x,p_x,pt_x,E_x):
    for i in range(1,np.size(p_x)-1):
        p_x[i] += dt * ((p_x[i]*mu_h*E_x[i] + Dp * (p_x[i-1]+p_x[i+1]-2*p_x[i])/(dx**2)) - alpha_3*p_x[i]*nt_x[i] - alpha_2*n_x[i]*p_x[i] - p_x[i]/tau_p + pt_x[i]/tau_pt)
    p_x[0] += 0
    p_x[-1] += 0
    return(p_x)
def evol_t_nt_x(n_x,nt_x,p_x):
    for i in range(1,np.size(p_x)-1):
        nt_x[i] += dt * (- alpha_3 * p_x[i] * nt_x[i] + n_x[i]/tau_n - nt_x[i]/tau_nt)
    nt_x[0] = 0
    nt_x[-1] = 0
    return(nt_x)
def evol_t_pt_x(n_x,p_x,pt_x):
    for i in range(np.size(p_x)):
        pt_x[i] += dt * (- alpha_1 * pt_x[i] * n_x[i] + p_x[i]/tau_n - pt_x[i]/tau_nt)
    pt_x[0] = 0
    pt_x[-1] = 0
    return(pt_x)


def equation_poisson(rho_x,x):
    return(rho_x/(eps_0*eps_r))

def coupled_resolution_t(Vector_x,Vector_x_dx,Vector_x_mdx,t):
    rho_x = np.sum(Vector_x)
    E_x = odeint(equation_poisson(rho_x,length),E_x,length)

    A = Vector_x[0]
    B = Vector_x[1]
    C = Vector_x[2]
    D = Vector_x[3]

    A_mdx = Vector_x_mdx[0]
    B_mdx = Vector_x_mdx[1]

    A_dx = Vector_x_dx[0]
    B_dx = Vector_x_dx[1]


    # constantes

    a1 = -mu_e*E_x - 1/tau_n
    a2 = -Dn
    a3 = -alpha_1
    a4 = -alpha_2
    a5 = 1/tau_nt

    b1 = mu_h*E_x - 1/tau_p
    b2 = -Dp
    b3 = -alpha_3
    b4 = -alpha_2
    b5 = 1/tau_pt

    c1 = -alpha_3
    c2 = 1/tau_n
    c3 = -1/tau_nt

    d1 = -alpha_1
    d2 = 1/tau_p
    d3 = -1/tau_pt


    # derivee

    Aseconde = (A_mdx+A_dx-2*A)/dx**2
    Bseconde = (B_mdx+B_dx-2*B)/dx**2

    dndt = a1 * A + a2 * Aseconde + a3 * D * A + a4 * B * A + a5 * C
    dpdt = b1 * B + b2 * Bseconde + b3 * C * B + b4 * B * A + b5 * D
    dntdt = c1 * B * C + c2 * A + c3 * C
    dptdt = d1 * A * D + d2 * B + d3 * D

    Vector_dx = [dndt,dpdt,dntdt,dptdt]
    return(Vector_dx)

for i in range(1,np.size(length)-1):
    init_x = [n_x[i],p_x[i],nt_x[i],pt_x[i]]
    init_mx = [n_x[i-1], p_x[i-1], nt_x[i-1], pt_x[i-1]]
    init_dx = [n_x[i+1], p_x[i+1], nt_x[i+1], pt_x[i+1]]
    Vector_x = odeint(coupled_resolution_t,init_x,init_dx,init_mx,time_simu)






nombre_bunch = np.minimum(temps_fin*1E-3%(duree_bunch_on+duree_bunch_off),duree_train*1.E3%(duree_bunch_on+duree_bunch_off))
bunch = 0
time_i = time_simu[0]
affichage_proba = 0.05
while time_i+dt<0.:
    line1.set_data(length, n_x)
    line2.set_data(length, p_x)
    leg = [str(round(time_i, 2)) + " ps", "electron", "hole"]
    plt.legend(leg)
    maxi = np.maximum(np.max(n_x),1)
    maxi = np.maximum(np.max(p_x),maxi)
    mini = np.minimum(np.min(n_x), -1)
    mini = np.minimum(np.min(p_x), mini)
    axes.set_ylim((1+affichage_proba)*mini, maxi*(1+affichage_proba))
    plt.pause(0.00001)  # pause avec duree en secondes
    time_i+=dt

print("end t_0")
bunch +=1
compteur = 0
nb_proton_temps_bunch_on = Poisson_bunch(nombre_ion_par_bunch,dt)
while time_i+dt<duree_bunch_on*1.E3:
    n_x -= nb_proton_temps_bunch_on[compteur]*np.ones(np.size(length))
    p_x += nb_proton_temps_bunch_on[compteur]*np.ones(np.size(length))
    line1.set_data(length, n_x)
    line2.set_data(length, p_x)
    maxi = np.max(n_x)
    maxi = np.maximum(np.max(p_x),maxi)
    mini = np.minimum(np.min(n_x), -1)
    mini = np.minimum(np.min(p_x), mini)
    print(mini,maxi)
    axes.set_ylim((1+affichage_proba)*mini, maxi*(1+affichage_proba))
    leg = [str(round(time_i, 2)) + " ps", "electron", "hole"]
    plt.legend(leg)
    plt.pause(0.00001)  # pause avec duree en secondes
    compteur+=1
    time_i += dt
    rho_x = evol_tx_rho_x(n_x,nt_x,p_x,pt_x)
    E_x,V_x = evol_x_potentiel(V_x,E_x,dx,rho_x)
    n_x = evol_t_n_x(n_x,nt_x,p_x,pt_x,E_x)
    p_x = evol_t_p_x(n_x,nt_x,p_x,pt_x,E_x)
    nt_x = evol_t_nt_x(n_x,nt_x,p_x)
    pt_x = evol_t_pt_x(n_x,p_x,pt_x)
print("end _bunch_on")
while time_i+dt<(duree_bunch_on+duree_bunch_off)*1.E3:
    line1.set_data(length, n_x)
    line2.set_data(length, p_x)
    leg = [str(round(time_i, 2)) + " ps","electron", "hole"]
    plt.legend(leg)
    maxi = np.max(n_x)
    maxi = np.maximum(np.max(p_x),maxi)
    mini = np.minimum(np.min(n_x), -1)
    mini = np.minimum(np.min(p_x), mini)
    axes.set_ylim((1+affichage_proba)*mini, maxi*(1+affichage_proba))
    plt.pause(0.00001)  # pause avec duree en secondes
    time_i += dt
    rho_x = evol_tx_rho_x(n_x,nt_x,p_x,pt_x)
    E_x,V_x = evol_x_potentiel(V_x,E_x,dx,rho_x)
    n_x = evol_t_n_x(n_x,nt_x,p_x,pt_x,E_x)
    p_x = evol_t_p_x(n_x,nt_x,p_x,pt_x,E_x)
    nt_x = evol_t_nt_x(n_x,nt_x,p_x)
    pt_x = evol_t_pt_x(n_x,p_x,pt_x)
print("end _bunch")
bunch +=1
compteur = 0
nb_proton_temps_bunch_on = Poisson_bunch(nombre_ion_par_bunch,dt)
while time_i+dt<(duree_bunch_on*2+duree_bunch_off)*1.E3 and time_i<=time_simu[-1]:
    n_x -= nb_proton_temps_bunch_on[compteur]*np.ones(np.size(length))
    p_x += nb_proton_temps_bunch_on[compteur] * np.ones(np.size(length))
    line1.set_data(length, n_x)
    line2.set_data(length, p_x)
    maxi = np.max(n_x)
    maxi = np.maximum(np.max(p_x),maxi)
    mini = np.minimum(np.min(n_x), -1)
    mini = np.minimum(np.min(p_x), mini)
    axes.set_ylim((1+affichage_proba)*mini, maxi*(1+affichage_proba))
    leg = [str(round(time_i, 2)) + " ps", "electron", "hole"]
    plt.legend(leg)
    plt.pause(0.00001)  # pause avec duree en secondes
    compteur+=1
    time_i += dt
    rho_x = evol_tx_rho_x(n_x,nt_x,p_x,pt_x)
    E_x,V_x = evol_x_potentiel(V_x,E_x,dx,rho_x)
    n_x = evol_t_n_x(n_x,nt_x,p_x,pt_x,E_x)
    p_x = evol_t_p_x(n_x,nt_x,p_x,pt_x,E_x)
    nt_x = evol_t_nt_x(n_x,nt_x,p_x)
    pt_x = evol_t_pt_x(n_x,p_x,pt_x)



plt.show()



