from pylab import *
import numpy as np
from scipy.signal import butter,lfilter
from scipy.integrate import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import *

# donnees diamants
polarisation_haut = 500. # V
polarisation_bas = 0. # V
rho_0 = 0. # C.cm^-1
mu_e = 2.23E2 # µm^2/V.ns
vsat_e = 0.813E2 # µm/ns
mu_h = 2.650E2 # µm^2/V.ns
vsat_h = 1.21E2 # µm/ns
epaisseur_diamant = 540. # µm
surface_irradiee = 7.1E-2 # cm^2
tau_n = 5. # ns
tau_nt = 5. # ns
tau_p = 5. # ns
tau_pt = 5. # ns
alpha_1 = 0.01  # µm^3.ns^-1
alpha_2 = 0.01  # µm^3.ns^-1
alpha_3 = 0.01  # µm^3.ns^-1

# donnees ion
energie_perdue_particule_simple = 3.2E3*epaisseur_diamant # eV
Flux_faisceau = 4.42E14 # protons/cm²/s
duree_bunch_on = 4. # ns
duree_bunch_off = 29. # ns
duree_train = 10. # µs
duree_inter_train = 100. # µs

# donnees simulation
h = 10. # ps   pas de temps
temps_debut = -1. # ns
temps_fin = 1. # ns
temperature = 300.
bins = 150


# constantes
eps_0 = 8.85E12 #F.m^-1
kB = 1.381E-23 # J.K^-1
q = 1.6E-19 # C
eps_r = 5.7 # sans unite
creation_paires_electron_trou = 13.6 # eV

Dn = mu_e*kB*temperature/q
Dp = mu_h*kB*temperature/q
champ_electrique_E = (polarisation_haut-polarisation_bas)/epaisseur_diamant # V/µm


def intensite_faisceau_crete(Flux,duree_bunch_ON,surface_irradiee):
    return(Flux*surface_irradiee*q)

def intensite_faisceau_bunch(Flux,duree_bunch_OFF,duree_bunch_ON,surface_irradiee):
    return(Flux*surface_irradiee*q)

def initialisation():
    length = np.linspace(-epaisseur_diamant/(bins), epaisseur_diamant*(1+1/(bins)), num=bins + 3)
    dx = (length[2] - length[1]) # um
    dt = h # ps
    time_simu = np.linspace(temps_debut, temps_fin, num=int((temps_fin - temps_debut) / h) + 1)

    intensite_diamant = intensite_faisceau_bunch(Flux_faisceau,duree_bunch_off,duree_bunch_on,surface_irradiee)

    densite_par_ion = energie_perdue_particule_simple/(creation_paires_electron_trou*epaisseur_diamant*(bins)) # .um^-1
    nombre_ion_par_bunch = (duree_bunch_off+duree_bunch_on)*1E-9*intensite_diamant/q

    n_x = np.ones(np.size(length))*densite_par_ion# µm^-1
    p_x = np.ones(np.size(length))*densite_par_ion # µm^-1
    nt_x = np.zeros(np.size(length))# µm^-1
    pt_x = np.zeros(np.size(length))# µm^-1
    E_x = np.ones(np.size(length))*(polarisation_haut-polarisation_bas)/epaisseur_diamant # V/um
    rho_x = q*(n_x+p_x+nt_x+pt_x) # C.m^-1

    return(length,dx,dt,time_simu,densite_par_ion,nombre_ion_par_bunch,n_x,p_x,nt_x,pt_x,E_x,rho_x)

length,dx,dt,time_simu,densite_par_ion,nombre_ion_par_bunch,n_x,p_x,nt_x,pt_x,E_x,rho_x=initialisation()

def evol_x_potentiel(E_x,dx,rho_x):
    for i in range(np.size(E_x)-1):
        E_x[i+1] = E_x[i] + dx*rho_x[i]/(eps_0*eps_r*1.E6) # V/um
    return(E_x)



figure()

line, = plt.plot(length,n_x*bins,label=r"$electrons$")
line2, = plt.plot(length,p_x*bins,label=r"$holes$")
line3, = plt.plot(length,nt_x,label=r"$electrons ~ trapped$")
line4, = plt.plot(length,pt_x,label=r"$holes ~ trapped$")
# line5, = plt.plot(length,rho_x,label=r" $\rho$")
plt.ylim([-0.1*densite_par_ion*bins,1.1*densite_par_ion*bins])
plt.xlabel(r"$Diamond ~ thickness (µm)$")
plt.ylabel(r"$Density (µm^{-3})$")


def f_n(t,n):
    dn_dt = np.zeros(np.size(n))
    champ_electrique = E_x
    if champ_electrique_E>0:
        for i in range(0,np.size(n)-1):
            deriv = (- n[i] * mu_e * champ_electrique[i] / (1 + mu_e * abs(champ_electrique[i]) / vsat_e) + n[i+1] * mu_e * champ_electrique[i]/ (1 + mu_e * abs(champ_electrique[i]) / vsat_e))
            diffusion = 0
            recombinaison = 0#- alpha_1*n[i]*pt_x[i] - alpha_2*n[i]*p_x[i]
            piegeage = 0#- n[i]/tau_n + nt_x[i]/tau_nt
            dn_dt[i] = deriv + diffusion + recombinaison + piegeage
        deriv0 = (- n[-1] * mu_e * champ_electrique[i] / (1 + mu_e * abs(champ_electrique[i]) / vsat_e))
        diffusion0 = 0
        recombinaison0 = 0#- alpha_1*n[-1]*pt_x[-1] - alpha_2*n[-1]*p_x[-1]
        piegeage0 = 0#- n[-1]/tau_n + nt_x[-1]/tau_nt
        dn_dt[-1] = deriv0 + diffusion0 + recombinaison0 + piegeage0
    else :
        for i in range(1,np.size(n)):
            deriv = - n[i] * mu_e * champ_electrique[i] / (1 + mu_e * abs(champ_electrique[i]) / vsat_e) + n[i+1] * mu_e * champ_electrique[i] / (1 + mu_e * abs(champ_electrique[i]) / vsat_e)
            diffusion = 0
            recombinaison = 0#- alpha_1*n[i]*pt_x[i] - alpha_2*n[i]*p_x[i]
            piegeage = 0#- n[i]/tau_n + nt_x[i]/tau_nt
            dn_dt[i] = deriv + diffusion + recombinaison + piegeage
        deriv0 = - n[0] * mu_e * champ_electrique[i] / (1 + mu_e * abs(champ_electrique[i]) / vsat_e)
        diffusion0 = 0
        recombinaison0 = 0#- alpha_1*n[0]*pt_x[0] - alpha_2*n[0]*p_x[0]
        piegeage0 = 0#- n[0]/tau_n + nt_x[0]/tau_nt
        dn_dt[0] = deriv0 + diffusion0 + recombinaison0 + piegeage0
    return dn_dt
def f_p(t,p):
    dp_dt = np.zeros(np.size(p))
    champ_electrique = E_x
    if champ_electrique_E<0:
        for i in range(1,np.size(p)):
            deriv = - p[i] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h) + p[i-1] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h)
            diffusion = 0
            recombinaison = 0#- alpha_3*nt_x[i]*p[i] - alpha_2*n_x[i]*p[i]
            piegeage = 0#- p[i]/tau_p + pt_x[i]/tau_pt
            dp_dt[i] = deriv + diffusion + recombinaison + piegeage
        deriv0 = - p[0] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h)
        diffusion0 = 0
        recombinaison0 = 0#- alpha_3*nt_x[0]*p[0] - alpha_2*n_x[0]*p[0]
        piegeage0 = 0#- p[0]/tau_p + pt_x[0]/tau_pt
        dp_dt[0] = deriv0 + diffusion0 + recombinaison0 + piegeage0

    else :
        for i in range(0,np.size(p)-1):
            deriv = - p[i] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h) + p[i-1] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h)
            diffusion = 0
            recombinaison = 0#- alpha_3*nt_x[i]*p[i] - alpha_2*n_x[i]*p[i]
            piegeage = 0#- p[i]/tau_p + pt_x[i]/tau_pt
            dp_dt[i] = deriv + diffusion + recombinaison + piegeage
        deriv0 = - p[-1] * mu_h * champ_electrique[i] / (1 + mu_h * abs(champ_electrique[i]) / vsat_h)
        diffusion0 = 0
        recombinaison0 = 0#- alpha_3*nt_x[-1]*p[-1] - alpha_2*n_x[-1]*p[-1]
        piegeage0 = 0#- p[-1]/tau_p + pt_x[-1]/tau_pt
        dp_dt[-1] = deriv0 + diffusion0 + recombinaison0 + piegeage0
    return dp_dt
def f_nt(t,n):
    dn_dt = np.zeros(np.size(n))
    for i in range(np.size(n)):
        dn_dt[i] = 0#-alpha_3*p_x[i]*n[i] + n_x[i]/tau_n -n[i]/tau_nt
    return dn_dt
def f_pt(t,p):
    dp_dt = np.zeros(np.size(p))
    for i in range(1,np.size(p)):
        dp_dt[i] =0#-alpha_3*p[i]*n_x[i] + p_x[i]/tau_p -p[i]/tau_pt
    return dp_dt

temps_final = 6
max_pas_temps = 0.01
sol = solve_ivp(fun=f_n, y0=n_x, t_span=[0,temps_final], max_step=max_pas_temps)
sol2 = solve_ivp(fun=f_p, y0=p_x, t_span=[0,temps_final], max_step=max_pas_temps)

amplification = 10**(43/20) # gain
resistance = 50 # Ohm
signal_electrode_haut = [0]
signal_electrode_bas = [0]
signal_trous_haut = [0]
signal_electrons_haut=[0]

for i in range(np.size(sol.t)):
    n_x = sol.y[:,i]
    p_x = sol2.y[:, i]
    rho_x = -n_x + p_x - nt_x + pt_x
    signal_electrode_haut.append((np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h+np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    signal_electrode_bas.append(-(np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h+np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    signal_trous_haut.append((np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h)*resistance*amplification*10**(9))
    signal_electrons_haut.append((np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    line.set_data(length,n_x*bins)
    line2.set_data(length,p_x*bins)
    line3.set_data(length, nt_x)
    line4.set_data(length, pt_x)
    # line5.set_data(length, rho_x)
    plt.legend()
    plt.title("Density of charge carrier in diamond : time=" + str(np.round(sol.t[i],2)) + "ns" )
    plt.pause(0.001)  # pause avec duree en secondes
sol.t = np.insert(sol.t,0,0)
plt.figure()
# plt.plot(sol.t,signal_electrode_bas,label="signal electrode down")
plt.plot(sol.t,signal_electrode_haut,label="signal electrode up")
plt.plot(sol.t,signal_electrons_haut,label="signal electrons up")
plt.plot(sol.t,signal_trous_haut,label="signal holes up")
plt.plot([0,temps_final],[0,0],color='black',linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.title("Diamond signal function of time")
plt.legend()
plt.show()







