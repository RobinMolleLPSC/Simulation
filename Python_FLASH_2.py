from pylab import *
import numpy as np
from scipy.signal import butter,lfilter
from scipy.integrate import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import *
import time
import os
import glob
import sys
from PIL import Image

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
tau_n = 40. # ns
tau_nt = 100. # ns
tau_p = 40 # ns
tau_pt = 100. # ns
alpha_1 = 0.1  # µm^3.ns^-1
alpha_2 = 0.1  # µm^3.ns^-1
alpha_3 = 0.1  # µm^3.ns^-1

# donnees ion
energie_perdue_particule_simple = 5.4E6 #3.2E3*epaisseur_diamant # eV
Flux_faisceau = 8.1E8 #3.525E9 # protons/cm²/s
duree_bunch_on = 4. # ns
duree_bunch_off = 29. # ns
duree_train = 10. # µs
duree_inter_train = 100. # µs

# donnees simulation
h = 0.1 # ns   pas de temps
temps_debut = -0.1 # ns
temps_fin = 12. # ns
temperature = 300. #K
bins = int(epaisseur_diamant)

temps_final = temps_fin
max_pas_temps = 0.4
amplification = 10**(46/20) # gain
resistance = 50 # Ohm

# constantes
eps_0 = 8.85418782E-12 #F.m^-1
kB = 1.381E-23 # J.K^-1
q = 1.602E-19 # C
eps_r = 5.7 # sans unite
creation_paires_electron_trou = 13.6 # eV

Dn = mu_e*kB*temperature/q
Dp = mu_h*kB*temperature/q
champ_electrique_E = (polarisation_haut-polarisation_bas)/epaisseur_diamant # V/µm

implantation = 13.

time_0 = time.time()

def intensite_faisceau_crete(Flux,surface_irradiee):
    return(Flux*surface_irradiee*q)
def intensite_faisceau_bunch(Flux,duree_bunch_OFF,duree_bunch_ON,surface_irradiee):
    return(Flux*surface_irradiee*q*duree_bunch_ON/(duree_bunch_OFF+duree_bunch_ON))
def initialisation():
    length = np.linspace(-epaisseur_diamant/(bins), epaisseur_diamant*(1+1/(bins)), num=bins + 2)
    dx = (length[2] - length[1]) # um
    dt = h # ps

    intensite_diamant = intensite_faisceau_bunch(Flux_faisceau,duree_bunch_off,duree_bunch_on,surface_irradiee)

    densite_par_ion = energie_perdue_particule_simple/(creation_paires_electron_trou*epaisseur_diamant*(bins)) # .um^-1
    nombre_ion_par_bunch = (duree_bunch_off+duree_bunch_on)*1E-9*intensite_diamant/q
    # print("nombre d'ions par bunch :",nombre_ion_par_bunch)

    n_x = np.ones(np.size(length))*densite_par_ion*nombre_ion_par_bunch# µm^-1
    for i in range(int(epaisseur_diamant-implantation)):
        n_x[i] = 0
    p_x = np.ones(np.size(length))*densite_par_ion*nombre_ion_par_bunch # µm^-1
    for i in range(int(epaisseur_diamant-implantation)):
        p_x[i] = 0
    nt_x = np.zeros(np.size(length))# µm^-1
    pt_x = np.zeros(np.size(length))# µm^-1
    E_x = np.ones(np.size(length))*(polarisation_haut-polarisation_bas)/epaisseur_diamant # V/um
    rho_x = q*(n_x+p_x+nt_x+pt_x) # C.m^-1

    Data = np.concatenate((n_x,p_x))
    Data = np.concatenate((Data,nt_x))
    Data = np.concatenate((Data,pt_x))
    Data = np.concatenate((Data,E_x))
    Data = np.concatenate((Data,rho_x))

    return(length,dx,dt,densite_par_ion,nombre_ion_par_bunch,Data)

length,dx,dt,densite_par_ion,nombre_ion_par_bunch,Data=initialisation()

fig1 = plt.figure()
line, = plt.plot(length,Data[0:bins+2]*bins,label=r"$electrons$")
line2, = plt.plot(length,Data[bins+2:2*bins+4]*bins,label=r"$holes$")
line3, = plt.plot(length,Data[2*bins+4:3*bins+6]*bins,label=r"$electrons ~ trapped$")
line4, = plt.plot(length,Data[3*bins+6:4*bins+8]*bins,label=r"$holes ~ trapped$")
# line6, = plt.plot(length,Data[4*bins+8:5*bins+10],label=r"$\vec{E}$")
line5, = plt.plot(length,Data[5*bins+10:]*bins,label=r"$\rho$")

plt.ylim([-0.1*densite_par_ion*nombre_ion_par_bunch*bins,1.1*densite_par_ion*nombre_ion_par_bunch*bins])
plt.xlabel(r"$Diamond ~ thickness ~ (µm)$")
plt.ylabel(r"$Density ~ (µm^{-3})$")

def dData_dt(t,Data):
    if int(t*100/temps_final)>int((t-max_pas_temps/100)*100/temps_final):
        print(str(int(t*100/temps_final)) + " %")
    n = Data[0:bins+2]
    p = Data[bins+2:2*bins+4]
    nt = Data[2*bins+4:3*bins+6]
    pt = Data[3*bins+6:4*bins+8]
    deri,diffu,pieg,recomb = 1, 1, 1, 1

    rho = np.zeros(np.size(length))
    for i in range(np.size(n)):
        rho[i] = q*(p[i]+pt[i]-n[i]-nt[i])

    E = Data[4*bins+8:5*bins+10]
    for i in range(np.size(E) - 1):
        # if E[i] + dx * rho[i] / (eps_0 * eps_r*1E-6)<10:
        #     E[i + 1] = E[i] + dx * rho[i] / (eps_0 * eps_r*1E-6)
        # else:
        #     E[i + 1] = E[i]
        E[i + 1] = E[i] + dx * rho[i] / (eps_0 * eps_r * 1E-6)
    dn_dt = np.zeros(np.size(n))
    for i in range(np.size(E)):
        if E[i]>0:
            if i == np.size(n)-1:
                if deri ==1:
                    deriv0 = (- n[-1] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e))
                else :
                    deriv0 = 0
                diffusion0 = 0
                if recomb == 1:
                    recombinaison0 = - alpha_1*n[-1]*pt[-1] - alpha_2*n[-1]*p[-1]
                else :
                    recombinaison0 = 0
                if pieg == 1:
                    piegeage0 = - n[-1]/tau_n + nt[-1]/tau_nt
                else :
                    piegeage0 = 0
                dn_dt[-1] = deriv0 + diffusion0 + recombinaison0 + piegeage0
            elif i == 0:
                if deri == 1 :
                    deriv = (- n[i] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e) + n[i + 1] * mu_e * E[i] / (
                            1 + mu_e * abs(E[i]) / vsat_e))
                else :
                    deriv = 0
                diffusion = 0
                if recomb == 1 :
                    recombinaison = - alpha_1 * n[i] * pt[i] - alpha_2 * n[i] * p[i]
                else :
                    recombinaison = 0
                if pieg == 1 :
                    piegeage = - n[i] / tau_n + nt[i] / tau_nt
                else :
                    piegeage = 0
                dn_dt[i] = deriv + diffusion + recombinaison + piegeage
            else:
                if deri == 1:
                    deriv = (- n[i] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e) + n[i + 1] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e))
                else :
                    deriv = 0
                if diffu == 1:
                    diffusion = -Dn*1/(dx**2)*(n[i+1]+n[i-1]-2*n[i])
                else :
                    diffusion = 0
                if recomb == 1:
                    recombinaison = - alpha_1*n[i]*pt[i] - alpha_2*n[i]*p[i]
                else :
                    recomb = 0
                if pieg == 1:
                    piegeage = - n[i]/tau_n + nt[i]/tau_nt
                else :
                    piegeage =0
                dn_dt[i] = deriv + diffusion + recombinaison + piegeage
        else:
            if i == 0:
                if deri == 1 :
                    deriv0 = - n[0] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e)
                else :
                    deriv0 = 0
                diffusion0 = 0
                if recomb == 1 :
                    recombinaison0 =  - alpha_1*n[0]*pt[0] - alpha_2*n[0]*p[0]
                else :
                    recombinaison0 = 0
                if pieg == 1 :
                    piegeage0 =  - n[0]/tau_n + nt[0]/tau_nt
                else :
                    piegeage0 = 0
                dn_dt[0] = deriv0 + diffusion0 + recombinaison0 + piegeage0
            elif i == np.size(n)-1:
                if deri == 1:
                    deriv = - n[i] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e) + n[i - 1] * mu_e * E[i] / (
                            1 + mu_e * abs(E[i]) / vsat_e)
                else :
                    deriv = 0
                diffusion = 0
                if recomb == 1 :
                    recombinaison = - alpha_1 * n[i] * pt[i] - alpha_2 * n[i] * p[i]
                else :
                    recombinaison = 0
                if pieg == 1:
                    piegeage = - n[i] / tau_n + nt[i] / tau_nt
                else :
                    piegeage = 0
                dn_dt[i] = deriv + diffusion + recombinaison + piegeage
            else:
                if deri == 1:
                    deriv = - n[i] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e) + n[i - 1] * mu_e * E[i] / (1 + mu_e * abs(E[i]) / vsat_e)
                else :
                    deriv = 0
                if diffu == 1:
                    diffusion = -Dn*1/(dx**2)*(n[i+1]+n[i-1]-2*n[i])
                else :
                    diffusion = 0
                if recomb == 1:
                    recombinaison = - alpha_1*n[i]*pt[i] - alpha_2*n[i]*p[i]
                else :
                    recombinaison = 0
                if pieg == 1:
                    piegeage = - n[i]/tau_n + nt[i]/tau_nt
                else :
                    piegeage = 0
                dn_dt[i] = deriv + diffusion + recombinaison + piegeage

    dp_dt = np.zeros(np.size(p))
    for i in range(np.size(E)):
        if E[i] < 0:
            if i == np.size(n) - 1:
                if deri == 1:
                    deriv0 = - p[-1] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h)
                else :
                    deriv0 = 0
                diffusion0 = 0
                if recomb == 1:
                    recombinaison0 =  - alpha_3*nt[-1]*p[-1] - alpha_2*n[-1]*p[-1]
                else :
                    recombinaison0 = 0
                if piegeage0 == 1:
                    piegeage0 =   - p[-1]/tau_p + pt[-1]/tau_pt
                else :
                    piegeage0 = 0
                dp_dt[-1] = deriv0 + diffusion0 + recombinaison0 + piegeage0
            elif i == 0:
                if deri == 1:
                    deriv = - p[i] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h) + p[i + 1] * mu_h * E[i] / (
                            1 + mu_h * abs(E[i]) / vsat_h)
                else :
                    deriv = 0
                diffusion = 0
                if recomb == 1:
                    recombinaison = - alpha_3 * nt[i] * p[i] - alpha_2 * n[i] * p[i]
                else :
                    recombinaison = 0
                if pieg == 1:
                    piegeage = - p[i] / tau_p + pt[i] / tau_pt
                else :
                    piegeage=0
                dp_dt[i] = deriv + diffusion + recombinaison + piegeage
            else:
                if deri == 1:
                    deriv = - p[i] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h) + p[i + 1] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h)
                else :
                    deriv = 0
                if diffu ==1:
                    diffusion = Dp*1/(dx**2)*(p[i+1]+p[i-1]-2*p[i])
                else :
                    diffusion = 0
                if recomb == 1:
                    recombinaison =  - alpha_3*nt[i]*p[i] - alpha_2*n[i]*p[i]
                else :
                    recombinaison = 0
                if pieg == 1:
                    piegeage =  - p[i]/tau_p + pt[i]/tau_pt
                else :
                    piegeage = 0
                dp_dt[i] = deriv + diffusion + recombinaison + piegeage
        else:
            if i == 0:
                if deri == 1:
                    deriv0 = - p[0] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h)
                else :
                    deriv0 = 0
                diffusion0 = 0
                if recomb == 1:
                    recombinaison0 = - alpha_3*nt[0]*p[0] - alpha_2*n[0]*p[0]
                else :
                    recombinaison0 = 0
                if pieg == 1:
                    piegeage0 = - p[0]/tau_p + pt[0]/tau_pt
                else :
                    piegeage0 = 0
                dp_dt[0] = deriv0 + diffusion0 + recombinaison0 + piegeage0
            elif i == i == np.size(n) - 1:
                if deri == 1:
                    deriv = - p[i] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h) + p[i - 1] * mu_h * E[i] / (
                            1 + mu_h * abs(E[i]) / vsat_h)
                else :
                    deriv = 0
                diffusion = 0
                if recomb == 1:
                    recombinaison = - alpha_3 * nt[i] * p[i] - alpha_2 * n[i] * p[i]
                else:
                    recombinaison = 0
                if pieg == 1:
                    piegeage = - p[i] / tau_p + pt[i] / tau_pt
                else :
                    piegeage = 0
                dp_dt[i] = deriv + diffusion + recombinaison + piegeage
            else:
                if deri == 1:
                    deriv = - p[i] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h) + p[i - 1] * mu_h * E[i] / (1 + mu_h * abs(E[i]) / vsat_h)
                else:
                    deriv = 0
                if diffu == 1:
                    diffusion = Dp*1/(dx**2)*(p[i+1]+p[i-1]-2*p[i])
                else :
                    diffusion = 0
                if recomb == 1:
                    recombinaison =  - alpha_3*nt[i]*p[i] - alpha_2*n[i]*p[i]
                else:
                    recombinaison = 0
                if pieg == 1:
                    piegeage =  - p[i]/tau_p + pt[i]/tau_pt
                else :
                    piegeage = 0
                dp_dt[i] = deriv + diffusion + recombinaison + piegeage
    dnt_dt = np.zeros(np.size(nt))
    for i in range(np.size(nt)):
        if recomb == 1:
            recombinaison = -alpha_3*p[i]*n[i]
        else :
            recombinaison = 0
        if pieg == 1:
            piegeage = n[i]/tau_n -nt[i]/tau_nt
        else :
            piegeage = 0
        dnt_dt[i] =  recombinaison + piegeage
    dpt_dt = np.zeros(np.size(pt))
    for i in range(np.size(pt)):
        if recomb == 1:
            recombinaison = -alpha_3 * p[i] * n[i]
        else:
            recombinaison = 0
        if pieg == 1:
            piegeage =p[i]/tau_p -pt[i]/tau_pt
        else:
            piegeage = 0
        dpt_dt[i] =  recombinaison + piegeage

    drho_dt = np.zeros(np.size(rho))
    dE_dt = np.zeros(np.size(rho))
    dData_dt_val = np.concatenate((dn_dt, dp_dt))
    dData_dt_val = np.concatenate((dData_dt_val, dnt_dt))
    dData_dt_val = np.concatenate((dData_dt_val, dpt_dt))
    dData_dt_val = np.concatenate((dData_dt_val, dE_dt))
    dData_dt_val = np.concatenate((dData_dt_val, drho_dt))
    return(dData_dt_val)

print("solve")
print("Etat simulation :")
sol = solve_ivp(fun=dData_dt, y0=Data,method='DOP853', t_span=[0,temps_final], max_step=max_pas_temps, vectorized=True)
print("end solve")

signal_electrode_haut = []
signal_electrode_bas = []
signal_trous_haut = []
signal_electrons_haut=[]

temps_before = np.linspace(temps_debut,0,10)
for i in range(np.size(temps_before)):
    n_x = np.zeros(np.size(length))
    p_x = np.zeros(np.size(length))
    nt_x = np.zeros(np.size(length))
    pt_x = np.zeros(np.size(length))
    rho_x = np.zeros(np.size(n_x))
    line.set_data(length, n_x * bins)
    line2.set_data(length, p_x * bins)
    line3.set_data(length, nt_x * bins)
    line4.set_data(length, pt_x * bins)
    line5.set_data(length, rho_x * bins)
    plt.legend(loc='upper right')
    plt.title("Density of charge carrier in diamond : time=" + str(np.round(temps_before[i], 2)) + "ns")
    fig1.savefig('C:\\Users\\molle\\Desktop\\png\\png_simu\\'+str(i).zfill(4)+'.png')
    plt.pause(0.01)  # pause avec duree en secondes
    signal_electrode_haut.append(0)
    signal_electrode_bas.append(0)
    signal_trous_haut.append(0)
    signal_electrons_haut.append(0)
for i in range(np.size(sol.t)):
    Data = sol.y[:,i]
    n_x = Data[0:bins + 2]
    p_x = Data[bins + 2:2 * bins + 4]
    nt_x = Data[2 * bins + 4:3 * bins + 6]
    pt_x = Data[3 * bins + 6:4 * bins + 8]
    rho_x = np.zeros(np.size(n_x))
    for j in range(np.size(n_x)):
        rho_x[j] = -n_x[j]  - nt_x[j]  + p_x[j]  + pt_x[j]
    E_x = Data[4*bins+8:5*bins+10]
    signal_electrode_haut.append((np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h+np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    signal_electrode_bas.append(-(np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h+np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    signal_trous_haut.append((np.dot(p_x[1:-1],E_x[1:-1])*q*mu_h)*resistance*amplification*10**(9))
    signal_electrons_haut.append((np.dot(n_x[1:-1],E_x[1:-1])*(-q)*mu_e*(-1))*resistance*amplification*10**(9))
    line.set_data(length,n_x*bins)
    line2.set_data(length,p_x*bins)
    line3.set_data(length, nt_x*bins)
    line4.set_data(length, pt_x*bins)
    line5.set_data(length, rho_x*bins)
    plt.legend(loc='upper right')
    plt.title("Density of charge carrier in diamond : time=" + str(np.round(sol.t[i], 2)) + "ns")
    fig1.savefig('C:\\Users\\molle\\Desktop\\png\\png_simu\\' + str(i+np.size(temps_before)).zfill(4) + '.png')
    plt.pause(0.01)  # pause avec duree en secondes

for i in range(np.size(temps_before)):
    sol.t = np.insert(sol.t,0,temps_before[np.size(temps_before)-1-i])
fig2 = plt.figure()
# plt.plot(sol.t,signal_electrode_bas,label="signal electrode down")
plt.plot(sol.t,signal_electrode_haut,label="signal electrode up")
plt.plot(sol.t,signal_electrons_haut,label="signal electrons up")
plt.plot(sol.t,signal_trous_haut,label="signal holes up")
plt.plot([0,temps_final],[0,0],color='black',linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (V)')
plt.title("Diamond signal function of time")
fig2.savefig('C:\\Users\\molle\\Desktop\\png\\image_simu\\diamant_TCT_courant.png')
plt.legend()



def png_gif(duration):
    os.chdir("C:\\Users\\molle\\Desktop\\png\\png_simu\\")
    fGIF = "diamant_TCT.gif"
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
    os.chdir("C:\\Users\\molle\\Desktop\\png\\Gif\\")

    frames[0].save(fGIF, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration, loop=0)
    return()

png_gif(duration=60)

time_1 = time.time()

print("Temps de simulation : " + str(int((time_1-time_0)//60)) + " min " + str(int((time_1-time_0)%60)) + " s ")

plt.show()







