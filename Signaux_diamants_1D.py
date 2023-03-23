from pylab import *
import numpy as np
from scipy.signal import butter,lfilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import *


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
global Energie_perdue
global Energie_perdue_Entry
global Epaisseur_implantation
global Epaisseur_implantation_Entry
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
global entree_part
global energie_perdue
global epaisseur_implantation
global Epaisseur
global Epaisseur_Entry
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
global bruit_val
global Bruit_Entry
global Bruit
global Bruit_mean
global Bruit_mean_Entry
global Bruit_rms
global Bruit_rms_Entry
global ampli
global Amplificateur
global Amplificateur_Entry
global electrode
global Electrode
global Electrode_Entry
global Polarisation_haut
global Polarisation_bas
global Polarisation_haut_Entry
global Polarisation_bas_Entry

## Constantes
eps_0 = 8.854E-12  # permittivite du vide
eps = 13.1 #eV
nombre_particule = 1
epaisseur = 0.15 #mm
q = 1.6*10**-19 #C
masse_effective = 9.1*10**-31 #kg
kB = 1.38 * 10 ** -23


def Signaux_diamants_1D():
    plt.close("all")
    mu_e = float(Mobilite_electron_Entry.get())*0.0001
    mu_h = float(Mobilite_trou_Entry.get())*0.0001
    vsat_e = float(Vitesse_saturation_electron_Entry.get())*1E-8
    vsat_h = float(Vitesse_saturation_trou_Entry.get())*1E-8
    energie_deposee = float(Energie_perdue_Entry.get())
    epaisseur_implantation = float(Epaisseur_implantation_Entry.get())*1000

    frequence_coupure_bas = float(Freq_Cividec_haut_Entry.get())
    frequence_coupure_haut = float(Freq_Cividec_bas_Entry.get())
    frequence_coupure_haut_Lecroy = float(Freq_Lecroy_Entry.get())
    freq_rc = float(Freq_Cable_Entry.get())
    save2 = save.get()
    filtre2 = filtre.get()
    ampli2 = ampli.get()

    polarisation_haut = float(Polarisation_haut_Entry.get())
    polarisation_bas = float(Polarisation_bas_Entry.get())
    delta_temps = float(Pas_de_temps_Entry.get())
    nom_simu = nom_simu_entry.get()

    epaisseur = float(Epaisseur_diamant_Entry.get())*1000
    champ_electrique_interne = 0 ; #negatif si du haut vers le bas

    ## Electronique
    R_oscillo = 50.; # Ohm
    if ampli2 == "ON":
        amplification_cividec = 10**(54/20);
    else :
        amplification_cividec = 1

    electrode2 = electrode.get()
    if electrode2 == "haut":
        electrode_observee = -1 ; # 1 pour electrode du dessus, -1 pour electrode du dessous
    else :
        electrode_observee = 1

    #parametre affichage et calculs
    temps_fin = float(temps_final_Entry.get())
    temps_debut = -temps_fin/10. # ps
    bruit2 = bruit_val.get()
    if bruit2 == "ON":
        moyenne_bruit = float(Bruit_mean_Entry.get())
        rms_bruit = float(Bruit_rms_Entry.get())
    else :
        moyenne_bruit = 0.
        rms_bruit = 0.

    # #Dessin du diamant

    fig, ax = plt.subplots()
    ax.add_patch(patches.Rectangle((-2300,0),4600, 540,edgecolor = 'blue',facecolor = '#AEB6BF',fill=True))
    ax.add_patch(patches.Rectangle((-2200,540),4400, 0.1,edgecolor = 'orange',facecolor = 'orange',fill=True))
    ax.add_patch(patches.Rectangle((-2200,-0.1),4400, 0.1,edgecolor = 'orange',facecolor = 'orange',fill=True))

    plt.arrow(0, 1000, 0, -400, width=10, color='green')
    plt.text(30,750, r'Faisceau', fontsize=15, color='green')
    plt.arrow(-1000,540.1,0,500,width = 10,color = 'black')
    plt.text(-1300,1170, r'Polarisation haut', fontsize=10, color='black')
    plt.arrow(-1000,-0.1,0,-500,width = 10,color = 'black')
    plt.text(-1300,-700, r'Polarisation bas', fontsize=10, color='black')
    plt.arrow(-1600,500,0,-450,width = 10,color = 'black')
    plt.arrow(-1600,40,0,450,width = 10,color = 'black')
    plt.text(-1550,250, r'Epaisseur', fontsize=10, color='black')
    ax.set(xlim =[-2500,2500], ylim =[-1000,1500])
    plt.title("Schema du diamant")
    plt.xlabel("largeur en um")
    plt.ylabel("epaisseur en um")

    ## fonctions



    def temps(temps_debut,temps_final,delta_temps):
        return np.linspace(temps_debut,temps_final,int(floor(((temps_final-temps_debut)+1)/delta_temps)))
    def bruit(moyenne,rms,temps):
        return np.random.normal(moyenne,rms,len(temps))
    def champ_electrique():
        return champ_electrique_interne - (polarisation_haut - polarisation_bas) / epaisseur ;  # V/um
    if(champ_electrique()>0):
        plt.text(900, 220, r'$\overrightarrow{E}>0$', fontsize=15, color='#873600')
        plt.arrow(1250, 150, 0, 300, width=10, color='#873600')
    elif (champ_electrique()<0) :
        plt.text(900, 220, r'$\overrightarrow{E}<0$', fontsize=15, color='#873600')
        plt.arrow(1250, 450, 0, -300, width=10, color='#873600')
    else :
        print("ATTENTION, champ electrique nul")
    def drift_velocity_hole():
        return mu_h*champ_electrique()/(1+mu_h*abs(champ_electrique())/vsat_h)
    def drift_velocity_electron():
        return -mu_e * champ_electrique() / (1 + mu_e * abs(champ_electrique()) / vsat_e)
    def weighting_field():
        return electrode_observee * (1/epaisseur)
    def ne(temps_arrivee,energie_deposee,temps_i):
        v_drift_e = abs(drift_velocity_electron())
        if champ_electrique()<0:
            Te = epaisseur_implantation / v_drift_e
            collecte = 1
        if champ_electrique()>0:
            Te = epaisseur / v_drift_e
            collecte = 0
        if champ_electrique()==0:
            exit()
        if collecte==1:
            if -temps_arrivee + temps_i <= 0:
                n_e = 0
            elif (-temps_arrivee + temps_i)<Te:
                n_e = energie_deposee / eps * (1 - (temps_i-temps_arrivee) / (Te))
            elif (-temps_arrivee + temps_i)>=Te :
                n_e = 0
        if collecte==0:
            if -temps_arrivee + temps_i <= 0:
                n_e = 0
            elif (-temps_arrivee + temps_i)<=(Te - epaisseur_implantation/v_drift_e):
                n_e = energie_deposee / eps
            elif (-temps_arrivee + temps_i)<Te:
                n_e = energie_deposee / eps * (1 - (-temps_arrivee + temps_i - Te + epaisseur_implantation/v_drift_e )/ (epaisseur_implantation / v_drift_e))
            elif (-temps_arrivee + temps_i)>=Te :
                n_e = 0

        return n_e
    def nh(temps_arrivee,energie_deposee,temps_i):
        v_drift_h = abs(drift_velocity_hole())
        if champ_electrique()<0:
            Th = epaisseur / v_drift_h
            collecte = 1
        if champ_electrique()>0:
            Th = epaisseur_implantation / v_drift_h
            collecte = 0
        if champ_electrique()==0:
            exit()
        if collecte == 1:
            if -temps_arrivee + temps_i <= 0:
                n_h = 0
            elif (-temps_arrivee + temps_i) <= (Th-epaisseur_implantation/v_drift_h):
                n_h = energie_deposee / eps
            elif (-temps_arrivee + temps_i) < (Th):
                n_h = energie_deposee / eps * (1 - (-temps_arrivee + temps_i - (Th - epaisseur_implantation / v_drift_h)) / (epaisseur_implantation / v_drift_h))
            elif (-temps_arrivee + temps_i) >= Th:
                n_h = 0
        if collecte == 0:
            if -temps_arrivee + temps_i <= 0:
                n_h = 0
            elif (-temps_arrivee + temps_i) < Th:
                n_h = energie_deposee / eps * (1 - (temps_i-temps_arrivee)/Th)
            elif (-temps_arrivee + temps_i) >= Th:
                n_h = 0
        return n_h
    def intensite(temps_arrivee,energie_deposee,nombre_particule,temps_i):
        n_e = ne(temps_arrivee,energie_deposee,temps_i)
        n_h = nh(temps_arrivee, energie_deposee, temps_i)
        i_electron = -nombre_particule * 10 ** 12 * (n_e * (-q) * drift_velocity_electron() * weighting_field())
        i_trou = -nombre_particule * 10 ** 12 * (n_h * q * drift_velocity_hole() * weighting_field())
        i = i_trou + i_electron
        return [i, i_electron, i_trou]
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

    time = temps(temps_debut,temps_fin,delta_temps);
    fs = 1.E12/(time[1]-time[0])
    str1=[]
    part = 1

    #init
    d=figure()
    tension = []
    tension_reelle = []
    signal_trou = []
    signal_electron = []
    noise = bruit(moyenne_bruit, rms_bruit, time)

    #calcul des signaux
    for i in range(len(time)):
        tension_i = amplification_cividec*R_oscillo*intensite(temps_arrivee=0, energie_deposee=energie_deposee, nombre_particule=part, temps_i=time[i])[0] + noise[i]
        signal_electron.append(amplification_cividec*R_oscillo*intensite(temps_arrivee=0, energie_deposee=energie_deposee, nombre_particule=part, temps_i=time[i])[1])
        signal_trou.append(amplification_cividec * R_oscillo * intensite(temps_arrivee=0, energie_deposee=energie_deposee, nombre_particule=part, temps_i=time[i])[2])
        tension.append(tension_i+noise[i])
        tension_reelle.append(tension_i)

    #filtrage
    if filtre2 == "ON":
        cut_signal = butter_lowpass_filter(tension,freq_rc,fs,order=1)
        cut_signal = butter_bandpass_filter(cut_signal,frequence_coupure_bas,frequence_coupure_haut,fs,order=1)
        cut_signal = butter_lowpass_filter(cut_signal,frequence_coupure_haut_Lecroy,fs,order=1)

        #affichage
        plot(time/1000.,cut_signal)
        str1.append( str(nom_simu) + " signal filtre")
    plot(time/1000., signal_electron,'--')
    str1.append( str(nom_simu) + " signal electron")
    plot(time/1000., signal_trou,'--')
    str1.append( str(nom_simu) + " signal trou")
    plot(time/1000.,tension_reelle)
    str1.append( str(nom_simu) + " signal reel")

    print(time/1000.)
    print("cut_signal ", cut_signal)
    print("tension_reelle ",tension_reelle)

    xlabel("Temps en nanosecondes")
    ylabel("Amplitude en V")
    title("Signal oscilloscope de : "+str(nom_simu))
    legend(str1)

    if save2 == "ON":
        d.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\signal\\'+str(nom_simu)+'.png')

    show();


###################  Interface Graphique ########################
#
# window = Tk()
# window.title("PyCom-signal : simulation 1D du signal d'un ion traversant le diamant")
# window.geometry("1000x480")
#
#
# StudyType_Txt = Label(window, text = "Etude PyCom-signal : simulation 1D du signal d'un ion traversant le diamant: \n"
#                                      ,font="Helvetica", fg='black')
# StudyType_Txt.place (x=150, y=50)
# StudyType_Txt = Label(window, text = "Parametres :", fg='black')
# StudyType_Txt.place (x=480, y=80)
#
#
# Mobilite_electron = Label(window, text="Mobilite des electrons : [en cm2V.s] ", fg='blue')
# Mobilite_electron.place(x=50, y=100)
# Mobilite_electron_Entry = Entry(window, width=10)
# Mobilite_electron_Entry.place(x=400, y=100)
# Mobilite_electron_Entry.insert(0,"2200.")
#
# Mobilite_trou = Label(window, text="Mobilite des trous : [en cm2/V.s]", fg='red')
# Mobilite_trou.place(x=50,y=120)
# Mobilite_trou_Entry = Entry(window, width=10)
# Mobilite_trou_Entry.place(x=400,y=120)
# Mobilite_trou_Entry.insert(0,"2640.")
#
# Vitesse_saturation_electron = Label(window, text="Vitesse de saturation des electrons : [en cm/s] ", fg='blue')
# Vitesse_saturation_electron.place(x=50,y=140)
# Vitesse_saturation_electron_Entry = Entry(window, width=10)
# Vitesse_saturation_electron_Entry.place(x=400,y=140)
# Vitesse_saturation_electron_Entry.insert(0,"0.821E7")
#
# Vitesse_saturation_trou = Label(window, text="Vitesse de saturation des trous : [en cm/s] ", fg='red')
# Vitesse_saturation_trou.place(x=50,y=160)
# Vitesse_saturation_trou_Entry = Entry(window, width=10)
# Vitesse_saturation_trou_Entry.place(x=400,y=160)
# Vitesse_saturation_trou_Entry.insert(0,"1.2E7")
#
#
# Energie_perdue = Label(window, text="Energie perdue par la particule : [en eV] ", fg='black')
# Energie_perdue.place(x=50,y=200)
# Energie_perdue_Entry = Entry(window, width=10)
# Energie_perdue_Entry.place(x=400,y=200)
# Energie_perdue_Entry.insert(0,"5.4E6")
#
# Epaisseur_implantation = Label(window, text="Parcours de la particule dans le diamant : [en mm] ", fg='black')
# Epaisseur_implantation.place(x=50,y=220)
# Epaisseur_implantation_Entry = Entry(window, width=10)
# Epaisseur_implantation_Entry.place(x=400,y=220)
# Epaisseur_implantation_Entry.insert(0,"0.015")
#
# Epaisseur_diamant = Label(window, text="Epaisseur diamant : [en mm] ", fg='black')
# Epaisseur_diamant.place(x=50,y=240)
# Epaisseur_diamant_Entry = Entry(window, width=10)
# Epaisseur_diamant_Entry.place(x=400,y=240)
# Epaisseur_diamant_Entry.insert(0,"0.150")
#
# Polarisation_haut = Label(window, text="Polarisation face superieure : [en V] ", fg='black')
# Polarisation_haut.place(x=50,y=260)
# Polarisation_haut_Entry = Entry(window, width=10)
# Polarisation_haut_Entry.place(x=400,y=260)
# Polarisation_haut_Entry.insert(0,"150.")
#
# Polarisation_bas = Label(window, text="Polarisation face inferieure : [en V] ", fg='black')
# Polarisation_bas.place(x=50,y=280)
# Polarisation_bas_Entry = Entry(window, width=10)
# Polarisation_bas_Entry.place(x=400,y=280)
# Polarisation_bas_Entry.insert(0,"0.")
#
#
# Freq_Cividec_haut = Label(window, text="Frequence coupure passe haut Cividec : [en Hz] ", fg='green')
# Freq_Cividec_haut.place(x=50,y=320)
# Freq_Cividec_haut_Entry = Entry(window, width=10)
# Freq_Cividec_haut_Entry.place(x=400,y=320)
# Freq_Cividec_haut_Entry.insert(0,"4.E6")
#
# Freq_Cividec_bas = Label(window, text="Frequence coupure passe bas Cividec : [en Hz] ", fg='green')
# Freq_Cividec_bas.place(x=50,y=340)
# Freq_Cividec_bas_Entry = Entry(window, width=10)
# Freq_Cividec_bas_Entry.place(x=400,y=340)
# Freq_Cividec_bas_Entry.insert(0,"2.E9")
#
# Freq_Lecroy = Label(window, text="Frequence de coupure de l'oscilloscope : [en Hz] ", fg='green')
# Freq_Lecroy.place(x=50,y=360)
# Freq_Lecroy_Entry = Entry(window, width=10)
# Freq_Lecroy_Entry.place(x=400,y=360)
# Freq_Lecroy_Entry.insert(0,"2.5E9")
#
# Freq_Cable = Label(window, text="Frequence de coupure des cables : [en Hz] ", fg='green')
# Freq_Cable.place(x=50,y=380)
# Freq_Cable_Entry = Entry(window, width=10)
# Freq_Cable_Entry.place(x=400,y=380)
# Freq_Cable_Entry.insert(0,"5.E7")
#
#
#
#
# Pas_de_temps = Label(window, text="Pas de temps : [en ps] ", fg='black')
# Pas_de_temps.place(x=550, y=100)
# Pas_de_temps_Entry = Entry(window, width=10)
# Pas_de_temps_Entry.place(x=900, y=100)
# Pas_de_temps_Entry.insert(0,"40.")
#
# Temps_final = Label(window, text="Temps estime de derive : [en ps] ", fg='black')
# Temps_final.place(x=550, y=120)
# temps_final_Entry = Entry(window, width=10)
# temps_final_Entry.place(x=900, y=120)
# temps_final_Entry.insert(0,"7500.")
#
# Save_parameter = Label(window, text="Sauvegarde : ", fg='purple4')
# Save_parameter.place(x=550,y=160)
# save = StringVar(value="OFF")
# Save_parameter_Entry = Checkbutton(window,text="Active", variable=save, onvalue="ON", offvalue="OFF")
# Save_parameter_Entry.place(x=900,y=160)
#
# Electrode = Label(window, text="Electrode observee : [haut ou bas] ", fg='purple4')
# Electrode.place(x=550,y=180)
# electrode = StringVar(value="haut")
# Electrode_Entry = Checkbutton(window,variable=electrode, onvalue="haut", offvalue="bas",text=electrode.get())
# Electrode_Entry.place(x=900,y=180)
#
# Filtre = Label(window, text="Prise en compte de l'electronique : [Oui/Non] ", fg='purple4')
# Filtre.place(x=550,y=200)
# filtre = StringVar(value="OFF")
# Filtre_Entry = Checkbutton(window, variable=filtre, onvalue="ON", offvalue="OFF",text="Oui")
# Filtre_Entry.place(x=900,y=200)
#
# Bruit = Label(window, text="Bruit : [Oui/Non] ", fg='purple4')
# Bruit.place(x=550,y=220)
# bruit_val = StringVar(value="OFF")
# Bruit_Entry = Checkbutton(window, variable=bruit_val, onvalue="ON", offvalue="OFF",text="Oui")
# Bruit_Entry.place(x=900,y=220)
#
# Bruit_mean = Label(window, text="Moyenne du bruit : [en V] ", fg='purple4')
# Bruit_mean.place(x=550,y=240)
# Bruit_mean_Entry = Entry(window, width=10)
# Bruit_mean_Entry.place(x=900,y=240)
# Bruit_mean_Entry.insert(0,"0.")
#
# Bruit_rms = Label(window, text="RMS du bruit : [en V] ", fg='purple4')
# Bruit_rms.place(x=550,y=260)
# Bruit_rms_Entry = Entry(window, width=10)
# Bruit_rms_Entry.place(x=900,y=260)
# Bruit_rms_Entry.insert(0,"0.")
#
# Amplificateur = Label(window, text="Amplificateur : [Oui/Non] ", fg='purple4')
# Amplificateur.place(x=550,y=280)
# ampli = StringVar(value="OFF")
# Amplificateur_Entry = Checkbutton(window, variable=ampli, onvalue="ON", offvalue="OFF",text="Oui")
# Amplificateur_Entry.place(x=900,y=280)
#
#
#
#
#
# nom_exp = Label(window, text="Nom de la Simulation ", fg='red3')
# nom_exp.place(x=550,y=380)
# nom_simu_entry = Entry(window, width=43)
# nom_simu_entry.place(x=700, y=380)
#
#
#
# Validate = Button(window, bg="chartreuse", activebackground="darkgreen", relief="raised", text="Valider",
#                   command=lambda: Signaux_diamants_1D())
# Validate.place(x=333, y=430)
#
# QUITTER = Button(window, bg='orangered', activebackground="darkred", text="QUITTER", command=exit)
# QUITTER.place(x=666, y=430)


window = Tk()
window.title("PyDiam-signal:  ")
window.geometry("1000x480")


StudyType_Txt = Label(window, text = "Study PyDiam-signal: Simulation of the diamond signal with an incident particle: \n"
                                     ,font="Helvetica", fg='black')
StudyType_Txt.place (x=150, y=50)
StudyType_Txt = Label(window, text = "Parameters:", fg='black')
StudyType_Txt.place (x=480, y=80)


Mobilite_electron = Label(window, text="Electron mobility: [en cm2V.s] ", fg='blue')
Mobilite_electron.place(x=50, y=100)
Mobilite_electron_Entry = Entry(window, width=10)
Mobilite_electron_Entry.place(x=400, y=100)
Mobilite_electron_Entry.insert(0,"2200.")

Mobilite_trou = Label(window, text="Holes mobility: [en cm2/V.s]", fg='red')
Mobilite_trou.place(x=50,y=120)
Mobilite_trou_Entry = Entry(window, width=10)
Mobilite_trou_Entry.place(x=400,y=120)
Mobilite_trou_Entry.insert(0,"2640.")

Vitesse_saturation_electron = Label(window, text="Electrons saturation velocity: [en cm/s] ", fg='blue')
Vitesse_saturation_electron.place(x=50,y=140)
Vitesse_saturation_electron_Entry = Entry(window, width=10)
Vitesse_saturation_electron_Entry.place(x=400,y=140)
Vitesse_saturation_electron_Entry.insert(0,"0.821E7")

Vitesse_saturation_trou = Label(window, text="Holes saturation velocity: [en cm/s] ", fg='red')
Vitesse_saturation_trou.place(x=50,y=160)
Vitesse_saturation_trou_Entry = Entry(window, width=10)
Vitesse_saturation_trou_Entry.place(x=400,y=160)
Vitesse_saturation_trou_Entry.insert(0,"1.2E7")


Energie_perdue = Label(window, text="Energy loss by te particle: [en eV] ", fg='black')
Energie_perdue.place(x=50,y=200)
Energie_perdue_Entry = Entry(window, width=10)
Energie_perdue_Entry.place(x=400,y=200)
Energie_perdue_Entry.insert(0,"5.4E6")

Epaisseur_implantation = Label(window, text="Range of the particle: [en mm] ", fg='black')
Epaisseur_implantation.place(x=50,y=220)
Epaisseur_implantation_Entry = Entry(window, width=10)
Epaisseur_implantation_Entry.place(x=400,y=220)
Epaisseur_implantation_Entry.insert(0,"0.015")

Epaisseur_diamant = Label(window, text="Diamond thickness: [en mm] ", fg='black')
Epaisseur_diamant.place(x=50,y=240)
Epaisseur_diamant_Entry = Entry(window, width=10)
Epaisseur_diamant_Entry.place(x=400,y=240)
Epaisseur_diamant_Entry.insert(0,"0.150")

Polarisation_haut = Label(window, text="Voltage upper metallization: [en V] ", fg='black')
Polarisation_haut.place(x=50,y=260)
Polarisation_haut_Entry = Entry(window, width=10)
Polarisation_haut_Entry.place(x=400,y=260)
Polarisation_haut_Entry.insert(0,"150.")

Polarisation_bas = Label(window, text="Voltage lower metallization: [en V] ", fg='black')
Polarisation_bas.place(x=50,y=280)
Polarisation_bas_Entry = Entry(window, width=10)
Polarisation_bas_Entry.place(x=400,y=280)
Polarisation_bas_Entry.insert(0,"0.")


Freq_Cividec_haut = Label(window, text="Highpass filter preamplifier: [en Hz] ", fg='green')
Freq_Cividec_haut.place(x=50,y=320)
Freq_Cividec_haut_Entry = Entry(window, width=10)
Freq_Cividec_haut_Entry.place(x=400,y=320)
Freq_Cividec_haut_Entry.insert(0,"4.E6")

Freq_Cividec_bas = Label(window, text="Lowpass filter preamplifier: [en Hz] ", fg='green')
Freq_Cividec_bas.place(x=50,y=340)
Freq_Cividec_bas_Entry = Entry(window, width=10)
Freq_Cividec_bas_Entry.place(x=400,y=340)
Freq_Cividec_bas_Entry.insert(0,"2.E9")

Freq_Lecroy = Label(window, text="Lowpass filter oscilloscope: [en Hz] ", fg='green')
Freq_Lecroy.place(x=50,y=360)
Freq_Lecroy_Entry = Entry(window, width=10)
Freq_Lecroy_Entry.place(x=400,y=360)
Freq_Lecroy_Entry.insert(0,"2.5E9")

Freq_Cable = Label(window, text="Lowpass filter cable: [en Hz] ", fg='green')
Freq_Cable.place(x=50,y=380)
Freq_Cable_Entry = Entry(window, width=10)
Freq_Cable_Entry.place(x=400,y=380)
Freq_Cable_Entry.insert(0,"5.E7")




Pas_de_temps = Label(window, text="Time stamp: [en ps] ", fg='black')
Pas_de_temps.place(x=550, y=100)
Pas_de_temps_Entry = Entry(window, width=10)
Pas_de_temps_Entry.place(x=900, y=100)
Pas_de_temps_Entry.insert(0,"40.")

Temps_final = Label(window, text="Drift time estimated: [en ps] ", fg='black')
Temps_final.place(x=550, y=120)
temps_final_Entry = Entry(window, width=10)
temps_final_Entry.place(x=900, y=120)
temps_final_Entry.insert(0,"7500.")

Save_parameter = Label(window, text="Save? ", fg='purple4')
Save_parameter.place(x=550,y=160)
save = StringVar(value="OFF")
Save_parameter_Entry = Checkbutton(window,text="Active", variable=save, onvalue="ON", offvalue="OFF")
Save_parameter_Entry.place(x=900,y=160)

Electrode = Label(window, text="Observed electrode : [Up ou Down] ", fg='purple4')
Electrode.place(x=550,y=180)
electrode = StringVar(value="Up")
Electrode_Entry = Checkbutton(window,variable=electrode, onvalue="Up", offvalue="Down",text=electrode.get())
Electrode_Entry.place(x=900,y=180)

Filtre = Label(window, text="Readout electronic into account: [Yes/No] ", fg='purple4')
Filtre.place(x=550,y=200)
filtre = StringVar(value="OFF")
Filtre_Entry = Checkbutton(window, variable=filtre, onvalue="ON", offvalue="OFF",text="Oui")
Filtre_Entry.place(x=900,y=200)

Bruit = Label(window, text="Noise: [Yes/No] ", fg='purple4')
Bruit.place(x=550,y=220)
bruit_val = StringVar(value="OFF")
Bruit_Entry = Checkbutton(window, variable=bruit_val, onvalue="ON", offvalue="OFF",text="Oui")
Bruit_Entry.place(x=900,y=220)

Bruit_mean = Label(window, text="Noise mean: [en V] ", fg='purple4')
Bruit_mean.place(x=550,y=240)
Bruit_mean_Entry = Entry(window, width=10)
Bruit_mean_Entry.place(x=900,y=240)
Bruit_mean_Entry.insert(0,"0.")

Bruit_rms = Label(window, text="Noise rms: [en V] ", fg='purple4')
Bruit_rms.place(x=550,y=260)
Bruit_rms_Entry = Entry(window, width=10)
Bruit_rms_Entry.place(x=900,y=260)
Bruit_rms_Entry.insert(0,"0.")

Amplificateur = Label(window, text="Preamplifieur: [Yes/No] ", fg='purple4')
Amplificateur.place(x=550,y=280)
ampli = StringVar(value="OFF")
Amplificateur_Entry = Checkbutton(window, variable=ampli, onvalue="ON", offvalue="OFF",text="Oui")
Amplificateur_Entry.place(x=900,y=280)





nom_exp = Label(window, text="Name of the experiment", fg='red3')
nom_exp.place(x=550,y=380)
nom_simu_entry = Entry(window, width=43)
nom_simu_entry.place(x=700, y=380)



Validate = Button(window, bg="chartreuse", activebackground="darkgreen", relief="raised", text="Launch",
                  command=lambda: Signaux_diamants_1D())
Validate.place(x=333, y=430)

QUITTER = Button(window, bg='orangered', activebackground="darkred", text="QUIT", command=exit)
QUITTER.place(x=666, y=430)



window.mainloop()

