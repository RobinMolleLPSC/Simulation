
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy.signal import butter,lfilter
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
global Duree_intertrain
global Duree_intertrain_Entry
global Duree_train
global Duree_train_Entry
global Duree_bunch_on
global Duree_bunch_on_Entry
global Duree_bunch_off
global Duree_bunch_off_Entry
global Temps_debut
global Temps_debut_Entry
global Temps_fin
global Temps_fin_Entry
global I_faisceau
global I_faisceau_Entry
global Temps_montee
global Temps_montee_Entry
global Temps_descente
global Temps_descente_Entry

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
        amplification_cividec = 10**(45.6/20);
    else :
        amplification_cividec = 1

    electrode2 = electrode.get()
    if electrode2 == "haut":
        electrode_observee = -1 ; # 1 pour electrode du dessus, -1 pour electrode du dessous
    else :
        electrode_observee = 1

    #parametre affichage et calculs
    temps_fin = float(temps_final_Entry.get())
    temps_debut = 0. # ps
    bruit2 = bruit_val.get()
    if bruit2 == "ON":
        moyenne_bruit = float(Bruit_mean_Entry.get())*0.001
        rms_bruit = float(Bruit_rms_Entry.get())*0.001
    else :
        moyenne_bruit = 0.
        rms_bruit = 0.

    # #Dessin du diamant

    # fig, ax = plt.subplots()
    # ax.add_patch(patches.Rectangle((-2300,0),4600, 540,edgecolor = 'blue',facecolor = '#AEB6BF',fill=True))
    # ax.add_patch(patches.Rectangle((-2200,540),4400, 0.1,edgecolor = 'orange',facecolor = 'orange',fill=True))
    # ax.add_patch(patches.Rectangle((-2200,-0.1),4400, 0.1,edgecolor = 'orange',facecolor = 'orange',fill=True))
    #
    # plt.arrow(0, 1000, 0, -400, width=10, color='green')
    # plt.text(30,750, r'Faisceau', fontsize=15, color='green')
    # plt.arrow(-1000,540.1,0,500,width = 10,color = 'black')
    # plt.text(-1300,1170, r'Polarisation haut', fontsize=10, color='black')
    # plt.arrow(-1000,-0.1,0,-500,width = 10,color = 'black')
    # plt.text(-1300,-700, r'Polarisation bas', fontsize=10, color='black')
    # plt.arrow(-1600,500,0,-450,width = 10,color = 'black')
    # plt.arrow(-1600,40,0,450,width = 10,color = 'black')
    # plt.text(-1550,250, r'Epaisseur', fontsize=10, color='black')
    # ax.set(xlim =[-2500,2500], ylim =[-1000,1500])
    # plt.title("Schema du diamant")
    # plt.xlabel("largeur en um")
    # plt.ylabel("epaisseur en um")

    ## fonctions



    def temps(temps_debut,temps_final,delta_temps):
        return np.linspace(temps_debut,temps_final,int(floor(((temps_final-temps_debut)+1)/delta_temps)))
    def bruit(moyenne,rms,temps):
        return np.random.normal(moyenne,rms,len(temps))
    def champ_electrique():
        return champ_electrique_interne - (polarisation_haut - polarisation_bas) / epaisseur ;  # V/um
    # if(champ_electrique()>0):
    #     plt.text(900, 220, r'$\overrightarrow{E}>0$', fontsize=15, color='#873600')
    #     plt.arrow(1250, 150, 0, 300, width=10, color='#873600')
    # elif (champ_electrique()<0) :
    #     plt.text(900, 220, r'$\overrightarrow{E}<0$', fontsize=15, color='#873600')
    #     plt.arrow(1250, 450, 0, -300, width=10, color='#873600')
    # else :
    #     print("ATTENTION, champ electrique nul")
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


    time = temps(temps_debut,temps_fin,delta_temps);

    part = 1

    #init
    # d=figure()
    tension = []
    tension_reelle = []
    signal_trou = []
    signal_electron = []
    noise = bruit(moyenne_bruit, rms_bruit, time)

    #calcul des signaux
    for i in range(len(time)):
        tension_i = amplification_cividec*R_oscillo*intensite(temps_arrivee=0, energie_deposee=energie_deposee, nombre_particule=part, temps_i=time[i])[0]
        tension.append(tension_i)

    return(tension)
    #     #affichage
    #     plot(time/1000.,cut_signal)
    #     str1.append( str(nom_simu) + " signal filtre")
    # plot(time/1000., signal_electron,'--')
    # str1.append( str(nom_simu) + " signal electron")
    # plot(time/1000., signal_trou,'--')
    # str1.append( str(nom_simu) + " signal trou")
    # plot(time/1000.,tension_reelle)
    # str1.append( str(nom_simu) + " signal reel")
    #
    # xlabel("Temps en nanosecondes")
    # ylabel("Amplitude en V")
    # title("Signal oscilloscope de : "+str(nom_simu))
    # legend(str1)

    # if save2 == "ON":
    #     d.savefig('C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\2D-python-comsol\\Resultats\\signal\\'+str(nom_simu)+'.png')

    # show();

def Faisceau():
    plt.close('all')
    import time

    start = time.time()

    duree_on_bunch = float(Duree_bunch_on_Entry.get())  # ns
    duree_off_bunch = float(Duree_bunch_off_Entry.get()) # ns
    dt = float(Duree_train_Entry.get())  # us
    dit = float(Duree_intertrain_Entry.get())  # us
    intensite_faisceau_onD = float(I_faisceau_Entry.get())  # nA
    temps_debut = float(Temps_debut_Entry.get()) #us
    temps_final = float(Temps_fin_Entry.get()) #us
    pas_de_temps = float(Pas_de_temps_Entry.get()) # ps

    frequence_coupure_bas = float(Freq_Cividec_haut_Entry.get())
    frequence_coupure_haut = float(Freq_Cividec_bas_Entry.get())
    frequence_coupure_haut_Lecroy = float(Freq_Lecroy_Entry.get())
    freq_rc = float(Freq_Cable_Entry.get())

    temps_montee_HT = float(Temps_montee_Entry.get())
    temps_descente_HT = float(Temps_descente_Entry.get())

    time_t = np.linspace(temps_debut,temps_final,int((temps_final-temps_debut)*10**6/pas_de_temps))

    def bruit(moyenne,rms,temps):
        return np.random.normal(moyenne,rms,len(temps))

    bruit2 = bruit_val.get()
    if bruit2 == "ON":
        moyenne_bruit = float(Bruit_mean_Entry.get()) * 0.001
        rms_bruit = float(Bruit_rms_Entry.get()) * 0.001
    else:
        moyenne_bruit = 0.
        rms_bruit = 0.

    structure_faisceau = np.zeros(np.size(time_t))

    #bunch :
    for index_time_i in range(np.size(time_t)):
        if dit != 0:
            if time_t[index_time_i]%((duree_off_bunch+duree_on_bunch)*1E-3)<=duree_on_bunch*1E-3 and time_t[index_time_i]%(dit)<=dt:
                structure_faisceau[index_time_i] = intensite_faisceau_onD*10**-9
        else :
            if time_t[index_time_i] % ((duree_off_bunch + duree_on_bunch) * 1E-3) <= duree_on_bunch*1E-3:
                structure_faisceau[index_time_i] = intensite_faisceau_onD * 10 ** -9

    intensite_moyenne_faisceau = mean(structure_faisceau)
    intensite_moyenne_pulse = mean(intensite_faisceau_onD*10**-9*duree_on_bunch/duree_off_bunch)
    # plt.figure()
    # plt.plot(time*10**-12,structure_faisceau,label='Structure faisceau')
    # plt.plot([temps_debut*10**-12, temps_final*10**-12],[intensite_moyenne_faisceau,intensite_moyenne_faisceau],'--',color='red',label="Intensite moyenne faisceau")
    # plt.plot([temps_debut*10**-12, temps_final*10**-12], [intensite_moyenne_pulse,intensite_moyenne_pulse], '--', color='green',label='Intensite moyenne bunch')
    # plt.xlabel("Temps en s")
    # plt.ylabel("Intensite faisceau")
    # plt.legend()
    # plt.title("Structure faisceau : dt=" + str(dt) + "us et dit="+str(dit)+"us")


    lambda_nombre_proton_pas_de_temps = intensite_faisceau_onD * 10 ** -9 * (pas_de_temps) * 10 ** -12 / q
    nb_proton_par_bunch = np.zeros(np.size(time_t))
    nb_dt = 0
    for index_time_i in range(np.size(time_t)):
        if structure_faisceau[index_time_i]>0.:
            if dit==0:
                nb_proton_par_bunch[index_time_i] = np.random.poisson(lambda_nombre_proton_pas_de_temps, 1)
            else :
                if time_t[index_time_i]%dit<=dt:
                    nb_dt = time_t[index_time_i]//dit+1
                    nb_proton_par_bunch[index_time_i] = np.random.poisson(lambda_nombre_proton_pas_de_temps, 1)*(1-np.exp(-(time_t[index_time_i]-(nb_dt-1)*(dit))/temps_montee_HT))
        elif structure_faisceau[index_time_i]==0 and time_t[index_time_i]%dit>(dt) and time_t[index_time_i]%((duree_off_bunch+duree_on_bunch)*1E-3)<=duree_on_bunch*1E-3:
            if nb_dt==1:
                nb_proton_par_bunch[index_time_i] = np.random.poisson(lambda_nombre_proton_pas_de_temps, 1) * (np.exp(-(time_t[index_time_i]- dt )/ temps_descente_HT))*(1-np.exp(-dt/temps_montee_HT))
            if nb_dt>1:
                nb_proton_par_bunch[index_time_i] = np.random.poisson(lambda_nombre_proton_pas_de_temps, 1) * (np.exp(-(time_t[index_time_i]- dt - (nb_dt-1)*(dit)) / temps_descente_HT))*(1-np.exp(-dt/temps_montee_HT))

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

    somme = int(np.sum(nb_proton_par_bunch))
    print("nombre de protons : " + str(somme))
    signal_reel = bruit(moyenne_bruit,rms_bruit,time_t)
    num_proton = 0
    num_proton0 = 0
    indices_proton = [ idx for idx, element in enumerate(nb_proton_par_bunch) if element>0]
    signal_proton = Signaux_diamants_1D()
    for i in indices_proton:
        num_proton0 = num_proton
        num_proton+= nb_proton_par_bunch[i]
        if (num_proton*10//somme) != (num_proton0*10//somme) :
            print(str(int(num_proton)) + "/"+str(int(somme)) + "  -  " + str(int(num_proton*100//somme)) + "%")
        pas = 0
        for pas in range(np.size(signal_proton)):
            if i+pas<np.size(time_t):
                signal_reel[i+pas] += signal_proton[pas]*nb_proton_par_bunch[i]


    fs = 1.E6/(time_t[1]-time_t[0])
    filtre2 = filtre.get()
    ampli2 = ampli.get()
    if filtre2 == "ON":
        cut_signal = butter_lowpass_filter(signal_reel,freq_rc,fs,order=1)
        if ampli2 == "ON":
            cut_signal = butter_bandpass_filter(cut_signal,frequence_coupure_bas,frequence_coupure_haut,fs,order=1)
        cut_signal = butter_lowpass_filter(cut_signal,frequence_coupure_haut_Lecroy,fs,order=1)


    plt.figure()
    # plt.plot(time_t*10**-6,signal_reel,label="Signal sans electronique")
    if filtre2 == "ON":
        plt.plot(time_t*10**-6,cut_signal,label="Signal avec electronique")
    plt.legend()
    plt.xlabel("Temps en s")
    plt.ylabel("Signal diamant (V)")
    plt.title("Signal de diamant sous faisceau ARRONAX : i_faisceau=" + str(intensite_faisceau_onD) + "nA, dt=" + str(dt) + "us et dit=" + str(dit) + "us")

    plt.figure()
    for i in range(np.size(structure_faisceau)):
        if structure_faisceau[i]>0:
            structure_faisceau[i] = 1
    plt.plot(time_t * 10 ** -6, nb_proton_par_bunch, label='Nombre proton par bunch')
    plt.plot(time_t * 10 ** -6, structure_faisceau,'--', label='structure faisceau')
    plt.xlabel("Temps en s")
    plt.ylabel("Nombre de proton par pas de temps et structure faisceau")
    plt.title("Nombre proton par bunch : dt=" + str(dt) + "us et dit=" + str(dit) + "us")
    print("Affichage")

    end = time.time()
    print("temps total : " + str(round(end-start,2)) + "s")
    plt.show()




###################  Interface Graphique ########################

window = Tk()
window.title("PyCom-train : structure faisceau d'ARRONAX")
window.geometry("1000x650")


StudyType_Txt = Label(window, text = "Etude PyCom-train : simulation 1D de la structure faisceau d'ARRONAX: \n"
                                     ,font="Helvetica", fg='black')
StudyType_Txt.place (x=150, y=50)
StudyType_Txt = Label(window, text = "Parametres :", fg='black')
StudyType_Txt.place (x=480, y=80)


Mobilite_electron = Label(window, text="Mobilite des electrons : [en cm2V.s] ", fg='blue')
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


Energie_perdue = Label(window, text="Energie perdue par la particule : [en eV] ", fg='black')
Energie_perdue.place(x=50,y=200)
Energie_perdue_Entry = Entry(window, width=10)
Energie_perdue_Entry.place(x=400,y=200)
Energie_perdue_Entry.insert(0,"1.7E6")

Epaisseur_implantation = Label(window, text="Parcours de la particule dans le diamant : [en mm] ", fg='black')
Epaisseur_implantation.place(x=50,y=220)
Epaisseur_implantation_Entry = Entry(window, width=10)
Epaisseur_implantation_Entry.place(x=400,y=220)
Epaisseur_implantation_Entry.insert(0,"0.150")

Epaisseur_diamant = Label(window, text="Epaisseur diamant : [en mm] ", fg='black')
Epaisseur_diamant.place(x=50,y=240)
Epaisseur_diamant_Entry = Entry(window, width=10)
Epaisseur_diamant_Entry.place(x=400,y=240)
Epaisseur_diamant_Entry.insert(0,"0.150")

Polarisation_haut = Label(window, text="Polarisation face superieure : [en V] ", fg='black')
Polarisation_haut.place(x=50,y=260)
Polarisation_haut_Entry = Entry(window, width=10)
Polarisation_haut_Entry.place(x=400,y=260)
Polarisation_haut_Entry.insert(0,"150.")

Polarisation_bas = Label(window, text="Polarisation face inferieure : [en V] ", fg='black')
Polarisation_bas.place(x=50,y=280)
Polarisation_bas_Entry = Entry(window, width=10)
Polarisation_bas_Entry.place(x=400,y=280)
Polarisation_bas_Entry.insert(0,"0.")


Freq_Cividec_haut = Label(window, text="Frequence coupure passe haut Cividec : [en Hz] ", fg='green')
Freq_Cividec_haut.place(x=50,y=320)
Freq_Cividec_haut_Entry = Entry(window, width=10)
Freq_Cividec_haut_Entry.place(x=400,y=320)
Freq_Cividec_haut_Entry.insert(0,"4.E6")

Freq_Cividec_bas = Label(window, text="Frequence coupure passe bas Cividec : [en Hz] ", fg='green')
Freq_Cividec_bas.place(x=50,y=340)
Freq_Cividec_bas_Entry = Entry(window, width=10)
Freq_Cividec_bas_Entry.place(x=400,y=340)
Freq_Cividec_bas_Entry.insert(0,"2.E9")

Freq_Lecroy = Label(window, text="Frequence de coupure de l'oscilloscope : [en Hz] ", fg='green')
Freq_Lecroy.place(x=50,y=360)
Freq_Lecroy_Entry = Entry(window, width=10)
Freq_Lecroy_Entry.place(x=400,y=360)
Freq_Lecroy_Entry.insert(0,"2.5E9")

Freq_Cable = Label(window, text="Frequence de coupure des cables : [en Hz] ", fg='green')
Freq_Cable.place(x=50,y=380)
Freq_Cable_Entry = Entry(window, width=10)
Freq_Cable_Entry.place(x=400,y=380)
Freq_Cable_Entry.insert(0,"3.E8")




Pas_de_temps = Label(window, text="Pas de temps : [en ps] ", fg='black')
Pas_de_temps.place(x=550, y=100)
Pas_de_temps_Entry = Entry(window, width=10)
Pas_de_temps_Entry.place(x=900, y=100)
Pas_de_temps_Entry.insert(0,"40.")

Temps_final = Label(window, text="Temps estime de derive : [en ps] ", fg='black')
Temps_final.place(x=550, y=120)
temps_final_Entry = Entry(window, width=10)
temps_final_Entry.place(x=900, y=120)
temps_final_Entry.insert(0,"10000.")

Save_parameter = Label(window, text="Sauvegarde : ", fg='purple4')
Save_parameter.place(x=550,y=160)
save = StringVar(value="OFF")
Save_parameter_Entry = Checkbutton(window,text="Active", variable=save, onvalue="ON", offvalue="OFF")
Save_parameter_Entry.place(x=900,y=160)

Electrode = Label(window, text="Electrode observee : [haut ou bas] ", fg='purple4')
Electrode.place(x=550,y=180)
electrode = StringVar(value="bas")
Electrode_Entry = Checkbutton(window,variable=electrode, onvalue="haut", offvalue="bas",text=electrode.get())
Electrode_Entry.place(x=900,y=180)

Filtre = Label(window, text="Prise en compte de l'electronique : [Oui/Non] ", fg='purple4')
Filtre.place(x=550,y=200)
filtre = StringVar(value="ON")
Filtre_Entry = Checkbutton(window, variable=filtre, onvalue="ON", offvalue="OFF",text="Oui")
Filtre_Entry.place(x=900,y=200)

Bruit = Label(window, text="Bruit : [Oui/Non] ", fg='purple4')
Bruit.place(x=550,y=220)
bruit_val = StringVar(value="OFF")
Bruit_Entry = Checkbutton(window, variable=bruit_val, onvalue="ON", offvalue="OFF",text="Oui")
Bruit_Entry.place(x=900,y=220)

Bruit_mean = Label(window, text="Moyenne du bruit : [en mV] ", fg='purple4')
Bruit_mean.place(x=550,y=240)
Bruit_mean_Entry = Entry(window, width=10)
Bruit_mean_Entry.place(x=900,y=240)
Bruit_mean_Entry.insert(0,"0.")

Bruit_rms = Label(window, text="RMS du bruit : [en mV] ", fg='purple4')
Bruit_rms.place(x=550,y=260)
Bruit_rms_Entry = Entry(window, width=10)
Bruit_rms_Entry.place(x=900,y=260)
Bruit_rms_Entry.insert(0,"0.")

Amplificateur = Label(window, text="Amplificateur : [Oui/Non] ", fg='purple4')
Amplificateur.place(x=550,y=280)
ampli = StringVar(value="OFF")
Amplificateur_Entry = Checkbutton(window, variable=ampli, onvalue="ON", offvalue="OFF",text="Oui")
Amplificateur_Entry.place(x=900,y=280)


Duree_train = Label(window, text="Duree train (dt) : [en us] ", fg='orange')
Duree_train.place(x=550,y=340)
Duree_train_Entry = Entry(window, width=10)
Duree_train_Entry.place(x=900,y=340)
Duree_train_Entry.insert(0,"40.")

Duree_intertrain = Label(window, text="Duree intertrain (dit) : [en us] ", fg='orange')
Duree_intertrain.place(x=550,y=360)
Duree_intertrain_Entry = Entry(window, width=10)
Duree_intertrain_Entry.place(x=900,y=360)
Duree_intertrain_Entry.insert(0,"100.")

Duree_bunch_on = Label(window, text="Duree bunch ON : [en ns] ", fg='orange')
Duree_bunch_on.place(x=550,y=380)
Duree_bunch_on_Entry = Entry(window, width=10)
Duree_bunch_on_Entry.place(x=900,y=380)
Duree_bunch_on_Entry.insert(0,"4.")

Duree_bunch_off = Label(window, text="Duree bunch OFF : [en ns] ", fg='orange')
Duree_bunch_off.place(x=550,y=400)
Duree_bunch_off_Entry = Entry(window, width=10)
Duree_bunch_off_Entry.place(x=900,y=400)
Duree_bunch_off_Entry.insert(0,"29.")

Temps_debut = Label(window, text="Temps debut : [en us] ", fg='orange')
Temps_debut.place(x=550,y=420)
Temps_debut_Entry = Entry(window, width=10)
Temps_debut_Entry.place(x=900,y=420)
Temps_debut_Entry.insert(0,"-5.")

Temps_fin = Label(window, text="Temps fin : [en us] ", fg='orange')
Temps_fin.place(x=550,y=440)
Temps_fin_Entry = Entry(window, width=10)
Temps_fin_Entry.place(x=900,y=440)
Temps_fin_Entry.insert(0,"100.")

I_faisceau = Label(window, text="Intensite faisceau sur diamant : [nA] ", fg='orange')
I_faisceau.place(x=550,y=460)
I_faisceau_Entry = Entry(window, width=10)
I_faisceau_Entry.place(x=900,y=460)
I_faisceau_Entry.insert(0,"0.0046")

Temps_montee = Label(window, text="Temps montee HT: [en us] ", fg='orange')
Temps_montee.place(x=550,y=480)
Temps_montee_Entry = Entry(window, width=10)
Temps_montee_Entry.place(x=900,y=480)
Temps_montee_Entry.insert(0,"1.")

Temps_descente = Label(window, text="Temps descente HT : [uA] ", fg='orange')
Temps_descente.place(x=550,y=500)
Temps_descente_Entry = Entry(window, width=10)
Temps_descente_Entry.place(x=900,y=500)
Temps_descente_Entry.insert(0,"1.")


nom_exp = Label(window, text="Nom de la Simulation ", fg='red3')
nom_exp.place(x=550,y=530)
nom_simu_entry = Entry(window, width=43)
nom_simu_entry.place(x=700, y=530)



Validate = Button(window, bg="chartreuse", activebackground="darkgreen", relief="raised", text="Valider",
                  command=lambda: Faisceau())
Validate.place(x=333, y=590)

QUITTER = Button(window, bg='orangered', activebackground="darkred", text="QUITTER", command=exit)
QUITTER.place(x=666, y=590)



window.mainloop()

