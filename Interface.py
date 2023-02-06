from tkinter import *

import Derive_2D

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


window = Tk()
window.title("PyCom : simulation de chambre d'ionisation solide diamant")
window.geometry("920x650")

StudyType_Txt = Label(window, text = "Etude PyCom de la dérive des charges dans la chambre d'ionisation solide en diamant\n\n"
                                     "Parametres :", fg='black')
StudyType_Txt.place (x=100, y=50)

Mobilite_electron = Label(window, text="Mobilite des electrons : [en cm²/V.s] ", fg='blue')
Mobilite_electron.place(x=100, y=100)
Mobilite_electron_Entry = Entry(window, width=10)
Mobilite_electron_Entry.place(x=400, y=100)
Mobilite_electron_Entry.insert(0,"2200.")
global mu_e
mu_e = Mobilite_electron_Entry.get()

Mobilite_trou = Label(window, text="Mobilite des trous : [en cm²/V.s]", fg='red')
Mobilite_trou.place(x=100,y=120)
Mobilite_trou_Entry = Entry(window, width=10)
Mobilite_trou_Entry.place(x=400,y=120)
Mobilite_trou_Entry.insert(0,"2640.")
global mu_h
mu_h = Mobilite_trou_Entry.get()

Vitesse_saturation_electron = Label(window, text="Vitesse de saturation des electrons : [en cm/s] ", fg='blue')
Vitesse_saturation_electron.place(x=100,y=140)
Vitesse_saturation_electron_Entry = Entry(window, width=10)
Vitesse_saturation_electron_Entry.place(x=400,y=140)
Vitesse_saturation_electron_Entry.insert(0,"0.821E7")
global vsat_e
vsat_e = Vitesse_saturation_electron_Entry.get()

Vitesse_saturation_trou = Label(window, text="Vitesse de saturation des trous : [en cm/s] ", fg='red')
Vitesse_saturation_trou.place(x=100,y=160)
Vitesse_saturation_trou_Entry = Entry(window, width=10)
Vitesse_saturation_trou_Entry.place(x=400,y=160)
Vitesse_saturation_trou_Entry.insert(0,"1.2E7")
global vsat_h
vsat_h = Vitesse_saturation_trou_Entry.get()


Pas_de_temps = Label(window, text="Pas de temps : [en ps] ", fg='black')
Pas_de_temps.place(x=100, y=200)
Pas_de_temps_Entry = Entry(window, width=10)
Pas_de_temps_Entry.place(x=400, y=200)
Pas_de_temps_Entry.insert(0,"40.")
global delta_temps
delta_temps = Pas_de_temps_Entry.get()

Paquets_charges = Label(window, text="Paquets de charges : [nombre]", fg='black')
Paquets_charges.place(x=100,y=220)
Paquets_charges_Entry = Entry(window, width=10)
Paquets_charges_Entry.place(x=400,y=220)
Paquets_charges_Entry.insert(0,"10000.")
global paquet_paires
paquet_paires = Paquets_charges_Entry.get()

Entree_particule = Label(window, text="Abscisse d'entree de la particule : [en mm] ", fg='black')
Entree_particule.place(x=100,y=240)
Entree_particule_Entry = Entry(window, width=10)
Entree_particule_Entry.place(x=400,y=240)
Entree_particule_Entry.insert(0,"0.5")
global entree_part
entree_part = [Entree_particule_Entry.get()]

Energie_perdue = Label(window, text="Energie perdue par la particule : [en eV] ", fg='black')
Energie_perdue.place(x=100,y=260)
Energie_perdue_Entry = Entry(window, width=10)
Energie_perdue_Entry.place(x=400,y=260)
Energie_perdue_Entry.insert(0,"5.4E6")
global energie_perdue
energie_perdue = Energie_perdue_Entry.get()

Epaisseur_implantation = Label(window, text="Parcours de la particule dans le diamant : [en mm] ", fg='black')
Epaisseur_implantation.place(x=100,y=280)
Epaisseur_implantation_Entry = Entry(window, width=10)
Epaisseur_implantation_Entry.place(x=400,y=280)
Epaisseur_implantation_Entry.insert(0,"0.015")
global epaisseur_implantation
epaisseur_implantation = Epaisseur_implantation_Entry.get()

position_particule_haut_bas = Label(window, text="Incidence de la particule : [haut ou bas] ", fg='black')
position_particule_haut_bas.place(x=100,y=300)
position_particule_haut_bas_Entry = Entry(window, width=10)
position_particule_haut_bas_Entry.place(x=400,y=300)
position_particule_haut_bas_Entry.insert(0,"haut")
global entree
entree = position_particule_haut_bas_Entry.get()

Dispersion_angulaire = Label(window, text="Angle par rapport à la normale (dispersion) : [en °] ", fg='black')
Dispersion_angulaire.place(x=100,y=320)
Dispersion_angulaire_Entry = Entry(window, width=10)
Dispersion_angulaire_Entry.place(x=400,y=320)
Dispersion_angulaire_Entry.insert(0,"0.5")
global disp_angle
disp_angle = Dispersion_angulaire_Entry.get()

Temperature = Label(window, text="Temperature : [en K] ", fg='black')
Temperature.place(x=100,y=340)
Temperature_Entry = Entry(window, width=10)
Temperature_Entry.place(x=400,y=340)
Temperature_Entry.insert(0,"300")
global temp
temp = Temperature_Entry.get()


Freq_Cividec_haut = Label(window, text="Frequence coupure passe bas Cividec : [en Hz] ", fg='green')
Freq_Cividec_haut.place(x=100,y=380)
Freq_Cividec_haut_Entry = Entry(window, width=10)
Freq_Cividec_haut_Entry.place(x=400,y=380)
Freq_Cividec_haut_Entry.insert(0,"4.E6")
global frequence_coupure_bas
frequence_coupure_bas = Freq_Cividec_haut_Entry.get()

Freq_Cividec_bas = Label(window, text="Frequence coupure passe haut Cividec : [en Hz] ", fg='green')
Freq_Cividec_bas.place(x=100,y=400)
Freq_Cividec_bas_Entry = Entry(window, width=10)
Freq_Cividec_bas_Entry.place(x=400,y=400)
Freq_Cividec_bas_Entry.insert(0,"2.E9")
global frequence_coupure_haut
frequence_coupure_haut = Freq_Cividec_bas_Entry.get()

Freq_Lecroy = Label(window, text="Fréquence de coupure de l'oscilloscope : [en Hz] ", fg='green')
Freq_Lecroy.place(x=100,y=420)
Freq_Lecroy_Entry = Entry(window, width=10)
Freq_Lecroy_Entry.place(x=400,y=420)
Freq_Lecroy_Entry.insert(0,"2.5E9")
global frequence_coupure_haut_Lecroy
frequence_coupure_haut_Lecroy = Freq_Lecroy_Entry.get()

Freq_Cable = Label(window, text="Fréquence de coupure de l'oscilloscope : [en Hz] ", fg='green')
Freq_Cable.place(x=100,y=440)
Freq_Cable_Entry = Entry(window, width=10)
Freq_Cable_Entry.place(x=400,y=440)
Freq_Cable_Entry.insert(0,"2.E8")
global freq_rc
freq_rc = Freq_Cable_Entry.get()


global Save_parameter
global Save_parameter_Entry

global Modele
global Modele_Entry

global Affichage_derive
global Affichage_derive_Entry

global Affichage_intensite
global Affichage_intensite_Entry

global Filtre
global Filtre_Entry


Save_parameter = Label(window, text="Sauvegarde : ", fg='black')
Save_parameter.place(x=100,y=500)
global save
save = StringVar(value="OFF")
Save_parameter_Entry = Checkbutton(window, width=35, variable=save, onvalue="ON", offvalue="OFF")
Save_parameter_Entry.place(x=375,y=500)

Modele = Label(window, text="Modele de vitesse de derive : [Canali ou simplifie] ", fg='black')
Modele.place(x=100,y=520)
global model
model = StringVar(value="Canali")
Modele_Entry = Checkbutton(window, width=40,variable=model, onvalue="Canali", offvalue="Autre",text=model.get())
Modele_Entry.place(x=375,y=520)

Affichage_derive = Label(window, text="Affichage de la derive des charges : [Oui/Non] ", fg='black')
Affichage_derive.place(x=100,y=540)
global deriv
deriv = StringVar(value="ON")
Affichage_derive_Entry = Checkbutton(window, width=35,variable=deriv, onvalue="ON", offvalue="OFF",text="Oui")
Affichage_derive_Entry.place(x=375,y=540)

Affichage_intensite = Label(window, text="Affichage de l'intensite : [Oui/Non] ", fg='black')
Affichage_intensite.place(x=100,y=560)
global intensite
intensite = StringVar(value="ON")
Affichage_intensite_Entry = Checkbutton(window, width=35,variable=intensite, onvalue="ON", offvalue="OFF",text="Oui")
Affichage_intensite_Entry.place(x=375,y=560)

Filtre = Label(window, text="Prise en compte de l'electronique : [Oui/Non] ", fg='black')
Filtre.place(x=100,y=580)
global fitre
filtre = StringVar(value="ON")
Filtre_Entry = Checkbutton(window, width=38,variable=filtre, onvalue="ON", offvalue="OFF",text="Oui")
Filtre_Entry.place(x=375,y=580)

Validate = Button(window, text = "Validate",command=main.main2)
Validate.place(x=200, y=640)

STOP = Button(window, text = "STOP", command = exit)
STOP.place(x=400, y=600)

window.mainloop()