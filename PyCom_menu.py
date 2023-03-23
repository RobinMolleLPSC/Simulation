from tkinter import *
import os
import glob


############## Variables #############

global Filtre_Entry
global fitre
global parametre

def run_Derive_2D():
    os.system('python Derive_2D.py')
def run_Signaux_diamants_1D():
    os.system('python Signaux_diamants_1D.py')
def run_Train_simu():
    os.system('python Train_simu.py')

###################  Interface Graphique ########################


window_menu = Tk()
window_menu.title("PyCom : Simulation de detecteur diamant")
window_menu.geometry("500x300")

StudyType_Txt = Label(window_menu, text = "PyCom : Simulation de detecteur diamant: \n"
                                     ,font="Helvetica", fg='black')
StudyType_Txt.place (x=50, y=50)

PyComDrift = Button(window_menu, bg="chartreuse", activebackground="darkgreen", relief="raised", text="PyCom-drift (2D)",
                  command=lambda: run_Derive_2D())
PyComDrift.place(x=150, y=100)

PyComSignal = Button(window_menu, bg="orange", activebackground="orangered", relief="raised", text="PyCom-signal (1D)",
                  command=lambda: run_Signaux_diamants_1D())
PyComSignal.place(x=150, y=150)

PyComSignal = Button(window_menu, bg="aqua", activebackground="dodgerblue", relief="raised", text="PyCom-Train (1D)",
                  command=lambda: run_Train_simu())
PyComSignal.place(x=150, y=200)


window_menu.mainloop()


