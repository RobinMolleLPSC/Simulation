import Interface

epsilon = 13.1 #eV
nombre_particule = 1
epaisseur_implantation = 0.014 #mm
energie_perdue = 5400000 #eV
epaisseur = 0.15 #mm
mu_e = 2200; # cm2/V.s
mu_h = 2640; # cm2/V.s
vsat_e = 0.821*10**7; # cm/s
vsat_h = 1.200*10**7; # cm/s
q = 1.6*10**-19 #C
delta_temps = 100. #ps
masse_effective = 9.1*10**-31 #kg
kB = 1.38 * 10 ** -23
paquet_paires = 2000.
nombre_de_paire = energie_perdue / epsilon
temperature = 300.
frequence_coupure_bas = 4.E6 #preampli cividec etc... #Hz
frequence_coupure_haut = 2.E9 #preampli cividec etc... #Hz
frequence_coupure_haut_Lecroy = 2.5*1E9 #Hz
freq_rc = 2.E8 # 1/(resistance_cividec*1E-12)
entree_particule = [0.51]

