# ouverture des fichiers comsol et creation des tableaux

import os


adresse_dossier = "C:\\Users\\molle\\Documents\\LPSC_Stage\\Simulations\\COMSOL\\Modelisation2D\\PyCom_2striphaut_1bas_150um_150Vhaut_interstrip_100um\\"
file_champ_electrique = "Champ_electrique"
file_weighting_field_1 = "Weighting_field_1"
file_weighting_field_2 ="Weighting_field_2"
file_weighting_field_3 = "Weighting_field_bas"
file_champ_electrique2 = "Champ_electrique_4strip"
file_weighting_field_1_4 = "Weighting_field_1_4strip"
file_weighting_field_2_4 ="Weighting_field_2_4strip"
file_weighting_field_3_4 = "Weighting_field_3_4strip"
file_weighting_field_4_4 ="Weighting_field_4_4strip"
file_weighting_field_down_4 = "Weighting_field_down_4strip"


def ouverture(file):
    x = []  # premiere ligne : coordonnee x
    y = []
    Ex = []
    Ey = []
    os.chdir(adresse_dossier)
    with open(file + ".txt", 'r') as f:
        compteur_ligne = -1
        for line in f:
            compteur_ligne += 1

            # x
            if compteur_ligne == 9:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        x.append(float(nb))
            # y
            if compteur_ligne == 10:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        y.append(float(nb))
            # champ Ex
            if compteur_ligne >= 13 and compteur_ligne <= 162:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        Ex.append(float(line[caractere - taille_nombre:caractere]))

            # champ Ey
            if compteur_ligne >= 165 and compteur_ligne <= 314:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        Ey.append(float(line[caractere - taille_nombre:caractere]))
    return(x,y,Ex,Ey)

def ouverture2(file):
    x = []  # premiere ligne : coordonnee x
    y = []
    Ex = []
    Ey = []
    os.chdir(adresse_dossier)
    with open(file + ".txt", 'r') as f:
        compteur_ligne = -1
        for line in f:
            compteur_ligne += 1
            # x
            if compteur_ligne == 9:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        x.append(float(nb))
            # y
            if compteur_ligne == 10:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        y.append(float(nb))
            # champ Ex
            if compteur_ligne >= 13 and compteur_ligne <= 1012:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        nb.replace(" ", "")
                        nb.replace("\n", "")
                        if nb == "NaN":
                            Ex.append(0)
                        else:
                            Ex.append(float(line[caractere - taille_nombre:caractere]))

            # champ Ey
            if compteur_ligne >= 1015 and compteur_ligne <= 2014:
                caractere = 0
                while caractere < len(line) - 1:
                    if line[caractere] == ' ':
                        while caractere < len(line) - 1 and line[caractere] == ' ':
                            caractere += 1
                    else:
                        taille_nombre = 0
                        while caractere < len(line) - 1 and line[caractere] != ' ':
                            caractere += 1
                            taille_nombre += 1
                        # print(line[caractere-taille_nombre:caractere])
                        nb = line[caractere - taille_nombre:caractere]
                        if nb == "NaN":
                            Ey.append(0)
                        else:
                            Ey.append(float(line[caractere - taille_nombre:caractere]))
    return(x,y,Ex,Ey)

x_maillage,y_maillage,Ex,Ey = ouverture(file_champ_electrique)
_,_,Ewx1,Ewy1 = ouverture(file_weighting_field_1)
_,_,Ewx2,Ewy2 = ouverture(file_weighting_field_2)
_,_,Ewx3,Ewy3 = ouverture(file_weighting_field_3)
# x_maillage2,y_maillage2,Ex2,Ey2 = ouverture2(file_champ_electrique2)
# _,_,Ewx1,Ewy1 = ouverture2(file_weighting_field_1_4)
# _,_,Ewx2,Ewy2 = ouverture2(file_weighting_field_2_4)
# _,_,Ewx3,Ewy3 = ouverture2(file_weighting_field_3_4)
# _,_,Ewx4,Ewy4 = ouverture2(file_weighting_field_4_4)
# _,_,Ewx5,Ewy5 = ouverture2(file_weighting_field_down_4)

# z = np.zeros(( len(x_maillage),len(y_maillage)))
# for i in range(len(x_maillage)):
#     for j in range(len(y_maillage)):
#         z[i][j] = np.sqrt(Ewx1[i*150+j]**2+Ewy1[i*150+j]**2)
#         z[i][j] = -Ewy2[i*150+j]
# #
# plot = plt.pcolormesh(x_maillage, y_maillage, z, cmap='rainbow', shading='auto',vmax=20000)#,norm=SymLogNorm(linthresh=100, linscale=0.001))    #norm=SymLogNorm(linthresh=100, linscale=0.001))
# #
# # contour needs the centers
# # cset = plt.contour(x, y,z, cmap='gray')
# # plt.clabel(cset, inline=True)
# plt.title("Champ Weighting")
# plt.xlabel("Length in mm")
# plt.ylabel("Width in um")
#
# plt.colorbar(plot)
# #
# plt.show()


