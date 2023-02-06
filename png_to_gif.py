import os
import sys
from PIL import Image
import glob

def png_gif(path_png,path_gif,duration,entree_part):
    os.chdir(path_png)
    fGIF = "x_alpha_"+ str(entree_part[0])+".gif"
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
    os.chdir(path_gif)
    frames[0].save(fGIF, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration, loop=0)
    return()

pngf_gif()