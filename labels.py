import numpy as np
import glob
import os
path = r'./Data'
nb=8
nb_of_images=1500
def lb(path):
    labels=os.listdir(path)[0:nb]
    return labels
labels=lb(path)
np.save('labes.npy', labels)