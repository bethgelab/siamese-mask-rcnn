import os
import numpy as np
import matplotlib.pyplot as plt

def loadAde20K(filename):
    """
    Python translation of the original MATLAB function.

    The outputs are
        - Object class masks (n x m)
        - Object instance masks (n x m)
    """
    # Full obeject masks
    fileseg = filename.replace('.jpg', '_seg.png')

    seg = plt.imread(fileseg)

    R, G, B = (seg[:,:,0], seg[:,:,1], seg[:,:,2])

    ObjectClassMasks = (R * 255 * 256 / 10 + G * 255).astype(np.int)

    _, Minstances_hat = np.unique(B.ravel(), return_inverse=True)
    ObjectInstanceMasks = np.reshape(Minstances_hat, B.shape).astype(np.int)

    # Parts masks
    level = 0
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while True:
        level = level + 1
        file_parts = filename.replace('.jpg', '_parts_{}.png'.format(level))
        if os.path.isfile(file_parts):
            partsseg = plt.imread(file_parts);
            R, G, B = (partsseg[:,:,0], partsseg[:,:,1], partsseg[:,:,2])
            PartsClassMasks.append((R * 255 * 256 / 10 + G * 255).astype(np.int))
            _, Minstances_hat = np.unique(B.ravel(), return_inverse=True)
            PartsInstanceMasks.append(np.reshape(Minstances_hat, B.shape).astype(np.int))
        else:
            if len(PartsClassMasks) > 0:
                PartsClassMasks = np.stack(PartsClassMasks, -1)
                PartsInstanceMasks = np.stack(PartsInstanceMasks, -1)
            break

    return ObjectClassMasks, ObjectInstanceMasks, PartsClassMasks, PartsInstanceMasks

# loadAde20K('ADE20K_2016_07_26/images/training/b/bathroom/ADE_train_00000100.jpg')

