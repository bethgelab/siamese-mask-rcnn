import numpy as np
import matplotlib.pyplot as plt

def loadADE20K(filename):
    """
    Python translation of the original MATLAB function.
    So far it only loads the semantic and instance segemntation masks.

    The outputs are
        - Object class masks (n x m)
        - Object instance masks (n x m)
    """
    fileseg = filename.replace('.jpg', '_seg.png')

    seg = plt.imread(fileseg)

    R, G, B = (seg[:,:,0], seg[:,:,1], seg[:,:,2])

    ObjectClassMasks = (R * 255 * 256 / 10 + G * 255).astype(np.int)

    _, Minstances_hat = np.unique(B.ravel(), return_inverse=True)
    ObjectInstanceMasks = np.reshape(Minstances_hat, B.shape).astype(np.int)

    return ObjectClassMasks, ObjectInstanceMasks

# loadAde20K('ADE20K_2016_07_26/images/training/a/airport_terminal/ADE_train_00000001.jpg')

