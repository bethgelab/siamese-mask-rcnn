import numpy as np
import scipy.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class ADE20KConfig(Config):
    NAME = 'ADE20K'
    IMAGES_PER_GPU = 2
    # GPU_count = 8
    NUM_CLASSES = 1 + 3147

class ADE20KDataset(util.Dataset):
    def load_ade20k(self, dataset_dir, subset, class_ids=None, class_map=None):
        index = scipy.io.loatmat(dataset_dir + '/index_ade20k.mat')['index'][0][0]

        filenames = [f[0] for f in index[0][0]]
