import numpy as np
import scipy.io

import pickle

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

        filenames      = [folder[0] + '/' + filename[0] for folder, filename in zip(index[1][0], index[0][0])]
        objectnames    = [f[0] for f in index[6][0]]
        objectPresence = index[4].astype(np.int)

        with open(dataset_dir + '/image_sizes.pkl', 'rb') as f:
            image_sizes = pickle.load(f)

        if subset == 'train':
            filter_fn = lambda s: 'training' in s
        elif subset == 'val':
            filter_fn = lambda s: 'validation' in s
        else:
            raise Exception('Unknown subset: {}'.format(subset))

        idx = [i for i, f in enumerate(filenames) if filter_fn(f)]
        filenames      = [filenames[i] for i in idx]
        objectnames    = [objectnames[i] for i in idx]
        objectPresence = objectPresence[:, idx]

        if not class_ids:
            # All classes with existing instances
            class_ids = np.where(np.sum(objectPresence, 1) > 0)[0]

        if class_ids:
            # Take images of corresponding classes
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(np.where(objectPresence[class_id][0])))
            image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class('ade20k', i, objectnames[i])

        # Add images
        for i in image_ids:
            self.add_image(
                    'ade20k', image_id=i,
                    path=filenames[i],
                    width=image_size[filenames[i]][0],
                    height=image_size[filenames[i]][1])

    def load_mask(self, image_id):
        pass