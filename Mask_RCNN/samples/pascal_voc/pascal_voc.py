import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image

import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class PascalVOCConfig(Config):
    NAME = 'PascalVOC'

class PascalVOCDataset(utils.Dataset):
    def load_pascal_voc(self, dataset_dir, subset, class_map=None):
        metadata = scipy.io.loadmat(dataset_dir + '/' + '{}_metadata.mat'.format(subset))['metadata'][0][0]

        filenames = [f[0] for f in metadata[0][0]]
        object_mask_filenames = [f[0] for f in metadata[1][0]]
        instance_mask_filenames = [f[0] for f in metadata[2][0]]
        image_sizes = [(int(s[0][0][0][0][0]), int(s[0][0][1][0][0])) for s in metadata[3][0]]
        image_classes = [list(set([c [0] for c in metadata[4][0][i][0]])) for i in range(len(filenames))]

        pascal_voc_classes = ['aeroplane',
                              'bicycle',
                              'bird',
                              'boat',
                              'bottle',
                              'bus',
                              'car',
                              'cat',
                              'chair',
                              'cow',
                              'diningtable',
                              'dog',
                              'horse',
                              'motorbike',
                              'person',
                              'pottedplant',
                              'sheep',
                              'sofa',
                              'train',
                              'tvmonitor']

        classes_mapping = {(i + 1) : c for i, c in enumerate(pascal_voc_classes)}
        classes_mapping_inverse = {c : i for i, c in classes_mapping.items()}

        if not len(self.active_classes) == 0:
            # All classes with existing instances
            class_ids = list(classes_mapping.keys())
        else:
            class_ids = self.active_classes

        # Take images of corresponding classes
        image_ids = []
        for class_id in class_ids:
            image_ids.extend([i for i in range(len(filenames))
                if classes_mapping[class_id] in image_classes[i]])
        image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class('pascal_voc', i, classes_mapping[i])

        # Build annotations (list of classes for all images)
        annotations = []
        for i in image_ids:
            annotations.append({'class_index': [classes_mapping_inverse[c] for c in image_classes[i]]})

        # Add images
        for i in image_ids:
            self.add_image(
                    'pascal_voc', image_id=i,
                    path=filenames[i],
                    object_mask_path=object_mask_filenames[i], 
                    instance_mask_path=instance_mask_filenames[i], 
                    width=image_sizes[i][0],
                    height=image_sizes[i][1],
                    annotations=annotations[i])

    def load_mask(self, image_id):
        instance_masks = []
        class_ids = []

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        object_mask, instance_mask = loadPascalVOC(self.image_info[image_id]['object_mask_path'],
                                                   self.image_info[image_id]['instance_mask_path'])

        for instance_id in np.unique(instance_mask)[1:]: # skip 0 (background with no instances)
            instance_id_mask = (instance_mask == instance_id)
            if np.sum(instance_id_mask) < 1:
                continue
            instance_masks.append(instance_id_mask)
            class_ids.append(int(np.median(object_mask[instance_id_mask])))

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(PascalVOCDataset, self).load_mask(image_id)

def loadPascalVOC(object_filename, instance_filename):
    object_mask = np.array(Image.open(object_filename))
    object_mask[object_mask == 255] = 0

    instance_mask = np.array(Image.open(instance_filename))
    instance_mask[instance_mask == 255] = 0

    return object_mask, instance_mask

