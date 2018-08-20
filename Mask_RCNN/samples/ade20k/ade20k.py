import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
    NUM_CLASSES = 1 + 2693

class ADE20KDataset(utils.Dataset):
    def load_ade20k(self, dataset_dir, subset, class_map=None):
        self.class_map = class_map
        index = scipy.io.loadmat(dataset_dir + '/index_ade20k.mat')['index'][0][0]

        filenames_relative = [folder[0] + '/' + filename[0] for folder, filename in zip(index[1][0], index[0][0])]
        filenames          = [dataset_dir + '/' + '/'.join(f.split('/')[1:]) for f in filenames_relative]
        objectnames        = ['dummy_background'] + [f[0] for f in index[6][0]]
        objectPresence     = np.load(dataset_dir + '/' + 'objectPresence.npy')
        # objectPresence     = index[4] > 0
        # objectPresence     = (index[4] > 0) & (index[3] == 0) # subtract the object parts

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
        objectPresence = objectPresence[:, idx]
        
        # Add dummy background class
        # objectPresence = np.concatenate([np.zeros((1, objectPresence.shape[1])), objectPresence])

        if len(self.active_classes) == 0:
            # All classes with existing instances
            class_ids = list(np.where(np.sum(objectPresence, 1) > 0)[0])
        else:
            class_ids = list(np.intersect1d(self.active_classes, np.where(np.sum(objectPresence, 1) > 0)[0]))

        self.class_ids_with_holes = class_ids

        # Take images of corresponding classes
        image_ids = []
        for class_id in class_ids:
            image_ids.extend(list(np.where(objectPresence[class_id])[0]))
        image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class('ade20k', i, objectnames[i])

        # Build annotations (list of classes for all images)
        self.annotations = {}
        for i in image_ids:
            image_classes = np.unique(np.where(objectPresence[:,i] > 0)[0])
            self.annotations[i] = {'class_index': image_classes}

        def invert_class_map(class_map):
            inverse_class_map = {}
            for k, v in class_map.items():
                for c in v:
                    inverse_class_map[c] = k
            return inverse_class_map

        if self.class_map is not None:
            self.inverse_class_map = invert_class_map(self.class_map)
            mapped_classes    = sum(list(class_map.values()), [])
            mapped_class_ids  = list(filter(lambda x: x in mapped_classes, class_ids))
            mapped_image_ids  = [i for i in image_ids if len(np.intersect1d(self.annotations[i]['class_index'], mapped_class_ids)) > 0]

            mapped_annotations = {}
            for i in mapped_image_ids:
                mapped_image_classes = [self.inverse_class_map[c] for c in self.annotations[i]['class_index'] if c in mapped_classes]
                mapped_annotations[i] = {'class_index': mapped_image_classes}
            
            class_ids = [c for c in class_map.keys() if len(np.intersect1d(class_map[c], mapped_class_ids)) > 0]
            image_ids = mapped_image_ids
            self.annotations = mapped_annotations

        # Add images
        for i in image_ids:
            self.add_image(
                    'ade20k', image_id=i,
                    path=filenames[i],
                    width=image_sizes[filenames_relative[i]][0],
                    height=image_sizes[filenames_relative[i]][1],
                    annotations=self.annotations[i])

    def load_mask(self, image_id):
        def cluster_classes(mask, class_ids, class_map, inverse_class_map):
            objects_to_keep = [i for i, c in enumerate(class_ids) if c in inverse_class_map]
            mask = mask[:,:,objects_to_keep]
            class_ids = [inverse_class_map[class_ids[i]] for i in objects_to_keep]
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids

        instance_masks = []
        class_ids = []

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        object_mask, instance_mask, parts_object_mask, parts_instance_mask = loadADE20K(self.image_info[image_id]['path'])

        for instance_id in np.unique(instance_mask)[1:]:
            instance_id_mask = (instance_mask == instance_id)
            if np.sum(instance_id_mask) < 1:
                continue
            instance_masks.append(instance_id_mask)
            class_ids.append(int(np.median(object_mask[instance_id_mask])))

        if len(parts_object_mask) > 0:
            _, _, n_levels = parts_object_mask.shape
            for level in range(n_levels):
                for instance_id in np.unique(parts_instance_mask[:,:,level])[1:]:
                    instance_id_mask = (parts_instance_mask[:,:,level] == instance_id)
                    if np.sum(instance_id_mask) < 1:
                        continue
                    instance_masks.append(instance_id_mask)
                    class_ids.append(int(np.median(parts_object_mask[:,:,level][instance_id_mask])))

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            if self.class_map is not None:
                mask, class_ids = cluster_classes(mask, class_ids, self.class_map, self.inverse_class_map)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ADE20KDataset, self).load_mask(image_id)

def loadADE20K(filename):
    """
    Python translation of the original MATLAB function.

    The outputs are
        - Object class masks (n x m)
        - Object instance masks (n x m)
    """
    # Full object masks
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
