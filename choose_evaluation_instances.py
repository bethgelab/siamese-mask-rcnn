import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import utils as siamese_utils
from IPython import display

def run_insatances_selection(dataset, config, save_folder):
    image_ids = dataset.image_ids

    plt.figure(figsize=(10, 5))

    for i, image_id in enumerate(image_ids):
        coco_image_id = dataset.image_info[image_id]["id"]
        
        save_filename = os.path.join(save_folder, '{}.pkl'.format(coco_image_id))
        if os.path.isfile(save_filename):
            continue
        
        dataset_instances = {}
        image = dataset.load_image(image_id)
        
        gt_class_ids = dataset.load_mask(image_id)[1]
        if not np.any(gt_class_ids > 0):
                continue
                
        categories = np.unique(gt_class_ids)
        _idx = categories > 0
        categories = categories[_idx]
        
        for category in categories:
            while True:
                # try:
                #     target, target_size = siamese_utils.get_one_target(category, dataset, config, return_original_size=True)
                # except:
                #     print('Error extracting target. Imade ID {}, category {}'.format(image_id, category))
                    
                target, target_size = siamese_utils.get_one_target(category, dataset, config, return_original_size=True)
                if max(target_size[:2]) < 20:
                    continue

                plt.subplot(1,2,1)
                plt.imshow(target)
                plt.title(dataset.class_names[category], fontsize=18)
                plt.axis('off')

                plt.subplot(1,2,2)
                plt.imshow(image)
                plt.title('Input image', fontsize=18)
                plt.axis('off')

                display.display(plt.gcf())
                
                target_chosen = False

                while True:
                    user_input = input()

                    if user_input == 'n':
                        target_chosen = False
                        display.clear_output()
                        break
                    elif user_input == 'y':
                        print('Accepted. Imade ID {}, category {}'.format(coco_image_id, category))
                        dataset_instances[category] = target
                        target_chosen = True
                        display.clear_output()
                        break
                    else:
                        print('Type y/n to accept/reject the target')
                        
                if target_chosen:
                    break
                    
        with open(save_filename, 'wb') as f:
            pickle.dump(dataset_instances, f)
