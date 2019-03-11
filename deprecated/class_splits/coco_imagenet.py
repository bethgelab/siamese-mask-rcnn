
import numpy as np
import wordnet_functions as wf


def get_ms_coco_categories():
    """Return the MS COCO categories from a file."""

    coco_categories = []

    with open("MS_COCO_categories.txt") as f:
        for line in f:
            l = line.replace("\n", "")
            coco_categories.append(l)

    return coco_categories

def compare_coco_imagenet():
    """Return, for each MS COCO category, the number of categ. in ImageNet."""

    categories = get_ms_coco_categories()
    nums = []

    for c in categories:
        num = wf.num_hypernyms_in_ilsvrc2012(str(c))
        nums.append(num)

    return nums

