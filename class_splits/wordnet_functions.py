
import numpy as np
from nltk.corpus import wordnet as wn
from shutil import copyfile
import os
import linecache as lc
# import pydnn.utility as ut

def get_filenames_of_category(category, image_labels_path, categories):
    """Return a list of filenames of all images belonging to a category.

    category - a string specifying a (perhaps broad) category
    image_labels_path - a filepath to a file with all image labels,
                        formatted in the ilsvrc2012 format
    categories - a list of all categories of the dataset. The order of
                 categories has to be the same as used for the labelling.

    """
  
    # get indices of all subcategories that belong to the category
    subcategories_list = []
    counter = 0
    for c in categories:
        if is_hypernym(c, category):
            subcategories_list.append(counter)
        counter += 1


    image_list = []
    with open(image_labels_path) as labels_file:
        for line in labels_file:
            image_name, image_label = line.split(" ")

            if int(image_label) in subcategories_list:
                image_list.append(image_name)

    return image_list


def hypernyms_in_ilsvrc2012_categories(entity):
    """Return all hypernyms of categories.txt for a given entity.

    entity - a string, e.g. "furniture"
  
    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in categories.txt (the imagenet categories).
    If the entity itself is contained, it will be returned as well.
    """

    return get_hypernyms("categories.txt", entity)

    
def get_hypernyms(categories_file, entity):
    """Return all hypernyms of categories for a given entity.

    entity - a string, e.g. "furniture"

    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the categories.
    If the entity itself is contained, it will be returned as well.
    """

    hypers = []
    with open(categories_file) as f:
        for line in f:
            category = get_category_from_line(line)
            cat_synset = wn.synsets(category)[0]
            if is_hypernym(category, entity):
                hypers.append(category)

    return hypers


def get_ilsvrc2012_training_WNID(entity):
    """Return a WNID for each hypernym of entity.

    entity - a string, e.g. "furniture"

    Returns the WNIDs of the children of the entity,
    e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the ilsvrc2012 categories.
    If the entity itself is contained, it will be returned as well.
    """

    results = []

    hypernyms = hypernyms_in_ilsvrc2012_categories(entity)
    
    for hyper in hypernyms:

        with open("WNID_synsets_mapping.txt") as f:
            for line in f:
                category = get_category_from_line(line)

                if category == hyper:
                    print(line[:9])
                    results.append(line[:9])

    return results


def make_directory_for_category(category, directory, print_status):
    """Create a directory for category and store all category images there.

    cagegory - the (perhaps large) category of interest
    directory - the path of the target directory's PARENT directory,
                where all the images should be stored
                The directory will then be created and named after category.
    print_status - a boolean whether you'd like to get a status update
    """   

    subcategories = get_ilsvrc2012_training_WNID(category)

    target_directory = os.path.join(directory, category)   

    if not os.path.exists(target_directory):
        os.mkdir(target_directory)


    if print_status:
        print("---COPYING for category "+category+": STARTED---")

    for image in os.listdir(directory):
        if image[:9] in subcategories:
            if print_status:
                print("copying: "+image)
            copyfile(os.path.join(directory, image),
                     os.path.join(target_directory, image))
    if print_status:
        print("---COPYING DONE---")


def num_hypernyms_in_ilsvrc2012(entity):
    """Return number of hypernyms in the ilsvrc2012 categories for entity."""

    return len(hypernyms_in_ilsvrc2012_categories(entity))


def num_images_for_entity(entity, do_print=False):
    """Return the number of images that belong to category/entity."""
 
    categories = get_ilsvrc2012_categories();
    counter = 0

    with open("image_labels.txt") as file:
        for line in file.readlines():
            label_num = int(line.split(" ")[1])
            
            category_of_label = categories[label_num]
            if is_hypernym(category_of_label, entity):
                if do_print:
                    print(category_of_label)
                counter += 1

    return counter


def is_hypernym(entity1, entity2):
    """Return whether entity1 is a (kind of) entity2."""
    
    synset1 = wn.synsets(entity1)
    synset2 = wn.synsets(entity2)
    if len(synset1) == 0:
        print("entity1: "+str(entity1)+" is not a synset!")
        return False
    if len(synset2) == 0:
        print("entity2: "+str(entity2)+" is not a synset!")
        return False

    synset1 = synset1[0]
    synset2 = synset2[0]
    hypoentity2 = set([i for i in synset2.closure(
                                  lambda s:s.hyponyms())])
    
    return (synset1 in hypoentity2) or (synset1 is synset2)


def get_ilsvrc2012_categories():
    """Return the first item of each synset of the ilsvrc2012 categories."""

    categories = []

    with open("categories.txt") as f:
        for line in f:
           categories.append(get_category_from_line(line))

    return categories


def get_category_from_line(line):
    """Return the category without anything else from categories.txt"""

    category = line.split(",")[0][10:]
    category = category.replace(" ", "_")
    category = category.replace("\n", "")
    return category           


def get_WNID_from_index(index):
    """Return WNID given an index of categories.txt"""
    assert(index >= 0 and index < 1000), "index needs to be within [0, 999]"

    filepath = ut.project_dir("code/wordnet/")
    line = lc.getline(os.path.join(filepath, "categories.txt"), index+1)
    return line.split(" ")[0]


def get_similarity_matrix(categories, distance_function):
    """Return the similarity matrix of all categories."""

    num_labels = len(categories)
    matrix = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(i+1, num_labels):
            synset1 = wn.synset(categories[i] + ".n.01")
            synset2 = wn.synset(categories[j] + ".n.01")
            matrix[i,j] = matrix[j,i] = distance_function(synset1, synset2)

    return matrix


def get_2012_similarity_matrix(distance_function):
    """Return the similarity matrix of the ilsvrc2012 categories."""

    categories = get_ilsvrc2012_categories()
    return get_similarity_matrix(categories, distance_function)


def path_similarity(synset1, synset2):
    """Return the path similarity (shortest connecting path in wordnet)."""

    return synset1.path_similarity(synset2)


def lch_similarity(synset1, synset2):
    """Return the Leacock-Chodorow similarity.

       lch similarity is a score denoting how similar two word
       senses are, based on the shortest path that connects the
       senses and the maximum depth of the taxonomy in which the
       senses occur.   
    """ 

    return synset1.lch_similarity(synset2)


def wup_similarity(synset1, synset2):
    """Return the Wu-Palmer similarity.

       The wup similarity is a score denoting how similar two word
       senses are, based on the depth of the two senses in the
       taxonomy and that of their least common subsumer
       (=most specific ancestor node).
    """

    return synset1.wup_similarity(synset2)

