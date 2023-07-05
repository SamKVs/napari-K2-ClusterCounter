import matplotlib.pyplot as plt
import numpy as np
import fil_finder
from skimage import measure
from skimage.morphology import skeletonize, remove_small_objects
import astropy.units as u
from warnings import filterwarnings

def area_to_length(area):

    filterwarnings('ignore', category=UserWarning)
    filterwarnings('ignore', category=RuntimeWarning)

    def clean(array):
        # assuming mask is a binary image
        # label and calculate parameters for every cluster in mask
        labelled = measure.label(array)
        rp = measure.regionprops(labelled)

        # get size of largest cluster
        size = max([i.area for i in rp])

        # remove everything smaller than largest
        out = remove_small_objects(array, min_size=size - 1)
        return out

    image = clean(area)
    skeleton_lee = skeletonize(image, method='lee')


    fil = fil_finder.FilFinder2D(skeleton_lee, mask=skeleton_lee)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')


    if len(fil.lengths()) == 0:
        return np.nan
    else:
        return max(fil.lengths())