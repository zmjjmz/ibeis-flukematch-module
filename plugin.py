import ibeis
import numpy as np
from os.path import join, exists
import cPickle as pickle
from flukematch import (
        find_trailing_edge,
        block_integral_curvatures_cpp,
)

# register : name, parent(s), cols, dtypes
@ibeis.register_preproc('Notch-Tips', ['annot'], ['notch', 'left', 'right'], [np.ndarray, np.ndarray, np.ndarray])
def preproc_notch_tips(depc_obj, aid_list, config={}):
    ibs = depc_obj.controller
    # TODO: Implement manual annotation options
    # HACK: Read in a file that associates image names w/these annotations, and try to associate these w/the image names
    # HACK: hardcode this filename relative to the IBEIS directory

    fn = join(ibs.getdbdir(), 'fluke_image_points.pkl')
    if not exists(fn):
        print("[fluke-module] ERROR: Could not find image points file")
        raise NotImplementedError

    with open(fn, 'r') as f:
        # this is a dict of img: dict of left/right/notch to the corresponding point
        img_points_map = pickle.load(f)

    img_names = ibs.get_annot_image_names(aid_list)
    for imgn in img_names:
        try:
            yield (img_points_map[imgn]['notch'],
                   img_points_map[imgn]['left'],
                   img_points_map[imgn]['right'],)
        except KeyError:
            print("[fluke-module] ERROR: aid given that does not have points associated")
            raise NotImplementedError

@ibeis.register_preproc('Trailing-Edge', ['Notch-Tips'], ['edge', 'cost'], [np.ndarray, np.float32])
def preproc_trailing_edge(depc_obj, aid_list, config={'n_neighbors':5}):
    ibs = depc_obj.controller
    # get the notch / left / right points
    points = ibs.depc.get_property('Notch-Tips', aid_list)
    # get the actual images
    image_paths = ibs.get_annot_image_paths(aid_list)
    # call flukematch.get_trailing_edge on each image
    try:
        n_neighbors = config['n_neighbors']
    except KeyError:
        print("[fluke-module] WARNING: Number of neighbors for trailing edge extraction not provided, defaulting to 5")
        n_neighbors = 5
    for imagen, point_set in zip(image_paths, points):
        img = cv2.imread(imagen)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tedge, cost = find_trailing_edge_cpp(img_grey, point_set[1], point_set[2], point_set[0], n_neighbors=n_neighbors)
        yield (tedge, cost)

@ibeis.register_preproc('Block-Curvature', ['Trailing-Edge'], ['curvature'], [np.ndarray])
def preproc_block_curvature(depc_obj, aid_list, config={'sizes':[5,10,15,20]}):
    ibs = depc_obj.controller
    # get the trailing edges
    tedges, _ = zip(*ibs.depc.get_property('Trailing-Edge', aid_list))
    try:
        sizes = config['sizes']
    except KeyError:
        print("[fluke-module] WARNING: Sizes for block curvature extraction not provided, defaulting to [5, 10, 15, 20]")
        sizes = [5,10,15,20]

    # call flukematch.block_integral_curvatures_cpp
    for tedge in tedges:
        yield block_integral_curvatures_cpp(tedge, sizes)






