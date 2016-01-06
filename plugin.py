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
def preproc_notch_tips(ibs, aid_list, cfg={}):
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
    ret_annots = []
    for imgn in img_names:
        try:
            ret_annots.append((img_points_map[imgn]['notch'],
                               img_points_map[imgn]['left'],
                               img_points_map[imgn]['right'],))
        except KeyError:
            print("[fluke-module] ERROR: aid given that does not have points associated")
            raise NotImplementedError
    return ret_annots

@ibeis.register_preproc('Trailing-Edge', ['Notch-Tips'], ['edge', 'cost'], [np.ndarray, np.float32])
def preproc_trailing_edge(ibs, aid_list, cfg={'n_neighbors':5}):
    # get the notch / left / right points
    points = ibs.depc.get_property('Notch-Tips', aid_list)
    # get the actual images
    image_paths = ibs.get_annot_image_paths(aid_list)
    # call flukematch.get_trailing_edge on each image
    try:
        n_neighbors = cfg['n_neighbors']
    except KeyError:
        print("[fluke-module] WARNING: Number of neighbors for trailing edge extraction not provided, defaulting to 5")
        n_neighbors = 5
    ret_edges = []
    for imagen, point_set in zip(image_paths, points):
        img = cv2.imread(imagen)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tedge, cost, _ = find_trailing_edge(img_grey, point_set[1], point_set[2], center=point_set[0], n_neighbors=n_neighbors)
        ret_edges.append((tedge, cost))

    return ret_edges

@ibeis.register_preproc('Block-Curvature', ['Trailing-Edge'], ['curvature'], [np.ndarray])
def preproc_block_curvature(ibs, aid_list, cfg={'sizes':[5,10,15,20]}):
    # get the trailing edges
    tedges, _ = zip(*ibs.depc.get_property('Trailing-Edge', aid_list))
    try:
        sizes = cfg['sizes']
    except KeyError:
        print("[fluke-module] WARNING: Sizes for block curvature extraction not provided, defaulting to [5, 10, 15, 20]")
        sizes = [5,10,15,20]

    # call flukematch.block_integral_curvatures_cpp
    ret_bcs = []
    for tedge in tedges:
        ret_bcs.append(block_integral_curvatures_cpp(tedge, sizes))

    return ret_bcs




