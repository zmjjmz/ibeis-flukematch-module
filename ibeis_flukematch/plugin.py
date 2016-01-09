from __future__ import division, print_function
import ibeis
import utool as ut
import numpy as np
import cv2
from os.path import join, exists
import cPickle as pickle
from collections import defaultdict
from flukematch import (find_trailing_edge_cpp, block_integral_curvatures_cpp,
                        get_distance_curvweighted,)

# register : name, parent(s), cols, dtypes


@ibeis.register_preproc('has_notch_tips', ['annot'], ['exists'], [bool])
def preproc_has_tips(depc_obj, aid_list, config={}):
    r"""
    HACK TO FIND ONLY ANNTS THAT HAVE TIPS

    Args:
        depc_obj (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {})

    Yields:
        tuple: (np.ndarray, np.ndarray, np.ndarray)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --db testdb1
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> aid_list = ibs.get_valid_aids()
        >>> config = {}
        >>> result = preproc_has_tips(ibs.depc, aid_list, config)
        >>> hasnotch_list = list(result)
        >>> print('%d annots have notches' % (sum(hasnotch_list)))
    """
    ibs = depc_obj.controller
    fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
    if not exists(fn):
        print("[fluke-module] ERROR: Could not find image points file")
        raise NotImplementedError('Could not find image points file')

    with open(fn, 'r') as f:
        # this is a dict of img: dict of left/right/notch to the corresponding
        # point
        img_points_map = pickle.load(f)

    img_names = ibs.get_annot_image_names(aid_list)

    for imgn in ut.ProgIter(img_names, lbl='Checking Notches'):
        yield (imgn in img_points_map,)


@ibeis.register_preproc('Notch_Tips', ['annot'], ['notch', 'left', 'right'], [np.ndarray, np.ndarray, np.ndarray])
def preproc_notch_tips(depc_obj, aid_list, config={}):
    r"""
    Args:
        depc_obj (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {})

    Yields:
        tuple: (np.ndarray, np.ndarray, np.ndarray)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb('humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('has_notch_tips', all_aids, 'exists')
        >>> aid_list = ut.compress(all_aids, isvalid)
        >>> config = {}
        >>> result = preproc_notch_tips(ibs.depc, aid_list, config)
        >>> result = list(result)
        >>> print(len(filter(lambda x: x is not None, result)))
    """
    ibs = depc_obj.controller
    # TODO: Implement manual annotation options
    # HACK: Read in a file that associates image names w/these annotations, and
    #   try to associate these w/the image names
    # HACK: hardcode this filename relative to the IBEIS directory

    fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
    if not exists(fn):
        print("[fluke-module] ERROR: Could not find image points file")
        raise NotImplementedError('Could not find image points file')

    with open(fn, 'r') as f:
        # this is a dict of img: dict of left/right/notch to the corresponding
        # point
        img_points_map = pickle.load(f)

    img_names = ibs.get_annot_image_names(aid_list)
    for imgn in ut.ProgIter(img_names, lbl='Reading Notches'):
        try:
            yield (img_points_map[imgn]['notch'],
                   img_points_map[imgn]['left'],
                   img_points_map[imgn]['right'],)
        except KeyError:
            print(
                "[fluke-module] ERROR: aid given that does not have points associated")
            yield None
            #raise NotImplementedError


@ibeis.register_preproc('Trailing_Edge', ['Notch_Tips'], ['edge', 'cost'], [np.ndarray, float])
def preproc_trailing_edge(depc_obj, aid_list, config={'n_neighbors': 5}):
    r"""
    Args:
        depc_obj (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {'n_neighbors': 5})

    Yields:
        tuple: (tedge, cost)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> depc_obj = '?'
        >>> aid_list = ibs.get_valid_aids()
        >>> config = {'n_neighbors': 5}
        >>> (tedge, cost) = preproc_trailing_edge(depc_obj, aid_list, config)
    """
    ibs = depc_obj.controller
    # get the notch / left / right points
    points = ibs.depc.get_property('Notch-Tips', aid_list)
    # get the actual images
    image_paths = ibs.get_annot_image_paths(aid_list)
    # call flukematch.get_trailing_edge on each image
    try:
        n_neighbors = config['n_neighbors']
    except KeyError:
        print("[fluke-module] WARNING: Number of neighbors for trailing edge"
              "extraction not provided, defaulting to 5")
        n_neighbors = 5
    for imagen, point_set in zip(image_paths, points):
        img = cv2.imread(imagen)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tedge, cost = find_trailing_edge_cpp(img_grey, point_set[1],
                                             point_set[2], point_set[0],
                                             n_neighbors=n_neighbors)
        yield (tedge, cost)


@ibeis.register_preproc('Block_Curvature', ['Trailing_Edge'], ['curvature'], [np.ndarray])
def preproc_block_curvature(depc_obj, aid_list, config={'sizes': [5, 10, 15, 20]}):
    r"""
    Args:
        depc_obj (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {'sizes': [5, 10, 15, 20]})

    Yields:
        list: [np.ndarray]

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_block_curvature --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> depc_obj = '?'
        >>> aid_list = ibs.get_valid_aids()
        >>> config = {'sizes': [5, 10, 15, 20]}
        >>> result = preproc_block_curvature(depc_obj, aid_list, config)
    """
    ibs = depc_obj.controller
    # get the trailing edges
    tedges, _ = zip(*ibs.depc.get_property('Trailing_Edge', aid_list))
    try:
        sizes = config['sizes']
    except KeyError:
        sizes = [5, 10, 15, 20]
        print(("[fluke-module] WARNING: Sizes for block curvature extraction"
               "not provided, defaulting to %r ") % (sizes,))

    # call flukematch.block_integral_curvatures_cpp
    for tedge in tedges:
        yield block_integral_curvatures_cpp(tedge, sizes)


DEFAULT_ALGO_CONFIG = {
    'verbose': False,
    'daid_list': None,
    'decision': np.average,
    'sizes': [5, 10, 15, 20],
    'weights': None
}


@ibeis.register_algo('BC_DTW', ['Block_Curvature'], ['bcdtwmatch'], [ibeis.AnnotMatch.load_from_fpath])
def id_algo_bc_dtw(depc_obj, qaid_list, config=None):
    r"""
    Args:
        depc_obj (DependencyCache):
        qaid_list (list):
        config (dict): (default = {'weights': None,
            'decision': <function average at 0x7ff71b2bd7d0>, 'daid_list':
                None, 'verbose': False, 'sizes': [5, 10, 15, 20]})

    Yields:
        ibeis.AnnotMatch:

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-id_algo_bc_dtw --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> depc_obj = '?'
        >>> qaid_list = '?'
        >>> config = {'weights': None, 'decision': np.average, 'daid_list': None,
        >>>           'verbose': False, 'sizes': [5, 10, 15, 20]}
        >>> result = id_algo_bc_dtw(depc_obj, qaid_list, config)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    if config is None:
        config = DEFAULT_ALGO_CONFIG

    assert(config['daid_list'] is not None)
    weights = config['weights']
    sizes = config['sizes']
    if weights is not None:
        assert(len(weights) == len(sizes))
    else:
        weights = [1.] * len(sizes)

    ibs = depc_obj.controller
    daid_list = config['daid_list']
    block_config = ut.dict_subset(config, ['sizes'])
    query_curvs = depc_obj.get_property(
        'Block_Curvature', qaid_list, config=block_config)
    db_curvs = depc_obj.get_property(
        'Block_Curvature', daid_list, config=block_config)

    qnid_list = ibs.get_annot_nids(qaid_list)
    dnid_list = ibs.get_annot_nids(daid_list)

    _iter = zip(query_curvs, qaid_list, qnid_list)
    _progiter = ut.ProgressIter(_iter, lbl='QueryAID',
                                enabled=config['verbose'])

    for query_curv, qaid, qnid in _progiter:
        dists_by_nid = defaultdict(list)
        daid_dists = []
        for db_curv, daid, dnid in zip(db_curvs, daid_list, dnid_list):
            distance = get_distance_curvweighted(query_curv, db_curv, weights)
            daid_dists.append(-1 * distance)
            dists_by_nid[dnid].append(-1 * distance)

        dists_by_nid = {dnid: config['decision'](
            dists_by_nid[dnid]) for dnid in dists_by_nid}
        dnid_dists = [dists_by_nid[dnid] for dnid in dnid_list]

        yield ibeis.AnnotMatch(qaid, qnid, daid_list, dnid_list, daid_dists, dnid_dists)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_flukematch.plugin
        python -m ibeis_flukematch.plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
