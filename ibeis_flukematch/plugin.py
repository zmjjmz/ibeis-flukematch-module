# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ibeis
import utool as ut
import dtool  # NOQA
import numpy as np
import vtool as vt
import cv2
from os.path import join, exists
import cPickle as pickle
from ibeis import constants as const
#from collections import defaultdict
from ibeis_flukematch.flukematch import (find_trailing_edge_cpp,
                                         block_integral_curvatures_cpp,
                                         get_distance_curvweighted,)
(print, rrr, profile) = ut.inject2(__name__, '[flukeplug]')

ROOT = ibeis.const.ANNOTATION_TABLE

# register : name, parent(s), cols, dtypes


def debug_depcache(ibs):
    r"""
    CommandLine:
        python -m ibeis_flukematch.plugin --exec-debug_depcache
        python -m ibeis_flukematch.plugin --exec-debug_depcache --show --no-cnn
        python -m ibeis_flukematch.plugin --exec-debug_depcache --clear-all-depcache --db humbpacks

    Example:
        >>> # SCRIPT
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> debug_depcache(ibs)
        >>> ut.show_if_requested()
    """
    print(ibs.depc)
    nas_notch_deps = ibs.depc.get_dependencies('Has_Notch')
    print('nas_notch_deps = %r' % (nas_notch_deps,))
    te_deps = ibs.depc.get_dependencies('Trailing_Edge')
    print('te_deps = %r' % (te_deps,))
    notch_tip_deps = ibs.depc.get_dependencies('Notch_Tips')
    print('notch_tip_deps = %r' % (notch_tip_deps,))
    ibs.depc.print_schemas()
    try:
        ibs.depc.show_digraph()
    except Exception as ex:
        ut.printex(ex, iswarning=True)
    # from dtool import depends_cache
    # print(ut.repr3(depends_cache.__PREPROC_REGISTER__))
    # print(ut.repr3(depends_cache.__ALGO_REGISTER__))


@ibeis.register_preproc('Has_Notch', [ROOT], ['flag'], [bool])
def preproc_has_tips(depc, aid_list, config=None):
    r"""
    HACK TO FIND ONLY ANNTS THAT HAVE TIPS

    Args:
        depc (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {})

    Yields:
        tuple: (np.ndarray, np.ndarray, np.ndarray)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --db testdb1
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn --clear-all-depcache
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --db humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_has_tips --db humpbacks --no-cnn --clear-all-depcache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> aid_list = ibs.get_valid_aids()
        >>> config = {}
        >>> propgen = preproc_has_tips(ibs.depc, aid_list, config)
        >>> result = list(propgen)
        >>> hasnotch_list = ut.take_column(result, 0)
        >>> num_with = sum(hasnotch_list)
        >>> print('%r / %r annots have notches' % (num_with, len(aid_list)))
    """
    print('Preprocess Has_Notch')
    print(config)
    if config is None:
        config = {}

    config = config.copy()
    ibs = depc.controller
    fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
    if not exists(fn):
        print('[fluke-module] ERROR: Could not find image points file')
        raise NotImplementedError('Could not find image points file')

    with open(fn, 'r') as f:
        # this is a dict of img: dict of left/right/notch to the corresponding
        # point
        img_points_map = pickle.load(f)

    img_names = ibs.get_annot_image_names(aid_list)

    for imgn in ut.ProgIter(img_names, lbl='Checking Has_Notch'):
        try:
            (img_points_map[imgn]['notch'], img_points_map[imgn]['left'],
             img_points_map[imgn]['right'],)
        except KeyError:
            yield (False,)
        else:
            yield (True,)


# TODO: depricate old chips with this one in ibeis core

#DEFAULT_CHIP_CONFIG = {
#    'resize_dim': 'width',
#    'dim_size': 128,
#    'preserve_aspect': True,
#    'ext': '.png'
#}


class ChipConfig(dtool.TableConfig):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('resize_dim', 'width',
                         valid_values=['area', 'width', 'heigh', 'diag']),
            ut.ParamInfo('dim_size', 128, 'sz'),
            ut.ParamInfo('preserve_aspect', True, hideif=True),
            ut.ParamInfo('histeq', False, hideif=False),
            ut.ParamInfo('ext', '.png'),
        ]


@ibeis.register_preproc(
    const.CHIP_TABLE,
    parents=[const.ANNOTATION_TABLE],
    colnames=['img', 'width', 'height', 'M'],
    coltypes=[('extern', vt.imread), int, int, np.ndarray],
    config_class=ChipConfig,
    docstr='Used to store *processed* annots as chips',
    #fname='chipcache2'
)
def preproc_chip(depc, aid_list, config=None):
    r"""
    Example of using the dependency cache.

    Args:
        depc (ibeis.depends_cache.DependencyCache):
        aid_list (list):  list of annotation rowids
        config2_ (dict): (default = None)

    Yields:
        (uri, int, int): tup

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_chip --show --db humpbacks

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> depc = ibs.depc
        >>> config = None
        >>> aid_list = ibs.get_valid_aids()[0:20]
        >>> chips = depc.get_property(ibs.const.CHIP_TABLE, aid_list, 'img')
        >>> depc[const.CHIP_TABLE].print_csv()
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(chips, nPerPage=4)
        >>> pt.show_if_requested()
    """
    print('Preprocess Chips')
    print(config)

    ibs = depc.controller
    chip_dpath = ibs.get_chipdir() + '2'
    if config is None:
        config = ChipConfig()

    ext = config['ext']

    cfghashid = config.get_hashid()
    avuuid_list = ibs.get_annot_visual_uuids(aid_list)

    _fmt = 'chip_avuuid_{avuuid}_{cfghashid}{ext}'
    cfname_list = [_fmt.format(avuuid=avuuid, ext=ext, cfghashid=cfghashid)
                   for avuuid in avuuid_list]
    cfpath_list = [ut.unixjoin(chip_dpath, chip_fname)
                   for chip_fname in cfname_list]

    gfpath_list = ibs.get_annot_image_paths(aid_list)
    bbox_list   = ibs.get_annot_bboxes(aid_list)
    theta_list  = ibs.get_annot_thetas(aid_list)
    bbox_size_list = ut.take_column(bbox_list, [2, 3])

    dim_size = config['dim_size']
    resize_dim = config['resize_dim']
    scale_func_dict = {
        'width': vt.get_scaled_size_with_width,
        'root_area': vt.get_scaled_size_with_area,
    }
    scale_func = scale_func_dict[resize_dim]
    if resize_dim == 'root_area':
        dim_size = dim_size ** 2
    newsize_list = [scale_func(dim_size, w, h) for (w, h) in bbox_size_list]

    invalid_flags = [w == 0 or h == 0 for (w, h) in bbox_size_list]
    invalid_aids = ut.compress(aid_list, invalid_flags)

    assert len(invalid_aids) == 0, 'invalid aids=%r' % (invalid_aids,)

    arg_iter = zip(cfpath_list, gfpath_list, bbox_list, theta_list,
                   newsize_list)
    arg_list = list(arg_iter)

    flags = cv2.INTER_LANCZOS4
    borderMode = cv2.BORDER_CONSTANT
    warpkw = dict(flags=flags, borderMode=borderMode)

    ut.ensuredir(chip_dpath)
    for tup in ut.ProgIter(arg_list, lbl='computing chips'):
        cfpath, gfpath, bbox, theta, new_size = tup
        chipBGR = vt.compute_chip(gfpath, bbox, theta, new_size)
        # Build transformation from image to chip
        M = vt.get_image_to_chip_transform(bbox, new_size, theta)
        # Read parent image
        imgBGR = vt.imread(gfpath)
        # Warp chip
        chipBGR = cv2.warpAffine(imgBGR, M[0:2], tuple(new_size), **warpkw)
        height, width = vt.get_size(chipBGR)
        # Write chip to disk
        vt.imwrite(cfpath, chipBGR)
        yield (cfpath, width, height, M)


DEFAULT_NTIP_CONFIG = {}


@ibeis.register_preproc('Notch_Tips', [const.CHIP_TABLE], ['notch', 'left', 'right'], [np.ndarray, np.ndarray, np.ndarray])
def preproc_notch_tips(depc, cid_list, config=None):
    r"""
    Args:
        depc (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {})

    Yields:
        tuple: (np.ndarray, np.ndarray, np.ndarray)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --no-cnn --clear-all-depcache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
        >>> aid_list = ut.compress(all_aids, isvalid)
        >>> cid_list = ibs.depc.get_rowids(const.CHIP_TABLE, aid_list)
        >>> config = {}
        >>> result = preproc_notch_tips(ibs.depc, cid_list, config)
        >>> result = list(result)
        >>> #print(len(filter(lambda x: x is not None, result)))
        >>> print('depth_profile(notch_tips) = %r' % (ut.depth_profile(result),))
    """
    print('Preprocess Notch_Tips')
    print(config)

    if config is None:
        config = DEFAULT_NTIP_CONFIG
    config = config.copy()

    ibs = depc.controller
    # TODO: Implement manual annotation options
    # HACK: Read in a file that associates image names w/these annotations, and
    #   try to associate these w/the image names
    # HACK: hardcode this filename relative to the IBEIS directory

    # this is a dict of img: dict of left/right/notch to the corresponding
    # point
    fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
    img_points_map = ut.load_cPkl(fn)

    aid_list = depc.get_root_rowids(const.CHIP_TABLE, cid_list)
    img_names = ibs.get_annot_image_names(aid_list)

    M_list = ibs.depc.get_native_property(const.CHIP_TABLE, cid_list, 'M')

    for aid, imgn, M in ut.ProgIter(zip(aid_list, img_names, M_list),
                                    lbl='Reading Notch_Tips'):
        try:
            # Need to scale notch tips as they are
            # specified relative to the image, not the chip.
            ptdict = img_points_map[imgn]
            notch, left, right = ut.dict_take(ptdict, ['notch', 'left', 'right'])

            notch_ = notch.dot(M[0:2])[0:2]
            left_ = left.dot(M[0:2])[0:2]
            right_ = right.dot(M[0:2])[0:2]
            yield (notch_, left_, right_)
        except KeyError:
            print(
                '[fluke-module] ERROR: aid=%r does not have points associated' % (aid,))
            # yield None
            raise NotImplementedError(
                'ERROR: aid=%r does not have points associated' % (aid,))


DEFAULT_TE_CONFIG = {'n_neighbors': 5}


@ibeis.register_preproc('Trailing_Edge', ['Notch_Tips'], ['edge', 'cost'], [np.ndarray, float])
def preproc_trailing_edge(depc, ntid_list, config=None):
    r"""
    Args:
        depc (DependencyCache):
        ntid_list (list):  list of notch tip rowids
        config (dict): (default = {'n_neighbors': 5})

    Yields:
        tuple: (tedge, cost)

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --show
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn --clear-all-depcache
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --db humpbacks --no-cnn --clear-all-depcache
        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --db humpbacks --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
        >>> aid_list = ut.compress(all_aids, isvalid)[0:10]
        >>> print('aid_list = %r' % (aid_list,))
        >>> depc = ibs.depc
        >>> config = {'n_neighbors': 5}
        >>> ntid_list = ibs.depc.get_rowids('Notch_Tips', aid_list)
        >>> print('ntid_list = %r' % (ntid_list,))
        >>> propgen = preproc_trailing_edge(depc, ntid_list, config)
        >>> results = list(propgen)
        >>> tedge_list, cost_list = list(zip(*results))
        >>> print('tedge_list = %r' % (tedge_list,))
        >>> print('cost_list = %r' % (cost_list,))
    """
    print('Preprocess Trailing_Edge')
    print(config)

    if config is None:
        config = DEFAULT_TE_CONFIG
    config = config.copy()
    ibs = depc.controller
    # get the notch / left / right points
    # points = ibs.depc.get_property('Notch_Tips', aid_list)
    points = ibs.depc.get_native_property('Notch_Tips', ntid_list)
    # get the actual images
    #aid_list = depc.get_root_rowids('Notch_Tips', ntid_list)
    #image_paths = ibs.get_annot_image_paths(aid_list)

    cid_list = depc.get_ancestor_rowids('Notch_Tips', ntid_list, const.CHIP_TABLE)
    #image_paths = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img')
    image_paths = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img',
                                           read_extern=False)

    # call flukematch.get_trailing_edge on each image
    try:
        n_neighbors = config['n_neighbors']
    except KeyError:
        print('[fluke-module] WARNING: Number of neighbors for trailing edge'
              'extraction not provided, defaulting to 5')
        n_neighbors = 5
    _iter = zip(image_paths, points)
    progiter = ut.ProgIter(_iter, lbl='compute Trailing_Edge')
    for imagen, point_set in progiter:
        img = cv2.imread(imagen)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        left, right, notch = point_set[1], point_set[2], point_set[0]
        left = left.round().astype(np.int64)
        right = right.round().astype(np.int64)
        notch = notch.round().astype(np.int64)
        # TODO: find_trailing_edge should work to subpixel accuracy
        tedge, cost = find_trailing_edge_cpp(
            img_grey, left, right, notch,
            n_neighbors=n_neighbors)
        yield (tedge, cost)


#def preproc_binarized(coords, sizes):
#    """
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis_flukematch.plugin import *  # NOQA
#        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
#        >>> all_aids = ibs.get_valid_aids()
#        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag', _debug=True)
#        >>> aid_list = ut.compress(all_aids, isvalid)[0:1]
#        >>> tedges = depc.get_property('Trailing_Edge', aid_list, 'edge', config)
#        >>> coords = tedges[0]
#        >>> sizes = [20]
#    """
#    coords = np.array(coords, dtype=np.int32)
#    fit_size = (np.max(coords, axis=0) - np.min(coords, axis=0)) + 1
#    binarized = np.zeros(fit_size[::-1], dtype=np.float32)
#    fixed_coords = np.array((coords - np.min(coords, axis=0)))[:, ::-1]
#    fixed_coords = np.ascontiguousarray(fixed_coords)
#    binarized[zip(*fixed_coords)] = 1
#    binarized = binarized.cumsum(axis=0)
#    binarized[np.where(binarized > 0)] = 1
#    summed_table = binarized.cumsum(axis=0).cumsum(axis=1)
#    yield (summed_table, fixed_coords)


#def preproc_block_curve(summed_table, fixed_coords, config):
#    """
#    size = 10
#    """
#    from ibeis_flukematch import flukematch
#    size = config['size']
#    curv = np.zeros((fixed_coords.shape[0], 1), dtype=np.float32)
#    flukematch.block_curv(summed_table, summed_table.shape[0],
#                          summed_table.shape[1], fixed_coords,
#                          fixed_coords.shape[0], size, curv)


@ibeis.register_preproc('Block_Curvature', ['Trailing_Edge'], ['curvature'], [np.ndarray])
def preproc_block_curvature(depc, te_rowids, config={'sizes': [5, 10, 15, 20]}):
    r"""
    Args:
        depc (DependencyCache):
        aid_list (list):  list of annotation rowids
        config (dict): (default = {'sizes': [5, 10, 15, 20]})

    Yields:
        list: [np.ndarray]

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_block_curvature --dbdir /home/zach/data/IBEIS/humpbacks --no-cnn
        python -m ibeis_flukematch.plugin --exec-preproc_block_curvature --db humpbacks --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag', _debug=True)
        >>> aid_list = ut.compress(all_aids, isvalid)[0:4]
        >>> print('\n!!![test] aid_list = %r' % (aid_list,))
        >>> depc = ibs.depc
        >>> config = {'sizes': [5, 10, 15, 20]}
        >>> te_rowids = depc.get_rowids('Trailing_Edge', aid_list, config)
        >>> print('te_rowids = %r' % (te_rowids,))
        >>> propgen = preproc_block_curvature(depc, te_rowids, config)
        >>> curve_arr_list = list(propgen)
        >>> result = ut.depth_profile(curve_arr_list)
        >>> print(result)
    """
    print('Preprocess Block_Curvature')
    print(config)

    ibs = depc.controller
    # NOTE: Need to use get_native_property because the take the type
    # of the parent (trailing ege) ids, not the root (annot) ids.
    # get the trailing edges
    # NOTE: Can specify a single column, so unpacking is done automatically
    tedges = ibs.depc.get_native_property('Trailing_Edge', te_rowids, 'edge')
    try:
        sizes = config['sizes']
    except KeyError:
        sizes = [5, 10, 15, 20]
        print(('[fluke-module] WARNING: Sizes for block curvature extraction'
               'not provided, defaulting to %r ') % (sizes,))

    # call flukematch.block_integral_curvatures_cpp
    progiter = ut.ProgIter(tedges, lbl='compute Block_Curvature')
    for tedge in progiter:
        curve_arr = block_integral_curvatures_cpp(sizes, tedge)
        yield (curve_arr,)


DEFAULT_ALGO_CONFIG = {
    'verbose': False,
    'decision': 'average',
    'sizes': (5, 10, 15, 20),
    'weights': None
}


@ibeis.register_algo('BC_DTW', algo_result_class=ibeis.AnnotMatch,
                     config_class=DEFAULT_ALGO_CONFIG)
def id_algo_bc_dtw(depc, request):
    r"""
    Args:
        depc (DependencyCache):
        qaid_list (list):
        config (dict): (default = {'weights': None,
            'decision': <function average at 0x7ff71b2bd7d0>, 'daid_list':
                None, 'verbose': False, 'sizes': [5, 10, 15, 20]})

    Yields:
        ibeis.AnnotMatch:

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-id_algo_bc_dtw --show
        ibeis -e rank_cdf --db humpbacks -t default:pipeline_root=BC_DTW --qaid-override=1,9,15,16,18 --daid-override=1,9,15,16,18,21,22  --show --clear-all-depcache --nocache
        ibeis -e rank_cdf --db humpbacks -t default:pipeline_root=BC_DTW --qaid-override=1,9,15,16,18 --daid-override=1,9,15,16,18,21,22  --show --nocache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
        >>> aid_list = ut.compress(all_aids, isvalid)
        >>> depc = ibs.depc
        >>> qaids = aid_list[0:5]
        >>> daids = aid_list[0:7]
        >>> qaids = aid_list
        >>> daids = aid_list
        >>> cfgdict = {'weights': None, 'decision': 'average',
        >>>           'verbose': True, 'sizes': (5, 10, 15, 20)}
        >>> algoname = 'BC_DTW'
        >>> request = depc.new_algo_request(algoname, qaids, daids, cfgdict)
        >>> propgen = id_algo_bc_dtw(depc, request)
        >>> am_list = list(propgen)
        >>> print('\n'.join(ut.lmap(str, am_list)))
        >>> result1 = (ut.repr2(np.vstack([am.annot_score_list for am in am_list]), precision=2))
        >>> am = am_list[0]
        >>> # Execute via cache
        >>> am_list2 = request.execute()
        >>> result2 = (ut.repr2(np.vstack([am.annot_score_list for am in am_list2]), precision=2))
        >>> print(result1)
        >>> assert result1 == result2
    """
    print('Executing BC_DTW')
    print(request)

    qaid_list = request.qaids
    daid_list = request.daids
    config = request.config

    #if config is None:
    #    config = DEFAULT_ALGO_CONFIG

    #assert(config['daid_list'] is not None)
    curv_weights = config['weights']
    sizes = config['sizes']
    if curv_weights is not None:
        assert(len(curv_weights) == len(sizes))
    else:
        curv_weights = [1.] * len(sizes)

    ibs = depc.controller
    block_config = ut.dict_subset(config, ['sizes'])
    query_curvs = depc.get_property(
        'Block_Curvature', qaid_list, 'curvature', config=block_config)
    db_curvs = depc.get_property(
        'Block_Curvature', daid_list, 'curvature', config=block_config)

    qnid_list = ibs.get_annot_nids(qaid_list)
    dnid_list = ibs.get_annot_nids(daid_list)

    _iter = zip(query_curvs, qaid_list, qnid_list)
    _progiter = ut.ProgressIter(_iter, lbl='QueryAID',
                                enabled=config['verbose'])

    for query_curv, qaid, qnid in _progiter:
        #dists_by_nid = defaultdict(list)
        daid_dists = []
        for db_curv, daid, dnid in zip(db_curvs, daid_list, dnid_list):
            distance = get_distance_curvweighted(query_curv, db_curv, curv_weights)
            daid_dists.append(-1 * distance)
            #dists_by_nid[dnid].append(-1 * distance)

        decision_func = getattr(np, config['decision'])
        #dists_by_nid = {dnid: decision_func(
        #    dists_by_nid[dnid]) for dnid in dists_by_nid}
        #dnid_dists = [dists_by_nid[dnid] for dnid in dnid_list]

        # Remove distance to self
        annot_scores = np.array(daid_dists)
        daid_list_ = np.array(daid_list)
        dnid_list_ = np.array(dnid_list)

        isself = (daid_list_ != qaid)
        daid_list_ = daid_list_.compress(isself)
        dnid_list_ = dnid_list_.compress(isself)
        annot_scores = annot_scores.compress(isself)

        # Hacked in version of creating an annot match object
        match_result = ibeis.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([decision_func(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


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
