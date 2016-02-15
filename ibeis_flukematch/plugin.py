# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m ibeis --tf autogen_ipynb --db humpbacks --ipynb -t default:proot=BC_DTW -a default:has_any=hasnotch
    python -m ibeis --tf autogen_ipynb --db humpbacks -a default:has_any=hasnotch -t default:proot=BC_DTW,manual_extract=True default:proot=BC_DTW,manual_extract=True
    python -m ibeis --tf autogen_ipynb --db humpbacks --ipynb -t default:proot=BC_DTW default:proot=vsmany -a default:has_any=hasnotch,mingt=2,qindex=0:50,dindex=0:50 --noexample
    python -m ibeis --tf autogen_ipynb --db humpbacks --ipynb -t default:proot=BC_DTW_NEW default:proot=vsmany -a default:has_any=hasnotch,mingt=2,qindex=0:50,dindex=0:50 --noexample

    python -m ibeis -e rank_cdf --db humpbacks -a default:has_any=hasnotch,mingt=2,qsize=100,dsize=100 -t default:proot=[BC_DTW_NEW,BC_DTW] --show

"""
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
from ibeis import register_preproc, register_algo
from ibeis_flukematch.flukematch import (find_trailing_edge_cpp,
                                         block_integral_curvatures_cpp,
                                         get_distance_curvweighted,
                                         setup_kp_network,
                                         infer_kp,
                                         setup_te_network,
                                         score_te,)
(print, rrr, profile) = ut.inject2(__name__, '[flukeplug]')

#register_preproc = lambda *args, **kwargs: ut.identity
#register_algo = lambda *args, **kwargs: ut.identity

ROOT = ibeis.const.ANNOTATION_TABLE

# register : name, parent(s), cols, dtypes


def testdata_humpbacks():
    import ibeis
    ibs = ibeis.opendb(defaultdb='humpbacks')
    all_aids = ibs.get_valid_aids()
    isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
    aid_list = ut.compress(all_aids, isvalid)
    aid_list = aid_list[0:10]
    return ibs, aid_list


def bound_point(point, size):
    return np.min(np.hstack([np.array(size, dtype=np.int).reshape(-1, 1) - 1,
                             point.reshape(-1, 1)]), axis=1)


def debug_depcache(ibs):
    r"""
    CommandLine:
        python -m ibeis_flukematch.plugin --exec-debug_depcache
        python -m ibeis_flukematch.plugin --exec-debug_depcache --show --no-cnn
        python -m ibeis_flukematch.plugin --exec-debug_depcache --clear-all-depcache --db humbpacks
        python -m ibeis_flukematch.plugin --exec-debug_depcache --show --no-cnn --db humpbacks

        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --no-cnn --show

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

    all_aids = ibs.get_valid_aids()
    isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
    aid_list = ut.compress(all_aids, isvalid)
    aid_list = aid_list[0:10]
    ibs.depc.print_config_tables()
    #import utool
    #utool.embed()
    # from dtool import depends_cache
    # print(ut.repr3(depends_cache.__PREPROC_REGISTER__))
    # print(ut.repr3(depends_cache.__ALGO_REGISTER__))


@register_preproc('Has_Notch', [ROOT], ['flag'], [bool])
def preproc_has_tips(depc, aid_list, config=None):
    r"""
    HACK TO FIND ONLY ANNOTS THAT HAVE TIPS

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
        >>> valid_aids = ut.compress(aid_list, hasnotch_list)
        >>> ibs.append_annot_case_tags(valid_aids, ['hasnotch'] * len(valid_aids))
        >>> print(ibs.get_annot_info(valid_aids[2], default=True))
        >>> print('%r / %r annots have notches' % (num_with, len(aid_list)))
    """
    print('Preprocess Has_Notch')
    print(config)

    config = config.copy()
    ibs = depc.controller
    fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
    if not exists(fn):
        print('[fluke-module] ERROR: Could not find image points file')
        raise NotImplementedError('Could not find image points file')

    # this is a dict of img: dict of left/right/notch to the xy-point
    img_points_map = ut.load_cPkl(fn)

    img_names = ibs.get_annot_image_names(aid_list)

    for imgn in ut.ProgIter(img_names, lbl='Checking Has_Notch'):
        try:
            (img_points_map[imgn]['notch'], img_points_map[imgn]['left'],
             img_points_map[imgn]['right'],)
        except KeyError:
            yield (False,)
        else:
            yield (True,)


class NotchTipConfig(dtool.TableConfig):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('manual_extract', False, hideif=False),
            #ut.ParamInfo('ntversion', 1)
            ut.ParamInfo('version', 1),
        ]


def show_notch_tips(depc, aid, config={}, fnum=None, pnum=None):
    import plottool as pt
    pt.figure(fnum=fnum, pnum=pnum)
    notch = depc.get('Notch_Tips', aid, config=config)
    chip = depc.get('chips', aid, 'img', config=config)
    pt.imshow(chip)
    pt.draw_kpts2(np.array(notch), pts=True, ell=False, pts_size=20)


@register_preproc('Notch_Tips', [const.CHIP_TABLE], ['notch', 'left', 'right'], [np.ndarray, np.ndarray, np.ndarray],
                    configclass=NotchTipConfig)
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
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --no-cnn --show
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --show --manual_extract=False
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --show --manual_extract=True
        python -m ibeis_flukematch.plugin --exec-preproc_notch_tips --db humpbacks --no-cnn --clear-all-depcache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
        >>> aid_list = ut.compress(all_aids, isvalid)
        >>> aid_list = aid_list[0:10]
        >>> #config = dict(dim_size=None)
        >>> config = NotchTipConfig.from_argv_dict()
        >>> depc = ibs.depc
        >>> config['dim_size'] = 480
        >>> cid_list = depc.get_rowids('chips', aid_list, config)
        >>> notch_tips = list(preproc_notch_tips(depc, cid_list, config))
        >>> result = ut.depth_profile(notch_tips)
        >>> print('depth_profile(notch_tips) = %r' % (result,))
        >>> ut.quit_if_noshow()
        >>> chip_list1 = depc.get_native_property('chips', cid_list, 'img')
        >>> chip_list2 = depc.get('chips', aid_list, 'img', config=config)
        >>> assert np.all(chip_list2[0] == chip_list1[0])
        >>> chip_list = chip_list2
        >>> import plottool as pt
        >>> ut.ensure_pylab_qt4()
        >>> overlay_chips = [overlay_fluke_feats(chip, tips=tips) for chip,  tips in zip(chip_list, notch_tips)]
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(overlay_chips, nPerPage=4, autostart=True)
        >>> ut.show_if_requested()
    """
    print('Preprocess Notch_Tips')
    print(config)

    config = config.copy()

    ibs = depc.controller

    aid_list = depc.get_root_rowids(const.CHIP_TABLE, cid_list)
    img_names = ibs.get_annot_image_names(aid_list)

    M_list = ibs.depc.get_native_property(const.CHIP_TABLE, cid_list, 'M')
    size_list = ibs.depc.get_native_property(const.CHIP_TABLE, cid_list, ('width', 'height'))

    if config['manual_extract']:
        # TODO: Implement manual annotation options
        # HACK: Read in a file that associates image names w/these annotations, and
        #   try to associate these w/the image names
        # HACK: hardcode this filename relative to the IBEIS directory

        # this is a dict of img: dict of left/right/notch to the corresponding
        # point
        fn = join(ibs.get_dbdir(), 'fluke_image_points.pkl')
        img_points_map = ut.load_cPkl(fn)
    else:
        network_data = setup_kp_network()
        # process all the points at once
        # TODO: Is this the best way to do this? Or should we do it in the main loop? Another preproc node?
        img_paths = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img',
                                             read_extern=False)
        # assume infer_kp handles the bounds checking / snapping
        # TODO: Add config for batch size and image size
        networkfn = network_data['networkfn']
        mean = network_data['mean']
        std = network_data['std']
        pt_preds = infer_kp(img_paths, networkfn, mean, std)
        img_points_map = {img_name: pt_pred for img_name, pt_pred in zip(img_names, pt_preds)}

    def inbounds(size, point):
        return (point[0] >= 0 and point[0] < size[0]) and (point[1] >= 0 and point[1] < size[1])

    for aid, imgn, M, size in ut.ProgIter(zip(aid_list, img_names, M_list, size_list),
                                          lbl='Reading Notch_Tips'):
        try:
            # Need to scale notch tips as they are
            # specified relative to the image, not the chip.
            ptdict = img_points_map[imgn]
            notch, left, right = ut.dict_take(ptdict, ['notch', 'left', 'right'])

            if config['manual_extract']:
                notch_ = bound_point(M[0:2].T.dot(notch)[0:2], size)
                left_  = bound_point(M[0:2].T.dot(left)[0:2], size)
                right_ = bound_point(M[0:2].T.dot(right)[0:2], size)
            else:
                notch_ = notch
                left_  = left
                right_ = right

            # verify that the notch / left / right are within the bounds specified by size
            assert(inbounds(size, notch_) and inbounds(size, left_) and inbounds(size, right_))

            yield (notch_, left_, right_)
        except KeyError:
            print(
                '[fluke-module] ERROR: aid=%r does not have points associated' % (aid,))
            # yield None
            raise NotImplementedError(
                'ERROR: aid=%r does not have points associated' % (aid,))
        except AssertionError:
            print(
                '[fluke-module] ERROR: aid=%r has associated points that are out of bounds' % (aid,))
            print(
                '[fluke-module] ERROR: Points: Notch: %s, Left: %s, Right: %s -- Chip Size: %s' % (notch_, left_, right_, size))
            raise NotImplementedError(
                'ERROR: aid=%r has associated points that are out of bounds' % (aid,))


class CropChipConfig(dtool.TableConfig):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('crop_dim_size', 960, 'sz', hideif=lambda cfg: cfg['crop_enabled'] is False or cfg['crop_dim_size'] is None),
            ut.ParamInfo('crop_enabled', False, hideif=False),
            #ut.ParamInfo('ccversion', 1)
            ut.ParamInfo('version', 1),
        ]


# Custom chip table
@register_preproc(
    'Cropped_Chips',
    parents=[const.CHIP_TABLE, 'Notch_Tips'],
    colnames=['img', 'width', 'height', 'M', 'notch', 'left', 'right'],
    coltypes=[('extern', vt.imread), int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=CropChipConfig,
    fname='cropped_chip'
)
def preproc_cropped_chips(depc, cid_list, tipid_list, config=None):
    """
    CommandLine:
        python -m ibeis_flukematch.plugin --exec-preproc_cropped_chips --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs, aid_list = testdata_humpbacks()
        >>> config = CropChipConfig(crop_enabled=True)
        >>> cid_list = ibs.depc.get_rowids('chips', aid_list, config)
        >>> tipid_list = ibs.depc.get_rowids('Notch_Tips', aid_list, config)
        >>> depc = ibs.depc
        >>> list(preproc_cropped_chips(depc, cid_list, tipid_list, config))
        >>> #cpid_list = ibs.depc.d.get_Cropped_Chips_rowids(aid_list, config)
        >>> #cpid_list = ibs.depc.w.Cropped_Chips.get_rowids(aid_list, config)
        >>> chip_list = ibs.depc.get_property('Cropped_Chips', aid_list, 'img', config)
        >>> notch_tips = ibs.depc.get_property('Cropped_Chips', aid_list, ('notch', 'left', 'right'), config)
        >>> import plottool as pt
        >>> ut.ensure_pylab_qt4()
        >>> for notch, chip in ut.InteractiveIter(zip(notch_tips, chip_list)):
        >>>     pt.reset()
        >>>     pt.imshow(chip)
        >>>     kpts_ = np.array(notch)
        >>>     pt.draw_kpts2(kpts_, pts=True, ell=False, pts_size=20)
        >>>     pt.update()
        >>> ut.show_if_requested()
    """
    # crop first
    img_list = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img')
    tips_list = depc.get_native_property('Notch_Tips', tipid_list, ('left', 'notch', 'right'))

    imgpath_list = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img',
                                            read_extern=False)

    cropped_chip_dpath = depc.controller.get_chipdir() + '_crop'
    ut.ensuredir(cropped_chip_dpath)
    crop_path_list = [ut.augpath(path, '_crop' + config.get_hashid())
                      for path in imgpath_list]

    for img, tips, path in zip(img_list, tips_list, crop_path_list):
        left, notch, right = tips
        bbox = (0, 0, img.shape[1], img.shape[0])  # default to bbox being whole image
        chip_size = (img.shape[1], img.shape[0])
        if left[0] > right[0]:
            # HACK: Ugh, I don't like this
            # TODO: maybe move this to infer_kp?
            right, left = (left, right)
        if config['crop_enabled']:
            # figure out bbox (x, y, w, h) w/x, y on top left
            # assume left is on the left note: this may not be a good assumption
            # note: lol that's not a good assumption
            # what do when network predicts left on right and right on left?
            bbox = (left[0],  # leftmost x value
                    0,  # top of the image
                    (right[0] - left[0]),  # width
                    img.shape[0],  # height
                    )
        if config['crop_dim_size'] is not None:
            # we're only resizing in x, but after the crop
            # as a result we need to make sure we use the image dimensions apparent to get the chip size
            # we want to preserve the aspect ratio of the crop, not the whole image
            new_x = config['crop_dim_size']
            #ratio = bbox[2] / bbox[3]  # w/h
            #new_y = int(new_x / ratio)
            #chip_size = (new_x, new_y)
            chip_size = vt.get_scaled_size_with_width(new_x, bbox[2], bbox[3])
        M = vt.get_image_to_chip_transform(bbox, chip_size, 0)
        with ut.embed_on_exception_context:
            new_img = cv2.warpAffine(img, M[:-1, :], chip_size)

        notch_, left_, right_ = vt.transform_points_with_homography(M, np.array([notch, left, right]).T).T

        notch_ = bound_point(notch_, chip_size)
        left_  = bound_point(left_, chip_size)
        right_ = bound_point(right_, chip_size)
        vt.imwrite(path, new_img)

        yield (path, bbox[2], bbox[3], M, notch_, left_, right_)


def overlay_fluke_feats(img, path=None, tips=None):
    img_copy = img[:]
    # assume path is x, y
    if path is not None:
        for j, i in path:
            if (j >= img_copy.shape[1] or j < 0) or (i >= img_copy.shape[0] or i < 0):
                    continue
            cv2.circle(img_copy, (j, i), 2, (255, 0, 0), thickness=-1)

    if tips is not None:
        for pt in tips:
            pt1 = np.array(np.round(pt), dtype=np.int)
            pt1 = tuple(pt1.tolist())
            cv2.circle(img_copy, pt1, 7, (0, 128, 255), thickness=-7)
    return img_copy


class TrailingEdgeConfig(dtool.TableConfig):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('n_neighbors', 5, 'n_nb'),
            ut.ParamInfo('ignore_notch', False, 'ign_n', hideif=False),
            #ut.ParamInfo('teversion', 1),
            ut.ParamInfo('version', 3),
            ut.ParamInfo('use_te_scorer', False, 'te_s', hideif=False),
            ut.ParamInfo('te_score_weight', 0.5, 'w_tes'),
        ]


@register_preproc('Trailing_Edge', ['Cropped_Chips'], ['edge', 'cost'], [np.ndarray, float],
                    configclass=TrailingEdgeConfig, chunksize=256)
def preproc_trailing_edge(depc, cpid_list, config=None):
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

        python -m ibeis_flukematch.plugin --exec-preproc_trailing_edge --db humpbacks --no-cnn --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag')
        >>> aid_list = ut.compress(all_aids, isvalid)[0:10]
        >>> print('aid_list = %r' % (aid_list,))
        >>> depc = ibs.depc
        >>> config = {'n_neighbors': 5, 'crop_enabled': True}
        >>> cpid_list = ibs.depc.get_rowids('Cropped_Chips', aid_list, config)
        >>> propgen = preproc_trailing_edge(depc, cpid_list, config)
        >>> results = list(propgen)
        >>> tedge_list, cost_list = list(zip(*results))
        >>> print('tedge_list = %r' % (tedge_list,))
        >>> print('cost_list = %r' % (cost_list,))
        >>> ut.quit_if_noshow()
        >>> # Visualize
        >>> #aid_list = [2826]
        >>> #chipcfg = ibeis.algo.preproc.preproc_chip.ChipConfig(dim_size=None)
        >>> chips = depc.get_property('Cropped_Chips', aid_list, 'img', config=config, _debug=True)
        >>> notches = depc.get_property('Cropped_Chips', aid_list, ('notch', 'left', 'right'), config=config, _debug=True)
        >>> overlay_chips = [overlay_fluke_feats(chip, path, tips=tips) for chip, path, tips in zip(chips, tedge_list, notches)]
        >>> import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(overlay_chips, nPerPage=4)
        >>> iteract_obj.start()
        >>> pt.show_if_requested()

    """
    print('Preprocess Trailing_Edge')
    print(config)

    config = config.copy()
    ibs = depc.controller
    # get the notch / left / right points
    # points = ibs.depc.get_property('Notch_Tips', aid_list)
    img_paths = ibs.depc.get_native_property('Cropped_Chips', cpid_list, 'img', read_extern=False)
    points = ibs.depc.get_native_property('Cropped_Chips', cpid_list, ('notch', 'left', 'right'))
    # get the actual images
    #aid_list = depc.get_root_rowids('Notch_Tips', ntid_list)
    #image_paths = ibs.get_annot_image_paths(aid_list)
    if config['use_te_scorer']:
        network_data = setup_te_network()
        networkfn = network_data['networkfn']
        mean = network_data['mean']
        std = network_data['std']
        score_preds = score_te(img_paths, networkfn, mean, std)
    else:
        score_preds = [None for _ in img_paths]

    #image_paths = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img')
    #image_paths = depc.get_native_property(const.CHIP_TABLE, cid_list, 'img',
    #                                       read_extern=False)

    # call flukematch.get_trailing_edge on each image
    try:
        n_neighbors = config['n_neighbors']
    except KeyError:
        print('[fluke-module] WARNING: Number of neighbors for trailing edge'
              'extraction not provided, defaulting to 5')
        n_neighbors = 5
    _iter = zip(img_paths, points, score_preds)
    progiter = ut.ProgIter(_iter, lbl='compute Trailing_Edge')

    def fix_point(point):
        return np.max(np.hstack([np.zeros((2, 1), dtype=np.int), (point - 1).reshape(-1, 1)]), axis=1)

    for img_path, point_set, score_pred in progiter:
        img = cv2.imread(img_path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        left, right, notch = point_set[1], point_set[2], point_set[0]
        left = fix_point(left.round().astype(np.int))
        right = fix_point(right.round().astype(np.int))
        notch = fix_point(notch.round().astype(np.int))
        # TODO: find_trailing_edge should work to subpixel accuracy
        tedge, cost = find_trailing_edge_cpp(
            img_grey, left, right, notch,
            n_neighbors=n_neighbors, ignore_notch=config['ignore_notch'],
            score_mat=score_pred, score_weight=config['te_score_weight'])
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

class BlockCurvConfig(dtool.TableConfig):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('sizes', (5, 10, 15, 20)),
        ]


@register_preproc('Block_Curvature', ['Trailing_Edge'], ['curvature'], [np.ndarray],
                  configclass=BlockCurvConfig, chunksize=256)
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
    # FIXME: CONFIG
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


class BC_DTW_Config(dtool.AlgoConfig):
    """
    CommandLine:
        python -m ibeis_flukematch.plugin --exec-BC_DTW_Config --show

    IPython:
        ut.execute_doctest('BC_DTW_Config', module='ibeis_flukematch.plugin')

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> config = BC_DTW_Config()
        >>> result = config.get_cfgstr()
        >>> print(result)
        BC_DTW(decision=average,sizes=(5, 10, 15, 20),weights=None,version=1)_CropChip()
    """
    #def get_config_name(self):
    #    return 'BC_DTW'

    def get_sub_config_list(self):
        # Different pipeline compoments can go here
        # as well as dependencies that were not
        # explicitly enumerated in the tree structure
        return [
            # I guess different annots might want different configs ...
            NotchTipConfig,
            CropChipConfig,
            TrailingEdgeConfig,
            BlockCurvConfig,
        ]

    def get_param_info_list(self):
        return [
            #ut.ParamInfo('score_method', 'csum'),
            # should this be the only thing here?
            #ut.ParamInfo('daids', None),
            ut.ParamInfo('decision', 'average'),
            #ut.ParamInfo('sizes', (5, 10, 15, 20)),
            ut.ParamInfo('weights', None),
            ut.ParamInfo('window', 50),
            #ut.ParamInfo('bcdtwversion', 1),
            ut.ParamInfo('version', 1),
        ]


class BC_DTW_Request(dtool.AlgoRequest):
    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(self, aid_list, config=None):
        """
        import plottool as pt
        pt.ensure_pylab_qt4()
        pt.imshow(overlay_chips[0])
        """
        # FIXME: THIS STRUCTURE OF TELLING HOW FEATURE
        # MATCHES SHOULD BE VISUALIZED IS NOT FINAL.
        depc = self.depc
        chips = depc.get_property('Cropped_Chips', aid_list, 'img', config=config)
        points = depc.get_property('Cropped_Chips', aid_list, ('notch', 'left', 'right'),
                                   config=config)
        tedge_list = depc.get_property('Trailing_Edge', aid_list, 'edge', config=config)
        overlay_chips = [overlay_fluke_feats(chip, path=path, tips=tips)
                         for chip, path, tips in zip(chips, tedge_list, points)]
        return overlay_chips


@register_algo(
    'BC_DTW', algo_result_class=ibeis.AnnotMatch,
    algo_request_class=BC_DTW_Request, configclass=BC_DTW_Config, chunksize=8)
def id_algo_bc_dtw(depc, request):
    r"""
    Args:
        depc (DependencyCache):
        qaid_list (list):

    Yields:
        ibeis.AnnotMatch:

    CommandLine:
        python -m ibeis_flukematch.plugin --exec-id_algo_bc_dtw:0 --show
        python -m ibeis_flukematch.plugin --exec-id_algo_bc_dtw --show --clear-all-depcache
        ibeis -e rank_cdf --db humpbacks -t default:pipeline_root=BC_DTW,crop_enabled=True --qaid-override=1 --daid-override=1,9,15  --show --clear-all-depcache --nocache  --debug-depc
        ibeis -e rank_cdf --db humpbacks -t default:pipeline_root=BC_DTW -a timectrl:has_any=hasnotch --show --nocache
        ibeis -e rank_cdf --db humpbacks -a default:has_any=hasnotch,mingt=2 -t default:pipeline_root=BC_DTW --show  --clear-all-depcache

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids(
        >>>     defaultdb='humpbacks', a='default:has_any=hasnotch,pername=2,mingt=2,size=10')
        >>> ibs.print_annot_stats(aid_list)
        >>> depc = ibs.depc
        >>> qaids = aid_list
        >>> daids = aid_list
        >>> cfgdict = dict(weights=None, decision='average', sizes=(5, 10, 15, 20))
        >>> request = depc.new_algo_request('BC_DTW', qaids, daids, cfgdict)
        >>> # Execute function
        >>> propgen = id_algo_bc_dtw(depc, request)
        >>> am_list = list(propgen)
        >>> print('\n'.join(ut.lmap(str, am_list)))
        >>> result1 = (ut.repr2(np.vstack([am.annot_score_list for am in am_list]), precision=2))
        >>> print(result1)
        >>> am = am_list[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> import ibeis
        >>> ibs, aid_list = ibeis.testdata_aids(
        >>>     defaultdb='humpbacks', a='default:has_any=hasnotch,pername=2,mingt=2,size=10')
        >>> depc = ibs.depc
        >>> qaids = aid_list[0:10]
        >>> daids = aid_list[0:10]
        >>> cfgdict = {'weights': None, 'decision': 'average',
        >>>            'sizes': (5, 10, 15, 20), 'crop_enabled': True}
        >>> request = depc.new_algo_request('BC_DTW', qaids, daids, cfgdict)
        >>> # Execute function
        >>> propgen = id_algo_bc_dtw(depc, request)
        >>> am_list = list(propgen)
        >>> result1 = (ut.repr2(np.vstack([am.annot_score_list for am in am_list]),
        >>>                     precision=2))
        >>> am = am_list[0]
        >>> # Execute via cache
        >>> am_list2 = request.execute()
        >>> result2 = (ut.repr2(np.vstack([am.annot_score_list
        >>>                                for am in am_list2]), precision=2))
        >>> print(result1)
        >>> assert result1 == result2, 'cache does not aggree with non-cache'
    """
    print('Executing BC_DTW')
    print(request)

    qaid_list = request.qaids
    daid_list = request.daids
    config = request.config

    #assert(config['daid_list'] is not None)

    curv_weights = config['weights']
    sizes = config.block_curv_cfg['sizes']
    if curv_weights is not None:
        assert(len(curv_weights) == len(sizes))
    else:
        curv_weights = [1.] * len(sizes)
    ibs = depc.controller

    # Get block curvatures
    #block_config = config.block_curv_cfg
    query_curvs = depc.get_property(
        'Block_Curvature', qaid_list, 'curvature', config=config)
    db_curvs = depc.get_property(
        'Block_Curvature', daid_list, 'curvature', config=config)

    qnid_list = ibs.get_annot_nids(qaid_list)
    dnid_list = ibs.get_annot_nids(daid_list)

    #for qaid, daid in request.get_parent_rowids():
    _iter = zip(query_curvs, qaid_list, qnid_list)
    _progiter = ut.ProgressIter(_iter, lbl='Query BC_DTW',
                                enabled=not ut.QUIET)

    for query_curv, qaid, qnid in _progiter:
        #dists_by_nid = defaultdict(list)
        daid_dists = []
        for db_curv, daid, dnid in zip(db_curvs, daid_list, dnid_list):
            distance = get_distance_curvweighted(query_curv, db_curv,
                                                 curv_weights,
                                                 window=config['window'])
            #score = -1 * distance
            score = np.exp(-distance / 50)
            daid_dists.append(score)
            #dists_by_nid[dnid].append(-1 * distance)

        decision_func = getattr(np, config['decision'])
        #dists_by_nid = {dnid: decision_func(
        #    dists_by_nid[dnid]) for dnid in dists_by_nid}
        #dnid_dists = [dists_by_nid[dnid] for dnid in dnid_list]

        # Remove distance to self
        annot_scores = np.array(daid_dists)
        daid_list_ = np.array(daid_list)
        dnid_list_ = np.array(dnid_list)

        is_valid = (daid_list_ != qaid)
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

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


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    #qaid_list, daid_list = request.get_parent_rowids()
    #score_list = request.score_list
    #config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    #grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)
    # FIXME: decision should not be part of the config for the one-vs-one
    # scores
    decision_func = getattr(np, config['decision'])
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = (daid_list_ != qaid)
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

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


class BC_DTW_NEW_Config(BC_DTW_Config):
    pass


class BC_DTW_NEW_Request(dtool.base.OneVsOneSimilarityRequest):
    _tablename = 'BC_DTW_NEW'
    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, config=None):
        """
        import plottool as pt
        pt.ensure_pylab_qt4()
        pt.imshow(overlay_chips[0])
        """
        # FIXME: THIS STRUCTURE OF TELLING HOW FEATURE
        # MATCHES SHOULD BE VISUALIZED IS NOT FINAL.
        depc = request.depc
        chips = depc.get_property('Cropped_Chips', aid_list, 'img', config=config)
        points = depc.get_property('Cropped_Chips', aid_list, ('notch', 'left', 'right'),
                                   config=config)
        tedge_list = depc.get_property('Trailing_Edge', aid_list, 'edge', config=config)
        overlay_chips = [overlay_fluke_feats(chip, path=path, tips=tips)
                         for chip, path, tips in zip(chips, tedge_list, points)]
        return overlay_chips

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list,
                                         score_list, config))
        return cm_list


@register_preproc(
    tablename='BC_DTW_NEW', parents=[ROOT, ROOT],
    colnames=['score'], coltypes=[float],
    configclass=BC_DTW_NEW_Config,
    requestclass=BC_DTW_NEW_Request,
    chunksize=2056)
def id_algo_bc_dtw_new(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_flukematch.plugin --exec-id_algo_bc_dtw_new:0 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.plugin import *  # NOQA
        >>> import ibeis
        >>> # Setup Inputs
        >>> ibs, aid_list = ibeis.testdata_aids(
        >>>     defaultdb='humpbacks', a='default:has_any=hasnotch,pername=2,mingt=2,size=10')
        >>> depc = ibs.depc
        >>> root_rowids = list(zip(*ut.iprod(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> cfgdict = dict(weights=None, decision='average', sizes=(5, 10, 15, 20))
        >>> config = BC_DTW_NEW_Config(**cfgdict)
        >>> # Call function via request
        >>> request = BC_DTW_NEW_Request.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> # Call function via depcache
        >>> prop_list = depc.get('BC_DTW_NEW', root_rowids, config=config)
        >>> # Call function normally
        >>> score_list = list(id_algo_bc_dtw_new(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    print('Executing BC_DTW_new')
    curv_weights = config['weights']
    sizes = config.block_curv_cfg['sizes']
    if curv_weights is not None:
        assert(len(curv_weights) == len(sizes))
    else:
        curv_weights = [1.] * len(sizes)
    # Group pairs by qaid
    if False:
        unique_qaids, groupxs = ut.group_indices(qaid_list)
        grouped_daids = ut.apply_grouping(daid_list, groupxs)
        # Get block curvatures
        query_curvs = depc.get('Block_Curvature', unique_qaids, 'curvature', config=config)
        _progiter = ut.ProgressIter(
            zip(qaid_list, grouped_daids, query_curvs),
            lbl='Query BC_DTW', enabled=not ut.QUIET)
        for qaid, daids, query_curv in _progiter:
            db_curvs = depc.get('Block_Curvature', daid_list, 'curvature', config=config)
            for daid, db_curv in zip(daids, db_curvs):
                distance = get_distance_curvweighted(query_curv, db_curv,
                                                     curv_weights,
                                                     window=config['window'])
                score = np.exp(-distance / 50)
                yield (score,)
    else:
        all_aids = np.unique(ut.flatten([qaid_list, daid_list]))
        all_curves = depc.get('Block_Curvature', all_aids, 'curvature', config=config)
        aid_to_curves = dict(zip(all_aids, all_curves))
        for qaid, daid in zip(qaid_list, daid_list):
            query_curv = aid_to_curves[qaid]
            db_curv = aid_to_curves[daid]
            distance = get_distance_curvweighted(query_curv, db_curv, curv_weights,
                                                 window=config['window'])
            score = np.exp(-distance / 50)
            yield (score,)


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
