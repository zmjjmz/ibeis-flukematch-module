# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import ctypes
from os.path import dirname, join
from ibeis_flukematch.networks import (
        # tescorers
        build_segmenter_simple,
        build_segmenter_upsample,
        build_segmenter_jet,
        build_segmenter_jet_2,
        build_segmenter_simple_absurd_res,
        # kpextractors
        build_kpextractor64_decoupled,
        build_kpextractor128_decoupled,
        build_kpextractor256_decoupled,
        )
import utool as ut
import theano.tensor as T
from theano import function as tfn
import lasagne.layers as ll
from itertools import chain
import math

KP_NETWORK_OPTIONS = {
'64_decoupled':{'url':'kpext_64_decoupled.pickle', 'exp':build_kpextractor64_decoupled, 'size':(64,64)},
'128_decoupled':{'url':'kpext_128_decoupled.pickle', 'exp':build_kpextractor128_decoupled, 'size':(128,128)},
'128_decoupled_nofb':{'url':'kpext_128_decoupled_nofb.pickle', 'exp':build_kpextractor128_decoupled, 'size':(128,128)},
'256_decoupled':{'url':'kpext_256_decoupled.pickle', 'exp':build_kpextractor256_decoupled, 'size':(256,256)},
}

def setup_kp_network(network_str):
    fn = KP_NETWORK_OPTIONS[network_str]['url']
    file_url = join('http://lev.cs.rpi.edu/public/models/',fn)
    network_params_path = ut.grab_file_url(file_url, appname='ibeis')
    network_params = ut.load_cPkl(network_params_path)
    # network_params also includes normalization constants needed for the dataset, and is assumed to be a dictionary
    # with keys mean, std, and params
    network_exp = KP_NETWORK_OPTIONS[network_str]['exp']()
    ll.set_all_param_values(network_exp, network_params['params'])
    X = T.tensor4()
    network_fn = tfn([X], ll.get_output(network_exp, X, deterministic=True))
    return {'mean': network_params['mean'], 'std': network_params['std'], 'networkfn': network_fn,
            'input_size':KP_NETWORK_OPTIONS[network_str]['size']}


def bound_output(output, size_mat):
    # make sure the output doessn't exceed the boundaries of the image
    # if it does, snap it to the edge of each dimension it exceeds
    bound_below = np.max(np.stack([output, np.zeros(output.shape)], axis=2), axis=2)
    bound_above = np.min(np.stack([bound_below, size_mat-1], axis=2), axis=2)
    return bound_above


def infer_kp(img_paths, networkfn, mean, std, batch_size=32, input_size=(128, 128)):
    """
    >>> from ibeis_flukematch.flukematch import *
    >>> pt.imshow(overlay_fluke_feats((img[0] * std + mean), tips=batch_outputs[0] * 128))
    """
    # load up the images in batches
    nbatches = int(math.ceil(len(img_paths) / batch_size))
    predictions = []
    for batch_ind in range(nbatches):
        batch_slice = slice(batch_ind * batch_size, (batch_ind + 1) * batch_size)
        print("[infer_kp] Batch %d/%d, processing %d images" % (batch_ind, nbatches, len(img_paths[batch_slice])))
        batch_imgs = []
        batch_sizes = []
        for img_path in img_paths[batch_slice]:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            original_size = img.shape[::-1]
            batch_sizes.append(original_size)
            img = cv2.resize(img, input_size, cv2.INTER_LANCZOS4)
            img = np.expand_dims(img, axis=0)  # add a dummy channel
            # assume zscore normalization
            img = (img.astype(np.float32) - mean) / std
            batch_imgs.append(img)
        nd_imgs = np.stack(batch_imgs, axis=0)
        # get outputs and convert to dict format
        batch_outputs = networkfn(nd_imgs)
        batch_ptdicts = []
        for output, size in zip(batch_outputs, batch_sizes):
            size_mat = np.array([size] * 3, dtype=np.float32).reshape(3, 2)
            output = (output * size_mat).astype(np.int)
            output = bound_output(output, size_mat)
            ptdict = {'left': output[0, :],
                      'right': output[1, :],
                      'notch': output[2, :], }
            batch_ptdicts.append(ptdict)
        predictions = chain(predictions, batch_ptdicts)
    predictions = list(predictions)
    return predictions


TE_NETWORK_OPTIONS = {
'annot_simple':{'url':'tescorer_annot_simple.pickle', 'exp':build_segmenter_simple},
'fbannot_simple':{'url':'tescorer_fbannot_simple.pickle', 'exp':build_segmenter_simple},
'annot_upsample':{'url':'tescorer_annot_upsample.pickle', 'exp':build_segmenter_upsample},
#'fbannot_upsample':{'url':'tescorer_fbannot_upsample.pickle', 'exp':build_segmenter_upsample},
'annot_jet':{'url':'tescorer_annot_jet.pickle', 'exp':build_segmenter_jet},
'fbannot_jet':{'url':'tescorer_fbannot_jet.pickle', 'exp':build_segmenter_jet},
'annot_jet2':{'url':'tescorer_annot_jet2.pickle', 'exp':build_segmenter_jet_2},
'fbannot_jet2':{'url':'tescorer_fbannot_jet2.pickle', 'exp':build_segmenter_jet_2},
#'fbannot_jet_preconv':{'url':'tescorer_fbannot_jet_preconv.pickle', 'exp':build_segmenter_jet_preconv},
'annot_res':{'url':'tescorer_annot_res.pickle', 'exp':build_segmenter_simple_absurd_res},
}

def make_acceptable_shape(acceptable_mult, shape):
    new_shape = []
    for shp in shape:
        if shp % acceptable_mult != 0:
            rem = shp % acceptable_mult
            new_shp = shp - rem
            assert(new_shp != 0 and new_shp % acceptable_mult == 0)
            new_shape.append(new_shp)
        else:
            new_shape.append(shp)
    return tuple(new_shape)

def setup_te_network(network_str):
    fn = TE_NETWORK_OPTIONS[network_str]['url']
    file_url = join('http://lev.cs.rpi.edu/public/models/',fn)
    network_params_path = ut.grab_file_url(file_url, appname='ibeis')
    network_params = ut.load_cPkl(network_params_path)
    # network_params also includes normalization constants needed for the dataset, and is assumed to be a dictionary
    # with keys mean, std, and params
    network_exp = TE_NETWORK_OPTIONS[network_str]['exp']()
    ll.set_all_param_values(network_exp, network_params['params'])
    X = T.tensor4()
    network_fn = tfn([X], ll.get_output(network_exp[-1], X, deterministic=True))
    retdict = {'mean': network_params['mean'], 'std': network_params['std'], 'networkfn': network_fn}
    if any([i in network_str for i in ('upsample', 'jet')]):
        retdict['mod_acc'] = 8
    return retdict

def safe_load(networkfn, img):
    try:
        return networkfn(img)
    except MemoryError:
        print("[score_te] ERROR: GPU ran out of memory trying to process an image of size %r" % (img.shape,))
        return None

def score_te(img_paths, networkfn, mean, std, mod_acc=None, batch_size=32, input_size=None):
    # load up the images in batches
    nbatches = int(math.ceil(len(img_paths) / batch_size))
    predictions = []
    for batch_ind in range(nbatches):
        batch_slice = slice(batch_ind * batch_size, (batch_ind + 1) * batch_size)
        print("[score_te] Batch %d/%d, processing %d images" % (batch_ind, nbatches, len(img_paths[batch_slice])))
        batch_imgs = []
        batch_sizes = []
        for img_path in img_paths[batch_slice]:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            original_size = img.shape[::-1]
            batch_sizes.append(original_size)
            if mod_acc is not None:
                new_shape = make_acceptable_shape(mod_acc, img.shape)
                img = cv2.resize(img, new_shape[::-1], cv2.INTER_LANCZOS4)

            if input_size is not None:
                # note that input_size should be w, h
                img = cv2.resize(img, input_size, cv2.INTER_LANCZOS4)
            img = np.expand_dims(img, axis=0)  # add a dummy channel
            # assume zscore normalization
            img = (img.astype(np.float32) - mean) / std
            batch_imgs.append(img)
        # we only want the background probabilities
        bg_ind = 1
        if input_size is not None:
            nd_imgs = np.stack(batch_imgs, axis=0)
            batch_outputs = safe_load(networkfn, nd_imgs)
            # denumpy it
            if batch_outputs is not None:
                batch_outputs = [i[bg_ind] for i in batch_outputs]
        else:
            nd_imgs = [np.expand_dims(img, axis=0) for img in batch_imgs]
            # denumpy it
        batch_outputs = [safe_load(networkfn, img) for img in nd_imgs]
        # resize it to the original img shape if necessary
        batch_outputs_r = []
        for label, size in zip(batch_outputs, batch_sizes):
            if label is None:
                batch_outputs_r.append(label)
                continue
            # img.shape is (1, h, w)
            # label.shape is (1, 2, h, w)
            if label.shape[2:] != img.shape[1:]:
                batch_outputs_r.append(cv2.resize(label[0][bg_ind], size, cv2.INTER_LANCZOS4))
            else:
                batch_outputs_r.append(label[0][bg_ind])

        predictions = chain(predictions, batch_outputs_r)
    predictions = list(predictions)
    return predictions




def find_trailing_edge(img, start, end, center=None, n_neighbors=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert(n_neighbors % 2 == 1)
    #n_neighbors = img.shape[0]
    neighbor_range = range(-1 * (n_neighbors // 2), 1 + (n_neighbors // 2))
    # start and end are x,y
    # take the vertical gradients of the image
    gradient_y_image = 1 * cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # TODO maybe: blur before taking gradients to denoise
    # set the start point's vertical gradient to a very high value
    # for each row from start to end, the cost of each cell is the max
    # of the three neighbors to its left + its own cost
    # the max that is chosen is then stored for each pixel in a column so we
    # can backtrack
    # make sure this is where the path starts
    gradient_y_image[:, start[0]] = np.inf
    # np.max(gradient_y_image) + 1 # yeah I think this makes sense
    gradient_y_image[start[1], start[0]] = 0

    gradient_y_image[:, end[0]] = np.inf
    gradient_y_image[end[1], end[0]] = 0  # np.max(gradient_y_image) + 1
    if center is not None:
        # force it to go through the center
        gradient_y_image[:, center[0]] = np.inf
        gradient_y_image[center[1], center[0]] = 0
    # the goal of the above is to make the path end at 'end', but that
    # probably can't be guaranteed regardless
    cost = np.zeros(gradient_y_image.shape)
    back = np.zeros(gradient_y_image.shape, dtype=np.int32)
    def get_cost(row, col, i):
        return (
            np.inf
            if ((row + i < 0) or (row + i >= gradient_y_image.shape[0])) else
            cost[row + i, col - 1] + gradient_y_image[row, col]
        )
    for col in range(start[0], end[0] + 1):
        # this is the slow part
        for row in range(gradient_y_image.shape[0]):
            # candidates = [((cost[cell+i, col-1]+gradient_y_image[cell,col]) if
            #              ((cell+i > 0) and (cell+i < gradient_y_image.shape[0])) else -np.inf)
            #              for i in (-1,0,1)]
            candidates = [get_cost(row, col, i) for i in neighbor_range]
            best = np.argmin(candidates)
            # print(candidates)
            back[row, col] = best - (n_neighbors // 2)
            cost[row, col] = candidates[best]
    # backtrack the seam
    path = []  # we know that the path is from end to start so we don't need to store the x values
    curr_y = end[1]
    path_cost = 0
    # we know that the optimal path must end at the end point since otherwise
    # its cost is -inf
    for col in range(start[0], end[0] + 1)[::-1]:
        path_cost += cost[curr_y, col]
        path.append((col, curr_y))
        next_ = curr_y + back[curr_y, col]
        curr_y = next_
    # plt.imshow(cost)
    return path, path_cost, cost


try:
    lib = ctypes.cdll.LoadLibrary(join(dirname(__file__), 'flukematch_lib.so'))
    HAS_LIB = True
except Exception as ex:
    ut.printex(ex, iswarning=True)
    HAS_LIB = False


if HAS_LIB:
    ndmat_f_type = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
    ndmat_i_type = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')

    find_te = lib.find_trailing_edge
    find_te.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,  # image and size info
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # startcol, endrow, endcol
                        ctypes.c_int, ndmat_i_type]  # number of neighbors, output path
    find_te.restype = ctypes.c_float


def normalize_01(img):
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm_img


def find_trailing_edge_cpp(img, start, end, center, n_neighbors=5, ignore_notch=False, score_mat=None,
                           score_method='avg', score_weight=0.5, tol=None):
    # points are x,y
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert(n_neighbors % 2 == 1)
    gradient_y_image = 1 * cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    # normalize so that all gradients lie in -1, 0
    norm_grad = normalize_01(gradient_y_image)
    # so now we still want to be minimizing, which means that the score_mat
    # can be directly used (as it is going to be in 0,1 probability of it being bg)

    # next question: how do we factor in the score?
    # Option 1: Just blend it in
    if score_mat is not None:
        try:
            assert(score_mat.shape == norm_grad.shape)
        except AssertionError:
            print(score_mat.shape)
            print(norm_grad.shape)
            raise
        if score_method == 'avg':
            score_grad = score_weight*score_mat + (1-score_weight)*norm_grad
        elif score_method == 'allow':
            # basically make everything that isn't classified as being part of the trailing edge (using score_weight as the thresh)
            # into np.inf
            # YUUUGGE EROSION (i.e. dilation but inverted)
            erosion_k = np.ones((20,img.shape[1]),dtype=np.float32)
            score_mat = cv2.erode(score_mat, erosion_k)

            norm_grad[np.where(score_mat > 0.5)] = np.inf # since we're given the bg classification
            score_grad = norm_grad
            # this is dangerous, since if we miss any of the vital points (i.e. start / end) or disconnect them from the rest
            # of the trailing edge, we'll end up w/disastrous results. If we keep the threshold low we should (big should)
            # be able to avoid this
        elif score_method == 'avg_thresh':
            # instead of giving a direct average, we'll basically snap a (inverted) pixel in score_mat to 0 if it's above the threshold
            # and snap it to 1 if it's below the threshold, and then average. This should emphasize those classifications that are
            # part of the predicted trailing edge without producing artifacts from unconfident predictions
            thresholded_mat = np.copy(score_mat)
            thresholded_mat[np.where(score_mat < 0.5)] = 0
            thresholded_mat[np.where(score_mat > 0.5)] = 1
            score_grad = score_weight*score_mat + (1-score_weight)*norm_grad
        #score_grad = np.average(np.stack([norm_grad, score_mat],axis=0),axis=0)
    else:
        score_grad = norm_grad


    try:
        score_grad[:, start[0]] = 1 * np.inf
        score_grad[start[1], start[0]] = 0

        score_grad[:, end[0]] = 1 * np.inf
        score_grad[end[1], end[0]] = 0

        if not ignore_notch:
            score_grad[:, center[0]] = 1 * np.inf
            score_grad[center[1], center[0]] = 0

        if tol is not None:
            # between start, end, and center find the highest / lowest points
            # we're then going to bound the trailing edge below and above these
            lowest_point = max(start[1], center[1], end[1])
            bound_low = min(lowest_point + int(img.shape[0] * (tol / 100)) + 1, img.shape[0])

            highest_point = min(start[1], center[1], end[1])
            bound_high = max(highest_point - int(img.shape[0] * (tol / 100)) - 1, 0)

            score_grad[bound_low:,:] = 1 * np.inf
            score_grad[:bound_high,:] = 1 * np.inf
    except IndexError:
        print("[find_te] Bad points: start: %s, end: %s, center: %s" % (start, end, center))
        raise

    outpath = np.zeros((end[0] - start[0], 2), dtype=np.int32)
    cost = find_te(score_grad, score_grad.shape[0],
                   score_grad.shape[1], start[0], end[1], end[0],
                   n_neighbors, outpath)
    return outpath, cost

if HAS_LIB:
    block_curv = lib.block_curvature
    block_curv.argtypes = [ndmat_f_type,  # summed_area_table,
                           ctypes.c_int, ctypes.c_int,  # summed_area_table shape
                           ndmat_i_type, ctypes.c_int,  # trailing edge and length
                           ctypes.c_int, ndmat_f_type, ]  # size, output


def block_integral_curvatures_cpp(sizes, coords):
    """
        >>> from ibeis_flukematch.flukematch import *  # NOQA
    """
    # assume coords are in x, y
    coords = np.array(coords, dtype=np.int32)
    sizes = map(lambda x: int(math.ceil(coords.shape[0]*x)), sizes)

    fit_size = (np.max(coords, axis=0) -
                np.min(coords, axis=0)) + (max(sizes) + 1)
    binarized = np.zeros(fit_size[::-1], dtype=np.float32)
    fixed_coords = np.array(
        (coords - np.min(coords, axis=0)) + max(sizes) // 2)[:, ::-1]
    fixed_coords = np.ascontiguousarray(fixed_coords)
    binarized[zip(*fixed_coords)] = 1
    binarized = binarized.cumsum(axis=0)
    binarized[np.where(binarized > 0)] = 1
    summed_table = binarized.cumsum(axis=0).cumsum(axis=1)
    curvs = {}

    #coords_flat = fixed_coords.flatten()
    #sat_flat = summed_table.flatten()
    for size in sizes:
        # compute curvature using separate calls to block_curv for each
        curvs[size] = np.zeros((fixed_coords.shape[0], 1), dtype=np.float32)
        block_curv(summed_table, summed_table.shape[0], summed_table.shape[1],
                   fixed_coords, fixed_coords.shape[0], size, curvs[size])
    #return curvs

    curv_arr = np.concatenate([curvs[size] for size in sizes], axis=1)
    return curv_arr

if HAS_LIB:
    dtw_curvweighted = lib.dtw_curvweighted
    dtw_curvweighted.argtypes = [ndmat_f_type, ndmat_f_type,  # curvatures
                                 ctypes.c_int, ctypes.c_int,  # lengths
                                 ctypes.c_int, ctypes.c_int,  # window, number of sizes
                                 ndmat_f_type, ndmat_f_type]  # weights, output


def get_distance_curvweighted(query_curv, db_curv, curv_weights, window=50):
    """
    Ignore:
        aid1 = ibs.get_image_aids(ibs.get_valid_gids()[ibs.get_image_gnames(ibs.get_valid_gids()).index('CINMS-20090307-B9984.jpg')])[0]
        aid2 = ibs.get_image_aids(ibs.get_valid_gids()[ibs.get_image_gnames(ibs.get_valid_gids()).index('CINMS_20100427_A1860.jpg')])[0]

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_flukematch.flukematch import *  # NOQA
        >>> import ibeis
        >>> #from ibeis_flukematch.plugin import *  # NOQA
        >>> ibs = ibeis.opendb(defaultdb='humpbacks')
        >>> all_aids = ibs.get_valid_aids()
        >>> isvalid = ibs.depc.get_property('Has_Notch', all_aids, 'flag', _debug=True)
        >>> aid = ut.compress(all_aids, isvalid)[0]
        >>> aid1 = 2826
        >>> aid2 = 2827
        >>> query_curv_nd, db_curv_nd = depc.get_property('Block_Curvature', [aid1, aid2], 'curvature')
        >>> depc = ibs.depc
        >>> window = 50
        >>> sizes = [5, 10, 15, 20]
        >>> curv_weights_nd = np.array([1.] * len(sizes))
        >>> get_distance_curvweighted(query_curv_nd, db_curv_nd, curv_weights, window=50)
        >>> print(curve_arr.shape())
        >>> print(curve_arr.sum())
        >>> window=50
    """
    #ordered_sizes = sorted(curv_weights.keys())
    # we just need to stack the curvatures and make sure that the ordering is
    # consistent w/curv_weights
    #curv_weights_nd = np.array([curv_weights[i]
    #                            for i in ordered_sizes], dtype=np.float32)
    curv_weights_nd = np.array(curv_weights, dtype=np.float32).reshape(-1, 1)
    curv_weights_nd = np.ascontiguousarray(curv_weights_nd)
    query_curv_nd = query_curv
    db_curv_nd = db_curv
    #query_curv_nd = np.hstack(
    #    [np.array(query_curv[i], dtype=np.float32).reshape(-1, 1) for i in ordered_sizes])
    #db_curv_nd = np.hstack(
    #    [np.array(db_curv[i], dtype=np.float32).reshape(-1, 1) for i in ordered_sizes])

    query_len = query_curv_nd.shape[0]
    db_len = db_curv_nd.shape[0]
    #distance_mat = (np.zeros((query_len, db_len), dtype=np.float32) + np.inf)
    distance_mat = np.full((query_len, db_len), np.inf, dtype=np.float32)
    distance_mat[0, 0] = 0
    dtw_curvweighted(
        query_curv_nd, db_curv_nd, query_len, db_len, window,
        curv_weights_nd.shape[0], curv_weights_nd, distance_mat)
    distance = distance_mat[-1, -1]
    return distance

def curv_weight_gen(rel_importance, sizes):
    # ok, so let's do this: basically each curvature is weighted as rel_importance compared to the previous one
    # so essentially we're doing each weight as rel_importance^(i)
    # then we'll divide each by the sum of the weights to keep the sum of these weights at one while preserving the ratio
    weights = [1]
    for _ in range(1,len(sizes)): # reduce by one to account for the first weight
        weights.append(weights[-1] * rel_importance)
    weights = map(lambda x: x / sum(weights), weights)
    return weights

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_flukematch.flukematch
        python -m ibeis_flukematch.flukematch --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
