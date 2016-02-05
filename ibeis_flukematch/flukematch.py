# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import ctypes
from os.path import dirname, join
from ibeis_flukematch.kpextractor import build_kpextractor128
import utool as ut
import theano.tensor as T
from theano import function as tfn
import lasagne.layers as ll
from itertools import chain


def setup_kp_network():
    network_params_path = ut.grab_file_url('http://lev.cs.rpi.edu/public/models/kpextractor_weights.pickle', appname='ibs')
    network_params = ut.load_cPkl(network_params_path)
    # network_params also includes normalization constants needed for the dataset, and is assumed to be a dictionary
    # with keys mean, std, and params
    network_exp = build_kpextractor128()
    ll.set_all_param_values(network_exp, network_params['params'])
    X = T.tensor4()
    network_fn = tfn([X], ll.get_output(network_exp, X, deterministic=True))
    return {'mean': network_params['mean'], 'std': network_params['std'], 'networkfn': network_fn}


def bound_output(output, size_mat):
    # make sure the output doessn't exceed the boundaries of the image
    # if it does, snap it to the edge of each dimension it exceeds
    bound_below = np.max(np.stack([output, np.zeros(output.shape)], axis=2), axis=2)
    bound_above = np.min(np.stack([bound_below, size_mat], axis=2), axis=2)
    return bound_above


def infer_kp(img_paths, networkfn, mean, std, batch_size=32, input_size=(128, 128)):
    """
    >>> from ibeis_flukematch.flukematch import *
    >>> pt.imshow(overlay_fluke_feats((img[0] * std + mean), tips=batch_outputs[0] * 128))
    """
    # load up the images in batches
    nbatches = (len(img_paths) // batch_size) + 1
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


def find_trailing_edge_cpp(img, start, end, center, n_neighbors=5):
    # points are x,y
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert(n_neighbors % 2 == 1)
    gradient_y_image = 1 * cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)

    gradient_y_image[:, start[0]] = np.inf
    gradient_y_image[start[1], start[0]] = 0

    gradient_y_image[:, end[0]] = np.inf
    gradient_y_image[end[1], end[0]] = 0

    gradient_y_image[:, center[0]] = np.inf
    gradient_y_image[center[1], center[0]] = 0

    outpath = np.zeros((end[0] - start[0], 2), dtype=np.int32)
    cost = find_te(gradient_y_image, gradient_y_image.shape[0],
                   gradient_y_image.shape[1], start[0], end[1], end[0],
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

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_flukematch.flukematch
        python -m ibeis_flukematch.flukematch --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
