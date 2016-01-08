import cv2
import numpy as np
import ctypes


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


lib = ctypes.cdll.LoadLibrary('./flukematch_lib.so')

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
    cost = find_trailing_edge_cpp(gradient_y_image, gradient_y_image.shape[0],
                                  gradient_y_image.shape[1], start[0], end[1],
                                  end[0], n_neighbors, outpath)
    return outpath, cost

block_curv = lib.block_curvature
block_curv.argtypes = [ndmat_f_type,  # summed_area_table,
                       ctypes.c_int, ctypes.c_int,  # summed_area_table shape
                       ndmat_i_type, ctypes.c_int,  # trailing edge and length
                       ctypes.c_int, ndmat_f_type, ]  # size, output


def block_integral_curvatures_cpp(sizes, coords):
    # assume coords are in x, y
    coords = np.array(coords, dtype=np.int32)
    fit_size = (np.max(coords, axis=0) -
                np.min(coords, axis=0)) + (max(sizes) + 1)
    binarized = np.zeros(fit_size[::-1], dtype=np.float32)
    fixed_coords = np.ascontiguousarray(
        (coords - np.min(coords, axis=0)) + max(sizes) // 2)[:, ::-1]
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
    return curvs

dtw_curvweighted = lib.dtw_curvweighted
dtw_curvweighted.argtypes = [ndmat_f_type, ndmat_f_type,  # curvatures
                             ctypes.c_int, ctypes.c_int,  # lengths
                             ctypes.c_int, ctypes.c_int,  # window, number of sizes
                             ndmat_f_type, ndmat_f_type]  # weights, output


def get_distance_curvweighted(query_curv, db_curv, curv_weights, window=50):
    ordered_sizes = sorted(curv_weights.keys())
    # we just need to stack the curvatures and make sure that the ordering is
    # consistent w/curv_weights
    curv_weights_nd = np.array([curv_weights[i]
                                for i in ordered_sizes], dtype=np.float32)
    query_curv_nd = np.hstack(
        [np.array(query_curv[i], dtype=np.float32).reshape(-1, 1) for i in ordered_sizes])
    db_curv_nd = np.hstack(
        [np.array(db_curv[i], dtype=np.float32).reshape(-1, 1) for i in ordered_sizes])

    query_len = query_curv_nd.shape[0]
    db_len = db_curv_nd.shape[0]
    distance_mat = (np.zeros((query_len, db_len), dtype=np.float32) + np.inf)
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
