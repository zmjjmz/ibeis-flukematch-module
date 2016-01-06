#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
//#include <stdint.h>
#include <ctime>
//#include <cstdint>
#include <Eigen/Dense>
#include <iostream>
//#include <cblas>
// this is bad don't do this kids
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//#define DLLEXPORT extern "C" __declspec(dllexport)

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> NDArrayFlattened;
typedef Map<NDArrayFlattened> ExternNDArrayf;

typedef Matrix<int, Dynamic, Dynamic, RowMajor> NDArrayFlattenedi;
typedef Map<NDArrayFlattenedi> ExternNDArrayi;

extern "C" void dtw_curvweighted(float * seq1curvv, float * seq2curvv,
                                 int seq1_len, int seq2_len, int window,
                                 int curv_hist_size, float * curv_hist_weightsv,
                                 float * distmat_outv) {

    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    ExternNDArrayf distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    ExternNDArrayf seq1curv((float*)seq1curvv, seq1_len, curv_hist_size);
    ExternNDArrayf seq2curv((float*)seq2curvv, seq2_len, curv_hist_size);


    ExternNDArrayf curv_hist_weights((float*)curv_hist_weightsv, curv_hist_size, 1);

    float dist;
    for (int i = 1; i < seq1_len; i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2_len, i + window); j++) {
            dist = ((seq1curv.row(i).array() * curv_hist_weights.transpose().array()) - 
                    (seq2curv.row(j).array() * curv_hist_weights.transpose().array())).matrix().norm();
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
        }
    }
}

extern "C" void block_curvature(float * summed_area_tabv, int binarized_rows, int binarized_cols,
                                int * seq_posv, int seq_len, 
                                int curvature_size, float * curvature_vecv) {

    ExternNDArrayf summed_area_tab((float*)summed_area_tabv, binarized_rows, binarized_cols);
    ExternNDArrayi seq_pos((int*)seq_posv, seq_len, 2);
    ExternNDArrayf curvature_vec((float*)curvature_vecv, seq_len, 1);
    float area = std::pow((float)curvature_size,2.0);

    for (int ind = 0; ind < seq_len; ind++) {
        // assume y, x
        int i = seq_pos(ind, 0);
        int j = seq_pos(ind, 1);
        int starti = MAX(0, i - (curvature_size / 2));
        int startj = MAX(0, j - (curvature_size / 2));
        int endi = MIN(binarized_rows - 1, i + (curvature_size / 2));
        int endj = MIN(binarized_cols - 1, j + (curvature_size / 2));
        //printf("i %d\n j %d\n", i, j);
        //printf("starti %d\nstartj %d\nendi %d\nendj %d\n", starti, startj, endi, endj);
        float this_summed_area = (float)(summed_area_tab(starti, startj) + summed_area_tab(endi, endj) -
                            (summed_area_tab(starti, endj) + summed_area_tab(endi, startj)));
        this_summed_area /= area; // may be a little wrong on the edges, but we shouldn't need to worry about that
        //printf("Point (%d, %d) has curvature %0.2f at size %d\n", i, j, this_summed_area, curvature_size);
        curvature_vec(ind) = this_summed_area; 
    }
}
