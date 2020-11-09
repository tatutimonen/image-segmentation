#include "segment.hpp"

//----------------------------------------------------------------------------
// Auxiliary data structure for storing intermediate results along with the
// corresponding utility.

struct ResultAux {
    segment::Result res;
    double util;
};

//----------------------------------------------------------------------------
// Computes the sum over the RGB values in a vector register.

static inline double sum3(const __m256d& x)
{
    double res = 0;
    for (int k = 0; k < 3; ++k) {
        res += x.m256d_f64[k];
    }
    return res;
}

//----------------------------------------------------------------------------

namespace segment
{
Result segment(const cv::Mat& image)
{
    using aligned_alloc::alloc, aligned_alloc::free;

    const int ny = image.rows;
    const int nx = image.cols;
    const int nc = image.channels();

    // Vectorize and rescale the data.
    __m256d* data_vec = alloc(ny*nx);
    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            int idx_r = i*nx*nc + j*nc;
            int idx_w = i*nx + j;
            data_vec[idx_w] = _mm256_set_pd(
                                  0.0,                   // A
                                  image.data[idx_r + 0], // B
                                  image.data[idx_r + 1], // G
                                  image.data[idx_r + 2]  // R
                              );
            data_vec[idx_w] = _mm256_div_pd(data_vec[idx_w], _mm256_set1_pd(255.0));
        }
    }

    // Cumulate row-wise sums.
    __m256d* cum_row_sums = alloc((ny+1)*(nx+1));
    for (int j = 0; j < nx+1; ++j) {
        cum_row_sums[j] = _mm256_setzero_pd();
    }
    #pragma omp parallel for
    for (int i = 1; i < ny+1; ++i) {
        cum_row_sums[i*(nx+1)] = _mm256_setzero_pd();
        for (int j = 0; j < nx; ++j) {
            cum_row_sums[i*(nx+1) + j+1] = _mm256_add_pd(cum_row_sums[i*(nx+1) + j], data_vec[(i-1)*nx + j]);
        }
    }

    // Create a summed-area table to enable computing arbitrary rectangles in O(1) time.
    __m256d* sum_table = alloc((ny+1)*(nx+1));
    for (int j = 0; j < nx+1; ++j) {
        sum_table[j] = _mm256_setzero_pd();
    }
    for (int i = 1; i < ny+1; ++i) {
        for (int j = 0; j < nx+1; ++j) {
            sum_table[i*(nx+1) + j] = _mm256_add_pd(sum_table[(i-1)*(nx+1) + j], cum_row_sums[i*(nx+1) + j]);
        }
    }
    auto last = sum_table[ny*(nx+1) + nx];

    // Compute all possible segmentations and return the one minimizing the SSE.
    ResultAux global_max = {};
    #pragma omp parallel
    {
        ResultAux thread_max = {};
        #pragma omp for schedule(dynamic)
        for (int dy = 1; dy < ny+1; ++dy) {
            for (int dx = 1; dx < nx+1; ++dx) {
                auto card_X_inv = _mm256_set1_pd(1.0/(dy*dx));
                auto card_Y_inv = _mm256_set1_pd(ny*nx - dy*dx > 0 ? 1.0/(ny*nx - dy*dx) : 0.0);
                auto A = _mm256_add_pd(card_X_inv, card_Y_inv);
                auto B = _mm256_mul_pd(last, _mm256_mul_pd(_mm256_set1_pd(-2), card_Y_inv));
                auto C = _mm256_mul_pd(_mm256_mul_pd(last, last), card_Y_inv);
                for (int y1 = dy; y1 < ny+1; ++y1) {
                    int y0 = y1 - dy;
                    for (int x1 = dx; x1 < nx+1; ++x1) {
                        int x0 = x1 - dx;
                        auto vx = _mm256_add_pd(
                                      _mm256_sub_pd(
                                          _mm256_sub_pd(
                                              sum_table[y1*(nx+1) + x1],
                                              sum_table[y1*(nx+1) + x0]
                                          ),
                                          sum_table[y0*(nx+1) + x1]
                                      ),
                                      sum_table[y0*(nx+1) + x0]
                                  );
                        auto h4 = _mm256_add_pd(
                                      _mm256_mul_pd(
                                          _mm256_add_pd(
                                              _mm256_mul_pd(
                                                  vx,
                                                  A
                                              ),
                                              B
                                          ),
                                          vx
                                      ),
                                      C
                                  );
                        auto a = _mm256_mul_pd(vx, card_X_inv);
                        auto b = _mm256_mul_pd(_mm256_sub_pd(last, vx), card_Y_inv);
                        double h = sum3(h4);
                        if (h > thread_max.util) {
                            thread_max.res = {
                                y0,
                                x0,
                                y1,
                                x1,
                                {(float)b.m256d_f64[0], (float)b.m256d_f64[1], (float)b.m256d_f64[2]},
                                {(float)a.m256d_f64[0], (float)a.m256d_f64[1], (float)a.m256d_f64[2]}
                            };
                            thread_max.util = h;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (thread_max.util > global_max.util) {
                global_max.res = thread_max.res;
                global_max.util = thread_max.util;
            }
        }
    }

    free(data_vec);
    free(cum_row_sums);
    free(sum_table);

    return global_max.res;
}
} // namespace segment

//----------------------------------------------------------------------------
