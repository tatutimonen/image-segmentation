#include "segment.hpp"

#include <vector>

//----------------------------------------------------------------------------
// Stores intermediate results along with the corresponding utility.

struct ResultAux
{
    segment::Result res;
    double util;
};

//----------------------------------------------------------------------------
// Computes the sum over the RGB values in a vector register.

static inline double sum3(const __m256d& x)
{
    double res = 0;
    for (int k = 0; k < 3; ++k)
    {
#ifdef _WIN32
        res += x.m256d_f64[k];
#else
        res += x[k];
#endif
    }
    return res;
}

//----------------------------------------------------------------------------

namespace segment
{

//----------------------------------------------------------------------------

Result segment(const cv::Mat& image)
{
    const int ny = image.rows;
    const int nx = image.cols;
    const int nc = image.channels();

    // Vectorize and rescale the data.
    std::vector<__m256d> dataVec;
    dataVec.resize(ny * nx);

    #pragma omp parallel for
    for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nx; ++j)
    {
        int readIdx = i*nx*nc + j*nc;
        int writeIdx = i*nx + j;
        dataVec[writeIdx] = _mm256_set_pd(
            0.0,                     // A
            image.data[readIdx + 0], // B
            image.data[readIdx + 1], // G
            image.data[readIdx + 2]  // R
        );
        dataVec[writeIdx] = _mm256_div_pd(dataVec[writeIdx], _mm256_set1_pd(255.0));
    }

    // Create a summed-area table to enable computing arbitrary rectangles in O(1) time.
    std::vector<__m256d> sumTable;
    sumTable.resize((ny+1) * (nx+1));

    for (int j = 0; j < nx+1; ++j)
        sumTable[j] = _mm256_setzero_pd();

    #pragma omp parallel for
    for (int i = 1; i < ny+1; ++i)
    {
        sumTable[i*(nx+1)] = _mm256_setzero_pd();
        for (int j = 0; j < nx; ++j)
        {
            sumTable[i*(nx+1) + j+1] = _mm256_add_pd(sumTable[i*(nx+1) + j], dataVec[(i-1)*nx + j]);
        }
    }

    for (int i = 1; i < ny+1; ++i)
    for (int j = 0; j < nx+1; ++j)
    {
        sumTable[i*(nx+1) + j] = _mm256_add_pd(sumTable[(i-1)*(nx+1) + j], sumTable[i*(nx+1) + j]);
    }

    auto last = sumTable[ny*(nx+1) + nx];

    // Compute all possible segmentations and return the one minimizing the SSE.
    ResultAux globalMax{};
    #pragma omp parallel
    {
        ResultAux threadMax{};
        #pragma omp for schedule(dynamic)
        for (int dy = 1; dy < ny+1; ++dy)
        for (int dx = 1; dx < nx+1; ++dx)
        {
            auto invCardX = _mm256_set1_pd(1.0/(dy*dx));
            auto invCardY = _mm256_set1_pd(ny*nx - dy*dx > 0 ? 1.0/(ny*nx - dy*dx) : 0.0);
            auto A = _mm256_add_pd(invCardX, invCardY);
            auto B = _mm256_mul_pd(last, _mm256_mul_pd(_mm256_set1_pd(-2), invCardY));
            auto C = _mm256_mul_pd(_mm256_mul_pd(last, last), invCardY);

            for (int y1 = dy; y1 < ny+1; ++y1)
            {
                int y0 = y1 - dy;
                for (int x1 = dx; x1 < nx+1; ++x1)
                {
                    int x0 = x1 - dx;
                    
                    auto vx = _mm256_add_pd(
                                  _mm256_sub_pd(
                                      _mm256_sub_pd(
                                          sumTable[y1*(nx+1) + x1],
                                          sumTable[y1*(nx+1) + x0]
                                      ),
                                      sumTable[y0*(nx+1) + x1]
                                  ),
                                  sumTable[y0*(nx+1) + x0]
                              );
                    auto h4 = _mm256_fmadd_pd(_mm256_fmadd_pd(vx, A, B), vx, C);
                    auto a = _mm256_mul_pd(vx, invCardX);
                    auto b = _mm256_mul_pd(_mm256_sub_pd(last, vx), invCardY);
                    double h = sum3(h4);

                    if (h > threadMax.util)
                    {
                        threadMax.res = {
                            y0,
                            x0,
                            y1,
                            x1,
#ifdef _WIN32
                            { (float)b.m256d_f64[0], (float)b.m256d_f64[1], (float)b.m256d_f64[2] },
                            { (float)a.m256d_f64[0], (float)a.m256d_f64[1], (float)a.m256d_f64[2] }
#else
                            { (float)b[0], (float)b[1], (float)b[2] },
                            { (float)a[0], (float)a[1], (float)a[2] }
#endif
                        };
                        threadMax.util = h;
                    }
                }
            }
        }
        #pragma omp critical
        {
            if (threadMax.util > globalMax.util)
            {
                globalMax.res = threadMax.res;
                globalMax.util = threadMax.util;
            }
        }
    }

    return globalMax.res;
}

//----------------------------------------------------------------------------

}  // namespace segment

//----------------------------------------------------------------------------
