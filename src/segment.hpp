#pragma once

#include "aligned_alloc.hpp"

#include <opencv2/imgproc.hpp>

extern "C" {
#include <omp.h>
}

//----------------------------------------------------------------------------

namespace segment
{
    
// Data structure storing the two defining cornerpoints of the rectangle, along
// with its color and the color of the background.
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

// Computes the utility of every possible segmentation in a brute-force manner
// and returns the optimal one.
Result segment(const cv::Mat& image);

} // namespace segment

//----------------------------------------------------------------------------