#include "segment.hpp"

#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>

//----------------------------------------------------------------------------

int main(int argc, char* const* argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <input> <output>" << std::endl;
        return 1;
    }
    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    const cv::Mat imgIn = cv::imread(inputPath);

    std::cout << "Segmenting...\n";

    using namespace std::chrono_literals;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto r = segment::segment(imgIn);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    
    std::cout << "Took " << std::setprecision(2) << timeTaken.count() << " seconds, writing results...\n";
    
    const cv::Mat imgOut = cv::Mat(imgIn.size(), CV_32FC3, cv::Scalar(r.outer[2], r.outer[1], r.outer[0]));
    cv::rectangle(imgOut, cv::Point(r.x0, r.y0), cv::Point(r.x1, r.y1),
                  cv::Scalar(r.inner[2], r.inner[1], r.inner[0]), cv::FILLED);
    cv::imwrite(outputPath, imgOut * 255.0);

    std::cout << "Done." << std::endl;

    return 0;
}

//----------------------------------------------------------------------------
