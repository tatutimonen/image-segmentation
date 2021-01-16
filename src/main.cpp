#include "Segment.hpp"

#include "opencv2/highgui.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char* const* argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input> <output>" << std::endl;
        return 1;
    }
    const char* input_path = argv[1];
    const char* output_path = argv[2];
    cv::Mat img_in = cv::imread(input_path);

    std::cout << "Segmenting...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r = Segment::segment(img_in);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    std::cout << "Took " << std::setprecision(2) << time_taken.count() << " seconds, writing results...\n";
    
    cv::Mat img_out = cv::Mat(img_in.size(), CV_32FC3, cv::Scalar(r.outer[2], r.outer[1], r.outer[0]));
    cv::rectangle(img_out, cv::Point(r.x0, r.y0), cv::Point(r.x1, r.y1),
                  cv::Scalar(r.inner[2], r.inner[1], r.inner[0]), cv::FILLED);
    cv::imwrite(output_path, img_out*255.0);
    std::cout << "Done." << std::endl;

    return 0;
}
