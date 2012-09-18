#include "super_resolution.hpp"
#include "exampled_based.hpp"

using namespace std;
using namespace cv;

Ptr<SuperResolution> SuperResolution::create(Method method)
{
    typedef Ptr<SuperResolution> (*func_t)();
    static const func_t funcs[] =
    {
        ExampledBased::create
    };

    CV_Assert(method >= EXAMPLE_BASED && method < METHOD_MAX);

    return funcs[method]();
}

SuperResolution::~SuperResolution()
{
}

void SuperResolution::train(const Mat& image)
{
    vector<Mat> images(1);
    images[0] = image;
    train(images);
}
