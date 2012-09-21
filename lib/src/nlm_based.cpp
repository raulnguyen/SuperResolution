#include "nlm_based.hpp"
#include <opencv2/core/internal.hpp>
#include "extract_patch.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

CV_INIT_ALGORITHM(NlmBased, "VideoSuperResolution.NlmBased",
                  obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                       "Scale factor.");
                  obj.info()->addParam(obj, "searchAreaRadius", obj.searchAreaRadius, false, 0, 0,
                                       "Radius of the patch search area in low resolution image (-1 means whole image).");
                  obj.info()->addParam(obj, "timeRadius", obj.timeRadius, false, 0, 0,
                                       "Radius of the time search area.");
                  obj.info()->addParam(obj, "lowResPatchSize", obj.lowResPatchSize, false, 0, 0,
                                       "Size of tha patch at in low resolution image.");
                  obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0,
                                       "Weight of the patch difference");
                  obj.info()->addParam(obj, "doDeblurring", obj.doDeblurring, false, 0, 0,
                                       "Performs deblurring operation"));

bool NlmBased::init()
{
    return !NlmBased_info_auto.name().empty();
}

Ptr<VideoSuperResolution> NlmBased::create()
{
    return Ptr<VideoSuperResolution>(new NlmBased);
}

NlmBased::NlmBased()
{
    scale = 2;
    searchAreaRadius = 15;
    timeRadius = 7;
    lowResPatchSize = 7;
    sigma = 7.5;
    doDeblurring = true;

    motionEstimator = new KeypointBasedMotionEstimator(new MotionEstimatorRansacL2());
    deblurer = new WeightingDeblurer;

    deblurer->setFrames(Y);
    deblurer->setMotions(motions);
    deblurer->setBlurrinessRates(blurrinessRates);
}

void NlmBased::resetImpl()
{
    Y.clear();
    y.clear();
}

void NlmBased::initImpl(cv::Ptr<IFrameSource>& frameSource)
{
    y.resize(2 * timeRadius + 2);
    Y.resize(2 * timeRadius + 2);

    if (doDeblurring)
    {
        deblurer->setRadius(timeRadius);
        motions.resize(2 * timeRadius + 2);
        blurrinessRates.resize(2 * timeRadius + 2);
    }

    curPos = -1;
    curOutPos = -(2 * timeRadius + 1);

    for (int t = -timeRadius; t <= timeRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();

        if (frame.empty())
            return;

        addNewFrame(frame, true);
    }

    if (doDeblurring)
    {
        for (int i = 0; i <= 2 * timeRadius; ++i)
        {
            if (i < 2 * timeRadius)
                motions[i] = motionEstimator->estimate(Y[i], Y[i + 1]);
            blurrinessRates[i] = calcBlurriness(Y[i]);
        }
    }
}

Mat NlmBased::processImpl(const Mat& frame)
{
    const int s = scale;           // the desired scaling factor
    const int q = lowResPatchSize; // the size of the low resolution patch
    const int p = s * (q - 1) + 1; // the size of the high resolution patch
    const double weightScale = 1.0 / (2.0 * sigma * sigma);

    addNewFrame(frame);

    Mat& curY = at(curOutPos, Y);

    curY.convertTo(V, CV_64F);
    W.create(V.size());
    W.setTo(Scalar::all(1));

    for (int k = 0; k < V.rows; ++k)
    {
        for (int l = 0; l < V.cols; ++l)
        {
            for (int t = -timeRadius; t <= timeRadius; ++t)
            {
                const int iStart = searchAreaRadius == -1 ? 0 : k / s - searchAreaRadius;
                const int iEnd = searchAreaRadius == -1 ? y.front().rows : k / s + searchAreaRadius + 1;

                for (int i = iStart; i < iEnd; ++i)
                {
                    const int jStart = searchAreaRadius == -1 ? 0 : l / s - searchAreaRadius;
                    const int jEnd = searchAreaRadius == -1 ? y.front().cols : l / s + searchAreaRadius + 1;

                    for (int j = jStart; j < jEnd; ++j)
                    {
                        // Compute Weight

                        extractPatch(curY, Point2d(l, k), patch1, p, INTER_NEAREST);
                        extractPatch(Y[t + timeRadius], Point2d(s * j, s * i), patch2, p, INTER_NEAREST);

                        double norm2 = 0.0;
                        for (size_t n = 0; n < patch1.size(); ++n)
                        {
                            const double diff = patch1[n] - patch2[n];
                            norm2 += diff * diff;
                        }

                        const double w = exp(-norm2 * weightScale);

                        // Accumulate Inputs & Weights

                        // Extract the low-resolution patch
                        extractPatch(y[t + timeRadius], Point2d(j, i), patch1, q, INTER_NEAREST);

                        // Accumulate it in its proper location
                        Mat_<Point3d> lowResPatch(q, q, (Point3d*) &patch1[0]);
                        for (int y = 0; y < q; ++y)
                        {
                            const int hy = k + s * (y - q / 2);

                            if (hy >= 0 && hy < V.rows)
                            {
                                for (int x = 0; x < q; ++x)
                                {
                                    const int hx = l + s * (x - q / 2);

                                    if (hx >= 0 && hx < V.cols)
                                    {
                                        const Point3d val = lowResPatch(y, x);

                                        V(hy, hx) += w * val;
                                        W(hy, hx) += Point3d(w, w, w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalization
    divide(V, W, Z);

    Z.convertTo(dst, CV_8U);
    dst.copyTo(curY);

    // Deblurring
    if (doDeblurring)
    {
        deblurer->deblur(curOutPos, dst);
        dst.copyTo(curY);
    }

    return dst;
}

void NlmBased::addNewFrame(const cv::Mat& frame, bool init)
{
    ++curPos;
    ++curOutPos;

    frame.copyTo(at(curPos, y));

    Mat& highRes = at(curPos, Y);
    pyrUp(frame, highRes, Size(frame.cols * scale, frame.rows * scale));

    if (!init && doDeblurring)
    {
        at(curPos - 1, motions) = motionEstimator->estimate(at(curPos - 1, Y), highRes);
        at(curPos, blurrinessRates) = calcBlurriness(highRes);
    }
}
