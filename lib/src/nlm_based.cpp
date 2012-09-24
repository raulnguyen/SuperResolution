#include "nlm_based.hpp"
#include <opencv2/core/internal.hpp>

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
                                       "Weight of the patch difference"));

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
    searchAreaRadius = 5;
    timeRadius = 1;
    lowResPatchSize = 7;
    sigma = 7.5;
}

void NlmBased::initImpl(cv::Ptr<IFrameSource>& frameSource)
{
    y.resize(2 * timeRadius + 1);
    Y.resize(2 * timeRadius + 1);

    curPos = -1;
    curProcessedPos = curPos - timeRadius;
    curOutPos = curProcessedPos - timeRadius;

    for (int t = -timeRadius; t < timeRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();

        if (frame.empty())
            return;

        addNewFrame(frame);
    }
}

#ifdef HAVE_TBB
    typedef tbb::blocked_range<int> Range1D;
    typedef tbb::blocked_range2d<int> Range2D;

    template <class Body>
    void loop2D(const Range2D& range, const Body& body)
    {
        tbb::parallel_for(range, body);
    }
#else
    class Range1D
    {
    public:
        Range1D(int begin, int end) : begin_(begin), end_(end) {}

        int begin() const { return begin_; }
        int end() const { return end_; }

    private:
        int begin_, end_;
    };

    class Range2D
    {
    public:
        Range2D(int row_begin, int row_end, int col_begin, int col_end) : rows_(row_begin, row_end), cols_(col_begin, col_end) {}

        const Range1D& rows() const { return rows_; }
        const Range1D& cols() const { return cols_; }

    private:
        Range1D rows_, cols_;
    };

    template <class Body>
    void loop2D(const Range2D& range, const Body& body)
    {
        body(range);
    }
#endif

namespace
{
    struct LoopBody
    {
        void operator ()(const Range2D& range) const;

        int scale;
        int searchAreaRadius;
        int timeRadius;
        int lowResPatchSize;
        double sigma;

        int curProcessedPos;

        vector<Mat>* y;
        vector<Mat>* Y;

        mutable Mat_<Point3d> V;
        mutable Mat_<Point3d> W;
    };

    void LoopBody::operator ()(const Range2D& range) const
    {
        const int s = scale;           // the desired scaling factor
        const int q = lowResPatchSize; // the size of the low resolution patch
        const int p = s * (q - 1) + 1; // the size of the high resolution patch
        const double weightScale = 1.0 / (2.0 * sigma * sigma);

        const Mat& Z = at(curProcessedPos, *Y);

        const int kStart = max(range.rows().begin(), p/2);
        const int kEnd = min(range.rows().end(), Z.rows - p/2);

        const int lStart = max(range.cols().begin(), p/2);
        const int lEnd = min(range.cols().end(), Z.cols - p/2);

        for (int k = kStart; k < kEnd; ++k)
        {
            for (int l = lStart; l < lEnd; ++l)
            {
                const Mat_<Vec3b> Z_patch(p, p, const_cast<Vec3b*>(Z.ptr<Vec3b>(k - p/2) + l - p/2), Z.step);

                for (int t = -timeRadius; t <= timeRadius; ++t)
                {
                    const Mat& Yt = at(curProcessedPos + t, *Y);
                    const Mat& yt = at(curProcessedPos + t, *y);

                    int iStart, iEnd;
                    int jStart, jEnd;

                    if (searchAreaRadius > 0)
                    {
                        iStart = k / s - searchAreaRadius;
                        iEnd   = k / s + searchAreaRadius + 1;
                        jStart = l / s - searchAreaRadius;
                        jEnd   = l / s + searchAreaRadius + 1;
                    }
                    else
                    {
                        iStart = 0;
                        iEnd   = y->front().rows;
                        jStart = 0;
                        jEnd   = y->front().cols;
                    }

                    for (int i = iStart; i < iEnd; ++i)
                    {
                        if (s * i < p / 2 || s * i > Yt.rows)
                            continue;

                        for (int j = jStart; j < jEnd; ++j)
                        {
                            if (s * j < p / 2 || s * j > Yt.cols)
                                continue;

                            const Mat_<Vec3b> Yt_patch(p, p, const_cast<Vec3b*>(Yt.ptr<Vec3b>(s * i - p/2) + s * j - p/2), Yt.step);

                            double patchDiff = 0.0;
                            for (int ii = 0; ii < p; ++ii)
                            {
                                for (int jj = 0; jj < p; ++jj)
                                {
                                    Vec3b zVal = Z_patch(ii, jj);
                                    Point3d zVald(zVal[0], zVal[1], zVal[2]);

                                    Vec3b YtVal = Yt_patch(ii, jj);
                                    Point3d YtVald(YtVal[0], YtVal[1], YtVal[2]);

                                    Point3d diff = zVald - YtVald;

                                    patchDiff += diff.ddot(diff);
                                }
                            }

                            const double w = exp(-patchDiff * weightScale);

                            Vec3b val = yt.at<Vec3b>(i, j);
                            Point3d vald(val[0], val[1], val[2]);

                            V(k, l) += w * vald;
                            W(k, l) += Point3d(w, w, w);
                        }
                    }
                }
            }
        }
    }
}

Mat NlmBased::processImpl(const Mat& frame)
{
    addNewFrame(frame);

    Mat& curProcY = at(curProcessedPos, Y);

    curProcY.convertTo(V, CV_64F);
    W.create(V.size());
    W.setTo(Scalar::all(1));

    {
        LoopBody body;

        body.scale = scale;
        body.searchAreaRadius = searchAreaRadius;
        body.timeRadius = timeRadius;
        body.lowResPatchSize = lowResPatchSize;
        body.sigma = sigma;

        body.curProcessedPos = curProcessedPos;

        body.y = &y;
        body.Y = &Y;

        body.V = V;
        body.W = W;

        loop2D(Range2D(0, V.rows, 0, V.cols), body);
    }

    divide(V, W, Z);

    Z.convertTo(curProcY, CV_8U);

    return at(curOutPos, Y);
}

void NlmBased::addNewFrame(const cv::Mat& frame)
{
    ++curPos;
    ++curProcessedPos;
    ++curOutPos;

    frame.copyTo(at(curPos, y));
    resize(frame, at(curPos, Y), Size(), scale, scale, INTER_CUBIC);
}
