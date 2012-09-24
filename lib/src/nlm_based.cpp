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

void NlmBased::resetImpl()
{
    Y.clear();
    y.clear();
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

        int curPos;
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

        const Mat& curY = at(curProcessedPos, *Y);

        vector<double> patch1;
        vector<double> patch2;

        for (int k = range.rows().begin(); k < range.rows().end(); ++k)
        {
            for (int l = range.cols().begin(); l < range.cols().end(); ++l)
            {
                extractPatch(curY, Point2d(l, k), patch1, p, INTER_NEAREST);

                for (int t = -timeRadius; t <= timeRadius; ++t)
                {
                    const Mat& Yt = at(curProcessedPos + t, *Y);
                    const Mat& yt = at(curProcessedPos + t, *y);

                    const int iStart = searchAreaRadius == -1 ? 0 : k / s - searchAreaRadius;
                    const int iEnd = searchAreaRadius == -1 ? y->front().rows : k / s + searchAreaRadius + 1;

                    for (int i = iStart; i < iEnd; ++i)
                    {
                        const int jStart = searchAreaRadius == -1 ? 0 : l / s - searchAreaRadius;
                        const int jEnd = searchAreaRadius == -1 ? y->front().cols : l / s + searchAreaRadius + 1;

                        for (int j = jStart; j < jEnd; ++j)
                        {
                            extractPatch(Yt, Point2d(s * j, s * i), patch2, p, INTER_NEAREST);

                            double norm2 = 0.0;
                            for (size_t n = 0; n < patch1.size(); ++n)
                            {
                                const double diff = patch1[n] - patch2[n];
                                norm2 += diff * diff;
                            }

                            const double w = exp(-norm2 * weightScale);

                            Point3d val;
                            val.x = detail::readVal<uchar, double>(yt, i, j, 0, BORDER_REFLECT_101, Scalar());
                            val.y = detail::readVal<uchar, double>(yt, i, j, 1, BORDER_REFLECT_101, Scalar());
                            val.z = detail::readVal<uchar, double>(yt, i, j, 2, BORDER_REFLECT_101, Scalar());

                            V(k, l) += w * val;
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

        body.curPos = curPos;
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
