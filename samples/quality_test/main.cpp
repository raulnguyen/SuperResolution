// Copyright (c) 2012, Vladislav Vinogradov (jet47)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

namespace
{
    class GrayScaleFrameSource : public IFrameSource
    {
    public:
        GrayScaleFrameSource(const Ptr<IFrameSource>& base, int scale);

        void reset();
        Mat nextFrame();

    private:
        Mat gray;
        Ptr<IFrameSource> base;
        const int scale;
    };

    GrayScaleFrameSource::GrayScaleFrameSource(const Ptr<IFrameSource>& base, int scale) :
        base(base), scale(scale)
    {
        CV_Assert( !base.empty() );
    }

    void GrayScaleFrameSource::reset()
    {
        base->reset();
    }

    Mat GrayScaleFrameSource::nextFrame()
    {
        Mat frame = base->nextFrame();

        if (frame.rows % scale != 0 || frame.cols % scale != 0)
            frame = frame(Rect(0, 0, (frame.cols / scale) * scale, (frame.rows / scale) * scale));

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        return gray;
    }

    class DegradeFrameSource : public IFrameSource
    {
    public:
        DegradeFrameSource(const Ptr<IFrameSource>& base, int scale);

        void reset();
        Mat nextFrame();

    private:
        Mat blurred;
        Mat deg;
        Ptr<IFrameSource> base;
        const double iscale;
    };

    DegradeFrameSource::DegradeFrameSource(const Ptr<IFrameSource>& base, int scale) :
        base(base), iscale(1.0 / scale)
    {
        CV_Assert( !base.empty() );
    }

    void DegradeFrameSource::reset()
    {
        base->reset();
    }

    void addGaussNoise(Mat& image, double sigma)
    {
        Mat noise(image.size(), CV_32FC(image.channels()));
        theRNG().fill(noise, RNG::NORMAL, 0.0, sigma);

        addWeighted(image, 1.0, noise, 1.0, 0.0, image, image.depth());
    }

    void addSpikeNoise(Mat& image, int frequency)
    {
        Mat_<uchar> mask(image.size(), 0);

        for (int y = 0; y < mask.rows; ++y)
        {
            for (int x = 0; x < mask.cols; ++x)
            {
                if (theRNG().uniform(0, frequency) < 1)
                    mask(y, x) = 255;
            }
        }

        image.setTo(Scalar::all(255), mask);
    }

    Mat DegradeFrameSource::nextFrame()
    {
        Mat frame = base->nextFrame();

        GaussianBlur(frame, blurred, Size(5, 5), 0);
        resize(blurred, deg, Size(), iscale, iscale, INTER_NEAREST);

        addGaussNoise(deg, 10.0);
        addSpikeNoise(deg, 500);

        return deg;
    }

    double getPSNR(const Mat& src1, const Mat& src2)
    {
        CV_Assert( src1.type() == CV_8UC1 );
        CV_Assert( src2.size() == src1.size() );
        CV_Assert( src2.type() == src1.type() );

        double sse = 0.0;
        for (int y = 0; y < src1.rows; ++y)
        {
            const uchar* src1Row = src1.ptr(y);
            const uchar* src2Row = src2.ptr(y);

            for (int x = 0; x < src1.cols; ++x)
                sse += (src1Row[x] - src2Row[x]) * (src1Row[x] - src2Row[x]);
        }

        if (sse == 0.0)
            return 0;

        const double mse = sse / src1.size().area();

        return 10.0 * log10(255 * 255 / mse);
    }
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ @0           | car.avi   | Input video }"
        "{ s scale      | 4         | Scale factor }"
        "{ i iterations | 180       | Iteration count }"
        "{ t temporal   | 4         | Radius of the temporal search area }"
        "{ f opt-flow   | farneback | Optical flow algorithm (farneback, simple, tvl1, brox, pyrlk) }"
        "{ gpu          |           | Use GPU }"
        "{ h help       |           | Print help message }"
    );

    if (cmd.has("help"))
    {
        cmd.about("This sample measure quality of Super Resolution algorithm");
        cmd.printMessage();
        return 0;
    }

    const string inputVideoName = cmd.get<string>(0);
    const int scale = cmd.get<int>("scale");
    const int iterations = cmd.get<int>("iterations");
    const int temporalAreaRadius = cmd.get<int>("temporal");
    const string optFlow = cmd.get<string>("opt-flow");
    const bool useGpu = cmd.has("gpu");

    if (!cmd.check())
    {
        cmd.printErrors();
        return -1;
    }

    Ptr<DenseOpticalFlow> optFlowAlg;
    if (optFlow == "farneback")
    {
        if (useGpu)
            optFlowAlg = new Farneback_GPU;
        else
            optFlowAlg = new Farneback;
    }
    else if (optFlow == "simple")
        optFlowAlg = new Simple;
    else if (optFlow == "tvl1")
        optFlowAlg = new Dual_TVL1;
    else if (optFlow == "brox")
        optFlowAlg = new Brox_GPU;
    else if (optFlow == "pyrlk")
        optFlowAlg = new PyrLK_GPU;
    else
    {
        cerr << "Incorrect Optical Flow algorithm - " << optFlow << endl;
        cmd.printMessage();
        return -1;
    }

    Ptr<SuperResolution> superRes;
    if (useGpu)
        superRes = new BTV_L1_GPU;
    else
        superRes = new BTV_L1;

    const int btvKernelSize = superRes->getInt("btvKernelSize");
    Rect inner;

    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);
    superRes->set("opticalFlow", optFlowAlg);

    Ptr<IFrameSource> goldSource(new GrayScaleFrameSource(new VideoFileSource(inputVideoName), scale));
    Ptr<IFrameSource> lowResSource(new DegradeFrameSource(new GrayScaleFrameSource(new VideoFileSource(inputVideoName), scale), scale));
    Ptr<IFrameSource> lowResSource2(new DegradeFrameSource(new GrayScaleFrameSource(new VideoFileSource(inputVideoName), scale), scale));

    // skip first frame, it is usually corrupted
    {
        Mat frame = lowResSource->nextFrame();
        frame = lowResSource2->nextFrame();
        frame = goldSource->nextFrame();

        inner = Rect(btvKernelSize, btvKernelSize, frame.cols - 2 * btvKernelSize, frame.rows - 2 * btvKernelSize);

        cout << "Input             : " << inputVideoName << " " << frame.size() << endl;
        cout << "Scale factor      : " << scale << endl;
        cout << "Iterations        : " << iterations << endl;
        cout << "Frames to process : " << temporalAreaRadius * 2 + 1 << endl;
        cout << "Optical Flow      : " << optFlow << endl;
        cout << "Mode              : " << (useGpu ? "GPU" : "CPU") << endl;
    }

    superRes->setFrameSource(lowResSource);

    Mat biCubicFrame;

    double srAvgPSNR = 0.0;
    double bcAvgPSNR = 0.0;
    int count = 0;

    cout << "-------------------------------------------------" << endl;
    cout << "|   Ind    |  SuperRes PSNR  |   BiCubic PSNR   |" << endl;
    cout << "|----------|-----------------|------------------|" << endl;

    for (;; ++count)
    {
        Mat goldFrame = goldSource->nextFrame();
        if (goldFrame.empty())
            break;

        Mat lowResFrame = lowResSource2->nextFrame();
        if (lowResFrame.empty())
            break;

        Mat superResFrame = superRes->nextFrame();
        if (superResFrame.empty())
            break;

        resize(lowResFrame, biCubicFrame, Size(), scale, scale, INTER_CUBIC);

        const double srPSNR = getPSNR(goldFrame(inner), superResFrame);
        const double bcPSNR = getPSNR(goldFrame, biCubicFrame);

        srAvgPSNR += srPSNR;
        bcAvgPSNR += bcPSNR;

        cout << "|  [" << setw(4) << count << "]  |      " << fixed << setprecision(2) << srPSNR << "      |      " << fixed << setprecision(2) << bcPSNR << "       |" << endl;

        imshow("Gold", goldFrame);
        imshow("Low Res Frame", lowResFrame);
        imshow("Super Resolution", superResFrame);
        imshow("Bi Cubic", biCubicFrame);

        if (waitKey(1000) > 0)
            break;
    }

    cout << "-------------------------------------------------" << endl;

    destroyAllWindows();

    srAvgPSNR /= count;
    bcAvgPSNR /= count;

    cout << "Super Resolution Avg PSNR : " << srAvgPSNR << " dB" << endl;
    cout << "Bi-Cubic Resize  AVG PSNR : " << bcAvgPSNR << " dB" << endl;

    return 0;
}
