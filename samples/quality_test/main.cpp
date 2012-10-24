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
    class AllignedFrameSource : public IFrameSource
    {
    public:
        AllignedFrameSource(const Ptr<IFrameSource>& base, int scale);

        void reset();
        Mat nextFrame();

    private:
        Ptr<IFrameSource> base;
        const int scale;
    };

    AllignedFrameSource::AllignedFrameSource(const Ptr<IFrameSource>& base, int scale) :
        base(base), scale(scale)
    {
        CV_Assert( !base.empty() );
    }

    void AllignedFrameSource::reset()
    {
        base->reset();
    }

    Mat AllignedFrameSource::nextFrame()
    {
        Mat frame = base->nextFrame();

        if (frame.rows % scale != 0 || frame.cols % scale != 0)
            frame = frame(Rect(0, 0, (frame.cols / scale) * scale, (frame.rows / scale) * scale));

        return frame;
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

    double MSSIM(const Mat& i1, const Mat& i2)
    {
        const double C1 = 6.5025;
        const double C2 = 58.5225;

        const int depth = CV_32F;

        Mat I1, I2;
        i1.convertTo(I1, depth);
        i2.convertTo(I2, depth);

        Mat I2_2  = I2.mul(I2); // I2^2
        Mat I1_2  = I1.mul(I1); // I1^2
        Mat I1_I2 = I1.mul(I2); // I1 * I2

        Mat mu1, mu2;
        GaussianBlur(I1, mu1, Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, Size(11, 11), 1.5);

        Mat mu1_2   = mu1.mul(mu1);
        Mat mu2_2   = mu2.mul(mu2);
        Mat mu1_mu2 = mu1.mul(mu2);

        Mat sigma1_2, sigma2_2, sigma12;

        GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;

        GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;

        GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;

        Mat t1, t2;
        Mat numerator;
        Mat denominator;

        // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        numerator = t1.mul(t2);

        // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        denominator = t1.mul(t2);

        // ssim_map =  numerator./denominator;
        Mat ssim_map;
        divide(numerator, denominator, ssim_map);

        // mssim = average of ssim map
        Scalar mssim = mean(ssim_map);

        if (i1.channels() == 1)
            return mssim[0];

        return (mssim[0] + mssim[1] + mssim[3]) / 3;
    }
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ @0           | car.avi   | Input video }"
        "{ s scale      | 4         | Scale factor }"
        "{ i iterations | 180       | Iteration count }"
        "{ t temporal   | 4         | Radius of the temporal search area }"
        "{ f opt-flow   | farneback | Optical flow algorithm (farneback, simple, brox, pyrlk) }"
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

    Ptr<IFrameSource> goldSource(new AllignedFrameSource(new VideoFileSource(inputVideoName), scale));
    Ptr<IFrameSource> lowResSource(new DegradeFrameSource(new AllignedFrameSource(new VideoFileSource(inputVideoName), scale), scale));
    Ptr<IFrameSource> lowResSource2(new DegradeFrameSource(new AllignedFrameSource(new VideoFileSource(inputVideoName), scale), scale));

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
    double srAvgMSSIM = 0.0;
    double bcAvgPSNR = 0.0;
    double bcAvgMSSIM = 0.0;
    int count = 0;

    cout << "-----------------------------------------------------------------" << endl;
    cout << "|   Ind    |  SuperRes PSNR / MSSIM  |   BiCubic PSNR / MSSIM   |" << endl;
    cout << "|----------|-------------------------|--------------------------|" << endl;

    for (;;)
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

        const double srPSNR = PSNR(goldFrame(inner), superResFrame);
        const double srMSSIM = MSSIM(goldFrame(inner), superResFrame);

        const double bcPSNR = PSNR(goldFrame, biCubicFrame);
        const double bcMSSIM = MSSIM(goldFrame, biCubicFrame);

        srAvgPSNR += srPSNR;
        srAvgMSSIM += srMSSIM;
        bcAvgPSNR += bcPSNR;
        bcAvgMSSIM += bcMSSIM;
        ++count;

        cout << "|  [" << setw(4) << count << "]  "
             << "|      " << fixed << setprecision(2) << srPSNR << " / " << fixed << setprecision(3) << srMSSIM << "      "
             << "|       " << fixed << setprecision(2) << bcPSNR << " / "  << fixed << setprecision(3) << bcMSSIM << "      "
             << "|" << endl;

        imshow("Gold", goldFrame);
        imshow("Low Res Frame", lowResFrame);
        imshow("Super Resolution", superResFrame);
        imshow("Bi Cubic", biCubicFrame);

        if (waitKey(1000) > 0)
            break;
    }

    cout << "-----------------------------------------------------------------" << endl;

    destroyAllWindows();

    srAvgPSNR /= count;
    srAvgMSSIM /= count;
    bcAvgPSNR /= count;
    bcAvgMSSIM /= count;

    cout << "Super Resolution Avg PSNR  : " << srAvgPSNR << endl;
    cout << "Super Resolution Avg MSSIM : " << srAvgMSSIM << endl;

    cout << "Bi-Cubic Resize  AVG PSNR  : " << bcAvgPSNR << endl;
    cout << "Bi-Cubic Resize  AVG MSSIM : " << bcAvgMSSIM << endl;

    return 0;
}
