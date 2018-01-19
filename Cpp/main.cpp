/**
@brief main.cpp
main function for second alpha version
@author Shane Yuan
@date Jan 19, 2018
*/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <time.h>
// opencv
#include <opencv2/opencv.hpp>

#include "AutoWhiteBalance.h";

// main function
int main(int argc, char* argv[]) {
	AutoWhiteBalance autoWB;
	autoWB.loadModel("E:/Project/AutoWhiteBalance/data/model/model.bin");

	cv::Mat img = cv::imread("E:/data/giga/NanshanIPark/2/calibrate/ref_00.jpg");
	cv::cuda::GpuMat img_d;
	img_d.upload(img);

	float gain_r, gain_g, gain_b;

	time_t begin, end;
	begin = clock();

	autoWB.apply(img_d, gain_r, gain_g, gain_b);
	autoWB.apply(img_d, gain_r, gain_g, gain_b);

	end = clock();
	printf("Auto white balance update, cost %f milliseconds ...\n",
		static_cast<float>(end - begin) / static_cast<double>(CLOCKS_PER_SEC) * 1000);

	autoWB.applyWhiteBalance(img_d, gain_r, gain_g, gain_b);

	cv::Mat img2;
	img_d.download(img2);

	return 0;
}





