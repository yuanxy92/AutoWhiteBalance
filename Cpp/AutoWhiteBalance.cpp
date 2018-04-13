/**
@brief AutoWhiteBalance.cpp
class for auto white balance
@author Shane Yuan
@date Jan 19, 2018
*/
#include <fstream>
#include <cmath>
#include "AutoWhiteBalance.h"

#include <npp.h>
#include <nppi.h>

#include "Exceptions.h"
#include "helper_string.h"
#include "helper_cuda.h"

//#define DEBUG_AUTO_WHITE_BALANCE
//#define MEASURE_RUNTIME

AutoWhiteBalance::AutoWhiteBalance() : binsize(1.0f / 64.0f),
	smallSize(400,  300), uv0(-1.421875), frameInd(0) {}
AutoWhiteBalance::~AutoWhiteBalance() {}

/**
@brief calculate histogram
@return
*/
int AutoWhiteBalance::calcHistFeature() {
	// calculate histogram in CPU
	cv::log(smallImgf, smallImgf);
	int u, v;
	float totPixel = smallSize.width * smallSize.height;
	float singlePxWeight = 1.0f / totPixel;
	for (size_t i = 0; i < smallSize.height; i++) {
		for (size_t j = 0; j < smallSize.width; j++) {
			cv::Point3f val = smallImgf.at<cv::Point3f>(i, j);
			u = round((val.y - val.z - uv0) / binsize);
			v = round((val.y - val.x - uv0) / binsize);
			u = std::max<int>(std::min<int>(u, 255), 0);
			v = std::max<int>(std::min<int>(v, 255), 0);
			hist.at<float>(u, v) += singlePxWeight;
		}
	}
	// upload to gpu
	hist_d.upload(hist);
	return 0;
}

/**
@brief compute response map
@return int
*/
int AutoWhiteBalance::computeResponse() {
	// compute response
	cv::cuda::dft(hist_d, hist_fft, cv::Size(width, height));
	cv::cuda::mulSpectrums(filter_fft, hist_fft, response_fft, 0);
	cv::cuda::add(response_fft, bias_fft, response_fft);
	// ifft
	cv::cuda::dft(response_fft, response, cv::Size(width, height),
		cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	// get max position
#ifdef DEBUG_AUTO_WHITE_BALANCE
	cv::Mat response_h;
	response.download(response_h);
#endif
	cv::cuda::minMaxLoc(response, NULL, NULL, NULL, &pos);
	return 0;
}

/**
@brief load auto white balance model from file
@param std::string modelname: input model name
@return int
*/
int AutoWhiteBalance::loadModel(std::string modelname) {
	std::fstream fs(modelname, std::ios::in | std::ios::binary);
	// read size
	fs.read(reinterpret_cast<char*>(&width), sizeof(int));
	fs.read(reinterpret_cast<char*>(&height), sizeof(int));
	// read filter
	this->filter.create(height, width, CV_32F);
	this->bias.create(height, width, CV_32F);
	fs.read(reinterpret_cast<char*>(filter.data), sizeof(float) * width * height);
	fs.read(reinterpret_cast<char*>(bias.data), sizeof(float) * width * height);
	fs.close();
	filter = filter.t();
	bias = bias.t();
	// upload to GPU
	filter_d.upload(filter);
	bias_d.upload(bias);
	// calculate fft
	filter_fft.create(height, width, CV_32FC2);
	bias_fft.create(height, width, CV_32FC2);
	hist_fft.create(height, width, CV_32FC2);
	response_fft.create(height, width, CV_32FC2);
	response.create(height, width, CV_32FC2);
	cv::cuda::dft(filter_d, filter_fft, cv::Size(width, height));
	cv::cuda::dft(bias_d, bias_fft, cv::Size(width, height));
	// init hist
	hist.create(height, width, CV_32F);
	hist_d.create(height, width, CV_32F);
	hist.setTo(cv::Scalar(0));
	hist_d.setTo(cv::Scalar(0));
	response.create(height, width, CV_32F);
	// init kalman filter
	pos = cv::Point(height / 2, width / 2);
	measurement = cv::Mat::zeros(2, 1, CV_32F);
	kfPtr = std::make_shared<cv::KalmanFilter>(4, 2, 0);
	kfPtr->transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	kfPtr->statePre.at<float>(0) = 0;
	kfPtr->statePre.at<float>(1) = 0;
	kfPtr->statePre.at<float>(2) = 0;
	kfPtr->statePre.at<float>(3) = 0;
	setIdentity(kfPtr->measurementMatrix);
	setIdentity(kfPtr->processNoiseCov, cv::Scalar::all(1e-4));
	setIdentity(kfPtr->measurementNoiseCov, cv::Scalar::all(10));
	setIdentity(kfPtr->errorCovPost, cv::Scalar::all(.1));
	return 0;
}

/**
@brief predict new point using kalman filter
@return
*/
int AutoWhiteBalance::predictKalman() {
	float lu, lv, z;
	if (frameInd == 0) {
		kfPtr->statePost.at<float>(0) = pos.x;
		kfPtr->statePost.at<float>(1) = pos.y;
		kfPtr->statePost.at<float>(2) = 0;
		kfPtr->statePost.at<float>(3) = 0;
		// change position to gain
		lu = (pos.y + 1) * binsize + uv0;
		lv = (pos.x + 1) * binsize + uv0;
		z = sqrt(exp(-lu) * exp(-lu) + exp(-lv) * exp(-lv) + 1);
	}
	else {
		// First predict, to update the internal statePre variable
		cv::Mat prediction = kfPtr->predict();
		cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));
		// The update phase 
		measurement.at<float>(0, 0) = pos.x;
		measurement.at<float>(1, 0) = pos.y;
		estimated = kfPtr->correct(measurement);
		// change position to gain
		lu = prediction.at<float>(1, 0) * binsize + uv0;
		lv = prediction.at<float>(0, 0) * binsize + uv0;
		z = sqrt(exp(-lu) * exp(-lu) + exp(-lv) * exp(-lv) + 1);
	}
	gain_r = z / exp(-lu);
	gain_g = z;
	gain_b = z / exp(-lv);
	float gain_sum = gain_r + gain_g + gain_b;
	gain_r = gain_r / gain_sum * 3;
	gain_g = gain_g / gain_sum * 3;
	gain_b = gain_b / gain_sum * 3;
	frameInd++;
	return 0;
}

/**
@brief apply white balance
@param cv::cuda::GpuMat & img_d: input/output gpu image
@param float gain_r: input r channel gain
@param float gain_g: input g channel gain
@param float gain_b: input b channel gain
@return int
*/
int AutoWhiteBalance::applyWhiteBalance(cv::cuda::GpuMat & img_d, float gain_r,
	float gain_g, float gian_b) {
	// white balance color twist
	Npp32f wbTwist[3][4] = {
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 0.0, 1.0, 0.0, 0.0 },
		{ 0.0, 0.0, 1.0, 0.0 }
	};
	wbTwist[0][0] = gain_b;
	wbTwist[1][1] = gain_g;
	wbTwist[2][2] = gain_r;
	NppiSize osize;
	osize.width = img_d.cols;
	osize.height = img_d.rows;
	NPP_CHECK_NPP(nppiColorTwist32f_8u_C3IR(img_d.data, img_d.step, osize, wbTwist));
	return 0;
}

/**
@brief apply auto white balance
@param cv::cuda::GpuMat img_d: input gpu images
@param float & gain_r: output r channel gain
@param float & gain_g: output g channel gain
@param float & gain_b: output b channel gain
@return int
*/
int AutoWhiteBalance::apply(cv::cuda::GpuMat img_d, float & gain_r,
	float & gain_g, float & gain_b) {
#ifdef MEASURE_RUNTIME
	time_t t1, t2, t3, t4, t5, t6;
	t1 = clock();
#endif
	// resize image
	cv::cuda::resize(img_d, smallImg_d, smallSize);
#ifdef MEASURE_RUNTIME
	t2 = clock();
#endif
	// download to CPU
	cv::Mat smallImg;
	smallImg_d.download(smallImg);
	smallImg.convertTo(smallImgf, CV_32F);
#ifdef MEASURE_RUNTIME
	t3 = clock();
#endif
	// calculate histogram
	this->calcHistFeature();
#ifdef MEASURE_RUNTIME
	t4 = clock();
#endif
	// compute response map
	this->computeResponse();
#ifdef MEASURE_RUNTIME
	t5 = clock();
#endif
	// kalman filter
	this->predictKalman();
#ifdef MEASURE_RUNTIME
	t6 = clock();
	printf("Resize image, cost %f milliseconds ...\n",
		static_cast<float>(t2 - t1) / static_cast<float>(CLOCKS_PER_SEC) * 1000);
	printf("Download to CPU, cost %f milliseconds ...\n",
		static_cast<float>(t3 - t2) / static_cast<float>(CLOCKS_PER_SEC) * 1000);
	printf("Calculate histogram, cost %f milliseconds ...\n",
		static_cast<float>(t4 - t3) / static_cast<float>(CLOCKS_PER_SEC) * 1000);
	printf("Compute response mape, cost %f milliseconds ...\n",
		static_cast<float>(t5 - t4) / static_cast<float>(CLOCKS_PER_SEC) * 1000);
	printf("Kalman filter, cost %f milliseconds ...\n",
		static_cast<float>(t6 - t5) / static_cast<float>(CLOCKS_PER_SEC) * 1000);
#endif
	gain_r = this->gain_r;
	gain_g = this->gain_g;
	gain_b = this->gain_b;
	return 0;
}