/**
@brief AutoWhiteBalance.h
class for auto white balance
This algorithm is implemented based on google's two papers:
	https://github.com/yuanxy92/AutoWhiteBalance
	fft2(response) = fft2(filter_d) .* fft2(hist_d) + fft2(bias_d)
@author Shane Yuan
@date Jan 19, 2018
*/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <memory>
// opencv
#include <opencv2/opencv.hpp>

class AutoWhiteBalance {
private:
	// filter and bias matrix
	// fft2(response) = fft2(filter_d) .* fft2(hist_d) + fft2(bias_d)
	cv::Mat filter;
	cv::Mat bias;
	cv::cuda::GpuMat filter_d;
	cv::cuda::GpuMat bias_d;

	// histogram 
	int width;
	int height;
	cv::Mat hist;
	cv::cuda::GpuMat hist_d;

	// input image
	cv::cuda::GpuMat img_d;
	cv::cuda::GpuMat smallImg_d;
	cv::Mat smallImg;
	cv::Mat smallImgf;
	cv::Size smallSize;

	// histogram step
	float binsize;
	float uv0;

	// response map
	cv::cuda::GpuMat response_fft;
	cv::cuda::GpuMat response;
	cv::cuda::GpuMat filter_fft;
	cv::cuda::GpuMat bias_fft;
	cv::cuda::GpuMat hist_fft;

	// kalman filter use to smooth result
	size_t frameInd;
	float gain_r, gain_g, gain_b;
	cv::Point pos;
	cv::Mat measurement;
	cv::Mat estimated;
	std::shared_ptr<cv::KalmanFilter> kfPtr;
public:

private:
	/**
	@brief calculate histogram
	@return int
	*/
	int calcHistFeature();

	/**
	@brief compute response map
	@return int
	*/
	int computeResponse();

	/**
	@brief predict new point using kalman filter
	@return
	*/
	int predictKalman();

public:
	AutoWhiteBalance();
	~AutoWhiteBalance();

	/**
	@brief load auto white balance model from file
	@param std::string modelname: input model name
	@return int
	*/
	int loadModel(std::string modelname);

	/**
	@brief calc auto white balance
	@param cv::cuda::GpuMat img_d: input gpu image
	@param float & gain_r: output r channel gain
	@param float & gain_g: output g channel gain
	@param float & gain_b: output b channel gain
	@return int
	*/
	int calc(cv::cuda::GpuMat img_d, float & _gain_r,
		float & _gain_g, float & _gain_b);

	/**
	@brief apply white balance
	@param cv::cuda::GpuMat & img_d: input/output gpu image
	@param float gain_r: input r channel gain
	@param float gain_g: input g channel gain
	@param float gain_b: input b channel gain
	@return int
	*/
	static int applyWhiteBalance(cv::cuda::GpuMat & img_d, float _gain_r,
		float _gain_g, float _gain_b);

	/**
	@brief Twist Color
	*      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3]
	*      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3]
	*      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3]
	@param cv::cuda::GpuMat & img_d: input/output gpu image
	@param  float **twist_data: 2 dimension float twist array data
	@return int
	*/
	static int GpuMatColorTwist(cv::cuda::GpuMat & img_d, float twist_data[3][4]);
};
