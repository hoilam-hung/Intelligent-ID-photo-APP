/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face detection, the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <stack>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 //#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"

using namespace std;
/*
int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Usage: " << argv[0]
			<< " image_path model_path"
			<< endl;
		return -1;
	}

	const char* img_path = argv[1];
	seeta::FaceDetection detector(argv[2]);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::Mat image = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;

	cv::Rect rectangle(3, 50, image.cols, image.rows);
	cv::Mat result;
	//两个临时矩阵变量，作为算法的中间变量使用，不用care
	cv::Mat bgModel, fgModel;
	// GrabCut 分段
	cv::grabCut(image, //输入图像
		result, //分段结果
		rectangle,// 包含前景的矩形
		bgModel, fgModel, // 前景、背景
		13, // 迭代次数
		cv::GC_INIT_WITH_RECT); // 用矩形
	//printf("算法执行执行时间:%g ms\n", tt / cv::getTickFrequency() * 1000);
	// 得到可能是前景的像素
	//比较函数保留值为GC_PR_FGD 的像素
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// 产生输出图像
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	//背景值为GC_BGD=0，作为掩码
	image.copyTo(foreground, result);


	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Before", image);
	cv::imshow("Test", foreground);
	cv::waitKey(0);
	cv::destroyAllWindows();

}
*/
void holeFilling(cv::Mat ori, cv::Mat seg) {
	cv::Mat grayImage;
	cv::cvtColor(seg, grayImage, CV_BGR2GRAY);
	cv::Mat bw = grayImage.clone();
	cv::threshold(grayImage, bw, 254, 255, CV_THRESH_BINARY);
	cv::Mat element3 = cv::getStructuringElement(2, cv::Size(8, 8));
	cv::Rect top_f(0, 0, 300, 400);
	cv::Mat bw_top = bw(top_f).clone();
	cv::morphologyEx(bw_top, bw_top, cv::MORPH_OPEN, element3);
	bw_top.copyTo(bw(top_f));
	cv::Rect midddle(1, 250, 200, 110);
	cv::Rect middle2(1, 255, 190, 105);
	cv::Mat white(400, 300, CV_8UC1, cv::Scalar(255));
	cv::Mat temp = bw(midddle).clone();
	temp.copyTo(white(midddle));
	cv::Mat element4 = cv::getStructuringElement(2, cv::Size(20, 20));
	cv::morphologyEx(white, white, cv::MORPH_OPEN, element4);
	white(middle2).copyTo(bw(middle2));
	cv::Mat labelImg;
	bw.convertTo(labelImg, CV_32SC1);
	int label = 0; //start by 1
	int rows = bw.rows;
	int cols = bw.cols;
	cv::Mat mask(rows, cols, CV_8UC1);
	mask.setTo(0);
	vector< vector<std::pair<int, int>> > pixs;
	for (int i = 0; i < rows; i++)
	{
		int* data = labelImg.ptr<int>(i);
		uchar *masKptr = mask.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (data[j] == 255 && mask.at<uchar>(i, j) != 1)
			{
				mask.at<uchar>(i, j) = 1;
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j)); // pixel position: <i,j>
				++label; //begin with a new label
				vector<std::pair<int, int>> pix;
				while (!neighborPixels.empty())
				{
					//get the top pixel on the stack and label it with the same label
					std::pair<int, int> curPixel = neighborPixels.top();
					pix.push_back(curPixel);
					int curY = curPixel.first;
					int curX = curPixel.second;
					labelImg.at<int>(curY, curX) = label;
					//pop the top pixel
					neighborPixels.pop();
					//push the 4-neighbors(foreground pixels)
					if (curX - 1 >= 0)
					{
						if (labelImg.at<int>(curY, curX - 1) == 255 && mask.at<uchar>(curY, curX -
							1) != 1) //leftpixel
						{
							neighborPixels.push(std::pair<int, int>(curY, curX - 1));
							mask.at<uchar>(curY, curX - 1) = 1;
						}
						if (curX + 1 <= cols - 1)
						{
							if (labelImg.at<int>(curY, curX + 1) == 255 && mask.at<uchar>(curY, curX +
								1) != 1)
								// right pixel
							{
								neighborPixels.push(std::pair<int, int>(curY, curX + 1));
								mask.at<uchar>(curY, curX + 1) = 1;
							}
						}
						if (curY - 1 >= 0)
						{
							if (labelImg.at<int>(curY - 1, curX) == 255 && mask.at<uchar>(curY - 1,
								curX) != 1)
								// up pixel
							{
								neighborPixels.push(std::pair<int, int>(curY - 1, curX));
								mask.at<uchar>(curY - 1, curX) = 1;
							}
						}
						if (curY + 1 <= rows - 1)
						{
							if (labelImg.at<int>(curY + 1, curX) == 255 && mask.at<uchar>(curY + 1,
								curX) != 1)
								//down pixel
							{
								neighborPixels.push(std::pair<int, int>(curY + 1, curX));
								mask.at<uchar>(curY + 1, curX) = 1;
							}
						}
					}
					if (pix.size() < 6000)
					{
						pixs.push_back(pix);
						for (auto ff : pix)
						{
							if (ff.first <= 40)
								continue;
							if (ff.first <= 150 && (ff.second < 80 || ff.second>220))
								continue;
							seg.at<cv::Vec3b>(ff.first, ff.second) = ori.at<cv::Vec3b>(ff.first,
								ff.second);
							bw.at<uchar>(ff.first, ff.second) = 0;
						}
					}
				}
			}
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Usage: " << argv[0]
			<< " image_path model_path"
			<< endl;
		return -1;
	}

	const char* img_path = argv[1];

	cv::Mat image = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;

	cv::Rect rectangle(1, 49, image.cols-5, image.rows-49);
	cv::Mat result;
	cv::Mat bgModel, fgModel;

	//grabCut()最后一个参数为cv::GC_INIT_WITH_MASK时
	result = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(cv::GC_BGD));
	cv::Mat roi(result, rectangle);
	roi = cv::Scalar(cv::GC_PR_FGD);
	//这两步可以合并（此处体现了使用bgModel , fgModel的价值）
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 1,
		cv::GC_INIT_WITH_MASK);
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 4,
		cv::GC_INIT_WITH_MASK);


	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	//result = result & 1 ;
	cv::Mat foreground(image.size(), CV_8UC3,
		cv::Scalar(255, 255, 255));
	image.copyTo(foreground, result);
	holeFilling(image, foreground);

	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Before", image);
	cv::imshow("Test", foreground);
	cv::waitKey(0);
	cv::destroyAllWindows();

}


/*
int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Usage: " << argv[0]
			<< " image_path model_path"
			<< endl;
		return -1;
	}

	const char* img_path = argv[1];
	seeta::FaceDetection detector(argv[2]);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	long t0 = cv::getTickCount();
	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
	long t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();

	cout << "Detections takes " << secs << " seconds " << endl;
#ifdef USE_OPENMP
	cout << "OpenMP is used." << endl;
#else
	cout << "OpenMP is not used. " << endl;
#endif

#ifdef USE_SSE
	cout << "SSE is used." << endl;
#else
	cout << "SSE is not used." << endl;
#endif

	cout << "Image size (wxh): " << img_data.width << "x"
		<< img_data.height << endl;

	cv::Rect face_rect;
	int32_t num_face = static_cast<int32_t>(faces.size());

	for (int32_t i = 0; i < num_face; i++) {
		face_rect.x = faces[i].bbox.x;
		face_rect.y = faces[i].bbox.y;
		face_rect.width = faces[i].bbox.width;
		face_rect.height = faces[i].bbox.height;

		cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
	}

	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
*/