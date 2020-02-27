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
#pragma warning( disable : 4996)

#define _CRT_SECURE_NO_WARNINGS
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 //#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"

using namespace std;
int main(int argc, char** argv) {
	int flag;
	if (argc < 3) {
		cout << "Usage: " << argv[0]
			<< " image_path model_path"
			<< endl;
		return -1;
	}

	string path(argv[1]);

	const char* img_path = path.c_str();

	//cv::Mat detectAndResize(cv::Mat& img, int& flag) {}
	seeta::FaceDetection detector(argv[2]);
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	cv::Mat initImg = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	
	cv::Mat imgmini;
	cv::Rect mini;
	mini.x = img.cols*0.1;
	mini.y = img.rows*0.1;
	mini.height = img.rows*0.6;
	mini.width = img.cols*0.6;
	cv::Mat temps = img.clone();
	
	img = img(mini);

	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	cv::Mat centerROI, centerGray, cutROI, reImg;
	cv::Rect centerRect, cutRect;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	cv::Rect face_rect;
	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
	int32_t num_face = static_cast<int32_t>(faces.size());

	if (num_face != 0) {
		flag = 1;
		for (int32_t i = 0; i < 1; i++) {
			face_rect.x = mini.x + faces[i].bbox.x - faces[i].bbox.width*0.4375;
			face_rect.y = mini.y + faces[i].bbox.y - faces[i].bbox.width*0.75;

			face_rect.width = faces[i].bbox.width*1.875;
			face_rect.height = faces[i].bbox.width*2.5;
			cv::Mat result(face_rect.height, face_rect.width, CV_8UC3, cv::Scalar(255, 255, 255));

			for (int i = 0; i < face_rect.width; i++)
			{
				for (int j = 0; j < face_rect.height; j++)
				{
					if (i + face_rect.x < 0 || j + face_rect.y < 0 || i + face_rect.x >= temps.cols || j + face_rect.y >= temps.rows)
					{
						continue;
					}
					else
					{
						result.at<cv::Vec3b>(j, i) = temps.at<cv::Vec3b>(j + face_rect.y, i + face_rect.x);
					}
				}
			}

			double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
			int y = 0;

			if (face_rect.x < 0)
				x1 = 0 - face_rect.x;
			else
				x1 = 0;
			if (face_rect.width + face_rect.x > temps.cols)
				x2 = temps.cols - face_rect.x;
			else
				x2 = face_rect.width;
			y = (int)((x2 - x1)*4.0 / 3.0);
			int chazhi = temps.rows - y;

			if (face_rect.y < 0)
				y1 = 0 - face_rect.y;
			else
				y1 = 0;
			if (face_rect.height + face_rect.y > temps.rows)
				y2 = temps.rows - face_rect.y;
			else
				y2 = face_rect.height;

			int heng = x2 - x1;
			int zong = y2 - y1;
			cv::Rect xiaotu_rect(x1, y1, heng, zong);

			double zong_2 = (face_rect.width*1.0) / (heng*1.0)*zong;
			double heng_2 = face_rect.width;
			cv::Mat xiaotu = result(xiaotu_rect).clone();
			cv::resize(xiaotu, xiaotu, cv::Size(heng_2, zong_2));
			int tem = (int)zong_2 - face_rect.height;
			if (tem > 0)
			{
				cv::Rect R_A(0, 0, heng_2, face_rect.height);
				xiaotu = xiaotu(R_A).clone();
			}
			if (tem < 0)
			{
				tem = 0 - tem;

				cv::Mat background(face_rect.height, face_rect.width, CV_8UC3, cv::Scalar(0, 255, 0)); //±³¾°ÑÕÉ«
				cv::Rect R_B(0, tem, xiaotu.cols, xiaotu.rows);
				xiaotu.copyTo(background(R_B));
				xiaotu = background.clone();
			}
			cv::Size reSize = cv::Size(300, 400);
			cv::resize(xiaotu, reImg, reSize);
		}
	}
	else
	{
		flag = 0;
	}
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
	cout << path << endl;
	
	char *path_name = (char*)img_path;
	string s(path_name);
	
	s.erase(s.begin(),s.begin()+8);
	cout << s << endl;
	string store_path = "../resized_data/";
	store_path.append(s);
	cout << store_path << endl;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //jpeg
	compression_params.push_back(100); //quality of img
	cv::imwrite(store_path.c_str(), reImg, compression_params); //store the resized img

	cv::namedWindow("Resize", cv::WINDOW_AUTOSIZE);
	cv::imshow("Before", initImg);
	cv::imshow("Resize", reImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
}