#include <stdio.h>
#include <filesystem>
#include <windows.h>
#include <string>
#include <map>
#include <iostream>
#include <chrono>

#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "line_functions.h"
#include "hough.cuh"
#include "hough_images.cuh"

string file_path;
using namespace std::chrono;
using namespace cv;
using namespace std;

const char* CW_IMG_ORIGINAL = "Imagen original";
const char* CW_IMG_LINES = "Lineas detectadas";
const char* CW_IMG_INTERSECTIONS = "Intersecciones";
const char* CW_ACCUMULATOR = "Acumulador";
const char* CW_ACCUMULATOR2 = "Acumulador";

int images = 1024;

int countPixelsWithValue(Mat image, uchar targetValue) {
	int count = 0;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (image.at<uchar>(y, x) >= targetValue) {
				count++;
			}
		}
	}

	return count;
}

//Operaciones morfológicas + Canny

Mat alternativa_desechada(Mat input_image) {
	Mat img_edge;
	Mat morph_close;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(input_image, morph_close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);
	cv::Canny(morph_close, img_edge, 100, 150, 3);

	return img_edge;
}

void opencvTransform(vector <Mat> input_data, int threshold) {

	vector<Vec2f> lines;
	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < images; i++) {
		HoughLines(input_data[i], lines, 1, CV_PI / 180, threshold);
	}
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << "OPENCV execution time: " << duration.count() << " ms" << endl;
}

Mat getAccu(keymolen::Hough hough) {
	int aw, ah, maxa;
	aw = ah = maxa = 0;
	const unsigned int* accu = hough.GetAccu(&aw, &ah);

	for (int p = 0; p < (ah * aw); p++)
	{
		if ((int)accu[p] > maxa)
			maxa = accu[p];
	}
	double contrast = 1.0;
	double coef = 255.0 / (double)maxa * contrast;

	Mat img_accu(ah, aw, CV_8UC3);
	for (int p = 0; p < (ah * aw); p++)
	{
		unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
		img_accu.data[(p * 3) + 0] = 255;
		img_accu.data[(p * 3) + 1] = 255 - c;
		img_accu.data[(p * 3) + 2] = 255 - c;
	}
	return img_accu;
}

//Function that uses the CUDA Hough Transform implementation

void transformMultipleImages(vector <Mat> input_data) {

	//Specify if you want the result images to be displayed
	bool display_all_images = false;
	bool display_image_with_index = false;
	//Specify the index of the image for which you want the accumulator and lines calculated
	int image_to_show = 0;
	//Threshold to calculate the lines
	int threshold = 10;

	vector <unsigned char*> image_data;
	vector <pair <int, int>> dimensions;

	for (int i = 0; i < images; i++) {
		image_data.push_back(input_data[i].data);
		dimensions.push_back(make_pair(input_data[i].cols, input_data[i].rows));
	}

	int num_i = image_data.size();

	transformImages::Hough_I hough;
	hough.Transform(image_data, dimensions, num_i);

	vector <Mat> accu_images;

	for (int i = 0; i < num_i; ++i) {
		int aw, ah, maxa;
		aw = ah = maxa = 0;
		const unsigned int* accu = hough.GetAccu(&aw, &ah, i);

		for (int p = 0; p < (ah * aw); p++)
		{
			if ((int)accu[p] > maxa)
				maxa = accu[p];
		}
		double contrast = 1.0;
		double coef = 255.0 / (double)maxa * contrast;

		Mat accu_image(ah, aw, CV_8UC3);
		for (int p = 0; p < (ah * aw); p++)
		{
			unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
			accu_image.data[(p * 3) + 0] = 255;
			accu_image.data[(p * 3) + 1] = 255 - c;
			accu_image.data[(p * 3) + 2] = 255 - c;
		}
		accu_images.push_back(accu_image);
	}

	Mat res = input_data[image_to_show].clone();
	cvtColor(res, res, COLOR_GRAY2BGR);

	vector< pair< pair<int, int>, pair<int, int> > > lines = hough.GetLines(threshold, image_to_show);
	vector< pair< pair<int, int>, pair<int, int> > >::iterator it;

	for (it = lines.begin(); it != lines.end(); it++)
	{
		line(res, Point(it->first.first, it->first.second), Point(it->second.first, it->second.second), Scalar(0, 0, 255), 1, 8);
	}

	if (display_image_with_index)
		if (display_all_images) {
			for (int i = 0; i < num_i; ++i) {
				string result = "resultado " + to_string(i);
				imshow("resultado" + (char)i, accu_images[i]);
			}
			char c = cv::waitKey(360000);
		}else {
			imshow("resultado", res);
			imshow("acumulador", accu_images[image_to_show]);
			char c = cv::waitKey(360000);
		}
}

int main(int argc, char** argv) {

	utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
	cout << images << " images\n" << endl;

	//Initialize the vector of images to process

	vector <Mat> input_images;
	vector <Mat> input_data;

	//Add as many different images as you want to the vector and specify the number in the variable different_images

	int different_images = 1;
	Mat img1 = cv::imread("image_example/prueba.png", cv::IMREAD_GRAYSCALE);
	//Mat img2 = cv::imread("more images", cv::IMREAD_GRAYSCALE);

	input_images.push_back(img1);
	//input_images.push_back(more images);

	for (int i = 0; i < images; i++) {
		int image_index = rand() % different_images;
		input_data.push_back(input_images[image_index]);
	}

	transformMultipleImages(input_data);

	//int threshold = (int)(pixelCount / 12);
	int threshold = 6;

	//Transform
	keymolen::Hough hough1;

	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < images; i++) 
		hough1.Transform(input_data[i].data, input_data[i].cols, input_data[i].rows);
		
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << "CPU execution time: " << duration.count() << " ms" << endl;

	opencvTransform(input_data, threshold);
}