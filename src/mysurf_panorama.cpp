#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

long FRAME_MAX; // 最大フレーム数
#define FRAME_T 9       // フレーム飛ばし間隔
#define f_comp 1
#define f_center 1
#define f_deblur 1
using namespace std;
using namespace cv;

// Tilt :
int SetTiltRotationMatrix(Mat *tiltMatrix, double tilt_deg) {
	double tilt_angle;

	tilt_angle = tilt_deg / 180.0 * M_PI;
	(*tiltMatrix).at<double> (1, 1) = cos(tilt_angle);
	(*tiltMatrix).at<double> (1, 2) = -sin(tilt_angle);
	(*tiltMatrix).at<double> (2, 1) = sin(tilt_angle);
	(*tiltMatrix).at<double> (2, 2) = cos(tilt_angle);

	//	cvmSet(tiltMatrix, 1, 1, cos(tilt_angle));
	//	cvmSet(tiltMatrix, 1, 2, -sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 1, sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 2, cos(tilt_angle));
	return 0;
}

int SetPanRotationMatrix(Mat *panMatrix, double pan_deg) {
	double pan_angle;
	pan_angle = pan_deg / 180.0 * M_PI;

	(*panMatrix).at<double> (2, 2) = cos(pan_angle);
	(*panMatrix).at<double> (2, 0) = -sin(pan_angle);
	(*panMatrix).at<double> (0, 2) = sin(pan_angle);
	(*panMatrix).at<double> (0, 0) = cos(pan_angle);
	//	cvmSet(panMatrix, 2, 2, cos(pan_angle));
	//	cvmSet(panMatrix, 2, 0, -sin(pan_angle));
	//	cvmSet(panMatrix, 0, 2, sin(pan_angle));
	//	cvmSet(panMatrix, 0, 0, cos(pan_angle));
	return 0;
}

// Roll :
int SetRollRotationMatrix(Mat *rollMatrix, double roll_deg) {
	double roll_angle;

	roll_angle = roll_deg / 180.0 * M_PI;
	(*rollMatrix).at<double> (0, 0) = cos(roll_angle);
	(*rollMatrix).at<double> (0, 1) = -sin(roll_angle);
	(*rollMatrix).at<double> (1, 0) = sin(roll_angle);
	(*rollMatrix).at<double> (1, 1) = cos(roll_angle);
	//	cvmSet(rollMatrix, 0, 0, cos(roll_angle));
	//	cvmSet(rollMatrix, 0, 1, -sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 0, sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 1, cos(roll_angle));
	return 0;
}

// Pitch :
int SetPitchRotationMatrix(Mat *pitchMatrix, double pitch_deg) {
	double pitch_angle;

	pitch_angle = pitch_deg / 180.0 * M_PI;
	(*pitchMatrix).at<double> (1, 1) = cos(pitch_angle);
	(*pitchMatrix).at<double> (1, 2) = -sin(pitch_angle);
	(*pitchMatrix).at<double> (2, 1) = sin(pitch_angle);
	(*pitchMatrix).at<double> (2, 2) = cos(pitch_angle);
	//	cvmSet(pitchMatrix, 1, 1, cos(pitch_angle));
	//	cvmSet(pitchMatrix, 1, 2, -sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 1, sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 2, cos(pitch_angle));
	return 0;
}

// Yaw
int SetYawRotationMatrix(Mat *yawMatrix, double yaw_deg) {
	double yaw_angle;

	yaw_angle = yaw_deg / 180.0 * M_PI;
	(*yawMatrix).at<double> (2, 2) = cos(yaw_angle);
	(*yawMatrix).at<double> (2, 0) = -sin(yaw_angle);
	(*yawMatrix).at<double> (0, 2) = sin(yaw_angle);
	(*yawMatrix).at<double> (0, 0) = cos(yaw_angle);
	//	cvmSet(yawMatrix, 2, 2, cos(yaw_angle));
	//	cvmSet(yawMatrix, 2, 0, -sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 2, sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 0, cos(yaw_angle));
	return 0;
}

void setHomographyReset(Mat* homography) {
	cvZero(homography);
	(*homography).at<double> (0, 0) = 1;
	(*homography).at<double> (1, 1) = 1;
	(*homography).at<double> (2, 2) = 1;
	//cvmSet(homography, 0, 0, 1);
	//cvmSet(homography, 1, 1, 1);
	//cvmSet(homography, 2, 2, 1);
}

double compareSURFDescriptors(const float* d1, const float* d2, double best,
		int length) {
	double total_cost = 0;
	assert( length % 4 == 0 );
	for (int i = 0; i < length; i += 4) {
		double t0 = d1[i] - d2[i];
		double t1 = d1[i + 1] - d2[i + 1];
		double t2 = d1[i + 2] - d2[i + 2];
		double t3 = d1[i + 3] - d2[i + 3];
		total_cost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
		if (total_cost > best)
			break;
	}
	return total_cost;
}

void get_histimage(Mat image, Mat *hist_image) {
	MatND hist; // ヒストグラム
	Scalar mean, dev; // 平均と分散の格納先
	float hrange[] = { 0, 256 }; // ヒストグラムの輝度値レンジ
	const float* range[] = { hrange }; // チャネルごとのヒストグラムの輝度値レンジ（グレースケールなので要素数は１）
	int binNum = 126; // ヒストグラムの量子化の値
	int histSize[] = { binNum }; // チャネルごとのヒストグラムの量子化の値
	int channels[] = { 0 }; // ヒストグラムを求めるチャネル指定
	int dims = 1; // 求めるヒストグラムの数


	float max_dev = FLT_MIN, min_dev = FLT_MAX; // エッジ画像におけるヒストグラムの分散のmin max
	float max_mean = FLT_MIN, min_mean = FLT_MAX; // エッジ画像におけるヒストグラムの平均のmin max
	float sum_mean = 0.0;
	Mat count(10, 10, CV_32F, cv::Scalar(0)); // エッジの数を格納するカウンタ
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			double max_value;
			int bin_w;

			calcHist(&Mat(image, cv::Rect(j * 128, i * 72, 128, 72)), 1,
					channels, Mat(), hist, dims, histSize, range, true, false);

			meanStdDev(hist, mean, dev);
			count.at<float> (i, j) = dev[0];

			if (dev[0] < min_dev)
				min_dev = dev[0];
			if (dev[0] > max_dev)
				max_dev = dev[0];

			if (mean[0] < min_mean)
				min_mean = mean[0];
			if (mean[0] > max_mean)
				max_mean = mean[0];

			sum_mean += mean[0];
			std::cout << "count : " << mean << std::endl;

			minMaxLoc(hist, NULL, &max_value, NULL, NULL);

			hist *= hist_image[i * 10 + j].rows / max_value;
			bin_w = cvRound((double) 260 / 256);

			for (int k = 0; k < 256; k++)
				rectangle(hist_image[i * 10 + j], Point(k * bin_w, hist_image[i
						* 10 + j].rows), cvPoint((k + 1) * bin_w, hist_image[i
						* 10 + j].rows - cvRound(hist.data[k])),
						cvScalarAll(0), -1, 8, 0);
		}
	}

}
/*
 *  透視投影変換後の画像をパノラマ平面にマスクを用いて
 *  上書きせずに未投影の領域のみに投影する関数
 *
 * @Param  src パノラマ画像に投影したい画像
 * @Param  dst パノラマ画像
 * @Param mask 投影済みの領域を表したマスク画像
 * @Param  roi 投影したい画像の領域を表した画像
 *
 *  （＊maskは処理後に更新されて返される）
 */
void make_pano(Mat src, Mat dst, Mat mask, Mat roi) {

	//サイズの一致を確認
	if (src.cols == dst.cols && src.rows == dst.rows) {
		int h = src.rows;
		int w = src.cols;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				if (mask.at<unsigned char> (j, i) == 0) {

					dst.at<Vec3b> (j, i) = src.at<Vec3b> (j, i);
					if (roi.at<unsigned char> (j, i) == 255)
						mask.at<unsigned char> (j, i) = roi.at<unsigned char> (
								j, i);
				}
			}
		}
	}
}

int main(int argc, char** argv) {

	if (argc != 6) {
		cout << "argument error\n " << argv[0]
				<< "cam_data center start end blur" << endl;
		return -1;
	}

	VideoCapture cap; // ビデオファイル読み込み
	VideoWriter VideoWriter; // パノラマ動画
	Mat center_img = imread(argv[2]); // センターサークル画像読み込み
	int skip = atoi(argv[3]); // 合成開始フレーム番号
	int end = atoi(argv[4]); // 合成終了フレーム番号
	int frame_num = skip + 1; // 現在のフレーム位置
	int blur = atoi(argv[5]); // ブレのしきい値

	//	char object_filename[128];
	//	char scene_filename[128];

	char imagefileName[256];
	char timefileName[256];
	char sensorfileName[256];

	ofstream log("composition_log.txt");
	vector<string> v_log_str;
	string log_str;

	// 特徴点の集合と特徴量
	std::vector<KeyPoint> objectKeypoints, imageKeypoints;
	Mat objectDescriptors, imageDescriptors;

	//	double tt = (double) cvGetTickCount();

	// ホモグラフィ行列用変数
	Mat homography = Mat(3, 3, CV_64FC1); // 画像対におけるホモグラフィ−
	Mat h_base = cv::Mat::eye(3, 3, CV_64FC1); // センターサークル画像へのホモグラフィ−
	//	vector<int> ptpairs;
	vector<Point2f> pt1, pt2; // 画像対における特徴点の集合
	int n, w, h;

	// パノラマ平面の
	int roll = 0;
	int pitch = 0;
	int yaw = 0;
	Mat A1Matrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat A2Matrix = cv::Mat::eye(3, 3, CV_64FC1);

	Mat rollMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat pitchMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat yawMatrix = cv::Mat::eye(3, 3, CV_64FC1);

	FILE *SensorFP = fopen(argv[1], "r");
	fscanf(SensorFP, "%s", imagefileName);
	fscanf(SensorFP, "%s", timefileName);
	fscanf(SensorFP, "%s", sensorfileName);
	fclose(SensorFP);

	std::ofstream out_Hmat("mat_homography.txt", std::ios::out);

	//　パノラマ平面への射影行列の作成
	A1Matrix.at<double> (0, 2) = -640;
	A1Matrix.at<double> (1, 2) = -360;
	A1Matrix.at<double> (2, 2) = 1080;

	A2Matrix.at<double> (0, 0) = 290;
	A2Matrix.at<double> (1, 1) = 290;
	A2Matrix.at<double> (0, 2) = 640;
	A2Matrix.at<double> (1, 2) = 360;
	SetRollRotationMatrix(&rollMatrix, (double) roll);
	SetPitchRotationMatrix(&pitchMatrix, (double) pitch);
	SetYawRotationMatrix(&yawMatrix, (double) yaw);

	h_base = A2Matrix * A1Matrix * yawMatrix * pitchMatrix * rollMatrix;

	// 動画ファイルをオープン
	if (!(cap.open(string(imagefileName)))) {
		fprintf(stderr, "Avi Not Found!!\n");
		return -1;
	}
	cout << h_base << endl;

	// 総フレーム数の取得
	FRAME_MAX = cap.get(CV_CAP_PROP_FRAME_COUNT);
	float fps = 10;

	std::cout << "Video Property : total flame = " << FRAME_MAX << endl;
	cout << "fps = " << fps << endl;
	// 合成開始フレームまでスキップ
	cap.set(CV_CAP_PROP_POS_FRAMES, skip);

	// 動画から取得した画像対の格納先
	Mat image;
	Mat object;
	Mat gray_image;

	// 最初のフレームを取得（センターサークル画像に差し替え）
	if (f_center)
		image = center_img.clone();
	else
		cap >> image;

	cvtColor(image, gray_image, CV_RGB2GRAY);

	// 各種アルゴリズムによる特徴点検出および特徴量記述
	string algorithm_type = "SURF";

	// SIFT 
	//	cv::SIFT feature;
	//Surf
	SURF feature(5, 3, 4, true);
	feature.getParams(v_log_str);

	/*logging*/
	log << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " "
			<< argv[4] << " " << argv[5] << endl;

	log << "<avi_file_name>" << endl << imagefileName << endl;
	log << "<A1>" << endl << A1Matrix << endl;
	log << "<A2>" << endl << A2Matrix << endl;
	log << "<roll pitch yaw>" << endl;
	log << roll << " " << pitch << " " << yaw << endl;
	log << "<FRAME_ T> " << endl << FRAME_T << endl;
	log << "<Comp>" << endl;
	log << f_comp << endl;
	log << "<use center>" << endl;
	log << f_center << endl;
	log << "<deblur>" << endl << f_deblur << endl;
	log << "<Algorithm> " << endl << algorithm_type << endl;
	log << "<Algorithm Param>" << endl;
	for (int ii = 0; ii < v_log_str.size(); ii++)
		log << v_log_str[ii] << " " << feature.getDouble(v_log_str[ii]) << endl;


	log << "<Algorithm> " << endl << "SURF" << endl;
	log << "<Algorithm Param>" << endl;
	for (int ii = 0; ii < log_str.size(); ii++)
		log << log_str[ii] << " " << feature.getDouble(log_str[ii]) << endl;

	//SurfFeatureDetector detector(5, 3, 4, true);
	//SurfDescriptorExtractor extractor;

	//OrbFeatureDetector detector;
	//OrbDescriptorExtractor extractor;

	// 特徴点の検出と特徴量の記述
	feature(gray_image, Mat(), imageKeypoints, imageDescriptors);

	// ホモグラフィ−行列による変換後の画像格納先
	Mat transform_image; // 画像単体での変換結果
	Mat transform_image2 = cv::Mat::zeros(image.size(), CV_8UC3); // パノラマ平面への変換結果
	warpPerspective(image, transform_image2, h_base, image.size()); // 先頭フレームをパノラマ平面へ投影

	//  パノラマ平面へ射影する際のマスク処理
	w = image.cols;
	h = image.rows;

	Mat mask(h, w, CV_8U, cv::Scalar(0)); //pano black
	Mat matrixA(h, w, CV_8U, cv::Scalar(0)); //pano black
	Mat matrixB(image.rows, image.cols, CV_8U, cv::Scalar(255)); //mask white

	// パノラマ動画ファイルを作成
	VideoWriter.open("panorama.avi", CV_FOURCC('X', 'V', 'I', 'D'), (int) fps,
			cvSize(w, h), 1);

	// フレームを飛ばす
	for (int i = 0; i < FRAME_T; i++) {
		cap >> object;
		frame_num++;
	}

	// 特徴点のマッチャー（ユークリッド距離とハミング距離で使用する関数を変える）
	cv::FlannBasedMatcher matcher;
	//cv::BFMatcher matcher(cv::NORM_HAMMING, true);
	std::vector<cv::DMatch> matches;

	// 手ブレ検出用各種変数 //
	int img_num = 0;
	stringstream ss; // 書き出しファイル名
	cv::Mat tmp_img; //
	cv::Mat sobel_img; // エッジ画像格納先

	vector<Mat> hist_image;
	int blur_skip;

	if (f_center)
		blur_skip = 0;
	else
		blur_skip = FRAME_T;

	while (frame_num <= FRAME_MAX && frame_num <= end) {

		//		while(dev[0] < blur){
		cap >> object;
		frame_num++;
		printf("\nframe=%d\n", frame_num);

		cv::Laplacian(object, tmp_img, CV_32F, 1, 1);
		//Canny(cvarrToMat(objectc), sobel_img,50,100);
		cv::convertScaleAbs(tmp_img, sobel_img, 1, 0);

		// 縦横１０分割したエッジ画像の各ヒストグラムの領域確保
		for (int i = 0; i < 100; i++)
			hist_image.push_back(Mat(200, 260, CV_8U, cv::Scalar(255)));
		/*
		 // ヒストグラム画像を作成
		 get_histimage(sobel_img, hist_image.data());

		 // 各ヒストグラムを順次表示
		 cvNamedWindow("Histogram", CV_WINDOW_AUTOSIZE);
		 for(int i = 0; i < 100;i++){
		 imshow("Histogram", hist_image[i]);
		 cvWaitKey(0);
		 }
		 */
		/*
		 if(count < blur){
		 ss << "img/img_"<< frame_num << ".jpg";
		 std::cout << ss.str();
		 imwrite(ss.str(),cvarrToMat(imagec));
		 ss.clear();
		 ss.str("");

		 ss << "img/sobel_img_"<< frame_num << ".jpg";
		 std::cout << ss.str();
		 imwrite(ss.str(),sobel_img);
		 ss.clear();
		 ss.str("");
		 img_num++;
		 count = 0;
		 std::cout << "skip frame : " << frame_num << std::endl;
		 }
		 */

		/*
		 if (object.empty() || image.empty()) {
		 fprintf(stderr, "Can not load %s and/or %s\n"
		 "Usage: find_obj [<object_filename> <scene_filename>]\n",
		 object_filename, scene_filename);
		 exit(-1);
		 }
		 */
		cvtColor(object, gray_image, CV_RGB2GRAY);
		feature(gray_image, Mat(), objectKeypoints, objectDescriptors);

		cv::Laplacian(object, tmp_img, CV_32F, 1, 1);
		cv::convertScaleAbs(tmp_img, sobel_img, 1, 0);
		ss << "img/img_" << frame_num << ".jpg";
		std::cout << ss.str() << endl;

		imwrite(ss.str(), image);
		ss.clear();
		ss.str("");

		ss << "img/sobel_img_" << frame_num << ".jpg";
		std::cout << ss.str() << endl;
		imwrite(ss.str(), sobel_img);
		ss.clear();
		ss.str("");

		matcher.match(objectDescriptors, imageDescriptors, matches);
		cout << "matches : " << matches.size() << endl;

		double min_dist = DBL_MAX;
		for (int i = 0; i < (int) matches.size(); i++) {
			double dist = matches[i].distance;
			if (dist < min_dist)
				min_dist = dist;
		}

		cout << "min dist :" << min_dist << endl;

		//  対応点間の移動距離による良いマッチングの取捨選択
		std::vector<cv::DMatch> good_matches;
		std::vector<KeyPoint> good_objectKeypoints;
		std::vector<KeyPoint> good_imageKeypoints;
		good_matches.clear();
		pt1.clear();
		pt2.clear();
		for (int i = 0; i < (int) matches.size(); i++) {
			if (round(objectKeypoints[matches[i].queryIdx].class_id) == round(
					imageKeypoints[matches[i].trainIdx].class_id)) {
				if (matches[i].distance < min_dist * 4) {
					//		  &&	(fabs(objectKeypoints[matches[i].queryIdx].pt.y - imageKeypoints[matches[i].trainIdx].pt.y)
					//		/ fabs(objectKeypoints[matches[i].queryIdx].pt.x - 	imageKeypoints[matches[i].trainIdx].pt.x)) < 0.1) {

					good_matches.push_back(matches[i]);
					pt1.push_back(objectKeypoints[matches[i].queryIdx].pt);
					pt2.push_back(imageKeypoints[matches[i].trainIdx].pt);
					good_objectKeypoints.push_back(
							objectKeypoints[matches[i].queryIdx]);
					good_imageKeypoints.push_back(
							imageKeypoints[matches[i].trainIdx]);
				}
			}
		}

		// マッチング結果をリサイズして表示
		Mat result, r_result;
		cv::drawMatches(object, objectKeypoints, image, imageKeypoints,
				good_matches, result);
		resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);
		namedWindow("matches", CV_WINDOW_AUTOSIZE);
		imshow("matches", r_result);
		waitKey(30);

		imageKeypoints = objectKeypoints;
		objectDescriptors.copyTo(imageDescriptors);
		image = object.clone();
		/*
		 cout << "補完開始" << endl;
		 vector<Point2f> dist;
		 vector<Point2f> est_pt1 = pt1, est_pt2;
		 Mat est_h_base = h_base.clone();
		 float inv_skip = 1.0 / (float) (blur_skip + 1);
		 frame_num -= blur_skip;
		 cap.set(CV_CAP_PROP_POS_FRAMES, frame_num - blur_skip);
		 cout << "pt1 " << pt1[0] << endl;
		 cout << "pt2 " << pt2[0] << endl;
		 for (int k = 0; k < blur_skip; k++) {
		 est_pt2.clear();
		 for (int l = 0; l < good_objectKeypoints.size(); l++)
		 est_pt2.push_back(est_pt1[l] + (pt2[l] - pt1[l]) * inv_skip);
		 cout << "est_pt1 " << est_pt1[0] << endl;
		 cout << "est_pt2 " << est_pt2[0] << endl;
		 //waitKey(0);
		 // 補完した点でホモグラフィ−行列を計算
		 n = pt1.size() / 2;
		 printf("n = %d\n", n);
		 homography = findHomography(Mat(est_pt1), Mat(est_pt2), CV_RANSAC,
		 5.0);

		 // パノラマ平面へのホモグラフィーを計算
		 est_h_base = est_h_base * homography;
		 //h_base = h_base * homography;

		 // 飛ばしたフレームを取得しパノラマ平面へ投影


		 cap >> object;
		 warpPerspective(object, transform_image, est_h_base, object.size());
		 Mat h2 = est_h_base;
		 warpPerspective(matrixB, matrixA, h2, matrixB.size(),
		 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		 // 特徴点をコピー
		 est_pt1 = est_pt2;

		 for (int i = 0; i < w; i++) {
		 for (int j = 0; j < h; j++) {
		 cal = transform_image.at<Vec3b> (j, i);
		 //               if(cal.val[0]!=0||cal.val[1]!=0||cal.val[2]!=0){
		 tmpc = transform_image2.at<Vec3b> (j, i);
		 if (mask.at<unsigned char> (j, i) == 0) {
		 //if(tmpc.val[0]==0&&tmpc.val[1]==0&&tmpc.val[2]==0){
		 tmpc.val[0] = cal.val[0];
		 tmpc.val[1] = cal.val[1];
		 tmpc.val[2] = cal.val[2];
		 if (matrixA.at<unsigned char> (j, i) == 255)
		 mask.at<unsigned char> (j, i) = matrixA.at<
		 unsigned char> (j, i);

		 }
		 transform_image2.at<Vec3b> (j, i) = tmpc;
		 }
		 }
		 ss << "frame = " << frame_num;
		 putText(transform_image, ss.str(), Point(100, 100),
		 CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		 ss.clear();
		 ss.str("");
		 VideoWriter.write(transform_image);
		 imshow("Object Correspond", transform_image2);
		 frame_num++;
		 waitKey(30);
		 //cvWaitKey(0);

		 }
		 */
		//n = ptpairs.size()/2;

		n = pt1.size();
		printf("n = %d\n", n);
		printf("num_of_obj = %d\n", good_objectKeypoints.size());
		printf("num_of_img = %d\n", good_imageKeypoints.size());
		if (n >= 4) {

			//			_pt1 = Mat(pt1);
			//			_pt2 = Mat(pt2);
			homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
		} else {
			setHomographyReset(&homography);
			printf("frame_num = %d\n", frame_num);
		}

		cv::Mat tmp = homography.clone();
		out_Hmat << tmp << endl;

		h_base = h_base * homography;

		// 補完の際に上書きしているのでフレームを再取得
		cap >> object;
		warpPerspective(object, transform_image, h_base, object.size());

		Mat h2 = h_base;
		warpPerspective(matrixB, matrixA, h2, matrixB.size(), CV_INTER_LINEAR
				| CV_WARP_FILL_OUTLIERS);

		make_pano(transform_image, transform_image2, mask, matrixA);
		ss << "frame = " << frame_num;
		putText(transform_image, ss.str(), Point(100, 100),
				CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		ss.clear();
		ss.str("");
		VideoWriter.write(transform_image);
		imshow("Object Correspond", transform_image2);
		waitKey(30);
		blur_skip = FRAME_T;
		for (int i = 0; i < FRAME_T; i++) {

			cap >> object;
			frame_num++;
		}
	}

	namedWindow("Object Correspond", CV_WINDOW_AUTOSIZE);
	imshow("Object Correspond", transform_image2);
	cvWaitKey(0);
	imwrite("transform4.jpg", transform_image2);

	return 0;
}
