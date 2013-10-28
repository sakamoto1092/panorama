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
#include <boost/program_options.hpp>
#include"3dms-func.h"
#define PANO_W 6000
#define PANO_H 3000
using namespace std;
using namespace cv;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using boost::program_options::notify;

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
	int binNum = 256; // ヒストグラムの量子化の値
	int histSize[] = { binNum }; // チャネルごとのヒストグラムの量子化の値
	int channels[] = { 0 }; // ヒストグラムを求めるチャネル指定
	int dims = 1; // 求めるヒストグラムの数


	float max_dev = FLT_MIN, min_dev = FLT_MAX; // エッジ画像におけるヒストグラムの分散のmin max
	float max_mean = FLT_MIN, min_mean = FLT_MAX; // エッジ画像におけるヒストグラムの平均のmin max
	float sum_mean = 0.0;
	Rect roi_rect;
	Mat count(10, 10, CV_32F, cv::Scalar(0)); // エッジの数を格納するカウンタ

	cout << "making histgram" << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			double max_value;
			int bin_w;
			Mat tmp_img(image, cv::Rect(j * 128, i * 72, 128, 72));
			calcHist(&tmp_img, 1, channels, Mat(), hist, dims, histSize, range,
					true, false);

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
						* 10 + j].rows - cvRound(hist.at<float> (k))),
						cvScalarAll(0), -1, 8, 0);
			roi_rect.width = tmp_img.cols;
			roi_rect.height = tmp_img.rows;
			roi_rect.x = 260;
			Mat roi(hist_image[i * 10 + j], roi_rect);
			tmp_img.copyTo(roi);
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
/*
 * @Param descriptors1 特徴量１
 * @Param descriptors2 特徴量２
 * @Param key1         特徴点１
 * @Param key2         特徴点２
 * @Param matches      良いマッチングの格納先
 * @Param pt1          良いマッチングの特徴点座標１
 * @Param pt2          良いマッチングの特徴点座標２
 */
void good_matcher(Mat descriptors1, Mat descriptors2, vector<KeyPoint> *key1,
		vector<KeyPoint> *key2, std::vector<cv::DMatch> *matches, vector<
				Point2f> *pt1, vector<Point2f> *pt2) {

	FlannBasedMatcher matcher;
	vector<std::vector<cv::DMatch> > matches12, matches21;
	std::vector<cv::DMatch> tmp_matches;
	int knn = 1;
	//BFMatcher matcher(cv::NORM_HAMMING, true);
	//matcher.match(objectDescriptors, imageDescriptors, matches);

	cout << key1->size() << endl;

	matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
	matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
	tmp_matches.clear();
	// KNN探索で，1->2と2->1が一致するものだけがマッチしたとみなされる
	for (size_t m = 0; m < matches12.size(); m++) {
		bool findCrossCheck = false;
		for (size_t fk = 0; fk < matches12[m].size(); fk++) {
			cv::DMatch forward = matches12[m][fk];
			for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++) {
				cv::DMatch backward = matches21[forward.trainIdx][bk];
				if (backward.trainIdx == forward.queryIdx) {
					tmp_matches.push_back(forward);
					findCrossCheck = true;
					break;
				}
			}
			if (findCrossCheck)
				break;
		}
	}
	cout << "matches : " << tmp_matches.size() << endl;
	double min_dist = DBL_MAX;
	for (int i = 0; i < (int) tmp_matches.size(); i++) {
		double dist = tmp_matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
	}

	cout << "min dist :" << min_dist << endl;

	//  対応点間の移動距離による良いマッチングの取捨選択
	matches->clear();
	pt1->clear();
	pt2->clear();
	for (int i = 0; i < (int) tmp_matches.size(); i++) {
		if (round((*key1)[tmp_matches[i].queryIdx].class_id) == round(
				(*key2)[tmp_matches[i].trainIdx].class_id)) {
			if (tmp_matches[i].distance > 0 && tmp_matches[i].distance
					< (min_dist) * 3) {
				//		  &&	(fabs(objectKeypoints[matches[i].queryIdx].pt.y - imageKeypoints[matches[i].trainIdx].pt.y)
				//		/ fabs(objectKeypoints[matches[i].queryIdx].pt.x - 	imageKeypoints[matches[i].trainIdx].pt.x)) < 0.1) {
				//				cout << "i : " << i << endl;
				matches->push_back(tmp_matches[i]);
				pt1->push_back((*key1)[tmp_matches[i].queryIdx].pt);
				pt2->push_back((*key2)[tmp_matches[i].trainIdx].pt);
				//good_objectKeypoints.push_back(
				//		objectKeypoints[tmp_matches[i].queryIdx]);
				//good_imageKeypoints.push_back(
				//		imageKeypoints[tmp_matches[i].trainIdx]);
			}
		}
	}
}

int main(int argc, char** argv) {

	VideoWriter VideoWriter; // パノラマ動画
	VideoCapture cap; // ビデオファイル


	// 動画から取得した各種画像の格納先
	Mat image; // 投影先の画像
	Mat object; // 変換元の画像
	Mat gray_image; // 変換元の微分画像
	Mat center_img; // センターサークル画像

	// ホモグラフィ行列による変換後の画像格納先
	Mat transform_image; // 画像単体での変換結果
	Mat transform_image2; // パノラマ平面への変換結果

	// マスク関係画像（既存のパノラマの未描画領域のみに投影するため）
	Mat mask; // パノラマ画像のマスク
	Mat pano_black; // パノラマ画像と同じサイズの黒画像
	Mat white_img; // フレームと同じサイズの白画像

	int skip; // 合成開始フレーム番号
	long end; // 合成終了フレーム番号
	long frame_num; // 現在のフレーム位置
	int blur; // ブレのしきい値
	long FRAME_MAX; // 動画の最大フレーム数
	int FRAME_T; // フレーム飛ばし間隔

	// 各種フラグ
	bool f_comp = false; // 線形補完
	bool f_center = false; // センターサークル中心
	bool f_video = false; // ビデオ書き出し

	float fps = 20; // 書き出しビデオのfps
	string n_video; // 書き出しビデオファイル名
	string cam_data; // 映像センサファイル名
	string n_center; // センターサークル画像名

	// 映像 センサファイル名取得
	char imagefileName[256];
	char timefileName[256];
	char sensorfileName[256];
	FILE *SensorFP;

	// 取り出したフレームのセンサ情報(obj_sd)
	// パノラマ合成元フレームの中でobj_sdに最も近いフレームのセンサ情報
	SENSOR_DATA obj_sd, near_sd;

	// パノラマ背景に使われたフレームのセンサ情報のvector
	vector<SENSOR_DATA> pano_sds;

	// パノラマ背景に使われたフレーム番号のvector
	vector<int> vec_n_pano_frames;

	// パノラマ背景に使われたフレームのホモグラフィ行列
	vector<Mat> pano_monographys;

	// 合成に使ったホモグラフィ行列をxmlに出力
	FileStorage cvfs("aria_H.xml", cv::FileStorage::WRITE);

	// 手ブレ検出用各種変数
	int img_num = 0;
	stringstream ss; // 書き出しファイル名
	cv::Mat tmp_img; //
	cv::Mat sobel_img; // エッジ画像格納先

	vector<Mat> hist_image;
	int blur_skip;

	// 各種アルゴリズムによる特徴点検出および特徴量記述
	string algorithm_type;
	Ptr<Feature2D> feature;
	int hessian;
	// SIFT
	//SIFT feature;

	// SURF
	//	SURF feature(5, 3, 4, true);

	// ORB
	//ORB featur;


	// 対応点の対の格納先
	std::vector<cv::DMatch> matches; // matcherにより求めたおおまかな対を格納

	// 特徴点の集合と特徴量
	std::vector<KeyPoint> objectKeypoints, imageKeypoints;
	Mat objectDescriptors, imageDescriptors;
	vector<Point2f> pt1, pt2; // 画像対における特徴点座の集合

	// より良いペアの格納先
	std::vector<cv::DMatch> good_matches;
	std::vector<KeyPoint> good_objectKeypoints, good_imageKeypoints;
	Mat good_objectDescriptors, good_imageDescriptors;

	Mat homography = Mat(3, 3, CV_64FC1); // 画像対におけるホモグラフィ
	Mat h_base = cv::Mat::eye(3, 3, CV_64FC1); // パノラマ平面へのホモグラフィ
	int n, w, h;

	// パノラマ平面の構成
	int roll = 0;
	int pitch = 0;
	double yaw = 0;
	Mat A1Matrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat A2Matrix = cv::Mat::eye(3, 3, CV_64FC1);

	Mat rollMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat pitchMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat yawMatrix = cv::Mat::eye(3, 3, CV_64FC1);

	//　パノラマ平面への射影行列の作成
	//	A1Matrix.at<double> (0, 2) = -640;
	//	A1Matrix.at<double> (1, 2) = -360;
	//	A1Matrix.at<double> (2, 2) = 1080;

	// GCの内部パラメータの逆行列
	A1Matrix.at<double> (0, 0) = 1.1107246554597004e-003;
	A1Matrix.at<double> (0, 2) = -7.0476759105087217e-001;
	A1Matrix.at<double> (1, 1) = 1.0937849972977411e-003;
	A1Matrix.at<double> (1, 2) = -4.2040554903081440e-001;

	// AQの内部パラメータの逆行列
	//	A1Matrix.at<double> (0, 0) = 8.1632905612490970e-004;
	//	A1Matrix.at<double> (0, 2) = -5.2593546441192318e-001;
	//	A1Matrix.at<double> (1, 1) = 8.1390778599629236e-004;
	//	A1Matrix.at<double> (1, 2) = -2.7706041350882804e-001;

	cout << "<A1>" << endl << A1Matrix.inv() << endl;

	Mat inv_a1 = A1Matrix.inv();
	// GCでの仮想パノラマ平面
	//A2Matrix.at<double> (0, 0) = 900;
	//A2Matrix.at<double> (1, 1) = 900;
	//A2Matrix.at<double> (0, 2) = PANO_W / 2;
	//A2Matrix.at<double> (1, 2) = PANO_H / 2;

	// AQでの仮想パノラマ平面
	A2Matrix.at<double> (0, 0) = inv_a1.at<double> (0, 0);
	A2Matrix.at<double> (1, 1) = inv_a1.at<double> (1, 1);
	A2Matrix.at<double> (0, 2) = PANO_W / 2;
	A2Matrix.at<double> (1, 2) = PANO_H / 2;

	cout << "<A2>" << endl << A2Matrix << endl;

	try {
		// コマンドラインオプションの定義
		options_description opt("Usage");
		opt.add_options()("cam", value<std::string> ()->default_value(
				"cam_data.txt"), "動画名やセンサファイル名が記述されたファイルの指定")("center", value<
				std::string> (), "センター画像の指定")("start,s",
				value<int> ()->default_value(0), "スタートフレームの指定")("end,e", value<
				int> (), "終了フレームの指定")("comp", value<bool> ()->default_value(
				false), "補完の設定")("inter,i", value<int> ()->default_value(9),
				"取得フレームの間隔")("blur,b", value<int> ()->default_value(0),
				"ブラーによるフレームの破棄の閾値")("video,v", value<std::string> (),
				"書きだす動画ファイル名の指定")("yaw", value<double> ()->default_value(0),
				"初期フレーム投影時のyaw")("fps,f", value<int> ()->default_value(30),
				"書きだす動画のフレームレートの指定")("algo,a", value<string> ()->default_value(
				"SURF"), "特徴点抽出等のアルゴリズムの指定")("hessian",
				value<int> ()->default_value(50), "SURFのhessianの値")("help,h",
				"ヘルプの出力");

		// オプションのマップを作成
		variables_map vm;
		store(parse_command_line(argc, argv, opt), vm);
		notify(vm);

		// 必須オプションの確認
		if (vm.count("help")) {
			cout << "  [option]... \n" << opt << endl;
			return -1;
		}

		// 各種オプションの値を取得
		cam_data = vm["cam"].as<string> (); // 映像 センサファイル名

		if (vm.count("start")) // スタートフレーム番号
			skip = vm["start"].as<int> ();
		else
			skip = 0;

		if (vm.count("end")) // 終了フレーム番号
			end = vm["end"].as<int> ();
		else
			end = 20000;

		if (vm.count("center")) { // センターサークル画像名
			n_center = vm["center"].as<string> ();
			f_center = true;
		}

		if (vm.count("video")) { // 書き出し動画ファイル名
			n_video = vm["video"].as<string> ();
			f_video = true;
		}

		algorithm_type = vm["algo"].as<string> ();
		f_comp = vm["comp"].as<bool> (); // 補完の有効無効
		FRAME_T = vm["inter"].as<int> (); // フレーム取得間隔
		blur = vm["blur"].as<int> (); // 手ブレ閾値
		fps = vm["fps"].as<int> (); // 書き出し動画のfps
		yaw = vm["yaw"].as<double> (); // 初期フレーム角度
		hessian = vm["hessian"].as<int> ();

	} catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		return -1;
	} catch (...) {
		cerr << "Exception of unknown type!\n";
		return -1;
	}

	// 映像　センサファイル名を取得
	if ((SensorFP = fopen(cam_data.c_str(), "r")) == NULL) {
		cerr << "cant open cam_data.txt" << endl;
		return -1;
	}
	fscanf(SensorFP, "%s", imagefileName);
	fscanf(SensorFP, "%s", timefileName);
	fscanf(SensorFP, "%s", sensorfileName);
	fclose(SensorFP);

	if (f_center)
		center_img = imread(n_center);

	if (f_center && center_img.empty()) {
		cerr << "Cant open center_img " << n_center << endl;
		f_center = false;
		//return -1;
	}

	// 動画ファイルをオープン
	if (!(cap.open(string(imagefileName)))) {
		fprintf(stderr, "Avi Not Found!!\n");
		return -1;
	}

	// 総フレーム数の取得
	FRAME_MAX = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (FRAME_MAX < end)
		end = FRAME_MAX;

	std::cout << "Video Property : total flame = " << FRAME_MAX << endl;
	cout << "fps = " << fps << endl;

	// 合成開始フレームまでスキップ
	if (skip > 1) {
		cap.set(CV_CAP_PROP_POS_FRAMES, skip);
		frame_num = skip - 1; // 現在のフレーム位置を設定
	} else {
		frame_num = 0;
	}
	feature = Feature2D::create(algorithm_type);
	if (algorithm_type.compare("SURF") == 0) {
		feature->set("extended", 1);
		feature->set("hessianThreshold", hessian);
		feature->set("nOctaveLayers", 4);
		feature->set("nOctaves", 3);
		feature->set("upright", 0);
	}

	//	double tt = (double) cvGetTickCount();

	// 各種回転をパノラマ平面に適用
	SetRollRotationMatrix(&rollMatrix, (double) roll);
	SetPitchRotationMatrix(&pitchMatrix, (double) pitch);
	SetYawRotationMatrix(&yawMatrix, (double) yaw);

	// 最終的なパノラマ平面へのホモグラフィ行列を計算
	h_base = A2Matrix * rollMatrix * pitchMatrix * yawMatrix * A1Matrix;

	cout << "<h_base>" << endl << h_base << endl;

	/*logging*/
	ofstream log("composition_log.txt");
	vector<string> v_log_str;
	string log_str;

	//feature->getParams(v_log_str);
	feature->getParams(v_log_str);

	log << "<avi_file_name>" << endl << imagefileName << endl;
	log << "<A1>" << endl << A1Matrix << endl;
	log << "<A2>" << endl << A2Matrix << endl;
	log << "<roll pitch yaw>" << endl;
	log << roll << " " << pitch << " " << yaw << endl;
	log << "<FRAME_ T> " << endl << FRAME_T << endl;
	log << "<Comp>" << endl;
	log << f_comp << endl;
	log << "<use center>" << endl;
	if (center_img.empty())
		log << 0 << endl;
	else
		log << 1 << endl;
	log << "<deblur>" << endl << blur << endl;
	log << "<start>" << endl << skip << endl;
	log << "<end>" << endl << end << endl;
	log << "<inter>" << endl << FRAME_T << endl;
	log << "<Algorithm> " << endl << algorithm_type << endl;
	log << "<Algorithm Param>" << endl;
	for (int ii = 0; ii < v_log_str.size(); ii++)
		log << v_log_str[ii] << " " << feature->getDouble(v_log_str[ii])
				<< endl;
	/*end of logging*/

	SENSOR_DATA *sensor = (SENSOR_DATA *) malloc(sizeof(SENSOR_DATA) * 5000);
	double s_time;
	// 最初のフレームを取得（センターサークル画像に差し替え）
	if (f_center) {
		image = center_img.clone();
	} else {

		// 撮影時間とセンサ情報（ORIとTIME）を取得
		string time_buf;
		ifstream ifs_time(timefileName);

		if (!ifs_time.is_open()) {
			cerr << "cannnot open TIME file : " << timefileName << endl;
			return -1;
		}

		// 三回読み飛ばして秒数を文字で取得
		ifs_time >> time_buf;
		ifs_time >> time_buf;
		ifs_time >> time_buf;
		ifs_time >> s_time; // msec

		cout << "s_time : " << s_time / 1000.0 << "[sec]" << endl;

		// 対象フレームのセンサデータを一括読み込み
		LoadSensorData(sensorfileName, &sensor);

		frame_num++;//現在のフレーム位置を更新
		// 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
		double frame_msec = cap.get(CV_CAP_PROP_POS_MSEC) + s_time;
		GetSensorDataForTime(frame_msec / 1000.0, &sensor, &obj_sd);
		cap >> image;
	}

	cout << cap.get(CV_CAP_PROP_FPS) << "[frame / sec]" << endl;

	Mat
			dist =
					(Mat_<double> (1, 5) << 4.6607295014012916e-002, 4.1437801936723750e-001, -7.4809715282343212e-003, -2.9591314503800725e-003, -2.1417165056372101e+000);
	cout << "dist param : " << dist << endl;
	//w = image.cols;
	// h = image.rows;
	//CvArr cvimage = image;
	//IplImage dist_img = image;
	// IplImage *undist_img = cvCreateImage(cvSize(w, h), 8, 3);
	//Mat inv_a1 = A1Matrix.inv();
	//CvMat a1 = inv_a1;
	//CvMat cvdist(dist);
	//cvUndistort2(&dist_img, undist_img, &a1, &(CvMat) dist);

	//Mat dist_src = image.clone();
	//undistort(dist_src, image, A1Matrix.inv(), dist);

	cvtColor(image, gray_image, CV_RGB2GRAY);

	// マスクイメージ関係を生成
	//  パノラマ平面へ射影する際のマスク処理

	mask = Mat(PANO_H, PANO_W, CV_8U, cv::Scalar(0));
	pano_black = Mat(PANO_H, PANO_W, CV_8U, cv::Scalar(0));
	white_img = Mat(image.rows, image.cols, CV_8U, cv::Scalar(255));
	transform_image2 = cv::Mat::zeros(Size(PANO_W, PANO_H), CV_8UC3);

	// 特徴点の検出と特徴量の記述
	feature->operator ()(gray_image, Mat(), imageKeypoints, imageDescriptors);

	warpPerspective(image, transform_image2, h_base, Size(PANO_W, PANO_H)); // 先頭フレームをパノラマ平面へ投影


	ss << "homo_" << frame_num;
	write(cvfs, ss.str(), h_base);
	ss.clear();
	ss.str("");

	// パノラマ動画ファイルを作成
	if (f_video)
		VideoWriter.open(n_video.c_str(), CV_FOURCC('X', 'V', 'I', 'D'),
				(int) fps, cvSize(w, h), 1);

	// フレームを飛ばす
	if (!f_center) {
		for (int i = 0; i < FRAME_T; i++) {
			cap >> object;
			frame_num++;
		}
		// このフレームのセンサとフレーム番号，ホモグラフィ行列を記録
		pano_sds.push_back(obj_sd);
		vec_n_pano_frames.push_back(frame_num);
		pano_monographys.push_back(h_base);
	}

	// センターサークル画像を用いた場合，センターサークルと動画の最初のフレーム間は
	// 補完しないようにする
	if (f_center)
		blur_skip = 0;
	else
		blur_skip = FRAME_T;
	bool f_hist = false;
	bool f_blur = true;
	long count_blur = 100;
	double obj_frame_msec;
	namedWindow("Object Correspond", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);

	while (frame_num < end && frame_num < FRAME_MAX + FRAME_T + 1) {

		while (f_blur) {
			obj_frame_msec = cap.get(CV_CAP_PROP_POS_MSEC);
			cap >> object;
			if (object.empty())
				break;
			frame_num++;
			printf("\nframe=%d\n", frame_num);
			cvtColor(object, gray_image, CV_RGB2GRAY);
			//cv::Laplacian(gray_image, tmp_img, CV_16S, 3);
			//Canny(object, sobel_img, 50, 200);
			cv::Sobel(gray_image, tmp_img, CV_32F, 1, 1);
			cv::convertScaleAbs(tmp_img, sobel_img, 1, 0);

			cv::Mat_<float>::iterator it = sobel_img.begin<float> ();
			long count = 0;
			for (; it != sobel_img.end<float> (); ++it) {
				if ((*it) > 150)
					count++;
			}
			cout << "edge : " << count << endl;
			if (count > blur)
				f_blur = false;
			count = 0;
		}

		f_blur = true;
		if (object.empty())
			break;
		// 縦横１０分割したエッジ画像の各ヒストグラムの領域確保
		if (f_hist) {
			for (int i = 0; i < 100; i++)
				hist_image.push_back(
						Mat(200, 260 + 128, CV_8U, cv::Scalar(255)));

			// ヒストグラム画像を作成
			get_histimage(sobel_img, hist_image.data());

			// 各ヒストグラムを順次表示
			//cvNamedWindow("Histogram", CV_WINDOW_AUTOSIZE);
			for (int i = 0; i < 100; i++) {
				//imshow("Histogram", hist_image[i]);
				ss << "img/hist_img_" << frame_num << "_" << i << ".jpg";
				imwrite(ss.str(), hist_image[i]);
				ss.clear();
				ss.str("");
				//cvWaitKey(0);
			}

			// if(count < blur){
			ss << "img/img_" << frame_num << ".jpg";
			std::cout << ss.str();
			imwrite(ss.str(), image);
			ss.clear();
			ss.str("");

			ss << "img/sobel_img_" << frame_num << ".jpg";
			std::cout << ss.str();
			imwrite(ss.str(), sobel_img);
			ss.clear();
			ss.str("");
			//img_num++;
			//count = 0;
			//std::cout << "skip frame : " << frame_num << std::endl;
			// }
		}

		// 歪み補正
		//Mat	dist =	(Mat_<double> (1, 5) << 4.0557296604988635e-002, -7.0495680844213993e-001, 1.4154080043873203e-002, -2.7104946840592046e-003, 2.9299467217460284e+000);
		//Mat dist_src = object.clone();

		//dist_img = object.clone();
		//IplImage *undist_img = cvCreateImage(cvSize(w, h), 8, 3);
		//Mat inv_a1 = A1Matrix.inv();
		//CvMat a1 = inv_a1;
		//CvMat cvdist(dist);

		//cvUndistort2(&dist_img, undist_img, &a1, &(CvMat) dist);
		//object = cvarrToMat(undist_img);

		//undistort(dist_src, object, A1Matrix.inv(), dist);


		cvtColor(object, gray_image, CV_RGB2GRAY);
		feature->operator ()(gray_image, Mat(), objectKeypoints,
				objectDescriptors);

		// TODO : 取り出したフレーム(object)に近いフレーム(image)を
		//         センサ情報から探して特徴点抽出をしてマッチングする

		// 取り出したフレームの秒数を取得
		double frame_msec = obj_frame_msec + s_time;
		GetSensorDataForTime(frame_msec / 1000.0, &sensor, &obj_sd);
		pano_sds.push_back(obj_sd);
		vec_n_pano_frames.push_back(frame_num);


		// 近い角度のフレームを計算する
		Mat vec1(3, 1, CV_64F), vec2(3, 1, CV_64F);
		Mat near_vec(3, 1, CV_64F);
		long near_frame;

		SetYawRotationMatrix(&yawMatrix, obj_sd.alpha);
		SetPitchRotationMatrix(&pitchMatrix, obj_sd.beta);
		SetRollRotationMatrix(&rollMatrix, obj_sd.gamma);
		//warpPerspective(Point3d(1, 0, 0), vec1, yawMatrix * pitchMatrix* rollMatrix);
		vec1 = yawMatrix * pitchMatrix * rollMatrix * (cv::Mat_<double>(3, 1)
				<< 1, 0, 0);
		double dist, min = DBL_MAX;
		cout << obj_sd.TT<< " [sec]" << endl;
		for (vector<SENSOR_DATA>::iterator sd_it = pano_sds.begin(); sd_it
				< pano_sds.end(); sd_it++) {
			SetYawRotationMatrix(&yawMatrix, (*sd_it).alpha);
			SetPitchRotationMatrix(&pitchMatrix, (*sd_it).beta);
			SetRollRotationMatrix(&rollMatrix, (*sd_it).gamma);
			vec2 = yawMatrix * pitchMatrix * rollMatrix * (cv::Mat_<double>(3,
					1) << 1, 0, 0);

			dist = sqrt(pow(vec1.at<double> (0, 0) - vec2.at<double> (0, 0), 2)
					+ pow(vec1.at<double> (0, 1) - vec2.at<double> (0, 1), 2)
					+ pow(vec1.at<double> (0, 2) - vec2.at<double> (0, 2), 2));
			//cout << "dist : " << dist << endl;
			// 近いものがあったらnear_sdを更新
			if (dist < min) {
				min = dist;
				near_sd = (*sd_it);
				cout << "update min " << (*sd_it).TT <<endl;
			}

		}
		min = DBL_MAX;
		cout << vec2 << endl;

		long tmp_frame_num;
		tmp_frame_num = cap.get(CV_CAP_PROP_POS_FRAMES);
		cap.set(CV_CAP_PROP_POS_MSEC, near_sd.TT*1000.0 - s_time);
		//cout << "detect near frame : " << near_sd.TT << " [sec]" << endl;
		cout << cap.get(CV_CAP_PROP_POS_FRAMES) << " [frame]" << endl;

		cap.set(CV_CAP_PROP_POS_FRAMES, tmp_frame_num);

		good_matcher(objectDescriptors, imageDescriptors, &objectKeypoints,
				&imageKeypoints, &matches, &pt1, &pt2);

		cout << "selected good_matches : " << pt1.size() << endl;
		// マッチング結果をリサイズして表示
		Mat result, r_result;
		drawMatches(object, objectKeypoints, image, imageKeypoints, matches,
				result);
		resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);
		namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		imshow("matches", r_result);
		waitKey(30);

		imageKeypoints = objectKeypoints;
		objectDescriptors.copyTo(imageDescriptors);
		image = object.clone();
		if (f_comp && blur_skip != 0) {
			cout << "start comp" << endl;
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
				for (int l = 0; l < pt2.size(); l++)
					est_pt2.push_back(est_pt1[l] + (pt2[l] - pt1[l]) * inv_skip);
				cout << "est_pt1 " << est_pt1[0] << endl;
				cout << "est_pt2 " << est_pt2[0] << endl;

				// 補完した点でホモグラフィ−行列を計算
				n = pt1.size() / 2;
				printf("n = %d\n", n);
				homography = findHomography(Mat(est_pt1), Mat(est_pt2),
						CV_RANSAC, 5.0);

				// パノラマ平面へのホモグラフィーを計算
				est_h_base = est_h_base * homography;

				// 飛ばしたフレームを取得しパノラマ平面へ投影
				cap >> object;
				warpPerspective(object, transform_image, est_h_base,
						object.size());

				warpPerspective(white_img, pano_black, est_h_base,
						white_img.size(), CV_INTER_LINEAR
								| CV_WARP_FILL_OUTLIERS);

				// 特徴点をコピー
				est_pt1 = est_pt2;
				make_pano(transform_image, transform_image2, mask, pano_black);

				ss << "frame = " << frame_num;
				putText(transform_image, ss.str(), Point(100, 100),
						CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1,
						8);
				ss.clear();
				ss.str("");
				if (f_video)
					VideoWriter.write(transform_image);
				imshow("Object Correspond", transform_image2);
				frame_num++;
				waitKey(30);
				//cvWaitKey(0);

			}
			// 補完の際に上書きしているのでフレームを再取得
			cap >> object;
		}

		n = pt1.size();
		printf("n = %d\n", n);
		printf("num_of_obj = %d\n", pt1.size());
		printf("num_of_img = %d\n", pt2.size());
		if (n >= 4) {
			homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
		} else {
			setHomographyReset(&homography);
			printf("frame_num = %d\n", frame_num);
		}

		//cout << Mat(pt1) << endl;

		cv::Mat tmp = homography.clone();
		h_base = h_base * homography;

		warpPerspective(object, transform_image, h_base, Size(PANO_W, PANO_H));
		pano_monographys.push_back(h_base);


		Mat h2 = h_base;
		warpPerspective(white_img, pano_black, h2, Size(PANO_W, PANO_H),
				CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		make_pano(transform_image, transform_image2, mask, pano_black);

		ss << "frame = " << frame_num;
		putText(transform_image, ss.str(), Point(100, 100),
				CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		ss.clear();
		ss.str("");

		// 合成に使ったフレームのホモグラフィ行列とフレーム番号を記録
		//write(cvfs, "frame" ,(int)frame_num);
		//cv::WriteStructContext ws(cvfs, ss.str(), CV_NODE_SEQ);
		ss << "homo_" << frame_num;
		write(cvfs, ss.str(), h_base);
		ss.clear();
		ss.str("");

		ss << "img/frame_" << frame_num << ".jpg";
		imwrite(ss.str(), object);
		ss.clear();
		ss.str("");

		if (f_video) {
			Mat movie(h, w, CV_8UC3, Scalar::all(0));
			Mat tmp;
			Rect roi;
			double fx = (double) w / (double) transform_image.cols;
			double fy = (double) h / (double) transform_image.rows;
			double f = fx < fy ? fx : fy;
			roi.width = (double) transform_image.cols * f;
			roi.height = (double) transform_image.rows * f;
			roi.y = ((double) h - (double) transform_image.rows * f) / 2.0;
			Mat roi_movie(movie, roi);
			resize(transform_image2, roi_movie, cv::Size(0, 0), f, f);
			//tmp.copyTo(roi_movie);
			VideoWriter.write(movie);
		}

		imshow("Object Correspond", transform_image2);
		waitKey(30);
		blur_skip = FRAME_T;

		for (int i = 0; i < FRAME_T; i++) {
			cap >> object;
			frame_num++;
		}
	}

	imshow("Object Correspond", transform_image2);
	cout << "finished making the panorama" << endl;
	waitKey(30);
	imwrite("transform4.jpg", transform_image2);
	imwrite("mask.jpg", mask);

	return 0;
}
