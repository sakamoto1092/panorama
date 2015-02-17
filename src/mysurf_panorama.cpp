#include <opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sstream>
#include <string>
#include <iostream>
#include<iomanip>
#include <vector>
#include <fstream>
#include <boost/program_options.hpp>
#include"3dms-func.h"

// パノラマ画像の解像度
//#define PANO_W 12000
#define PANO_W 6000
#define PANO_H 3000
#define F_SCALE

using namespace std;
using namespace cv;

// コマンド引数を解析するためのライブラリ
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using boost::program_options::notify;

int main(int argc, char** argv) {

	VideoWriter VideoWriter; // パノラマ動画
	VideoCapture cap; // ビデオファイル

	// 動画から取得した各種画像の格納先
	Mat buf;
	Mat image; // 投影先の画像
	Mat object; // 変換元の画像
	Mat gray_image; // 変換元の微分画像
	Mat center_img; // センターサークル画像
	//vector<Mat> pano_imgs; // パノラマ背景のキーとなるフレーム

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
	float dist_direction; // センサから計算されるフレーム間の視線方向ベクトルの距離のしきい値
	float focus_scale; // パノラマ画像の焦点距離に掛かる係数(panorama_focus = focus_scale * camera_focus)

	// 各種フラグ
	bool f_comp = false; // 線形補完
	bool f_center = false; // センターサークル中心
	bool f_video = false; // ビデオ書き出し
	bool f_senser = false; // センサ情報の使用・不使用
	bool f_line = false; // 直線検出を利用
	bool f_undist = false; // レンズ歪み補正

	float fps = 20; // 書き出しビデオのfps
	string n_video; // 書き出しビデオファイル名
	string cam_data; // 映像センサファイル名
	string n_center; // センターサークル画像名
	string in_param; // 内部パラメータのxmlファイル名
	string save_dir; // 各ファイルの保存先ディレクトリ

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
	FileStorage cvfs;	//("log.xml", cv::FileStorage::WRITE);

	// 手ブレ検出用各種変数
	int img_num = 0;
	stringstream ss; // 書き出しファイル名
	cv::Mat tmp_img; //
	cv::Mat sobel_img; // エッジ画像格納先

	vector<Mat> hist_image;
	int blur_skip;

	// 各種アルゴリズムによる特徴点検出および特徴量記述
	string algorithm_type;
	Ptr<Feature2D> feature; // 汎用特徴点抽出および特徴量記述クラスへのポインタ
	int hessian;

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

	Mat homography; // 画像対におけるホモグラフィ
	Mat h_base = cv::Mat::eye(3, 3, CV_64FC1); // パノラマ平面へのホモグラフィ
	int n, w, h;

	Mat dist; // カメラの歪み係数

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

	/*
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
	 */
	time_t timer;          // 日付取得用
	struct tm *local;       // 日付取得用

	struct timeval t0, t1; // 処理時間計算用変数 t0:start t1:end

	timer = time(NULL);			// 現在時間の取得
	local = localtime(&timer);  // 地方時間に変換
	gettimeofday(&t0, NULL); // 開始時刻の取得

	try {
		// コマンドラインオプションの定義
		options_description opt("Usage");
		opt.add_options()("cam", value<std::string>(),
				"動画名やセンサファイル名が記述されたファイルの指定")("center", value<std::string>(),
				"センター画像の指定")("start,s", value<int>()->default_value(0),
				"スタートフレームの指定")("end,e", value<int>()->default_value(INT_MAX),
				"終了フレームの指定")("comp", value<bool>()->default_value(false),
				"補完の設定")("inter,i", value<int>()->default_value(9), "取得フレームの間隔")(
				"blur,b", value<int>()->default_value(0), "ブラーによるフレームの破棄の閾値")(
				"video,v", value<std::string>(), "書きだす動画ファイル名の指定")("yaw",
				value<double>()->default_value(0), "初期フレーム投影時のyaw")("fps,f",
				value<int>()->default_value(30), "書きだす動画のフレームレートの指定")("algo,a",
				value<string>()->default_value("SIFT"), "特徴点抽出等のアルゴリズムの指定")(
				"hessian", value<int>()->default_value(20), "SURFのhessianの値")(
				"senser", value<float>(), "センサー情報を使う際の視線方向のしきい値")("line",
				value<bool>()->default_value(false), "直線検出の利用")("undist",
				value<bool>()->default_value(false), "画像のレンズ歪み補正")("outdir,o",
				value<string>()->default_value("./"), "各種ファイルの出力先ディレクトリの指定")
				("f_scale,f",value<float>()->default_value(1.0), "パノラマ平面の焦点距離に掛かる係数")
				(
				"in_param", value<string>(), "内部パラメータ(.xml)ファイル名の指定")("help,h",
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
		if (!vm.count("cam")) {
			cout << "cam_dataファイル名は必ず指定して下さい" << endl;
			return -1;
		}

		if (vm.count("center")) { // センターサークル画像名
			n_center = vm["center"].as<string>();
			f_center = true;
		}

		if (vm.count("video")) { // 書き出し動画ファイル名
			n_video = vm["video"].as<string>();
			f_video = true;
		}

		if (vm.count("in_param")) { // 内部パラメータファイル名
			in_param = vm["in_param"].as<string>();
		} else {
			cout << "内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}

		if (vm.count("undist") && !vm.count("in_param")) {
			cout << "歪み補正をかけるには内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}
		if (vm.count("senser")) {
			cout << "test " << endl;
			f_senser = true; // センサ情報の使用/不使用
			dist_direction = vm["senser"].as<float>();
		} else {
			f_senser = false; // センサ情報の使用/不使用
			dist_direction = 0.0;
		}

		cam_data = vm["cam"].as<string>(); // 映像 センサファイル名
		skip = vm["start"].as<int>(); // 合成開始フレーム番号
		end = vm["end"].as<int>(); // 合成終了フレーム番号
		algorithm_type = vm["algo"].as<string>(); // 特徴点抽出記述アルゴリズム名
		f_comp = vm["comp"].as<bool>(); // 補完の有効無効
		FRAME_T = vm["inter"].as<int>(); // フレーム取得間隔
		blur = vm["blur"].as<int>(); // 手ブレ閾値
		fps = vm["fps"].as<int>(); // 書き出し動画のfps
		yaw = vm["yaw"].as<double>(); // 初期フレーム角度
		hessian = vm["hessian"].as<int>(); // SURFのhessianパラメータ

		focus_scale = vm["f_scale"].as<float>();
		f_undist = vm["undist"].as<bool>(); // レンズ歪み補正
		save_dir = vm["outdir"].as<string>();
		f_line = vm["line"].as<bool>();

	} catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		return -1;
	} catch (...) {
		cerr << "Exception of unknown type!\n";
		return -1;
	}

	FileStorage cvfs_inparam(in_param, CV_STORAGE_READ);
	if (!cvfs_inparam.isOpened()) {
		cout << "cannt open internal parameta file " << in_param << endl;
		return -1;
	}
	Mat a_tmp;
	FileNode node_inparam(cvfs_inparam.fs, NULL);
	read(node_inparam["intrinsic"], a_tmp);
	if (f_undist) {
		read(node_inparam["distortion"], dist);
	}
	Mat inv_a1 = a_tmp.inv();
	A1Matrix = a_tmp.clone();

	// 画像をリサイズするのに合わせて，内部パラメータの画像中心座標を半分にする
	//A1Matrix.at<double>(0,2) /= 2.0;
	//A1Matrix.at<double>(1,2) /= 2.0;

	A2Matrix.at<double>(0, 0) = A1Matrix.at<double>(0, 0)*focus_scale;
	A2Matrix.at<double>(1, 1) = A1Matrix.at<double>(1, 1)*focus_scale;
	A2Matrix.at<double>(0, 2) = PANO_W / 2;
	A2Matrix.at<double>(1, 2) = PANO_H / 2;

	cout << "<A1>" << endl << A1Matrix << endl;
	cout << "<A2>" << endl << A2Matrix << endl;

	// 映像　センサファイル名を取得
	if ((SensorFP = fopen(cam_data.c_str(), "r")) == NULL) {
		cerr << "cant open " << cam_data << endl;
		return -1;
	}
	fscanf(SensorFP, "%s", imagefileName);
	fscanf(SensorFP, "%s", timefileName);
	fscanf(SensorFP, "%s", sensorfileName);
	fclose(SensorFP);

	ss << save_dir << "log.xml";
	cvfs.open(ss.str(), cv::FileStorage::WRITE);
	ss.str("");
	ss.clear();

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
	cout << "out put video fps = " << fps << endl;

	cout << "start frame : " << cap.get(CV_CAP_PROP_POS_FRAMES) << endl;
	// 合成開始フレームまでスキップ
	if (skip > 0) {
		cout << "skip frame to : " << skip << endl;
		for (int i = 0; i < skip; i++)
			cap >> image;
		frame_num = skip - 1; // 現在のフレーム位置を設定
	} else {
		frame_num = 0;
	}

	// 特徴点抽出記述器を生成
	feature = Feature2D::create(algorithm_type);
	// 指定されたアルゴリズムが存在しなければSURFを使用する
	if (feature == NULL) {
		cout << algorithm_type << " algorithm was not found " << endl;
		feature = Feature2D::create("SURF");
		cout << "using SURF algorithm" << endl;
	}
	// SURFのときのみパラメータを設定（SURFしかやったことないんで...）
	if (algorithm_type.compare("SURF") == 0) {
		feature->set("extended", 1);
		feature->set("hessianThreshold", hessian);
		feature->set("nOctaveLayers", 8);
		feature->set("nOctaves", 6);
		feature->set("upright", 0);
	} else if (algorithm_type.compare("SIFT") == 0) {
		//feature->set("nfeatures", 0);
		feature->set("nOctaveLayers", 10);
		feature->set("contrastThreshold", 0.02);
		feature->set("edgeThreshold", 20);
		feature->set("sigma", 2.0);

	}

	// 各種回転をパノラマ平面に適用
	SetRollRotationMatrix(&rollMatrix, (double) roll);
	SetPitchRotationMatrix(&pitchMatrix, (double) pitch);
	SetYawRotationMatrix(&yawMatrix, (double) yaw);

	// 最終的なパノラマ平面へのホモグラフィ行列を計算
	h_base = A2Matrix * rollMatrix * pitchMatrix * yawMatrix * A1Matrix.inv();

	cout << "<h_base>" << endl << h_base << endl;

	/*logging*/
	ss << save_dir << "composition_log.txt";
	ofstream log(ss.str().c_str());
	ss.str("");
	ss.clear();

	vector<String> v_log_str;
	string log_str;

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
	log << "<use senser>" << f_senser << endl;
	log << "<deblur>" << endl << blur << endl;
	log << "<start frame>" << endl << skip << endl;
	log << "<end frame>" << endl << end << endl;
	log << "<inter>" << endl << FRAME_T << endl;
	log << "<Algorithm> " << endl << algorithm_type << endl;
	log << "<Algorithm Param>" << endl;
	for (int ii = 0; ii < v_log_str.size(); ii++)
		log << v_log_str[ii] << " " << feature->getDouble(v_log_str[ii])
				<< endl;
	log << "<start_time>" << endl << local->tm_year + 1900 << "/"
			<< local->tm_mon + 1 << "/" << local->tm_mday << " "
			<< local->tm_hour << ":" << local->tm_min << ":" << local->tm_sec
			<< " " << local->tm_isdst << endl;
	;;
	/*end of logging*/

	// センサ情報の領域確保
	SENSOR_DATA *sensor;
	double s_time;
	if (f_senser)
		sensor = (SENSOR_DATA *) malloc(sizeof(SENSOR_DATA) * 5000);

	// 最初のフレームを取得（センターサークル画像に差し替え）
	// 最初のフレームを動画から取得する場合（else以降）では，同時にセンサ情報を取得
	if (f_center) {
		image = center_img.clone();
	} else {

		// 撮影時間とセンサ情報（ORIとTIME）を取得
		string time_buf;
		ifstream ifs_time(timefileName);

		if (f_senser && !ifs_time.is_open()) {
			cerr << "cannnot open TIME file : " << timefileName << endl;
			return -1;
		}

		if (f_senser) {

			// 三回読み飛ばして秒数を文字で取得
			ifs_time >> time_buf;
			ifs_time >> time_buf;
			ifs_time >> time_buf;
			ifs_time >> s_time; // msec

			cout << "senser_time : " << s_time / 1000.0 << "[sec]" << endl;

			// 対象フレームのセンサデータを一括読み込み
			LoadSensorData(sensorfileName, &sensor);
		}

		if (f_senser) {
			// 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
			double frame_msec = cap.get(CV_CAP_PROP_POS_MSEC) + s_time;
			cout << "object frame time :" << cap.get(CV_CAP_PROP_POS_MSEC)
					<< endl;
			GetSensorDataForTime(frame_msec / 1000.0, &sensor, &obj_sd);
		}

		cap >> buf;
		image = buf.clone();
		// S-09がツインカメラでの映像のため半分トリミングして横を倍にして利用
		//resize(Mat(image,Rect(0,0,image.cols/2,image.rows)),image,Size(),1.0,0.5);

		//resize(image,image,Size(image.cols/2,image.rows/2)); // convert image to half image
		frame_num++;
		//undistort(buf, object, A1Matrix, dist);
	}

	cout << cap.get(CV_CAP_PROP_FPS) << "[frame / sec]" << endl;

	if (f_undist && !in_param.empty()) {
		//dist = (Mat_<double> (1, 5) << 4.6607295014012916e-002, 4.1437801936723750e-001, -7.4809715282343212e-003, -2.9591314503800725e-003, -2.1417165056372101e+000);
		Mat dist_src = image.clone();
		undistort(dist_src, image, A1Matrix, dist);
		cout << "dist param : " << dist << endl;
	}

	if (image.empty()) {
		cout << "cannot get frame from video : " << endl;
		return -1;
	}
	w = image.cols;
	h = image.rows;

	cvtColor(image, gray_image, CV_RGB2GRAY);

	// マスクイメージ関係を生成
	//  パノラマ平面へ射影する際のマスク処理

	mask = Mat(PANO_H, PANO_W, CV_8U, cv::Scalar(0));
	pano_black = Mat(PANO_H, PANO_W, CV_8U, cv::Scalar(0));
	white_img = Mat(image.rows, image.cols, CV_8U, cv::Scalar(255));
	transform_image2 = cv::Mat::zeros(Size(PANO_W, PANO_H), CV_8UC3);

	// 初期フレームの特徴点の検出と特徴量の記述
	//PyramidAdaptedFeatureDetector g_feature_detector(new SiftFeatureDetector,5);
	GridAdaptedFeatureDetector g_feature_detector(new SiftFeatureDetector,
			1000000, 2, 2);

	// gridadaptedfeaturedetectorで特徴点の検出
	//g_feature_detector.detect(gray_image, imageKeypoints);

	// 検出した特徴点の特徴量を記述（sift or surf）　特徴点検出は行わない (trueの引数)
	feature->operator ()(gray_image, Mat(), imageKeypoints, imageDescriptors,
			false);

	/*
	 //AKAZEで検出と記述を行う
	 {
	 Mat img32;
	 gray_image.convertTo(img32,CV_32F,1.0/255.0);
	 akaze.Create_Nonlinear_Scale_Space(img32);
	 akaze.Feature_Detection(imageKeypoints);
	 akaze.Compute_Descriptors(imageKeypoints,imageDescriptors);
	 }
	 */
	// 先頭フレームをパノラマ平面へ投影
	warpPerspective(image, transform_image, h_base, Size(PANO_W, PANO_H));
	warpPerspective(white_img, pano_black, h_base, Size(PANO_W, PANO_H));

	make_pano(transform_image, transform_image2, mask, pano_black);
	ss.str("");
	ss.clear();
	ss << save_dir << "pano_src_" << frame_num << ".jpg";
	imwrite(ss.str(), image);
	ss.str("");
	ss.clear();

	// 合成に使ったホモグラフィ行列を記録
	ss << "homo_" << frame_num;
	write(cvfs, ss.str(), h_base);
	ss.clear();
	ss.str("");

	ss << "keypoints_" << frame_num;
	write(cvfs, ss.str(), imageKeypoints);
	ss.clear();
	ss.str("");

	ss << "descriptors_" << frame_num;
	write(cvfs, ss.str(), imageDescriptors);
	ss.clear();
	ss.str("");

	// パノラマ動画ファイルを作成
	if (f_video) {
		ss.str("");
		ss.clear();
		ss << save_dir << n_video;
		cout << ss.str() << endl;
		VideoWriter.open(ss.str(), CV_FOURCC('D', 'I', 'V', 'X'), (int) fps,
				Size(w, h), 1);
	}
	// フレームを飛ばす
	if (!f_center) {
		if (f_senser) {
			// 最初に投影した動画フレームのセンサとフレーム番号，ホモグラフィ行列を記録
			pano_sds.push_back(obj_sd);
			vec_n_pano_frames.push_back(frame_num);
			cout << "pushback background image src frame : " << frame_num
					<< endl;
			pano_monographys.push_back(h_base.clone());
			ss << save_dir << "pano_src_" << frame_num << ".jpg";
			imwrite(ss.str(), image);
			ss.str("");
			ss.clear();
		}
		for (int i = 0; i < FRAME_T; i++) {
			cap >> buf;
			object = buf.clone();
			frame_num++;
		}
		//undistort(buf, object, A1Matrix, dist);

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

	// 中間生成物等の確認用ウインドウの設定
	namedWindow("Object Correspond", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO); // 合成途中のパノラマ画像
	namedWindow("detected lines", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO); // 確率的ハフ変換で検出された直線からなるマスク画像
	namedWindow("object image ", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO); // 取り出した動画フレーム

	cv::Mat tmp(3, 3, CV_64FC1, Scalar(0));
	Mat x;
	Mat y;
	Mat yt;
	Mat q;
	Mat tmp_pt1;
	Mat tmp_pt2;

	Mat roi_x;
	Mat roi_y;
	Mat svd_w, svd_u, svd_vt;
	Mat svd_v, svd_ut;
	Mat est_R, H;

	// 指定したフレーム番号まで処理を繰り返す(あるいは，動画の最終フレームまで繰り返す)
	while (frame_num < end && frame_num < FRAME_MAX + FRAME_T + 1) {

		// ブラーがかかっていないフレーム画像を選択
		//（ブラーによる判定をしない場合はここではループしない）
		while (f_blur) {
			obj_frame_msec = cap.get(CV_CAP_PROP_POS_MSEC);
			cout << "object frame : " << obj_frame_msec << "[msec]" << endl;
			cout << cap.get(CV_CAP_PROP_FPS) << "[frame / sec]" << endl;
			cap >> buf;

			if (buf.empty())
				break;

			if (f_undist)
				undistort(buf, object, A1Matrix, dist);
			else
				object = buf.clone();
			frame_num++;
			//resize(object,object,Size(object.cols/2,object.rows/2));

			printf("\nframe=%d\n", frame_num);
			cvtColor(object, gray_image, CV_RGB2GRAY);
			//cv::Laplacian(gray_image, tmp_img, CV_16S, 3);
			//Canny(object, sobel_img, 50, 200);
			cv::Sobel(gray_image, tmp_img, CV_32F, 1, 1);
			cv::convertScaleAbs(tmp_img, sobel_img, 1, 0);

			cv::Mat_<float>::iterator it = sobel_img.begin<float>();
			long count = 0;
			for (; it != sobel_img.end<float>(); ++it) {
				if ((*it) > 150)
					count++;
			}
			cout << "edge : " << count << endl;
			if (count > blur)
				f_blur = false;
			count = 0;
		}

		f_blur = true;

		/*
		 cap >> buf;
		 if (f_undist)
		 undistort(buf, object, A1Matrix, dist);
		 else
		 object = buf.clone();
		 frame_num++;
		 */
		// 動画のフレームが取得出来なかったらbreak
		if (object.empty())
			break;

		// S-09がツインカメラでの映像のため半分トリミングして横を倍にして利用
		//	resize(Mat(object,Rect(0,0,object.cols/2,object.rows)),object,Size(),1.0,0.5);
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

			// debug 合成に使う元画像と，微分画像を保存
			// if(count < blur){
			/*			ss << "img/img_" << frame_num << ".jpg";
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
			 * */

		}

		//undistort(dist_src, object, A1Matrix.inv(), dist);
		imshow("object image ", object);
		waitKey(30);

		// 確率的ハフ変換で線を検出し，特徴点抽出のマスクに利用する
		if (f_line) {
			Mat hough_src, hough_dst; // ハフ変換の入力と，検出した線の出力先
			cvtColor(object, gray_image, CV_RGB2GRAY);
			Canny(gray_image, hough_src, 30, 50, 3);

			imshow("detected lines", hough_src);
			//			waitKey(0);

			hough_dst = Mat(hough_src.size(), CV_8U, Scalar::all(0));

			vector<Vec4i> lines;
			HoughLinesP(hough_src, lines, 1, CV_PI / 180, 80, 100, 200);

			for (size_t i = 0; i < lines.size(); i++) {
				Vec4i l = lines[i];
				line(hough_dst, Point(l[0], l[1]), Point(l[2], l[3]),
						Scalar(255, 255, 255), 3, CV_AA);
			}

			// 検出した直線を膨張させて，マスク画像を作成
			Mat t = hough_dst;
			dilate(t, hough_dst, cv::Mat(), cv::Point(-1, -1), 5);

			imshow("detected lines", hough_dst);
			waitKey(30);

			cvtColor(object, gray_image, CV_RGB2GRAY);
			g_feature_detector.detect(gray_image, objectKeypoints);
			feature->operator ()(gray_image, hough_dst, objectKeypoints,
					objectDescriptors, true);
		} else {
			//	cvtColor(object, gray_image, CV_RGB2GRAY);
			//	feature->operator ()(gray_image, Mat(), objectKeypoints,
			//			objectDescriptors);
		}
		// TODO : 取り出したフレーム(object)に近いフレーム(image)を
		//         センサ情報から探して特徴点抽出をしてマッチングする

		Mat near_homography = h_base.clone();
		bool f_detect_near = false;

		if (f_senser) {
			double dist, min = 1.0;
			min = 1.0;
			f_detect_near = false;
			// 取り出したフレームセンサ情報と秒数を取得
			double frame_msec = obj_frame_msec + s_time;
			GetSensorDataForTime(frame_msec / 1000.0, &sensor, &obj_sd);
			//pano_sds.push_back(obj_sd);
			//vec_n_pano_frames.push_back(frame_num);

			// 近い角度のフレームを計算する
			Mat vec1(3, 1, CV_64F), vec2(3, 1, CV_64F);
			Mat near_vec(3, 1, CV_64F);
			long near_frame;

			SetYawRotationMatrix(&yawMatrix, obj_sd.alpha);
			SetPitchRotationMatrix(&pitchMatrix, obj_sd.beta);
			SetRollRotationMatrix(&rollMatrix, obj_sd.gamma);
			//warpPerspective(Point3d(1, 0, 0), vec1, yawMatrix * pitchMatrix* rollMatrix);
			vec1 = yawMatrix * pitchMatrix * rollMatrix
					* (cv::Mat_<double>(3, 1) << 1, 0, 0);

			for (int i = 0; i < pano_sds.size(); i++) {
				SetYawRotationMatrix(&yawMatrix, pano_sds[i].alpha);
				SetPitchRotationMatrix(&pitchMatrix, pano_sds[i].beta);
				SetRollRotationMatrix(&rollMatrix, pano_sds[i].gamma);
				vec2 = yawMatrix * pitchMatrix * rollMatrix
						* (cv::Mat_<double>(3, 1) << 1, 0, 0);

				dist = sqrt(
						pow(vec1.at<double>(0, 0) - vec2.at<double>(0, 0), 2)
								+ pow(
										vec1.at<double>(0, 1)
												- vec2.at<double>(0, 1), 2)
								+ pow(
										vec1.at<double>(0, 2)
												- vec2.at<double>(0, 2), 2));
				cout << "dist : " << dist << endl;
				// 近いものがあったらnear_sdを更新
				if (dist < dist_direction && dist < min) {
					min = dist;
					near_sd = pano_sds[i];
					cout << "detect near frame : " << vec_n_pano_frames[i]
							<< endl;

					f_detect_near = true;
					near_frame = vec_n_pano_frames[i];
					near_homography = pano_monographys[i].clone();
				}
				i++;
			}

			// 近いフレームがあったならそのフレームとマッチング
			// 見つからなかったら，そのフレームのセンサ情報とホモグラフィー行列などを保存
			long tmp_frame_num;
			if (f_detect_near) {
//				if (!f_comp)
//					continue;
				tmp_frame_num = cap.get(CV_CAP_PROP_POS_FRAMES); // 現在のフレーム位置を退避
				cap.set(CV_CAP_PROP_POS_FRAMES, near_frame);
				//cout << "detect near frame : " << near_sd.TT << " [sec]" << endl;
				//cout << cap.get(CV_CAP_PROP_POS_FRAMES) << " [frame]" << endl;
				cap >> buf;
				if (f_undist)
					undistort(buf, image, A1Matrix, dist);
				else
					image = buf.clone();
				cap.set(CV_CAP_PROP_POS_FRAMES, tmp_frame_num); // フレーム位置を復元

				if (image.empty()) {
					cout << "cannnot get near frame" << endl;
					return -1;
				}

				// 近いフレームの特徴点抽出を再度実行
				cvtColor(image, gray_image, CV_RGB2GRAY);
				g_feature_detector.detect(gray_image, objectKeypoints);
				feature->operator ()(gray_image, Mat(), imageKeypoints,
						imageDescriptors, true);

				pano_sds.push_back(obj_sd);
				vec_n_pano_frames.push_back(frame_num);
			} else {
				pano_sds.push_back(obj_sd);
				vec_n_pano_frames.push_back(frame_num);
				cout << "pushback background image src frame : " << frame_num
						<< endl;
				ss << save_dir << "pano_src_" << frame_num << ".jpg";
				imwrite(ss.str(), object);
				ss.str("");
				ss.clear();

			}

		}

		// ここで，近いフレームが見つかっている場合はそのフレームがimageに
		// 見つかっていない場合は，FRAME_T + a 前のフレームがimageに格納されているはず
		cvtColor(object, gray_image, CV_RGB2GRAY);
		//g_feature_detector.detect(gray_image, objectKeypoints);
		feature->operator ()(gray_image, Mat(), objectKeypoints,
				objectDescriptors, false);

		/*
		 {
		 Mat img32;
		 gray_image.convertTo(img32,CV_32F,1.0/255.0);
		 akaze.Create_Nonlinear_Scale_Space(img32);
		 akaze.Feature_Detection(objectKeypoints);
		 akaze.Compute_Descriptors(objectKeypoints,objectDescriptors);
		 }
		 */
		good_matcher(objectDescriptors, imageDescriptors, &objectKeypoints,
				&imageKeypoints, &matches, &pt1, &pt2);

		cout << "selected good_matches : " << pt1.size() << endl;
		// マッチング結果をリサイズして表示
		Mat result, r_result;
		drawMatches(object, objectKeypoints, image, imageKeypoints, matches,
				result);

		namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);

		imshow("matches", result);
		waitKey(30);

		/*
		 // 補完作業
		 if (f_comp && blur_skip != 0) {
		 cout << "start comp" << endl;
		 vector<Point2d> dist;
		 vector<Point2d> est_pt1 = pt1, est_pt2;
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
		 cap >> buf;
		 if(f_undist)
		 undistort(buf, object, A1Matrix, dist);
		 else
		 object = buf.clone();

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
		 cap >> buf;
		 if(f_undist)
		 undistort(buf, object, A1Matrix, dist);
		 else
		 object = buf.clone();
		 }
		 */
		n = pt1.size();
		printf("n = %d\n", n);
		printf("num_of_obj = %d\n", pt1.size());
		printf("num_of_img = %d\n", pt2.size());
		vector<uchar> state;
		if (n >= 4) {
			homography = findHomography(Mat(pt1), Mat(pt2), state, RANSAC, 5.0);

		} else {
			setHomographyReset(&homography);
			printf("frame_num = %d\n", frame_num);
		}

		vector<detail::ImageFeatures> features;
		features.clear();
		detail::ImageFeatures aa;
		aa.descriptors = imageDescriptors.clone();
		aa.img_idx = 0;
		aa.img_size = gray_image.size();
		aa.keypoints = imageKeypoints;
		features.push_back(aa);

		aa.descriptors = objectDescriptors.clone();
		aa.img_idx = 1;
		aa.img_size = gray_image.size();
		aa.keypoints = objectKeypoints;
		features.push_back(aa);

		Mat estA1 = A1Matrix.clone();
		Mat estA2 = A2Matrix.clone();
		homography = rotation_estimater(A1Matrix, A2Matrix, features, estA1,
				estA2);

		if (near_homography.empty()) {
			cout << "near_homography is empty." << endl;
			cout << "num_sd : " << pano_sds.size() << endl;
			;
			cout << "num_pano : " << vec_n_pano_frames.size() << endl;
			;
			cout << "num_homo :" << pano_monographys.size() << endl;
			;
			return -1;
		}
		vector<DMatch> current_adopt;
		for (int i = 0; i < matches.size(); i++)
			if (state[i] == 1)
				current_adopt.push_back(matches[i]);
		drawMatches(object, objectKeypoints, image, imageKeypoints,
				current_adopt, result, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), 2);
		ss.clear();
		ss.str("");
		ss << "adopt_matches_" << frame_num << ".jpg";
		imwrite(ss.str(), result);
		ss.clear();
		ss.str("");

		warpPerspective(object, transform_image, near_homography * homography,
				Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		warpPerspective(white_img, pano_black, near_homography * homography,
				Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		imshow("detected lines", pano_black);
		make_pano(transform_image, transform_image2, mask, pano_black);
		//tmp = homography.clone();

		h_base = Mat(near_homography * homography).clone();
		if (f_senser)
			pano_monographys.push_back(h_base.clone());
		cout << pano_monographys.size() << endl;

		imageKeypoints = objectKeypoints;
		objectDescriptors.copyTo(imageDescriptors);
		image = object.clone();

		ss << "frame = " << frame_num;
		putText(transform_image, ss.str(), Point(100, 100),
				CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		ss.clear();
		ss.str("");

		// 合成に使ったフレームのホモグラフィ行列とフレーム番号を記録

		ss << "homo_" << frame_num;
		write(cvfs, ss.str(), h_base);
		ss.clear();
		ss.str("");

		ss << "keypoints_" << frame_num;
		write(cvfs, ss.str(), objectKeypoints);
		ss.clear();
		ss.str("");

		ss << "descriptors_" << frame_num;
		write(cvfs, ss.str(), objectDescriptors);
		ss.clear();
		ss.str("");

		ss << save_dir << "pano_src_" << frame_num << ".jpg";
		imwrite(ss.str(), object);
		ss.str("");
		ss.clear();

		// 動画に書き出し
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
			//resize(transform_image, roi_movie, cv::Size(0, 0), f, f);

			waitKey(30);
			//tmp.copyTo(roi_movie);
			VideoWriter.write(movie);
		}

		imshow("Object Correspond", transform_image2);
		waitKey(30);
		ss << save_dir << "transform4.jpg";
		imwrite(ss.str(), transform_image2);
		ss.str("");
		ss.clear();

		blur_skip = FRAME_T;

		// フレーム飛ばし
		for (int i = 0; i < FRAME_T; i++) {
			cap >> buf;
			frame_num++;
		}

	}

	// 最終的なパノラマ画像の表示と保存（後の処理に使うマスク画像も保存）

	imshow("Object Correspond", transform_image2);
	cout << "finished making the panorama" << endl;
	waitKey(30);

	ss << save_dir << "transform4.jpg";
	imwrite(ss.str(), transform_image2);
	ss.str("");
	ss.clear();

	ss << save_dir << "mask.jpg";
	imwrite(ss.str(), mask);
	ss.str("");
	ss.clear();
/*
	 Mat fine_pano(cv::Mat::zeros(Size(PANO_W, PANO_H), CV_8UC3));
	 get_refine_panorama(fine_pano, mask);
	 cout << "refine panorama" << endl;
	 if (fine_pano.empty()) {
	 cout << "fine pano is empty" << endl;
	 exit(1);
	 }
	 imwrite("fine_panorama.jpg", fine_pano);
*/
	gettimeofday(&t1, NULL); // 開始時刻の取得
	log << "<processing_start>" << "\n" << t0.tv_sec << "." << t0.tv_usec
			<< "[sec]" << endl;
	log << "<processing_end>" << "\n" << t1.tv_sec << "." << t1.tv_usec
			<< "[sec]" << endl;
	if (t1.tv_usec < t0.tv_usec)
		log << "<processing_time>" << "\n" << t1.tv_sec - t0.tv_sec - 1.0 << "."
				<< 1000000 + t1.tv_usec - t0.tv_usec << "[sec]" << endl;
	else
		log << "<processing_time>" << "\n" << t1.tv_sec - t0.tv_sec << "."
				<< t1.tv_usec - t0.tv_usec << "[sec]" << endl;
	cout << "finished" << endl;
	return 0;
}

