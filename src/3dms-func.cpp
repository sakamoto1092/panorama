//
// 3dms-func.c
//

#include <stdio.h>
#include <stdlib.h>
#include<opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
//#include<opencv2/stitching/stitcher.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "3dms-func.h"
#include<assert.h>

using namespace std;
//#define PANO_W 12000
#define PANO_W 6000
#define PANO_H 3000
cv::Mat pano_count = Mat(Size(PANO_W, PANO_H), CV_32S, Scalar::all(1));

//vector<vector<Vec4i> > color_map(PANO_H * PANO_W, vector<Vec4i>()); // パノラマの各画素にどんな色が出てきたかを格納する

// ���ĤΥ��󥵥ǡ�����ɽ�������ߤϻ���Ȧ�, ��, ��, ��-north ��ɽ��
int DispSensorData(SENSOR_DATA sd) {
	fprintf(stderr, "%f %f %f %f ", sd.alpha, sd.beta, sd.gamma, sd.north);
	fprintf(stderr, "%f\n", sd.TT);

	return 0;
}

/* flash.dat ���ɤ߹��� */
int LoadSensorData(const char *oridatafile, SENSOR_DATA *sd_array[]) {
	FILE *fp_ori;

	double t1, t2, t3;

	int i;
	char cdum[256];

	SENSOR_DATA *p;

	if ((fp_ori = fopen(oridatafile, "r")) == NULL) {
		fprintf(stderr, "\nError : Cannot open %s\n\n", oridatafile);
		exit(0);
	}

	//fgets( cdum, 1024, fp); // �����ɤ����Ф�
	//fgets( cdum, 1024, fp); // �����ɤ����Ф�
	//fgets( cdum, 1024, fp); // �����ɤ����Ф�

	p = sd_array[0];
	for (i = 0; i < MAXDATA_3DMS; i++) {
		fscanf(fp_ori, "%lf,%lf,%lf,%lf", &t1, &(p->alpha), &(p->beta),
				&(p->gamma));
		p->north = p->alpha;
		;

		p->TT = t1 / 1000.0; // ���л���(��)�λ���
		//printf("i : %d t1 :%lf \n",i,p->TT);
		p++;
	}
	fclose(fp_ori);

	return 0;
}

// ���󥵥ǡ�������֤��ƻ���Υѥ�᡼���򻻽Ф���
int GetSensorDataForTime(double TT, // �������
		SENSOR_DATA *in_sd_array[], // ���ϥǡ���(����)
		SENSOR_DATA *out_sd // ���ϥǡ���(����ʬ)
		) {
	int i = 0;
	double s;
	SENSOR_DATA *sd0, *sd1;
	int flag = 0;

	sd0 = in_sd_array[0];
	sd1 = sd0 + 1;

	while (sd0->TT < TT) {
		flag = 1;
		sd0++;
		sd1++;
		i++;
		if (i > MAXDATA_3DMS) {
			fprintf(stderr, "OVER MAXDATA\n");
			exit(0);
		}
	}

	if (flag == 0) {
		out_sd->alpha = sd0->alpha;
		out_sd->beta = sd0->beta;
		out_sd->gamma = sd0->gamma;
		out_sd->north = sd0->north;
		out_sd->TT = sd0->TT;
		return 0;
	}

	sd1 = sd0;
	sd0--;

	//fprintf(stderr, "(%f) < [%f] < (%f)\n",
	//        sd0->TT, TT,  sd1->TT);

	s = (TT - sd0->TT) / (sd1->TT - sd0->TT);
	if (sd0->alpha > 180)
		sd0->alpha = sd0->alpha - 360;
	if (sd1->alpha > 180)
		sd1->alpha = sd1->alpha - 360;
	if (sd0->beta > 180)
		sd0->beta = sd0->beta - 360;
	if (sd1->beta > 180)
		sd1->beta = sd1->beta - 360;
	if (sd0->gamma > 180)
		sd0->gamma = sd0->gamma - 360;
	if (sd1->gamma > 180)
		sd1->gamma = sd1->gamma - 360;

	out_sd->alpha = (1 - s) * sd0->alpha + s * sd1->alpha;
	out_sd->beta = (1 - s) * sd0->beta + s * sd1->beta;
	out_sd->gamma = (1 - s) * sd0->gamma + s * sd1->gamma;
	out_sd->north = (1 - s) * sd0->north + s * sd1->north;
	out_sd->TT = (1 - s) * sd0->TT + s * sd1->TT;
	//printf("test\n");
	//DispSensorData(*out_sd);

	return 0;
}

// Tilt :
int SetTiltRotationMatrix(Mat *tiltMatrix, double tilt_deg) {
	double tilt_angle;
	(*tiltMatrix) = Mat::eye(3,3,CV_64FC1);
	tilt_angle = tilt_deg / 180.0 * M_PI;
	(*tiltMatrix).at<double>(1, 1) = cos(tilt_angle);
	(*tiltMatrix).at<double>(1, 2) = -sin(tilt_angle);
	(*tiltMatrix).at<double>(2, 1) = sin(tilt_angle);
	(*tiltMatrix).at<double>(2, 2) = cos(tilt_angle);

	//	cvmSet(tiltMatrix, 1, 1, cos(tilt_angle));
	//	cvmSet(tiltMatrix, 1, 2, -sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 1, sin(tilt_angle));
	//	cvmSet(tiltMatrix, 2, 2, cos(tilt_angle));
	return 0;
}

int SetPanRotationMatrix(Mat *panMatrix, double pan_deg) {
	double pan_angle;
	pan_angle = pan_deg / 180.0 * M_PI;
	(*panMatrix) = Mat::eye(3,3,CV_64FC1);
	(*panMatrix).at<double>(2, 2) = cos(pan_angle);
	(*panMatrix).at<double>(2, 0) = -sin(pan_angle);
	(*panMatrix).at<double>(0, 2) = sin(pan_angle);
	(*panMatrix).at<double>(0, 0) = cos(pan_angle);
	//	cvmSet(panMatrix, 2, 2, cos(pan_angle));
	//	cvmSet(panMatrix, 2, 0, -sin(pan_angle));
	//	cvmSet(panMatrix, 0, 2, sin(pan_angle));
	//	cvmSet(panMatrix, 0, 0, cos(pan_angle));
	return 0;
}

// Roll :
int SetRollRotationMatrix(Mat *rollMatrix, double roll_deg) {
	double roll_angle;
	(*rollMatrix) = Mat::eye(3,3,CV_64FC1);
	roll_angle = roll_deg / 180.0 * M_PI;
	(*rollMatrix).at<double>(0, 0) = cos(roll_angle);
	(*rollMatrix).at<double>(0, 1) = -sin(roll_angle);
	(*rollMatrix).at<double>(1, 0) = sin(roll_angle);
	(*rollMatrix).at<double>(1, 1) = cos(roll_angle);
	//	cvmSet(rollMatrix, 0, 0, cos(roll_angle));
	//	cvmSet(rollMatrix, 0, 1, -sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 0, sin(roll_angle));
	//	cvmSet(rollMatrix, 1, 1, cos(roll_angle));
	return 0;
}

// Pitch :
int SetPitchRotationMatrix(Mat *pitchMatrix, double pitch_deg) {
	double pitch_angle;
	(*pitchMatrix) = Mat::eye(3,3,CV_64FC1);
	pitch_angle = pitch_deg / 180.0 * M_PI;
	(*pitchMatrix).at<double>(1, 1) = cos(pitch_angle);
	(*pitchMatrix).at<double>(1, 2) = -sin(pitch_angle);
	(*pitchMatrix).at<double>(2, 1) = sin(pitch_angle);
	(*pitchMatrix).at<double>(2, 2) = cos(pitch_angle);
	//	cvmSet(pitchMatrix, 1, 1, cos(pitch_angle));
	//	cvmSet(pitchMatrix, 1, 2, -sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 1, sin(pitch_angle));
	//	cvmSet(pitchMatrix, 2, 2, cos(pitch_angle));
	return 0;
}

// Yaw
int SetYawRotationMatrix(Mat *yawMatrix, double yaw_deg) {
	double yaw_angle;
	(*yawMatrix)= Mat::eye(3,3,CV_64FC1);
	yaw_angle = yaw_deg / 180.0 * M_PI;
	(*yawMatrix).at<double>(2, 2) = cos(yaw_angle);
	(*yawMatrix).at<double>(2, 0) = -sin(yaw_angle);
	(*yawMatrix).at<double>(0, 2) = sin(yaw_angle);
	(*yawMatrix).at<double>(0, 0) = cos(yaw_angle);
	//	cvmSet(yawMatrix, 2, 2, cos(yaw_angle));
	//	cvmSet(yawMatrix, 2, 0, -sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 2, sin(yaw_angle));
	//	cvmSet(yawMatrix, 0, 0, cos(yaw_angle));
	return 0;
}

void setHomographyReset(Mat* homography) {
	//cvZero(homography);
	*homography = Mat::zeros(homography->cols,homography->rows,homography->type());
	(*homography).at<double>(0, 0) = 1;
	(*homography).at<double>(1, 1) = 1;
	(*homography).at<double>(2, 2) = 1;
	//cvmSet(homography, 0, 0, 1);
	//cvmSet(homography, 1, 1, 1);
	//cvmSet(homography, 2, 2, 1);
}

double compareSURFDescriptors(const float* d1, const float* d2, double best,
		int length) {
	double total_cost = 0;

	assert(length % 4 == 0);
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

			// 画像のトリミング
			Mat tmp_img(image, cv::Rect(j * 128, i * 72, 128, 72));
			calcHist(&tmp_img, 1, channels, Mat(), hist, dims, histSize, range,
					true, false);

			meanStdDev(hist, mean, dev);
			count.at<float>(i, j) = dev[0];

			if (dev[0] < min_dev)
				min_dev = dev[0];
			if (dev[0] > max_dev)
				max_dev = dev[0];

			if (mean[0] < min_mean)
				min_mean = mean[0];
			if (mean[0] > max_mean)
				max_mean = mean[0];

			sum_mean += mean[0];
			//std::cout << "count : " << mean << std::endl;

			minMaxLoc(hist, NULL, &max_value, NULL, NULL);
			hist *= hist_image[i * 10 + j].rows / max_value;
			bin_w = cvRound((double) 260 / 256);

			for (int k = 0; k < 256; k++)
				rectangle(hist_image[i * 10 + j],
						Point( k * bin_w      , hist_image[i * 10 + j].rows),
						Point( (k + 1) * bin_w, hist_image[i * 10 + j].rows - round(hist.at<float>(k)) ),
						Scalar::all(0), -1, 8, 0);
			// ヒストグラムを計算した部分の画像を切り出して
			// ヒストグラム画像の横に連結
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

	static int cc = 1;
	Vec3f a,b;
	//サイズの一致を確認
	if (src.cols == dst.cols && src.rows == dst.rows) {
		int h = src.rows;
		int w = src.cols;
		int i, j;
#pragma omp parallel for private(j)
		for (i = 0; i < w; i++) {
			for (j = 0; j < h; j++) {
				/*if (mask.at<unsigned char>(j, i) == 255&& roi.at<unsigned char>(j, i) == 255) {
					// 書き込みたい画素位置にすでに画素が書き込まれていた場合
					a = dst.at<Vec3b>(j, i);
					b = src.at<Vec3b>(j, i);
					dst.at<Vec3b>(j, i) = (a * pano_count.at<int>(j, i) + b)
							/ (pano_count.at<int>(j, i) + 1.0);
					pano_count.at<int>(j, i)++;
				}*/
				if (mask.at<unsigned char>(j, i) == 0)
					if (roi.at<unsigned char>(j, i) == 255){
						// 書き込みたい画素位置にまだ書き込まれていない場合
						mask.at<unsigned char>(j, i) = roi.at<unsigned char>(j,	i);
						pano_count.at<int>(j, i)++;
						dst.at<Vec3b>(j, i) = src.at<Vec3b>(j, i);
					}
/*
				if (roi.at<unsigned char>(j, i) == 255){
					// 書き込みたい各画素において，書き込んだ画素を記録

					b = src.at<Vec3b>(j, i);
					size_t tmp_size = color_map[j * PANO_W + i].size();
					if(tmp_size==0)
						color_map[j * PANO_W + i].push_back(Vec4i(b[0],b[1],b[2],1));
					else{
						size_t l ;
						for (l = 0; l < tmp_size; l++) {
							if ((int)(b[0]) == color_map[j * PANO_W + i][l][0] &&
									(int)(b[1]) == color_map[j * PANO_W + i][l][1]&&
									(int)(b[2]) == color_map[j * PANO_W + i][l][2] ) {
								// 同じ画素値だったらカウントアップ
								color_map[j * PANO_W + i][l][3]++;
							}
						}
						if(l == tmp_size)
							color_map[j * PANO_W + i].push_back(Vec4i(b[0],b[1],b[2],1));
					}
				}
*/
			}
		}
	}
	stringstream ss;
	ss.str("");
	ss.clear();
ss <<  "pano_" << cc << ".jpg";
	imwrite(ss.str(), dst);
	cc++;
}

	/* より良い対応点を選択する
	 *
	 * @Param descriptors1 特徴量１
	 * @Param descriptors2 特徴量２
	 * @Param key1         特徴点１
	 * @Param key2         特徴点２
	 * @Param matches      良いマッチングの格納先
	 * @Param pt1          良いマッチングの特徴点座標１
	 * @Param pt2          良いマッチングの特徴点座標２
	 */
void good_matcher(Mat descriptors1, Mat descriptors2, vector<KeyPoint> *key1,
		vector<KeyPoint> *key2, std::vector<cv::DMatch> *matches,
		vector<Point2f> *pt1, vector<Point2f> *pt2) {

	static float gain =1.0;

	int MATCH_TYPE = KNN2_DIST;
	float knn2_conf = 0.01f;
	FlannBasedMatcher matcher;
	//BFMatcher matcher(cv::NORM_L2,true);
	vector<std::vector<cv::DMatch> > matches12, matches21;
	std::vector<cv::DMatch> tmp_matches;
	int knn = 2;
	//BFMatcher matcher(cv::NORM_HAMMING);
	//matcher.match(descriptors1, descriptors2, tmp_matches);

	cout << key1->size() << endl;
	cout << key2->size() << endl;
	cout << descriptors1.size() << endl;
	cout << descriptors2.size() << endl;

	matcher.knnMatch(descriptors1, descriptors2, matches12, knn);
	matcher.knnMatch(descriptors2, descriptors1, matches21, knn);
	cout << "test"<<endl;
	tmp_matches.clear();
	if (MATCH_TYPE == CROSS) {
		// KNN探索で，1->2と2->1が一致するものだけがマッチしたとみなされる
		for (size_t m = 0; m < matches12.size(); m++) {
			bool findCrossCheck = false;
			for (size_t fk = 0; fk < matches12[m].size(); fk++) {
				cv::DMatch forward = matches12[m][fk];
				for (size_t bk = 0; bk < matches21[forward.trainIdx].size();
						bk++) {
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
			if (round((*key1)[tmp_matches[i].queryIdx].class_id)
					== round((*key2)[tmp_matches[i].trainIdx].class_id)) {
				if (tmp_matches[i].distance > 0
						&& tmp_matches[i].distance < (min_dist + 0.1) * 20) {
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
	} else if (MATCH_TYPE == KNN2_DIST) {
		// opencvのdetailクラスのmatcherを参考にした
		set<pair<int, int> > match_set;
		matches->clear();
		pt1->clear();
		pt2->clear();
		for (auto i : matches12) {
			if (i.size() < 2)
				continue;

			const DMatch& m1 = i[0];
			const DMatch& m2 = i[1];
			if (m1.distance < (1.f - knn2_conf) * m2.distance) {
				matches->push_back(m1);
				match_set.insert(make_pair(m1.queryIdx, m1.trainIdx));
				pt1->push_back((*key1)[m1.queryIdx].pt);
				pt2->push_back((*key2)[m1.trainIdx].pt);
			}
		}
		for (auto i : matches21) {
			if (i.size() < 2)
				continue;

			const DMatch& m1 = i[0];
			const DMatch& m2 = i[1];
			if (m1.distance < (1.f - knn2_conf) * m2.distance) {
				if (match_set.find(make_pair(m1.trainIdx, m1.queryIdx))	== match_set.end()) {
					matches->push_back(DMatch(m1.trainIdx, m1.queryIdx, m1.distance));
					pt1->push_back((*key1)[m1.trainIdx].pt);
					pt2->push_back((*key2)[m1.queryIdx].pt);
				}
			}
		}
	}
}
/*
void get_refine_panorama(Mat out, Mat mask) {

	// 出力用のMatを確保
	//out = Mat(Size(PANO_W, PANO_H), CV_8UC3, Scalar::all(0)).clone();
	//vector<vector<Vec3b> > panorama_color(PANO_H * PANO_W, vector<Vec3b>(1));
	vector<Vec4i> pixel_color;
	vector<Vec4i> tmp_pixel_color; 			//一時的な票の格納先
	// パノラマの各画素に描画された色のリストからその画素を，最瀕値で置き換える
	// 最瀕地は，投票された画素値の平均
	int max_count = 0;
	int num = 0;
	// パノラマ画像の各画素は，複数枚の画像が重なっているため画素値の候補が複数ある
	// そこで，候補をクラスタリングして最も投票数が多かった画素値をパノラマ画像の画素値とする
	for (int i = 0; i < PANO_W; i++) {
		for (int j = 0; j < PANO_H; j++) {
			if (mask.at<unsigned char>(j, i) == 255){
				max_count=0;
				for (int l = 0; l < color_map[PANO_W * j + i].size(); l++) {

					Vec3b c;
					c[0] = (uchar)color_map[PANO_W * j + i][l][0];	// 基準の画素
					c[1] = (uchar)color_map[PANO_W * j + i][l][1];	// 基準の画素
					c[2] = (uchar)color_map[PANO_W * j + i][l][2];	// 基準の画素


					num = 0;
					tmp_pixel_color.clear();

					for (int m = 0; m < color_map[PANO_W * j + i].size(); m++) {
						// l番目を基準とし，投票する
						// 類似度は内積により判断
						// 一定以上の類似度ならその色を票として格納し
						// 票数に１を加算

						Vec3f tt;
						Vec3b t;

						tt[0] = pow(abs(color_map[PANO_W * j + i][m][0] - c[0]),2);
						tt[1] = pow(abs(color_map[PANO_W * j + i][m][1] - c[1]),2);
						tt[2] = pow(abs(color_map[PANO_W * j + i][m][2] - c[2]),2);

						t[0] = color_map[PANO_W * j + i][m][0];
						t[1] = color_map[PANO_W * j + i][m][1];
						t[2] = color_map[PANO_W * j + i][m][2];
						float d = tt[0] + tt[1] + tt[2];

						if (sqrt(d) < 30) {
							tmp_pixel_color.push_back(Vec4i(t[0],t[1],t[2],color_map[PANO_W * j + i][m][3]));

							num++;
						}

					}
					// この段階でlを基準とした投票が終わっており，
					// 票の内容はtmp_pixel_colorに
					// 票数はnumに格納されている
					if (num > max_count) {
						// l番目を基準として投票した結果，投票数が過去最大になった時のみ，票の色一覧を更新する
						max_count = num;
						pixel_color = tmp_pixel_color;

					}
				}
				// ここでパノラマの１画素の画素値を決定する
				// 最大投票数が得られた　ある画素を基準とする票群が
				// pixel_colorに格納されている
				// これらの平均をパノラマの画素値とする
				if(color_map[PANO_W * j + i].size()==0){
					// １度も画素値が書き込まれていない画素は飛ばす
					continue;
				}
				int sum_r = 0, sum_g = 0, sum_b = 0;
				int pixel_num =0;
				for (int l = 0; l < pixel_color.size(); l++) {
					sum_r += pixel_color[l][0]*pixel_color[l][3];
					sum_g += pixel_color[l][1]*pixel_color[l][3];
					sum_b += pixel_color[l][2]*pixel_color[l][3];
					pixel_num+=pixel_color[l][3];
				}

				out.at<Vec3b>(j, i) = Vec3b((uchar)((float)sum_r / (float)pixel_num),
						(uchar)((float)sum_g / (float)pixel_num),(uchar)((float)sum_b / (float)pixel_num));
				 sum_r = 0, sum_g = 0, sum_b = 0,pixel_num =0;
			}

		}
	}
}
*/

Mat rotation_estimater(Mat A1, Mat A2, vector<detail::ImageFeatures> features,
		Mat &outA1, Mat &outA2, vector<DMatch>& adopted) {
	Mat homography;

	vector<detail::MatchesInfo> pairwise_matches;

	// MatchesInfo を生成するためにopencvのmatcherを使用
	detail::BestOf2NearestMatcher matcher(false, 0.3f);

	matcher.operator ()(features,pairwise_matches);
	//matcher(features, pairwise_matches);

	matcher.collectGarbage();

	// detail::HomographyBasedEstimatorと
	// detail::BundleAdjusterReprojを使った
	// 回転行列の推定（求めることはできたが合っているかの検証はまだ）
	detail::HomographyBasedEstimator estimator;
	vector<detail::CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);

	for (auto var = cameras.begin(); var < cameras.end(); var++) {
		// BundleBundleAdjusterはCV_32Fを要求しているので変換
		Mat R;
		(*var).R.convertTo(R, CV_32F);
		(*var).R = R.clone();
		//cout << "test : " << (*var).R << endl;
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = new detail::BundleAdjusterReproj();
	adjuster->setConfThresh(1.0f);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	refine_mask(0, 0) = 0; //focal
	refine_mask(0, 1) = 0; // useless
	refine_mask(0, 2) = 0; // ppx
	refine_mask(1, 1) = 0; // aspect
	refine_mask(1, 2) = 0; // ppy
	adjuster->setRefinementMask(refine_mask);

	cameras[0].focal = A1.at<double>(0, 0);
	cameras[0].aspect = A1.at<double>(1, 1) / cameras[0].focal;
	cameras[0].ppx = A1.at<double>(0, 2);
	cameras[0].ppy = A1.at<double>(1, 2);
	//cout << cameras[0].K() << endl;

	cameras[1].focal = A1.at<double>(0, 0);
	cameras[1].aspect = A1.at<double>(1, 1) / cameras[1].focal;
	cameras[1].ppx = A1.at<double>(0, 2);
	cameras[1].ppy = A1.at<double>(1, 2);
	//cout << cameras[1].K() << endl;
	(*adjuster)(features, pairwise_matches, cameras);

	// これは不要のはず
	for (auto var : cameras) {
		Mat R;
		var.R.convertTo(R, CV_64F);
		var.R = R.clone();

		// cout << "test : " << var.R << endl;
	}

	{
		Mat R, K, R0, K0;
		cameras[1].R.convertTo(R, CV_64F);
		cameras[0].R.convertTo(R0, CV_64F);
		Mat(cameras[1].K()).convertTo(K, CV_64F);
		Mat(cameras[0].K()).convertTo(K0, CV_64F);
		homography = K0 * R0.inv() * R * K.inv();
		outA1 = K.clone();
		outA2 = K0.clone();
		for(int i =0; i < pairwise_matches[2].inliers_mask.size();i++){
			if(pairwise_matches[2].inliers_mask[i] == 1)
				adopted.push_back(pairwise_matches[2].matches[i]);
		}
		//cout << cameras[0].K() << endl;
		//cout << cameras[1].K() << endl;
	}

	return homography;
}


