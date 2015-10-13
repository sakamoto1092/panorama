//
// 3dms-func.h
//

#define MAXDATA_3DMS  5000
using namespace cv;
typedef struct{
    double alpha, beta, gamma, north;
    //double Accx, Accy, Accz;
    //int wAccx, wAccy, wAccz;
    //int wGyrx, wGyry, wGyrz;
    //int wMagx, wMagy, wMagz;
    //double HH,MM,SS;
    double TT;
}SENSOR_DATA;
// CROSS     : クロスマッチング（1->2 & 2->1）
// KNN2_DIST : KNNでの2点で距離が離れているものを採用
enum{CROSS, KNN2_DIST};
// ���ĤΥ��󥵥ǡ�����ɽ�������ߤϻ���Ȧ�, ��, ��, ��-north ��ɽ��
int DispSensorData(SENSOR_DATA sd);

// ���󥵥ǡ����ե����뤫����ɤ߹���
int LoadSensorData(const char *oridatafile ,SENSOR_DATA *sd_array[]);
//int LoadSensorData(char *timedatafile,char *accdatafile,char *magdatafile,char *oridatafile , SENSOR_DATA *sd_array[]);

// ���󥵥ǡ�������֤��ƻ���Υѥ�᡼���򻻽Ф���
int GetSensorDataForTime(double TT, SENSOR_DATA *in_sd_array[], SENSOR_DATA *sd);

// Tilt :
int SetTiltRotationMatrix(Mat *tiltMatrix, double tilt_deg);


int SetPanRotationMatrix(Mat *panMatrix, double pan_deg);


// Roll :
int SetRollRotationMatrix(Mat *rollMatrix, double roll_deg);


// Pitch :
int SetPitchRotationMatrix(Mat *pitchMatrix, double pitch_deg);

// Yaw
int SetYawRotationMatrix(Mat *yawMatrix, double yaw_deg);

void setHomographyReset(Mat* homography);

double compareSURFDescriptors(const float* d1, const float* d2, double best,
		int length);


/*
 * 画像を縦横１０分割し，それぞれの部分画像の
 * ヒストグラムを計算し，画像として返す関数
 *
 *
 * @Param       image ヒストグラムを計算したい画像
 * @Param *hist_image ヒストグラム画像の格納先
 *
 * 分割数は10で固定
 * *hist_imageには予め10個のMat型領域が確保されていると想定
 *
 */
void get_histimage(Mat image, Mat *hist_image);



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
void make_pano(Mat src, Mat dst, Mat mask, Mat roi);

/* より良い対応点を選択する
 *
 * @Param descriptors1 特徴量１
 * @Param descriptors2 特徴量２
 * @Param key1         特徴点１
 * @Param key2         特徴点２
 * @Param matches      良いマッチングの格納先
 * @Param pt1          良いマッチングの特徴点座標1の格納先
 * @Param pt2          良いマッチングの特徴点座標2の格納先
 */
void good_matcher(Mat descriptors1, Mat descriptors2, std::vector<KeyPoint> *key1,
		std::vector<KeyPoint> *key2, std::vector<cv::DMatch> *matches, std::vector<
				Point2f> *pt1, std::vector<Point2f> *pt2);
void get_refine_panorama(Mat out,Mat mask);

/* 特徴点のペアから回転行列と内部パラメータによる
 * 射影変換行列を計算する関数
 * 初期値としてA1とA2をとり，レーベンバーグマーカート法で
 * 計算する．
 * @Param A1       queryの内部パラメータ
 * @Param A2       trainの内部パラメータ
 * @Param features 特徴点と特徴量　size 2
 * @Param outA1    推定されたA1
 * @Param outA2    推定されたA2
 */
Mat rotation_estimater(Mat A1, Mat A2, std::vector<cv::detail::ImageFeatures> features, Mat &outA1, Mat &outA2 ,std::vector<DMatch>& adopted);


/*!
 * パスから拡張子を小文字にして取り出す
 * @param[in] path ファイルパス
 * @return (小文字化した)拡張子
 */
inline std::string GetExtension(const std::string &path) {
    std::string ext;
    size_t pos1 = path.rfind('.');
    if(pos1 != std::string::npos){
        ext = path.substr(pos1+1, path.size()-pos1);
        std::string::iterator itr = ext.begin();
        while(itr != ext.end()){
            *itr = tolower(*itr);
            itr++;
        }
        itr = ext.end()-1;
        while(itr != ext.begin()){    // パスの最後に\0やスペースがあったときの対策
            if(*itr == 0 || *itr == 32){
                ext.erase(itr--);
            }
            else{
                itr--;
            }
        }
    }

    return ext;
}

/*
 * 文字列が画像あるいは動画の拡張子かどうかを判別
 * @param[in] ext 拡張子
 * @return 動画なら1,静止画なら2,その他なら0
 */
inline int checkMediaExtention(const std::string &ext){
	int filetype = 0;

	filetype = ext.compare("avi")  ? 1 :
				ext.compare("mp4")  ? 1 :
				ext.compare("3gp")  ? 1 :
				ext.compare("jpg")  ? 2 :
				ext.compare("jpeg") ? 2:
				ext.compare("jpe")  ? 2 :
				ext.compare("png")  ? 2 : 0;


	return filetype;
}
