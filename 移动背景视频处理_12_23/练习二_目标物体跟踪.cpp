#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//基于HOG的图片相似度计算，以来追踪目标物体
int cellsize = 16;//每个cell的大小
int nAngle = 8;//角度8等分量化

//函数声明
int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSize);
float normL2(float * Hist1, float * Hist2, int siz);

int main()
{
	cv::Mat objMat = imread("C:\\Users\\Lenovo\\Pictures\\数图\\元件.png",0);
	cv::Mat srcMat = imread("C:\\Users\\Lenovo\\Pictures\\数图\\元件大图.png",0);
	cv::Mat dstMat;

	srcMat.copyTo(dstMat);
	int nAngle = 8;
	int blockSize = 16;
	int nX = objMat.cols / blockSize;
	int nY = objMat.rows / blockSize;

	int bins = nX*nY*nAngle;

	float * obj_hist = new float[bins];
	memset(obj_hist, 0, sizeof(float)*bins);

	int reCode = 0;
	//计算输入模板的HOG
	reCode = calcHOG(objMat, obj_hist, nAngle, blockSize);
	if (reCode != 0) {
		return -1;
	}

	Rect2d r1;
	cv::Mat refMat1;
	Point dst_pos;
	int cnt = 0;
	float dist, dist1;
	

	Mat dstanceMat = Mat(srcMat.size(), CV_32FC1, Scalar(0));


	for (int j = 0; j < srcMat.rows - objMat.rows+1; j++)
	{
		for (int i = 0; i < srcMat.cols - objMat.cols+1; i++)
		{
			//初始化内存这句要放在for循环里面！ 否则结果会一直累加！
			float * ref_hist_1 = new float[bins];
			memset(ref_hist_1, 0, sizeof(float)*bins);

			r1 = Rect(i, j, objMat.cols, objMat.rows);
			//选取原图特定区域的矩形 
			refMat1 = srcMat(r1); 

			//计算指定Mat的HOG
			calcHOG(refMat1, ref_hist_1, nAngle, blockSize);
		    dist1 = normL2(obj_hist, ref_hist_1, bins);


			dstanceMat.at<float>(j,i)=dist1;

			if (cnt == 0)
			{
				dist = dist1;
				dst_pos.x = i;  dst_pos.y = j;//当前最相似的矩形的左上角点
				cnt = cnt + 1;
			}
			else if(cnt > 0)
			{
				//当前值与之前最相似的比
				if (dist1 < dist)
				{
					dist = dist1;
					dst_pos.x = i;  dst_pos.y = j;
				}
			}
			delete[] ref_hist_1;

		}
	}

	//遍历完整张图的各个矩形后，得到相似度最高的矩形坐标，并据此绘制矩形
	rectangle(dstMat, dst_pos, Point(dst_pos.x + objMat.cols, dst_pos.y + objMat.rows), Scalar(255, 0, 0), 2, 8, 0);

	delete[] obj_hist;
	//delete[] ref_hist_1;
	destroyAllWindows();

	imshow("追踪后", dstMat);
	waitKey(0);

}


int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSiz)
{
	cv::Mat gx, gy;
	cv::Mat mag, angle;
	//计算梯度
	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);

	//x方向梯度，y方向梯度，梯度，角度，决定输出弧度/角度
	cartToPolar(gx, gy, mag, angle, true);

	//nY表示纵向cell个数，nX表示横向cell个数
	int nX = src.cols / cellsize;
	int nY = src.rows / cellsize;

	int binAngle = 360 / nAngle;

	for (int i = 0; i < nY; i++)
	{
		for (int j = 0; j < nX; j++)
		{
			cv::Mat roiMat;
			cv::Mat roiMag;
			cv::Mat roiAgl;
			Rect roi = Rect(j*cellsize, i*cellsize, cellsize, cellsize);

			roi.x = j*cellsize;
			roi.y = i*cellsize;

			//赋值图像,处理每个cell
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);

			//当前cell第一个元素在数组中的位置
			int head = (i*nX + j)*nAngle;


			for (int n = 0; n < roiMat.rows; n++)
			{
				for (int m = 0; m < roiMat.cols; m++)
				{
					//计算角度在哪个bin，通过int自动取整实现
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					//以像素点的值为权重
					hist[head + pos] += roiMag.at<float>(n, m);
				}
			}


		}
	}
	return 0;
}

float normL2(float * Hist1, float * Hist2, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += (Hist1[i] - Hist2[i])*(Hist1[i] - Hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}

