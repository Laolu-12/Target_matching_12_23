#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//����HOG��ͼƬ���ƶȼ��㣬����׷��Ŀ������
int cellsize = 16;//ÿ��cell�Ĵ�С
int nAngle = 8;//�Ƕ�8�ȷ�����

//��������
int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSize);
float normL2(float * Hist1, float * Hist2, int siz);

int main()
{
	cv::Mat objMat = imread("C:\\Users\\Lenovo\\Pictures\\��ͼ\\Ԫ��.png",0);
	cv::Mat srcMat = imread("C:\\Users\\Lenovo\\Pictures\\��ͼ\\Ԫ����ͼ.png",0);
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
	//��������ģ���HOG
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
			//��ʼ���ڴ����Ҫ����forѭ�����棡 ��������һֱ�ۼӣ�
			float * ref_hist_1 = new float[bins];
			memset(ref_hist_1, 0, sizeof(float)*bins);

			r1 = Rect(i, j, objMat.cols, objMat.rows);
			//ѡȡԭͼ�ض�����ľ��� 
			refMat1 = srcMat(r1); 

			//����ָ��Mat��HOG
			calcHOG(refMat1, ref_hist_1, nAngle, blockSize);
		    dist1 = normL2(obj_hist, ref_hist_1, bins);


			dstanceMat.at<float>(j,i)=dist1;

			if (cnt == 0)
			{
				dist = dist1;
				dst_pos.x = i;  dst_pos.y = j;//��ǰ�����Ƶľ��ε����Ͻǵ�
				cnt = cnt + 1;
			}
			else if(cnt > 0)
			{
				//��ǰֵ��֮ǰ�����Ƶı�
				if (dist1 < dist)
				{
					dist = dist1;
					dst_pos.x = i;  dst_pos.y = j;
				}
			}
			delete[] ref_hist_1;

		}
	}

	//����������ͼ�ĸ������κ󣬵õ����ƶ���ߵľ������꣬���ݴ˻��ƾ���
	rectangle(dstMat, dst_pos, Point(dst_pos.x + objMat.cols, dst_pos.y + objMat.rows), Scalar(255, 0, 0), 2, 8, 0);

	delete[] obj_hist;
	//delete[] ref_hist_1;
	destroyAllWindows();

	imshow("׷�ٺ�", dstMat);
	waitKey(0);

}


int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSiz)
{
	cv::Mat gx, gy;
	cv::Mat mag, angle;
	//�����ݶ�
	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);

	//x�����ݶȣ�y�����ݶȣ��ݶȣ��Ƕȣ������������/�Ƕ�
	cartToPolar(gx, gy, mag, angle, true);

	//nY��ʾ����cell������nX��ʾ����cell����
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

			//��ֵͼ��,����ÿ��cell
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);

			//��ǰcell��һ��Ԫ���������е�λ��
			int head = (i*nX + j)*nAngle;


			for (int n = 0; n < roiMat.rows; n++)
			{
				for (int m = 0; m < roiMat.cols; m++)
				{
					//����Ƕ����ĸ�bin��ͨ��int�Զ�ȡ��ʵ��
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					//�����ص��ֵΪȨ��
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

