#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap;
	cap.open(0);
	int cnt = 0;
	cv::Mat frame;
	cv::Mat tempMat;
	cv::Mat refMat;
	cv::Mat resultMat;
	cv::Mat dispMat;

	while (1)
	{
		cap >> frame;
		frame.copyTo(dispMat);
		if (cnt == 0)
		{
			Rect2d r;
			r = selectROI(frame, true);
			tempMat = frame(r);
			tempMat.copyTo(refMat);
			destroyAllWindows();
		}

		
		//ģ��ƥ��
		matchTemplate(frame, refMat, resultMat, TM_SQDIFF);
		//��һ��
		normalize(resultMat, resultMat, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point mathLoc;

		//Ѱ�Ҽ�ֵ
		minMaxLoc(resultMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		//���ƾ��ο�
		rectangle(dispMat, minLoc, Point(minLoc.x + refMat.cols, minLoc.y + refMat.rows),Scalar(255,0,0), 2, 8, 0);
		
		cnt++;
		imshow("ģ��ƥ���", dispMat);
		waitKey(30);


	}
}