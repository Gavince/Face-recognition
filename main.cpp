#include<opencv2/face/facerec.hpp>
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;
string haar_face_datapath = "/home/gavin/Software/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt2.xml";
int main()
{
    //1.打开人脸数据集
    string filename = string("/home/gavin/GProject/Face-recognition/mylist.csv");

    ifstream file(filename.c_str(), ifstream::in);
    if(!file)
    {
        cout<< "Can't open!" <<endl;
        return -1;
    }
    string line, path, classlabel;
    vector<Mat> images;
    vector<int> labels;
    char separtor = ';';

    while(getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separtor);
        getline(liness, classlabel);
        if(!path.empty()&&!classlabel.empty())
        {
            images.push_back(imread(path, 0));//读取灰度图片
            labels.push_back(atoi(classlabel.c_str()));
        }

    }
    if(images.size()<1 || labels.size()<1)
    {
        cout<< "路径错误！" << endl;
        return -1;
    }

    int heigh = images[0].rows;
    int width = images[0].cols;

    //2.创建测试样本
    cout<< "开始训练人脸数据集"<<endl;
    Mat testsample = images[images.size()-1];
    int testlabel = labels[labels.size()-1];
    images.pop_back();
    labels.pop_back();

    //3.训练数据
    Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    CascadeClassifier faceDetector;
    faceDetector.load(haar_face_datapath);

    //4.打开摄像头测试人脸
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
        cout<< "Can't open "<< endl;
        return -1;
	}

    Mat frame;
    //namedWindow("Facerecongnize",CV_WINDOW_AUTOSIZE);
    vector<Rect> faces;
    Mat dst;
    Mat gray;
    while(cap.read(frame))
    {
        flip(frame, frame, 1);
        cvtColor(frame, gray,COLOR_BGR2GRAY);
        equalizeHist(gray,gray);

        // 6. 人脸检测
        faceDetector.detectMultiScale(gray, faces, 1.1, 4, 0, Size(80, 100), Size(380, 400));
        for(int i = 0;i < faces.size(); i++)
        {
            Mat roi = frame(faces[i]);
            cvtColor(roi, dst, COLOR_BGR2GRAY);

            resize(dst, testsample, testsample.size());
            int label = model->predict(testsample);
            cout<< "label:" << label << endl;
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
            switch(label)//查询身份信息
            {
                case 13 : putText(frame, format("%s",("zhangwanyu")), faces[i].tl(),FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 2, 255), 2, 8);break;
                case 14 : putText(frame, format("%s",("wuqiyou")), faces[i].tl(),FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 2, 255), 2, 8);break;
                default : putText(frame, format("%s",("Unknow")), faces[i].tl(),FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 2, 255), 2, 8);
            }
        }

        imshow("Face_recongnize", frame);
        char c = waitKey(10);
        if(c == 27)
        {
            break;
        }
    }
    waitKey(0);
    return 0;
}


