#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

string hear_face_datapath = "/home/gavin/Software/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt.xml";

int main()
{

    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout<< " 不能打开视频" << endl;
        return -1;
    }
    Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    int fps = cap.get(CV_CAP_PROP_FPS);

    CascadeClassifier faceDetect;
    faceDetect.load(hear_face_datapath);
    vector<Rect> faces;
    int count = 1;

    Mat frame, dst;
    namedWindow("camera", CV_WINDOW_AUTOSIZE);

    while(cap.read(frame))
    {

        flip(frame, frame,1);
        faceDetect.detectMultiScale(frame, faces, 1.1, 1, 0, Size(100, 120),Size(380, 400));
        //绘制人脸
        for (int i = 0; i<faces.size();i++)
        {
            if(count % 10 == 0)
            {
                resize(frame(faces[i]), dst, Size(92, 112));
                cout<<"存入图片！"<<endl;
                imwrite(format("/home/gavin/C++progarmm/Detectface/face/face_%d.jpg", count),dst);//写入人脸
            }

            count += 1;
            rectangle(frame,faces[i], Scalar(0, 0, 255), 2, 8, 0);
        }

        imshow("camera", frame);
        char c = waitKey(100);
        if (c == 27)
        {
            break;
        }
        count += 1;
    }
    cap.release();

    return 0;
}
