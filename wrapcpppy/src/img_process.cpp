#include "img_process.hpp"
#include "pyboostcvconverter.hpp"

void Face_Detector::load_cascade(std::string filename){
    this->classifer.load(filename);
}

bp::list Face_Detector::detect(PyObject *pyimg, float scale){
    cv::Mat img =pbcvt::fromNDArrayToMat(pyimg);
    cv::Mat gray, smallImg;
    std::vector<cv::Rect> faces;
    cv::cvtColor( img, gray, cv::COLOR_BGR2GRAY ); // Convert to Gray Scale 
    double fx = 1 / scale; 
  
    // Resize the Grayscale Image  
    cv::resize( gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR );  
    cv::equalizeHist( smallImg, smallImg ); 
  
    // Detect faces of different sizes using cascade classifier  
    this->classifer.detectMultiScale( smallImg, faces, 1.1,  
                            2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) ); 

    bp::list l;
    
    for (auto r: faces){
        Rect_ x;
        x.x = r.x;
        x.y = r.y;
        x.w = r.width;
        x.h = r.height;
        l.append(x);
    }
    return l;
}