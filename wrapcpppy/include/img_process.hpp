#ifndef __IMG_PROCESS_HPP__
#define __IMG_PROCESS_HPP__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/python.hpp>

namespace bp = boost::python;

class Rect_ {
public:
    int x;
    int y;
    int w;
    int h;
public:
    
    int get_x(){return x;}
    int get_y(){return y;}
    int get_h(){return h;}
    int get_w(){return w;}
};

class Face_Detector {
private:
    cv::CascadeClassifier classifer;
public:
    Face_Detector(){};
    ~Face_Detector(){};
    void load_cascade(std::string filename);
    bp::list detect(PyObject *pyimg, float);
};

#endif