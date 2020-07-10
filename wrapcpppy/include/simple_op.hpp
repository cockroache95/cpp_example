#ifndef __SIMPLE_OP_HPP__
#define __SIMPLE_OP_HPP__

#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <opencv2/opencv.hpp>
namespace bp = boost::python;

namespace simple_op {
    int add_(int, int);

    int find_max_(bp::object &);

    bp::list split_str_(std::string, std::string);
    PyObject* rgb2gray(PyObject*);
}

#endif