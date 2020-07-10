#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <iostream>
#include <boost/python.hpp>
#include "pyboostcvconverter.hpp"
#include "simple_op.hpp"
#include "img_process.hpp"

namespace bp = boost::python;

#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
	static void init_ar(){
#endif
	Py_Initialize();

	import_array();
	return NUMPY_IMPORT_ARRAY_RETVAL;
}

BOOST_PYTHON_MODULE(wrapcpppy){

	init_ar();
	//initialize converters
	bp::to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
	pbcvt::matFromNDArrayBoostConverter();

    bp::def("add", simple_op::add_, bp::args("a","b"),"Simple add function");
    bp::def("split_str", simple_op::split_str_, bp::args("src_str", "delimiter"), "Split string function");
    bp::def("max", simple_op::find_max_, bp::args("arr"), "Find index of maximum in array");
    bp::def("rgb2gray", simple_op::rgb2gray, bp::args("input_arr"), "Convert color img to gray img");

    bp::class_<Face_Detector>("Face_Detector", bp::init<>())
        .def("load_cascade", &Face_Detector::load_cascade)
        .def("detect", &Face_Detector::detect);

    bp::class_<Rect_>("Rect", bp::init<>())
        .add_property("x", &Rect_::get_x)
        .add_property("y", &Rect_::get_y)
        .add_property("w", &Rect_::get_w)
        .add_property("h", &Rect_::get_h);
}