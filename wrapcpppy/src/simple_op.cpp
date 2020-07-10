#include "simple_op.hpp"
#include <boost/python/stl_iterator.hpp>
#include "pyboostcvconverter.hpp"

int simple_op::add_(int a, int b)
{
    return a + b;
}

int simple_op::find_max_( bp::object &pyarr)
{
    const std::vector<int> arr = std::vector<int>( boost::python::stl_input_iterator<int>( pyarr ),
                             boost::python::stl_input_iterator<int>( ) );
    int len = arr.size();
    if (len <= 0)
    {
        throw("Length of array is invalid!");
    }
    int max_tmp = arr[0];
    int max_idx = 0;
    for (int i = 1; i < len; i++)
    {
        if (arr[i] > max_tmp)
        {
            max_tmp = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bp::list simple_op::split_str_(std::string src_string, std::string delimiter)
{
    size_t pos = 0;
    std::string token;
    bp::list l;
    while ((pos = src_string.find(delimiter)) != std::string::npos)
    {
        token = src_string.substr(0, pos);
        l.append(token);
        src_string.erase(0, pos + delimiter.length());
    }
    
    l.append(src_string);
    return l;
}

PyObject* simple_op::rgb2gray(PyObject *pymat){
    cv::Mat input =pbcvt::fromNDArrayToMat(pymat);
    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_RGB2GRAY);
    PyObject *res = pbcvt::fromMatToNDArray(output);
    return res;
}