#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "prelu.h"
using namespace nvinfer1;

ILayer* convBnNoActivation(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    int num_filters, int k, int s, int p, int g, std::string lname);

ILayer* convBnLeakyRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    int num_filters, int k, int s, int p, int g, std::string lname);

ILayer* convBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    int num_filters, int k, int s, int p, int g, std::string lname);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    std::string lname, float eps);    


ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    std::string lname);

ILayer* addRes3Block(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    int num_filters, std::string block_name);

ILayer* addDconvBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
    int num_filters, std::string block_name);

#endif