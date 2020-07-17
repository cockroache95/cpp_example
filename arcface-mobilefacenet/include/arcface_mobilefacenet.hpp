#ifndef __ARCFACE_MOBILEFACENET_HPP__
#define __ARCFACE_MOBILEFACENET_HPP__

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

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1

using namespace nvinfer1;

static const int INPUT_H = 112;
static const int INPUT_W = 112;
static const int OUTPUT_SIZE = 128;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

REGISTER_TENSORRT_PLUGIN(PReluPluginCreator);


std::map<std::string, Weights> loadWeights(const std::string file);

ICudaEngine* createEngine(unsigned int maxBatchSize, std::string weight_path,  IBuilder* builder, IBuilderConfig* config, DataType dt);
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
void doInference(IExecutionContext& context, float* input, float* output, int batchSize);


#endif