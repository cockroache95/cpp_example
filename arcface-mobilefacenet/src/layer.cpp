#include "layer.hpp"


ILayer* convBnNoActivation(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = network->addConvolutionNd(input, num_filters, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setNbGroups(g);
    ILayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + "_batchnorm", 1e-3);
    return bn;
}

ILayer* convBnLeakyRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, int g, std::string lname) {
    ILayer* convBn = convBnNoActivation(network, weightMap, input, num_filters, k, s, p, g, lname);
    ILayer* leakeyRelu = addPRelu(network, weightMap, *convBn->getOutput(0), lname + "_relu");
    assert(leakeyRelu);
    return leakeyRelu;
}

ILayer* convBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, int g, std::string lname) {
    ILayer* convBn = convBnNoActivation(network, weightMap, input, num_filters, k, s, p, g, lname);
    IActivationLayer* relu = network->addActivation(*convBn->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    auto creator = getPluginRegistry()->getPluginCreator("PRelu_TRT", "1");
    PluginFieldCollection pfc;
    PluginField pf("gamma", weightMap[lname + "_gamma"].values, PluginFieldType::kFLOAT32, weightMap[lname + "_gamma"].count);
    pfc.nbFields = 1;
    pfc.fields = &pf;
    IPluginV2 *pluginObj = creator->createPlugin(lname.c_str(), &pfc);
    ITensor* inputTensors[] = {&input};
    auto prelu = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
    assert(prelu);
    return prelu;
}


ILayer* addRes3Block(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, std::string block_name){

    auto conv_sep = convBnLeakyRelu(network, weightMap, input, num_filters, 1, 1, 0, 1, block_name + "_conv_sep");
    auto conv_dw = convBnLeakyRelu(network, weightMap, *conv_sep->getOutput(0), num_filters, 3, 1, 1, num_filters, block_name + "_conv_dw");

    auto conv_proj_bn = convBnNoActivation(network, weightMap, *conv_dw->getOutput(0), num_filters/2, 1, 1, 0, 1,  block_name + "_conv_proj");
    IElementWiseLayer* ew1 = network->addElementWise(input, *conv_proj_bn->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    return ew1;
}

ILayer* addDconvBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, std::string block_name){

    auto conv_sep = convBnLeakyRelu(network, weightMap, input, num_filters, 1, 1, 0, 1, "dconv_" + block_name + "_conv_sep");
    auto conv_dw = convBnLeakyRelu(network, weightMap, *conv_sep->getOutput(0), num_filters, 3, 2, 1, num_filters, "dconv_" + block_name + "_conv_dw");
    int tmp = num_filters/2;
    if (num_filters == 512){
        tmp = 128;
    }
    auto conv_proj_bn = convBnNoActivation(network, weightMap, *conv_dw->getOutput(0), tmp, 1, 1, 0, 1, "dconv_" + block_name + "_conv_proj");
    // IActivationLayer* relu = network->addActivation(*conv_proj_bn->getOutput(0), ActivationType::kRELU);
    // assert(relu);
    return conv_proj_bn;
}