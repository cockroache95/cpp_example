#include "arcface_mobilefacenet.hpp"
#include "layer.hpp"



// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, std::string weight_path, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(weight_path);

    std::cout << "Load weight done\n";
    // Weights emptywts{DataType::kFLOAT, nullptr, 0};


    auto conv_1 = convBnLeakyRelu(network, weightMap, *data, 64, 3, 2, 1, 1, "conv_1");

    auto conv_2 = convBnLeakyRelu(network, weightMap, *conv_1->getOutput(0), 64, 3, 1, 1, 64, "conv_2_dw");

    auto dconv23 = addDconvBlock(network, weightMap, *conv_2->getOutput(0), 128, "23");

    auto res_3_block0 = addRes3Block(network, weightMap, *dconv23->getOutput(0), 128, "res_3_block0");

    auto res_3_block1 = addRes3Block(network, weightMap, *res_3_block0->getOutput(0), 128, "res_3_block1");
    auto res_3_block2 = addRes3Block(network, weightMap, *res_3_block1->getOutput(0), 128, "res_3_block2");
    auto res_3_block3 = addRes3Block(network, weightMap, *res_3_block2->getOutput(0), 128, "res_3_block3");

    auto dconv34 = addDconvBlock(network, weightMap, *res_3_block3->getOutput(0), 256, "34");

    auto res_4_block0 = addRes3Block(network, weightMap, *dconv34->getOutput(0), 256, "res_4_block0");
    auto res_4_block1 = addRes3Block(network, weightMap, *res_4_block0->getOutput(0), 256, "res_4_block1");
    auto res_4_block2 = addRes3Block(network, weightMap, *res_4_block1->getOutput(0), 256, "res_4_block2");
    auto res_4_block3 = addRes3Block(network, weightMap, *res_4_block2->getOutput(0), 256, "res_4_block3");
    auto res_4_block4 = addRes3Block(network, weightMap, *res_4_block3->getOutput(0), 256, "res_4_block4");
    auto res_4_block5 = addRes3Block(network, weightMap, *res_4_block4->getOutput(0), 256, "res_4_block5");

    auto dconv45 = addDconvBlock(network, weightMap, *res_4_block5->getOutput(0), 512, "45");

    auto res_5_block0 = addRes3Block(network, weightMap, *dconv45->getOutput(0), 256, "res_5_block0");
    auto res_5_block1 = addRes3Block(network, weightMap, *res_5_block0->getOutput(0), 256, "res_5_block1");

    auto conv6sep = convBnLeakyRelu(network, weightMap, *res_5_block1->getOutput(0), 512, 1, 1, 0, 1, "conv_6sep");

    auto conv_6dw7_7 = convBnRelu(network, weightMap, *conv6sep->getOutput(0), 512, 7, 1, 0, 512, "conv_6dw7_7");

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*conv_6dw7_7->getOutput(0), 128, weightMap["fc1_weight"], weightMap["pre_fc1_bias"]);
    
    assert(fc1);
    auto bn2 = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);

    bn2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*bn2->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string weight_path) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, weight_path, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, "../arcface_mobilefacenet.wts");
        assert(modelStream != nullptr);
        std::ofstream p("arcface_mobilefacenet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("arcface_mobilefacenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./arcface_mobilefacenet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./arcface_mobilefacenet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cv::Mat img = cv::imread("../images/joey0.ppm");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(128, 1, CV_32FC1, prob);
    cv::Mat out_norm;
    cv::normalize(out, out_norm);

    img = cv::imread("../images/x1.png");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
    }

    // Run inference
    start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out1(1, 128, CV_32FC1, prob);
    cv::Mat out_norm1;
    cv::normalize(out1, out_norm1);

    cv::Mat res = out_norm1 * out_norm;

    std::cout << "similarity score: " << *(float*)res.data << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    //Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << p_out_norm[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}