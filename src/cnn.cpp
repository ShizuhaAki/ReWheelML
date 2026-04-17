#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint.hpp"
#include "functional.hpp"
#include "image.hpp"
#include "module/batchnorm.hpp"
#include "module/conv2d.hpp"
#include "module/dense.hpp"
#include "module/dropout.hpp"
#include "module/module.hpp"
#include "module/pooling.hpp"
#include "optim/adam.hpp"
#include "optim/optim.hpp"
#include "scheduler.hpp"
#include "vendor/inih.hpp"

using namespace std;

namespace {
constexpr const char* kDefaultConfigPath = "cnn.ini";
constexpr const char* kDefaultCheckpointPath = "checkpoints/best_cnn.ckpt";

string read_text_file(const string& path) {
    ifstream in(path);
    if (!in) {
        throw runtime_error("failed to open file: " + path);
    }
    return string((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
}

INIReader load_ini_from_string(const string& config_text) {
    FILE* file = tmpfile();
    if (file == nullptr) {
        throw runtime_error("failed to create temporary file for INI parsing");
    }

    size_t written = fwrite(config_text.data(), 1, config_text.size(), file);
    if (written != config_text.size()) {
        fclose(file);
        throw runtime_error("failed to write temporary INI data");
    }
    rewind(file);

    INIReader ini(file);
    fclose(file);
    return ini;
}

struct InferenceOptions {
    string checkpoint_path = kDefaultCheckpointPath;
    string image_path;
    string dir_path;
};

InferenceOptions parse_inference_args(int argc, char** argv) {
    InferenceOptions options;
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--checkpoint") {
            if (i + 1 >= argc) {
                throw invalid_argument("missing value for --checkpoint");
            }
            options.checkpoint_path = argv[++i];
        } else if (arg == "--image") {
            if (i + 1 >= argc) {
                throw invalid_argument("missing value for --image");
            }
            options.image_path = argv[++i];
        } else if (arg == "--dir") {
            if (i + 1 >= argc) {
                throw invalid_argument("missing value for --dir");
            }
            options.dir_path = argv[++i];
        } else if (arg.starts_with("--")) {
            throw invalid_argument("unknown inference flag: " + arg);
        } else if (options.checkpoint_path == kDefaultCheckpointPath) {
            options.checkpoint_path = arg;
        } else {
            throw invalid_argument("unexpected positional argument: " + arg);
        }
    }
    return options;
}
}  // namespace

vector<int> slice_labels(const vector<int>& labels, int start, int end) {
    return vector<int>(labels.begin() + start, labels.begin() + end);
}

ImageDataset shuffle_dataset(const ImageDataset& dataset, mt19937& rng) {
    vector<int> indices(dataset.X.shape[0]);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    ImageDataset shuffled{
        NDArray({dataset.X.shape[0], dataset.X.shape[1]}),
        vector<int>(dataset.y.size()),
    };

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        int src = indices[i];
        for (int j = 0; j < dataset.X.shape[1]; ++j) {
            shuffled.X.set({i, j}, dataset.X.get({src, j}));
        }
        shuffled.y[i] = dataset.y[src];
    }

    return shuffled;
}

float accuracy(const NDArray& logits, const vector<int>& labels) {
    int correct = 0;
    for (int i = 0; i < logits.shape[0]; ++i) {
        correct += (argmax_row(logits, i) + 1 == labels[i]);
    }
    return labels.empty() ? 0.0f
                          : static_cast<float>(correct) /
                                static_cast<float>(labels.size());
}

float max_probability(const NDArray& probs, int row) {
    float best = probs.get({row, 0});
    for (int j = 1; j < probs.shape[1]; ++j) {
        best = max(best, probs.get({row, j}));
    }
    return best;
}

NDArray flat_to_nchw(const NDArray& x) {
    auto out = x;
    out.reshape({x.shape[0], 1, 28, 28});
    return out;
}

NDArray nchw_to_flat(const NDArray& x) {
    auto out = x;
    out.reshape({x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]});
    return out;
}

class ConvBlock : public Module {
   private:
    Conv2D conv1;
    BatchNorm2D bn1;
    Activation act1;
    Conv2D conv2;
    BatchNorm2D bn2;
    Activation act2;
    MaxPool2D pool;
    Dropout dropout;

   public:
    ConvBlock(int in_channels, int out_channels, int kernel_size,
              float dropout_p, const string& activation, unsigned int seed)
        : conv1(in_channels, out_channels, kernel_size, 1, kernel_size / 2),
          bn1(out_channels),
          act1(activation),
          conv2(out_channels, out_channels, kernel_size, 1, kernel_size / 2),
          bn2(out_channels),
          act2(activation),
          pool(2, 2),
          dropout(dropout_p, seed) {}

    NDArray forward(const NDArray& x) override {
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = act1.forward(out);
        out = conv2.forward(out);
        out = bn2.forward(out);
        out = act2.forward(out);
        out = pool.forward(out);
        out = dropout.forward(out);
        return out;
    }

    NDArray backward(const NDArray& grad_out) override {
        auto grad = dropout.backward(grad_out);
        grad = pool.backward(grad);
        grad = act2.backward(grad);
        grad = bn2.backward(grad);
        grad = conv2.backward(grad);
        grad = act1.backward(grad);
        grad = bn1.backward(grad);
        grad = conv1.backward(grad);
        return grad;
    }

    void step(float lr) override {
        conv1.step(lr);
        bn1.step(lr);
        conv2.step(lr);
        bn2.step(lr);
    }

    void train() override {
        conv1.train();
        bn1.train();
        act1.train();
        conv2.train();
        bn2.train();
        act2.train();
        pool.train();
        dropout.train();
    }

    void eval() override {
        conv1.eval();
        bn1.eval();
        act1.eval();
        conv2.eval();
        bn2.eval();
        act2.eval();
        pool.eval();
        dropout.eval();
    }

    vector<Parameter> parameters() override {
        vector<Parameter> result;
        for (auto params : {conv1.parameters(), bn1.parameters(),
                            conv2.parameters(), bn2.parameters()}) {
            result.insert(result.end(), params.begin(), params.end());
        }
        return result;
    }
};

class CNNClassifier : public Module {
   private:
    vector<unique_ptr<ConvBlock>> blocks;
    int final_channels;
    int final_spatial;
    unique_ptr<Linear> hidden;
    Activation hidden_activation;
    unique_ptr<Linear> classifier;

   public:
    CNNClassifier(int num_blocks, int base_channels, int channel_multiplier,
                  int kernel_size, float block_dropout, int hidden_size,
                  int num_classes, const string& activation, unsigned int seed)
        : final_channels(1), final_spatial(28), hidden_activation(activation) {
        if (num_blocks <= 0) {
            throw invalid_argument("num_blocks must be positive");
        }
        if (channel_multiplier <= 0) {
            throw invalid_argument("channel_multiplier must be positive");
        }

        int in_channels = 1;
        int out_channels = base_channels;
        for (int i = 0; i < num_blocks; ++i) {
            if (final_spatial < 2) {
                throw invalid_argument(
                    "too many pooling blocks for 28x28 inputs");
            }
            blocks.push_back(make_unique<ConvBlock>(in_channels, out_channels,
                                                    kernel_size, block_dropout,
                                                    activation, seed + i));
            in_channels = out_channels;
            final_channels = out_channels;
            final_spatial = (final_spatial - 2) / 2 + 1;
            out_channels *= channel_multiplier;
        }

        hidden = make_unique<Linear>(
            final_channels * final_spatial * final_spatial, hidden_size);
        classifier = make_unique<Linear>(hidden_size, num_classes);
    }

    NDArray forward(const NDArray& x) override {
        auto out = flat_to_nchw(x);
        for (auto& block : blocks) {
            out = block->forward(out);
        }
        out = nchw_to_flat(out);
        out = hidden->forward(out);
        out = hidden_activation.forward(out);
        out = classifier->forward(out);
        return out;
    }

    NDArray backward(const NDArray& grad_out) override {
        auto grad = classifier->backward(grad_out);
        grad = hidden_activation.backward(grad);
        grad = hidden->backward(grad);
        grad.reshape(
            {grad.shape[0], final_channels, final_spatial, final_spatial});
        for (auto& block : blocks | ranges::views::reverse) {
            grad = block->backward(grad);
        }
        return grad;
    }

    void step(float lr) override {
        for (auto& block : blocks) {
            block->step(lr);
        }
        hidden->step(lr);
        classifier->step(lr);
    }

    void train() override {
        for (auto& block : blocks) {
            block->train();
        }
        hidden->train();
        hidden_activation.train();
        classifier->train();
    }

    void eval() override {
        for (auto& block : blocks) {
            block->eval();
        }
        hidden->eval();
        hidden_activation.eval();
        classifier->eval();
    }

    vector<Parameter> parameters() override {
        vector<Parameter> result;
        for (auto& block : blocks) {
            auto params = block->parameters();
            result.insert(result.end(), params.begin(), params.end());
        }

        auto hidden_params = hidden->parameters();
        result.insert(result.end(), hidden_params.begin(), hidden_params.end());
        auto classifier_params = classifier->parameters();
        result.insert(result.end(), classifier_params.begin(),
                      classifier_params.end());
        return result;
    }
};

int main(int argc, char** argv) {
    bool inference_mode = argc > 1 && string(argv[1]) == "--inference";
    if (inference_mode) {
        try {
            auto options = parse_inference_args(argc, argv);

            CheckpointMetadata metadata;
            if (!load_checkpoint_metadata(options.checkpoint_path, &metadata)) {
                cerr << "failed to load checkpoint metadata: "
                     << options.checkpoint_path << endl;
                return 1;
            }
            if (metadata.config_text.empty()) {
                cerr << "checkpoint does not contain embedded INI: "
                     << options.checkpoint_path << endl;
                return 1;
            }

            auto ini = load_ini_from_string(metadata.config_text);
            if (ini.ParseError() != 0) {
                cerr << "failed to parse embedded INI from checkpoint" << endl;
                return 1;
            }

            bool invert = ini.GetBoolean("data", "invert", false);
            int num_classes = ini.GetInteger("model", "num_classes", 12);
            int num_blocks = ini.GetInteger("model", "blocks", 2);
            int base_channels = ini.GetInteger("model", "conv_channels", 16);
            int channel_multiplier =
                ini.GetInteger("model", "channel_multiplier", 2);
            int kernel_size = ini.GetInteger("model", "kernel_size", 3);
            int hidden_size = ini.GetInteger("model", "hidden_size", 128);
            float block_dropout = ini.GetFloat("model", "dropout", 0.1f);
            string activation = ini.Get("model", "activation", "relu");
            int seed = ini.GetInteger("train", "seed", 42);
            string inference_dir = options.dir_path.empty()
                                       ? ini.Get("data", "eval_dir", "data/test")
                                       : options.dir_path;

            CNNClassifier model(num_blocks, base_channels, channel_multiplier,
                                kernel_size, block_dropout, hidden_size,
                                num_classes, activation, seed);
            if (!load_checkpoint(options.checkpoint_path, model.parameters())) {
                cerr << "failed to load checkpoint weights: "
                     << options.checkpoint_path << endl;
                return 1;
            }

            model.eval();
            cout << format("Loaded checkpoint {} (epoch {}, {} = {:.6f})",
                           options.checkpoint_path, metadata.epoch,
                           metadata.metric_name, metadata.metric_value)
                 << endl;

            if (!options.image_path.empty()) {
                auto image = load_bmp_28x28_flat(options.image_path, invert);
                auto logits = model.forward(image);
                auto probs = softmax(logits);
                int pred = argmax_row(logits, 0) + 1;
                cout << format("Prediction for {}: class {}, confidence {:.4f}",
                               options.image_path, pred,
                               max_probability(probs, 0))
                     << endl;
                return 0;
            }

            cout << "Loading inference dataset from " << inference_dir << "..."
                 << endl;
            auto inference_data = load_bmp_dataset(inference_dir, invert);
            if (inference_data.y.empty()) {
                cerr << "inference dataset is empty" << endl;
                return 1;
            }

            auto logits = model.forward(inference_data.X);
            auto probs = softmax(logits);
            auto targets = one_hot(inference_data.y, num_classes);
            cout << format("Inference loss: {:.6f}, accuracy: {:.4f}",
                           cross_entropy(probs, targets),
                           accuracy(logits, inference_data.y))
                 << endl;
            return 0;
        } catch (const exception& e) {
            cerr << e.what() << endl;
            return 1;
        }
    }

    string config_path = argc > 1 ? argv[1] : kDefaultConfigPath;
    string config_text;
    try {
        config_text = read_text_file(config_path);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }

    INIReader ini(config_path);
    if (ini.ParseError() != 0) {
        cerr << "failed to parse " << config_path << endl;
        return 1;
    }

    string train_dir = ini.Get("data", "train_dir", "data/train");
    string eval_dir = ini.Get("data", "eval_dir", "data/test");
    bool invert = ini.GetBoolean("data", "invert", false);

    int num_classes = ini.GetInteger("model", "num_classes", 12);
    int num_blocks = ini.GetInteger("model", "blocks", 2);
    int base_channels = ini.GetInteger("model", "conv_channels", 16);
    int channel_multiplier = ini.GetInteger("model", "channel_multiplier", 2);
    int kernel_size = ini.GetInteger("model", "kernel_size", 3);
    int hidden_size = ini.GetInteger("model", "hidden_size", 128);
    float block_dropout = ini.GetFloat("model", "dropout", 0.1f);
    string activation = ini.Get("model", "activation", "relu");
    float lr = ini.GetFloat("train", "lr", 0.001f);
    int epoches = ini.GetInteger("train", "epoches", 20);
    int batch_size = ini.GetInteger("train", "batch_size", 64);
    int seed = ini.GetInteger("train", "seed", 42);
    string optimizer_name = ini.Get("train", "optimizer", "adam");
    float beta1 = ini.GetFloat("train", "beta1", 0.9f);
    float beta2 = ini.GetFloat("train", "beta2", 0.999f);
    float eps = ini.GetFloat("train", "eps", 1e-8f);
    bool scheduler_enabled = ini.GetBoolean("scheduler", "enabled", false);
    string scheduler_name = ini.Get("scheduler", "name", "plateau");
    string scheduler_metric = ini.Get("scheduler", "metric", "eval_loss");
    string scheduler_mode = ini.Get("scheduler", "mode", "min");
    float scheduler_factor = ini.GetFloat("scheduler", "factor", 0.3f);
    int scheduler_patience = ini.GetInteger("scheduler", "patience", 3);
    float scheduler_min_lr = ini.GetFloat("scheduler", "min_lr", 1e-5f);
    float scheduler_threshold =
        ini.GetFloat("scheduler", "threshold", 1e-4f);
    AugmentConfig augment_config{
        .max_shift =
            static_cast<int>(ini.GetInteger("augment", "max_shift", 0)),
        .max_rotation_deg = ini.GetFloat("augment", "max_rotation_deg", 0.0f),
        .morphology_prob = ini.GetFloat("augment", "morphology_prob", 0.0f),
        .noise_std = ini.GetFloat("augment", "noise_std", 0.0f),
    };
    bool checkpoint_enabled = ini.GetBoolean("checkpoint", "enabled", false);
    string checkpoint_path =
        ini.Get("checkpoint", "path", kDefaultCheckpointPath);
    string checkpoint_metric = ini.Get("checkpoint", "metric", "eval_loss");
    string checkpoint_mode = ini.Get("checkpoint", "mode", "min");

    mt19937 rng(seed);
    CNNClassifier model(num_blocks, base_channels, channel_multiplier,
                        kernel_size, block_dropout, hidden_size, num_classes,
                        activation, seed);

    unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "sgd") {
        optimizer = make_unique<SGD>(model.parameters(), lr);
    } else if (optimizer_name == "adam") {
        optimizer =
            make_unique<Adam>(model.parameters(), lr, beta1, beta2, eps);
    } else {
        cerr << "unknown optimizer: " << optimizer_name << endl;
        return 1;
    }

    unique_ptr<Scheduler> scheduler;
    if (scheduler_enabled) {
        if (scheduler_name == "plateau") {
            scheduler = make_unique<ReduceLROnPlateau>(
                optimizer.get(), scheduler_factor, scheduler_patience,
                scheduler_min_lr, scheduler_threshold, scheduler_mode);
        } else {
            cerr << "unknown scheduler: " << scheduler_name << endl;
            return 1;
        }
    }

    cout << "Loading training images from " << train_dir << "..." << endl;
    auto train = load_bmp_dataset(train_dir, invert);
    cout << "Loaded " << train.y.size() << " training samples" << endl;

    cout << "Loading evaluation images from " << eval_dir << "..." << endl;
    auto eval = load_bmp_dataset(eval_dir, invert);
    cout << "Loaded " << eval.y.size() << " evaluation samples" << endl;

    if (train.y.empty() || eval.y.empty()) {
        cerr << "dataset is empty" << endl;
        return 1;
    }

    for (int epoch = 1; epoch <= epoches; ++epoch) {
        float epoch_loss = 0.0f;
        int seen = 0;
        auto shuffled_train = shuffle_dataset(train, rng);
        auto augmented_train =
            augment_dataset(shuffled_train, augment_config, rng);
        model.train();
        for (int start = 0; start < augmented_train.X.shape[0];
             start += batch_size) {
            int end = min(start + batch_size, augmented_train.X.shape[0]);
            auto batch_x = augmented_train.X.slice(start, end);
            auto batch_labels = slice_labels(augmented_train.y, start, end);
            auto batch_y = one_hot(batch_labels, num_classes);

            auto logits = model.forward(batch_x);
            auto probs = softmax(logits);
            float batch_loss = cross_entropy(probs, batch_y);
            epoch_loss += batch_loss * (end - start);
            seen += end - start;

            auto grad = cross_entropy_grad(probs, batch_y);
            model.backward(grad);
            optimizer->step();
            optimizer->zero_grad();
        }
        model.eval();
        auto train_logits = model.forward(train.X);
        auto eval_logits = model.forward(eval.X);
        auto eval_targets = one_hot(eval.y, num_classes);
        auto eval_probs = softmax(eval_logits);
        float eval_loss = cross_entropy(eval_probs, eval_targets);
        float eval_acc = accuracy(eval_logits, eval.y);

        cout << format(
                    "Epoch {}/{} - loss: {:.6f}, train_acc: {:.4f}, "
                    "eval_loss: {:.6f}, eval_acc: {:.4f}, lr: {:.6g}",
                    epoch, epoches, epoch_loss / seen,
                    accuracy(train_logits, train.y), eval_loss, eval_acc,
                    optimizer->get_lr())
             << endl;

        if (scheduler) {
            float scheduler_value =
                scheduler_metric == "eval_acc" ? eval_acc : eval_loss;
            if (scheduler->step(scheduler_value)) {
                cout << format("Reduced learning rate to {:.6g}",
                               optimizer->get_lr())
                     << endl;
            }
        }

        if (checkpoint_enabled) {
            float current_metric =
                checkpoint_metric == "eval_acc" ? eval_acc : eval_loss;
            static bool has_best = false;
            static float best_metric = 0.0f;

            bool improved = !has_best || (checkpoint_mode == "max"
                                              ? current_metric > best_metric
                                              : current_metric < best_metric);
            if (improved) {
                save_checkpoint(checkpoint_path, model.parameters(), epoch,
                                current_metric, checkpoint_metric,
                                config_text);
                best_metric = current_metric;
                has_best = true;
                cout << format("Saved checkpoint: {} ({})", checkpoint_path,
                               current_metric)
                     << endl;
            }
        }
    }

    return 0;
}
