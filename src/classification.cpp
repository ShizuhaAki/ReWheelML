#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "checkpoint.hpp"
#include "functional.hpp"
#include "image.hpp"
#include "module/dense.hpp"
#include "module/dropout.hpp"
#include "module/module.hpp"
#include "ndarray.hpp"
#include "optim/adam.hpp"
#include "optim/optim.hpp"
#include "vendor/inih.hpp"

using namespace std;

NDArray one_hot(const vector<int>& labels, int num_classes) {
    NDArray encoded({static_cast<int>(labels.size()), num_classes});
    encoded.fill(0.0f);
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        encoded.set({i, labels[i] - 1}, 1.0f);
    }
    return encoded;
}

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

int argmax_row(const NDArray& x, int row) {
    int best = 0;
    float best_value = x.get({row, 0});
    for (int j = 1; j < x.shape[1]; ++j) {
        float value = x.get({row, j});
        if (value > best_value) {
            best_value = value;
            best = j;
        }
    }
    return best;
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

class DigitClassifier : public Module {
   private:
    vector<unique_ptr<Module>> layers;

   public:
    DigitClassifier(int input_dim, int hidden_size, int hidden_layers,
                    int num_classes, const string& activation, float dropout_p,
                    unsigned int seed) {
        layers.push_back(make_unique<Linear>(input_dim, hidden_size));
        layers.push_back(make_unique<Activation>(activation));
        if (dropout_p > 0.0f) {
            layers.push_back(make_unique<Dropout>(dropout_p, seed));
        }
        for (int i = 1; i < hidden_layers; ++i) {
            layers.push_back(make_unique<Linear>(hidden_size, hidden_size));
            layers.push_back(make_unique<Activation>(activation));
            if (dropout_p > 0.0f) {
                layers.push_back(make_unique<Dropout>(dropout_p, seed + i));
            }
        }
        layers.push_back(make_unique<Linear>(hidden_size, num_classes));
    }

    NDArray forward(const NDArray& x) override {
        auto out = x;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    NDArray backward(const NDArray& grad) override {
        auto grad_out = grad;
        for (auto& layer : layers | ranges::views::reverse) {
            grad_out = layer->backward(grad_out);
        }
        return grad_out;
    }

    void step(float lr) override {
        for (auto& layer : layers) {
            layer->step(lr);
        }
    }

    void train() override {
        for (auto& layer : layers) {
            layer->train();
        }
    }

    void eval() override {
        for (auto& layer : layers) {
            layer->eval();
        }
    }

    vector<Parameter> parameters() override {
        vector<Parameter> result;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            result.insert(result.end(), layer_params.begin(), layer_params.end());
        }
        return result;
    }
};

int main(int argc, char** argv) {
    string config_path = argc > 1 ? argv[1] : "model.ini";
    INIReader ini(config_path);
    if (ini.ParseError() != 0) {
        cerr << "failed to parse " << config_path << endl;
        return 1;
    }

    string train_dir = ini.Get("data", "train_dir", "data/train");
    string eval_dir = ini.Get("data", "eval_dir", "data/test");
    bool invert = ini.GetBoolean("data", "invert", false);

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

    int input_dim = train.X.shape[1];
    int num_classes = ini.GetInteger("model", "num_classes", 10);
    int hidden_size = ini.GetInteger("model", "hidden_size", 128);
    int hidden_layers = ini.GetInteger("model", "hidden_layers", 2);
    string activation = ini.Get("model", "activation", "relu");
    float lr = ini.GetFloat("train", "lr", 0.01f);
    int epoches = ini.GetInteger("train", "epoches", 20);
    int batch_size = ini.GetInteger("train", "batch_size", 64);
    int seed = ini.GetInteger("train", "seed", 42);
    float dropout_p = ini.GetFloat("model", "dropout", 0.0f);
    string optimizer_name = ini.Get("train", "optimizer", "adam");
    float beta1 = ini.GetFloat("train", "beta1", 0.9f);
    float beta2 = ini.GetFloat("train", "beta2", 0.999f);
    float eps = ini.GetFloat("train", "eps", 1e-8f);
    AugmentConfig augment_config{
        .max_shift = static_cast<int>(ini.GetInteger("augment", "max_shift", 0)),
        .noise_std = ini.GetFloat("augment", "noise_std", 0.0f),
    };
    bool checkpoint_enabled = ini.GetBoolean("checkpoint", "enabled", false);
    string checkpoint_path =
        ini.Get("checkpoint", "path", "checkpoints/best_classification.ckpt");
    string checkpoint_metric = ini.Get("checkpoint", "metric", "eval_loss");
    string checkpoint_mode = ini.Get("checkpoint", "mode", "min");

    mt19937 rng(seed);

    DigitClassifier model(input_dim, hidden_size, hidden_layers, num_classes,
                          activation, dropout_p, seed);
    unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "sgd") {
        optimizer = make_unique<SGD>(model.parameters(), lr);
    } else if (optimizer_name == "adam") {
        optimizer = make_unique<Adam>(model.parameters(), lr, beta1, beta2, eps);
    } else {
        cerr << "unknown optimizer: " << optimizer_name << endl;
        return 1;
    }

    for (int epoch = 1; epoch <= epoches; ++epoch) {
        float epoch_loss = 0.0f;
        int seen = 0;
        auto shuffled_train = shuffle_dataset(train, rng);
        auto augmented_train = augment_dataset(shuffled_train, augment_config, rng);
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
                    "eval_loss: {:.6f}, eval_acc: {:.4f}",
                    epoch, epoches, epoch_loss / seen,
                    accuracy(train_logits, train.y), eval_loss, eval_acc)
             << endl;

        if (checkpoint_enabled) {
            float current_metric =
                checkpoint_metric == "eval_acc" ? eval_acc : eval_loss;
            static bool has_best = false;
            static float best_metric = 0.0f;

            bool improved =
                !has_best ||
                (checkpoint_mode == "max" ? current_metric > best_metric
                                           : current_metric < best_metric);
            if (improved) {
                save_checkpoint(checkpoint_path, model.parameters(), epoch,
                                current_metric, checkpoint_metric);
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
