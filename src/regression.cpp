#include <cfenv>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>

#include "functional.hpp"
#include "module/dense.hpp"
#include "module/module.hpp"
#include "ndarray.hpp"
#include "optim/adam.hpp"
#include "optim/optim.hpp"
#include "vendor/inih.hpp"


using namespace std;

random_device rd;
mt19937 mt(rd());
uniform_real_distribution<float> dist(-numbers::pi, +numbers::pi);

void make_dataset(int cnt, NDArray& X, NDArray& Y) {
    X = NDArray({cnt, 1});
    Y = NDArray({cnt, 1});
    for (int i = 0; i < cnt; i++) {
        auto xi = dist(mt);
        X.set({i, 0}, xi);
        Y.set({i, 0}, sin(xi));
    }
}

class SinePredictor : public Module {
   private:
    std::vector<std::unique_ptr<Module>> layers;

   public:
    SinePredictor(int hidden_size, int n_hidden_layers, string activation) {
        layers.push_back(std::make_unique<Linear>(1, hidden_size));
        layers.push_back(std::make_unique<Activation>(activation));
        if (n_hidden_layers > 1) {
            for (int i = 0; i < n_hidden_layers - 1; i++) {
                layers.push_back(
                    std::make_unique<Linear>(hidden_size, hidden_size));
                layers.push_back(std::make_unique<Activation>(activation));
            }
        }
        layers.push_back(std::make_unique<Linear>(hidden_size, 1));
    }
    NDArray forward(const NDArray& x) {
        auto out = x;
        for (auto& l : layers) {
            out = l->forward(out);
        }
        return out;
    }
    NDArray backward(const NDArray& grad) {
        auto grad_out = grad;
        for (auto& l : layers | ranges::views::reverse) {
            grad_out = l->backward(grad_out);
        }
        return grad_out;
    }
    void step(float lr) {
        for (auto& l : layers) {
            l->step(lr);
        }
    }

    std::vector<Parameter> parameters() override {
        std::vector<Parameter> result;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            result.insert(result.end(), layer_params.begin(), layer_params.end());
        }
        return result;
    }
};

int main(int argc, char** argv) {
    // feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
    //  load configuration
    INIReader ini(argv[1]);
    if (ini.ParseError() != 0) {
        cerr << "failed to parse INI configuration" << endl;
        return 0;
    }

    NDArray train_x, train_y;
    int train_samples, eval_samples;
    make_dataset(train_samples = ini.GetInteger("data", "train_size", 100),
                 train_x, train_y);
    NDArray eval_x, eval_y;
    make_dataset(eval_samples = ini.GetInteger("data", "eval_size", 100),
                 eval_x, eval_y);

    int hidden_size = ini.GetInteger("model", "hidden_size", 16);
    int n_hidden_layers = ini.GetInteger("model", "hidden_layers", 2);
    string activation = ini.Get("model", "activation", "sigmoid");
    SinePredictor model(hidden_size, n_hidden_layers, activation);
    auto lr = ini.GetFloat("train", "lr", 0.01);
    auto epoches = ini.GetInteger("train", "epoches", 20);
    string optimizer_name = ini.Get("train", "optimizer", "sgd");
    float beta1 = ini.GetFloat("train", "beta1", 0.9f);
    float beta2 = ini.GetFloat("train", "beta2", 0.999f);
    float eps = ini.GetFloat("train", "eps", 1e-8f);

    int batch_size = ini.GetInteger("train", "batch_size", 16);
    unique_ptr<Optimizer> optimizer;
    if (optimizer_name == "sgd") {
        optimizer = make_unique<SGD>(model.parameters(), lr);
    } else if (optimizer_name == "adam") {
        optimizer = make_unique<Adam>(model.parameters(), lr, beta1, beta2, eps);
    } else {
        cerr << "unknown optimizer: " << optimizer_name << endl;
        return 1;
    }

    for (int i = 1; i <= epoches; i++) {
        cout << format("Epoch {}/{}: ", i, epoches);
        float epoch_loss = 0.0;

        for (int start = 0; start < train_samples; start += batch_size) {
            int end = min(start + batch_size, train_samples);
            auto batch_x = train_x.slice(start, end);
            auto batch_y = train_y.slice(start, end);

            auto pred = model.forward(batch_x);
            auto loss = squared_error(pred, batch_y).mean(0, false);
            epoch_loss += loss[0] * (end - start);

            auto grad = (pred - batch_y) * 2 / (end - start);
            model.backward(grad);
            optimizer->step();
            optimizer->zero_grad();
        }

        cout << "loss: " << (epoch_loss / train_samples) << endl;

        cout << "Evaluating on validation set...";
        auto eval_pred = model.forward(eval_x);
        auto eval_loss = squared_error(eval_pred, eval_y).mean(0, false);
        cout << "Evaluation loss: " << eval_loss[0] << endl;

        cout << "Example predictions:\n";
        NDArray example_x({5, 1});
        NDArray example_y({5, 1});
        string fname = format("sine_plot_{}.csv", i);
        std::ofstream fout(fname);
        int points = 200;
        for (int i = 0; i <= points; i++) {
            float x_val = -numbers::pi + i * 2 * numbers::pi / points;
            NDArray x({1, 1});
            x.set({0, 0}, x_val);
            NDArray y_pred = model.forward(x);
            fout << x_val << "," << y_pred[0] << "," << sin(x_val) << "\n";
        }
        fout.close();
    }

    return 0;
}
