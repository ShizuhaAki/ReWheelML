#include <cfenv>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>

#include "dense.hpp"
#include "functional.hpp"
#include "module.hpp"
#include "ndarray.hpp"
#include "vendor/inih.hpp"


using namespace std;

random_device rd;
mt19937 mt(rd());
uniform_real_distribution<double> dist(-numbers::pi, +numbers::pi);

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
    SinePredictor() {
        layers.push_back(std::make_unique<Dense>(1, 16));
        layers.push_back(std::make_unique<Dense>(16, 1, "linear"));
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
    void step(double lr) {
        for (auto& l : layers) {
            l->step(lr);
        }
    }
};

int main(int argc, char** argv) {
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
    // load configuration
    INIReader ini(argv[1]);
    if (ini.ParseError() != 0) {
        cerr << "failed to parse model.ini" << endl;
        return 0;
    }

    NDArray train_x, train_y;
    int train_samples, eval_samples;
    make_dataset(train_samples = ini.GetInteger("data", "train_size", 100),
                 train_x, train_y);
    NDArray eval_x, eval_y;
    make_dataset(eval_samples = ini.GetInteger("data", "eval_size", 100),
                 eval_x, eval_y);

    SinePredictor model;
    auto lr = ini.GetFloat("train", "lr", 0.01);
    auto epoches = ini.GetInteger("train", "epoches", 20);

    for (int i = 1; i <= epoches; i++) {
        cout << format("Epoch {}/{}: ", i, epoches);
        auto pred = model.forward(train_x);
        auto loss = mean_squared_error(pred, train_y).mean(0, false);
        auto grad = (pred - train_y) * 2 / train_samples;
        model.backward(grad);
        model.step(lr);
        auto loss_value = loss[0];
        cout << "loss: " << loss_value << endl;

        // cout << "Evaluating on validation set...";
        // auto eval_pred = model.forward(eval_x);
        // auto eval_loss = mean_squared_error(eval_pred, eval_y).mean(0,
        // false); cout << "Evaluation loss: " << eval_loss[0] << endl;
    }

    return 0;
}