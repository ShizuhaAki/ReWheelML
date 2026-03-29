#include "ndarray.hpp"

#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

NDArray::NDArray(const vector<int>& shape_list) {
    shape = shape_list;
    ndim = shape_list.size();
    numel = reduce(shape.begin(), shape.end(), 1, multiplies<int>());
    data.resize(numel);
    compute_strides();
}

NDArray::NDArray() {}

void NDArray::compute_strides() {
    stride.resize(ndim);
    stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

int NDArray::offset(const vector<int>& index) const {
    if (index.size() != ndim) {
        throw invalid_argument("index length is not the same as ndim");
    }
    int offset = 0;
    for (int i = 0; i < ndim; i++) {
        offset += index[i] * stride[i];
    }
    return offset;
}

bool NDArray::check_same_shape(const NDArray& other) {
    return shape == other.shape;
}

void NDArray::fill(double value) {
    for (int i = 0; i < numel; i++) {
        data[i] = value;
    }
}

void NDArray::reshape(const vector<int>& shape_list) {
    int new_numel =
        reduce(shape_list.begin(), shape_list.end(), 1, multiplies<int>());
    if (new_numel != numel) {
        throw invalid_argument("reshape cannot change the number of elements");
    }
    shape = shape_list;
    ndim = shape_list.size();
    this->compute_strides();
}

double NDArray::get(const vector<int>& index) const {
    return data[offset(index)];
}

void NDArray::set(const vector<int>& index, double value) {
    data[offset(index)] = value;
}

NDArray NDArray::for_each_element(function<double(double)> fn) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = fn(data[i]);
    }
    return result;
}

NDArray NDArray::for_each_element_indexed(
    function<double(double, int)> fn) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = fn(data[i], i);
    }
    return result;
}

NDArray NDArray::operator+(const double s) const {
    return for_each_element([=](double x) { return x + s; });
}

NDArray NDArray::operator-(const double s) const {
    return for_each_element([=](double x) { return x - s; });
}

NDArray NDArray::operator*(const double s) const {
    return for_each_element([=](double x) { return x * s; });
}

NDArray NDArray::operator/(const double s) const {
    return for_each_element([=](double x) { return x / s; });
}

NDArray NDArray::operator+(const NDArray& r) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = data[i] + r.data[i];
    }
    return result;
}

NDArray NDArray::operator-(const NDArray& r) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = data[i] - r.data[i];
    }
    return result;
}

NDArray NDArray::operator*(const NDArray& r) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = data[i] * r.data[i];
    }
    return result;
}

NDArray NDArray::operator/(const NDArray& r) const {
    auto result = NDArray(shape);
    for (int i = 0; i < numel; i++) {
        result.data[i] = data[i] / r.data[i];
    }
    return result;
}

NDArray NDArray::matmul(const NDArray& r) const {
    if (shape.at(shape.size() - 1) != r.shape.at(r.shape.size() - 2)) {
        throw runtime_error(std::format("dimension mismatch: this {}, that {}",
                                        shape.at(shape.size() - 1),
                                        r.shape.at(r.shape.size() - 2)));
    }
    auto M = shape.at(shape.size() - 2);
    auto K = shape.at(shape.size() - 1);
    auto N = r.shape.at(r.shape.size() - 1);
    auto result = NDArray({M, N});
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                sum += get({i, k}) * r.get({k, j});
            }
            result.set({i, j}, sum);
        }
    }
    return result;
}

NDArray NDArray::sum(int axis, bool keepdims) {
    if (axis < 0 || axis >= shape.size()) {
        throw out_of_range("axis out of bounds");
    }

    vector<int> new_shape;
    for (int i = 0; i < shape.size(); i++) {
        if (i == axis) {
            if (keepdims) new_shape.push_back(1);
        } else {
            new_shape.push_back(shape[i]);
        }
    }

    NDArray result(new_shape);
    result.fill(0.0);

    for (int i = 0; i < numel; i++) {
        vector<int> idxs(ndim);
        int tmp = i;
        for (int j = 0; j < ndim; j++) {
            idxs[j] = tmp / stride[j];
            tmp %= stride[j];
        }
        vector<int> res_idx;
        for (int j = 0; j < ndim; j++) {
            if (j == axis) {
                if (keepdims) res_idx.push_back(0);
            } else {
                res_idx.push_back(idxs[j]);
            }
        }
        result.set(res_idx, result.get(res_idx) + data[i]);
    }

    return result;
}

const double& NDArray::operator[](int offset) const { return data[offset]; }

double& NDArray::operator[](int offset) { return data[offset]; }

NDArray NDArray::transpose2d() {
    if (ndim != 2) {
        throw invalid_argument("transpose only makes sense with 2D matrices");
    }

    auto result = NDArray({shape[1], shape[0]});
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            result.set({j, i}, this->get({i, j}));
        }
    }
    return result;
}

NDArray NDArray::mean(int axis, bool keepdims) {
    if (axis < 0 || axis >= shape.size()) {
        throw out_of_range("axis out of bounds");
    }

    NDArray s = sum(axis, keepdims);
    double divisor = static_cast<double>(shape[axis]);

    return s / divisor;  // uses operator/(double) defined earlier
}