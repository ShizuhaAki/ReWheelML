#include "ndarray.hpp"

#include <cblas.h>

#include <format>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

NDArray::NDArray(const vector<int>& shape_list) {
    shape = shape_list;
    ndim = shape_list.size();
    numel = reduce(shape.begin(), shape.end(), 1, multiplies<int>());

    owned = make_shared<vector<float>>(numel);
    data = span<float>(*owned);

    compute_strides();
}

NDArray::NDArray() : owned(make_shared<vector<float>>()), data(*owned) {}

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

bool NDArray::check_same_shape(const NDArray& other) const {
    return shape == other.shape;
}

void NDArray::fill(float value) {
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

float NDArray::get(const vector<int>& index) const {
    return data[offset(index)];
}

void NDArray::set(const vector<int>& index, float value) {
    data[offset(index)] = value;
}

NDArray NDArray::for_each_element(function<float(float)> fn) const {
    auto result = NDArray(shape);
#pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        result.data[i] = fn(data[i]);
    }
    return result;
}

NDArray NDArray::for_each_element_indexed(
    function<float(float, int)> fn) const {
    auto result = NDArray(shape);
#pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        result.data[i] = fn(data[i], i);
    }
    return result;
}

NDArray NDArray::operator+(const float s) const {
    return for_each_element([=](float x) { return x + s; });
}

NDArray NDArray::operator-(const float s) const {
    return for_each_element([=](float x) { return x - s; });
}

NDArray NDArray::operator*(const float s) const {
    return for_each_element([=](float x) { return x * s; });
}

NDArray NDArray::operator/(const float s) const {
    return for_each_element([=](float x) { return x / s; });
}

NDArray NDArray::operator+(const NDArray& r) const {
    return elementwise_broadcast(r, [](float a, float b) { return a + b; });
}

NDArray NDArray::operator-(const NDArray& r) const {
    return elementwise_broadcast(r, [](float a, float b) { return a - b; });
}

NDArray NDArray::operator*(const NDArray& r) const {
    return elementwise_broadcast(r, [](float a, float b) { return a * b; });
}

NDArray NDArray::operator/(const NDArray& r) const {
    return elementwise_broadcast(r, [](float a, float b) { return a / b; });
}

NDArray NDArray::matmul(const NDArray& r) const {
    if (ndim != 2 || r.ndim != 2) {
        throw runtime_error("matmul currently requires two 2D arrays");
    }
    if (shape[1] != r.shape[0]) {
        throw runtime_error(std::format("dimension mismatch: this {}, that {}",
                                        shape[1], r.shape[0]));
    }

    const int M = shape[0];
    const int K = shape[1];
    const int N = r.shape[1];

    NDArray result({M, N});

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                data.data(), K, r.data.data(), N, 0.0, result.data.data(), N);

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

const float& NDArray::operator[](int offset) const { return data[offset]; }

float& NDArray::operator[](int offset) { return data[offset]; }

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
    float divisor = static_cast<float>(shape[axis]);

    return s / divisor;  // uses operator/(float) defined earlier
}

string NDArray::shape_str() const {
    stringstream ret;
    ret << ndim;
    ret << "[";
    for (auto l : shape) ret << l << ",";
    ret << "]";
    string s;
    ret >> s;
    return s;
}
std::vector<int> NDArray::broadcast_shape(const NDArray& r) const {
    int n = max(ndim, r.ndim);
    vector<int> result_shape(n);

    for (int i = 0; i < n; i++) {
        int dim_a = (i < n - ndim) ? 1 : shape[i - (n - ndim)];
        int dim_b = (i < n - r.ndim) ? 1 : r.shape[i - (n - r.ndim)];

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw runtime_error(format("cannot broadcast shapes {} and {}",
                                       shape_str(), r.shape_str()));
        }
        result_shape[i] = max(dim_a, dim_b);
    }

    return result_shape;
}
NDArray NDArray::elementwise_broadcast(const NDArray& r,
                                       function<float(float, float)> fn) const {
    vector<int> result_shape = broadcast_shape(r);
    NDArray result(result_shape);

    int n = result_shape.size();
    vector<int> idx(n, 0);

    for (int i = 0; i < result.numel; i++) {
        int offset_a = 0, offset_b = 0;
        for (int j = 0; j < n; j++) {
            int dim_a = (j < n - ndim) ? 1 : shape[j - (n - ndim)];
            int dim_b = (j < n - r.ndim) ? 1 : r.shape[j - (n - r.ndim)];
            int idx_j = idx[j];

            if (dim_a > 1) offset_a += idx_j * stride[j - (n - ndim)];
            if (dim_b > 1) offset_b += idx_j * r.stride[j - (n - r.ndim)];
        }

        result.data[i] = fn(data[offset_a], r.data[offset_b]);
        for (int j = n - 1; j >= 0; j--) {
            idx[j]++;
            if (idx[j] < result_shape[j]) break;
            idx[j] = 0;
        }
    }

    return result;
}
NDArray NDArray::slice(int row_start, int row_end) const {
    int rows = row_end - row_start;
    int cols = shape[1];
    int start = row_start * stride[0];
    int count = rows * cols;

    NDArray out;
    out.owned = owned;
    out.data = data.subspan(start, count);
    out.shape = {rows, cols};
    out.ndim = 2;
    out.numel = count;
    out.compute_strides();
    return out;
}

void NDArray::init_weights(InitType type) {
    if (numel == 0) return;

    std::random_device rd;
    std::mt19937 gen(rd());

    float scale = 1.0;

    if (ndim < 2)
        throw std::runtime_error(
            "Weight initialization requires at least 2D tensor");

    int fan_in = shape[1];
    int fan_out = shape[0];

    switch (type) {
        case InitType::Xavier:
            scale = std::sqrt(2.0 / (fan_in + fan_out));
            break;
        case InitType::Kaiming:
            scale = std::sqrt(2.0 / fan_in);
            break;
        case InitType::Uniform:
            scale = 0.05;  // small uniform range
            break;
        case InitType::Normal:
            scale = 0.02;  // small normal std
            break;
    }

    std::uniform_real_distribution<float> uni_dist(-scale, scale);
    std::normal_distribution<float> norm_dist(0.0, scale);

    for (int i = 0; i < numel; ++i) {
        if (type == InitType::Normal || type == InitType::Kaiming ||
            type == InitType::Xavier) {
            data[i] = norm_dist(gen);
        } else {
            data[i] = uni_dist(gen);
        }
    }
}