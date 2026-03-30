#pragma once
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
using namespace std;

enum class InitType { Xavier, Kaiming, Uniform, Normal };

class NDArray {
   private:
    shared_ptr<vector<float>> owned;
    span<float> data;

    void compute_strides();
    bool check_same_shape(const NDArray& r) const;
    int offset(const vector<int>& index) const;

   public:
    vector<int> shape;
    vector<int> stride;
    int ndim;
    int numel;
    NDArray(const vector<int>& shape_list);
    NDArray();

    float get(const vector<int>& index) const;
    void set(const vector<int>& index, float value);
    void reshape(const vector<int>& shape_list);
    void fill(float value);
    NDArray for_each_element(function<float(float)> fn) const;
    NDArray for_each_element_indexed(function<float(float, int)> fn) const;
    NDArray operator+(const float s) const;
    NDArray operator-(const float s) const;
    NDArray operator*(const float s) const;
    NDArray operator/(const float s) const;
    NDArray operator+(const NDArray& r) const;
    NDArray operator-(const NDArray& r) const;
    NDArray operator*(const NDArray& r) const;
    NDArray operator/(const NDArray& r) const;
    float& operator[](int offset);
    const float& operator[](int offset) const;
    NDArray matmul(const NDArray& r) const;
    NDArray sum(int axis, bool keepdims);
    NDArray transpose2d();
    NDArray mean(int axis, bool keepdims);
    string shape_str() const;

    vector<int> broadcast_shape(const NDArray& r) const;
    NDArray elementwise_broadcast(const NDArray& r,
                                  function<float(float, float)> fn) const;

    // Add this method to NDArray
    NDArray slice(int row_start, int row_end) const;
    void init_weights(InitType type = InitType::Kaiming);
};