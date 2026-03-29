#pragma once
#include <functional>
#include <string>
#include <vector>
using namespace std;

class NDArray {
   private:
    vector<double> data;

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

    double get(const vector<int>& index) const;
    void set(const vector<int>& index, double value);
    void reshape(const vector<int>& shape_list);
    void fill(double value);
    NDArray for_each_element(function<double(double)> fn) const;
    NDArray for_each_element_indexed(function<double(double, int)> fn) const;
    NDArray operator+(const double s) const;
    NDArray operator-(const double s) const;
    NDArray operator*(const double s) const;
    NDArray operator/(const double s) const;
    NDArray operator+(const NDArray& r) const;
    NDArray operator-(const NDArray& r) const;
    NDArray operator*(const NDArray& r) const;
    NDArray operator/(const NDArray& r) const;
    double& operator[](int offset);
    const double& operator[](int offset) const;
    NDArray matmul(const NDArray& r) const;
    NDArray sum(int axis, bool keepdims);
    NDArray transpose2d();
    NDArray mean(int axis, bool keepdims);
    string shape_str() const;

    vector<int> broadcast_shape(const NDArray& r) const;
    NDArray elementwise_broadcast(const NDArray& r,
                                  function<double(double, double)> fn) const;
};