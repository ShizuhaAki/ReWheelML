#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "ndarray.hpp"
#include "vendor/doctest.hpp"

#include <vector>


using namespace std;

TEST_CASE("NDArray basic construction and properties") {
    NDArray arr({2, 3});
    CHECK(arr.ndim == 2);
    CHECK(arr.numel == 6);
    CHECK(arr.stride[0] == 3);
    CHECK(arr.stride[1] == 1);
}

TEST_CASE("NDArray set and get") {
    NDArray arr({2, 2});
    arr.set({0, 0}, 1.0);
    arr.set({0, 1}, 2.0);
    arr.set({1, 0}, 3.0);
    arr.set({1, 1}, 4.0);

    CHECK(arr.get({0, 0}) == 1.0);
    CHECK(arr.get({0, 1}) == 2.0);
    CHECK(arr.get({1, 0}) == 3.0);
    CHECK(arr.get({1, 1}) == 4.0);
}

TEST_CASE("NDArray fill") {
    NDArray arr({2, 2});
    arr.fill(5.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) CHECK(arr.get({i, j}) == 5.0);
}

TEST_CASE("NDArray reshape") {
    NDArray arr({2, 3});
    arr.reshape({3, 2});
    CHECK(arr.shape[0] == 3);
    CHECK(arr.shape[1] == 2);
    CHECK(arr.numel == 6);
}

TEST_CASE("NDArray arithmetic with scalar") {
    NDArray arr({2, 2});
    arr.fill(2.0);
    auto res_add = arr + 3.0;
    auto res_mul = arr * 2.0;

    CHECK(res_add.get({0, 0}) == 5.0);
    CHECK(res_add.get({1, 1}) == 5.0);
    CHECK(res_mul.get({0, 0}) == 4.0);
    CHECK(res_mul.get({1, 1}) == 4.0);
}

TEST_CASE("NDArray arithmetic with another NDArray") {
    NDArray a({2, 2});
    NDArray b({2, 2});
    a.fill(1.0);
    b.fill(2.0);

    auto res_add = a + b;
    auto res_mul = a * b;

    CHECK(res_add.get({0, 0}) == 3.0);
    CHECK(res_mul.get({1, 1}) == 2.0);
}

TEST_CASE("NDArray offset error") {
    NDArray arr({2, 2});
    CHECK_THROWS_AS(arr.get({0}), invalid_argument);
}

TEST_CASE("NDArray matmul") {
    NDArray a({2, 3});
    NDArray b({3, 2});
    a.set({0, 0}, 1);
    a.set({0, 1}, 2);
    a.set({0, 2}, 3);
    a.set({1, 0}, 4);
    a.set({1, 1}, 5);
    a.set({1, 2}, 6);

    b.set({0, 0}, 7);
    b.set({0, 1}, 8);
    b.set({1, 0}, 9);
    b.set({1, 1}, 10);
    b.set({2, 0}, 11);
    b.set({2, 1}, 12);

    auto c = a.matmul(b);
    CHECK(c.get({0, 0}) == 58);
    CHECK(c.get({0, 1}) == 64);
    CHECK(c.get({1, 0}) == 139);
    CHECK(c.get({1, 1}) == 154);
}

TEST_CASE("NDArray sum") {
    NDArray a({2, 3});
    a.set({0, 0}, 1);
    a.set({0, 1}, 2);
    a.set({0, 2}, 3);
    a.set({1, 0}, 4);
    a.set({1, 1}, 5);
    a.set({1, 2}, 6);
    
    auto sum0 = a.sum(0, false);
    CHECK(sum0.shape == std::vector<int>{3});
    CHECK(sum0.get({0}) == 5);   // 1 + 4
    CHECK(sum0.get({1}) == 7);   // 2 + 5
    CHECK(sum0.get({2}) == 9);   // 3 + 6

    auto sum1 = a.sum(1, false);
    CHECK(sum1.shape == std::vector<int>{2});
    CHECK(sum1.get({0}) == 6);   // 1 + 2 + 3
    CHECK(sum1.get({1}) == 15);  // 4 + 5 + 6

    auto sum0k = a.sum(0, true);
    CHECK(sum0k.shape == std::vector<int>{1, 3});
    CHECK(sum0k.get({0, 0}) == 5);
    CHECK(sum0k.get({0, 1}) == 7);
    CHECK(sum0k.get({0, 2}) == 9);

    auto sum1k = a.sum(1, true);
    CHECK(sum1k.shape == std::vector<int>{2, 1});
    CHECK(sum1k.get({0, 0}) == 6);
    CHECK(sum1k.get({1, 0}) == 15);
}