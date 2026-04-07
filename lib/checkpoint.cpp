#include "checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

namespace {
constexpr const char kMagic[] = "FDUAIH";
constexpr size_t kMagicSize = 8;

template <typename T>
void write_binary(ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw runtime_error("failed to write checkpoint");
}

template <typename T>
void read_binary(ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw runtime_error("failed to read checkpoint");
}
}  // namespace

void save_checkpoint(const string& path, const vector<Parameter>& params,
                     int epoch, float metric_value, const string& metric_name) {
    fs::path checkpoint_path(path);
    if (checkpoint_path.has_parent_path()) {
        fs::create_directories(checkpoint_path.parent_path());
    }

    ofstream out(path, ios::binary);
    if (!out) {
        throw runtime_error("failed to open checkpoint for writing: " + path);
    }

    out.write(kMagic, kMagicSize);
    write_binary(out, epoch);
    write_binary(out, metric_value);

    int metric_name_size = static_cast<int>(metric_name.size());
    write_binary(out, metric_name_size);
    out.write(metric_name.data(), metric_name_size);

    int param_count = static_cast<int>(params.size());
    write_binary(out, param_count);

    for (const auto& param : params) {
        int ndim = param.value->ndim;
        write_binary(out, ndim);
        for (int dim : param.value->shape) {
            write_binary(out, dim);
        }
        for (int i = 0; i < param.value->numel; ++i) {
            write_binary(out, (*param.value)[i]);
        }
    }
}

bool load_checkpoint(const string& path, const vector<Parameter>& params,
                     int* epoch, float* metric_value, string* metric_name) {
    ifstream in(path, ios::binary);
    if (!in) return false;

    char magic[kMagicSize];
    in.read(magic, kMagicSize);
    if (!in || string(magic, kMagicSize) != string(kMagic, kMagicSize)) {
        return false;
    }

    int saved_epoch = 0;
    float saved_metric_value = 0.0f;
    int metric_name_size = 0;
    read_binary(in, saved_epoch);
    read_binary(in, saved_metric_value);
    read_binary(in, metric_name_size);

    string saved_metric_name(metric_name_size, '\0');
    in.read(saved_metric_name.data(), metric_name_size);
    if (!in) throw runtime_error("failed to read checkpoint metric name");

    int param_count = 0;
    read_binary(in, param_count);
    if (param_count != static_cast<int>(params.size())) {
        return false;
    }

    for (int i = 0; i < param_count; ++i) {
        int ndim = 0;
        read_binary(in, ndim);
        vector<int> shape(ndim);
        for (int j = 0; j < ndim; ++j) {
            read_binary(in, shape[j]);
        }

        if (shape != params[i].value->shape) {
            return false;
        }

        for (int j = 0; j < params[i].value->numel; ++j) {
            float value = 0.0f;
            read_binary(in, value);
            (*params[i].value)[j] = value;
        }
    }

    if (epoch) *epoch = saved_epoch;
    if (metric_value) *metric_value = saved_metric_value;
    if (metric_name) *metric_name = saved_metric_name;
    return true;
}
