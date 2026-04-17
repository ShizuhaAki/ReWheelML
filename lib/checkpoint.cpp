#include "checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

constexpr char kOldMagic[] = {'F', 'D', 'U', 'A', 'I', 'H', '\0', '\0'};
constexpr char kNewMagic[] = {'F', 'D', 'U', 'A', 'I', '2', '\0', '\0'};
constexpr size_t kMagicSize = sizeof(kNewMagic);

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

void write_string(ofstream& out, const string& value) {
    int size = static_cast<int>(value.size());
    write_binary(out, size);
    out.write(value.data(), size);
    if (!out) throw runtime_error("failed to write checkpoint string");
}

string read_string(ifstream& in) {
    int size = 0;
    read_binary(in, size);
    if (size < 0) throw runtime_error("invalid checkpoint string length");

    string value(size, '\0');
    in.read(value.data(), size);
    if (!in) throw runtime_error("failed to read checkpoint string");
    return value;
}

bool read_header(ifstream& in, CheckpointMetadata* metadata, bool* has_config) {
    char magic[kMagicSize];
    in.read(magic, kMagicSize);
    if (!in) return false;

    const string actual_magic(magic, kMagicSize);
    if (actual_magic != string(kOldMagic, kMagicSize) &&
        actual_magic != string(kNewMagic, kMagicSize)) {
        return false;
    }

    read_binary(in, metadata->epoch);
    read_binary(in, metadata->metric_value);
    metadata->metric_name = read_string(in);
    *has_config = actual_magic == string(kNewMagic, kMagicSize);
    if (*has_config) {
        metadata->config_text = read_string(in);
    } else {
        metadata->config_text.clear();
    }
    return true;
}

void save_checkpoint(const string& path, const vector<Parameter>& params,
                     int epoch, float metric_value, const string& metric_name,
                     const string& config_text) {
    fs::path checkpoint_path(path);
    if (checkpoint_path.has_parent_path()) {
        fs::create_directories(checkpoint_path.parent_path());
    }

    ofstream out(path, ios::binary);
    if (!out) {
        throw runtime_error("failed to open checkpoint for writing: " + path);
    }

    out.write(kNewMagic, kMagicSize);
    write_binary(out, epoch);
    write_binary(out, metric_value);
    write_string(out, metric_name);
    write_string(out, config_text);

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
                     int* epoch, float* metric_value, string* metric_name,
                     string* config_text) {
    ifstream in(path, ios::binary);
    if (!in) return false;

    CheckpointMetadata metadata;
    bool has_config = false;
    if (!read_header(in, &metadata, &has_config)) {
        return false;
    }

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

    if (epoch) *epoch = metadata.epoch;
    if (metric_value) *metric_value = metadata.metric_value;
    if (metric_name) *metric_name = metadata.metric_name;
    if (config_text) *config_text = metadata.config_text;
    return true;
}

bool load_checkpoint_metadata(const string& path, CheckpointMetadata* metadata) {
    if (metadata == nullptr) return false;

    ifstream in(path, ios::binary);
    if (!in) return false;

    bool has_config = false;
    if (!read_header(in, metadata, &has_config)) {
        return false;
    }
    return true;
}
