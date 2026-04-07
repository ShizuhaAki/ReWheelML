#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

bool has_bmp_extension(const fs::path& path) {
    auto ext = path.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });
    return ext == ".bmp";
}

vector<fs::path> collect_class_dirs(const fs::path& input_root) {
    vector<fs::path> class_dirs;
    for (const auto& entry : fs::directory_iterator(input_root)) {
        if (entry.is_directory()) {
            class_dirs.push_back(entry.path());
        }
    }
    sort(class_dirs.begin(), class_dirs.end());
    return class_dirs;
}

vector<fs::path> collect_bmp_files(const fs::path& class_dir) {
    vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(class_dir)) {
        if (entry.is_regular_file() && has_bmp_extension(entry.path())) {
            files.push_back(entry.path());
        }
    }
    sort(files.begin(), files.end());
    return files;
}

void recreate_dir(const fs::path& dir) {
    if (fs::exists(dir)) {
        fs::remove_all(dir);
    }
    fs::create_directories(dir);
}

void copy_split(const vector<fs::path>& files, int train_count,
                const fs::path& train_dir, const fs::path& test_dir) {
    fs::create_directories(train_dir);
    fs::create_directories(test_dir);

    for (int i = 0; i < static_cast<int>(files.size()); ++i) {
        const fs::path& dst_dir = i < train_count ? train_dir : test_dir;
        fs::copy_file(files[i], dst_dir / files[i].filename(),
                      fs::copy_options::overwrite_existing);
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        cerr << "usage: split_dataset <input_root> <train_root> <test_root> "
                "[train_ratio] [seed]"
             << endl;
        return 1;
    }

    fs::path input_root = argv[1];
    fs::path train_root = argv[2];
    fs::path test_root = argv[3];
    double train_ratio = argc >= 5 ? stod(argv[4]) : 0.8;
    unsigned int seed = argc >= 6 ? static_cast<unsigned int>(stoul(argv[5]))
                                  : 42u;

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw invalid_argument("train_ratio must be between 0 and 1");
    }
    if (!fs::exists(input_root) || !fs::is_directory(input_root)) {
        throw runtime_error("input root does not exist or is not a directory");
    }

    auto class_dirs = collect_class_dirs(input_root);
    if (class_dirs.empty()) {
        throw runtime_error("no class directories found under input root");
    }

    recreate_dir(train_root);
    recreate_dir(test_root);

    mt19937 rng(seed);
    int total_train = 0;
    int total_test = 0;

    for (const auto& class_dir : class_dirs) {
        auto files = collect_bmp_files(class_dir);
        if (files.empty()) {
            continue;
        }

        shuffle(files.begin(), files.end(), rng);

        int train_count = static_cast<int>(files.size() * train_ratio);
        train_count = max(1, min(train_count, static_cast<int>(files.size()) - 1));
        int test_count = static_cast<int>(files.size()) - train_count;

        fs::path class_name = class_dir.filename();
        copy_split(files, train_count, train_root / class_name,
                   test_root / class_name);

        total_train += train_count;
        total_test += test_count;

        cout << format("class {}: train {}, test {}", class_name.string(),
                       train_count, test_count)
             << endl;
    }

    cout << format("done: total train {}, total test {}", total_train,
                   total_test)
         << endl;
    return 0;
}
