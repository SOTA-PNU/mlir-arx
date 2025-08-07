#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>

#include "mnist_onnx.hpp"

std::vector<float> load_raw_floats(const std::string& filepath) {
    // 1) 파일 크기 구하기
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    std::streamsize byte_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    size_t num_floats = byte_size / sizeof(float);

    std::vector<float> data(num_floats);
    ifs.read(reinterpret_cast<char*>(data.data()), byte_size);

    return data;
}

std::vector<std::string> list_files_in_dir(const std::string& folder) {
    std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            filenames.emplace_back(entry.path().filename().string());
        }
    }
    return filenames;
}


int main(int argc, char** argv) { 
    auto file_list = list_files_in_dir("/workdir/mlir-arx/test_so/test");
    std::cout << file_list[0] << "\n";
    
    float output[10];
    int progress = 0;
    int success = 0;


    for (const auto& file : file_list) { 
        progress += 1;

        auto path = std::string("/workdir/mlir-arx/test_so/test/") + file;
        int answer = file[0] - '0';
        auto input = load_raw_floats(path);

        // for (int y = 0; y < 28; y++) { 
        //     for (int x = 0; x < 28; x++) {
        //         auto p = input[y * 28 + x];
        //         printf("%d ", p > 0.01);
        //     }
        //     printf("\n");
        // }
        
        main_graph(input.data(), output);
        
        
        float max_val = 0;
        int predicted_answer = 0;
        for(int j = 0; j < 10; j++) {
            if(output[j] > max_val) {
                max_val = output[j];
                predicted_answer = j;
            }
        }
        // printf("Answer : %d, %d \n", answer, predicted_answer);
        // for(int j = 0; j < 10; j++) {
        //     printf("%.3f ", output[j]);
        // }
        // printf("\n\n");

        success += (answer == predicted_answer);
        if (progress % 100 == 0) { 
            printf("Progress [%d %%] : %d / %d\n", success * 100 / progress, success, progress);
        }

        // if (progress > 10) {
        //     break;
        // }
    }

    return 0;
}