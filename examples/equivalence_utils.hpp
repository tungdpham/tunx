#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>
#include <iostream>

#include "tensor/tensor.hpp"
#include "tensor/ops.hpp"

namespace tunx {

inline void save_tensor_bin(const Tensor& tensor, const std::string& path) {
    if (tensor.size() == 0) return;
    
    // Create host tensor to save
    Tensor host_tensor = to_host(tensor);
    
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    out.write(reinterpret_cast<const char*>(host_tensor.data_as<void>()), host_tensor.num_bytes());
    if (!out) {
        throw std::runtime_error("Failed to write to file: " + path);
    }
}

inline void load_tensor_bin(Tensor& tensor, const std::string& path) {
    if (tensor.size() == 0) return;
    
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    if (size != tensor.num_bytes()) {
        throw std::runtime_error("File size (" + std::to_string(size) + 
                               ") does not match tensor size (" + 
                               std::to_string(tensor.num_bytes()) + ") for " + path);
    }
    
    // Create host tensor
    Tensor host_tensor(tensor.shape(), tensor.dtype(), getHost());
    
    if (in.read(reinterpret_cast<char*>(host_tensor.data_as<void>()), size)) {
        // Upload to device
        tensor = to_device(host_tensor, tensor.device());
    } else {
        throw std::runtime_error("Failed to read from file: " + path);
    }
}

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

} // namespace tunx
