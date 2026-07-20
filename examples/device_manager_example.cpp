#include <iostream>
#include <vector>

#include "device/device_manager.hpp"
#include "device/device_type.hpp"

using namespace tunx;
using namespace std;

int main() {
  try {
    cout << "=== Device Manager Example ===" << endl;

    cout << "Initializing devices..." << endl;
    initializeDefaultDevices();

    DeviceManager &manager = DeviceManager::instance();

    vector<DeviceID> device_ids = manager.get_all();
    cout << "Found " << device_ids.size() << " device(s):" << endl;

    for (DeviceID device_id : device_ids) {
      const Device &device = manager.get(device_id);
      cout << "  Device " << device_id << ": " << device.get_name()
           << " (Type: " << device_type_to_string(device.device_type()) << ")" << endl;

      size_t total_mem = device.get_total_memory();
      size_t avail_mem = device.get_available_memory();
      cout << "    Memory - Total: " << total_mem / (1024 * 1024) << " MB, "
           << "Available: " << avail_mem / (1024 * 1024) << " MB" << endl;
    }

    if (manager.has(DeviceType::CPU, 0)) {
      cout << "Testing allocation on CPU device..." << endl;
      const Device &cpu_device = manager.get({DeviceType::CPU, 0});

      size_t test_size = 1024 * 1024;
      void *ptr = cpu_device.allocate_memory(test_size);

      if (ptr != nullptr) {
        cout << "  Successfully allocated " << test_size << " bytes" << endl;

        memset(ptr, 0xAA, test_size);
        char *char_ptr = static_cast<char *>(ptr);
        if (char_ptr[0] == (char)0xAA && char_ptr[test_size - 1] == (char)0xAA) {
          cout << "  Memory access test passed" << endl;
        } else {
          cout << "  Memory access test failed" << endl;
        }

        cpu_device.deallocate_memory(ptr);
        cout << "  Successfully deallocated memory" << endl;
      } else {
        cout << "  Failed to allocate memory" << endl;
      }
    }

    bool found_gpu = false;
    if (manager.has({DeviceType::CUDA, 0})) {
      cout << "Testing allocation on CUDA device " << 0 << "..." << endl;

      auto &device = manager.get({DeviceType::CUDA, 0});

      size_t test_size = 1024 * 1024 * 1024;
      void *ptr = device.allocate_memory(test_size);

      if (ptr != nullptr) {
        cout << "  Successfully allocated " << test_size << " bytes on CUDA" << endl;
        device.deallocate_memory(ptr);
        cout << "  Successfully deallocated CUDA memory" << endl;
      } else {
        cout << "  Failed to allocate CUDA memory" << endl;
      }
      found_gpu = true;
    }

    if (!found_gpu) {
      cout << "No CUDA devices available for testing" << endl;
    }
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}