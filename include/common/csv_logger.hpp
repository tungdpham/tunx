/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <chrono>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

#include "logging/logger.hpp"

namespace tunx {

inline std::string csv_timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf {};
  localtime_r(&t, &tm_buf);
  char ts[20];
  std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
  return ts;
}

inline long long get_timestamp_ms() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

class CsvLogger {
private:
  Logger logger_;
  std::vector<std::string> headers_;

public:
  CsvLogger(const std::string &name, const std::string &log_file,
            const std::vector<std::string> &headers);

  void log(const std::unordered_map<std::string, std::string> &row_data);

  void flush() { logger_.flush(); }
};

}  // namespace tunx