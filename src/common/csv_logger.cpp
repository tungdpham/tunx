/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "common/csv_logger.hpp"

#include <filesystem>
#include <sstream>

namespace tunx {

CsvLogger::CsvLogger(const std::string &name, const std::string &log_file,
                     const std::vector<std::string> &headers)
    : logger_(name, ""), headers_(headers) {
  if (log_file != "") {
    std::filesystem::path p(log_file);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    logger_.set_log_file(log_file);
  }

  logger_.set_pattern("%v");

  if (!headers_.empty()) {
    std::ostringstream header_stream;
    for (size_t i = 0; i < headers_.size(); ++i) {
      if (i > 0) header_stream << ",";
      header_stream << headers_[i];
    }
    logger_.info(header_stream.str());
    logger_.flush();
  }
}

void CsvLogger::log(const std::unordered_map<std::string, std::string> &row_data) {
  std::ostringstream row_stream;
  for (size_t i = 0; i < headers_.size(); ++i) {
    if (i > 0) row_stream << ",";
    auto it = row_data.find(headers_[i]);
    if (it != row_data.end()) {
      row_stream << it->second;
    }
  }
  logger_.info(row_stream.str());
  logger_.flush();
}

}  // namespace tunx
