#pragma once

#include <memory>
#include <string>

namespace tunx {
class Graph;

class NodeImpl {
public:
  explicit NodeImpl(Graph *graph, const std::string &uid = "")
      : graph_(graph),
        uid_(uid) {}

  ~NodeImpl() = default;

  Graph *graph() const { return graph_; }

  const std::string &uid() const { return uid_; }
  void set_uid(const std::string &uid) { uid_ = uid; }

private:
  Graph *graph_;
  std::string uid_;
};

using Node = std::shared_ptr<NodeImpl>;

}  // namespace tunx