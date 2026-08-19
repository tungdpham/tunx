#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

#include "graph.h"

struct TensorLife {
    std::string id;
    size_t size;
    int birth;
    int death;
    size_t offset = 0;
};

inline std::vector<TensorLife> extract_forward_lifespans(Graph& graph, const std::vector<std::string>& order) {
    std::vector<TensorLife> lives;
    std::unordered_map<std::string, size_t> active_tensors;
    int current_time = 0;
    
    auto allocate = [&](std::string id, size_t size) {
        active_tensors[id] = lives.size();
        lives.push_back({id, size, current_time, -1, 0});
    };
    
    auto free_t = [&](std::string id) {
        lives[active_tensors[id]].death = current_time;
        active_tensors.erase(id);
    };

    std::unordered_map<ActivationNode*, size_t> ref_count;
    std::unordered_map<ActivationNode*, std::vector<OperationNode*>> act_dependents;
    for (auto& [uuid, op_node] : graph.op_nodes()) {
        for (ActivationNode* dep : op_node.inputs()) {
            act_dependents[dep].push_back(&op_node);
        }
    }

    for (ActivationNode* input : graph.inputs()) {
        ref_count[input] = act_dependents[input].size();
        allocate(input->uuid(), input->size());
    }
    for (ActivationNode* output : graph.outputs()) {
        ref_count[output]++;
    }
    
    for (const auto& op_id : order) {
        current_time++;
        const auto& node = graph.get_op(op_id);
        
        if (node.workspace_req() > 0) {
            allocate(node.uuid() + "_workspace", node.workspace_req());
        }
        
        for (ActivationNode* output : node.outputs()) {
            ref_count[output] += act_dependents[output].size();
            allocate(output->uuid(), output->size());
        }
        
        current_time++;
        
        for (ActivationNode* input : node.inputs()) {
            if (ref_count[input] > 0) {
                ref_count[input]--;
                if (ref_count[input] == 0) {
                    free_t(input->uuid());
                }
            }
        }
        
        if (node.workspace_req() > 0) {
            free_t(node.uuid() + "_workspace");
        }
    }
    
    current_time++;
    for (auto& [id, idx] : active_tensors) {
        lives[idx].death = current_time;
    }
    
    return lives;
}

inline std::vector<TensorLife> extract_forward_backward_lifespans(Graph& graph, const std::vector<std::string>& bw_order) {
    std::vector<std::string> fw_order = bw_order;
    std::reverse(fw_order.begin(), fw_order.end());

    std::vector<TensorLife> lives;
    std::unordered_map<std::string, size_t> active_tensors;
    int current_time = 0;
    
    auto allocate = [&](std::string id, size_t size) {
        if (size == 0) return;
        active_tensors[id] = lives.size();
        lives.push_back({id, size, current_time, -1, 0});
    };
    
    auto free_t = [&](std::string id) {
        if (active_tensors.count(id)) {
            lives[active_tensors[id]].death = current_time;
            active_tensors.erase(id);
        }
    };

    std::unordered_map<ActivationNode*, size_t> ref_count;
    std::unordered_map<ActivationNode*, std::vector<OperationNode*>> act_dependents;
    std::unordered_map<ActivationNode*, OperationNode*> act_deps;
    for (auto& [uuid, op_node] : graph.op_nodes()) {
        for (ActivationNode* dep : op_node.inputs()) {
            act_dependents[dep].push_back(&op_node);
        }
        for (ActivationNode* out : op_node.outputs()) {
            act_deps[out] = &op_node;
        }
    }

    // --- Forward Phase ---
    for (ActivationNode* input : graph.inputs()) {
        ref_count[input] = act_dependents[input].size();
        allocate(input->uuid(), input->size());
    }
    for (ActivationNode* output : graph.outputs()) {
        ref_count[output]++;
    }
    
    for (const auto& op_id : fw_order) {
        current_time++;
        const auto& node = graph.get_op(op_id);
        
        if (node.workspace_req() > 0) {
            allocate(node.uuid() + "_workspace", node.workspace_req());
        }
        
        if (node.residual_mem() > 0) {
            allocate(node.uuid() + "_residual", node.residual_mem());
        }
        
        for (ActivationNode* output : node.outputs()) {
            ref_count[output] += act_dependents[output].size();
            allocate(output->uuid(), output->size());
        }
        
        current_time++;
        
        for (ActivationNode* input : node.inputs()) {
            if (ref_count[input] > 0) {
                ref_count[input]--;
                if (ref_count[input] == 0) {
                    free_t(input->uuid());
                }
            }
        }
        
        if (node.workspace_req() > 0) {
            free_t(node.uuid() + "_workspace");
        }
    }
    
    current_time++;

    // --- Transition Phase ---
    for (ActivationNode* output : graph.outputs()) {
        free_t(output->uuid());
    }

    std::unordered_map<ActivationNode*, size_t> grad_ref_count;
    std::unordered_map<ActivationNode*, size_t> grad_accumulated_count;

    for (ActivationNode* output : graph.outputs()) {
        grad_ref_count[output] = act_deps.count(output) ? 1 : 0;
        allocate(output->uuid() + "_grad", output->size());
    }
    for (ActivationNode* input : graph.inputs()) {
        grad_ref_count[input]++;
    }
    
    // --- Backward Phase ---
    for (const auto& op_id : bw_order) {
        current_time++;
        const auto& node = graph.get_op(op_id);
        
        if (node.workspace_req() > 0) {
            allocate(node.uuid() + "_workspace", node.workspace_req());
        }
        
        for (ActivationNode* input : node.inputs()) {
            if (grad_accumulated_count[input] == 0) {
                allocate(input->uuid() + "_grad", input->size());
                grad_ref_count[input] = (act_deps.count(input) > 0 ? 1 : 0);
                if (std::find(graph.inputs().begin(), graph.inputs().end(), input) != graph.inputs().end()) {
                    grad_ref_count[input]++;
                }
            }
            grad_accumulated_count[input]++;
        }
        
        current_time++;
        
        for (ActivationNode* output : node.outputs()) {
            if (grad_ref_count[output] > 0) {
                grad_ref_count[output]--;
                if (grad_ref_count[output] == 0) {
                    free_t(output->uuid() + "_grad");
                }
            }
        }
        
        if (node.residual_mem() > 0) {
            free_t(node.uuid() + "_residual");
        }
        
        if (node.workspace_req() > 0) {
            free_t(node.uuid() + "_workspace");
        }
    }
    
    current_time++;
    for (auto& [id, idx] : active_tensors) {
        lives[idx].death = current_time;
    }
    
    return lives;
}

inline size_t pack_memory_ffd(std::vector<TensorLife>& lives) {
    std::sort(lives.begin(), lives.end(), [](const TensorLife& a, const TensorLife& b) {
        return a.size > b.size;
    });

    size_t peak_memory = 0;
    std::vector<TensorLife*> placed;

    for (auto& life : lives) {
        if (life.size == 0) continue;
        
        std::vector<std::pair<size_t, size_t>> occupied;
        for (auto* p : placed) {
            if (std::max(life.birth, p->birth) < std::min(life.death, p->death)) {
                occupied.push_back({p->offset, p->offset + p->size});
            }
        }

        std::sort(occupied.begin(), occupied.end());

        size_t current_offset = 0;
        for (const auto& interval : occupied) {
            if (interval.first >= current_offset && interval.first - current_offset >= life.size) {
                break;
            }
            current_offset = std::max(current_offset, interval.second);
        }

        life.offset = current_offset;
        peak_memory = std::max(peak_memory, life.offset + life.size);
        placed.push_back(&life);
    }

    return peak_memory;
}

inline size_t compute_ffd_peak_memory(Graph& graph, const std::vector<std::string>& order, bool backward = false) {
    if (order.empty()) return std::numeric_limits<size_t>::max();
    auto lives = backward ? extract_forward_backward_lifespans(graph, order) : extract_forward_lifespans(graph, order);
    return pack_memory_ffd(lives);
}

