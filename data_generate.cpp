#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#define value_t u_int8_t
#define idx_t u_int64_t
#define pq_idx_t u_int8_t
#define pq_value_t _Float32

template<size_t VALUE_SIZE, size_t INDEX_SIZE>
struct graph_node {
    value_t values[VALUE_SIZE];
    idx_t indexes[INDEX_SIZE];
};

int main() {
    const int num_vertices = 1000;
    const int dim = 128;
    const int pq_dim = 128;
    const int num_queries = 100;
    const int degree = 100;
    const int TOPK = 10;

    // 内存中保存pq量化数据
    pq_idx_t* h_data = new pq_idx_t[num_vertices * pq_dim];
    for (int i = 0; i < num_vertices * pq_dim; ++i) {
        h_data[i] = static_cast<pq_idx_t>(rand());
    }
    
    // vamana图结构，包含完整向量和索引
    graph_node<dim,degree>* h_graph = new graph_node<dim,degree>[num_vertices];
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < dim; ++j)
            h_graph[i].values[j] = static_cast<value_t>(rand()); // Random vertex index
        for (int j = 0; j < degree; ++j)
            h_graph[i].indexes[j] = static_cast<idx_t>(rand()) % num_vertices;
    }

    // 查询
    std::vector<std::vector<std::pair<int,value_t>>> queries(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        // Each query has random values for different dimensions
        for (int j = 0; j < dim; ++j) {
            queries[i].push_back(std::make_pair(j, static_cast<value_t>(rand())));
        }
    }

    // pq中心表
    pq_value_t* pq_centroid=new pq_value_t[256 * dim];
        for (int i = 0; i < 256 * dim; ++i) {
            pq_centroid[i] = static_cast<pq_value_t>(rand());
        }

    // 结果保存
    std::vector<std::vector<idx_t>> results;

    // Call the function
    // astar_multi_start_search_batch(queries, TOPK, results, h_data, h_graph, 0, num_vertices, dim);

    // 结果输出
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < results.size(); ++i) {
        std::cout << "Query " << i << ":" << std::endl;
        for (int j = 0; j < results[i].size(); ++j) {
            std::cout << results[i][j] << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}