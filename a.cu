#include<vector>
#include"config.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include <memory>
#include"cublas_v2.h"

#include"smmh2.h"
#include"bin_heap.h"
#include"bloomfilter.h"

#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#define value_t u_int8_t
#define idx_t u_int64_t
#define size_t int
#define pq_idx_t u_int8_t
#define pq_value_t _Float32
#define dist_t _Float32

#define TOPK 10

const int num_vertices = 1000;
const int dim = 128;
const int pq_dim = 128;
const int num_queries = 100;
const int degree = 100;
const int k = 256;

template<class A,class B>
struct KernelPair{
    A first;
    B second;
	
	__device__
	KernelPair(){}


	__device__
    bool operator <(KernelPair& kp) const{
        return first < kp.first;
    }


	__device__
    bool operator >(KernelPair& kp) const{
        return first > kp.first;
    }
};

template<int VALUE_SIZE, int INDEX_SIZE>
struct graph_node {
    value_t values[VALUE_SIZE];
    idx_t indexes[INDEX_SIZE];
};


__device__
dist_t device_l2_distance(value_t* d_data,idx_t idx,value_t* d_query,int qid){
	dist_t ret = 0;
	for(int i = 0;i < dim;++i){
		dist_t diff = d_data[idx * dim + i] - d_query[qid * dim + i];
		ret += diff * diff;
	}
	return ret;
}

__device__
dist_t device_distance(pq_idx_t* d_data,idx_t idx,pq_value_t* pq_table){
    dist_t rest=0;
	for(int i=0;i<dim;i++)
    {
        rest+=pq_table[i*256+d_data[idx*dim+i]];
    }
    return rest;
}

__device__ void computePQTable(
    pq_value_t* d_pq_centroid, // PQ中心表
    value_t* d_query,       // 单个查询向量
    pq_value_t* pq_table,            // PQ table的结果，已预分配内存
    int m,                      // 子空间数量
    int k,                      // 每个子空间的中心数量
    int dim,                    // 查询向量的维度
    int tid)                    
{
    int subvector_dim = dim / m; // 子空间的维度

    for (int subvector_idx = 0; subvector_idx < m; ++subvector_idx) {
        for (int centroid_idx = 0; centroid_idx < k; ++centroid_idx) {
            pq_value_t distance = 0;
            for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
                int query_dim_idx = subvector_idx * subvector_dim + dim_idx;
                int centroid_dim_idx = (subvector_idx * k + centroid_idx) * subvector_dim + dim_idx;
                
                pq_value_t diff = d_query[dim*tid+query_dim_idx] - d_pq_centroid[centroid_dim_idx];
                distance += diff * diff;
            }
            
            pq_table[subvector_idx * k + centroid_idx] = distance;
        }
    }
}


__global__
void independent_search_kernel(pq_idx_t* d_data,value_t* d_query,idx_t* d_result,graph_node<dim,degree>* d_graph,pq_value_t* d_pq_centroid,int num_query){
	const int QUEUE_SIZE = TOPK;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= num_query)
		return;
    pq_value_t pq_table[256*k];
    computePQTable(d_pq_centroid,d_query,pq_table,128,256,128,tid);
    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    BloomFilter<64,6,7> bf;
    //BloomFilter<64,6,3> bf;
    //VanillaList bf;
    KernelPair<dist_t,idx_t> q[QUEUE_SIZE + 2];
    const idx_t start_point = 0;
    dist_t d = device_distance(d_data,start_point,pq_table);
    int heap_size = 1;
	KernelPair<dist_t,idx_t> kp;
	kp.first = d;
	kp.second = start_point;
	smmh2::insert(q,heap_size,kp);
    bf.add(start_point);

    KernelPair<dist_t,idx_t> topk[TOPK + 1];
	int topk_heap_size = 0;
    while(heap_size > 1){
        auto now = smmh2::pop_min(q,heap_size);
		if(topk_heap_size == TOPK && topk[0].first < now.first){
        	break;
        }
       	topk[topk_heap_size++] = now;
		push_heap(topk,topk + topk_heap_size);
        if(topk_heap_size > TOPK){
        	pop_heap(topk,topk + topk_heap_size);
			--topk_heap_size;
		}
		

        for(int i = 0;i < degree;++i){
            auto idx = d_graph[now.second].indexes[i];
            if(bf.test(idx)){
                continue;
			}
            bf.add(idx);
            dist_t d = device_distance(d_data,idx,pq_table);
			KernelPair<dist_t,idx_t> kp;
			kp.first = d;
			kp.second = idx;
			smmh2::insert(q,heap_size,kp);
			if(heap_size >= QUEUE_SIZE + 2){
				smmh2::pop_max(q,heap_size);
			}
        }
    }
	for(int i = 0;i < TOPK;++i){
		auto now = pop_heap(topk,topk + topk_heap_size - i);
		d_result[tid * TOPK + TOPK - 1 - i] = now.second;
	}
}

void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,value_t>>>& queries,int k,\
    std::vector<std::vector<idx_t>>& results,pq_idx_t* h_data,graph_node<dim,degree>* h_graph,pq_value_t* pq_centroid,int num){
    pq_idx_t* d_data;
    value_t* d_query;
    idx_t* d_result;
    pq_value_t* d_pq_centroid;
    graph_node<dim,degree>* d_graph;
    
    
    std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
    memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
    for(int i = 0;i < queries.size();++i){
        for(auto p : queries[i]){
            *(h_query.get() + i * dim + p.first) = p.second;
        }
    }
    std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

    cudaMalloc(&d_data,sizeof(pq_idx_t*) * num * dim);
    cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
    cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
    cudaMalloc(&d_pq_centroid,sizeof(pq_value_t) * 256 * dim);
    cudaMalloc(&d_graph,sizeof(graph_node<dim,degree>) * num);
    
    cudaMemcpy(d_data,h_data,sizeof(value_t) * num * dim,cudaMemcpyHostToDevice);
    cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);
    cudaMemcpy(d_pq_centroid,pq_centroid,sizeof(pq_value_t) * 256 * dim,cudaMemcpyHostToDevice);
    // cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);
    independent_search_kernel<<<10,32>>>(d_data,d_query,d_result,d_graph,d_pq_centroid,queries.size());
    cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);
    results.clear();
    for(int i = 0;i < queries.size();++i){
        std::vector<idx_t> v(TOPK);
        for(int j = 0;j < TOPK;++j)
            v[j] = h_result[i * TOPK + j];
        results.push_back(v);
    }
}






int main() {


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
    astar_multi_start_search_batch(queries, TOPK, results, h_data, h_graph, pq_centroid, num_vertices);

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