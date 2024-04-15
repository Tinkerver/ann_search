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
#include"blocked_bloomfilter.h"

#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#include "lightbam.cuh"
#include "gemm.cuh"

#define pq_idx_t u_int8_t
#define pq_value_t _Float32
#define value_t u_int8_t
#define idx_t u_int64_t
#define size_t int
#define pq_value_t _Float32
#define dist_t _Float32

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32
#define N_MULTIQUERY 1
#define CRITICAL_STEP (N_THREAD_IN_WARP/N_MULTIQUERY)
#define N_MULTIPROBE 1
#define FINISH_CNT 1

const int num_vertices = 1000;
const int dim = 128;
const int pq_dim = 128;
const int num_queries = 100;
const int degree = 100;
const int k = 256;
const int num_queues_per_ssd = 128;
const int queue_depth = 128;
const int max_io_size = 4096;

#define TOPK 100


template<size_t VALUE_SIZE, size_t INDEX_SIZE>
struct graph_node {
    value_t values[VALUE_SIZE];
    idx_t indexes[INDEX_SIZE];
};

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

__device__ void computePQTable(
    pq_value_t* d_pq_centroid,  // PQ中心表,在GM中
    value_t* d_query,           // 单个查询向量
    pq_value_t* pq_table,       // PQ table的结果，已预分配内存，在shared_memory中
    int pq_dim,                      // 子空间数量
    int k,                      // 每个子空间的中心数量
    int dim,                    // 查询向量的维度
    int tid)                    
{
    int subvector_dim = dim / pq_dim; // 子空间的维度
    int step = 32;


	for(int i = 0;i < N_MULTIQUERY;++i){
		for (int subvector_idx = 0; subvector_idx < pq_dim; ++subvector_idx) {
			for (int centroid_idx = tid; centroid_idx < k; centroid_idx += step) {
				pq_value_t distance = 0;
				for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
					int query_dim_idx = i * dim + subvector_idx * subvector_dim + dim_idx;
					int centroid_dim_idx = (subvector_idx * k + centroid_idx) * subvector_dim + dim_idx;
					
					pq_value_t diff = d_query[query_dim_idx] - d_pq_centroid[centroid_dim_idx];
					distance += diff * diff;
				}
				
				pq_table[i * dim * k * pq_dim + subvector_idx * k + centroid_idx] = distance;
			}
		}
	}
}

__device__ static void read_data(uint64_t start_lb, uint64_t num_lb, IoQueuePair *ssdqp, uint64_t *prp1)
{
    uint32_t cid;
	// otherwise require cross-block synchronization
	assert(blockIdx.x < num_queues_per_ssd);
	assert(max_io_size <= AEOLUS_HOST_PGSIZE * 2);
    int global_queue_id = blockIdx.x;
    uint64_t global_pos = (uint64_t)global_queue_id * queue_depth;
	uint64_t offset = global_pos * max_io_size;
    uint64_t io_addr = prp1[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
	offset += AEOLUS_HOST_PGSIZE;
    uint64_t io_addr2 = prp1[offset / AEOLUS_DEVICE_PGSIZE] + offset % AEOLUS_DEVICE_PGSIZE;
    ssdqp[global_queue_id].submit(cid, NVME_OPCODE_READ, io_addr, io_addr2, start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, NVME_RW_LIMITED_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[global_queue_id].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

__global__
void warp_independent_search_kernel(pq_idx_t* d_data,value_t* d_query,idx_t* d_result,pq_value_t* d_pq_centroid, int num_query, IoQueuePair *ssdqp, uint64_t *prp1, void *iobuf){
	const int QUEUE_SIZE = TOPK;
    int bid = blockIdx.x * N_MULTIQUERY;
	const int step = N_THREAD_IN_WARP;
    int tid = threadIdx.x;
	int cid = tid / CRITICAL_STEP;
	int subtid = tid % CRITICAL_STEP;
#define BLOOM_FILTER_BIT64 8
#define BLOOM_FILTER_BIT_SHIFT 3
#define BLOOM_FILTER_NUM_HASH 7

#ifndef __ENABLE_VISITED_DEL
#define HASH_TABLE_CAPACITY (TOPK*4*16)
#else
#define HASH_TABLE_CAPACITY (TOPK*4*2)
#endif

#ifdef __DISABLE_SELECT_INSERT
#undef HASH_TABLE_CAPACITY
#define HASH_TABLE_CAPACITY (TOPK*4*16+500)
#endif

    BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
    KernelPair<dist_t,idx_t>* q;
    KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(subtid == 0){
		dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
		q= new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    	pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();

	//pbf = new VanillaList();
	}
    __shared__ int heap_size[N_MULTIQUERY];
	int topk_heap_size;


	__shared__ int finished[N_MULTIQUERY];
	__shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
	__shared__ char index_list_len[N_MULTIQUERY];
	// __shared__ pq_value_t pq_table[N_MULTIQUERY][pq_dim][k];
	extern __shared__ pq_value_t dynamic_shared_memory[];
	pq_value_t (*pq_table)[pq_dim][k] = (pq_value_t (*)[pq_dim][k])dynamic_shared_memory;
	
	value_t start_distance;
	__syncthreads();

	computePQTable(d_pq_centroid,d_query,&pq_table[0][0][0],pq_dim,k,dim,tid);
	__syncthreads();

	value_t tmp[N_MULTIQUERY];
	for(int j = 0;j < N_MULTIQUERY;++j){
		tmp[j] = 0;
		for(int i = tid;i < dim;i += step){
			tmp[j] += (pq_table[j][i][d_data[i]]) * (pq_table[j][i][d_data[i]]); 

		}
		for (int offset = 16; offset > 0; offset /= 2){
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
		}
	}
	if(subtid == 0){
		start_distance = tmp[cid];
	}
	__syncthreads();
	
	if(subtid == 0){
    	heap_size[cid] = 1;
		topk_heap_size = 0;
		finished[cid] = false;
		dist_t d = start_distance;
		KernelPair<dist_t,idx_t> kp;
		kp.first = d;
		kp.second = 0;
		smmh2::insert(q,heap_size[cid],kp);
		pbf->add(0);
	}
	__syncthreads();
    while(heap_size[cid] > 1){
		index_list_len[cid] = 0;
		int current_heap_elements = heap_size[cid] - 1;
		for(int k = 0;k < N_MULTIPROBE && k < current_heap_elements;++k){
			KernelPair<dist_t,idx_t> now;
			if(subtid == 0){
				now = smmh2::pop_min(q,heap_size[cid]);
#ifdef __ENABLE_VISITED_DEL
				pbf->del(now.second);
#endif
				if(k == 0 && topk_heap_size == TOPK && (topk[0].first <= now.first)){
					++finished[cid];
				}
			}
			__syncthreads();
			if(finished[cid] >= FINISH_CNT)
				break;
			if(subtid == 0){
				topk[topk_heap_size++] = now;
				push_heap(topk,topk + topk_heap_size);
#ifdef __ENABLE_VISITED_DEL
				pbf->add(now.second);
#endif
				if(topk_heap_size > TOPK){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(topk[0].second);
#endif
					pop_heap(topk,topk + topk_heap_size);
					--topk_heap_size;
				}
				int num_lbs = max_io_size / AEOLUS_LB_SIZE;
				read_data(now.second*num_lbs,num_lbs,ssdqp,prp1);
				graph_node<dim,degree> *now_node = (graph_node<dim,degree>*)(iobuf+1ll*blockIdx.x*queue_depth*max_io_size);
				for(int i = 0;i < degree;++i){
					auto idx = now_node->indexes[i];
					if(subtid == 0){
						if(pbf->test(idx)){
							continue;
						}
#ifdef __DISABLE_SELECT_INSERT
						pbf->add(idx);
#endif
						index_list[cid][index_list_len[cid]++] = idx;
					}
				}
			}
		}
		if(finished[cid] >= FINISH_CNT)
			break;
		__syncthreads();

		for(int nq = 0;nq < N_MULTIQUERY;++nq){
			for(int i = 0;i < index_list_len[nq];++i){
				value_t tmp = 0;
				for(int j = tid;j < dim;j += step){
					tmp += pq_table[nq][j][d_data[index_list[nq][i] * dim + j]]*pq_table[nq][j][d_data[index_list[nq][i] * dim + j]];
				}
				for (int offset = 16; offset > 0; offset /= 2){
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
				}
				if(tid == nq * CRITICAL_STEP){
					dist_list[i] = tmp;
				}
			}
		}

		__syncthreads();

		if(subtid == 0){
			for(int i = 0;i < index_list_len[cid];++i){
				dist_t d = dist_list[i];
				KernelPair<dist_t,idx_t> kp;
				kp.first = d;
				kp.second = index_list[cid][i];

				if(heap_size[cid] >= QUEUE_SIZE + 1 && q[2].first < kp.first){
					continue;
				}
#ifdef __ENABLE_MULTIPROBE_DOUBLE_CHECK
				if(pbf->test(kp.second))
					continue;
#endif
				smmh2::insert(q,heap_size[cid],kp);
#ifndef __DISABLE_SELECT_INSERT
				pbf->add(kp.second);
#endif
				if(heap_size[cid] >= QUEUE_SIZE + 2){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(q[2].second);
#endif
					smmh2::pop_max(q,heap_size[cid]);
				}
			}
		}
		__syncthreads();
    }

	if(subtid == 0){
		for(int i = 0;i < TOPK;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[(bid + cid) * TOPK + TOPK - 1 - i] = now.second;
		}
		delete[] q;
		delete[] topk;
    	delete pbf;
    	delete[] dist_list;
	}
}


static void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,value_t>>>& queries,int k,\
	std::vector<std::vector<idx_t>>& results,pq_idx_t* h_data,graph_node<dim,degree>* h_graph,pq_value_t* pq_centroid,int num){
	pq_idx_t* d_data;
	value_t* d_query;
	idx_t* d_result;
	pq_value_t* d_pq_centroid;
	// graph_node<dim,degree>* d_graph;
	
	cudaMalloc(&d_data,sizeof(pq_idx_t*) * num * dim);
	// cudaMalloc(&d_graph,sizeof(graph_node<dim,degree>) * num);
	cudaMalloc(&d_pq_centroid,sizeof(pq_value_t) * 256 * dim);
	cudaMemcpy(d_data,h_data,sizeof(pq_idx_t*) * num * dim,cudaMemcpyHostToDevice);
	// cudaMemcpy(d_graph,h_graph,sizeof(graph_node<dim,degree>) * num,cudaMemcpyHostToDevice);
	cudaMemcpy(d_pq_centroid,pq_centroid,sizeof(pq_value_t) * 256 * dim,cudaMemcpyHostToDevice);


	std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
	memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
	for(int i = 0;i < queries.size();++i){
		for(auto p : queries[i]){
			*(h_query.get() + i * dim + p.first) = p.second;
		}
	}
	std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

	cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
	cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
	
	cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);

	// init ssd controller
	std::vector<Device *> devices{new Device(0)};
	Controller *ctrl = new ControllerDecoupled(devices, num_queues_per_ssd, max_io_size, queue_depth, AEOLUS_DIST_STRIPE, AEOLUS_BUF_PINNED);
	PinnedBuffer *buf = new PinnedBuffer(devices[0], 1ll * num_queues_per_ssd * queue_depth * max_io_size, max_io_size);

	// warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,32>>>(d_data,d_query,d_result,d_graph,d_pq_centroid,queries.size());
	warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,32,sizeof(pq_value_t) * N_MULTIQUERY * pq_dim * 256>>>(d_data,d_query,d_result,d_pq_centroid,queries.size(), ctrl->get_io_queue_pair(), buf->get_d_prp_phys(), *buf);

	cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);

	results.clear();
	for(int i = 0;i < queries.size();++i){
		std::vector<idx_t> v(TOPK);
		for(int j = 0;j < TOPK;++j)
			v[j] = h_result[i * TOPK + j];
		results.push_back(v);
	}
	cudaFree(d_data);
	cudaFree(d_query);
	cudaFree(d_result);
	// cudaFree(d_graph);
	cudaFree(d_pq_centroid);
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
