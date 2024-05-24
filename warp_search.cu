#include<vector>
#include<fstream>
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

#include<iostream>

// #include"type_def.h"
// #include"graph_node.h"
#include"read_bin.h"
#include"result_save.h"
#include"vanilla_list.h"

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32
#define N_MULTIQUERY 4
#define CRITICAL_STEP 32
#define N_MULTIPROBE 8
#define FINISH_CNT 1
#define N_THREAD_IN_BLOCK 128
//#define __HASH_TEST

const int num_vertices = 1000000;
const int dim = 128;
const int pq_dim = 128;
const int num_queries = 10000;
const int degree = 64;
const int k = 256;

#define TOPK 100
#define QUEUE_SIZE 128




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
    int subvector_dim = (dim+pq_dim-1) / pq_dim; // 子空间的维度
    int step = 32;


	for (int subvector_idx = 0; subvector_idx < pq_dim; ++subvector_idx) {
		for (int centroid_idx = tid; centroid_idx < k; centroid_idx += step) {
			pq_value_t distance = 0;
			for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
				int query_dim_idx = subvector_idx * subvector_dim + dim_idx;
				int centroid_dim_idx = (centroid_idx * pq_dim + subvector_idx) * subvector_dim + dim_idx;
				if(subvector_idx>48&&dim_idx==1)
					continue;
				pq_value_t diff = d_query[query_dim_idx] - d_pq_centroid[centroid_dim_idx];
				distance += diff * diff;
			}
			
			pq_table[subvector_idx * k + centroid_idx] = distance;
		}
	}
	
}

__global__
void warp_independent_search_kernel(pq_idx_t* d_data,value_t* d_query,idx_t* d_result,graph_node<dim,degree>* d_graph,pq_value_t* d_pq_centroid,pq_value_t* d_pq_table, int num_query){
    int bid = blockIdx.x * N_MULTIQUERY;
	const int step = N_THREAD_IN_WARP;
    int tid = threadIdx.x;
	int cid = tid / CRITICAL_STEP;
	int subtid = tid % CRITICAL_STEP;
#define BLOOM_FILTER_BIT64 1500
#define BLOOM_FILTER_BIT_SHIFT 3
#define BLOOM_FILTER_NUM_HASH 20

#ifndef __ENABLE_VISITED_DEL
#define HASH_TABLE_CAPACITY (TOPK*4*16)
#else
#define HASH_TABLE_CAPACITY (TOPK*4*2)
#endif

#ifdef __DISABLE_SELECT_INSERT
#undef HASH_TABLE_CAPACITY
#define HASH_TABLE_CAPACITY (TOPK*4*16+500)
#endif

#ifdef __HASH_TEST
	VanillaList* pbf;
#else
    BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
#endif
    KernelPair<dist_t,idx_t>* q;
    KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;

	// dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
	// q= new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
	// topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
	// pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();



	if(subtid == 0){
		//dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE*2];
		value_t dist_list2[FIXED_DEGREE * N_MULTIPROBE];
		dist_list=dist_list2;
		q = new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
#ifdef __HASH_TEST
		pbf = new VanillaList();
#else
    	pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
#endif

	//pbf = new VanillaList();
	}
    __shared__ int heap_size[N_MULTIQUERY];
	int topk_heap_size;


	__shared__ int finished[N_MULTIQUERY];
	__shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
	__shared__ int index_list_len[N_MULTIQUERY];
	// __shared__ pq_value_t pq_table[N_MULTIQUERY][pq_dim][k];
	pq_value_t (*pq_table)[pq_dim][k] = (pq_value_t (*)[pq_dim][k])d_pq_table;
	
	value_t start_distance;
	__syncthreads();

	computePQTable(d_pq_centroid,d_query+(bid+cid)*dim,&pq_table[bid+cid][0][0],pq_dim,k,dim,subtid);
	__syncthreads();

	value_t tmp[N_MULTIQUERY];
	int j=cid;
	tmp[j] = 0;
	for(int i = subtid;i < dim;i += step){
		tmp[j] += (pq_table[bid+j][i][d_data[i]]); 
	}
	if(bid==1)
		printf("tid:%d,sum:%f\t",tid,tmp[j]);
	for (int offset = 16; offset > 0; offset /= 2){
			tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
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
	int step_num;
	step_num=0;
    while(heap_size[cid] > 1){
		step_num++;
		index_list_len[cid] = 0;
		int current_heap_elements = heap_size[cid] - 1;
		for(int k = 0;k < N_MULTIPROBE && k < current_heap_elements;++k){
			KernelPair<dist_t,idx_t> now;
			if(subtid == 0){
				now = smmh2::pop_min(q,heap_size[cid]);
				// if(bid==0)
				// {
				// 	printf("now:%f,%d\t",now.first,now.second);
				// 	for(int i=0;i<heap_size[0];i++)
				// 		printf("i:%d,idx:%d,dst:%f\t",i,q[i].second,q[i].first);
				// }
// #ifdef __ENABLE_VISITED_DEL
// 				pbf->del(now.second);
// #endif
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
// #ifdef __ENABLE_VISITED_DEL
// 				pbf->add(now.second);
// #endif
				if(topk_heap_size > TOPK){
// #ifdef __ENABLE_VISITED_DEL
// 					pbf->del(topk[0].second);
// #endif
					pop_heap(topk,topk + topk_heap_size);
					--topk_heap_size;
				}
				for(int i = 0;i < degree;++i){
					auto idx = d_graph[now.second].indexes[i];
					if(subtid == 0){
						if(pbf->test(idx)){
							// if(bid==0)
							// 	printf("so big%d\n",idx);
							continue;
						}
// #ifdef __DISABLE_SELECT_INSERT
// 						pbf->add(idx);
// #endif
						index_list[cid][index_list_len[cid]++] = idx;
					}
				}
			}
		}
		if(finished[cid] >= FINISH_CNT)
			break;
		__syncthreads();
		
		int nq=cid;
		for(int i = 0;i < index_list_len[nq];++i){
			value_t tmp = 0;
			for(int j = subtid;j < dim;j += step){
				tmp += pq_table[bid+nq][j][d_data[index_list[nq][i] * dim + j]];
			}
			for (int offset = 16; offset > 0; offset /= 2){
				tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
			}
			if(subtid==0){
				//printf("tmp:%f\n",tmp);
				dist_list[i] = tmp;
				//printf("dist_list:%f\n",dist_list[i]);
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
#if N_MULTIPROBE > 1
				if(pbf->test(kp.second))
					continue;
#endif
				smmh2::insert(q,heap_size[cid],kp);
#ifndef __DISABLE_SELECT_INSERT
				pbf->add(kp.second);
#endif
				if(heap_size[cid] >= QUEUE_SIZE + 2){
// #ifdef __ENABLE_VISITED_DEL
// 					pbf->del(q[2].second);
// #endif
					smmh2::pop_max(q,heap_size[cid]);
				}
			}
		}
		__syncthreads();
		if(bid+cid==1 && subtid == 0)
		{
			//printf("\nquery:%d,step:%d,heapsize:%d\n",bid+cid,step_num,heap_size[cid]);
			//printf("dist:%f,idx:%d\t",q[1].first,q[1].second);
			// for(int i=0;i<heap_size[0];i++)
			// 	printf("i:%d,idx:%d\t",i,q[i].second);
    	}
	}


	if(subtid == 0){
		for(int i = 0;i < TOPK;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[(bid + cid) * TOPK + TOPK - 1 - i] = now.second;
		}
		//printf("\nquery:%d,pbflen:%d\n",bid+cid,pbf->len);
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
	pq_value_t* d_pq_table;
	graph_node<dim,degree>* d_graph;
	cudaFuncSetAttribute(warp_independent_search_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
	
	cudaMalloc(&d_data,sizeof(pq_idx_t) * num * dim);
	cudaMalloc(&d_graph,sizeof(graph_node<dim,degree>) * num);
	cudaMalloc(&d_pq_centroid,sizeof(pq_value_t) * 256 * dim);
	cudaMemcpy(d_data,h_data,sizeof(pq_idx_t) * num * dim,cudaMemcpyHostToDevice);
	cudaMemcpy(d_graph,h_graph,sizeof(graph_node<dim,degree>) * num,cudaMemcpyHostToDevice);
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
	cudaMalloc(&d_pq_table,sizeof(pq_value_t) * pq_dim * 256 * 10000);
	
	cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);


	// warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,32>>>(d_data,d_query,d_result,d_graph,d_pq_centroid,queries.size());
	warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,N_THREAD_IN_BLOCK,sizeof(pq_value_t)  * pq_dim * 256>>>(d_data,d_query,d_result,d_graph,d_pq_centroid,d_pq_table,queries.size());

	cudaDeviceSynchronize();

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
	cudaFree(d_graph);
	cudaFree(d_pq_centroid);
}


int main() {
	cudaSetDevice(0);

    // 内存中保存pq量化数据
    // pq_idx_t* h_data = new pq_idx_t[num_vertices * pq_dim];
    // for (int i = 0; i < num_vertices * pq_dim; ++i) {
    //     h_data[i] = static_cast<pq_idx_t>(rand());
    // }
	u_int64_t npts_u64, nchunks_u64;
    pq_idx_t *h_data = nullptr;
	load_bin<pq_idx_t>("/home/xy/anns-2/ann_search/mini_graph/disk_index_sift_learn_R64_L128_A1.2_pq_compressed.bin", h_data, npts_u64, nchunks_u64);

    
    // vamana图结构，包含完整向量和索引
    // graph_node<dim,degree>* h_graph = new graph_node<dim,degree>[num_vertices];
    // for (int i = 0; i < num_vertices; ++i) {
    //     for (int j = 0; j < dim; ++j)
    //         h_graph[i].values[j] = static_cast<value_t>(rand()); // Random vertex index
    //     for (int j = 0; j < degree; ++j)
    //         h_graph[i].indexes[j] = static_cast<idx_t>(rand()) % num_vertices;
    // }
	std::vector<graph_node<dim,degree>> nodes;
	read_node_bin("/home/xy/anns-2/ann_search/mini_graph/disk_index_sift_learn_R64_L128_A1.2_disk.index", nodes);
	graph_node<dim,degree>* h_graph=nodes.data();
	// for(int i=0;i<degree;i++)
	// {
	// 	std::cout<<"indexs:"<<static_cast<unsigned>(h_graph[0].indexes[i])<<std::endl;
	// }


    // 查询
    // std::vector<std::vector<std::pair<int,value_t>>> queries(num_queries);
    // for (int i = 0; i < num_queries; ++i) {
    //     // Each query has random values for different dimensions
    //     for (int j = 0; j < dim; ++j) {
    //         queries[i].push_back(std::make_pair(j, static_cast<value_t>(rand())));
    //     }
    // }

	float *query = nullptr;
	load_bin<value_t>("/home/xy/anns-2/ann_search/mini_graph/sift_query.fbin", query, npts_u64, nchunks_u64);
	std::vector<std::vector<std::pair<int,value_t>>> queries(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        for (int j = 0; j < dim; ++j) {
            queries[i].push_back(std::make_pair(j, query[i*dim+j]));
        }
    }
	
	


    // pq中心表

    // pq_value_t* pq_centroid=new pq_value_t[256 * dim];
	// for (int i = 0; i < 256 * dim; ++i) {
	// 	pq_centroid[i] = static_cast<pq_value_t>(rand());
	// }

	uint32_t *chunk_offsets = nullptr;
    float *centroid = nullptr;
    float *tables = nullptr;
    float *tables_tr = nullptr;
	size_t num_chunks = pq_dim;
    load_pq_centroid_bin("/home/xy/anns-2/ann_search/mini_graph/disk_index_sift_learn_R64_L128_A1.2_pq_pivots.bin",num_chunks,chunk_offsets,centroid,tables,tables_tr);


	pq_value_t* pq_centroid_tmp=new pq_value_t[256 * dim];
	for (int i = 0; i < 256 * dim; ++i) {
		pq_centroid_tmp[i] = tables[i]+centroid[i%128];
	}

	int subvector_dim = (dim+pq_dim-1) / pq_dim;

	pq_value_t* pq_centroid=new pq_value_t[256 * pq_dim * subvector_dim];

	if(pq_dim==80)
		for(int i=0;i<256;i++)
		{
			for(int j=0;j<pq_dim;j++)
			{
				for(int h=0;h<subvector_dim;h++)
				{
					if(j<48)
						pq_centroid[(i*pq_dim+j)*subvector_dim+h]=pq_centroid_tmp[i*128+j*2+h];
					else
					{
						if(h==0)
							pq_centroid[(i*pq_dim+j)*subvector_dim+h]=pq_centroid_tmp[i*128+j+48];
						else
							pq_centroid[(i*pq_dim+j)*subvector_dim+h]=0;
					}
				}
			}
		}
	else
		pq_centroid=pq_centroid_tmp;



    // 结果保存
    std::vector<std::vector<idx_t>> results;

    // Call the function
    astar_multi_start_search_batch(queries, TOPK, results, h_data, h_graph, pq_centroid, num_vertices);

	saveToCSV(results, "/home/xy/anns-2/ann_search/results.csv");

    // 结果输出
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < results.size(); ++i) {
        std::cout << "Query " << i << ":" << std::endl;
        for (int j = 0; j < results[i].size(); ++j) {
            std::cout << ' '<<results[i][j] << ",";
        }
        std::cout << std::endl;
    }


    return 0;
}
