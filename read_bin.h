#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include"type_def.h"
#include"graph_node.h"


void load_pq_centroid_bin(const char *file_path, size_t num_chunks,uint32_t *&chunk_offsets,float *&centroid,\
    float *&tables,float *&tables_tr);

template <typename T>
inline void load_aligned_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t &rounded_dim);

template <typename T>
inline void load_aligned_bin_impl(std::basic_istream<char> &reader, size_t actual_file_size, T *&data, size_t &npts,
                                  size_t &dim, size_t &rounded_dim);

template <typename T>
inline void load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                     size_t offset = 0);

template <typename T>
inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0);

template <typename T>
inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim, size_t file_offset = 0);

int read_node_bin(std::string filename,std::vector<graph_node<128,64>> nodes) {
    const size_t HeaderSize = 4096; // 文件头部大小
    const size_t BlockSize = 4096; // 对齐块大小
    const size_t NodesPerBlock = 5; // 每个块中的节点数
    // const std::string filename = "disk_index_sift_learn_R64_L128_A1.2_disk.index"; // 文件路径

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    // 跳过头部
    file.seekg(HeaderSize, std::ios::beg);

    
    while (file) { // 检查流的状态
        for (size_t i = 0; i < NodesPerBlock && file; ++i) {
            graph_node<128,64> node;
            file.read(reinterpret_cast<char*>(node.values), sizeof(node.values));
            // 跳过一个int
            file.seekg(sizeof(int), std::ios::cur);
            file.read(reinterpret_cast<char*>(node.indexes), sizeof(node.indexes));

            if (!file) {
                // 如果在读取过程中遇到文件结束或读取失败，退出循环
                break;
            }
            nodes.push_back(node);
        }
        // 跳到下一个4KB对齐的位置，为了简化代码，这里直接计算下一个位置
        file.seekg(HeaderSize + (nodes.size() / NodesPerBlock) * BlockSize, std::ios::beg);
    }

    file.close();

    std::cout<<"load node size"<<nodes.size()<<std::endl;

    return 0;
}

void load_pq_centroid_bin(const char *file_path, size_t num_chunks,uint32_t *&chunk_offsets,float *&centroid,\
    float *&tables,float *&tables_tr)
{

    uint64_t nr, nc;
    std::string pq_table_file(file_path);
    std::unique_ptr<size_t[]> file_offset_data;

    load_bin<size_t>(pq_table_file, file_offset_data, nr, nc);

    bool use_old_filetype = false;

    if (nr != 4 && nr != 5)
    {
        std::cout << "Error reading pq_pivots file " << pq_table_file
                      << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting " << 4
                      << " or " << 5;
    }

    if (nr == 4)
    {
        std::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                      << " " << file_offset_data[3] << std::endl;
    }
    else if (nr == 5)
    {
        use_old_filetype = true;
        std::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                      << " " << file_offset_data[3] << file_offset_data[4] << std::endl;
    }

    load_bin<float>(pq_table_file, tables, nr, nc, file_offset_data[0]);

    if ((nr != NUM_PQ_CENTROIDS))
    {
        std::cout << "Error reading pq_pivots file " << pq_table_file << ". file_num_centers  = " << nr
                      << " but expecting " << NUM_PQ_CENTROIDS << " centers";
    }

    int ndims=nc;


    load_bin<float>(pq_table_file, centroid, nr, nc, file_offset_data[1]);

    if ((nr != ndims) || (nc != 1))
    {
        std::cout << "Error reading centroids from pq_pivots file " << pq_table_file << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting " << ndims << " entries in 1 dimension.";

    }

    int chunk_offsets_index = 2;
    if (use_old_filetype)
    {
        chunk_offsets_index = 3;
    }
    load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);


    if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0))
    {
        std::cout << "Error loading chunk offsets file. numc: " << nc << " (should be 1). numr: " << nr
                      << " (should be " << num_chunks + 1 << " or 0 if we need to infer)" << std::endl;
    }

    int n_chunks = nr - 1;
    std::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << ndims
                  << ", #chunks: " << n_chunks << std::endl;


    // alloc and compute transpose
    tables_tr = new float[256 * ndims];
    for (size_t i = 0; i < 256; i++)
    {
        for (size_t j = 0; j < ndims; j++)
        {
            tables_tr[j * 256 + i] = tables[i * ndims + j];
        }
    }
}


template <typename T>
inline void load_aligned_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t &rounded_dim)
{
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    
    std::cout << "Reading (with alignment) bin file " << bin_file << " ..." << std::flush;
    reader.open(bin_file, std::ios::binary | std::ios::ate);

    uint64_t fsize = reader.tellg();
    reader.seekg(0);
    load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);

}

template <typename T>
inline void load_aligned_bin_impl(std::basic_istream<char> &reader, size_t actual_file_size, T *&data, size_t &npts,
                                  size_t &dim, size_t &rounded_dim)
{
    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::cout << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
    }
    // rounded_dim = ROUND_UP(dim, 8);
    rounded_dim = dim;
    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << ", aligned_dim = " << rounded_dim << "... "
                  << std::flush;
    size_t allocSize = npts * rounded_dim * sizeof(T);
    std::cout << "allocating aligned memory of " << allocSize << " bytes... " << std::flush;
    //alloc_aligned(((void **)&data), allocSize, 8 * sizeof(T));
    data = aligned_alloc(8 * sizeof(T), allocSize);
    std::cout << "done. Copying data to mem_aligned buffer..." << std::flush;

    for (size_t i = 0; i < npts; i++)
    {
        reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    std::cout << " done." << std::endl;
}

template <typename T>
inline void load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                     size_t offset)
{
    T *ptr;
    load_bin<T>(bin_file, ptr, npts, dim, offset);
    data.reset(ptr);
}

template <typename T>
inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset)
{
    std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);


    std::cout << "Opening bin file " << bin_file.c_str() << "... " << std::endl;
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    reader.seekg(0);
    load_bin_impl<T>(reader, data, npts, dim, offset);


    std::cout << "done." << std::endl;
}

template <typename T>
inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim, size_t file_offset)
{
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    data = new T[npts * dim];
    reader.read((char *)data, npts * dim * sizeof(T));
}