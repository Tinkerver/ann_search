#include <cstdio>
#include <vector>
#include "lightbam.cuh"
#include "gemm.cuh"
#include "graph_node.h"
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdlib.h>
const int num_queues_per_ssd = 128;
const int queue_depth = 128;
const int max_io_size = 4096;
const int iter = 10000;

__device__ static void read_data(uint64_t start_lb, uint64_t num_lb, IoQueuePair *ssdqp, uint64_t *prp1)
{
    uint32_t cid;
    int global_queue_id = blockIdx.x;
    uint64_t global_pos = (uint64_t)global_queue_id * queue_depth;
	uint64_t offset = global_pos * max_io_size;
    uint64_t io_addr = prp1[0] + offset;
    ssdqp[global_queue_id].submit_fence(cid, NVME_OPCODE_READ, io_addr, 0, start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, NVME_RW_LIMITED_RETRY_MASK | (num_lb - 1));
	// printf("qid %d cid %d sq_tail %d\n", global_queue_id, cid, ssdqp[global_queue_id].sq_tail);
    uint32_t status;
    ssdqp[global_queue_id].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

__global__ void test_kernel(IoQueuePair *ssdqp, uint64_t *prp1, void *iobuf)
{
    for (int i = 0; i < iter; i++)
    {
        if (threadIdx.x == 0)
        {
            int num_lbs = max_io_size / AEOLUS_LB_SIZE;
            read_data(blockIdx.x*num_lbs,num_lbs,ssdqp,prp1); // see read_bin.h
            printf("*");
        }
        __syncthreads();
    }
}

int main()
{
    // init ssd controller
	std::vector<Device *> devices{new Device(0)};
	Controller *ctrl = new ControllerDecoupled(devices, num_queues_per_ssd, max_io_size, queue_depth, AEOLUS_DIST_STRIPE, AEOLUS_BUF_PINNED);
	PinnedBuffer *buf = new PinnedBuffer(devices[0], 1ll * num_queues_per_ssd * queue_depth * max_io_size, max_io_size);

    test_kernel<<<100, 32>>>(ctrl->get_io_queue_pair(), buf->get_d_iobuf_phys(), *buf);
    cudaDeviceSynchronize();

    char resourceFilename[100];
    sprintf(resourceFilename, "/sys/bus/pci/devices/0000:%02x:%02x.%x/resource%x", 0x50, 0, 0, 0x1);
    int fd = open(resourceFilename, O_RDWR);
    if (fd < 0) {
        printf("Open resource error, maybe need sudo or you can check whether if %s exists\n", resourceFilename);
        exit(1);
    }

    int sq_size = MAX(AEOLUS_HOST_PGSIZE, queue_depth*NVME_SQ_ENTRY_SIZE);
    size_t size = 2*sq_size*num_queues_per_ssd;
    uint32_t *buffer = (uint32_t *)mmap(NULL, size, PROT_WRITE, MAP_SHARED | MAP_LOCKED , fd, ctrl->qp_phys[0]-0x1b0000000000);
    puts("debug:");
    for (int i = 0; i < 100; i++)
    {
        printf("QP %d:\n", i);
        for (int j = 0; j < 10; j++)
        {
            printf("SQ %d", j);
            for (int k = 0; k < 16; k++)
                printf(" %08x", buffer[2*i*sq_size/4+j*16+k]);
            puts("");
        }
        for (int j = 0; j < 10; j++)
        {
            printf("CQ %d", j);
            for (int k = 0; k < 4; k++)
                printf(" %08x", buffer[(2*i+1)*sq_size/4+j*4+k]);
            puts("");
        }
    }

	delete buf;
	delete ctrl;
	delete devices[0];
    return 0;
}