// #include <cuda_runtime.h>
// #include <iostream>

// int main() {
//     int nDevices;
//     cudaGetDeviceCount(&nDevices);
//     for (int i = 0; i < nDevices; i++) {
//         cudaDeviceProp prop;
//         cudaGetDeviceProperties(&prop, i);
//         std::cout << "Device Number: " << i << std::endl;
//         std::cout << "  Device name: " << prop.name << std::endl;
//         std::cout << "  Maximum shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
//     }

//     return 0;
// }

#include<stdio.h>
__global__ void myKernel() {
    //extern __shared__ int sharedArray[]; // 动态共享内存数组的声明
    int tid = threadIdx.x;
    printf("%d\n",tid);
    // for(int i=tid;i<1024;i+=32)
    // {
    //     sharedArray[i]=1;
    //     printf("%d:%d\n",i,sharedArray[i]);
    // }

}

int main(){
    //int size = sizeof(int) * 1024; // N是你希望分配的整数数量
    myKernel<<<1, 32>>>(); // 内核调用，动态指定共享内存大小
    return 0;
}
