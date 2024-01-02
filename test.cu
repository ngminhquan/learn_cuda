#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include<cstring>
#include<time.h>

// compare two arrays
void array_comparison(int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different!!\n");
            return;
        }
    }
    printf("Arrays are same!!\n");
}

//common error handling
#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}
/*
__global__ void print_details () {
    printf("threadIdx.x: %d,threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d,blockIdx.y: %d, blockIdx.z: %d, gridDim.x: %d, gridDim.y: %d \n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y);
}

int main() {
    //number of threads in each dimension for grid
    int nx, ny, nz;
    nx = 4;
    ny = 4;
    nz = 4;
    dim3 block(2, 2, 2);
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);
    print_details<<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
*/

/*
__global__ void unique_idx_calc_threadIdx(int *input)
{
    int tid = threadIdx.x;
    printf("threadIdx: %d, value: %d\n", tid, input[tid]);
}
int main()
{
    int array_size = 8;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33 };
    for (int i = 0; i < array_size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n\n");
    int * d_data;
    cudaMalloc((void **)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(8);
    dim3 grid(1);
    unique_idx_calc_threadIdx<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    
    cudaDeviceReset();
    return 0;
}
*/

/*
__global__ void unique_gid_cal_2d_2d(int *input)
{
    int tid = blockDim.x * threadIdx.x + threadIdx.y;
    int num_thread_in_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_thread_in_block;

    int num_thread_in_row = num_thread_in_block * gridDim.x;
    int row_offset = num_thread_in_row * blockIdx.y;

    int gid = tid + block_offset + row_offset;
    printf("blockIdx.x: %d, blockIdx.y: %d, tid: %d, gid: %d - value: %d\n", blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}
int main()
{
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33,22,43,56,4,76,81,94,32};
    
    for (int i = 0; i < array_size; i++)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n\n");
    
    int *d_data;
    cudaMalloc((void **)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2);
    dim3 grid(2,2);
    unique_gid_cal_2d_2d<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
*/

//memory transfer from host to device
/*
__global__ void mem_trs_test(int * input, int size) {
    int num_thread_in_block = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_offset = num_thread_in_block * blockIdx.x + num_thread_in_block * blockIdx.y * blockDim.x + num_thread_in_block * blockIdx.z * blockDim.x * blockDim.y;
    int gid = tid + block_offset;
    if (gid < size)
        printf("tid: %d, gid: %d, value: %d\n", threadIdx.x, gid, input[gid]);
}
int main() {
    int size = 64;
    int byte_size = size * sizeof(int);

    int * h_input;
    h_input = (int *)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++){
        h_input[i] = (int)(rand() & 0xff);
    }

    int *d_input;
    cudaMalloc((void **)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2,2);
    dim3 grid(2,2,2);
    mem_trs_test<<<grid, block>>>(d_input, size);
    cudaDeviceSynchronize();
    cudaFree(d_input);
    free(h_input);

    cudaDeviceReset();
    return 0;
}
*/


/*
__global__ void sum_array_gpu(int * a, int * b, int * c, int size){
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid<size) {
        c[gid] = a[gid] + b[gid];
    }
}

void sum_array_cpu(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; i++){
        c[i] = a[i] + b[i];
    }
}

int main() {
    int size = 2^1000000;
    int block_size = 128;
    int byte_size = size * sizeof(int);
    // host pointer
    int * h_a, * h_b, *gpu_result, *h_c;
    h_a = (int *)malloc(byte_size);
    h_b = (int *)malloc(byte_size);
    h_c = (int *)malloc(byte_size);
    gpu_result = (int *)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() & 0xff);
    }
    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)(rand() & 0xff);
    }
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, size);
    cpu_end = clock();

    memset(gpu_result, 0, byte_size);
    // device pointer
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, byte_size);
    cudaMalloc((void **) &d_b, byte_size);
    cudaMalloc((void **) &d_c, byte_size);

    //transfer memory to device
    clock_t htod_start, htod_end;
    htod_start = clock();
    cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);
    htod_end = clock();
    // launching grid
    dim3 block(block_size);
    dim3 grid((size/block.x) + 1);

    // memory transfer back to host
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    cudaMemcpy(gpu_result, d_c, byte_size, cudaMemcpyDeviceToHost);
    dtoh_end = clock();
    // array comparison
    array_comparison(gpu_result, h_c, size);

    //Timing comparison
    printf("array sum CPU execution time: %4.6f\n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
    printf("array sum GPU execution time: %4.6f\n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
    printf("htod execution time: %4.6f\n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
    printf("dtoh execution time: %4.6f\n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
    printf("total GPU execution time: %4.6f\n", (double)((double)(gpu_end+htod_end+dtoh_end - gpu_start-htod_start-dtoh_start) / CLOCKS_PER_SEC));

    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    free(gpu_result);
    free(h_a);
    free(h_b);

    cudaDeviceReset();
    return 0;
}
*/

//Device properties
/*
void query_device() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if(count == 0){
        printf("No CUDA support device found\n");

    }
    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Device %d: %s\n", devNo, iProp.name);
    printf("Number of multiprocessors:                  %d\n", iProp.multiProcessorCount);
    printf("Clock rate:                 %d\n", iProp.clockRate);
    printf("Compute capability:                         %d.%d\n", iProp.major, iProp.minor);
    printf("Total amount of global memory:              %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
    printf("Total amount of constant memory:            %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block:    %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("Total amount of shared memory per MP:       %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
}

int main() {
    query_device();
    return 0;
}
//warp divergence
/*
__global__ void code_without_divergence() {
    int gid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = gid / 32;
    float a, b;
    a = b = 0;
    if(warp_id % 2 == 0){
        a = 100.00;
        b = 50.00;
    }
    else {
        a = 200.00;
        b = 75.00;
    }
}
__global__ void code_divergence()
{
    int gid = threadIdx.x + threadIdx.y * blockDim.x;
    float a, b;
    a = b = 0;
    if (gid % 2 == 0)
    {
        a = 100.00;
        b = 50.00;
    }
    else
    {
        a = 200.00;
        b = 75.00;
        
    }
}

int main(int argc, char** argv){
    int size = 1 << 22;
    dim3 block(128);
    dim3 grid((size + block.x - 1) / block.x);

    code_without_divergence<<<grid, block>>>();
    cudaDeviceSynchronize();

    code_divergence<<<grid, block>>>();
    cudaDeviceSynchronize;
    cudaDeviceReset();
    return 0;
}
*/

__global__ void test(){
    int a, b;
    if (threadIdx.x % 2 == 0)
    {
        a = 50;
        b = 100;
    }
    else{
        a = 60;
        b = 70;
    }
}
int main() {
    test<<<12, 1>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}