#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define DRAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05


__device__ struct XY {
    double x;
    double y;
} XY;

__device__ double d_f_pred;
__device__ int n;
__device__ int count;


void calculate_h_xy(struct XY *h_xy, int m, double h){
    int id = 0;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < m; j++) {
            h_xy[id].x = (i + 1) * h;
            h_xy[id].y = (j + 1) * h;
            id++;
        }
    }
}

void calculate_f(double* host_f, struct XY *h_xy, int n){
    for(int id = 0; id < n; id++) {
        host_f[id] = 1 - (((h_xy[id].x - 0.5) * (h_xy[id].x - 0.5)) +
                       ((h_xy[id].y - 0.5) * (h_xy[id].y - 0.5))) + DRAND;
    }
}

int _ConvertSMVer2Cores(int major, int minor)
{
    typedef struct
    {
        int SM;
        int Cores;
    } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 },{ 0x21, 48 },{ 0x30, 192},{ 0x32, 192},{ 0x35, 192},{ 0x37, 192},{ 0x50, 128},
        { 0x52, 128},{ 0x53, 128},{ 0x60, 64 },{ 0x61, 128},{ 0x62, 128},{   -1, -1 }
    };
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1){
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

int getNumThreads(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        int curr_t =  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        if(curr_t > num_threads)
            num_threads = curr_t;
    }
    return(num_threads);
}

__global__ void device_compute_K(int num_threads, int m, double *K, struct XY *xy)
{
    n = m * m;
    double d_x, d_y;
    for(int i = threadIdx.x; i < n; i += num_threads) {
        for(int j = 0; j < n; j++) {
            d_x = pow(xy[i].x - xy[j].x, 2);
            d_y = pow(xy[i].y - xy[j].y, 2);
            K[i*n + j] = exp(-1 * (d_x + d_y));
            if(i == j)
                K[i*n + j] = K[i*n + j] + 0.01;
        }
    }
}

__device__ void get_total_sum(double *partial_sum, int dummy) {
    if(threadIdx.x == 0) {
        count = dummy;
        if(count % 2 != 0)
            count++;
        for(int i = 1; i < count; i++)
            partial_sum[0] += partial_sum[i];
    }
}

__global__ void device_run_LU(int num_threads, double *K, double *L, double *partial_sum) {
    for(int k = 0; k < n; k++) {
        for(int i = 0; i < n; i++) {
            if(i == k) {
                partial_sum[threadIdx.x] = 0;
                for(int j = threadIdx.x; j < k; j += num_threads) {
                    partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + pow((L[k * n + j]), 2);
                }
                __syncthreads();
                get_total_sum(partial_sum, (num_threads<k)?num_threads:k);
                if(threadIdx.x == 0) {
                    L[i * n + i] = sqrt(K[k * n + k] - partial_sum[0]);
                }
            } else if( i > k) {
                partial_sum[threadIdx.x] = 0;
                for(int j = threadIdx.x; j < k; j += num_threads) {
                    partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + L[i * n + j] * L[k * n + j];
                }
                __syncthreads();
                get_total_sum(partial_sum, (num_threads<k)?num_threads:k);
                if(threadIdx.x == 0) {
                    L[i * n + k] = K[i * n + k] - partial_sum[0];
                    L[i * n + k] /= L[k * n + k];
                }
            } else {
                if(threadIdx.x == 0)
                    L[i * n + k] = 0;
            }
            __syncthreads();
        }
    }
}

__global__ void device_run_solver(int num_threads, double *Y, double *z, double *L, double *partial_sum, double *f)
{
    for(int i = 0; i < n; i++) {
        partial_sum[threadIdx.x] = 0;
        for(int j = threadIdx.x; j < i; j += num_threads) {
            partial_sum[threadIdx.x] += (L[i * n + j] * Y[j]);
        }
        __syncthreads();
        get_total_sum(partial_sum, (num_threads<i)?num_threads:i);
        if(threadIdx.x == 0) {
            Y[i] = (f[i] - partial_sum[0]) / L[i * n + i];
        }
        __syncthreads();
    }
    for(int i = n-1; i >= 0; i--) {
        partial_sum[threadIdx.x] = 0;
        for(int j = n-1-threadIdx.x; j > i; j -= num_threads) {
            partial_sum[threadIdx.x] += (L[j * n + i] * z[j]);
        }
        __syncthreads();
        get_total_sum(partial_sum, (num_threads < (n - 1 - i))?num_threads:(n-1-i));
        if(threadIdx.x == 0) {
            z[i] = (Y[i] - partial_sum[0]) / L[i * n + i];
        }
        __syncthreads();
    }
}

__global__ void device_predict_value(int num_threads, int m, double rstar_x, double rstar_y, double *k, double *z, double *partial_sum, struct XY *xy)
{
    double d_x, d_y;
    for(int i = threadIdx.x; i < n; i += num_threads) {
        d_x = pow(xy[i].x - rstar_x, 2);
        d_y = pow(xy[i].y - rstar_y, 2);
        k[i] = exp(-1 * (d_x + d_y));
    }
    partial_sum[threadIdx.x] = 0.0;
    for(int i = threadIdx.x; i < n; i += num_threads) {
        partial_sum[threadIdx.x] += k[i] * z[i];
    }
    __syncthreads();
    get_total_sum(partial_sum, (num_threads<n)?num_threads:n);
    if(threadIdx.x == 0) {
            d_f_pred = partial_sum[0];
    }
}

int main(int argc, char* argv[]) {
    int m;
    double rstar_x, rstar_y;
    int num_threads = getNumThreads();

    if(argc != 4) {
        printf("Error: Invalid number of input arguments. Required 3.\n");
        return 0;
    } else {
        m = atoi(argv[1]);
        rstar_x = atof(argv[2]);
        rstar_y = atof(argv[3]);
        printf("m = %d \n", m);
        printf("rstarx = %f, rstary = %f\n", rstarx, rstary);
    }
    if(rstar_x < 0 || rstar_x >= m || rstar_y < 0 || rstar_y >= m ) {
        printf("Error: rstar passed is out of range of m \n");
        return 0;
    }

    double h = 1.0 / (double)(m + 1);
    int n = m * m;

    struct XY *h_xy = (struct XY *) malloc(n * sizeof(struct XY));
    calculate_h_xy(h_xy, m, h);
    struct XY *xy;
    cudaMalloc(&xy, (n * sizeof(struct XY)));
    cudaMemcpy(xy, h_xy, n * sizeof(struct XY), cudaMemcpyHostToDevice);

    double *host_f = (double*) malloc(n * sizeof(double));
    calculate_f(host_f, h_xy, n);
    double *f;
    cudaMalloc(&f, (m * m * sizeof(double)));
    cudaMemcpy(f, host_f, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start_overall, end_overall, start_LU, end_LU, start_solver, end_solver;

    double *K;
    cudaMalloc(&K, (n * n * sizeof(double)));
    cudaEventCreate(&start_overall);
    cudaEventCreate(&end_overall);
    cudaEventRecord(start_overall, 0);
    device_compute_K<<<1,num_threads>>>(num_threads, m, K, xy);

    double *L, *partial_sum;
    cudaMalloc(&L, (n * n * sizeof(double)));
    cudaMalloc(&partial_sum, num_threads * sizeof(double));
    cudaEventCreate(&start_LU);
    cudaEventCreate(&end_LU);
    cudaEventRecord(start_LU, 0);
    device_run_LU<<<1,num_threads>>>(num_threads, K, L, partial_sum);
    cudaEventRecord(end_LU, 0);
    cudaEventSynchronize(end_LU);

    cudaEventCreate(&start_solver);
    cudaEventCreate(&end_solver);
    cudaEventRecord(start_solver, 0);
    device_run_solver<<<1,num_threads>>>(num_threads, Y, z, L, partial_sum, f);
    cudaEventRecord(end_solver, 0);
    cudaEventSynchronize(end_solver);

    device_predict_value<<<1,num_threads>>>(num_threads, m, rstar_x, rstar_y, k, z, partial_sum, xy);

    cudaEventRecord(end_overall, 0);
    cudaEventSynchronize(end_overall);

    typeof(d_f_pred) f_pred;
    cudaMemcpyFromSymbol(&f_pred, d_f_pred, sizeof(d_f_pred), 0, cudaMemcpyDeviceToHost);
    printf("The predicted value of f at r_star is %f\n", f_pred);

    float time_overall, time_LU, time_solver;
    cudaEventElapsedTime(&time_overall, start_overall, end_overall);
    cudaEventElapsedTime(&time_LU, start_LU, end_LU);
    cudaEventElapsedTime(&time_solver, start_solver, end_solver);
    cudaEventDestroy(start_overall);
    cudaEventDestroy(end_overall);
    cudaEventDestroy(start_LU);
    cudaEventDestroy(end_LU);
    cudaEventDestroy(start_solver);
    cudaEventDestroy(end_solver);

    printf("Time Taken by LU = %f ms\n", time_LU);
    printf("Time Taken by Solver = %f ms\n", time_solver);
    printf("Overall Time Taken = %f ms\n", time_overall);
    return(0);
}