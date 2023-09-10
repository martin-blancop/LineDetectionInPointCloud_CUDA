#include "hough_images.cuh"
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <utility>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h" // helper utility functions 

#define DEG2RAD 0.017453293f
#define MAX_THREADS 800

extern dim3 block;
using namespace std;

cudaEvent_t c_start, c_stop;

inline void start_time() {
	cudaEventCreate(&c_start);
	cudaEventCreate(&c_stop);
	cudaEventRecord(c_start, 0);
}

inline float stop_time() {
	float elapsedTime = 0;
	cudaEventRecord(c_stop, 0);
	cudaEventSynchronize(c_stop);
	cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
	//if ( VERBOSE )
	printf("GPU execution time: %.3f ms\n", elapsedTime);
	cudaEventDestroy(c_start);
	cudaEventDestroy(c_stop);
	return elapsedTime;
}

__device__ int getIdx_2D_2D()
{
	int blockId = blockIdx.x
		+ blockIdx.y * gridDim.x;

	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;

	return threadId;
}

__device__ int getIdx_1D_2D()
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

 //CUDA kernel
__global__ void HoughTransformKernel(
	unsigned char* img_data,
	int* gpu_image_w,
	int* gpu_image_h,
	size_t* img_offsets,
	unsigned int** accu_ptrs,
	int* gpu_accu_w,
	int* gpu_accu_h,
	int num_images
) {
	unsigned int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x;

	// Obtain each block's image and their dimensions
	size_t img_offset = img_offsets[block_id];
	unsigned char* img_ptr = img_data + img_offset;
	int w = gpu_image_w[block_id];
	int h = gpu_image_h[block_id];

	int rows_per_thread = ((h / MAX_THREADS) > 1 ? (h / MAX_THREADS) : 1);
	int extra = (h > MAX_THREADS ? (h % MAX_THREADS) : 0);
	int start = (thread_id <= extra ? (thread_id * (rows_per_thread + 1)) : (extra * (rows_per_thread + 1) + (thread_id - extra) * rows_per_thread));
	int end = (thread_id <= extra ? start + rows_per_thread + 1 : start + rows_per_thread);

	//printf("thread_id = %d, start = %d, end = %d, rows = %d, extra = %d, w = %d, h = %d, cosa = %d\n", thread_id, start, end, rows_per_thread, extra, w, h, h / MAX_THREADS);

	// Obtain each block's specific accumulator and its dimensions
	unsigned int* accu_ptr = accu_ptrs[block_id];
	int accu_w = gpu_accu_w[block_id];
	int accu_h = gpu_accu_h[block_id];

	double center_x = w / 2;
	double center_y = h / 2;
	double hough_h = ((sqrt(2.0) * (double)(h > w ? h : w)) / 2.0);

	if (start < h)
		for (int i = start; i < end; i++) {

			for (int x = 0; x < w; x++)
			{
				if (img_ptr[(i * w) + x] == 255)
				{
					for (int theta = 0; theta < accu_w; theta++)
					{
						double rho = ((x - center_x) * cos(theta * DEG2RAD)) + ((i - center_y) * sin(theta * DEG2RAD));
						atomicAdd(&(accu_ptr[(int)((round(rho + hough_h) * accu_w)) + theta]), 1);
					}
				}
			}
		}
}


namespace transformImages {

	Hough_I::Hough_I() :accu(0), accu_w(0), accu_h(0), img_w(0), img_h(0)
	{

	}

	Hough_I::~Hough_I() {
		for (unsigned int* accu_ptr : accu) {
			delete[] accu_ptr;
		}
		accu.clear();
	}

	void Hough_I::Transform(vector<unsigned char*>& images, vector <pair<int, int>> dimensions, size_t num_images) {

		start_time();

		int max_h = 0;
		num_img = num_images;
		vector<size_t> image_sizes;
		vector<size_t> accu_sizes;
		size_t total_image_memory = 0;
		size_t total_accu_memory = 0;

		for (int i = 0; i < num_images; ++i) {

			int w = dimensions[i].first;
			int h = dimensions[i].second;

			max_h = (h > max_h ? h : max_h);

			img_w.push_back(w);
			img_h.push_back(h);

			//Create the accu
			double hough = (sqrt(2.0) * (double)(h > w ? h : w)) / 2.0;
			accu_h.push_back(hough * 2.0); // -r -> +r
			accu_w.push_back(180);

			size_t accu_size = sizeof(unsigned int) * accu_h[i] * accu_w[i];
			accu_sizes.push_back(accu_size);
			total_accu_memory += accu_size;
			//cout << "accu size = " << accu_size << ", accu_w = " << accu_w[i] << ", accu_h = " << accu_h[i] << endl;

			size_t image_size = sizeof(char) * w * h;
			image_sizes.push_back(image_size);
			total_image_memory += image_size;
		}

		//cout << "total accu memory = " << total_accu_memory << endl;

		// Allocate memory for the images in the GPU
		unsigned char* gpu_img_data;
		cudaMalloc((void**)&gpu_img_data, total_image_memory);

		// Allocate memory for the offsets that correspond to each image
		size_t* gpu_img_offsets;
		cudaMalloc((void**)&gpu_img_offsets, num_images * sizeof(size_t));

		// Allocate memory for arrays that contain the images dimensions
		int* gpu_image_w;
		cudaMalloc((void**)&gpu_image_w, num_images * sizeof(int));
		int* gpu_image_h;
		cudaMalloc((void**)&gpu_image_h, num_images * sizeof(int));

		// Allocate memory for the accumulator in the GPU
		unsigned int** gpu_accu_ptrs;
		cudaMalloc((void**)&gpu_accu_ptrs, num_images * sizeof(unsigned int*));

		// Allocate memory for arrays that contain the accumulators dimensions
		int* gpu_accu_w;
		cudaMalloc((void**)&gpu_accu_w, num_images * sizeof(int));
		int* gpu_accu_h;
		cudaMalloc((void**)&gpu_accu_h, num_images * sizeof(int));

		size_t image_offset = 0;

		for (size_t i = 0; i < num_images; ++i) {

			cudaMemcpyAsync(gpu_image_w + i, &dimensions[i].first, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_image_h + i, &dimensions[i].second, sizeof(int), cudaMemcpyHostToDevice);

			cudaMemcpyAsync(gpu_accu_w + i, &accu_w[i], sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_accu_h + i, &accu_h[i], sizeof(int), cudaMemcpyHostToDevice);

			cudaMemcpyAsync(gpu_img_data + image_offset, images[i], image_sizes[i], cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_img_offsets + i, &image_offset, sizeof(size_t), cudaMemcpyHostToDevice);

			unsigned int* gpu_accu_data_i;
			cudaMalloc((void**)&gpu_accu_data_i, accu_sizes[i]);
			cudaMemsetAsync(gpu_accu_data_i, 0, accu_sizes[i]);

			cudaMemcpyAsync(gpu_accu_ptrs + i, &gpu_accu_data_i, sizeof(unsigned int*), cudaMemcpyHostToDevice);

			image_offset += image_sizes[i];
		}

		int THREADS = (max_h > MAX_THREADS ? MAX_THREADS : max_h);
		int BLOCKS = num_images;
		size_t SHMEM;

		cout << "Threads_per_block-" << THREADS << " , Blocks-" <<BLOCKS << endl;
		
		HoughTransformKernel <<<BLOCKS, THREADS>>> (
			gpu_img_data,
			gpu_image_w,
			gpu_image_h,
			gpu_img_offsets,
			gpu_accu_ptrs,
			gpu_accu_w,
			gpu_accu_h,
			num_images
			);

		cudaDeviceSynchronize();

		unsigned int** acc = (unsigned int**)malloc(num_images * sizeof(size_t));
		cudaMemcpy(acc, gpu_accu_ptrs, num_images * sizeof(size_t), cudaMemcpyDeviceToHost);

		for (int i = 0; i < num_images; i++) {
			unsigned int* a = (unsigned int*)malloc(accu_sizes[i]);
			cudaMemcpy(a, acc[i], accu_sizes[i], cudaMemcpyDeviceToHost);
			accu.push_back(a);
			cudaFree(acc[i]);
		}

		cudaFree(gpu_image_w);
		cudaFree(gpu_image_h);
		cudaFree(gpu_accu_w);
		cudaFree(gpu_accu_h);
		cudaFree(gpu_img_data);
		cudaFree(gpu_img_offsets);
		cudaFree(gpu_accu_ptrs);

		stop_time();
	}

	vector< pair< pair<int, int>, pair<int, int> > > Hough_I::GetLines(int threshold, int pos)
	{
		unsigned int* a = accu[pos];
		int a_h = accu_h[pos], a_w = accu_w[pos];
		int i_h = img_h[pos], i_w = img_w[pos];

		vector< pair< pair<int, int>, pair<int, int> > > lines;

		if (a == 0)
			return lines;

		for (int rho = 0; rho < a_h; rho++)
		{
			for (int theta = 0; theta < a_w; theta++)
			{
				if ((int)a[(rho * a_w) + theta] >= threshold)
				{
					//Is this point a local maxima (NxN)
					int N = 9;
					N = N / 2;
					int max = a[(rho * a_w) + theta];
					for (int ly = -N; ly <= N; ly++)
					{
						for (int lx = -N; lx <= N; lx++)
						{
							if ((ly + rho >= 0 && ly + rho < a_h) && (lx + theta >= 0 && lx + theta < a_w))
							{
								if ((int)a[((rho + ly) * a_w) + (theta + lx)] > max)
								{
									max = a[((rho + ly) * a_w) + (theta + lx)];
									ly = lx = N + 1;
								}
							}
						}
					}
					if (max > (int)a[(rho * a_w) + theta])
						continue;

					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					if (theta >= 45 && theta <= 135)
					{
						//y = (r - x cos(t)) / sin(t)
						x1 = 0;
						y1 = ((double)(rho - (a_h / 2)) - ((x1 - (i_w / 2)) * cos(theta * DEG2RAD))) / sin(theta * DEG2RAD) + (i_h / 2);
						x2 = i_w - 0;
						y2 = ((double)(rho - (a_h / 2)) - ((x2 - (i_w / 2)) * cos(theta * DEG2RAD))) / sin(theta * DEG2RAD) + (i_h / 2);
					}
					else
					{
						//x = (r - y sin(t)) / cos(t);
						y1 = 0;
						x1 = ((double)(rho - (a_h / 2)) - ((y1 - (i_h / 2)) * sin(theta * DEG2RAD))) / cos(theta * DEG2RAD) + (i_w / 2);
						y2 = i_h - 0;
						x2 = ((double)(rho - (a_h / 2)) - ((y2 - (i_h / 2)) * sin(theta * DEG2RAD))) / cos(theta * DEG2RAD) + (i_w / 2);
					}

					lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1, y1), pair<int, int>(x2, y2)));
				}
			}
		}

		//cout << "lines: " << lines.size() << " ,threshold: " << threshold << "; img dim: w=" << i_w << " h=" << i_h << endl;
		return lines;
	}

	const unsigned int* Hough_I::GetAccu(int* w, int* h, int pos)
	{
		*w = accu_w[pos];
		*h = accu_h[pos];

		return accu[pos];
	}

}