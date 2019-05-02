#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void putPair(int key, int value) {

}

__global__ void kernel_insert(int *keys, int *values, int *hashmap[2]) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;


}

__global__ void kernel_get(int *keys, int *values, int *hashmap[2]) {

}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	cudaMalloc(&hashmap, size * sizeof(int));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;

	int memSize = numKeys * sizeof(int);
	cudaMalloc(&deviceKeys, memSize);
	cudaMalloc(&deviceValues, memSize);

	if (!deviceKeys || !deviceValues)
		return false;

	// load keys and values into VRAM
	cudaMemcpy(deviceKeys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, memSize, cudaMemcpyHostToDevice);

	// check if we need to increase the hashtable's size to reduce the load factor
	if (float(numInsertedPairs + numKeys) / hashtableSize >= MAX_LOAD_FACTOR) {
		hashtableSize = int((numInsertedPairs + numKeys) / MIN_LOAD_FACTOR);
		reshape(hashtableSize);
	}

	// load kernel for inserting pairs into hashtable
	kernel_insert<<< numKeys / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(deviceKeys,
	                                                                    deviceValues,
	                                                                    hashmap);

	// wait for all insertions to finish
	cudaDeviceSynchronize();

	// free device memory
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *deviceKeys, *values;

	int memSize = numKeys * sizeof(int);
	cudaMalloc(&deviceKeys, memSize);
	cudaMallocManaged(&values, memSize);

	if (!deviceKeys || !values)
		return nullptr;

	// load keys and values into VRAM
	cudaMemcpy(deviceKeys, keys, memSize, cudaMemcpyHostToDevice);

	kernel_get<<< numKeys / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(deviceKeys,
	                                                                 values,
	                                                                 hashmap);

	cudaDeviceSynchronize();

	// free device memory
	cudaFree(deviceKeys);

	return values;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return float(numInsertedPairs) / hashtableSize; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
