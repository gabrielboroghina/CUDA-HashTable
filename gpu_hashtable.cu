#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>

#include "gpu_hashtable.hpp"

#define HASH_A 823117
#define HASH_B 3452434812973

/** Get the hash of a value and reduce it in the range [0, limit) */
__device__ int getHash(int data, int limit) {
	return ((long long) abs(data) * HASH_A) % HASH_B % limit;
}

/** CUDA kernel which inserts a single (key, value) pair (selected based on
 *  block id & thread id) in the hash table.
 */
__global__ void kernel_insert(int *keys, int *values, int numEntries, hash_table hashmap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check thread index against task's upper bound
	if (idx >= numEntries) return;

	int oldKey, newKey;
	newKey = keys[idx];

	// compute hash value for the key
	int hash = getHash(newKey, hashmap.size);

	// ranges where to search an empty position
	int rangeBegin[2] = {hash, 0};
	int rangeEnd[2] = {hashmap.size, hash};

	for (int r = 0; r <= 1; r++)
		for (int i = rangeBegin[r]; i < rangeEnd[r]; i++) {
			// try slot 0
			oldKey = atomicCAS(&hashmap.map[0][i].key, KEY_INVALID, newKey);

			if (oldKey == KEY_INVALID || oldKey == newKey) {
				// the position was free (or was containing the same key)
				// only the current thread can enter here because this slot was acquired atomically
				// by the current thread (if oldKey == KEY_INVALID) or the slot was already
				// containing newKey (and no other thread can try to insert this key)

				// we can safely set the pair's value
				hashmap.map[0][i].value = values[idx];
				return;
			} else {
				// try slot 1
				oldKey = atomicCAS(&hashmap.map[1][i].key, KEY_INVALID, newKey);

				if (oldKey == KEY_INVALID || oldKey == newKey) {
					hashmap.map[1][i].value = values[idx];
					return;
				}
			}
		}
}

/** CUDA kernel which searches the value of a single key (selected based on the
 *  block id & thread id) in hash table.
 */
__global__ void kernel_get(int *keys, int *values, int numEntries, hash_table hashmap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check thread index against task's upper bound
	if (idx >= numEntries) return;

	int key = keys[idx];

	// compute hash value for the key
	int hash = getHash(keys[idx], hashmap.size);

	// ranges where to search the desired pair
	int rangeBegin[2] = {hash, 0};
	int rangeEnd[2] = {hashmap.size, hash};

	for (int r = 0; r <= 1; r++) {
		for (int i = rangeBegin[r]; i < rangeEnd[r]; i++) {
			if (hashmap.map[0][i].key == key) {
				// pair found in slot 0
				values[idx] = hashmap.map[0][i].value;
				return;
			} else if (hashmap.map[1][i].key == key) {
				// pair found in slot 1
				values[idx] = hashmap.map[1][i].value;
				return;
			}
		}
	}
}

/** CUDA kernel which copies a (key, value) pair from a hash table to another one. */
__global__ void kernel_rehash(hash_table oldHash, hash_table newHash) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check thread index against task's upper bound
	if (idx >= oldHash.size) return;

	for (int slot = 0; slot <= 1; slot++) {
		if (oldHash.map[slot][idx].key == KEY_INVALID)
			// not a pair here
			continue;

		int oldKey, newKey;
		newKey = oldHash.map[slot][idx].key;

		// compute hash value for the key
		int hash = getHash(newKey, newHash.size);

		// ranges where to search an empty position in the new hash table
		int rangeBegin[2] = {hash, 0};
		int rangeEnd[2] = {newHash.size, hash};

		bool inserted = false;
		for (int r = 0; r <= 1 && !inserted; r++)
			for (int i = rangeBegin[r]; i < rangeEnd[r]; i++) {
				// try slot 0
				oldKey = atomicCAS(&newHash.map[0][i].key, KEY_INVALID, newKey);

				if (oldKey == KEY_INVALID) {
					newHash.map[0][i].value = oldHash.map[slot][idx].value;
					inserted = true;
					break;
				} else {
					// try slot 1
					oldKey = atomicCAS(&newHash.map[1][i].key, KEY_INVALID, newKey);

					if (oldKey == KEY_INVALID) {
						newHash.map[1][i].value = oldHash.map[slot][idx].value;
						inserted = true;
						break;
					}
				}
			}
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	numInsertedPairs = 0;
	hashmap.size = size;
	hashmap.map[0] = nullptr;
	hashmap.map[1] = nullptr;

	// allocate memory for the new hash map
	for (int slot = 0; slot <= 1; slot++) {
		if (cudaMalloc(&hashmap.map[slot], size * sizeof(entry)) != cudaSuccess) {
			std::cerr << "Memory allocation error\n";
			return;
		}
		cudaMemset(hashmap.map[slot], 0, size * sizeof(entry));
	}
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	// free hash table memory
	cudaFree(hashmap.map[0]);
	cudaFree(hashmap.map[1]);
}

/* RESHAPE HASH
 */
/** Increase the size of the hash table and move all values from the old hash
 *  by rehashing them into the new hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// initialize the new hash table
	hash_table newHashmap;
	newHashmap.size = numBucketsReshape;

	for (int slot = 0; slot <= 1; slot++) {
		if (cudaMalloc(&newHashmap.map[slot], numBucketsReshape * sizeof(entry)) != cudaSuccess) {
			std::cerr << "Memory allocation error in reshape\n";
			return;
		}
		cudaMemset(newHashmap.map[slot], 0, numBucketsReshape * sizeof(entry));
	}

	// load kernel for rehashing all elements from hashmap
	unsigned int numBlocks = hashmap.size / THREADS_PER_BLOCK;
	if (hashmap.size % THREADS_PER_BLOCK != 0) numBlocks++;
	kernel_rehash<<< numBlocks, THREADS_PER_BLOCK >>>(hashmap, newHashmap);

	cudaDeviceSynchronize();

	// free old maps' memory
	for (int slot = 0; slot <= 1; slot++)
		cudaFree(hashmap.map[slot]);

	// switch to the new hash table
	hashmap = newHashmap;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;

	size_t memSize = numKeys * sizeof(int);
	cudaMalloc(&deviceKeys, memSize);
	cudaMalloc(&deviceValues, memSize);

	if (!deviceKeys || !deviceValues) {
		std::cerr << "Memory allocation error\n";
		return false;
	}

	// check if we need to increase the hashtable's size to reduce the load factor
	// or to be able to insert all the new entries.
	if (float(numInsertedPairs + numKeys) / hashmap.size >= MAX_LOAD_FACTOR)
		reshape(int((numInsertedPairs + numKeys) / MIN_LOAD_FACTOR));

	// load keys and values into VRAM
	cudaMemcpy(deviceKeys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, memSize, cudaMemcpyHostToDevice);

	// load kernel for inserting pairs into hashtable
	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	kernel_insert<<< numBlocks, THREADS_PER_BLOCK >>>(deviceKeys,
	                                                  deviceValues, numKeys,
	                                                  hashmap);

	// wait for all insertions to finish
	cudaDeviceSynchronize();

	numInsertedPairs += numKeys;

	// free device memory
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *deviceKeys, *values;

	size_t memSize = numKeys * sizeof(int);
	cudaMalloc(&deviceKeys, memSize);
	cudaMallocManaged(&values, memSize);

	if (!deviceKeys || !values) {
		std::cerr << "Memory allocation error\n";
		return nullptr;
	}

	// load keys and values into VRAM
	cudaMemcpy(deviceKeys, keys, memSize, cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	kernel_get<<< numBlocks, THREADS_PER_BLOCK >>>(deviceKeys,
	                                               values, numKeys,
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
	// avoid division by 0
	return (hashmap.size == 0)? 0 : (float(numInsertedPairs) / hashmap.size);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
