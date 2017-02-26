
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int DATA_SET_SIZE = 18;
const int MOVES = 3;
const int PERMUTATIONS = MOVES * MOVES;
int NAB[PERMUTATIONS];
int data[DATA_SET_SIZE];

const int N = DATA_SET_SIZE;

// Generate a random set of integers each being in the range 0 to MOVES - 1
// Save numbers to file to ensure tests can be repeated.
void GenerateData(){
	ofstream out("data.dat", ios::out | ios::binary);
	for (int n = 0; n < DATA_SET_SIZE; n++){
		int i = rand() % MOVES;
		out.write((char *)&i, sizeof(i));
	}
	out.close();
}

// Populate data array with contents from file
void GetData(){
	ifstream in("data.dat", ios::in | ios::binary);
	for (int n = 0; n < DATA_SET_SIZE; n++){
		in.read((char *)&data[n], sizeof(int));
	}
	in.close();
}

/*
Intialise NAB pointer allocated on GPU device
*/
__global__ void InitialiseNAB(int *NAB){
	int i = blockIdx.x;
	NAB[i] = 0;
}

/*
This device function is called from the global PopulateNABblocks/PopulateNABthreads kernel
*/
__device__ int GetIndex(int firstMove, int secondMove){
	if (firstMove == 0 && secondMove == 0) return 0;
	if (firstMove == 0 && secondMove == 1) return 1;
	if (firstMove == 0 && secondMove == 2) return 2;
	if (firstMove == 1 && secondMove == 0) return 3;
	if (firstMove == 1 && secondMove == 1) return 4;
	if (firstMove == 1 && secondMove == 2) return 5;
	if (firstMove == 2 && secondMove == 0) return 6;
	if (firstMove == 2 && secondMove == 1) return 7;
	if (firstMove == 2 && secondMove == 2) return 8;
}

__global__ void PopulateNABblocks(int *NAB, int *data){
	int index = blockIdx.x;
	int i;
	int previous = data[index];
	int next = data[index + 1];

	if (index < DATA_SET_SIZE)
	{
		i = GetIndex(previous, next);
	}
	NAB[i]++;
}

__global__ void PopulateNABThreads(int *NAB, int *data){
	int index = threadIdx.x;
	int i;
	int previous = data[index];
	if (index < DATA_SET_SIZE)
	{
		i = GetIndex(previous, data[index + 1]);
	}
	NAB[i]++;
}

void DisplayNAB(){
	cout << endl;
	for (int n = 0; n < PERMUTATIONS; n++){
		cout << "Index " << n << " : " << NAB[n] << endl;
	}
}

int main(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreads = prop.maxThreadsPerBlock;

	srand(time(NULL));
	GenerateData();
	GetData();
	
	int *dev_NAB;
	int *dev_data;

	cudaMalloc((void**)&dev_NAB, PERMUTATIONS * sizeof(int));
	cudaMalloc((void**)&dev_data, DATA_SET_SIZE * sizeof(int));

	cudaMemcpy(dev_data, &data, DATA_SET_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	InitialiseNAB << <PERMUTATIONS, 1 >> >(dev_NAB);
	PopulateNABblocks << <DATA_SET_SIZE, 1 >> >(dev_NAB, dev_data);
	//PopulateNABThreads << <1, DATA_SET_SIZE>> >(dev_NAB, dev_data);
	cudaMemcpy(NAB, dev_NAB, PERMUTATIONS * sizeof(int), cudaMemcpyDeviceToHost);
	DisplayNAB();

	cudaFree(dev_NAB);
	cudaFree(dev_data);
	getchar();
	return 0;
}