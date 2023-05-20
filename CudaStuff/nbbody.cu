#include "nbody.cuh"

void launchKernel(float3* positions, float3* velocity, float* mass) {
	kernel << <(N + 255) / 256, 256 >> > (positions, velocity, mass);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kernel(float3* positions, float3* velocity, float* mass) {
	// Compute forces between particles and update their positions and velocities
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N) return;

	float3 acc = { 0.0f, 0.0f, 0.0f };
#pragma unroll // N must be #define to use this
	for (int j = 0; j < N; j++) {
		if (j == i) continue;

		float dx = positions[j].x - positions[i].x;
		float dy = positions[j].y - positions[i].y;
		float dz = positions[j].z - positions[i].z;
		float distSqr = dx * dx + dy * dy + dz * dz + 1e-6f;
		float invDist = rsqrtf(distSqr);
		float invDist3 = invDist * invDist * invDist;

		acc.x += dx * G * mass[j] * invDist3;
		acc.y += dy * G * mass[j] * invDist3;
		acc.z += dz * G * mass[j] * invDist3;
	}

	velocity[i].x += acc.x * DT ;
	velocity[i].y += acc.y * DT;
	velocity[i].z += acc.z * DT;

	positions[i].x += velocity[i].x * DT;
	positions[i].y += velocity[i].y * DT;
	positions[i].z += velocity[i].z * DT;
}