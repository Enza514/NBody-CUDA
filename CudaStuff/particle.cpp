#include "particle.h"


void Particle::deleteParticle() {
	if (vbo) {
		glDeleteBuffers(1, &vbo);
	}
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo));
	checkCudaErrors(cudaFree(mass));
	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(positions));
}


void Particle::initilizeParticleValues() {

	checkCudaErrors(cudaMallocManaged((void**)&positions, N * sizeof(float3)));
	checkCudaErrors(cudaMallocManaged((void**)&velocity, N * sizeof(float3)));
	checkCudaErrors(cudaMallocManaged((void**)&mass, N * sizeof(float)));

	for (int i = 0; i < N; i++) {
		float r = ((float)rand() / RAND_MAX) * 10.0f + 10.0f;
		float theta = ((float)rand() / RAND_MAX) * 2.0f * 3.1415926f;
		float phi = ((float)rand() / RAND_MAX) * 2.0f * 3.1415926f;
		positions[i].x = r * sinf(theta) * cosf(phi);
		positions[i].y = r * sinf(theta) * sinf(phi);
		positions[i].z = r * cosf(theta);
		velocity[i].x = 0.0f;
		velocity[i].y = 0.0f;
		velocity[i].z = 0.0f;
		mass[i] = rand() / (float)RAND_MAX + 1e-6f;
	}

	launchKernel(positions, velocity, mass);

	createBuffers();
}


void Particle::updateBuffers() {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float* ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	// if the pointer is valid(mapped), update VBO
	if (ptr)
	{
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo, 0));
		size_t num_bytes = 0;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, cuda_vbo));

		launchKernel(positions, velocity, mass); // modify buffer data
		glUnmapBuffer(GL_ARRAY_BUFFER); // unmaps it after use

		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo, 0));
	}
}


void Particle::createBuffers() {
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	size_t size = N * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, positions, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, 0));

	updateBuffers();
}