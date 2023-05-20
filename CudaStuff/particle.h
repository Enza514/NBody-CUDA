#pragma once

#ifndef PARTICLE_H
#define PARTICLE_H

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "defines.h"
#include "helper_cuda.h"
#include "nbody.cuh"

class Particle {
public:
	Particle() = default;
	~Particle() = default;

	GLuint vbo;

	void deleteParticle();
	void initilizeParticleValues();
	void updateBuffers();

private:
	float3* positions;
	float3* velocity;
	float* mass;

	struct cudaGraphicsResource* cuda_vbo;

	void createBuffers();
};

#endif
