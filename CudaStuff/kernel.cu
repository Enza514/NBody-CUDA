// based off CUDA's SimpleGL sample for OpenGL interop

#include <math.h>
#include <stdio.h>
#include <time.h>

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "helper_timer.h"
#include "helper_cuda.h"


#define N 8192 // please set as a multiple of 1024
#define G 1.0f // gravity scalar
#define DT 0.01f // time step

#define REFRESH_DELAY     8.33 //ms

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const char* window_name = "N-Body Test";

// buffers
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// information storage
float3* positions;
float3* velocity;
float* mass;

//fps stuff
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
StopWatchInterface* timer = 0;

// variables for mouse zoom/movement
float zoom = .05f; // starting zoom value to fit most things in frame
float offsetX = 0.0f;
float offsetY = 0.0f;
int lastX = 0;
int lastY = 0;
bool dragging = false;

//forward declarations 
// misc
void cleanUp();
void computeFPS();
void initilizeArrays();

// GL stuff
void createBuffers();
void renderTick(int);
void physicsTick(int);
void updateBuffers();
void mouse(int, int, int, int);
void motion(int, int);

// Cuda Stuff
void launchKernel(float3*, float3*, float*);
__global__ void kernel(float3*, float3*, float*);

// rendering
void display();


int main(int argc, char** argv) {
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow(window_name);

	// glew initialization
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(-1);
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	glEnable(GL_DEPTH_TEST);

	checkCudaErrors(cudaMallocManaged(&positions, N * sizeof(float3)));
	checkCudaErrors(cudaMallocManaged(&velocity, N * sizeof(float3)));
	checkCudaErrors(cudaMallocManaged(&mass, N * sizeof(float)));

	sdkCreateTimer(&timer);
	
	initilizeArrays();

	launchKernel(positions, velocity, mass); // updating once

	createBuffers();

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, renderTick, 0);
	glutTimerFunc(16.6, physicsTick, 0);

	glutMainLoop();

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanUp);
#endif

	exit(0);
}


void cleanUp() {

	sdkDeleteTimer(&timer);

	if (vbo) {
		glDeleteBuffers(1, &vbo);
	}
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo));

	checkCudaErrors(cudaFree(mass));
	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(positions));
}


void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);
	}

	char fps[256];
	sprintf(fps, "N-Body Test: %3.1f fps", avgFPS);
	glutSetWindowTitle(fps);
	sdkResetTimer(&timer);
}


void createBuffers() {

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	size_t size = N * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, positions, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, 0));
}


void display() {

	sdkStartTimer(&timer);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glScalef(zoom, zoom, zoom);
	glTranslatef(offsetX, offsetY, 0.0f);

	glColor3f(1.0f, 1.0f, 1.0f);
	glPointSize(2.0f);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, N);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();

	sdkStopTimer(&timer);

	computeFPS();
}


void initilizeArrays() {

	std::srand(std::time(0));

	// black hole?
	positions[0].x = 0.0f;
	positions[0].y = 0;
	positions[0].z = 0;
	velocity[0].x = 0.0f;
	velocity[0].y = 0.0f;
	velocity[0].z = 0.0f;
	mass[0] = rand() / (((float)RAND_MAX + 1e-6f) / 10000.0f);

	for (int i = 1; i < N; i++) {
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
}


void mouse(int button, int state, int x, int y) {
	if (button == 3) {
		zoom *= 1.1f; // scroll up
	}
	else if (button == 4) {
		zoom *= 0.9f; // scroll down
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		dragging = true;
		lastX = x;
		lastY = y;
	}
	else {
		dragging = false;
	}
}


void motion(int x, int y) {
	if (dragging) {
		offsetX += ((x - lastX) * 0.001f) * (1 / zoom);
		offsetY -= ((y - lastY) * 0.001f) * (1 / zoom);
		lastX = x;
		lastY = y;
	}
}


void renderTick(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, renderTick, 0);
	}
}

void physicsTick(int value)
{
	if (glutGetWindow())
	{
		updateBuffers();
		glutTimerFunc(16.6, physicsTick, 0);
	}
}


void updateBuffers() {

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


// kernel related functions

void launchKernel(float3* positions, float3* velocity, float* mass) {
	kernel <<<(N + 255) / 256, 256>>> (positions, velocity, mass);
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

	velocity[i].x += acc.x * DT;
	velocity[i].y += acc.y * DT;
	velocity[i].z += acc.z * DT;

	positions[i].x += velocity[i].x * DT;
	positions[i].y += velocity[i].y * DT;
	positions[i].z += velocity[i].z * DT;
}