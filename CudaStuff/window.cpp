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

#include "Particle.h"
#include "helper_timer.h"

// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const char* window_name = "N-Body Test";

// variables
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
StopWatchInterface* timer = 0;
Particle particles;

// variables for mouse zoom/movement
float zoom = .05f; // starting zoom value
float offsetX = 0.0f;
float offsetY = 0.0f;
int lastX = 0;
int lastY = 0;
bool dragging = false;

// forward declarations
void display();
void cleanUp();
void computeFPS();
bool setupGL(int* argc, char** argv);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void renderTick(int value);
void physicsTick(int value);


int main(int argc, char** argv) {

	if (setupGL(&argc, argv) == false) {
		// TODO needs better error output
		std::cout << "There was a problem initializing glut!" << std::endl; 
		exit(EXIT_FAILURE);
	};
	sdkCreateTimer(&timer);
	
	particles.initilizeParticleValues();

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, renderTick, 0);
	glutTimerFunc(CALCULATION_DELAY, physicsTick, 0);

	glutMainLoop();

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanUp);
#endif

	exit(EXIT_SUCCESS);
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

	glBindBuffer(GL_ARRAY_BUFFER, particles.vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, N);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	sdkStopTimer(&timer);

	computeFPS();
}

void cleanUp() {
	sdkDeleteTimer(&timer);
	particles.deleteParticle();
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
	sprintf_s(fps, "N-Body Test: %3.1f fps", avgFPS);
	glutSetWindowTitle(fps);
	sdkResetTimer(&timer);
}

bool setupGL(int* argc, char** argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow(window_name);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		return false;
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	glEnable(GL_DEPTH_TEST);

	// TODO create another way to return false and check for errors
	return true;
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
		particles.updateBuffers();
		glutTimerFunc(CALCULATION_DELAY, physicsTick, 0);
	}
}
