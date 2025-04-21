#include "../include/walker.h"
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#ifdef _WIN32
    #define WALKER3D_EXPORT __declspec(dllexport)
#else
    #define WALKER3D_EXPORT __attribute__((visibility ("default")))
#endif

typedef struct {
    int* voidMap;
    int* solidMap;
    int layers;
    int height;
    int width;
    double minLen;
    double maxLen;
    double minTheta;
    double maxTheta;
    double minPhi;
    double maxPhi;
} HeatMap3D;

// Verify that data is received correctly
void printBoolImage3D(bool* image, int height, int width, int layers) {
    for (int i = 0; i < width * height * layers; ++i) {
        printf("%d ", image[i]);
        if (i > 0 && (i + 1) % width == 0) {
            printf("\n");
        }

        if ((i + 1) % (height * width) == 0) {
            printf("\n");
        }
    }
}

enum BorderStatus getBorderStatus3D(double posRow, double posCol, double posLayer, double height, double width, double layers) {
    if (posRow < 0.0) {
        return TOP;
    } else if (posRow >= height) {
        return BOTTOM;
    } else if (posCol < 0.0) {
        return LEFT;
    } else if (posCol >= width) {
        return RIGHT;
    } else if (posLayer < 0.0) {
        return FRONT;
    } else if (posLayer >= layers) {
        return BACK;
    } else {
        return INSIDE;
    }
}

// Find first wall
void getStartPos3D(double* distances, int height, int width, int layers, double* startRow, double* startCol, double* startLayer) {
    for (int i = height * width * layers / 2; i < height * width * layers; ++i) {
        if (distances[i] < 2.0 && distances[i] && distances[i] > 0) {
            int submatrixPos = i % (height * width); 
            *startRow = submatrixPos / width;
            *startCol = submatrixPos % width;
            *startLayer = i / (height * width);
            return;
        }
    }
}

void getStepSizes3D(double* stepRow, double* stepCol, double* stepLayer) {
    // Subtract RAND_MID to get negative numbers
    *stepRow = rand() - RAND_MID;
    *stepCol = rand() - RAND_MID;
    *stepLayer = rand() - RAND_MID;

    // Ensure we get non-zero values
    while (*stepCol == 0 || *stepRow == 0 || *stepLayer == 0) {
        if (*stepCol == 0) {
            *stepCol = rand() - RAND_MID;
        }
        
        if (*stepRow == 0) {
            *stepRow = rand() - RAND_MID;
        }

        if (*stepLayer == 0) {
            *stepLayer = rand() - RAND_MID;
        }
    }

    // Normalize the steps, so we always move to a new pixel
    double maxStep = max(fabs(*stepLayer), max(fabs(*stepRow), fabs(*stepCol)));
    *stepCol /= maxStep;
    *stepRow /= maxStep;
    *stepLayer /= maxStep;
}

void updateMap3D(HeatMap3D* heatMap, double voidLen, double solidLen, double theta, double phi) {
    int mapLayers = heatMap->layers;
    int mapHeight = heatMap->height;
    int mapWidth = heatMap->width;
    double minLen = heatMap->minLen;
    double maxLen = heatMap->maxLen;
    double minTheta = heatMap->minTheta;
    double maxTheta = heatMap->maxTheta;
    double minPhi = heatMap->minPhi;
    double maxPhi = heatMap->maxPhi;

    int posLayerVoid = (min(max(voidLen, minLen), maxLen) - minLen) / (maxLen - minLen) * (mapLayers - 1);
    int posLayerSolid = (min(max(solidLen, minLen), maxLen) - minLen) / (maxLen - minLen) * (mapLayers - 1);
    int posRow = (min(max(theta, minTheta), maxTheta) - minTheta) / (maxTheta - minTheta) * (mapHeight - 1);
    int posCol = (min(max(phi, minPhi), maxPhi) - minPhi) / (maxPhi - minPhi) * (mapWidth - 1);

    ++(heatMap->voidMap[posLayerVoid * mapWidth * mapHeight + posRow * mapWidth + posCol]);
    if (solidLen > 0.0) {
        ++(heatMap->solidMap[posLayerSolid * mapWidth * mapHeight + posRow * mapWidth + posCol]);
    }
}

// Allocates memory: caller must call destroyHeatMap3D
WALKER3D_EXPORT
HeatMap3D* createHeatMap3D(int* voidMap, int* solidMap, int layers, int height, int width, double minLen,
                           double maxLen, double minTheta, double maxTheta, double minPhi, double maxPhi)
{
    HeatMap3D* heatMap = malloc(sizeof(HeatMap3D));
    heatMap->voidMap = voidMap;
    heatMap->solidMap = solidMap;
    heatMap->layers = layers;
    heatMap->height = height;
    heatMap->width = width;
    heatMap->minLen = minLen;
    heatMap->maxLen = maxLen;
    heatMap->minTheta = minTheta;
    heatMap->maxTheta = maxTheta;
    heatMap->minPhi = minPhi;
    heatMap->maxPhi = maxPhi;

    return heatMap;
}

WALKER3D_EXPORT
void destroyHeatMap3D(HeatMap3D* heatMap) {
    free(heatMap);
}

WALKER3D_EXPORT
void walk3D(bool* image, int layers, int height, int width, double* distances, int iterations, HeatMap3D* heatMap, int* path)
{
    srand(time(NULL));

    double posLayer = 0;
    double posRow = 0;
    double posCol = 0;

    getStartPos3D(distances, height, width, layers, &posRow, &posCol, &posLayer);

    printf("Start layer: %lf\n", posLayer);
    printf("Start row: %lf\n", posRow);
    printf("Start col: %lf\n", posCol);

    time_t startTime = clock();
    for (int i = 0; i < iterations; ++i) {
        double stepLayer = 0;
        double stepRow = 0;
        double stepCol = 0;

        getStepSizes3D(&stepRow, &stepCol, &stepLayer);

        double solidSteps = 0;
        double voidSteps = 0;
        double lastStepSize = 0;
        bool lastPhase = false;
        enum BorderStatus borderStatus = INSIDE; // always start at a valid pixel
        int contPos = (int)posLayer * width * height + (int)posRow * width + (int)posCol; // position in contiguous array

        while (! lastPhase || image[contPos]) {
            if (borderStatus != INSIDE) {
                // Go back to last valid position
                posLayer -= lastStepSize * stepLayer;
                posRow -= lastStepSize * stepRow;
                posCol -= lastStepSize * stepCol;
                if (borderStatus == LEFT || borderStatus == RIGHT) {
                    stepCol *= -1;
                } else if (borderStatus == TOP || borderStatus == BOTTOM) {
                    stepRow *= -1;
                } else {
                    stepLayer *= -1;
                }
            }

            contPos = (int)posLayer * width * height + (int)posRow * width + (int)posCol;

            lastStepSize = max(distances[contPos], 1.0);

            posLayer += lastStepSize * stepLayer;
            posRow += lastStepSize * stepRow;
            posCol += lastStepSize * stepCol;

            contPos = (int)posLayer * width * height + (int)posRow * width + (int)posCol;            

            borderStatus = getBorderStatus3D(posRow, posCol, posLayer, height, width, layers);
            if (borderStatus != INSIDE) {
                continue;
            }

            if (image[contPos]) {
                path[contPos] += 1;
                voidSteps += lastStepSize;
                lastPhase = true;
            } else if (! image[contPos]) {
                path[contPos] -= 1;
                solidSteps += lastStepSize;
            }
        }

        // Go back to last valid position
        posRow -= lastStepSize * stepRow;
        posCol -= lastStepSize * stepCol;
        posLayer -= lastStepSize * stepLayer;

        double radius = sqrt(stepRow * stepRow + stepCol * stepCol + stepLayer * stepLayer);
        double solidLen = solidSteps * radius;
        double voidLen = voidSteps * radius;
        double theta = atan2(sqrt(stepRow * stepRow + stepCol * stepCol), stepLayer);
        double phi = atan2(stepRow, stepCol);
        updateMap3D(heatMap, voidLen, solidLen, theta, phi);

#ifdef DEBUG
        printf("---Iteration %d---\n", i);
        printf("Steps: %lf\n", voidSteps);
        printf("Final pos: (Row: %d), (Col: %d), (Layer: %d)\n", (int)posRow, (int)posCol, (int)posLayer);
        printf("stepRow: %lf\n", stepRow);
        printf("stepCol: %lf\n", stepCol);
        printf("stepLayer: %lf\n", stepLayer);
#endif
    }
    time_t endTime = clock();

    printf("End layer: %lf\n", posLayer);
    printf("End row: %lf\n", posRow);
    printf("End col: %lf\n", posCol);
    printf("Time elapsed for %d iterations: %lfms\n", iterations, (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000);
}
