#include "../include/walker.h"
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#ifdef _WIN32
    #define WALKER2D_EXPORT __declspec(dllexport)
#else
    #define WALKER2D_EXPORT __attribute__((visibility ("default")))
#endif

typedef struct {
    int* voidMap;
    int* solidMap;
    int height;
    int width;
    double minLen;
    double maxLen;
    double minAngle;
    double maxAngle;
} HeatMap2D;

// Verify that data is received correctly
void printBoolImage2D(bool* image, int height, int width) {
    for (int i = 0; i < width * height; ++i) {
        printf("%d ", image[i]);
        if (i > 0 && (i + 1) % width == 0) {
            printf("\n");
        }
    }
}

void printDoubleImage2D(double* image, int height, int width) {
    for (int i = 0; i < width * height; ++i) {
        printf("%lf ", image[i]);
        if (i > 0 && (i + 1) % width == 0) {
            printf("\n");
        }
    }
}

enum BorderStatus getBorderStatus2D(double posRow, double posCol, int height, int width) {
    if (posRow < 0.0) {
        return TOP;
    } else if (posRow >= (double)height) {
        return BOTTOM;
    } else if (posCol < 0.0) {
        return LEFT;
    } else if (posCol >= (double)width) {
        return RIGHT;
    } else {
        return INSIDE;
    }
}

// Find first wall
void getStartPos2D(double* distances, int height, int width, double* startRow, double* startCol) {
    for (int i = height * width / 2; i < height * width; ++i) {
        if (distances[i] < 2.0 && distances[i] && distances[i] > 0.0) {
            *startRow = i / height;
            *startCol = i % width;
            return;
        }
    }
}

void getStepSizes2D(double* stepRow, double* stepCol) {
    // Subtract RAND_MID to get negative numbers
    *stepRow = rand() - RAND_MID;
    *stepCol = rand() - RAND_MID;

    // Ensure we get non-zero values
    while (*stepCol == 0.0 || *stepRow == 0.0) {
        if (*stepCol == 0.0) {
            *stepCol = rand() - RAND_MID;
        } else {
            *stepRow = rand() - RAND_MID;
        }
    }

    // Normalize the steps, so we always move to a new pixel
    if (fabs(*stepCol) > fabs(*stepRow)) {
        *stepRow = (*stepRow / fabs(*stepCol));
        *stepCol = *stepCol > 0.0 ? 1.0 : -1.0;
    } else {
        *stepCol = (*stepCol / fabs(*stepRow));
        *stepRow = *stepRow > 0.0 ? 1.0 : -1.0;
    }
}

void updateMap2D(HeatMap2D* heatMap, double voidLen, double solidLen, double angle)
{
    double minLen = heatMap->minLen;
    double maxLen = heatMap->maxLen;
    double minAngle = heatMap->minAngle;
    double maxAngle = heatMap->maxAngle;
    int mapHeight = heatMap->height;
    int mapWidth = heatMap->width;

    int posRowVoid = (min(max(voidLen, minLen), maxLen) - minLen) / (maxLen - minLen) * (mapHeight - 1);
    int posRowSolid = (min(max(solidLen, minLen), maxLen) - minLen) / (maxLen - minLen) * (mapHeight - 1);
    int posCol = (min(max(angle, minAngle), maxAngle) - minAngle) / (maxAngle - minAngle) * (mapWidth - 1);

#ifdef DEBUG
    printf("angle: %lf\n", angle);
    printf("posCol: %d\n", posCol);
    printf("voidLen: %lf\n", voidLen);
    printf("posRowVoid %d\n", posRowVoid);
    printf("solidLen: %lf\n", solidLen);
    printf("posRowSolid %d\n", posRowSolid);
#endif

    ++(heatMap->voidMap[posRowVoid * mapWidth + posCol]);
    if (solidLen > 0.0) {
        ++(heatMap->solidMap[posRowSolid * mapWidth + posCol]);
    }
}

int getContPos2D(double posRow, double posCol, double stepRow, double stepCol, int height, int width) {
    return (int)(posRow) * width + (int)(posCol);
    // int a = (int)(posRow) * width + (int)(posCol);
    // int b = (int)(floor(posRow)) * width + (int)(floor(posCol));

    // int row = stepRow < 0.0 ? ceil(posRow) : floor(posRow);
    // int col = stepCol < 0.0 ? ceil(posCol) : floor(posCol);
    // int c = row * width + col;

    // if (a != c) {
    //     printf("posRow: %lf, posCol: %lf, a: %d, c: %d\n", posRow, posCol, a, c);
    // }

    // if (c < 0 || c >= width * height) {
    //     PRINT_DB_LINE
    //     printf("posRow: %lf, posCol: %lf, a: %d, c: %d\n", posRow, posCol, a, c);
    // }

    // return a;
}

// Allocates memory: caller must call destroyHeatMap2D
WALKER2D_EXPORT
HeatMap2D* createHeatMap2D(int* voidMap, int* solidMap, int height, int width,
                           double minLen, double maxLen, double minAngle, double maxAngle)
{
    HeatMap2D* heatMap = malloc(sizeof(HeatMap2D));
    heatMap->voidMap = voidMap;
    heatMap->solidMap = solidMap;
    heatMap->height = height;
    heatMap->width = width;
    heatMap->minLen = minLen;
    heatMap->maxLen = maxLen;
    heatMap->minAngle = minAngle;
    heatMap->maxAngle = maxAngle;

    return heatMap;
}

WALKER2D_EXPORT
void destroyHeatMap2D(HeatMap2D* heatMap) {
    free(heatMap);
}

WALKER2D_EXPORT
void walk2D(bool* image, int height, int width, double* distances, int iterations, HeatMap2D* heatMap, int* path)
{
    srand(time(NULL));

    double posRow = 0.0;
    double posCol = 0.0;

    getStartPos2D(distances, height, width, &posRow, &posCol);

    printf("Start row: %lf\n", posRow);
    printf("Start col: %lf\n", posCol);

    time_t startTime = clock();
    for (int i = 0; i < iterations; ++i) {
        double stepRow = 0.0;
        double stepCol = 0.0;

        getStepSizes2D(&stepRow, &stepCol);

        double solidSteps = 0.0;
        double voidSteps = 0.0;
        double lastStepSize = 0.0;
        bool lastPhase = false;
        enum BorderStatus borderStatus = INSIDE; // always start at a valid pixel
        int contPos = getContPos2D(posRow, posCol, stepRow, stepCol, height, width); // position in contiguous array

        while (! lastPhase || image[contPos]) {
            lastStepSize = max(distances[contPos], 1.0);
            posRow += lastStepSize * stepRow;
            posCol += lastStepSize * stepCol;

            borderStatus = getBorderStatus2D(posRow, posCol, height, width);
            if (borderStatus != INSIDE) {
                // Go back to last valid position
                posRow -= lastStepSize * stepRow;
                posCol -= lastStepSize * stepCol;

                contPos = getContPos2D(posRow, posCol, stepRow, stepCol, height, width);
                if (borderStatus == LEFT || borderStatus == RIGHT) {
                    stepCol *= -1;
                } else {
                    stepRow *= -1;
                }
                continue;
            }

            contPos = getContPos2D(posRow, posCol, stepRow, stepCol, height, width);

            if (image[contPos]) {
                path[contPos] += 1;
                voidSteps += lastStepSize;
                lastPhase = true;
            } else if (! image[contPos]) {
                path[contPos] -= 1;
                solidSteps += 1.0;
            }
        }

        // Go back to last valid position
        posRow -= lastStepSize * stepRow;
        posCol -= lastStepSize * stepCol;

        #ifdef DEBUG
        printf("---Iteration %d---\n", i);
        printf("Steps: %lf\n", voidSteps);
        printf("Final pos: (Row: %d), (Col: %d)\n", (int)posRow, (int)posCol);
        printf("stepRow: %lf\n", stepRow);
        printf("stepCol: %lf\n", stepCol);
#endif

        double hypotenuse = sqrt(stepRow * stepRow + stepCol * stepCol);
        double voidLen = voidSteps * hypotenuse;
        double solidLen = solidSteps * hypotenuse;
        double angle = atan(stepRow / stepCol);
        updateMap2D(heatMap, voidLen, solidLen, angle);
    }
    time_t endTime = clock();

    printf("End row: %lf\n", posRow);
    printf("End col: %lf\n", posCol);
    printf("Time elapsed for %d iterations: %lfms\n", iterations, (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000);
}
