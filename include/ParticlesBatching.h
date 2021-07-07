#ifndef PARTICLES_BATCHING_H
#define PARTICLES_BATCHING_H
#define TPB 64

#include <math.h>
#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

/** particle mover for GPU with batching */
int mover_GPU_batch(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** interpP2G for GPU with batching*/
void interpP2G_GPU_batch(struct particles*, struct interpDensSpecies*, struct grid*);

/** query amount of free memory available for use on GPU before splitting into batches*/
size_t queryFreeMemoryOnGPU(void);

#endif
