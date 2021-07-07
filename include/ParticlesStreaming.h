#ifndef PARTICLES_STREAMING_H
#define PARTICLES_STREAMING_H
#define TPB 64

#include <math.h>
#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

/** particle mover for GPU with batching */
int mover_GPU_stream(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** interpP2G for GPU with batching*/
void interpP2G_GPU_stream(struct particles*, struct interpDensSpecies*, struct grid*);

#endif
