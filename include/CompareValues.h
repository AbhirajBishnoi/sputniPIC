#ifndef CompareValues
#define CompareValues

#include <iostream>
#include <math.h>
#include "Alloc.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"
#include "Parameters.h"

void compareValues(struct parameters param, struct grid grd, struct interpDensSpecies* idsGPU, struct interpDensSpecies* idsCPU, struct interpDensNet idnGPU, struct interpDensNet idnCPU);

#endif
