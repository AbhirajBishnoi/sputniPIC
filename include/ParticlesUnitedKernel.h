#ifndef PARTICLES_UNITED_KERNEL_H
#define PARTICLES_UNITED_KERNEL_H
#define TPB 64

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"


__global__ void united_kernel(
    /* common parameters */
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, 
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat,
    int nxn, int nyn, int nzn,
    double xStart, double yStart, double zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    int npmax, int NiterMover, int stream_offset,

    /* single_particle_kernel */
    double Lx, double Ly, double Lz,
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat,
    FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
    bool PERIODICX, bool PERIODICY, bool PERIODICZ,
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int n_sub_cycles,

    /* interP2G_kernel */
    FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, FPinterp *rhon_flat,
    FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat, FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat
);

/** particle mover for GPU without batching*/
void mover_AND_interpP2G_stream(
    struct particles* part, 
    struct EMfield* field, 
    struct grid* grd, 
    struct parameters* param,
    struct interpDensSpecies* ids
);

#endif