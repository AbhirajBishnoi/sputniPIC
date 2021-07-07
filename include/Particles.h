#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);

__global__ void interP2G_kernel(
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, 
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, 
    int nxn, int nyn, int nzn, 
    double xStart, double yStart, double zStart, 
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL, 
    FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, 
    FPinterp *rhon_flat, FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat, FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat, 
    int npmax, int stream_offset = 0
);

__global__ void single_particle_kernel(
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, 
    int nxn, int nyn, int nzn, double xStart, double yStart, double zStart, FPfield invdx, FPfield invdy, FPfield invdz, double Lx, double Ly, double Lz, FPfield invVOL, 
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat, FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat, 
    bool PERIODICX, bool PERIODICY, bool PERIODICZ, 
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2, 
    int NiterMover, int npmax, int stream_offset = 0    
);


/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int, bool use_pinned_memory = false);

/** deallocate */
void particle_deallocate(struct particles*, bool use_pinned_memory = false);

/** particle mover */
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

/** particle mover for GPU without batching*/
int mover_PC_GPU_basic(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** interpP2G for GPU*/
void interpP2G_GPU_basic(struct particles*, struct interpDensSpecies*, struct grid*);

#endif
