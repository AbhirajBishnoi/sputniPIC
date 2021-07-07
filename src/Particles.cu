#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TPB 64

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is, bool use_pinned_memory)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////

    if(use_pinned_memory == false)
    {
        part->x = new FPpart[npmax];
        part->y = new FPpart[npmax];
        part->z = new FPpart[npmax];
        // allocate velocity
        part->u = new FPpart[npmax];
        part->v = new FPpart[npmax];
        part->w = new FPpart[npmax];

        // allocate charge = q * statistical weight
        part->q = new FPinterp[npmax];
    }

    else 
    {
        HANDLE_ERROR(cudaHostAlloc(&part->x, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->y, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->z, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->u, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->v, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->w, sizeof(FPpart) * npmax, cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc(&part->q, sizeof(FPinterp) * npmax, cudaHostAllocDefault));
    }
    
}
/** deallocate */
void particle_deallocate(struct particles* part, bool use_pinned_memory)
{

    // deallocate particle variables

   if (use_pinned_memory == false)
   {
        // deallocate particle variables
        delete[] part->x;
        delete[] part->y;
        delete[] part->z;
        delete[] part->u;
        delete[] part->v;
        delete[] part->w;
        delete[] part->q;
    }
    else
    {
        HANDLE_ERROR(cudaFreeHost(part->x));
        HANDLE_ERROR(cudaFreeHost(part->y));
        HANDLE_ERROR(cudaFreeHost(part->z));
        HANDLE_ERROR(cudaFreeHost(part->u));
        HANDLE_ERROR(cudaFreeHost(part->v));
        HANDLE_ERROR(cudaFreeHost(part->w));
        HANDLE_ERROR(cudaFreeHost(part->q));
    }    
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

/** particle kernel */

__global__ void single_particle_kernel(
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, 
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, 
    int nxn, int nyn, int nzn, 
    double xStart, double yStart, double zStart, 
    FPfield invdx, FPfield invdy, FPfield invdz, 
    double Lx, double Ly, double Lz, FPfield invVOL, 
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat, 
    FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat, 
    bool PERIODICX, bool PERIODICY, bool PERIODICZ, 
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2, 
    int NiterMover, int npmax, int stream_offset
){

    /* The stream_offset is to account for streaming, by default it is 0 and the result below results in idx = blockIdx.x * blockDim.x + threadIdx.x. 
        When streaming is used, the size of each array (x,y,z etc) is defined in terms of the stream size and hence we need to recalculate the offset to be new_global_offset = stream_offset + blockIdx.x * blockDim.x + threadIdx.x. */
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x + stream_offset; 
    int flat_idx = 0;
    
    if( (idx - stream_offset) >= npmax)
    {
        return;
    }

    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = x[idx];
    yptilde = y[idx];
    zptilde = z[idx];

    // calculate the average velocity iteratively
    for(int innter=0; innter < NiterMover; innter++){
        
        // interpolation G-->P
        ix = 2 +  int((x[idx] - xStart)*invdx);
        iy = 2 +  int((y[idx] - yStart)*invdy);
        iz = 2 +  int((z[idx] - zStart)*invdz);

        // calculate weights

        flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
        xi[0]   = x[idx] - XN_flat[flat_idx];

        flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
        eta[0]  = y[idx] - YN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
        zeta[0] = z[idx] - ZN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz, nyn, nzn);
        xi[1]   = XN_flat[flat_idx] - x[idx];
        eta[1]  = YN_flat[flat_idx] - y[idx];
        zeta[1] = ZN_flat[flat_idx] - z[idx];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;

        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){

                    flat_idx = get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn);
                    Exl += weight[ii][jj][kk]*Ex_flat[flat_idx];
                    Eyl += weight[ii][jj][kk]*Ey_flat[flat_idx];
                    Ezl += weight[ii][jj][kk]*Ez_flat[flat_idx];
                    Bxl += weight[ii][jj][kk]*Bxn_flat[flat_idx];
                    Byl += weight[ii][jj][kk]*Byn_flat[flat_idx];
                    Bzl += weight[ii][jj][kk]*Bzn_flat[flat_idx];
            
        } // end interpolation
        
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);

        // solve the position equation
        ut= u[idx] + qomdt2*Exl;
        vt= v[idx] + qomdt2*Eyl;
        wt= w[idx] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;

        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

        // update position
        x[idx] = xptilde + uptilde*dto2;
        y[idx] = yptilde + vptilde*dto2;
        z[idx] = zptilde + wptilde*dto2;


    } // end of iteration
    
    // update the final position and velocity
    u[idx]= 2.0*uptilde - u[idx];
    v[idx]= 2.0*vptilde - v[idx];
    w[idx]= 2.0*wptilde - w[idx];
    x[idx] = xptilde + uptilde*dt_sub_cycling;
    y[idx] = yptilde + vptilde*dt_sub_cycling;
    z[idx] = zptilde + wptilde*dt_sub_cycling;


    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (x[idx] > Lx){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] - Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = 2*Lx - x[idx];
        }
    }

    if (x[idx] < 0){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] + Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = -x[idx];
        }
    }


    // Y-DIRECTION: BC particles
    if (y[idx] > Ly){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] - Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = 2*Ly - y[idx];
        }
    }

    if (y[idx] < 0){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] + Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = -y[idx];
        }
    }

    // Z-DIRECTION: BC particles
    if (z[idx] > Lz){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] - Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = 2*Lz - z[idx];
        }
    }

    if (z[idx] < 0){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] + Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = -z[idx];
        }
    }
}

/** particle mover for GPU*/
int mover_PC_GPU_basic(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***GPU MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    // allocate memory for variables on device

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp *q_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL, *Ex_flat_dev = NULL, *Ey_flat_dev = NULL, *Ez_flat_dev = NULL, *Bxn_flat_dev = NULL, *Byn_flat_dev, *Bzn_flat_dev = NULL;
    
    cudaMalloc(&x_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&y_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&z_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&u_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&v_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&w_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    
    cudaMemcpy(x_dev, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMemcpy(y_dev, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMemcpy(z_dev, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMemcpy(u_dev, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMemcpy(v_dev, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMemcpy(w_dev, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);  
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){

        // Call GPU kernel

        single_particle_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(
            x_dev, y_dev, z_dev,u_dev, v_dev, w_dev, q_dev, 
            XN_flat_dev, YN_flat_dev, ZN_flat_dev, 
            grd->nxn, grd->nyn, grd->nzn, 
            grd->xStart, grd->yStart, grd->zStart, 
            grd->invdx, grd->invdy, grd->invdz, grd->Lx, grd->Ly, grd->Lz, grd->invVOL, 
            Ex_flat_dev, Ey_flat_dev, Ez_flat_dev, Bxn_flat_dev, Byn_flat_dev, Bzn_flat_dev, 
            param->PERIODICX, param->PERIODICY, param->PERIODICZ, 
            dt_sub_cycling, dto2, qomdt2, 
            part->NiterMover, part->nop, 0
        );

        cudaDeviceSynchronize();

    } // end of one particle

    // copy memory back to CPU (only the parts that have been modified inside the kernel)
    cudaMemcpy(part->x, x_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, y_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, z_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, u_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, v_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, w_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // clean up
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);
    cudaFree(Ex_flat_dev);
    cudaFree(Ey_flat_dev);
    cudaFree(Ez_flat_dev);
    cudaFree(Bxn_flat_dev);
    cudaFree(Byn_flat_dev);
    cudaFree(Bzn_flat_dev);

    return(0);
}

__global__ void interP2G_kernel(
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q,
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat,
    int nxn, int nyn, int nzn, 
    double xStart, double yStart, double zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, FPinterp *rhon_flat,
    FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat, FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat,
    int npmax, int stream_offset
)
{

    int idx = blockIdx.x*blockDim.x + threadIdx.x + stream_offset;
    if( (idx - stream_offset) >= npmax)
    {
        return;
    }

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // 3-D index of the cell
    int ix, iy, iz, flat_idx;

    ix = 2 + int (floor((x[idx] - xStart) * invdx));
    iy = 2 + int (floor((y[idx] - yStart) * invdy));
    iz = 2 + int (floor((z[idx] - zStart) * invdz));

    // distances from node
    flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
    xi[0]   = x[idx] - XN_flat[flat_idx];

    flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
    eta[0]  = y[idx] - YN_flat[flat_idx];

    flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
    zeta[0] = z[idx] - ZN_flat[flat_idx];

    flat_idx = get_idx(ix, iy, iz, nyn, nzn);
    xi[1]   = XN_flat[flat_idx] - x[idx];
    eta[1]  = YN_flat[flat_idx] - y[idx];
    zeta[1] = ZN_flat[flat_idx] - z[idx];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                // calculate the weights for different nodes
                weight[ii][jj][kk] = q[idx] * xi[ii] * eta[jj] * zeta[kk] * invVOL;


    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&rhon_flat[flat_idx], weight[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jx_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * weight[ii][jj][kk];
    
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&Jz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * u[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxx_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * v[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pxz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * v[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pyy_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) 
                temp[ii][jj][kk] = v[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pyz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }


    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * w[idx] * weight[ii][jj][kk];

    for (int ii=0; ii < 2; ii++)
        for (int jj=0; jj < 2; jj++)
            for(int kk=0; kk < 2; kk++) {
                flat_idx = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                atomicAdd(&pzz_flat[flat_idx], temp[ii][jj][kk] * invVOL);
            }

}

void interpP2G_GPU_basic(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp * q_dev = NULL, *Jx_flat_dev = NULL, *Jy_flat_dev = NULL, *Jz_flat_dev = NULL, *rhon_flat_dev = NULL, *pxx_flat_dev = NULL, *pxy_flat_dev = NULL, *pxz_flat_dev = NULL, *pyy_flat_dev = NULL, *pyz_flat_dev = NULL, *pzz_flat_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL;

    cudaMalloc(&x_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&y_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&z_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&u_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&v_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&w_dev, part->npmax * sizeof(FPpart));
    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMalloc(&Jx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&Jy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&Jz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&rhon_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pxx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pxy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pxz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pyy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pyz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&pzz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(x_dev, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(z_dev, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(u_dev, part->u, part->npmax* sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(w_dev, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(Jx_flat_dev, ids->Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Jy_flat_dev, ids->Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Jz_flat_dev, ids->Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(rhon_flat_dev, ids->rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pxx_flat_dev, ids->pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pxy_flat_dev, ids->pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pxz_flat_dev, ids->pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pyy_flat_dev, ids->pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pyz_flat_dev, ids->pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(pzz_flat_dev, ids->pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    interP2G_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(
        x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, 
        XN_flat_dev, YN_flat_dev, ZN_flat_dev, 
        grd->nxn, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, 
        grd->invdx, grd->invdy, grd->invdz, grd->invVOL, 
        Jx_flat_dev, Jy_flat_dev, Jz_flat_dev, rhon_flat_dev, 
        pxx_flat_dev , pxy_flat_dev, pxz_flat_dev, pyy_flat_dev, pyz_flat_dev, pzz_flat_dev, 
        part->nop
    );

    cudaDeviceSynchronize();

    // copy memory back to CPU (only the parts that have been modified inside the kernel)

    cudaMemcpy(ids->Jx_flat, Jx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, Jy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, Jz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhon_flat, rhon_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, pxx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, pxy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, pxz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, pyy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, pyz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, pzz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    
    // clean up

    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(q_dev);
    cudaFree(Jx_flat_dev);
    cudaFree(Jy_flat_dev);
    cudaFree(Jz_flat_dev);
    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);
    cudaFree(rhon_flat_dev);
    cudaFree(pxx_flat_dev);
    cudaFree(pxy_flat_dev);
    cudaFree(pxz_flat_dev);
    cudaFree(pyy_flat_dev);
    cudaFree(pyz_flat_dev);
    cudaFree(pzz_flat_dev);

}

