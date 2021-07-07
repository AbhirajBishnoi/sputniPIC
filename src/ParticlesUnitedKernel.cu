#include "Particles.h"
#include "ParticlesBatching.h"
#include "ParticlesUnitedKernel.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TPB 64
#define NUMBER_OF_PARTICLES_PER_BATCH 1024000
#define MAX_NUMBER_OF_STREAMS 5
#define NUMBER_OF_STREAMS_PER_BATCH 4

// Input parameters: 1) Common parameters 2) Only single_particle_kernel 3) Only interP2G_kernel 
__global__ void united_kernel(
    FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, 
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat,
    int nxn, int nyn, int nzn,
    double xStart, double yStart, double zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    int npmax, int NiterMover, int stream_offset,

    double Lx, double Ly, double Lz,
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat,
    FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
    bool PERIODICX, bool PERIODICY, bool PERIODICZ,
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int n_sub_cycles,

    FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, FPinterp *rhon_flat,
    FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat, FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat
){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x + stream_offset; 
    if( (idx - stream_offset) >= npmax)
    {
        return;
    }

    //////////////////////////////////
    ///// single_particle_kernel /////
    //////////////////////////////////
    
    for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){

        // The stream_offset is to account for streaming, by default it is 0 and the result below results in idx = blockIdx.x * blockDim.x + threadIdx.x. 
        // When streaming is used, the size of each array (x,y,z etc) is defined in terms of the stream size and hence we need to recalculate the offset to be new_global_offset = stream_offset + blockIdx.x * blockDim.x + threadIdx.x. 

        int flat_idx = 0;

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

    ///////////////////////////
    ///// interP2G_kernel /////
    ///////////////////////////

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




void mover_AND_interpP2G_stream(
    struct particles* part, 
    struct EMfield* field, 
    struct grid* grd, 
    struct parameters* param,
    struct interpDensSpecies* ids
)
{

    // print species and subcycling
    std::cout << "***GPU mover_AND_interpP2G_stream - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    // allocate memory for variables on device

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp *q_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL;

    // mover_PC:
    FPfield *Ex_flat_dev = NULL, *Ey_flat_dev = NULL, *Ez_flat_dev = NULL, *Bxn_flat_dev = NULL, *Byn_flat_dev, *Bzn_flat_dev = NULL;

    // interpP2G:
    FPinterp *Jx_flat_dev = NULL, *Jy_flat_dev = NULL, *Jz_flat_dev = NULL, *rhon_flat_dev = NULL, *pxx_flat_dev = NULL, *pxy_flat_dev = NULL, *pxz_flat_dev = NULL, *pyy_flat_dev = NULL, *pyz_flat_dev = NULL, *pzz_flat_dev = NULL;

    size_t free_bytes = 0;

    int i, total_size_particles, start_index_batch, end_index_batch, number_of_batches;

    // Calculation done later to compute free space after allocating space on the GPU for 
    // other variables below, the assumption is that these variables fit in the GPU memory 
    // and mini batching is implemented only taking into account particles

    // Common mover_PC & interpP2G:
    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    // mover_PC:
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    // interpP2G:
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

    // Common mover_PC & interpP2G:
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    // mover_PC:
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    // interpP2G:
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

    free_bytes = queryFreeMemoryOnGPU();
    total_size_particles = sizeof(FPpart) * part->npmax * 6 + sizeof(FPinterp) * part->npmax; //  for x,y,z,u,v,w and q

    start_index_batch = 0, end_index_batch = 0;

    // implement mini-batching only in the case where the free space on the GPU isn't enough

    if(free_bytes > total_size_particles)
    {
        start_index_batch = 0;
        end_index_batch = part->npmax - 1; // set end_index to the last particle as we are processing in in one batch
        number_of_batches = 1;
    }
    else
    {
        start_index_batch = 0;
        end_index_batch = start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH - 1; // NUM_PARTICLES_PER_BATCH is a hyperparameter set by tuning
        if(part->npmax % NUMBER_OF_PARTICLES_PER_BATCH != 0)
        {
            number_of_batches = part->npmax / NUMBER_OF_PARTICLES_PER_BATCH + 1; // works because of integer division
        }
        else
        {
            number_of_batches = part->npmax / NUMBER_OF_PARTICLES_PER_BATCH;
        }
    }


    cudaStream_t cudaStreams[MAX_NUMBER_OF_STREAMS];

    for(i = 0; i < number_of_batches; i++)
    {
        //std::cout << "  batch number: " << i << std::endl;

        long int number_of_particles_batch = end_index_batch - start_index_batch + 1; // number of particles in  a batch
        size_t batch_size_per_attribute = number_of_particles_batch * sizeof(FPpart); // size of the attribute per batch in bytes x,z,y,u,v,w

        long int number_of_particles_stream = 0, stream_size_per_attribute = 0, number_of_streams = 0, stream_offset = 0, offset = 0, start_index_stream = 0, end_index_stream = 0, max_num_particles_per_stream = 0;

        //std::cout << "  num_of_particles_batch: " << number_of_particles_batch << " batch_size : " << batch_size_per_attribute << std::endl;
        //std::cout << "  start_index: " << start_index_batch << " end_index: " << end_index_batch << std::endl;

        cudaMalloc(&x_dev, batch_size_per_attribute);
        cudaMalloc(&y_dev, batch_size_per_attribute);
        cudaMalloc(&z_dev, batch_size_per_attribute);
        cudaMalloc(&u_dev, batch_size_per_attribute);
        cudaMalloc(&v_dev, batch_size_per_attribute);
        cudaMalloc(&w_dev, batch_size_per_attribute);
        cudaMalloc(&q_dev, number_of_particles_batch * sizeof(FPinterp));

        start_index_stream = 0;
        end_index_stream = start_index_stream + (number_of_particles_batch / NUMBER_OF_STREAMS_PER_BATCH) - 1;
        max_num_particles_per_stream = number_of_particles_batch / NUMBER_OF_STREAMS_PER_BATCH;            

        if(number_of_particles_batch % NUMBER_OF_STREAMS_PER_BATCH != 0) // We have some leftover bytes
        {
            number_of_streams = NUMBER_OF_STREAMS_PER_BATCH + 1;
        }
        else
        {
           number_of_streams = NUMBER_OF_STREAMS_PER_BATCH;
        }

        for (int j = 0; j < number_of_streams; j++)
        {
            cudaStreamCreate(&cudaStreams[j]);
        }

        for (int stream_idx = 0; stream_idx < number_of_streams; stream_idx++)
        {

            number_of_particles_stream = end_index_stream - start_index_stream + 1;
            stream_size_per_attribute = number_of_particles_stream * sizeof(FPpart); // for x,y,z,u,v,w

            //std::cout << "      num_of_particles_stream: " << number_of_particles_stream << "stream_size : " << stream_size_per_attribute << std::endl;
            //std::cout << "      start_index: " << start_index_stream << " end_index: " << end_index_stream << std::endl;
        
            stream_offset = start_index_stream;
            offset = stream_offset + start_index_batch; // batch offset + stream_offset

            cudaMemcpyAsync(&x_dev[stream_offset], &part->x[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&y_dev[stream_offset], &part->y[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&z_dev[stream_offset], &part->z[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&u_dev[stream_offset], &part->u[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&v_dev[stream_offset], &part->v[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&w_dev[stream_offset], &part->w[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&q_dev[stream_offset], &part->q[offset], number_of_particles_stream * sizeof(FPinterp), cudaMemcpyHostToDevice, cudaStreams[stream_idx]);

            //std::cout << "      Before loop;" << " Offset: " << offset << " # of elems: " << number_of_particles_stream
            //       << " Stream index: " << stream_idx << std::endl;

            // Call GPU kernel
            united_kernel<<<(number_of_particles_stream + TPB - 1)/TPB, TPB, 0, cudaStreams[stream_idx]>>>(
                x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, 
                XN_flat_dev, YN_flat_dev, ZN_flat_dev, 
                grd->nxn, grd->nyn, grd->nzn, 
                grd->xStart, grd->yStart, grd->zStart, 
                grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
                number_of_particles_stream, part->NiterMover, stream_offset,

                grd->Lx, grd->Ly, grd->Lz, 
                Ex_flat_dev, Ey_flat_dev, Ez_flat_dev, 
                Bxn_flat_dev, Byn_flat_dev, Bzn_flat_dev, 
                param->PERIODICX, param->PERIODICY, param->PERIODICZ, 
                dt_sub_cycling, dto2, qomdt2,
                part->n_sub_cycles,
                
                Jx_flat_dev, Jy_flat_dev, Jz_flat_dev, rhon_flat_dev, 
                pxx_flat_dev , pxy_flat_dev, pxz_flat_dev, pyy_flat_dev, pyz_flat_dev, pzz_flat_dev
            );

            cudaMemcpyAsync(&part->x[offset], &x_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->y[offset], &y_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->z[offset], &z_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->u[offset], &u_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->v[offset], &v_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->w[offset], &w_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);

            cudaStreamSynchronize(cudaStreams[stream_idx]);

            start_index_stream = start_index_stream + max_num_particles_per_stream;

            if( (start_index_stream + max_num_particles_per_stream) > number_of_particles_batch)
            {
                end_index_stream = number_of_particles_batch - 1;
            }
            else
            {
                end_index_stream += max_num_particles_per_stream;
            }
        }
    
        for(int j = 0; j < number_of_streams; j++)
        {
            cudaStreamDestroy(cudaStreams[j]);
        }

        cudaFree(x_dev);
        cudaFree(y_dev);
        cudaFree(z_dev);
        cudaFree(u_dev);
        cudaFree(v_dev);
        cudaFree(w_dev);
        cudaFree(q_dev);

        // Update indices for next batch
        start_index_batch = start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH;

        if( (start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH) > part->npmax)
        {
            end_index_batch = part->npmax - 1;
        }
        else
        {
            end_index_batch += NUMBER_OF_PARTICLES_PER_BATCH;
        }

    }
    
    // mover_PC:
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    // interpP2G:
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


    // Common mover_PC & interpP2G:
    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);

    // mover_PC:
    cudaFree(Ex_flat_dev);
    cudaFree(Ey_flat_dev);
    cudaFree(Ez_flat_dev);
    cudaFree(Bxn_flat_dev);
    cudaFree(Byn_flat_dev);
    cudaFree(Bzn_flat_dev);

    // interpP2G:
    cudaFree(Jx_flat_dev);
    cudaFree(Jy_flat_dev);
    cudaFree(Jz_flat_dev);
    cudaFree(rhon_flat_dev);
    cudaFree(pxx_flat_dev);
    cudaFree(pxy_flat_dev);
    cudaFree(pxz_flat_dev);
    cudaFree(pyy_flat_dev);
    cudaFree(pyz_flat_dev);
    cudaFree(pzz_flat_dev);

}

