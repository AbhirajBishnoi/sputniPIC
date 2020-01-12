#include "Particles.h"
#include "ParticlesBatching.h"
#include "ParticlesStreaming.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define NUMBER_OF_PARTICLES_PER_BATCH 100000


/** particle mover for GPU with batching */

int mover_GPU_stream(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
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

    size_t free_bytes = 0;

    int i, total_size_particles, start_index_batch, end_index_batch, number_of_batches;

    // free_bytes = queryFreeMemoryOnGPU();

    // calculation done later to compute free space after allocating space on the GPU for other variables below, the assumption is that these variables fit in the GPU memory and mini batching is implemented only taking into account particles

    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);  

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    free_bytes = queryFreeMemoryOnGPU();
    total_size_particles = sizeof(FPpart) * part->npmax * 6; // for x,y,z,u,v,w
    
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
        number_of_batches = part->npmax / NUMBER_OF_PARTICLES_PER_BATCH + 1; // works because of integer division
    }
       
    cudaStream_t *cudaStreams = new cudaStream_t[number_of_batches];
    for(i = 0; i < number_of_batches; i++) {
        cudaStreamCreate(&cudaStreams[i]);
    }

    for(i = 0; i < number_of_batches; i++)
    {
        std::cout << "BATCH!" << std::endl;

        cudaStreamCreate(&cudaStreams[i]);

        int number_of_particles_batch = end_index_batch - start_index_batch + 1; // number of particles in  a batch
        size_t batch_size = number_of_particles_batch * sizeof(FPpart); // size of the batch in bytes

        std::cout << "num_of_particles_batch" << number_of_particles_batch << " batch_size : " << batch_size << std::endl;
        std::cout << "start_index" << start_index_batch << " end_index : " << end_index_batch << std::endl;

        if (number_of_batches > 1) {
            cudaMallocHost(part->x + start_index_batch, batch_size);
            cudaMallocHost(part->y + start_index_batch, batch_size);
            cudaMallocHost(part->z + start_index_batch, batch_size);
            cudaMallocHost(part->u + start_index_batch, batch_size);
            cudaMallocHost(part->v + start_index_batch, batch_size);
            cudaMallocHost(part->w + start_index_batch, batch_size);
        }
        
        cudaMalloc(&x_dev, batch_size);
        cudaMalloc(&y_dev, batch_size);
        cudaMalloc(&z_dev, batch_size);
        cudaMalloc(&u_dev, batch_size);
        cudaMalloc(&v_dev, batch_size);
        cudaMalloc(&w_dev, batch_size);
        cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));

        if (number_of_batches > 1) {
            cudaMemcpyAsync(x_dev, part->x + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(y_dev, part->y + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(z_dev, part->z + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(u_dev, part->u + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(v_dev, part->v + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(w_dev, part->w + start_index_batch, batch_size, cudaMemcpyHostToDevice, cudaStreams[i]);
            cudaMemcpyAsync(q_dev, part->q + start_index_batch, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice, cudaStreams[i]);
        } else {
            cudaMemcpy(x_dev, (part->x + start_index_batch), batch_size, cudaMemcpyHostToDevice); 
            cudaMemcpy(y_dev, (part->y + start_index_batch), batch_size, cudaMemcpyHostToDevice);
            cudaMemcpy(z_dev, (part->z + start_index_batch), batch_size, cudaMemcpyHostToDevice); 
            cudaMemcpy(u_dev, (part->u + start_index_batch), batch_size, cudaMemcpyHostToDevice); 
            cudaMemcpy(v_dev, (part->v +  start_index_batch), batch_size, cudaMemcpyHostToDevice); 
            cudaMemcpy(w_dev, (part->w + start_index_batch), batch_size, cudaMemcpyHostToDevice);
            cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);
        }

        // start subcycling
        for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++){

            // Call GPU kernel
            single_particle_kernel<<<(number_of_particles_batch + TPB - 1)/TPB, TPB>>>(
                x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, 
                XN_flat_dev, YN_flat_dev, ZN_flat_dev, 
                grd->nxn, grd->nyn, grd->nzn, 
                grd->xStart, grd->yStart, grd->zStart, 
                grd->invdx, grd->invdy, grd->invdz, 
                grd->Lx, grd->Ly, grd->Lz, grd->invVOL, 
                Ex_flat_dev, Ey_flat_dev, Ez_flat_dev, 
                Bxn_flat_dev, Byn_flat_dev, Bzn_flat_dev, 
                param->PERIODICX, param->PERIODICY, param->PERIODICZ, 
                dt_sub_cycling, dto2, qomdt2, 
                part->NiterMover, 
                part->nop  // TODO: Change this to number_of_particles_batch
            );

        } // end of one particle


        // copy memory back to CPU (only the parts that have been modified inside the kernel)

        cudaMemcpy( (part->x + start_index_batch), x_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->y + start_index_batch), y_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->z + start_index_batch), z_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->u + start_index_batch), u_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->v + start_index_batch), v_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->w + start_index_batch), w_dev, batch_size, cudaMemcpyDeviceToHost);

        cudaFree(x_dev);
        cudaFree(y_dev);
        cudaFree(z_dev);
        cudaFree(u_dev);
        cudaFree(v_dev);
        cudaFree(w_dev);
        cudaFree(q_dev);

        // update indices for next batch

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
        
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // clean up

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

/** Interpolation with batching */

void interpP2G_GPU_stream(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp * q_dev = NULL, *Jx_flat_dev = NULL, *Jy_flat_dev = NULL, *Jz_flat_dev = NULL, *rhon_flat_dev = NULL, *pxx_flat_dev = NULL, *pxy_flat_dev = NULL, *pxz_flat_dev = NULL, *pyy_flat_dev = NULL, *pyz_flat_dev = NULL, *pzz_flat_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL;

    size_t free_bytes = 0;

    int i, total_size_particles, start_index_batch, end_index_batch, number_of_batches;

    // free_bytes = queryFreeMemoryOnGPU();

    // calculation done later to compute free space after allocating space on the GPU for other variables below, the assumption is that these variables fit in the GPU memory and mini batching is implemented only taking into account particles


    cudaMalloc(&Jx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jx_flat_dev, ids->Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Jy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jy_flat_dev, ids->Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Jz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(Jz_flat_dev, ids->Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&rhon_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(rhon_flat_dev, ids->rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxx_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxx_flat_dev, ids->pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxy_flat_dev, ids->pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pxz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pxz_flat_dev, ids->pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pyy_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pyy_flat_dev, ids->pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pyz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pyz_flat_dev, ids->pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&pzz_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(pzz_flat_dev, ids->pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);


    free_bytes = queryFreeMemoryOnGPU();
    total_size_particles = sizeof(FPpart) * part->npmax * 6; // for x,y,z,u,v,w
    
    start_index_batch = 0, end_index_batch = 0;

    // implement mini-batching only in the case where the free space on the GPU isn't enough

    if(free_bytes > total_size_particles)
    {
        start_index_batch = 0;
        end_index_batch = part->npmax - 1 ; // set end_index to the last particle as we are processing in in one batch
        number_of_batches = 1;
    }
    else
    {
        start_index_batch = 0;
        end_index_batch = start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH - 1; // NUM_PARTICLES_PER_BATCH is a hyperparameter set by tuning
        number_of_batches = part->npmax / NUMBER_OF_PARTICLES_PER_BATCH + 1; // works because of integer division
    }
       

    for(i = 0; i < number_of_batches; i++)
    {

        std::cout << "BATCH!" << std::endl;

        int number_of_particles_batch = end_index_batch - start_index_batch + 1; // number of particles in  a batch
            
        size_t batch_size = number_of_particles_batch * sizeof(FPpart); // size of the batch in bytes

        std::cout << "num_of_particles_batch" << number_of_particles_batch << " batch_size : " << batch_size << std::endl;

        std::cout << "start_index" << start_index_batch << " end_index : " << end_index_batch << std::endl;

        cudaMalloc(&x_dev, batch_size);            
        cudaMemcpy(x_dev, (part->x + start_index_batch), batch_size, cudaMemcpyHostToDevice); 

        cudaMalloc(&y_dev, batch_size);
        cudaMemcpy(y_dev, (part->y + start_index_batch), batch_size, cudaMemcpyHostToDevice);

        cudaMalloc(&z_dev, batch_size);
        cudaMemcpy(z_dev, (part->z + start_index_batch), batch_size, cudaMemcpyHostToDevice); 

        cudaMalloc(&u_dev, batch_size);
        cudaMemcpy(u_dev, (part->u + start_index_batch), batch_size, cudaMemcpyHostToDevice); 
            
        cudaMalloc(&v_dev, batch_size);
        cudaMemcpy(v_dev, (part->v +  start_index_batch), batch_size, cudaMemcpyHostToDevice); 

        cudaMalloc(&w_dev, batch_size);
        cudaMemcpy(w_dev, (part->w + start_index_batch), batch_size, cudaMemcpyHostToDevice);
            
        cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
        cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice); 

        // Call GPU kernel

        interP2G_kernel<<<(number_of_particles_batch + TPB - 1)/TPB, TPB>>>( x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, XN_flat_dev, YN_flat_dev, ZN_flat_dev, grd->nxn, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->invVOL, Jx_flat_dev, Jy_flat_dev, Jz_flat_dev, rhon_flat_dev, pxx_flat_dev , pxy_flat_dev, pxz_flat_dev, pyy_flat_dev, pyz_flat_dev, pzz_flat_dev, part->nop);

        cudaDeviceSynchronize();


        // copy memory back to CPU (only the parts that have been modified inside the kernel)

        cudaMemcpy( (part->x + start_index_batch), x_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->y + start_index_batch), y_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->z + start_index_batch), z_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->u + start_index_batch), u_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->v + start_index_batch), v_dev, batch_size, cudaMemcpyDeviceToHost);
        cudaMemcpy( (part->w + start_index_batch), w_dev, batch_size, cudaMemcpyDeviceToHost);

        cudaFree(x_dev);
        cudaFree(y_dev);
        cudaFree(z_dev);
        cudaFree(u_dev);
        cudaFree(v_dev);
        cudaFree(w_dev);
        cudaFree(q_dev);

        // update indices for next batch

    
        start_index_batch = start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH;

        if ((start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH) > part->npmax)
        {
            end_index_batch = part->npmax - 1;
        }
        else
        {
            end_index_batch += NUMBER_OF_PARTICLES_PER_BATCH;
        }

    }

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