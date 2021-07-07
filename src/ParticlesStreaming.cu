#include "Particles.h"
#include "ParticlesBatching.h"
#include "ParticlesStreaming.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define NUMBER_OF_PARTICLES_PER_BATCH 1024000
#define MAX_NUMBER_OF_STREAMS 5
#define NUMBER_OF_STREAMS_PER_BATCH 4

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

    // Calculation done later to compute free space after allocating space on the GPU fo
    // other variables below, the assumption is that these variables fit in the GPU memory 
    // and mini batching is implemented only taking into account particles

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

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

        long int number_of_particles_batch = end_index_batch - start_index_batch + 1; // number of particles in  a batch
        size_t batch_size_per_attribute = number_of_particles_batch * sizeof(FPpart); // size of the attribute per batch in bytes x,z,y,u,v,w

        long int number_of_particles_stream = 0, stream_size_per_attribute = 0, number_of_streams = 0, stream_offset = 0, offset = 0, start_index_stream = 0, end_index_stream = 0, max_num_particles_per_stream = 0;

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

            stream_offset = start_index_stream;
            offset = stream_offset + start_index_batch; // batch offset + stream_offset

            cudaMemcpyAsync(&x_dev[stream_offset], &part->x[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&y_dev[stream_offset], &part->y[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&z_dev[stream_offset], &part->z[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&u_dev[stream_offset], &part->u[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&v_dev[stream_offset], &part->v[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&w_dev[stream_offset], &part->w[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&q_dev[stream_offset], &part->q[offset], number_of_particles_stream * sizeof(FPinterp), cudaMemcpyHostToDevice, cudaStreams[stream_idx]);

            // start subcycling
            for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++){

            // Call GPU kernel

                single_particle_kernel<<<(number_of_particles_stream + TPB - 1)/TPB, TPB, 0, cudaStreams[stream_idx]>>>(
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
                    part->NiterMover, number_of_particles_stream, stream_offset
                );

            } // end of one particle


            cudaMemcpyAsync(&part->x[offset], &x_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->y[offset], &y_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->z[offset], &z_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->u[offset], &u_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->v[offset], &v_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->w[offset], &w_dev[stream_offset], stream_size_per_attribute, cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&part->q[offset], &q_dev[stream_offset], number_of_particles_stream * sizeof(FPinterp), cudaMemcpyDeviceToHost, cudaStreams[stream_idx]);

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
        
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // Clean up
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

    // Calculation done later to compute free space after allocating space on the GPU for 
    // other variables below, the assumption is that these variables fit in the GPU memory 
    // and mini batching is implemented only taking into account particles

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

    free_bytes = queryFreeMemoryOnGPU();
    total_size_particles = sizeof(FPpart) * part->npmax * 6 + sizeof(FPinterp) * part->npmax; // for x,y,z,u,v,w and q
    
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

    cudaStream_t cudaStreams[MAX_NUMBER_OF_STREAMS];

    for(i = 0; i < number_of_batches; i++)
    {

        long int number_of_particles_batch = end_index_batch - start_index_batch + 1; // number of particles in  a batch
        size_t batch_size = number_of_particles_batch * sizeof(FPpart); // size of the batch in bytes

        long int number_of_particles_stream = 0, stream_size_per_attribute = 0, number_of_streams = 0, stream_offset = 0, offset = 0, start_index_stream = 0, end_index_stream = 0, max_num_particles_per_stream = 0;

        cudaMalloc(&x_dev, batch_size);
        cudaMalloc(&y_dev, batch_size);
        cudaMalloc(&z_dev, batch_size);
        cudaMalloc(&u_dev, batch_size);
        cudaMalloc(&v_dev, batch_size);
        cudaMalloc(&w_dev, batch_size);
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

            stream_offset = start_index_stream;
            offset = stream_offset + start_index_batch; // batch offset + stream_offset

            cudaMemcpyAsync(&x_dev[stream_offset], &part->x[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&y_dev[stream_offset], &part->y[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&z_dev[stream_offset], &part->z[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&u_dev[stream_offset], &part->u[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&v_dev[stream_offset], &part->v[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&w_dev[stream_offset], &part->w[offset], stream_size_per_attribute, cudaMemcpyHostToDevice, cudaStreams[stream_idx]);
            cudaMemcpyAsync(&q_dev[stream_offset], &part->q[offset], number_of_particles_stream * sizeof(FPinterp), cudaMemcpyHostToDevice, cudaStreams[stream_idx]);

            // Call GPU kernel
            interP2G_kernel<<<(number_of_particles_stream + TPB - 1)/TPB, TPB, 0, cudaStreams[stream_idx]>>>(
                x_dev, y_dev, z_dev, u_dev, v_dev, w_dev, q_dev, 
                XN_flat_dev, YN_flat_dev, ZN_flat_dev, 
                grd->nxn, grd->nyn, grd->nzn, 
                grd->xStart, grd->yStart, grd->zStart, 
                grd->invdx, grd->invdy, grd->invdz, grd->invVOL, 
                Jx_flat_dev, Jy_flat_dev, Jz_flat_dev, rhon_flat_dev, 
                pxx_flat_dev , pxy_flat_dev, pxz_flat_dev, pyy_flat_dev, pyz_flat_dev, pzz_flat_dev, 
                number_of_particles_stream, stream_offset
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
        if ((start_index_batch + NUMBER_OF_PARTICLES_PER_BATCH) > part->npmax)
        {
            end_index_batch = part->npmax - 1;
        }
        else
        {
            end_index_batch += NUMBER_OF_PARTICLES_PER_BATCH;
        }

    }

    // Copy memory back to CPU (only the parts that have been modified inside the kernel)
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
    
    // Clean up
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