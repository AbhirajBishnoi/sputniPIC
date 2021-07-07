# Particle-In-Cell Simulation

## About

This is the final project for the KTH course [Applied GPU Programming](https://www.kth.se/student/kurser/kurs/DD2360?l=en). All other assignments to this course can be seen [here](https://github.com/olapiv/kth-applied-gpu-programming).


This project provides a Particle-In-Cell (PIC) code-base, which simulates the movement of particles within an electromagnetic field. In simple terms, throughout multiple iterations, first the influence of the electromagnetic field is calculated upon particle positions & velocities and then the positions of the particles are used to calculate the accumulated charge, current and pressure along the grid. The first step is referenced as mover_pc (particle mover), whilst the second step is referred to as interpP2G (particle to grid interpolation). In a typical PIC calculation, one will also find a “fields solver” functionality, which essentially recalculates the electromagnetic field according to the movements of the particles. In our project, this is however assumed as constant for the sake of simplicity.

The skeleton of this code is derived from the open source project [iPIC3D](​https://ipic3d.github.io/​), which provides an implementation highly optimized for the usage with PCs, which is achieved by using MPI processes for communication. However, given the recent emergence of GPU-availability within supercomputers, the potential efficiency savings rise. This project thereby aims to implement GPU-parallelization with Cuda. According to the grading criteria, multiple versions were written to emphasize how special built-in CUDA functionalities can further fine-tune GPU code.

## Cuda Implementation Steps

1. **Grade D**: Porting mover_pc and interpP2G to GPU

    The first basic step to take is to rewrite the central loop of mover_pc and interpP2G, which iterates over part->nop (number of particles). We extract these loops to separated device kernels “single_particle_kernel” and "interP2G_kernel", which also provide the basis of our future steps. Both kernels are called with an arbitrary 64 threads per block, whereby our blocks only have one dimension. Within the kernels we can now calculate the thread index and replace the iteration variable referring to the number-of-particles-loop with this value. This is required to obtain the coordinates and the velocities of the particles, which are given in 1D arrays (x_flat, etc.).

    * All written code for this can be seen in [Particles.cu](src/Particles.cu)
    * The code is executed by calling mover_GPU_basic and interpP2G_GPU_basic in [sputniPIC.cpp](src/sputniPIC.cpp).

2. **Grade C**: Mini-batching GPU code

    This is done to handle a large number of particles, which would potentially not fit onto the GPU memory. In our implementation, we first allocate and copy over the data that is needed for all particles: for mover_pc, the grid & field and for interpP2G, the “densities”. This common data is assumed to fit into  GPU memory without the need for splitting into batches. Then, we check how much free memory is left on the GPU and whether it is less than the amount of memory needed for all particles to be copied over. If so, for the sake of simplicity, we decided for an arbitrary number of particles to be copied over per iteration and divide the total number of particles by this value to obtain the number of required batches. The number of particles is a hyperparameter which can be tuned according to the graphics card being used. In general, we found that larger batch sizes were quicker than using multiple smaller ones.

    * All written code for this can be seen in [ParticlesBatching.cu](src/ParticlesBatching.cu).
    * The code is run by calling mover_GPU_batch and interpP2G_GPU_batch in [sputniPIC.cpp](src/sputniPIC.cpp).

3. **Grade B**: Pinned memory, streams and asynchronous memory copying

    The purpose of asynchronous copying in relation to streams is that copying data to the GPU (in one stream) can be done concurrently to launching a kernel (in another stream). To do so, the data is required to be split up between the streams. Since asynchronous copying however requires the data on the CPU to be pinned, this method may not be effective in case not enough pinnable memory is available.

    After failing to pin an offsetted amount of values in the particle arrays for every stream, which would have had the benefit to be able to control the amount of pinned memory needed, we pinned the entire particle arrays at the beginning of our code. Due to the reasons mentioned beforehand, this may however cause the program to exit prematurely in case required pinned memory exceeds available memory (i.e. very large particle numbers).

    * All written code for this can be seen in [ParticlesStreaming.cu](src/ParticlesStreaming.cu).
    * The code is run by calling mover_GPU_stream and interpP2G_GPU_stream in [sputniPIC.cpp](src/sputniPIC.cpp).

4. **Grade A**: Combining mover_pc and interpP2G into single kernel

    For this step we created a new kernel called "united_kernel", which in essence executes mover_pc and interpP2G in sequence. However, the loop for number of cycles in the mover_pc is moved into the kernel, since one cycle affects the next cycle. Generally, we were also able to copy all other code from ParticlesStreaming.cu, which was called before and after the respective kernel calls. This meant that the field and the densities are moved entirely to and from the device before and after united_kernel is called.

    * All written code for this can be seen in [ParticlesUnitedKernel.cu](src/ParticlesUnitedKernel.cu).
    * The code is run by calling mover_AND_interpP2G_stream in [sputniPIC.cpp](src/sputniPIC.cpp).

## How To Run

1. Make sure to have a workstation with a NVIDIA graphics card.
2. Install CUDA
3. Run "make" on the root directory of this repository
4. Create a "data" directory on the root directory of this repository
5. Run ```./bin/sputniPIC.out inputfiles/GEM_2D.inp```. There is also GEM_3D.inp
6. See results in .data/

For grades D-B, uncomment the respective "mover_GPU_basic"/"mover_GPU_batch"/"mover_GPU_stream" and "interpP2G_GPU_basic"/"interpP2G_GPU_batch"/"interpP2G_GPU_stream" in [sputniPIC.cpp](src/sputniPIC.cpp).

For grade A, uncomment mover_AND_interpP2G_stream in [sputniPIC.cpp](src/sputniPIC.cpp).