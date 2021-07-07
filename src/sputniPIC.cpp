/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU
#include "ParticlesBatching.h"
#include "ParticlesStreaming.h"
#include "ParticlesUnitedKernel.h"

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include "CompareValues.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMoverCPU, iInterpCPU, eMoverCPU = 0.0, eInterpCPU= 0.0;
    double iMoverGPU, iInterpGPU, eMoverGPU = 0.0, eInterpGPU= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield fieldCPU;
    EMfield fieldGPU;
    field_allocate(&grd,&fieldCPU);
    field_allocate(&grd,&fieldGPU);
    EMfield_aux field_auxCPU;
    EMfield_aux field_auxGPU;
    field_aux_allocate(&grd,&field_auxCPU);
    field_aux_allocate(&grd,&field_auxGPU);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *idsCPU = new interpDensSpecies[param.ns];
    interpDensSpecies *idsGPU = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++) {
        interp_dens_species_allocate(&grd,&idsCPU[is],is);  // Only changes idsCPU
        interp_dens_species_allocate(&grd,&idsGPU[is],is);
    }
    // Net densities
    interpDensNet idnCPU;
    interpDensNet idnGPU;
    interp_dens_net_allocate(&grd,&idnCPU);  // Only changes idnCPU
    interp_dens_net_allocate(&grd,&idnGPU);
    
    // Allocate Particles
    particles *partCPU = new particles[param.ns];
    particles *partGPU = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&partCPU[is],is);
        particle_allocate(&param,&partGPU[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&fieldCPU,&field_auxCPU,partCPU,idsCPU);  // Changes fieldCPU, field_auxCPU, partCPU, idsCPU
    initGEM(&param,&grd,&fieldGPU,&field_auxGPU,partGPU,idsGPU);

    std::cout << " STARTING SIMULATION " << std::endl;

    // **********************************************************//
    // **** CPU CPU CPU CPU  *** //
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **** CPU CPU CPU CPU  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idnCPU,idsCPU,&grd,param.ns);  // Only idnCPU & idsCPU is changed
        
        // implicit mover
        iMoverCPU = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++) {
            mover_PC(&partCPU[is],&fieldCPU,&grd,&param);  // Only partCPU is changed
        }
        eMoverCPU += (cpuSecond() - iMoverCPU); // stop timer for mover
        
        // interpolation particle to grid
        iInterpCPU = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++) {
            interpP2G(&partCPU[is],&idsCPU[is],&grd);  // Only idsCPU is changed
        }
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++) {
            applyBCids(&idsCPU[is],&grd,&param);  // Only idsCPU is changed
        }
        // sum over species
        sumOverSpecies(&idnCPU,idsCPU,&grd,param.ns);  // Only idnCPU is changed
        // interpolate charge density from center to node
        applyBCscalarDensN(idnCPU.rhon,&grd,&param);  // Only idnCPU is changed
    
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd, &fieldCPU, "cpu");
            VTK_Write_Scalars(cycle, &grd,idsCPU,&idnCPU, "cpu");
        }
        
        eInterpCPU += (cpuSecond() - iInterpCPU); // stop timer for interpolation
    
    }  // end of one PIC cycle

    // **********************************************************//
    // **** GPU GPU GPU GPU  *** //
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **** GPU GPU GPU GPU  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idnGPU,idsGPU,&grd,param.ns);

        // Everything at once:
        /*for (int is=0; is < param.ns; is++) {

            mover_AND_interpP2G_stream(&partGPU[is], &fieldGPU, &grd, &param, &idsGPU[is]);

        }*/
        
        // implicit mover
        iMoverGPU = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++) {
            mover_PC_GPU_basic(&partGPU[is],&fieldGPU,&grd,&param);
            //mover_GPU_batch(&partGPU[is],&fieldGPU,&grd,&param);
            //mover_GPU_stream(&partGPU[is],&fieldGPU,&grd,&param);
        }
        eMoverGPU += (cpuSecond() - iMoverGPU); // stop timer for mover
        
        // interpolation particle to grid
        iInterpGPU = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++) {
            interpP2G_GPU_basic(&partGPU[is],&idsGPU[is],&grd);
            //interpP2G_GPU_stream(&partGPU[is],&idsGPU[is],&grd);
            //interpP2G_GPU_batch(&partGPU[is],&idsGPU[is],&grd);
        }
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++) {
            applyBCids(&idsGPU[is],&grd,&param);
        }
        // sum over species
        sumOverSpecies(&idnGPU,idsGPU,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idnGPU.rhon,&grd,&param);
    
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd, &fieldGPU, "gpu");
            VTK_Write_Scalars(cycle, &grd,idsGPU,&idnGPU, "gpu");
        }
        
        eInterpGPU += (cpuSecond() - iInterpGPU); // stop timer for interpolation
    
    }  // end of one PIC cycle

    compareValues(param, grd, idsGPU, idsCPU, idnGPU, idnCPU);
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);

    // **************************//
    // **** Deallocate CPU  *** //
    // **************************//
    field_deallocate(&grd,&fieldCPU);
    interp_dens_net_deallocate(&grd,&idnCPU);
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&idsCPU[is]);
        particle_deallocate(&partCPU[is]);
    }

    // **************************//
    // **** Deallocate GPU  *** //
    // **************************//
    field_deallocate(&grd,&fieldGPU);
    interp_dens_net_deallocate(&grd,&idnGPU);
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&idsGPU[is]);
        particle_deallocate(&partGPU[is]);
    }
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   CPU Mover Time / Cycle   (s) = " << eMoverCPU/param.ncycles << std::endl;
    std::cout << "   CPU Interp. Time / Cycle (s) = " << eInterpCPU/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   GPU Mover Time / Cycle   (s) = " << eMoverGPU/param.ncycles << std::endl;
    std::cout << "   GPU Interp. Time / Cycle (s) = " << eInterpGPU/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


