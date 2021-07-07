#include "CompareValues.h"

void compareValues(parameters param, grid grd, interpDensSpecies* idsGPU, interpDensSpecies* idsCPU, interpDensNet idnGPU, interpDensNet idnCPU)
{
    int flat_idx = 0;
    int nxn = grd.nxn;
    int nyn = grd.nyn;
    int nzn = grd.nzn;

    float maxErrorIdsRhon = 0.0f;
    for (int is=0; is < param.ns; is++) {
        for (register int i=0; i < nxn; i++) {
            for (register int j=0; j < nyn; j++) {
                for (register int k=0; k < nzn; k++){        
                    flat_idx = get_idx(i, j, k, nyn, nzn);
                    maxErrorIdsRhon = fmax(maxErrorIdsRhon, fabs(
                        idsCPU[is].rhon[i][j][k] - 
                        idsGPU[is].rhon_flat[flat_idx]
                    ));

                    // TODO: Implement more constants here
                }
            }
        }
    }
    std::cout << "Max error idsrhon: " << maxErrorIdsRhon << std::endl;
    std::cout << "------------" << std::endl;

    float valueFlat;
    float sumIdnRhon = 0.0f;
    float sumIdnRhonFlat = 0.0f;
    float avgIdnRhon = 0.0f;
    float avgIdnRhonFlat = 0.0f;

    float errorIdnRhon = 0.0f;
    float maxErrorIdnRhon = 0.0f;
    float sumErrorIdnRhon = 0.0f;
    for (register int i=0; i < nxn; i++){
        for (register int j=0; j < nyn; j++){
            for (register int k=0; k < nzn; k++){
                flat_idx = get_idx(i, j, k, nyn, nzn);
                valueFlat = idnGPU.rhon_flat[flat_idx];

                sumIdnRhon =+ idnCPU.rhon[i][j][k];
                sumIdnRhonFlat =+ valueFlat;

                errorIdnRhon = fabs(idnCPU.rhon[i][j][k] - valueFlat);
                maxErrorIdnRhon = fmax(maxErrorIdnRhon, errorIdnRhon);
                sumErrorIdnRhon =+ errorIdnRhon;

                // TODO: Implement more constants here
            }
        }
    }
    avgIdnRhon = sumIdnRhon / (nxn * nyn * nzn);
    avgIdnRhonFlat = sumIdnRhonFlat / (nxn * nyn * nzn);
    std::cout << "Avg idnrhon: " << avgIdnRhon << std::endl;
    std::cout << "Avg idnrhonFlat: " << avgIdnRhonFlat << std::endl;

    std::cout << "Max error idnrhon: " << maxErrorIdnRhon << std::endl;
    std::cout << "Avg error idnrhon: " << sumErrorIdnRhon / (nxn * nyn * nzn) << std::endl;

}