#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpc_mat_workspace.h" // Datos del solver generados
#include "gold_data.h"         // Tus datos de referencia de MATLAB

// Tolerancia para float (OSQP en FPGA suele variar ligeramente de MATLAB double)
#define TOLERANCIA 0.05f

// Prototipo de tu función TOP en HLS
void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2]);

int main() {
    printf("--- INICIANDO VALIDACION OSQP EN HLS (FLOAT) ---\n");
    printf("Muestras a procesar: %d\n", N);

    float x_in[3];
    float u0_in[2];
    float dut_output[2];
    float vsd_val, vsq_val;
    float iL_const = 380.0f;

    int error_count = 0;
    float max_error = 0.0f;

    for (int i = 0; i < N; i++) {
        // 1. Carga de datos de entrada
        x_in[0] = gold_x_ini[i][0];
        x_in[1] = gold_x_ini[i][1];
        x_in[2] = gold_x_ini[i][2];
        vsd_val = gold_Vsd[i][0];
        vsq_val = gold_Vsq[i][0];
        u0_in[0] = gold_u0[i][0];
        u0_in[1] = gold_u0[i][1];

        // 2. Ejecución del bloque HLS
        myFunction(x_in, vsd_val, vsq_val, iL_const, u0_in, dut_output);

        // 3. Comparación con Gold Data (Salidas 0 y 1)
        for (int j = 0; j < 2; j++) {
            float diff = fabsf(dut_output[j] - gold_outputVector[i][j]);

            if (diff > max_error) max_error = diff;

            if (diff > TOLERANCIA) {
                printf("Error Muestra %d [%d]: Gold=%f, HLS=%f, Diff=%f\n",
                        i, j, gold_outputVector[i][j], dut_output[j], diff);
                error_count++;
            }
        }

        if (error_count > 20) {
            printf("Demasiados errores. Abortando...\n");
            break;
        }
    }

    printf("--------------------------------------------\n");
    if (error_count == 0) {
        printf("RESULTADO: ¡TEST PASADO!\n");
        printf("Error max: %f\n", max_error);
    } else {
        printf("RESULTADO: TEST FALLIDO (%d errores)\n", error_count);
    }
    return (error_count == 0) ? 0 : 1;
}
