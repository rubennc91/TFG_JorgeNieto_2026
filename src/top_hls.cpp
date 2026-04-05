#include "mpc_mat_workspace.h"
#include <math.h>

extern "C" {
    #include "osqp.h"
    extern OSQPFloat mpc_mat_sol_x[15];
}

#define R_VAL 0.1f
#define L_VAL 5e-3f
#define CAP_VAL 1000e-6f
#define W_VAL 314.159265f
#define REF_VAL 380.0f
#define Q2_VAL 10000.0f
#define Q2N_VAL 100000.0f

void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2]) {
    #pragma HLS INTERFACE s_axilite port=x_ini
    #pragma HLS INTERFACE s_axilite port=Vsd
    #pragma HLS INTERFACE s_axilite port=Vsq
    #pragma HLS INTERFACE s_axilite port=iL
    #pragma HLS INTERFACE s_axilite port=u00
    #pragma HLS INTERFACE s_axilite port=outputVector
    #pragma HLS INTERFACE s_axilite port=return

    float vdc = x_ini[2];
    float Rl = (fabsf(iL) < 0.1f) ? 10000.0f : (vdc / iL);

    float Ex11 = 0.0f;
    float Ex12 = -1.0f / L_VAL;
    float Ex21 = -(3.0f * (Vsd - 2.0f * R_VAL * x_ini[0])) / (2.0f * CAP_VAL * L_VAL * vdc);
    float Ex22 = -(3.0f * (Vsq - 2.0f * R_VAL * x_ini[1])) / (2.0f * CAP_VAL * L_VAL * vdc);

    float det = (Ex11 * Ex22) - (Ex12 * Ex21);
    float invDet = 1.0f / det;
    float Einv00 =  Ex22 * invDet;
    float Einv01 = -Ex12 * invDet;
    float Einv10 = -Ex21 * invDet;
    float Einv11 =  Ex11 * invDet;

    OSQPFloat q_new[15] = {0};
    q_new[1] = -REF_VAL * Q2_VAL;
    q_new[4] = -REF_VAL * Q2_VAL;
    q_new[7] = -REF_VAL * Q2N_VAL;
    osqp_update_data_vec(&mpc_mat_solver, q_new, nullptr, nullptr);

    OSQPFloat A_new[12] = {
        (OSQPFloat)Einv00, (OSQPFloat)Einv01, (OSQPFloat)Einv00, -(OSQPFloat)Einv00,
        (OSQPFloat)Einv00, (OSQPFloat)Einv01, -(OSQPFloat)Einv01, (OSQPFloat)Einv01,
        (OSQPFloat)Einv00, -(OSQPFloat)Einv00, (OSQPFloat)Einv01, -(OSQPFloat)Einv01
    };
    OSQPInt A_idx[12] = {18, 21, 23, 25, 27, 31, 32, 33, 35, 37, 41, 42};
    osqp_update_data_mat(&mpc_mat_solver, nullptr, nullptr, 0, A_new, A_idx, 12);

    // Motor completo que SÍ refactoriza y funciona
    osqp_solve(&mpc_mat_solver);

    outputVector[0] = (float)mpc_mat_sol_x[11];
    outputVector[1] = (float)mpc_mat_sol_x[12];
}
