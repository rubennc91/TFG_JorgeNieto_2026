#include "mpc_util.h"
#include "workspace.h"
#include "osqp.h"
#include "lin_alg.h"
#include "simulink_block.h"
#include <math.h>

#define Q1 1000.0f
#define R 0.1f
#define L 0.005f
#define Cap 0.001f
#define w 314.159265f
#define REFERENCIA 380.0f
#define V_AMP 180.0f

#define OSQP_N 15
#define OSQP_M 19

// --- IMPORTANTE: Usamos extern para no redefinir la variable que ya está en workspace.c ---
extern OSQPWorkspace workspace;

void init_workspace_manually() {
    // Conectar punteros de estructura 'scaling' a arrays globales
    // Esto es necesario por si scaling.c accede a scaling.D
    scaling.D    = scaling_D;
    scaling.Dinv = scaling_Dinv;
    scaling.E    = scaling_E;
    scaling.Einv = scaling_Einv;
    scaling.c    = 1.0;
    scaling.cinv = 1.0;

    // Conectar solution pointers
    solution.x = xsolution;
    solution.y = ysolution;
}

// Declaraciones externas (asegúrate de que coinciden con mpc_util.h actualizado)
void referencia(c_float* q_new, c_float ref);
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]);
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00);
void atualizar_restricao_v(c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax);
void atualizar_A(c_float Einv[2][2]);
void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float m[2][2]);
void multiplyMatrixVector(c_float m[2][2], c_float v[2], c_float res[2]);

void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2])
{
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE s_axilite port=x_ini
    #pragma HLS INTERFACE s_axilite port=Vsd
    #pragma HLS INTERFACE s_axilite port=Vsq
    #pragma HLS INTERFACE s_axilite port=iL
    #pragma HLS INTERFACE s_axilite port=u00
    #pragma HLS INTERFACE s_axilite port=outputVector

    static int is_initialized = 0;
    if (!is_initialized) {
        init_workspace_manually();
        is_initialized = 1;
    }

    float vdc, Ex11, Ex12, Ex21, Ex22, Ax01, Ax02, Rl;
    float Ax[2];
    float Ex[2][2];
    float Einv[2][2];
    float z_ini[3];
    float v00[2];
    float v[2];
    float u[2];

    c_float q_new[OSQP_N];
    c_float l_new[OSQP_M];
    c_float u_new[OSQP_M];
    float ref = REFERENCIA;

    vdc = x_ini[2];

    if (fabsf(iL) < 0.1f) {
        Rl = 10000.0f;
    } else {
        Rl = vdc/iL;
    }

    prea_vec_copy(qdata, q_new, OSQP_N);
    prea_vec_copy(ldata, l_new, OSQP_M);
    prea_vec_copy(udata, u_new, OSQP_M);

    Ax01 = - w*x_ini[0] - (R*x_ini[1])/L;

    float term = (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap * x_ini[2]);
    float term2 = 1.0f / (2.0f * Cap * x_ini[2]);
    Ax02 = (1.0f / (Cap * Rl) + term / x_ini[2]) * (x_ini[2] / (Cap * Rl) - term) -
            (3.0f * (w * x_ini[0] + (R * x_ini[1]) / L) * (Vsq - 2.0f * R * x_ini[1])) * term2 +
            (3.0f * (Vsd - 2.0f * R * x_ini[0]) * (w * x_ini[1] + Vsd / L - (R * x_ini[0]) / L)) * term2;

    Ex11 = 0.0f; Ex12 = -1.0f/L;
    float termL = 1.0f / (2.0f * Cap * L * x_ini[2]);
    Ex21 = -(3.0f * (Vsd - 2.0f * R * x_ini[0])) * termL;
    Ex22 = -(3.0f * (Vsq - 2.0f * R * x_ini[1])) * termL;

    Ax[0] = Ax01; Ax[1] = Ax02;
    Ex[0][0] = Ex11; Ex[0][1] = Ex12;
    Ex[1][0] = Ex21; Ex[1][1] = Ex22;

    inverse_matrix_2x2(Ex11, Ex12, Ex21, Ex22, Einv);

    z_ini[0] = x_ini[1]; z_ini[1] = x_ini[2]; z_ini[2] = term - x_ini[2] / (Cap * Rl);

    referencia(q_new, ref);
    calculateV(Ax, Ex, u00, v00);
    atualizar_restricao(l_new, u_new, z_ini, v00);
    atualizar_restricao_v(l_new, u_new, z_ini[1], Einv, Ax);
    atualizar_A(Einv);

    osqp_solve();

    v[0] = work_x[11];
    v[1] = work_x[12];

    v[0] = v[0] - Ax[0];
    v[1] = v[1] - Ax[1];

    multiplyMatrixVector(Einv, v, u);

    u[0] = u[0]/vdc;
    u[1] = u[1]/vdc;

    outputVector[0] = work_x[11];
    outputVector[1] = work_x[12];
}
