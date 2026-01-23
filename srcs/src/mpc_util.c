#include "mpc_util.h"
#include "osqp.h"

#define Q2 10000.0f
#define Q2N 100000.0f

void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float inv[2][2]) {
    c_float determinant = a * d - b * c;
    c_float inv_determinant = 1.0f / determinant;
    inv[0][0] = d * inv_determinant;
    inv[0][1] = -b * inv_determinant;
    inv[1][0] = -c * inv_determinant;
    inv[1][1] = a * inv_determinant;
}

void multiplyMatrixVector(c_float Ex[2][2], c_float u[2], c_float result[2]) {
    int i,j;
    for (i = 0; i < 2; i++) {
        result[i] = 0;
        for (j = 0; j < 2; j++) {
            result[i] += Ex[i][j] * u[j];
        }
    }
}

void referencia(c_float* q_new, c_float ref){
    // YA NO HAY 'work' como argumento
    q_new[1] = -ref*10000.0f; // Q2
    q_new[4] = -ref*10000.0f;
    q_new[7] = -ref*100000.0f; // Q2N
    osqp_update_lin_cost(q_new);
}

void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]) {
    int i;
    c_float Exu[2];
    multiplyMatrixVector(Ex, u, Exu);
    for (i = 0; i < 2; i++) {
        v[i] = Ax[i] + Exu[i];
    }
}

void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00) {
     int i;
     for (i = 0; i < 3; i++) {
        l_new[i] = -x[i];
        u_new[i] = -x[i];
     }
     l_new[9] = v00[0]; u_new[9] = v00[0];
     l_new[10] = v00[1]; u_new[10] = v00[1];
    for (i = 15; i < 19; i++) {
            l_new[i] = -x[1]*0.3f;
            u_new[i] = x[1]*0.3f;
    }    
}

void atualizar_restricao_v(c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax) {
     c_float v_min[2], v_max[2], u_min[2], u_max[2];
     u_min[0] = -0.5f*vdc; u_min[1] = -0.5f*vdc;
     u_max[0] = 0.5f*vdc; u_max[1] = 0.5f*vdc;
    
     calculateV(u_min, Einv, Ax, v_min);
     calculateV(u_max, Einv, Ax, v_max);

     l_new[11] = v_min[0]; u_new[11] = v_max[0];
     l_new[12] = v_min[1]; u_new[12] = v_max[1];
     l_new[13] = v_min[0]; u_new[13] = v_max[0];
     l_new[14] = v_min[1]; u_new[14] = v_max[1];

     osqp_update_bounds(l_new, u_new);
}

void atualizar_A(c_float Einv[2][2]){
    c_float A_new[12] = {
    Einv[0][0], Einv[0][1], Einv[0][0], -Einv[0][0],
    Einv[0][0], Einv[0][1], -Einv[0][1], Einv[0][1],
    Einv[0][0], -Einv[0][0], Einv[0][1], -Einv[0][1]};
    c_int A_idx[12] = {18,21,23,25,27,31,32,33,35,37,41,42};
    osqp_update_A(A_new, A_idx, 12);
}
