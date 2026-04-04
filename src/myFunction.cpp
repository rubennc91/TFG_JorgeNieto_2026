#include "mpc_mat_workspace.h"
#include <math.h>

// Definiciones de constantes en float para HLS
#define Q1 1000.0f
#define Q2 10000.0f
#define Q2N 100000.0f
#define R_val 0.1f
#define L_val 5e-3f
#define Cap_val 1000e-6f
#define w_val 314.15926535f

// Función auxiliar: Inversa 2x2 en float
void inverse_matrix_2x2_f(float a, float b, float c, float d, float inv[2][2]) {
    float determinant = a * d - b * c;
    if (fabsf(determinant) < 1e-9f) return; // Protección contra división por cero
    float inv_det = 1.0f / determinant;
    inv[0][0] = d * inv_det;
    inv[0][1] = -b * inv_det;
    inv[1][0] = -c * inv_det;
    inv[1][1] = a * inv_det;
}

// Función auxiliar: Producto Matriz-Vector 2x2
void multiplyMatrixVector_f(float Ex[2][2], float u[2], float result[2]) {
    for (int i = 0; i < 2; i++) {
        result[i] = Ex[i][0] * u[0] + Ex[i][1] * u[1];
    }
}

void myFunction(const float x_ini[3], float Vsd, float Vsq, float iL, float ref, const float u0[2], float outputVector[4])
{
    float vdc, Rl, Ax01, Ax02;
    float Ax[2], Ex[2][2], Einv[2][2];
    float z_ini[3], v00[2], u00[2], v[2], u[2];
    
    // Vectores de actualización para OSQP
    float q_new[13];
    float l_new[20];
    float u_new[20];

    vdc = x_ini[2];
    Rl = (fabsf(iL) < 0.1f) ? 10000.0f : (vdc / iL);

    u00[0] = (isnan(u0[0])) ? 0.7f : u0[0];
    u00[1] = (isnan(u0[1])) ? 0.0f : u0[1];

    // --- Lógica del Modelo Dinámico ---
    Ax01 = -w_val * x_ini[0] - (R_val * x_ini[1]) / L_val;
    
    // Simplificación de tu fórmula Ax02 para legibilidad en HLS
    float term1 = (1.0f / (Cap_val * Rl) + (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R_val * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap_val * vdc * vdc));
    float term2 = (vdc / (Cap_val * Rl) - (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R_val * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap_val * vdc));
    
    Ax02 = term1 * term2 - 
           (3.0f * (w_val * x_ini[0] + (R_val * x_ini[1]) / L_val) * (Vsq - 2.0f * R_val * x_ini[1])) / (2.0f * Cap_val * vdc) +
           (3.0f * (Vsd - 2.0f * R_val * x_ini[0]) * (w_val * x_ini[1] + Vsd / L_val - (R_val * x_ini[0]) / L_val)) / (2.0f * Cap_val * vdc);

    Ax[0] = Ax01; Ax[1] = Ax02;
    Ex[0][0] = 0;      Ex[0][1] = -1.0f / L_val;
    Ex[2][0] = -(3.0f * (Vsd - 2.0f * R_val * x_ini[0])) / (2.0f * Cap_val * L_val * vdc);
    Ex[1][1] = -(3.0f * (Vsq - 2.0f * R_val * x_ini[1])) / (2.0f * Cap_val * L_val * vdc);

    inverse_matrix_2x2_f(Ex[0][0], Ex[0][1], Ex[2][0], Ex[1][1], Einv);

    z_ini[0] = x_ini[1];
    z_ini[1] = x_ini[2];
    z_ini[2] = (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R_val * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap_val * vdc) - vdc / (Cap_val * Rl);

    // --- Preparar datos para OSQP Solver ---
    for(int i=0; i<13; i++) q_new[i] = mpc_mat_solver.work->data->q[i];
    for(int i=0; i<20; i++) {
        l_new[i] = mpc_mat_solver.work->data->l[i];
        u_new[i] = mpc_mat_solver.work->data->u[i];
    }

    // Actualizar Coste (Referencia)
    q_new[1] = -ref * Q2;
    q_new[4] = -ref * Q2;
    q_new[7] = -ref * Q2N;
    osqp_update_lin_cost(&mpc_mat_solver, q_new);

    // Actualizar Restricciones
    for (int i = 0; i < 3; i++) {
        l_new[i] = -z_ini[i]; u_new[i] = -z_ini[i];
    }
    
    float Exu[2];
    multiplyMatrixVector_f(Ex, u00, Exu);
    v00[0] = Ax[0] + Exu[0];
    v00[1] = Ax[1] + Exu[1];
    
    l_new[9] = v00[0]; u_new[9] = v00[0];
    l_new[10] = v00[1]; u_new[10] = v00[1];

    for (int i = 15; i < 19; i++) {
        l_new[i] = -z_ini[1] * 0.3f;
        u_new[i] = z_ini[1] * 0.3f;
    }

    // Límites de saturación (v_min, v_max)
    float u_min[2] = {-0.5f * vdc, -0.5f * vdc};
    float u_max[2] = {0.5f * vdc, 0.5f * vdc};
    float v_min[2], v_max[2];
    
    // Reutilizamos calculateV logic aquí
    float tmp1[2], tmp2[2];
    multiplyMatrixVector_f(Einv, u_min, tmp1);
    multiplyMatrixVector_f(Einv, u_max, tmp2);
    for(int i=0; i<2; i++){
        v_min[i] = Ax[i] + tmp1[i];
        v_max[i] = Ax[i] + tmp2[i];
    }

    l_new[11] = v_min[0]; u_new[11] = v_max[0];
    l_new[12] = v_min[1]; u_new[12] = v_max[1];
    l_new[13] = v_min[0]; u_new[13] = v_max[0];
    l_new[14] = v_min[1]; u_new[14] = v_max[1];

    osqp_update_bounds(&mpc_mat_solver, l_new, u_new);

    // --- Actualizar Matriz A ---
    float A_new_val[12] = {
        Einv[0][0], Einv[0][1], Einv[0][0], -Einv[0][0],
        Einv[0][0], Einv[0][1], -Einv[0][1], Einv[0][1],
        Einv[0][0], -Einv[0][0], Einv[0][1], -Einv[0][1]
    };
    int A_idx[12] = {18,21,23,25,27,31,32,33,35,37,41,42};
    osqp_update_A(&mpc_mat_solver, A_new_val, A_idx, 12);

    // --- SOLVER ---
    osqp_solve(&mpc_mat_solver);

    // --- Salida ---
    v[0] = mpc_mat_solver.work->x->val[11];
    v[1] = mpc_mat_solver.work->x->val[12];
    
    u[0] = (v[0] - Ax[0]) * Einv[0][0] + (v[1] - Ax[1]) * Einv[0][1];
    u[1] = (v[0] - Ax[0]) * Einv[1][0] + (v[1] - Ax[1]) * Einv[1][1];

    // Saturación final
    for(int i=0; i<2; i++){
        u[i] /= (vdc * 0.5f);
        if (u[i] > 1.0f) u[i] = 1.0f;
        else if (u[i] < -1.0f) u[i] = -1.0f;
        if (isnan(u[i])) u[i] = (i==0) ? 0.7f : 0.0f;
    }

    outputVector[0] = u[0];
    outputVector[1] = u[1];
    outputVector[2] = u[0] * vdc * 0.5f;
    outputVector[3] = u[1] * vdc * 0.5f;
}