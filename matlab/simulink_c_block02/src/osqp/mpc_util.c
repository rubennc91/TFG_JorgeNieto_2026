#include "mpc_util.h"
#include "osqp.h"

#define Q2 10000.0f
#define Q2N 100000.0f

void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float inv[2][2]) {
    c_float determinant = a * d - b * c;

    c_float inv_determinant = 1.0 / determinant;

    // Calcula os elementos da matriz inversa
    inv[0][0] = d * inv_determinant;
    inv[0][1] = -b * inv_determinant;
    inv[1][0] = -c * inv_determinant;
    inv[1][1] = a * inv_determinant;
}
// Função para multiplicar uma matriz 2x2 por um vetor 2x1
void multiplyMatrixVector(c_float Ex[2][2], c_float u[2], c_float result[2]) {
    int i,j;
    for (i = 0; i < 2; i++) {
        result[i] = 0;
        for (j = 0; j < 2; j++) {
            result[i] += Ex[i][j] * u[j];
        }
    }
}
void referencia(OSQPWorkspace *work, c_float* q_new, c_float ref){
    q_new[1] = -ref*Q2;
    q_new[4] = -ref*Q2;
    q_new[7] = -ref*Q2N;

    osqp_update_lin_cost(work, q_new);
}
// Função para calcular v = Ax + Ex * u
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]) {
    int i;
    c_float Exu[2];
    multiplyMatrixVector(Ex, u, Exu);

    for (i = 0; i < 2; i++) {
        v[i] = Ax[i] + Exu[i];
    }
}
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00)
{
     int i;
//
     //atualiza as restrições das condições iniciais
     for (i = 0; i < 3; i++) {
        l_new[i] = -x[i];
        u_new[i] = -x[i];
     }
     //atualiza as restrições de v00
     l_new[9] = v00[0];
     u_new[9] = v00[0];
     l_new[10] = v00[1];
     u_new[10] = v00[1];
     //atualiza as restrições de delta_u
    for (i = 15; i < 19; i++) {
            l_new[i] = -x[1]*0.3f;
            u_new[i] = x[1]*0.3f;
    }    
}
void atualizar_restricao_v(OSQPWorkspace *work, c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax)
{
     int i;
     c_float v_min[2];
     c_float v_max[2];
     c_float u_min[2];
     c_float u_max[2];

     u_min[0] = -0.5f*vdc;
     u_min[1] = -0.5f*vdc;
     u_max[0] = 0.5f*vdc;
     u_max[1] = 0.5f*vdc;
    
     calculateV(u_min, Einv, Ax, v_min);
     calculateV(u_max, Einv, Ax, v_max);

     l_new[11] = v_min[0];
     u_new[11] = v_max[0];
     l_new[12] = v_min[1];
     u_new[12] = v_max[1];
     l_new[13] = v_min[0];
     u_new[13] = v_max[0];
     l_new[14] = v_min[1];
     u_new[14] = v_max[1];

     osqp_update_bounds(work, l_new, u_new);
}
void atualizar_A(OSQPWorkspace *work, c_float Einv[2][2]){

    c_float A_new[12] = {
    Einv[0][0],
    Einv[0][1],
    Einv[0][0],
    -Einv[0][0],
    Einv[0][0],
    Einv[0][1],
    -Einv[0][1],
    Einv[0][1],
    Einv[0][0],
    -Einv[0][0],
    Einv[0][1],
    -Einv[0][1]};
    c_int A_idx[12] = {18,21,23,25,27,31,32,33,35,37,41,42};


    // c_float A_new[18] = {Einv[0][0],Einv[1][0],Einv[0][1],Einv[0][0],Einv[1][0],-Einv[0][0],-Einv[1][0],
    // Einv[0][0],Einv[1][0],Einv[0][1],-Einv[0][1],Einv[0][1],Einv[0][0],
    // Einv[1][0],-Einv[0][0],-Einv[1][0],Einv[0][1],-Einv[0][1]};
    // c_int A_idx[18] = {18,19,21,23,24,25,26,27,28,31,32,33,35,36,37,38,41,42};
    osqp_update_A(work, A_new, A_idx, 12);
    /*work->data->Adata_x[18] = Einv[1][1];
    work->data->Adata_x[19] = Einv[2][1];
    work->data->Adata_x[21] = Einv[1][2];
    work->data->Adata_x[22] = Einv[2][2];

    work->data->Adata_x[24] = Einv[1][1];
    work->data->Adata_x[25] = Einv[2][1];
    work->data->Adata_x[26] = -Einv[1][1];
    work->data->Adata_x[27] = -Einv[2][1];

    work->data->Adata_x[28] = Einv[1][1];
    work->data->Adata_x[29] = Einv[2][1];
    work->data->Adata_x[31] = Einv[1][2];
    work->data->Adata_x[32] = Einv[2][2];

    work->data->Adata_x[33] = -Einv[1][2];
    work->data->Adata_x[34] = -Einv[2][2];
    work->data->Adata_x[35] = Einv[1][2];
    work->data->Adata_x[36] = Einv[2][2];

    work->data->Adata_x[38] = Einv[1][1];
    work->data->Adata_x[39] = Einv[2][1];
    work->data->Adata_x[40] = -Einv[1][1];
    work->data->Adata_x[41] = -Einv[2][1];

    work->data->Adata_x[43] = Einv[1][2];
    work->data->Adata_x[44] = Einv[2][2];
    work->data->Adata_x[45] = -Einv[1][2];
    work->data->Adata_x[46] = -Einv[2][2];
*/
}

// void atualizar_restricao(OSQPWorkspace *work, c_float* x, c_float* v00)
// {
//      int i;
// //
//      //atualiza as restrições das condições iniciais
//      for (i = 0; i < 3; i++) {
//          work->data->l[i] = -x[i];
//          work->data->u[i] = -x[i];
//      }
//      //atualiza as restrições de v00
//      work->data->l[9] = v00[0];
//      work->data->u[9] = v00[0];
//      work->data->l[10] = v00[1];
//      work->data->u[10] = v00[1];
//      //atualiza as restrições de delta_u
//     for (i = 15; i < 19; i++) {
//             work->data->l[i] = -x[1]*0.3f;
//             work->data->u[i] = x[1]*0.3f;
//     }    
// }