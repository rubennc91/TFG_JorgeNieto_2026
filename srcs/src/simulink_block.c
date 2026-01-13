// (Línea vacía o borrada)
#include "mpc_util.h"
#include "workspace.h"
#include "osqp.h"
#include "lin_alg.h"
#include "simulink_block.h"
#include <math.h> // Añade math.h si necesitas funciones matemáticas

// Definir constantes como float
#define Q1 1000.0f
#define R 0.1f
#define L 0.005f          // 5e-3f explícito
#define Cap 0.001f        // 1000e-6f explícito
#define w 314.159265f     // float
#define REFERENCIA 380.0f
#define V_AMP 180.0f

// Declaraciones externas si no están en headers
void referencia(OSQPWorkspace *work, c_float *q, c_float ref);
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u00[2], c_float v00[2]);
void atualizar_restricao(c_float *l, c_float *u, c_float z[3], c_float v[2]);
void atualizar_restricao_v(OSQPWorkspace *work, c_float *l, c_float *u, c_float z1, c_float Einv[2][2], c_float Ax[2]);
void atualizar_A(OSQPWorkspace *work, c_float Einv[2][2]);
void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float m[2][2]);
void multiplyMatrixVector(c_float m[2][2], c_float v[2], c_float res[2]);

// --- CAMBIO CRÍTICO: Eliminar dobles punteros en la interfaz ---
// Cambiamos double por float en toda la función
void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2])
{
    // Directivas HLS (opcional, recomendado para exportar IP)
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE s_axilite port=x_ini
    #pragma HLS INTERFACE s_axilite port=Vsd
    #pragma HLS INTERFACE s_axilite port=Vsq
    #pragma HLS INTERFACE s_axilite port=iL
    #pragma HLS INTERFACE s_axilite port=u00
    #pragma HLS INTERFACE s_axilite port=outputVector

    // Variables locales estáticas (float)
    float vdc, Ex11, Ex12, Ex21, Ex22, Ax01, Ax02, Rl;
    float Ax[2];
    float Ex[2][2];
    float Einv[2][2];
    float z_ini[3];
    float v00[2];
    float v[2];
    float u[2];

    // Buffers temporales para vectores de OSQP
    c_float q_new[15];
    c_float l_new[19];
    c_float u_new[19];
    float ref = REFERENCIA;

    vdc = x_ini[2];

    // CORRECCIÓN: fabsf para float
    if (fabsf(iL) < 0.1f) {
        Rl = 10000.0f;
    } else {
        Rl = vdc/iL;
    }

    // Copia segura de datos del workspace global a local
    // (Asumiendo que 'workspace' está definida en workspace.h/.c y tiene tamaño fijo)
    // Si workspace.data->q falla, puede que necesites acceder a workspace.data.q si la convertiste a estática.
    // En código generado original es un puntero, pero apunta a un array estático.
    // CAMBIA ESTAS LÍNEAS:
    // prea_vec_copy(workspace.data->q, q_new, 15);
    // prea_vec_copy(workspace.data->l, l_new, 19);
    // prea_vec_copy(workspace.data->u, u_new, 19);

    // POR ESTAS (Usando acceso directo a los arrays estáticos):
    prea_vec_copy(qdata, q_new, 15);
    prea_vec_copy(ldata, l_new, 19);
    prea_vec_copy(udata, u_new, 19);

    Ax01 = - w*x_ini[0] - (R*x_ini[1])/L;

    // Cálculo optimizado para float
    float term = (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap * x_ini[2]);
    float term2 = 1.0f / (2.0f * Cap * x_ini[2]);
    
    Ax02 = (1.0f / (Cap * Rl) + term / x_ini[2]) * (x_ini[2] / (Cap * Rl) - term) -
            (3.0f * (w * x_ini[0] + (R * x_ini[1]) / L) * (Vsq - 2.0f * R * x_ini[1])) * term2 +
            (3.0f * (Vsd - 2.0f * R * x_ini[0]) * (w * x_ini[1] + Vsd / L - (R * x_ini[0]) / L)) * term2;

    Ex11 = 0.0f;
    Ex12 = -1.0f/L;

    float termL = 1.0f / (2.0f * Cap * L * x_ini[2]);
    Ex21 = -(3.0f * (Vsd - 2.0f * R * x_ini[0])) * termL;
    Ex22 = -(3.0f * (Vsq - 2.0f * R * x_ini[1])) * termL;

    Ax[0] = Ax01;
    Ax[1] = Ax02;
    Ex[0][0] = Ex11;
    Ex[0][1] = Ex12;
    Ex[1][0] = Ex21;
    Ex[1][1] = Ex22;

    inverse_matrix_2x2(Ex11, Ex12, Ex21, Ex22, Einv);

    z_ini[0] = x_ini[1];
    z_ini[1] = x_ini[2];        
    z_ini[2] = term - x_ini[2] / (Cap * Rl);

    // Funciones auxiliares
    referencia(&workspace, q_new, ref);
    calculateV(Ax, Ex, u00, v00);
    atualizar_restricao(l_new, u_new, z_ini, v00);
    atualizar_restricao_v(&workspace, l_new, u_new, z_ini[1], Einv, Ax);

    // CUIDADO CON ESTA: Si modifica la estructura de la matriz (índices), fallará en HLS.
    // Si solo modifica valores, está bien.
    atualizar_A(&workspace, Einv);

    // Solve Problem
    osqp_solve(&workspace);

    v[0] = workspace.x[11]; // Verifica índices
    v[1] = workspace.x[12];

    v[0] = v[0] - Ax[0];
    v[1] = v[1] - Ax[1];

    multiplyMatrixVector(Einv, v, u);

    u[0] = u[0]/vdc;
    u[1] = u[1]/vdc;

    outputVector[0] = workspace.x[11];
    outputVector[1] = workspace.x[12];  
}
