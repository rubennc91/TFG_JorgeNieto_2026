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

// Usamos extern para referenciar el workspace que YA existe en workspace.c
extern OSQPWorkspace workspace;

// --- FUNCIONES AUXILIARES LOCALES (STATIC) ---
// Las definimos aqui para asegurar que HLS las pueda "aplanar" (inline)
// y evitar errores de punteros dobles con matrices externas.

static void local_inverse_matrix_2x2(float a, float b, float c, float d, float m[2][2]) {
    #pragma HLS INLINE
    float det = a*d - b*c;
    float detInv = 1.0f / det;
    m[0][0] =  d * detInv;
    m[0][1] = -b * detInv;
    m[1][0] = -c * detInv;
    m[1][1] =  a * detInv;
}

static void local_multiplyMatrixVector(float m[2][2], float v[2], float res[2]) {
    #pragma HLS INLINE
    res[0] = m[0][0]*v[0] + m[0][1]*v[1];
    res[1] = m[1][0]*v[0] + m[1][1]*v[1];
}

// Declaraciones externas necesarias (estas se mantienen porque son complejas)
void referencia(c_float* q_new, c_float ref);
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]);
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00);
// Nota: mantenemos el prototipo original, pero asegurate que en mpc_util.c
// esta funcion no use punteros dobles extraños.
void atualizar_restricao_v(c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax);
void atualizar_A(c_float Einv[2][2]);


void init_workspace_manually() {
    // Solo inicializamos escalares.
    scaling.c    = 1.0;
    scaling.cinv = 1.0;

    // --- CORRECCIÓN CRÍTICA PARA HLS ---
    // HLS no soporta la asignación dinámica de punteros dentro de structs de esta forma.
    // Comentamos estas lineas. La solución se guardará en 'xsolution' y 'ysolution'
    // automáticamente gracias a la función store_solution() del solver.

    // solution.x = xsolution;  // COMENTADO PARA EVITAR ERROR HLS 214-134
    // solution.y = ysolution;  // COMENTADO PARA EVITAR ERROR HLS 214-134
}

void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2])
{
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE s_axilite port=x_ini
    #pragma HLS INTERFACE s_axilite port=Vsd
    #pragma HLS INTERFACE s_axilite port=Vsq
    #pragma HLS INTERFACE s_axilite port=iL
    #pragma HLS INTERFACE s_axilite port=u00
    #pragma HLS INTERFACE s_axilite port=outputVector

    // Particionar arrays pequeños mejora el rendimiento y evita problemas de punteros
    #pragma HLS ARRAY_PARTITION variable=x_ini complete
    #pragma HLS ARRAY_PARTITION variable=u00 complete
    #pragma HLS ARRAY_PARTITION variable=outputVector complete

    static int is_initialized = 0;
    if (!is_initialized) {
        init_workspace_manually();
        is_initialized = 1;
    }

    float vdc, Ex11, Ex12, Ex21, Ex22, Ax01, Ax02, Rl;

    float Ax[2];
    #pragma HLS ARRAY_PARTITION variable=Ax complete

    float Ex[2][2];
    #pragma HLS ARRAY_PARTITION variable=Ex complete dim=0

    float Einv[2][2];
    #pragma HLS ARRAY_PARTITION variable=Einv complete dim=0

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

    // Usamos la funcion local estatica
    local_inverse_matrix_2x2(Ex11, Ex12, Ex21, Ex22, Einv);

    z_ini[0] = x_ini[1]; z_ini[1] = x_ini[2]; z_ini[2] = term - x_ini[2] / (Cap * Rl);

    referencia(q_new, ref);
    calculateV(Ax, Ex, u00, v00);
    atualizar_restricao(l_new, u_new, z_ini, v00);

    // Si esta funcion externa falla, asegúrate de que en su definicion
    // Einv se trate como array fijo y no como puntero dinamico.
    atualizar_restricao_v(l_new, u_new, z_ini[1], Einv, Ax);

    atualizar_A(Einv);

    osqp_solve();

    v[0] = work_x[11];
    v[1] = work_x[12];

    v[0] = v[0] - Ax[0];
    v[1] = v[1] - Ax[1];

    // Usamos la funcion local estatica
    local_multiplyMatrixVector(Einv, v, u);

    u[0] = u[0]/vdc;
    u[1] = u[1]/vdc;

    outputVector[0] = work_x[11];
    outputVector[1] = work_x[12];
}
