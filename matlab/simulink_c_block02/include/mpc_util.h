#include "types.h"

typedef struct{
    c_float a;
    c_float b;
    c_float c;
}v3ph;

void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float inv[2][2]);
// Função para multiplicar uma matriz 2x2 por um vetor 2x1
void multiplyMatrixVector(c_float Ex[2][2], c_float u[2], c_float result[2]);
void referencia(OSQPWorkspace *work, c_float* q_new, c_float ref);
// Função para calcular v = Ax + Ex * u
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]);
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00);
void atualizar_restricao_v(OSQPWorkspace *work, c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax);
void atualizar_A(OSQPWorkspace *work, c_float Einv[2][2]);

