#include "kkt.h"

void update_KKT_P(c_float      *KKT_x,
                  const c_float *P_x,
                  const c_int   *P_p,
                  c_int          P_n,
                  const c_int   *PtoKKT,
                  const c_float  param1,
                  const c_int   *Pdiag_idx,
                  const c_int    Pdiag_n) {
  c_int i, j;
  // Usamos P_p[P_n] para obtener el número de elementos no nulos
  c_int nnzP = P_p[P_n];

  for (i = 0; i < nnzP; i++) {
    KKT_x[PtoKKT[i]] = P_x[i];
  }

  for (i = 0; i < Pdiag_n; i++) {
    j = Pdiag_idx[i];
    KKT_x[PtoKKT[j]] += param1;
  }
}

void update_KKT_A(c_float      *KKT_x,
                  const c_float *A_x,
                  const c_int   *A_p,
                  c_int          A_n,
                  const c_int   *AtoKKT) {
  c_int i;
  c_int nnzA = A_p[A_n];

  for (i = 0; i < nnzA; i++) {
    KKT_x[AtoKKT[i]] = A_x[i];
  }
}

void update_KKT_param2(c_float       *KKT_x,
                       const c_float *param2,
                       const c_int   *param2toKKT,
                       const c_int    m) {
  c_int i;
  for (i = 0; i < m; i++) {
    KKT_x[param2toKKT[i]] = -param2[i];
  }
}
