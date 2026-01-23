#include "kkt.h"

// --- CAMBIO: Comentar/Borrar form_KKT entera ---
/*
csc* form_KKT(...) {
   ... (todo el código que había aquí con mallocs) ...
   return KKT;
}
*/
// -----------------------------------------------

// --- MANTENER SOLO ESTAS FUNCIONES ---

void update_KKT_P(csc          *KKT,
                  const csc    *P,
                  const c_int  *PtoKKT,
                  const c_float param1,
                  const c_int  *Pdiag_idx,
                  const c_int   Pdiag_n) {
  c_int i, j;

  for (i = 0; i < P->p[P->n]; i++) {
    KKT->x[PtoKKT[i]] = P->x[i];
  }

  for (i = 0; i < Pdiag_n; i++) {
    j                  = Pdiag_idx[i];
    KKT->x[PtoKKT[j]] += param1;
  }
}

void update_KKT_A(csc *KKT, const csc *A, const c_int *AtoKKT) {
  c_int i;

  for (i = 0; i < A->p[A->n]; i++) {
    KKT->x[AtoKKT[i]] = A->x[i];
  }
}

void update_KKT_param2(csc *KKT, const c_float *param2,
                       const c_int *param2toKKT, const c_int m) {
  c_int i;

  for (i = 0; i < m; i++) {
    KKT->x[param2toKKT[i]] = -param2[i];
  }
}
