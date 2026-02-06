#ifndef KKT_H
# define KKT_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"

// Actualizar KKT usando P
// Recibe los arrays crudos de KKT y P
void update_KKT_P(c_float      *KKT_x,
                  const c_float *P_x,
                  const c_int   *P_p,
                  c_int          P_n,
                  const c_int   *PtoKKT,
                  const c_float  param1,
                  const c_int   *Pdiag_idx,
                  const c_int    Pdiag_n);

// Actualizar KKT usando A
void update_KKT_A(c_float      *KKT_x,
                  const c_float *A_x,
                  const c_int   *A_p,
                  c_int          A_n,
                  const c_int   *AtoKKT);

// Actualizar KKT con rho
void update_KKT_param2(c_float       *KKT_x,
                       const c_float *param2,
                       const c_int   *param2toKKT,
                       const c_int    m);

# ifdef __cplusplus
}
# endif

#endif
