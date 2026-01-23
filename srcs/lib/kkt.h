#ifndef KKT_H
# define KKT_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"

// --- CAMBIO: Comentamos esto porque cs.h no existe y no lo queremos ---
// #  include "cs.h"

// Comentamos la declaración de form_KKT porque usa memoria dinámica
/*
csc* form_KKT(const csc  *P,
              const  csc *A,
              c_int       format,
              c_float     param1,
              c_float    *param2,
              c_int      *PtoKKT,
              c_int      *AtoKKT,
              c_int     **Pdiag_idx,
              c_int      *Pdiag_n,
              c_int      *param2toKKT);
*/
// ---------------------------------------------------------------------

// --- ESTAS SON LAS QUE SÍ NECESITAMOS (Mantener descomentadas) ---

void update_KKT_P(csc          *KKT,
                  const csc    *P,
                  const c_int  *PtoKKT,
                  const c_float param1,
                  const c_int  *Pdiag_idx,
                  const c_int   Pdiag_n);

void update_KKT_A(csc         *KKT,
                  const csc   *A,
                  const c_int *AtoKKT);

void update_KKT_param2(csc           *KKT,
                       const c_float *param2,
                       const c_int   *param2toKKT,
                       const c_int    m);

# ifdef __cplusplus
}
# endif

#endif
