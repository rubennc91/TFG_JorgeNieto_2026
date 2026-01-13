#include "scaling.h"
#include "workspace.h" // Necesario

#if EMBEDDED != 1
// (limit_scaling y compute_inf_norm... omitidos, igual que antes)
#endif

c_int scale_data(OSQPWorkspace *work) {
  // Usa global scaling y data
  vec_set_scalar(scaling.D, 1., data.n);
  vec_set_scalar(scaling.Dinv, 1., data.n);
  vec_set_scalar(scaling.E, 1., data.m);
  vec_set_scalar(scaling.Einv, 1., data.m);
  vec_set_scalar(scaling.c, 1., 1);
  scaling.cinv = 1.;

  // (resto de lógica de scaling usando globals si es necesario,
  // pero para embedded 2 con scaling activado ya está pre-calculado
  // y estas funciones suelen ser dummy o estáticas)
  return 0;
}

c_int unscale_data(OSQPWorkspace *work) {
  // Unscale cost (usando Pdata, qdata, scaling)
  mat_mult_scalar(&Pdata, scaling.cinv);
  mat_premult_diag(&Pdata, scaling.Dinv);
  mat_postmult_diag(&Pdata, scaling.Dinv);
  vec_mult_scalar(qdata, scaling.cinv, data.n);
  vec_ew_prod(scaling.Dinv, qdata, qdata, data.n);

  // Unscale constraints (Adata, ldata, udata)
  mat_premult_diag(&Adata, scaling.Einv);
  mat_postmult_diag(&Adata, scaling.Dinv);
  vec_ew_prod(scaling.Einv, ldata, ldata, data.m);
  vec_ew_prod(scaling.Einv, udata, udata, data.m);

  return 0;
}

c_int unscale_solution(OSQPWorkspace *work) {
  vec_ew_prod(scaling.D, work->x, work->x, data.n);
  vec_ew_prod(scaling.E, work->y, work->y, data.m);
  vec_mult_scalar(work->y, scaling.c, data.m);
  return 0;
}
