#include "scaling.h"
#include "workspace.h"

c_int scale_data(void) {
  scaling.c = 1.0;
  vec_set_scalar(scaling_D, 1., data.n);
  vec_set_scalar(scaling_Dinv, 1., data.n);
  vec_set_scalar(scaling_E, 1., data.m);
  vec_set_scalar(scaling_Einv, 1., data.m);
  scaling.cinv = 1.0;
  return 0;
}

c_int unscale_data(void) {
  // Usamos los arrays globales Pdata_x, Pdata_p, etc.
  mat_mult_scalar(Pdata_x, Pdata_p, data.n, scaling.cinv);
  mat_premult_diag(Pdata_x, Pdata_p, Pdata_i, data.n, scaling_Dinv);
  mat_postmult_diag(Pdata_x, Pdata_p, data.n, scaling_Dinv);

  vec_mult_scalar(qdata, scaling.cinv, data.n);
  vec_ew_prod(scaling_Dinv, qdata, qdata, data.n);

  mat_premult_diag(Adata_x, Adata_p, Adata_i, data.n, scaling_Einv);
  mat_postmult_diag(Adata_x, Adata_p, data.n, scaling_Dinv);

  vec_ew_prod(scaling_Einv, ldata, ldata, data.m);
  vec_ew_prod(scaling_Einv, udata, udata, data.m);
  return 0;
}

c_int unscale_solution(void) {
  vec_ew_prod(scaling_D, xsolution, xsolution, data.n);
  vec_ew_prod(scaling_E, ysolution, ysolution, data.m);
  vec_mult_scalar(ysolution, scaling.cinv, data.m);
  return 0;
}
