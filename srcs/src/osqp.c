#include "osqp.h"
#include "auxil.h"
#include "util.h"
#include "scaling.h"
#include "glob_opts.h"
#include "error.h"
#include "qdldl_interface.h"
#include "workspace.h"
#include "lin_alg.h"

void osqp_set_default_settings(OSQPSettings *sets) {
  sets->rho = (c_float)RHO;
  sets->sigma = (c_float)SIGMA;
  sets->scaling = SCALING;
  sets->max_iter = MAX_ITER;
  sets->eps_abs = (c_float)EPS_ABS;
  sets->eps_rel = (c_float)EPS_REL;
  sets->eps_prim_inf = (c_float)EPS_PRIM_INF;
  sets->eps_dual_inf = (c_float)EPS_DUAL_INF;
  sets->alpha = (c_float)ALPHA;
  sets->linsys_solver = LINSYS_SOLVER;
  sets->scaled_termination = SCALED_TERMINATION;
  sets->check_termination = CHECK_TERMINATION;
  sets->warm_start = WARM_START;
}

c_int osqp_solve(void) {
  c_int exitflag = 0;
  c_int iter;

  if (!settings.warm_start) cold_start();

  for (iter = 1; iter <= settings.max_iter; iter++) {
      prea_vec_copy(work_x, work_x_prev, data.n);
      prea_vec_copy(work_z, work_z_prev, data.m);

      update_xz_tilde();
      update_x();
      update_z();
      update_y();

      if (settings.check_termination && (iter % settings.check_termination == 0)) {
          update_info(iter, 0, 0);
          if (check_termination(0)) break;
      }
  }

  if (info.status_val == OSQP_UNSOLVED) {
      if (!check_termination(1)) update_status(&info, OSQP_MAX_ITER_REACHED);
  }
  store_solution();
  return exitflag;
}

// Funciones de actualización
c_int osqp_update_lin_cost(const c_float *q_new) {
  prea_vec_copy(q_new, qdata, data.n);
  if (settings.scaling) {
    vec_ew_prod(scaling_D, qdata, qdata, data.n);
    vec_mult_scalar(qdata, scaling.c, data.n);
  }
  reset_info(&info);
  return 0;
}

c_int osqp_update_bounds(const c_float *l_new, const c_float *u_new) {
  prea_vec_copy(l_new, ldata, data.m);
  prea_vec_copy(u_new, udata, data.m);
  if (settings.scaling) {
    vec_ew_prod(scaling_E, ldata, ldata, data.m);
    vec_ew_prod(scaling_E, udata, udata, data.m);
  }
  reset_info(&info);
  return osqp_update_rho(settings.rho);
}

c_int osqp_update_P(const c_float *Px_new, const c_int *Px_new_idx, c_int P_new_n) {
  c_int i, nnzP = Pdata.p[Pdata.n];
  if (settings.scaling) unscale_data();
  if (Px_new_idx) {
    for (i = 0; i < P_new_n; i++) Pdata.x[Px_new_idx[i]] = Px_new[i];
  } else {
    for (i = 0; i < nnzP; i++) Pdata.x[i] = Px_new[i];
  }
  if (settings.scaling) scale_data();
  update_linsys_solver_matrices_qdldl(&linsys_solver, &Pdata, &Adata);
  reset_info(&info);
  return 0;
}

c_int osqp_update_A(const c_float *Ax_new, const c_int *Ax_new_idx, c_int A_new_n) {
  c_int i, nnzA = Adata.p[Adata.n];
  if (settings.scaling) unscale_data();
  if (Ax_new_idx) {
    for (i = 0; i < A_new_n; i++) Adata.x[Ax_new_idx[i]] = Ax_new[i];
  } else {
    for (i = 0; i < nnzA; i++) Adata.x[i] = Ax_new[i];
  }
  if (settings.scaling) scale_data();
  update_linsys_solver_matrices_qdldl(&linsys_solver, &Pdata, &Adata);
  reset_info(&info);
  return 0;
}

c_int osqp_update_rho(c_float rho_new) {
  c_int i;
  if (rho_new <= 0) return 1;
  settings.rho = c_min(c_max(rho_new, RHO_MIN), RHO_MAX);
  for (i = 0; i < data.m; i++) {
    if (work_constr_type[i] == 0) {
      work_rho_vec[i] = settings.rho;
      work_rho_inv_vec[i] = 1. / settings.rho;
    } else if (work_constr_type[i] == 1) {
      work_rho_vec[i] = RHO_EQ_OVER_RHO_INEQ * settings.rho;
      work_rho_inv_vec[i] = 1. / work_rho_vec[i];
    }
  }
  return update_linsys_solver_rho_vec_qdldl(&linsys_solver, work_rho_vec);
}
