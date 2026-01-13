#include "osqp.h"
#include "auxil.h"
#include "util.h"
#include "scaling.h"
#include "glob_opts.h"
#include "error.h"
#include "qdldl_interface.h"
#include "workspace.h" // IMPORTANTE: Para acceder a 'data', 'settings', etc.

// Función para inicializar settings (simplificada)
void osqp_set_default_settings(OSQPSettings *settings) {
  settings->rho           = (c_float)RHO;
  settings->sigma         = (c_float)SIGMA;
  settings->scaling       = SCALING;
  settings->max_iter      = MAX_ITER;
  settings->eps_abs       = (c_float)EPS_ABS;
  settings->eps_rel       = (c_float)EPS_REL;
  settings->eps_prim_inf  = (c_float)EPS_PRIM_INF;
  settings->eps_dual_inf  = (c_float)EPS_DUAL_INF;
  settings->alpha         = (c_float)ALPHA;
  settings->linsys_solver = LINSYS_SOLVER;
  settings->scaled_termination = SCALED_TERMINATION;
  settings->check_termination  = CHECK_TERMINATION;
  settings->warm_start         = WARM_START;
}

c_int osqp_solve(OSQPWorkspace *work) {
  c_int exitflag = 0;
  c_int iter;
  c_int compute_cost_function = 0;
  c_int can_check_termination = 0;

  // Verificación básica
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Inicialización (Cold Start)
  if (!settings.warm_start) cold_start(work); // Usar global 'settings'

  // Bucle principal ADMM
  for (iter = 1; iter <= settings.max_iter; iter++) {

      // Swap manual de punteros (x <-> x_prev, z <-> z_prev)
      c_float *temp_ptr;
      temp_ptr = work->x;
      work->x = work->x_prev;
      work->x_prev = temp_ptr;

      temp_ptr = work->z;
      work->z = work->z_prev;
      work->z_prev = temp_ptr;

      // Pasos ADMM
      update_xz_tilde(work);
      update_x(work);
      update_z(work);
      update_y(work);

      // Chequeo de terminación
      can_check_termination = settings.check_termination &&
                              (iter % settings.check_termination == 0);

      if (can_check_termination) {
          update_info(work, iter, compute_cost_function, 0);
          if (check_termination(work, 0)) {
              break;
          }
      }
  }

  // Finalización
  if (!can_check_termination) {
      update_info(work, iter - 1, compute_cost_function, 0);
      check_termination(work, 0);
  }

  if (!compute_cost_function && has_solution(work->info)){
      work->info->obj_val = compute_obj_val(work, work->x);
  }

  if (work->info->status_val == OSQP_UNSOLVED) {
      if (!check_termination(work, 1)) {
          update_status(work->info, OSQP_MAX_ITER_REACHED);
      }
  }

  store_solution(work);
  return exitflag;
}

// --- Funciones de Actualización PARCHEADAS ---

c_int osqp_update_lin_cost(OSQPWorkspace *work, const c_float *q_new) {
  // Usamos 'qdata' global directamente para evitar indirección
  prea_vec_copy(q_new, qdata, data.n);

  if (settings.scaling) {
      vec_ew_prod(scaling.D, qdata, qdata, data.n);
      vec_mult_scalar(qdata, scaling.c, data.n);
  }
  reset_info(work->info);
  return 0;
}

c_int osqp_update_bounds(OSQPWorkspace *work, const c_float *l_new, const c_float *u_new) {
  c_int i;
  for (i = 0; i < data.m; i++) {
      if (l_new[i] > u_new[i]) return 1;
  }

  // Usamos 'ldata' y 'udata' globales
  prea_vec_copy(l_new, ldata, data.m);
  prea_vec_copy(u_new, udata, data.m);

  if (settings.scaling) {
      vec_ew_prod(scaling.E, ldata, ldata, data.m);
      vec_ew_prod(scaling.E, udata, udata, data.m);
  }
  reset_info(work->info);

  // Llamada directa parcheada
  return osqp_update_rho(work, settings.rho);
}

c_int osqp_update_P(OSQPWorkspace *work, const c_float *Px_new, const c_int *Px_new_idx, c_int P_new_n) {
  c_int i;
  // Acceso directo a Pdata
  c_int nnzP = Pdata.p[Pdata.n];

  if (settings.scaling) unscale_data(work);

  if (Px_new_idx) {
      for (i = 0; i < P_new_n; i++) Pdata.x[Px_new_idx[i]] = Px_new[i];
  } else {
      for (i = 0; i < nnzP; i++) Pdata.x[i] = Px_new[i];
  }

  if (settings.scaling) scale_data(work);

  // LLAMADA DIRECTA QDLDL
  update_linsys_solver_matrices_qdldl((qdldl_solver*)work->linsys_solver, &Pdata, &Adata);

  reset_info(work->info);
  return 0;
}

c_int osqp_update_A(OSQPWorkspace *work, const c_float *Ax_new, const c_int *Ax_new_idx, c_int A_new_n) {
  c_int i;
  // Acceso directo a Adata
  c_int nnzA = Adata.p[Adata.n];

  if (settings.scaling) unscale_data(work);

  if (Ax_new_idx) {
      for (i = 0; i < A_new_n; i++) Adata.x[Ax_new_idx[i]] = Ax_new[i];
  } else {
      for (i = 0; i < nnzA; i++) Adata.x[i] = Ax_new[i];
  }

  if (settings.scaling) scale_data(work);

  // LLAMADA DIRECTA QDLDL
  update_linsys_solver_matrices_qdldl((qdldl_solver*)work->linsys_solver, &Pdata, &Adata);

  reset_info(work->info);
  return 0;
}

c_int osqp_update_rho(OSQPWorkspace *work, c_float rho_new) {
  c_int i;
  if (rho_new <= 0) return 1;

  settings.rho = c_min(c_max(rho_new, RHO_MIN), RHO_MAX);

  for (i = 0; i < data.m; i++) {
      if (work->constr_type[i] == 0) {
          work->rho_vec[i] = settings.rho;
          work->rho_inv_vec[i] = 1. / settings.rho;
      }
      else if (work->constr_type[i] == 1) {
          work->rho_vec[i] = RHO_EQ_OVER_RHO_INEQ * settings.rho;
          work->rho_inv_vec[i] = 1. / work->rho_vec[i];
      }
  }

  // LLAMADA DIRECTA QDLDL
  return update_linsys_solver_rho_vec_qdldl((qdldl_solver*)work->linsys_solver, work->rho_vec);
}
