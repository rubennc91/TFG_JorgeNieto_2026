#include "osqp.h"
#include "auxil.h"
#include "proj.h"
#include "lin_alg.h"
#include "constants.h"
#include "scaling.h"
#include "util.h"
#include "qdldl_interface.h"
#include "workspace.h" // Necesario para 'data', 'qdata', etc.

void c_strcpy(char dest[], const char source[]) {
    int i = 0;
    while (source[i] != '\0') {
        dest[i] = source[i];
        i++;
    }
    dest[i] = '\0';
}

void swap_vectors(c_float **a, c_float **b) {
  c_float *temp;
  temp = *b;
  *b   = *a;
  *a   = temp;
}

void cold_start(OSQPWorkspace *work) {
  vec_set_scalar(work->x, 0., data.n); // Usa data.n
  vec_set_scalar(work->z, 0., data.m); // Usa data.m
  vec_set_scalar(work->y, 0., data.m); // Usa data.m
}

static void compute_rhs(OSQPWorkspace *work) {
  c_int i;
  for (i = 0; i < data.n; i++) {
    // Usa 'qdata' global
    work->xz_tilde[i] = settings.sigma * work->x_prev[i] - qdata[i];
  }
  for (i = 0; i < data.m; i++) {
    work->xz_tilde[i + data.n] = work->z_prev[i] - work->rho_inv_vec[i] * work->y[i];
  }
}

void update_xz_tilde(OSQPWorkspace *work) {
  compute_rhs(work);
  // LLAMADA DIRECTA SOLVER
  solve_linsys_qdldl((qdldl_solver*)work->linsys_solver, work->xz_tilde);
}

void update_x(OSQPWorkspace *work) {
  c_int i;
  for (i = 0; i < data.n; i++) {
    work->x[i] = settings.alpha * work->xz_tilde[i] +
                 ((c_float)1.0 - settings.alpha) * work->x_prev[i];
  }
  for (i = 0; i < data.n; i++) {
    work->delta_x[i] = work->x[i] - work->x_prev[i];
  }
}

void update_z(OSQPWorkspace *work) {
  c_int i;
  for (i = 0; i < data.m; i++) {
    work->z[i] = settings.alpha * work->xz_tilde[i + data.n] +
                 ((c_float)1.0 - settings.alpha) * work->z_prev[i] +
                 work->rho_inv_vec[i] * work->y[i];
  }
  project(work, work->z);
}

void update_y(OSQPWorkspace *work) {
  c_int i;
  for (i = 0; i < data.m; i++) {
    work->delta_y[i] = work->rho_vec[i] *
                       (settings.alpha *
                        work->xz_tilde[i + data.n] +
                        ((c_float)1.0 - settings.alpha) * work->z_prev[i] -
                        work->z[i]);
    work->y[i] += work->delta_y[i];
  }
}

c_float compute_obj_val(OSQPWorkspace *work, c_float *x) {
  c_float obj_val;
  // Usa Pdata y qdata globales
  obj_val = quad_form(&Pdata, x) + vec_prod(qdata, x, data.n);

  if (settings.scaling) {
    obj_val *= scaling.cinv;
  }
  return obj_val;
}

c_float compute_pri_res(OSQPWorkspace *work, c_float *x, c_float *z) {
  mat_vec(&Adata, x, work->Ax, 0); // Usa Adata
  vec_add_scaled(work->z_prev, work->Ax, z, data.m, -1);

  if (settings.scaling && !settings.scaled_termination) {
    return vec_scaled_norm_inf(scaling.Einv, work->z_prev, data.m);
  }
  return vec_norm_inf(work->z_prev, data.m);
}

c_float compute_pri_tol(OSQPWorkspace *work, c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;

  if (settings.scaling && !settings.scaled_termination) {
    max_rel_eps = vec_scaled_norm_inf(scaling.Einv, work->z, data.m);
    temp_rel_eps = vec_scaled_norm_inf(scaling.Einv, work->Ax, data.m);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  } else {
    max_rel_eps = vec_norm_inf(work->z, data.m);
    temp_rel_eps = vec_norm_inf(work->Ax, data.m);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }
  return eps_abs + eps_rel * max_rel_eps;
}

c_float compute_dua_res(OSQPWorkspace *work, c_float *x, c_float *y) {
  prea_vec_copy(qdata, work->x_prev, data.n); // Usa qdata

  mat_vec(&Pdata, x, work->Px, 0); // Usa Pdata
  mat_tpose_vec(&Pdata, x, work->Px, 1, 1);
  vec_add_scaled(work->x_prev, work->x_prev, work->Px, data.n, 1);

  if (data.m > 0) {
    mat_tpose_vec(&Adata, y, work->Aty, 0, 0); // Usa Adata
    vec_add_scaled(work->x_prev, work->x_prev, work->Aty, data.n, 1);
  }

  if (settings.scaling && !settings.scaled_termination) {
    return scaling.cinv * vec_scaled_norm_inf(scaling.Dinv, work->x_prev, data.n);
  }
  return vec_norm_inf(work->x_prev, data.n);
}

c_float compute_dua_tol(OSQPWorkspace *work, c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;

  if (settings.scaling && !settings.scaled_termination) {
    max_rel_eps = vec_scaled_norm_inf(scaling.Dinv, qdata, data.n);
    temp_rel_eps = vec_scaled_norm_inf(scaling.Dinv, work->Aty, data.n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    temp_rel_eps = vec_scaled_norm_inf(scaling.Dinv, work->Px, data.n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    max_rel_eps *= scaling.cinv;
  } else {
    max_rel_eps = vec_norm_inf(qdata, data.n);
    temp_rel_eps = vec_norm_inf(work->Aty, data.n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
    temp_rel_eps = vec_norm_inf(work->Px, data.n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }
  return eps_abs + eps_rel * max_rel_eps;
}

c_int is_primal_infeasible(OSQPWorkspace *work, c_float eps_prim_inf) {
  c_int i;
  c_float norm_delta_y;
  c_float ineq_lhs = 0.0;

  for (i = 0; i < data.m; i++) {
    if (udata[i] > OSQP_INFTY * MIN_SCALING) { // Usa udata
      if (ldata[i] < -OSQP_INFTY * MIN_SCALING) work->delta_y[i] = 0.0;
      else work->delta_y[i] = c_min(work->delta_y[i], 0.0);
    } else if (ldata[i] < -OSQP_INFTY * MIN_SCALING) {
      work->delta_y[i] = c_max(work->delta_y[i], 0.0);
    }
  }

  if (settings.scaling && !settings.scaled_termination) {
    vec_ew_prod(scaling.E, work->delta_y, work->Adelta_x, data.m);
    norm_delta_y = vec_norm_inf(work->Adelta_x, data.m);
  } else {
    norm_delta_y = vec_norm_inf(work->delta_y, data.m);
  }

  if (norm_delta_y > OSQP_DIVISION_TOL) {
    for (i = 0; i < data.m; i++) {
      ineq_lhs += udata[i] * c_max(work->delta_y[i], 0) + \
                  ldata[i] * c_min(work->delta_y[i], 0);
    }
    if (ineq_lhs < eps_prim_inf * norm_delta_y) {
      mat_tpose_vec(&Adata, work->delta_y, work->Atdelta_y, 0, 0); // Adata
      if (settings.scaling && !settings.scaled_termination) {
        vec_ew_prod(scaling.Dinv, work->Atdelta_y, work->Atdelta_y, data.n);
      }
      return vec_norm_inf(work->Atdelta_y, data.n) < eps_prim_inf * norm_delta_y;
    }
  }
  return 0;
}

c_int is_dual_infeasible(OSQPWorkspace *work, c_float eps_dual_inf) {
  c_int i;
  c_float norm_delta_x;
  c_float cost_scaling;

  if (settings.scaling && !settings.scaled_termination) {
    norm_delta_x = vec_scaled_norm_inf(scaling.D, work->delta_x, data.n);
    cost_scaling = scaling.c;
  } else {
    norm_delta_x = vec_norm_inf(work->delta_x, data.n);
    cost_scaling = 1.0;
  }

  if (norm_delta_x > OSQP_DIVISION_TOL) {
    if (vec_prod(qdata, work->delta_x, data.n) <
        cost_scaling * eps_dual_inf * norm_delta_x) {

      mat_vec(&Pdata, work->delta_x, work->Pdelta_x, 0); // Pdata
      mat_tpose_vec(&Pdata, work->delta_x, work->Pdelta_x, 1, 1);

      if (settings.scaling && !settings.scaled_termination) {
        vec_ew_prod(scaling.Dinv, work->Pdelta_x, work->Pdelta_x, data.n);
      }

      if (vec_norm_inf(work->Pdelta_x, data.n) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
        mat_vec(&Adata, work->delta_x, work->Adelta_x, 0); // Adata

        if (settings.scaling && !settings.scaled_termination) {
          vec_ew_prod(scaling.Einv, work->Adelta_x, work->Adelta_x, data.m);
        }

        for (i = 0; i < data.m; i++) {
          if (((udata[i] < OSQP_INFTY * MIN_SCALING) &&
               (work->Adelta_x[i] >  eps_dual_inf * norm_delta_x)) ||
              ((ldata[i] > -OSQP_INFTY * MIN_SCALING) &&
               (work->Adelta_x[i] < -eps_dual_inf * norm_delta_x))) {
            return 0;
          }
        }
        return 1;
      }
    }
  }
  return 0;
}

c_int has_solution(OSQPInfo * info){
  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));
}

void store_solution(OSQPWorkspace *work) {
  if (has_solution(work->info)) {
    prea_vec_copy(work->x, work->solution->x, data.n);
    prea_vec_copy(work->y, work->solution->y, data.m);
    if (settings.scaling) unscale_solution(work);
  } else {
    vec_set_scalar(work->solution->x, OSQP_NAN, data.n);
    vec_set_scalar(work->solution->y, OSQP_NAN, data.m);
    cold_start(work);
  }
}

void update_info(OSQPWorkspace *work, c_int iter, c_int compute_objective, c_int polish) {
  c_float *x, *z, *y;
  c_float *obj_val, *pri_res, *dua_res;

  x = work->x;
  y = work->y;
  z = work->z;
  obj_val = &work->info->obj_val;
  pri_res = &work->info->pri_res;
  dua_res = &work->info->dua_res;
  work->info->iter = iter;

  if (compute_objective) *obj_val = compute_obj_val(work, x);

  if (data.m == 0) *pri_res = 0.;
  else *pri_res = compute_pri_res(work, x, z);

  *dua_res = compute_dua_res(work, x, y);
}

void reset_info(OSQPInfo *info) {
  update_status(info, OSQP_UNSOLVED);
}

void update_status(OSQPInfo *info, c_int status_val) {
  info->status_val = status_val;
  if (status_val == OSQP_SOLVED) c_strcpy(info->status, "solved");
  else if (status_val == OSQP_SOLVED_INACCURATE) c_strcpy(info->status, "solved inaccurate");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE) c_strcpy(info->status, "primal infeasible");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) c_strcpy(info->status, "primal infeasible inaccurate");
  else if (status_val == OSQP_UNSOLVED) c_strcpy(info->status, "unsolved");
  else if (status_val == OSQP_DUAL_INFEASIBLE) c_strcpy(info->status, "dual infeasible");
  else if (status_val == OSQP_DUAL_INFEASIBLE_INACCURATE) c_strcpy(info->status, "dual infeasible inaccurate");
  else if (status_val == OSQP_MAX_ITER_REACHED) c_strcpy(info->status, "maximum iterations reached");
  else if (status_val == OSQP_SIGINT) c_strcpy(info->status, "interrupted");
  else if (status_val == OSQP_NON_CVX) c_strcpy(info->status, "problem non convex");
}

c_int check_termination(OSQPWorkspace *work, c_int approximate) {
  c_float eps_prim, eps_dual, eps_prim_inf, eps_dual_inf;
  c_int exitflag = 0;
  c_int prim_res_check = 0, dual_res_check = 0, prim_inf_check = 0, dual_inf_check = 0;
  c_float eps_abs, eps_rel;

  // Usa settings globales
  eps_abs = settings.eps_abs;
  eps_rel = settings.eps_rel;
  eps_prim_inf = settings.eps_prim_inf;
  eps_dual_inf = settings.eps_dual_inf;

  if ((work->info->pri_res > OSQP_INFTY) || (work->info->dua_res > OSQP_INFTY)){
    update_status(work->info, OSQP_NON_CVX);
    work->info->obj_val = OSQP_NAN;
    return 1;
  }

  if (approximate) {
    eps_abs *= 10;
    eps_rel *= 10;
    eps_prim_inf *= 10;
    eps_dual_inf *= 10;
  }

  if (data.m == 0) {
    prim_res_check = 1;
  } else {
    eps_prim = compute_pri_tol(work, eps_abs, eps_rel);
    if (work->info->pri_res < eps_prim) prim_res_check = 1;
    else prim_inf_check = is_primal_infeasible(work, eps_prim_inf);
  }

  eps_dual = compute_dua_tol(work, eps_abs, eps_rel);
  if (work->info->dua_res < eps_dual) dual_res_check = 1;
  else dual_inf_check = is_dual_infeasible(work, eps_dual_inf);

  if (prim_res_check && dual_res_check) {
    if (approximate) update_status(work->info, OSQP_SOLVED_INACCURATE);
    else update_status(work->info, OSQP_SOLVED);
    exitflag = 1;
  } else if (prim_inf_check) {
    if (approximate) update_status(work->info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    else update_status(work->info, OSQP_PRIMAL_INFEASIBLE);
    work->info->obj_val = OSQP_INFTY;
    exitflag = 1;
  } else if (dual_inf_check) {
    if (approximate) update_status(work->info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    else update_status(work->info, OSQP_DUAL_INFEASIBLE);
    work->info->obj_val = -OSQP_INFTY;
    exitflag = 1;
  }

  return exitflag;
}
