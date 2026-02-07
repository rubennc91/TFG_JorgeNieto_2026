#include "osqp.h"
#include "auxil.h"
#include "proj.h"
#include "lin_alg.h"
#include "constants.h"
#include "scaling.h"
#include "util.h"
#include "qdldl_interface.h"
#include "workspace.h"

//void c_strcpy(char dest[], const char source[]) {
//    int i = 0;
//    while (source[i] != '\0') {
//        dest[i] = source[i];
//        i++;
//    }
//    dest[i] = '\0';
//}

void cold_start(void) {
  vec_set_scalar(work_x, 0., data.n);
  vec_set_scalar(work_z, 0., data.m);
  vec_set_scalar(work_y, 0., data.m);
}

static void compute_rhs(void) {
  c_int i;
  for (i = 0; i < data.n; i++) work_xz_tilde[i] = settings.sigma * work_x_prev[i] - qdata[i];
  for (i = 0; i < data.m; i++) work_xz_tilde[i + data.n] = work_z_prev[i] - work_rho_inv_vec[i] * work_y[i];
}

void update_xz_tilde(void) {
  compute_rhs();
  solve_linsys_qdldl(work_xz_tilde);
}

void update_x(void) {
  c_int i;
  for (i = 0; i < data.n; i++) {
    work_x[i] = settings.alpha * work_xz_tilde[i] + ((c_float)1.0 - settings.alpha) * work_x_prev[i];
    work_delta_x[i] = work_x[i] - work_x_prev[i];
  }
}

void update_z(void) {
  c_int i;
  for (i = 0; i < data.m; i++) {
    work_z[i] = settings.alpha * work_xz_tilde[i + data.n] + ((c_float)1.0 - settings.alpha) * work_z_prev[i] + work_rho_inv_vec[i] * work_y[i];
  }
  project(work_z);
}

void update_y(void) {
  c_int i;
  for (i = 0; i < data.m; i++) {
    work_delta_y[i] = work_rho_vec[i] * (settings.alpha * work_xz_tilde[i + data.n] + ((c_float)1.0 - settings.alpha) * work_z_prev[i] - work_z[i]);
    work_y[i] += work_delta_y[i];
  }
}

c_float compute_obj_val(c_float *x) {
  c_float obj_val = quad_form(Pdata_x, Pdata_p, Pdata_i, data.n, x) + vec_prod(qdata, x, data.n);
  if (settings.scaling) obj_val *= scaling.cinv;
  return obj_val;
}

c_float compute_pri_res(c_float *x, c_float *z) {
  mat_vec(Adata_x, Adata_p, Adata_i, data.n, data.m, x, work_Ax, 0);
  vec_add_scaled(work_z_prev, work_Ax, z, data.m, -1);
  if (settings.scaling && !settings.scaled_termination) return vec_scaled_norm_inf(scaling_Einv, work_z_prev, data.m);
  return vec_norm_inf(work_z_prev, data.m);
}

c_float compute_pri_tol(c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;
  if (settings.scaling && !settings.scaled_termination) {
    max_rel_eps = vec_scaled_norm_inf(scaling_Einv, work_z, data.m);
    temp_rel_eps = vec_scaled_norm_inf(scaling_Einv, work_Ax, data.m);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  } else {
    max_rel_eps = vec_norm_inf(work_z, data.m);
    temp_rel_eps = vec_norm_inf(work_Ax, data.m);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }
  return eps_abs + eps_rel * max_rel_eps;
}

c_float compute_dua_res(c_float *x, c_float *y) {
  prea_vec_copy(qdata, work_x_prev, data.n);

  mat_vec(Pdata_x, Pdata_p, Pdata_i, data.n, data.m, x, work_Px, 0);
  mat_tpose_vec(Pdata_x, Pdata_p, Pdata_i, data.n, data.m, x, work_Px, 1, 1);

  vec_add_scaled(work_x_prev, work_x_prev, work_Px, data.n, 1);

  if (data.m > 0) {
    mat_tpose_vec(Adata_x, Adata_p, Adata_i, data.n, data.m, y, work_Aty, 0, 0);
    vec_add_scaled(work_x_prev, work_x_prev, work_Aty, data.n, 1);
  }

  if (settings.scaling && !settings.scaled_termination) return scaling.cinv * vec_scaled_norm_inf(scaling_Dinv, work_x_prev, data.n);
  return vec_norm_inf(work_x_prev, data.n);
}

c_float compute_dua_tol(c_float eps_abs, c_float eps_rel) {
  c_float max_rel_eps, temp_rel_eps;
  if (settings.scaling && !settings.scaled_termination) {
    max_rel_eps = vec_scaled_norm_inf(scaling_Dinv, qdata, data.n);
    temp_rel_eps = vec_scaled_norm_inf(scaling_Dinv, work_Aty, data.n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    temp_rel_eps = vec_scaled_norm_inf(scaling_Dinv, work_Px, data.n);
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    max_rel_eps *= scaling.cinv;
  } else {
    max_rel_eps = vec_norm_inf(qdata, data.n);
    temp_rel_eps = vec_norm_inf(work_Aty, data.n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
    temp_rel_eps = vec_norm_inf(work_Px, data.n);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }
  return eps_abs + eps_rel * max_rel_eps;
}

c_int is_primal_infeasible(c_float eps_prim_inf) {
  c_int i; c_float norm_delta_y, ineq_lhs = 0.0;
  for (i = 0; i < data.m; i++) {
    if (udata[i] > OSQP_INFTY * MIN_SCALING) {
      if (ldata[i] < -OSQP_INFTY * MIN_SCALING) work_delta_y[i] = 0.0;
      else work_delta_y[i] = c_min(work_delta_y[i], 0.0);
    } else if (ldata[i] < -OSQP_INFTY * MIN_SCALING) work_delta_y[i] = c_max(work_delta_y[i], 0.0);
  }
  if (settings.scaling && !settings.scaled_termination) {
    vec_ew_prod(scaling_E, work_delta_y, work_Adelta_x, data.m);
    norm_delta_y = vec_norm_inf(work_Adelta_x, data.m);
  } else {
    norm_delta_y = vec_norm_inf(work_delta_y, data.m);
  }
  if (norm_delta_y > OSQP_DIVISION_TOL) {
    for (i = 0; i < data.m; i++) ineq_lhs += udata[i] * c_max(work_delta_y[i], 0) + ldata[i] * c_min(work_delta_y[i], 0);
    if (ineq_lhs < eps_prim_inf * norm_delta_y) {
      mat_tpose_vec(Adata_x, Adata_p, Adata_i, data.n, data.m, work_delta_y, work_Atdelta_y, 0, 0);
      if (settings.scaling && !settings.scaled_termination) vec_ew_prod(scaling_Dinv, work_Atdelta_y, work_Atdelta_y, data.n);
      return vec_norm_inf(work_Atdelta_y, data.n) < eps_prim_inf * norm_delta_y;
    }
  }
  return 0;
}

c_int is_dual_infeasible(c_float eps_dual_inf) {
  c_int i; c_float norm_delta_x, cost_scaling;
  if (settings.scaling && !settings.scaled_termination) {
    norm_delta_x = vec_scaled_norm_inf(scaling_D, work_delta_x, data.n);
    cost_scaling = scaling.c;
  } else {
    norm_delta_x = vec_norm_inf(work_delta_x, data.n);
    cost_scaling = 1.0;
  }
  if (norm_delta_x > OSQP_DIVISION_TOL) {
    if (vec_prod(qdata, work_delta_x, data.n) < cost_scaling * eps_dual_inf * norm_delta_x) {
      mat_vec(Pdata_x, Pdata_p, Pdata_i, data.n, data.m, work_delta_x, work_Pdelta_x, 0);
      mat_tpose_vec(Pdata_x, Pdata_p, Pdata_i, data.n, data.m, work_delta_x, work_Pdelta_x, 1, 1);

      if (settings.scaling && !settings.scaled_termination) vec_ew_prod(scaling_Dinv, work_Pdelta_x, work_Pdelta_x, data.n);
      if (vec_norm_inf(work_Pdelta_x, data.n) < cost_scaling * eps_dual_inf * norm_delta_x) {

        mat_vec(Adata_x, Adata_p, Adata_i, data.n, data.m, work_delta_x, work_Adelta_x, 0);

        if (settings.scaling && !settings.scaled_termination) vec_ew_prod(scaling_Einv, work_Adelta_x, work_Adelta_x, data.m);
        for (i = 0; i < data.m; i++) {
          if (((udata[i] < OSQP_INFTY * MIN_SCALING) && (work_Adelta_x[i] > eps_dual_inf * norm_delta_x)) ||
              ((ldata[i] > -OSQP_INFTY * MIN_SCALING) && (work_Adelta_x[i] < -eps_dual_inf * norm_delta_x))) return 0;
        }
        return 1;
      }
    }
  }
  return 0;
}

// ... (El resto: store_solution, update_info, etc. sin cambios) ...
void store_solution(void) {
  if ((info.status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info.status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info.status_val != OSQP_DUAL_INFEASIBLE) &&
      (info.status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info.status_val != OSQP_NON_CVX)) {
    prea_vec_copy(work_x, xsolution, data.n);
    prea_vec_copy(work_y, ysolution, data.m);
    if (settings.scaling) unscale_solution();
  } else {
    vec_set_scalar(xsolution, OSQP_NAN, data.n);
    vec_set_scalar(ysolution, OSQP_NAN, data.m);
    cold_start();
  }
}

void update_info(c_int iter, c_int compute_objective, c_int polish) {
  info.iter = iter;
  if (compute_objective) info.obj_val = compute_obj_val(work_x);
  if (data.m == 0) info.pri_res = 0.;
  else info.pri_res = compute_pri_res(work_x, work_z);
  info.dua_res = compute_dua_res(work_x, work_y);
}

void reset_info(OSQPInfo *info_ptr) {
  update_status(info_ptr, OSQP_UNSOLVED);
}

void update_status(OSQPInfo *info_ptr, c_int status_val) {
  info_ptr->status_val = status_val;
  if (status_val == OSQP_SOLVED) c_strcpy(info_ptr->status, "solved");
  else if (status_val == OSQP_SOLVED_INACCURATE) c_strcpy(info_ptr->status, "solved inaccurate");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE) c_strcpy(info_ptr->status, "primal infeasible");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) c_strcpy(info_ptr->status, "primal infeasible inaccurate");
  else if (status_val == OSQP_UNSOLVED) c_strcpy(info_ptr->status, "unsolved");
  else if (status_val == OSQP_DUAL_INFEASIBLE) c_strcpy(info_ptr->status, "dual infeasible");
  else if (status_val == OSQP_DUAL_INFEASIBLE_INACCURATE) c_strcpy(info_ptr->status, "dual infeasible inaccurate");
  else if (status_val == OSQP_MAX_ITER_REACHED) c_strcpy(info_ptr->status, "maximum iterations reached");
  else if (status_val == OSQP_SIGINT) c_strcpy(info_ptr->status, "interrupted");
  else if (status_val == OSQP_NON_CVX) c_strcpy(info_ptr->status, "problem non convex");
}

c_int check_termination(c_int approximate) {
  c_float eps_prim, eps_dual, eps_prim_inf, eps_dual_inf;
  c_int exitflag = 0, prim_res_check = 0, dual_res_check = 0, prim_inf_check = 0, dual_inf_check = 0;
  c_float eps_abs = settings.eps_abs, eps_rel = settings.eps_rel;
  eps_prim_inf = settings.eps_prim_inf; eps_dual_inf = settings.eps_dual_inf;

  if ((info.pri_res > OSQP_INFTY) || (info.dua_res > OSQP_INFTY)){
    update_status(&info, OSQP_NON_CVX); info.obj_val = OSQP_NAN; return 1;
  }
  if (approximate) { eps_abs *= 10; eps_rel *= 10; eps_prim_inf *= 10; eps_dual_inf *= 10; }

  if (data.m == 0) prim_res_check = 1;
  else {
    eps_prim = compute_pri_tol(eps_abs, eps_rel);
    if (info.pri_res < eps_prim) prim_res_check = 1;
    else prim_inf_check = is_primal_infeasible(eps_prim_inf);
  }
  eps_dual = compute_dua_tol(eps_abs, eps_rel);
  if (info.dua_res < eps_dual) dual_res_check = 1;
  else dual_inf_check = is_dual_infeasible(eps_dual_inf);

  if (prim_res_check && dual_res_check) {
    if (approximate) update_status(&info, OSQP_SOLVED_INACCURATE);
    else update_status(&info, OSQP_SOLVED);
    exitflag = 1;
  } else if (prim_inf_check) {
    if (approximate) update_status(&info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    else update_status(&info, OSQP_PRIMAL_INFEASIBLE);
    info.obj_val = OSQP_INFTY; exitflag = 1;
  } else if (dual_inf_check) {
    if (approximate) update_status(&info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    else update_status(&info, OSQP_DUAL_INFEASIBLE);
    info.obj_val = -OSQP_INFTY; exitflag = 1;
  }
  return exitflag;
}
