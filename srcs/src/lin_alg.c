#include "lin_alg.h"

// Vectoriales estándar
void vec_add_scaled(c_float *c, const c_float *a, const c_float *b, c_int n, c_float sc) {
  c_int i; for (i = 0; i < n; i++) c[i] =  a[i] + sc * b[i];
}
c_float vec_scaled_norm_inf(const c_float *S, const c_float *v, c_int l) {
  c_int i; c_float abs_Sv_i, max = 0.0;
  for (i = 0; i < l; i++) { abs_Sv_i = c_absval(S[i] * v[i]); if (abs_Sv_i > max) max = abs_Sv_i; }
  return max;
}
c_float vec_norm_inf(const c_float *v, c_int l) {
  c_int i; c_float abs_v_i, max = 0.0;
  for (i = 0; i < l; i++) { abs_v_i = c_absval(v[i]); if (abs_v_i > max) max = abs_v_i; }
  return max;
}
c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l) {
  c_float nmDiff = 0.0, tmp; c_int i;
  for (i = 0; i < l; i++) { tmp = c_absval(a[i] - b[i]); if (tmp > nmDiff) nmDiff = tmp; }
  return nmDiff;
}
c_float vec_mean(const c_float *a, c_int n) {
  c_float mean = 0.0; c_int i;
  for (i = 0; i < n; i++) mean += a[i];
  return mean / (c_float)n;
}
void int_vec_set_scalar(c_int *a, c_int sc, c_int n) {
  c_int i; for (i = 0; i < n; i++) a[i] = sc;
}
void vec_set_scalar(c_float *a, c_float sc, c_int n) {
  c_int i; for (i = 0; i < n; i++) a[i] = sc;
}
void vec_add_scalar(c_float *a, c_float sc, c_int n) {
  c_int i; for (i = 0; i < n; i++) a[i] += sc;
}
void vec_mult_scalar(c_float *a, c_float sc, c_int n) {
  c_int i; for (i = 0; i < n; i++) a[i] *= sc;
}
void prea_int_vec_copy(const c_int *a, c_int *b, c_int n) {
  c_int i; for (i = 0; i < n; i++) b[i] = a[i];
}
void prea_vec_copy(const c_float *a, c_float *b, c_int n) {
  c_int i; for (i = 0; i < n; i++) b[i] = a[i];
}
void vec_ew_recipr(const c_float *a, c_float *b, c_int n) {
  c_int i; for (i = 0; i < n; i++) b[i] = (c_float)1.0 / a[i];
}
c_float vec_prod(const c_float *a, const c_float *b, c_int n) {
  c_float prod = 0.0; c_int i; for (i = 0; i < n; i++) prod += a[i] * b[i];
  return prod;
}
void vec_ew_prod(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i; for (i = 0; i < n; i++) c[i] = b[i] * a[i];
}

// --- MATRICIALES (Planos) ---

void mat_mult_scalar(c_float *Ax, const c_int *Ap, c_int An, c_float sc) {
  c_int i, nnzA = Ap[An];
  for (i = 0; i < nnzA; i++) Ax[i] *= sc;
}

void mat_premult_diag(c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, const c_float *d) {
  c_int j, i;
  for (j = 0; j < An; j++) {
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      Ax[i] *= d[Ai[i]];
    }
  }
}

void mat_postmult_diag(c_float *Ax, const c_int *Ap, c_int An, const c_float *d) {
  c_int j, i;
  for (j = 0; j < An; j++) {
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      Ax[i] *= d[j];
    }
  }
}

void mat_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
             const c_float *x, c_float *y, c_int plus_eq) {
  c_int i, j;
  if (!plus_eq) {
    for (i = 0; i < Am; i++) y[i] = 0;
  }
  if (Ap[An] == 0) return;

  for (j = 0; j < An; j++) {
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      if (plus_eq == -1) y[Ai[i]] -= Ax[i] * x[j];
      else               y[Ai[i]] += Ax[i] * x[j];
    }
  }
}

void mat_tpose_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
                   const c_float *x, c_float *y, c_int plus_eq, c_int skip_diag) {
  c_int i, j, k;
  if (!plus_eq) {
    for (i = 0; i < An; i++) y[i] = 0;
  }
  if (Ap[An] == 0) return;

  for (j = 0; j < An; j++) {
    for (k = Ap[j]; k < Ap[j + 1]; k++) {
      i = Ai[k];
      if (skip_diag && i == j) continue;
      if (plus_eq == -1) y[j] -= Ax[k] * x[i];
      else               y[j] += Ax[k] * x[i];
    }
  }
}

c_float quad_form(const c_float *Px, const c_int *Pp, const c_int *Pi, c_int Pn, const c_float *x) {
  c_float quad_form = 0.;
  c_int i, j, ptr;
  for (j = 0; j < Pn; j++) {
    for (ptr = Pp[j]; ptr < Pp[j + 1]; ptr++) {
      i = Pi[ptr];
      if (i == j) quad_form += (c_float)0.5 * Px[ptr] * x[i] * x[i];
      else if (i < j) quad_form += Px[ptr] * x[i] * x[j];
    }
  }
  return quad_form;
}
