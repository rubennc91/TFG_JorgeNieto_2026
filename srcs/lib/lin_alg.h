#ifndef LIN_ALG_H
# define LIN_ALG_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"

// Funciones vectoriales
void vec_add_scaled(c_float *c, const c_float *a, const c_float *b, c_int n, c_float sc);
c_float vec_scaled_norm_inf(const c_float *S, const c_float *v, c_int l);
c_float vec_norm_inf(const c_float *v, c_int l);
c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l);
c_float vec_mean(const c_float *a, c_int n);
void int_vec_set_scalar(c_int *a, c_int sc, c_int n);
void vec_set_scalar(c_float *a, c_float sc, c_int n);
void vec_add_scalar(c_float *a, c_float sc, c_int n);
void vec_mult_scalar(c_float *a, c_float sc, c_int n);
void prea_int_vec_copy(const c_int *a, c_int *b, c_int n);
void prea_vec_copy(const c_float *a, c_float *b, c_int n);
void vec_ew_recipr(const c_float *a, c_float *b, c_int n);
c_float vec_prod(const c_float *a, const c_float *b, c_int n);
void vec_ew_prod(const c_float *a, const c_float *b, c_float *c, c_int n);

// --- FUNCIONES MATRICIALES PLANAS (SIN STRUCTS CSC) ---
// Reciben x, p, i, n, m directamente

void mat_mult_scalar(c_float *Ax, const c_int *Ap, c_int An, c_float sc);

void mat_premult_diag(c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, const c_float *d);

void mat_postmult_diag(c_float *Ax, const c_int *Ap, c_int An, const c_float *d);

void mat_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
             const c_float *x, c_float *y, c_int plus_eq);

void mat_tpose_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
                   const c_float *x, c_float *y, c_int plus_eq, c_int skip_diag);

c_float quad_form(const c_float *Px, const c_int *Pp, const c_int *Pi, c_int Pn, const c_float *x);

# ifdef __cplusplus
}
# endif

#endif
