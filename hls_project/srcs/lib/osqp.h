#ifndef OSQP_H
# define OSQP_H

# ifdef __cplusplus
extern "C" {
# endif

#include "types.h"
#include "glob_opts.h"

// Configuración
void osqp_set_default_settings(OSQPSettings *settings);

// --- FUNCIONES MODIFICADAS (SIN ARGUMENTO WORKSPACE) ---

c_int osqp_solve(void);

c_int osqp_update_lin_cost(const c_float *q_new);

c_int osqp_update_bounds(const c_float *l_new,
                         const c_float *u_new);

c_int osqp_update_P(const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n);

c_int osqp_update_A(const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n);

c_int osqp_update_rho(c_float rho_new);

// Funciones internas expuestas
void update_xz_tilde(void);
void update_x(void);
void update_z(void);
void update_y(void);

# ifdef __cplusplus
}
# endif

#endif
