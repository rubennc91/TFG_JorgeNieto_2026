#include "glob_opts.h"
#include "qdldl.h"
#include "qdldl_interface.h"
#include "util.h"
#include "kkt.h"
#include "workspace.h" // NECESARIO

c_int init_linsys_solver_qdldl(void){ return 0; }

void permute_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[j] = b[P[j]];
}

void permutet_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[P[j]] = b[j];
}

static void LDLSolve(c_float *x, c_float *b,
                     const c_int *Lp, const c_int *Li, const c_float *Lx, // Argumentos crudos
                     c_int Ln,
                     const c_float *Dinv, const c_int *P, c_float *bp) {
    permute_x(Ln, bp, b, P);
    QDLDL_solve(Ln, Lp, Li, Lx, Dinv, bp);
    permutet_x(Ln, x, bp, P);
}

c_int solve_linsys_qdldl(c_float * b) {
    c_int j;
    // CORRECCIÓN: Usamos arrays globales (linsys_solver_L_p, etc)
    // 34 es la dimensión n+m del sistema
    LDLSolve(linsys_solver_sol, b,
             linsys_solver_L_p, linsys_solver_L_i, linsys_solver_L_x, 34,
             linsys_solver_Dinv, linsys_solver_P, linsys_solver_bp);

    for (j = 0 ; j < 15 ; j++) { // n=15
        b[j] = linsys_solver_sol[j];
    }

    for (j = 0 ; j < 19 ; j++) { // m=19
        b[j + 15] += linsys_solver_rho_inv_vec[j] * linsys_solver_sol[j + 15];
    }
    return 0;
}

c_int update_linsys_solver_matrices_qdldl(void) {
    // CORRECCIÓN: Usamos arrays globales para KKT, P y A
    // linsys_solver_KKT_x es el array de valores de la matriz KKT
    update_KKT_P(linsys_solver_KKT_x, Pdata_x, Pdata_p, 15, // 15=n
                 linsys_solver_PtoKKT, linsys_solver.sigma, linsys_solver_Pdiag_idx, 10);

    update_KKT_A(linsys_solver_KKT_x, Adata_x, Adata_p, 15, // 15=n
                 linsys_solver_AtoKKT);

    // Pasamos los arrays globales a QDLDL_factor
    return QDLDL_factor(34, linsys_solver_KKT_p, linsys_solver_KKT_i, linsys_solver_KKT_x,
        linsys_solver_L_p, linsys_solver_L_i, linsys_solver_L_x,
        linsys_solver_D, linsys_solver_Dinv, linsys_solver_Lnz,
        linsys_solver_etree, linsys_solver_bwork, linsys_solver_iwork, linsys_solver_fwork);
}

c_int update_linsys_solver_rho_vec_qdldl(const c_float * rho_vec){
    c_int i;
    for (i = 0; i < 19; i++){
        linsys_solver_rho_inv_vec[i] = 1. / rho_vec[i];
    }
    // CORRECCIÓN: Usamos arrays globales
    update_KKT_param2(linsys_solver_KKT_x, linsys_solver_rho_inv_vec, linsys_solver_rhotoKKT, 19);

    return QDLDL_factor(34, linsys_solver_KKT_p, linsys_solver_KKT_i, linsys_solver_KKT_x,
        linsys_solver_L_p, linsys_solver_L_i, linsys_solver_L_x,
        linsys_solver_D, linsys_solver_Dinv, linsys_solver_Lnz,
        linsys_solver_etree, linsys_solver_bwork, linsys_solver_iwork, linsys_solver_fwork);
}
