#ifndef QDLDL_INTERFACE_H
#define QDLDL_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "qdldl_types.h"

/**
 * QDLDL solver structure
 */
typedef struct qdldl qdldl_solver;

struct qdldl {
    enum linsys_solver_type type;

    // NOTA HLS: Eliminamos los punteros a funciones (*solve, *free, etc.)
    // porque el hardware no soporta llamadas dinámicas.

    /**
     * @name Attributes
     * @{
     */
    csc *L;                 ///< lower triangular matrix in LDL factorization
    c_float *Dinv;          ///< inverse of diag matrix in LDL (as a vector)
    c_int   *P;             ///< permutation of KKT matrix for factorization
    c_float *bp;            ///< workspace memory for solves
    c_float *sol;           ///< solution to the KKT system
    c_float *rho_inv_vec;   ///< parameter vector
    c_float sigma;          ///< scalar parameter

    c_int n;                ///< number of QP variables
    c_int m;                ///< number of QP constraints

    // Members required for matrix updates (Active in EMBEDDED=2)
    c_int * Pdiag_idx;
    c_int   Pdiag_n;        ///< index and number of diagonal elements in P
    csc   * KKT;            ///< Permuted KKT matrix in sparse form
    c_int * PtoKKT;         ///< Index of elements from P to KKT matrix
    c_int * AtoKKT;         ///< Index of elements from A to KKT matrix
    c_int * rhotoKKT;       ///< Index of rho places in KKT matrix

    // QDLDL Numeric workspace
    QDLDL_float *D;
    QDLDL_int   *etree;
    QDLDL_int   *Lnz;
    QDLDL_int   *iwork;
    QDLDL_bool  *bwork;
    QDLDL_float *fwork;

    /** @} */
};

// --- FUNCIONES ADAPTADAS A LA ESTRATEGIA GLOBAL (SIN PUNTEROS A ESTRUCTURAS) ---

/**
 * Initialize QDLDL Solver (Dummy in HLS static allocation)
 */
c_int init_linsys_solver_qdldl(void);

/**
 * Solve linear system and store result in b
 * Uses global 'linsys_solver' implicitly.
 * @param  b  Right-hand side (and solution output)
 */
c_int solve_linsys_qdldl(c_float * b);

/**
 * Update linear system solver matrices
 * Uses global 'linsys_solver', 'Pdata' and 'Adata' implicitly.
 */
c_int update_linsys_solver_matrices_qdldl(void);

/**
 * Update rho_vec parameter
 * Uses global 'linsys_solver' implicitly.
 * @param  rho_vec  new rho_vec value
 */
c_int update_linsys_solver_rho_vec_qdldl(const c_float * rho_vec);

#ifdef __cplusplus
}
#endif

#endif
