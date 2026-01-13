#include "glob_opts.h"
#include "qdldl.h"
#include "qdldl_interface.h"
#include "util.h"

// Nota: No incluimos amd.h ni kkt.h dinámicos si no son necesarios para evitar errores de linkado
// si tus archivos estáticos ya tienen la matriz KKT pre-calculada.

// Inicialización Dummy (En HLS estático ya viene todo inicializado en workspace.c)
c_int init_linsys_solver_qdldl(qdldl_solver ** sp, const csc * P, const csc * A, c_float sigma, const c_float * rho_vec, c_int polish){
    // En el modo estático, no necesitamos inicializar dinámicamente.
    // Devolvemos 0 para que OSQP crea que todo ha ido bien.
    return 0;
}

// Función auxiliar para permutaciones
void permute_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[j] = b[P[j]];
}

void permutet_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[P[j]] = b[j];
}

static void LDLSolve(c_float *x, c_float *b, const csc *L, const c_float *Dinv, const c_int *P, c_float *bp) {
    /* solves P'LDL'P x = b for x */
    permute_x(L->n, bp, b, P);
    QDLDL_solve(L->n, L->p, L->i, L->x, Dinv, bp);
    permutet_x(L->n, x, bp, P);
}

c_int solve_linsys_qdldl(qdldl_solver * s, c_float * b) {
    c_int j;

    // En HLS, asumimos que no hay polish o que la lógica es simple
    // Resolvemos el sistema KKT usando la factorización LDL almacenada en 's'

    // Paso 1: Resolver usando LDL
    // Usamos s->sol como buffer intermedio y s->bp como vector de permutación
    LDLSolve(s->sol, b, s->L, s->Dinv, s->P, s->bp);

    // Paso 2: Copiar resultados de vuelta a b
    // x_tilde (primeras n variables)
    for (j = 0 ; j < s->n ; j++) {
        b[j] = s->sol[j];
    }

    // z_tilde (siguientes m variables)
    // Recuperamos z_tilde a partir de y (solución dual)
    for (j = 0 ; j < s->m ; j++) {
        b[j + s->n] += s->rho_inv_vec[j] * s->sol[j + s->n];
    }

    return 0;
}

// Funciones de actualización (Activas porque EMBEDDED=2)

c_int update_linsys_solver_matrices_qdldl(qdldl_solver * s, const csc *P, const csc *A) {
    // Actualizar matriz KKT
    // Nota: update_KKT_P y update_KKT_A deben estar disponibles en kkt.c
    // Si no tienes kkt.c compilando, puedes necesitar incluir su contenido o añadir el archivo.

    #if EMBEDDED != 1
    // Solo intentamos actualizar si tenemos las funciones de KKT disponibles
    update_KKT_P(s->KKT, P, s->PtoKKT, s->sigma, s->Pdiag_idx, s->Pdiag_n);
    update_KKT_A(s->KKT, A, s->AtoKKT);
    #endif

    // Refactorizar la matriz KKT actualizada
    return QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork);
}

c_int update_linsys_solver_rho_vec_qdldl(qdldl_solver * s, const c_float * rho_vec){
    c_int i;

    // Actualizar vector rho interno
    for (i = 0; i < s->m; i++){
        s->rho_inv_vec[i] = 1. / rho_vec[i];
    }

    // Actualizar matriz KKT con nuevos valores de rho
    // update_KKT_param2 actualiza los elementos diagonales correspondientes a rho
    #if EMBEDDED != 1
    update_KKT_param2(s->KKT, s->rho_inv_vec, s->rhotoKKT, s->m);
    #endif

    // Refactorizar
    return QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork);
}

#ifndef EMBEDDED
void free_linsys_solver_qdldl(qdldl_solver * s) {
    // No-op en HLS
}
#endif
