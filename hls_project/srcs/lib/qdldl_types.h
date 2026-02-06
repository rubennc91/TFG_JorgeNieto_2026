#ifndef QDLDL_TYPES_H
#define QDLDL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h" // Para acceder a c_float, c_int y la macro DLONG
#include <limits.h> // <--- NECESARIO para INT_MAX y LLONG_MAX

// --- CAMBIO: Vinculamos los tipos de QDLDL a los de OSQP ---

typedef c_float QDLDL_float;
typedef c_int   QDLDL_int;
typedef c_int   QDLDL_bool;

// --- CORRECCIÓN: Definir el valor máximo ---
// Esto era lo que faltaba y causaba el error en qdldl.c
#ifdef DLONG
# define QDLDL_INT_MAX LLONG_MAX
#else
# define QDLDL_INT_MAX INT_MAX
#endif
// -------------------------------------------

#ifdef __cplusplus
}
#endif

#endif /* QDLDL_TYPES_H */
