#ifndef QDLDL_TYPES_H
#define QDLDL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h" // Importante: incluimos types.h para acceder a c_float/c_int

// --- CAMBIO: Vinculamos los tipos de QDLDL a los de OSQP ---

// En lugar de usar double/long por defecto, usamos lo que diga OSQP
typedef c_float QDLDL_float;  // Ahora será float
typedef c_int   QDLDL_int;    // Ahora será int (o long long si DLONG está activo)
typedef c_int   QDLDL_bool;   // Boolean compatible

// -----------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif /* QDLDL_TYPES_H */
