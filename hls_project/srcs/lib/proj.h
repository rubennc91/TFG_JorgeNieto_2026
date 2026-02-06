#ifndef PROJ_H
# define PROJ_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"
# include "lin_alg.h" // Necesario para c_max/c_min

// Función project simplificada (usa globales)
void project(c_float *z);

# ifdef __cplusplus
}
# endif

#endif
