#ifndef SCALING_H
# define SCALING_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"
# include "lin_alg.h" // Necesario para evitar avisos implícitos

// --- FIRMAS LIMPIAS ---
c_int scale_data(void);
c_int unscale_data(void);
c_int unscale_solution(void);

# ifdef __cplusplus
}
# endif

#endif
