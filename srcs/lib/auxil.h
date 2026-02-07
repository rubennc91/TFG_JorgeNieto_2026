#ifndef AUXIL_H
# define AUXIL_H

# ifdef __cplusplus
extern "C" {
# endif

# include "types.h"

// void c_strcpy(char dest[], const char source[]);

// --- FIRMAS LIMPIAS ---
void cold_start(void);
void update_info(c_int iter, c_int compute_objective, c_int polish);
void reset_info(OSQPInfo *info);
void update_status(OSQPInfo *info, c_int status_val);
c_int check_termination(c_int approximate);
void store_solution(void);
c_float compute_obj_val(c_float *x);

# ifdef __cplusplus
}
# endif

#endif
