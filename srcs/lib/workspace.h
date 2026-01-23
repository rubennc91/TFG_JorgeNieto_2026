#ifndef WORKSPACE_H
#define WORKSPACE_H

#include "types.h"
#include "qdldl_interface.h"

// Data structure prototypes
extern csc Pdata;
extern csc Adata;
extern c_float qdata[15];
extern c_float ldata[19];
extern c_float udata[19];
extern OSQPData data;

// Settings structure prototype
extern OSQPSettings settings;

// Scaling structure prototypes
extern OSQPScaling scaling;

// --- NUEVO: Arrays de escalado globales ---
extern c_float scaling_D[15];
extern c_float scaling_Dinv[15];
extern c_float scaling_E[19];
extern c_float scaling_Einv[19];
// ------------------------------------------

// Prototypes for linsys_solver structure
extern csc linsys_solver_L;
extern c_float linsys_solver_Dinv[34];
extern c_int linsys_solver_P[34];
extern c_float linsys_solver_bp[34];
extern c_float linsys_solver_sol[34];
extern c_float linsys_solver_rho_inv_vec[19];
extern c_int linsys_solver_Pdiag_idx[10];
extern csc linsys_solver_KKT;
extern c_int linsys_solver_PtoKKT[12];
extern c_int linsys_solver_AtoKKT[43];
extern c_int linsys_solver_rhotoKKT[19];
extern QDLDL_float linsys_solver_D[34];
extern QDLDL_int linsys_solver_etree[34];
extern QDLDL_int linsys_solver_Lnz[34];
extern QDLDL_int   linsys_solver_iwork[102];
extern QDLDL_bool  linsys_solver_bwork[34];
extern QDLDL_float linsys_solver_fwork[34];
extern qdldl_solver linsys_solver;

// Prototypes for solution
extern c_float xsolution[15];
extern c_float ysolution[19];

extern OSQPSolution solution;

// Prototype for info structure
extern OSQPInfo info;

// Prototypes for the workspace
extern c_float work_rho_vec[19];
extern c_float work_rho_inv_vec[19];
extern c_int work_constr_type[19];
extern c_float work_x[15];
extern c_float work_y[19];
extern c_float work_z[19];
extern c_float work_xz_tilde[34];
extern c_float work_x_prev[15];
extern c_float work_z_prev[19];
extern c_float work_Ax[19];
extern c_float work_Px[15];
extern c_float work_Aty[15];
extern c_float work_delta_y[19];
extern c_float work_Atdelta_y[15];
extern c_float work_delta_x[15];
extern c_float work_Pdelta_x[15];
extern c_float work_Adelta_x[19];
extern c_float work_D_temp[15];
extern c_float work_D_temp_A[15];
extern c_float work_E_temp[19];

extern OSQPWorkspace workspace;

#endif // ifndef WORKSPACE_H
