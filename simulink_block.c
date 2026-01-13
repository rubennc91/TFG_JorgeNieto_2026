#include <stdlib.h>
#include "mpc_util.h"
#include "workspace.h"
#include "osqp.h"
#include "lin_alg.h"
#include "simulink_block.h"

#define Q1 1000.0f
#define AC_FREQ_HZ 50.0f
#define ISR1_FREQUENCY_HZ 20000.0f
#define V_AMP 180.0f
#define R 0.1f
#define L 5e-3f
#define Cap 1000e-6f
#define w 3.141592653589793e+02f
#define REFERENCIA 380.0f


void myFunction(double x_ini[3], double Vsd, double Vsq, double iL,  double u00[2], double outputVector[2])
{
    double vdc, Ex11, Ex12, Ex21, Ex22, Ax01, Ax02, Rl;
    double Ax[2];
    double Ex[2][2];
    double Einv[2][2]; 
    double z_ini[3];
    double v00[2];
    double v[2];
    double u[2];
    double q_new[15];
    double l_new[19];
    double u_new[19];
    double ref = REFERENCIA;

    // x_ini[0] = 5.756828677424010f;
    // x_ini[1] = 0.502794263642644f;
    // x_ini[2] = 3.801079990283232e+02f;
    // Vsd = 1.357631599873370e+02f;
    // Vsq = -0.212146252529920f;
    // Rl = 1.000000109998001e+02f;
    // u00[0] = 1.351171606300369e+02f;
    // u00[1] = -30.237060892886790f;

    vdc = x_ini[2];

    if (abs(iL)<.1) {
        Rl = 10000;
    } else {
        Rl = vdc/iL;
    }

    prea_vec_copy(workspace.data->q,q_new, workspace.data->n);
    prea_vec_copy(workspace.data->l,l_new, workspace.data->m);
    prea_vec_copy(workspace.data->u,u_new, workspace.data->m);
    

    Ax01 = - w*x_ini[0] - (R*x_ini[1])/L;

    Ax02 = (1.0f / (Cap * Rl) + (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap * x_ini[2] * x_ini[2])) *
            (x_ini[2] / (Cap * Rl) - (3.0 * (Vsd * x_ini[0] + Vsq * x_ini[1] - R * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap * x_ini[2])) -
            (3.0f * (w * x_ini[0] + (R * x_ini[1]) / L) * (Vsq - 2.0f * R * x_ini[1])) / (2.0f * Cap * x_ini[2]) +
            (3.0f * (Vsd - 2.0f * R * x_ini[0]) * (w * x_ini[1] + Vsd / L - (R * x_ini[0]) / L)) / (2.0f * Cap * x_ini[2]);


    Ex11 =  0;
    Ex12 = -1/L;

    Ex21 = -(3*(Vsd - 2*R*x_ini[0]))/(2*Cap*L*x_ini[2]);
    Ex22 = -(3*(Vsq - 2*R*x_ini[1]))/(2*Cap*L*x_ini[2]);

    Ax[0] = Ax01;
    Ax[1] = Ax02;
    Ex[0][0] = Ex11;
    Ex[0][1] = Ex12;
    Ex[1][0] = Ex21;
    Ex[1][1] = Ex22;

    inverse_matrix_2x2(Ex11, Ex12, Ex21, Ex22, Einv);

    z_ini[0] = x_ini[1];
    z_ini[1] = x_ini[2];        
    z_ini[2] = (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - R * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * Cap * x_ini[2]) - x_ini[2] / (Cap * Rl);

    ////////////////////////
    referencia(&workspace, q_new, ref);
    calculateV(Ax, Ex, u00, v00);
    atualizar_restricao(l_new, u_new, z_ini, v00);
    atualizar_restricao_v(&workspace, l_new, u_new, z_ini[1], Einv, Ax);
    atualizar_A(&workspace, Einv);

    // Solve Problem
    osqp_solve(&workspace);

    v[0] = workspace.x[11];
    v[1] = workspace.x[12];

    v[0] = v[0] - Ax[0];
    v[1] = v[1] - Ax[1];

    multiplyMatrixVector(Einv, v, u);

    u[0] = u[0]/vdc;
    u[1] = u[1]/vdc;

    outputVector[0] = workspace.x[11];
    outputVector[1] = workspace.x[12];  




    // Example computation (replace with actual logic)
    // outputVector[0] = x_ini[0] + Vsd + u00[0];
    // outputVector[1] = x_ini[1] + Vsq + u00[1];    
}
