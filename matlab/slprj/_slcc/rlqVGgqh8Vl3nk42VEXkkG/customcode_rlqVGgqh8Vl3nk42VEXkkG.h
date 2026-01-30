#ifndef __customcode_rlqVGgqh8Vl3nk42VEXkkG_h__
#define __customcode_rlqVGgqh8Vl3nk42VEXkkG_h__

/* Include files */
#include "mex.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "tmwtypes.h"


/* Helper definitions for DLL support */
#if defined _WIN32 
  #define DLL_EXPORT_CC __declspec(dllexport)
#else
  #if __GNUC__ >= 4
    #define DLL_EXPORT_CC __attribute__ ((visibility ("default")))
  #else
    #define DLL_EXPORT_CC
  #endif
#endif
/* Custom Code from Simulation Target dialog */
#include "simulink_block.h"

/* Function Declarations */
#ifdef __cplusplus
extern "C" {
#endif
#define customcode_rlqVGgqh8Vl3nk42VEXkkG_initializer()

#define customcode_rlqVGgqh8Vl3nk42VEXkkG_terminator()
#ifdef __cplusplus
}
#endif

#endif

