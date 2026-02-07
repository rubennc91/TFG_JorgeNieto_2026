/* Provide Declarations */
#include <stdarg.h>
#include <setjmp.h>
#include <limits.h>
#ifdef NEED_CBEAPINT
#include <autopilot_cbe.h>
#else
#define aesl_fopen fopen
#define aesl_freopen freopen
#define aesl_tmpfile tmpfile
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#ifdef __STRICT_ANSI__
#define inline __inline__
#define typeof __typeof__ 
#endif
#define __isoc99_fscanf fscanf
#define __isoc99_sscanf sscanf
#undef ferror
#undef feof
/* get a declaration for alloca */
#if defined(__CYGWIN__) || defined(__MINGW32__)
#define  alloca(x) __builtin_alloca((x))
#define _alloca(x) __builtin_alloca((x))
#elif defined(__APPLE__)
extern void *__builtin_alloca(unsigned long);
#define alloca(x) __builtin_alloca(x)
#define longjmp _longjmp
#define setjmp _setjmp
#elif defined(__sun__)
#if defined(__sparcv9)
extern void *__builtin_alloca(unsigned long);
#else
extern void *__builtin_alloca(unsigned int);
#endif
#define alloca(x) __builtin_alloca(x)
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__) || defined(__arm__)
#define alloca(x) __builtin_alloca(x)
#elif defined(_MSC_VER)
#define inline _inline
#define alloca(x) _alloca(x)
#else
#include <alloca.h>
#endif

#ifndef __GNUC__  /* Can only support "linkonce" vars with GCC */
#define __attribute__(X)
#endif

#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __EXTERNAL_WEAK__ __attribute__((weak_import))
#elif defined(__GNUC__)
#define __EXTERNAL_WEAK__ __attribute__((weak))
#else
#define __EXTERNAL_WEAK__
#endif

#if defined(__GNUC__) && (defined(__APPLE_CC__) || defined(__CYGWIN__) || defined(__MINGW32__))
#define __ATTRIBUTE_WEAK__
#elif defined(__GNUC__)
#define __ATTRIBUTE_WEAK__ __attribute__((weak))
#else
#define __ATTRIBUTE_WEAK__
#endif

#if defined(__GNUC__)
#define __HIDDEN__ __attribute__((visibility("hidden")))
#endif

#ifdef __GNUC__
#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */
#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */
#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */
#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */
#define LLVM_INF           __builtin_inf()         /* Double */
#define LLVM_INFF          __builtin_inff()        /* Float */
#define LLVM_PREFETCH(addr,rw,locality) __builtin_prefetch(addr,rw,locality)
#define __ATTRIBUTE_CTOR__ __attribute__((constructor))
#define __ATTRIBUTE_DTOR__ __attribute__((destructor))
#define LLVM_ASM           __asm__
#else
#define LLVM_NAN(NanStr)   ((double)0.0)           /* Double */
#define LLVM_NANF(NanStr)  0.0F                    /* Float */
#define LLVM_NANS(NanStr)  ((double)0.0)           /* Double */
#define LLVM_NANSF(NanStr) 0.0F                    /* Float */
#define LLVM_INF           ((double)0.0)           /* Double */
#define LLVM_INFF          0.0F                    /* Float */
#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */
#define __ATTRIBUTE_CTOR__
#define __ATTRIBUTE_DTOR__
#define LLVM_ASM(X)
#endif

#if __GNUC__ < 4 /* Old GCC's, or compilers not GCC */ 
#define __builtin_stack_save() 0   /* not implemented */
#define __builtin_stack_restore(X) /* noop */
#endif

#if __GNUC__ && __LP64__ /* 128-bit integer types */
typedef int __attribute__((mode(TI))) llvmInt128;
typedef unsigned __attribute__((mode(TI))) llvmUInt128;
#endif

#define CODE_FOR_MAIN() /* Any target-specific code for main()*/

#ifndef __cplusplus
typedef unsigned char bool;
#endif


/* Support for floating point constants */
typedef unsigned long long ConstantDoubleTy;
typedef unsigned int        ConstantFloatTy;
typedef struct { unsigned long long f1; unsigned short f2; unsigned short pad[3]; } ConstantFP80Ty;
typedef struct { unsigned long long f1; unsigned long long f2; } ConstantFP128Ty;


/* Global Declarations */
/* Helper union for bitcasts */
typedef union {
  unsigned int Int32;
  unsigned long long Int64;
  float Float;
  double Double;
} llvmBitCastUnion;
/* Structure forward decls */
typedef struct l_struct_OC_csc l_struct_OC_csc;
typedef struct l_struct_OC_OSQPData l_struct_OC_OSQPData;
typedef struct l_struct_OC_OSQPSettings l_struct_OC_OSQPSettings;
typedef struct l_struct_OC_qdldl l_struct_OC_qdldl;
typedef struct l_struct_OC_OSQPSolution l_struct_OC_OSQPSolution;
typedef struct l_struct_OC_OSQPInfo l_struct_OC_OSQPInfo;
typedef struct l_struct_OC_OSQPScaling l_struct_OC_OSQPScaling;
typedef struct l_struct_OC_OSQPWorkspace l_struct_OC_OSQPWorkspace;
typedef struct l_struct_OC_linsys_solver l_struct_OC_linsys_solver;

/* Structure contents */
struct l_struct_OC_csc {
  unsigned long long field0;
  unsigned long long field1;
  unsigned long long field2;
  signed long long *field3;
  signed long long *field4;
  float *field5;
  unsigned long long field6;
};

struct l_struct_OC_OSQPData {
  unsigned long long field0;
  unsigned long long field1;
  l_struct_OC_csc *field2;
  l_struct_OC_csc *field3;
  float *field4;
  float *field5;
  float *field6;
};

struct l_struct_OC_OSQPSettings {
  float field0;
  float field1;
  unsigned long long field2;
  unsigned long long field3;
  unsigned long long field4;
  float field5;
  unsigned long long field6;
  float field7;
  float field8;
  float field9;
  float field10;
  float field11;
  bool field12;
  unsigned long long field13;
  unsigned long long field14;
  unsigned long long field15;
};

struct l_struct_OC_qdldl {
  bool field0;
  l_struct_OC_csc *field1;
  float *field2;
  signed long long *field3;
  float *field4;
  float *field5;
  float *field6;
  float field7;
  unsigned long long field8;
  unsigned long long field9;
  signed long long *field10;
  unsigned long long field11;
  l_struct_OC_csc *field12;
  signed long long *field13;
  signed long long *field14;
  signed long long *field15;
  float *field16;
  signed long long *field17;
  signed long long *field18;
  signed long long *field19;
  signed long long *field20;
  float *field21;
};

struct l_struct_OC_OSQPSolution {
  float *field0;
  float *field1;
};

struct l_struct_OC_OSQPInfo {
  unsigned long long field0;
   char field1[32];
  unsigned long long field2;
  float field3;
  float field4;
  float field5;
  unsigned long long field6;
  float field7;
};

struct l_struct_OC_OSQPScaling {
  float field0;
  float *field1;
  float *field2;
  float field3;
  float *field4;
  float *field5;
};

struct l_struct_OC_OSQPWorkspace {
  l_struct_OC_OSQPData *field0;
  l_struct_OC_linsys_solver *field1;
  float *field2;
  float *field3;
  signed long long *field4;
  float *field5;
  float *field6;
  float *field7;
  float *field8;
  float *field9;
  float *field10;
  float *field11;
  float *field12;
  float *field13;
  float *field14;
  float *field15;
  float *field16;
  float *field17;
  float *field18;
  float *field19;
  float *field20;
  float *field21;
  l_struct_OC_OSQPSettings *field22;
  l_struct_OC_OSQPScaling *field23;
  l_struct_OC_OSQPSolution *field24;
  l_struct_OC_OSQPInfo *field25;
};

struct l_struct_OC_linsys_solver {
  bool field0;
  unsigned long long  (*field1) (l_struct_OC_linsys_solver *, float *);
  unsigned long long  (*field2) (l_struct_OC_linsys_solver *, l_struct_OC_csc *, l_struct_OC_csc *);
  unsigned long long  (*field3) (l_struct_OC_linsys_solver *, float *);
};


/* External Global Variable Declarations */
extern signed long long Pdata_i[12];
extern signed long long Pdata_p[16];
extern float Pdata_x[12];
extern l_struct_OC_csc Pdata;
extern signed long long Adata_i[43];
extern signed long long Adata_p[16];
extern float Adata_x[43];
extern l_struct_OC_csc Adata;
extern float qdata[15];
extern float ldata[19];
extern float udata[19];
extern l_struct_OC_OSQPData data;
extern l_struct_OC_OSQPSettings settings;
extern signed long long linsys_solver_L_i[57];
extern signed long long linsys_solver_L_p[35];
extern float linsys_solver_L_x[57];
extern l_struct_OC_csc linsys_solver_L;
extern float linsys_solver_Dinv[34];
extern signed long long linsys_solver_P[34];
extern float linsys_solver_rho_inv_vec[19];
extern signed long long linsys_solver_Pdiag_idx[10];
extern signed long long linsys_solver_KKT_i[79];
extern signed long long linsys_solver_KKT_p[35];
extern float linsys_solver_KKT_x[79];
extern l_struct_OC_csc linsys_solver_KKT;
extern signed long long linsys_solver_PtoKKT[12];
extern signed long long linsys_solver_AtoKKT[43];
extern signed long long linsys_solver_rhotoKKT[19];
extern float linsys_solver_bp[34];
extern float linsys_solver_sol[34];
extern float linsys_solver_D[34];
extern signed long long linsys_solver_etree[34];
extern signed long long linsys_solver_Lnz[34];
extern signed long long linsys_solver_iwork[102];
extern signed long long linsys_solver_bwork[34];
extern float linsys_solver_fwork[34];
extern l_struct_OC_qdldl linsys_solver;
extern float xsolution[15];
extern float ysolution[19];
extern l_struct_OC_OSQPSolution solution;
extern l_struct_OC_OSQPInfo info;
extern float work_rho_vec[19];
extern float work_rho_inv_vec[19];
extern signed long long work_constr_type[19];
extern float work_x[15];
extern float work_y[19];
extern float work_z[19];
extern float work_xz_tilde[34];
extern float work_x_prev[15];
extern float work_z_prev[19];
extern float work_Ax[19];
extern float work_Px[15];
extern float work_Aty[15];
extern float work_delta_y[19];
extern float work_Atdelta_y[15];
extern float work_delta_x[15];
extern float work_Pdelta_x[15];
extern float work_Adelta_x[19];
extern float work_D_temp[15];
extern float work_D_temp_A[15];
extern float work_E_temp[19];
extern l_struct_OC_OSQPScaling scaling;
extern l_struct_OC_OSQPWorkspace workspace;
extern float scaling_D[15];
extern float scaling_Dinv[15];
extern float scaling_E[19];
extern float scaling_Einv[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);


/* Global Variable Definitions and Initialization */
signed long long Adata_p[16] = { 0ull, 2ull, 4ull, 7ull, 9ull, 11ull, 14ull, 15ull, 16ull, 17ull, 20ull, 22ull, 29ull, 34ull, 39ull, 43ull };
float Pdata_x[12] = { 0x1.e848p19, 0x1.388p13, 0x1.e848p19, 0x1.388p13, 0x1.86ap16, 0x1.86ap16, 0x1.9p6, 0x1.a36e2ep-14, -0x1.a36e2ep-14, 0x1.a36e2ep-13, -0x1.a36e2ep-14, 0x1.a36e2ep-14 };
float qdata[15] = { 0x0p0, -0x1.cfdep21, 0x0p0, 0x0p0, -0x1.cfdep21, 0x0p0, 0x0p0, -0x1.b2e02p26, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0 };
float udata[19] = { 0x1.571098p-27, -0x1.7bfa74p8, 0x1.2ffb8ap5, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, -0x1.461c34p-18, 0x1.5af4cap24, 0x1.448994p4, 0x1.7bfa74p7, 0x1.448994p4, 0x1.7bfa74p7, 0x1.c7f958p6, 0x1.c7f958p6, 0x1.c7f958p6, 0x1.c7f958p6 };
l_struct_OC_OSQPSettings settings = { 0x1.28b464p-3, 0x1.0c6f7ap-20, 0ull, 1ull, 25ull, 0x1.4p2, 10000ull, 0x1.0624dep-10, 0x1.0624dep-10, 0x1.a36e2ep-14, 0x1.a36e2ep-14, 0x1.99999ap0, 0, 0ull, 10000ull, 0ull };
signed long long linsys_solver_L_i[57] = { 31ull, 32ull, 30ull, 3ull, 8ull, 6ull, 31ull, 7ull, 31ull, 8ull, 31ull, 30ull, 31ull, 30ull, 31ull, 11ull, 15ull, 14ull, 14ull, 30ull, 15ull, 30ull, 29ull, 30ull, 29ull, 30ull, 18ull, 27ull, 20ull, 27ull, 28ull, 25ull, 25ull, 27ull, 24ull, 26ull, 32ull, 26ull, 27ull, 32ull, 27ull, 28ull, 32ull, 28ull, 29ull, 32ull, 29ull, 32ull, 30ull, 32ull, 33ull, 31ull, 32ull, 33ull, 32ull, 33ull, 33ull };
signed long long linsys_solver_L_p[35] = { 0ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 9ull, 11ull, 13ull, 15ull, 16ull, 17ull, 18ull, 20ull, 22ull, 24ull, 26ull, 27ull, 28ull, 29ull, 31ull, 32ull, 34ull, 35ull, 37ull, 40ull, 43ull, 46ull, 48ull, 51ull, 54ull, 56ull, 57ull, 57ull };
signed long long Pdata_i[12] = { 0ull, 1ull, 3ull, 4ull, 6ull, 7ull, 8ull, 9ull, 9ull, 11ull, 11ull, 13ull };
float Adata_x[43] = { -0x1p0, 0x1p0, -0x1p0, 0x1p0, -0x1p0, 0x1.a36e2ep-14, 0x1p0, -0x1p0, 0x1p0, -0x1p0, 0x1p0, -0x1p0, 0x1.a36e2ep-14, 0x1p0, -0x1p0, -0x1p0, -0x1p0, 0x1p0, -0x1.05e9bcp-17, -0x1.47ae14p-8, 0x1p0, -0x1.f4dda8p-18, 0x1.a36e2ep-14, -0x1.05e9bcp-17, -0x1.47ae14p-8, 0x1.05e9bcp-17, 0x1.47ae14p-8, -0x1.05e9bcp-17, -0x1.47ae14p-8, 0x1.5798eep-28, 0x1.a36e2ep-14, -0x1.f4dda8p-18, 0x1.f4dda8p-18, -0x1.f4dda8p-18, 0x1.a36e2ep-14, -0x1.05e9bcp-17, -0x1.47ae14p-8, 0x1.05e9bcp-17, 0x1.47ae14p-8, 0x1.5798eep-28, 0x1.a36e2ep-14, -0x1.f4dda8p-18, 0x1.f4dda8p-18 };
float ldata[19] = { 0x1.571098p-27, -0x1.7bfa74p8, 0x1.2ffb8ap5, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, 0x0p0, -0x1.461c34p-18, 0x1.5af4cap24, -0x1.67b1dap8, -0x1.7bfa74p7, -0x1.67b1dap8, -0x1.7bfa74p7, -0x1.c7f958p6, -0x1.c7f958p6, -0x1.c7f958p6, -0x1.c7f958p6 };
float linsys_solver_L_x[57] = { 0x1.2f5502p-20, 0x1.21e908p-20, 0x1.7bc74cp-11, 0x1.21c02ap7, 0x1.9d6ebcp-14, -0x1.4f3572p-17, 0x1.7bc74cp-11, -0x1.2154dcp7, -0x1.da0a62p-7, -0x1.9d6ebcp-14, 0x1.7d2da8p-20, -0x1.cd8b7ap-7, -0x1.a53686p-13, 0x1.7bc74cp-11, -0x1.7bc74cp-11, -0x1.21c02ap7, -0x1.ad7f2ap-25, -0x1.21c02ap7, 0x1.7bc74cp-11, -0x1.7bc74cp-11, -0x1.ad7f2ap-25, -0x1.7d2da8p-21, -0x1.21e908p-20, -0x1.2f5502p-20, 0x1.21e908p-20, 0x1.2f5502p-20, 0x1.21c02ap7, 0x1.9d6ebcp-14, 0x1.21c02ap7, 0x1.7270ep-21, 0x1.c45c66p-8, -0x1.bef1eep-19, 0x1.a36372p-14, -0x1.a36372p-14, -0x1.47aep-7, -0x1.d94d9ap5, -0x1.83bab6p-8, -0x1.d3b9a6p-7, -0x1.d3b9a6p-7, -0x1.7d2da8p-21, 0x1.5798eep-26, -0x1.14ee24p-6, 0x1.a36372p-14, 0x1.898276p-14, -0x1.77cf44p-21, 0x1.5798eep-27, -0x1.aad704p-9, -0x1.aad704p-9, 0x1.ab25dep-17, 0x1.f71882p-3, -0x1.68517p2, -0x1.f16d3cp-2, -0x1.5798eep-27, -0x1.2bd0b4p-5, 0x1.ad7f2ap-24, 0x1.150c9ap-4, 0x1.dda882p2 };
signed long long Adata_i[43] = { 0ull, 3ull, 1ull, 4ull, 2ull, 4ull, 5ull, 3ull, 6ull, 4ull, 7ull, 5ull, 7ull, 8ull, 6ull, 7ull, 8ull, 9ull, 15ull, 16ull, 10ull, 15ull, 3ull, 11ull, 12ull, 15ull, 16ull, 17ull, 18ull, 4ull, 5ull, 11ull, 15ull, 17ull, 6ull, 13ull, 14ull, 17ull, 18ull, 7ull, 8ull, 13ull, 17ull };
l_struct_OC_csc Adata = { 43ull, 19ull, 15ull, ((&Adata_p[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 16
#endif
])), ((&Adata_i[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 43
#endif
])), ((&Adata_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 43
#endif
])), 18446744073709551615ull };
signed long long Pdata_p[16] = { 0ull, 1ull, 2ull, 2ull, 3ull, 4ull, 4ull, 5ull, 6ull, 7ull, 8ull, 8ull, 10ull, 10ull, 12ull, 12ull };
l_struct_OC_csc Pdata = { 12ull, 15ull, 15ull, ((&Pdata_p[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 16
#endif
])), ((&Pdata_i[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 12
#endif
])), ((&Pdata_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 12
#endif
])), 18446744073709551615ull };
l_struct_OC_OSQPData data = { 15ull, 19ull, (&Pdata), (&Adata), ((&qdata[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&ldata[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&udata[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])) };
float linsys_solver_rho_inv_vec[19] = { 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2 };
signed long long linsys_solver_PtoKKT[12] = { 3ull, 28ull, 9ull, 34ull, 5ull, 33ull, 35ull, 20ull, 56ull, 57ull, 64ull, 65ull };
signed long long linsys_solver_iwork[102] __ATTRIBUTE_WEAK__;
float xsolution[15] __ATTRIBUTE_WEAK__;
float work_xz_tilde[34] __ATTRIBUTE_WEAK__;
float linsys_solver_fwork[34] __ATTRIBUTE_WEAK__;
float work_x_prev[15] __ATTRIBUTE_WEAK__;
float work_z_prev[19] __ATTRIBUTE_WEAK__;
float work_Aty[15] __ATTRIBUTE_WEAK__;
l_struct_OC_OSQPInfo info = { 0ull, "Unsolved\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 18446744073709551606ull, 0x0p0, 0x0p0, 0x0p0, 0ull, 0x0p0 };
signed long long linsys_solver_KKT_p[35] = { 0ull, 1ull, 2ull, 3ull, 5ull, 6ull, 7ull, 9ull, 11ull, 14ull, 15ull, 16ull, 18ull, 19ull, 20ull, 23ull, 26ull, 27ull, 28ull, 30ull, 31ull, 33ull, 34ull, 35ull, 36ull, 38ull, 41ull, 44ull, 48ull, 51ull, 56ull, 64ull, 70ull, 74ull, 79ull };
float work_delta_x[15] __ATTRIBUTE_WEAK__;
l_struct_OC_csc linsys_solver_L = { 57ull, 34ull, 34ull, ((&linsys_solver_L_p[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 35
#endif
])), ((&linsys_solver_L_i[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 57
#endif
])), ((&linsys_solver_L_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 57
#endif
])), 18446744073709551615ull };
signed long long linsys_solver_AtoKKT[43] = { 4ull, 11ull, 29ull, 44ull, 32ull, 45ull, 48ull, 12ull, 10ull, 46ull, 38ull, 49ull, 42ull, 43ull, 7ull, 39ull, 36ull, 21ull, 23ull, 22ull, 17ull, 24ull, 58ull, 59ull, 60ull, 61ull, 62ull, 74ull, 63ull, 52ull, 53ull, 54ull, 55ull, 75ull, 66ull, 67ull, 68ull, 76ull, 69ull, 71ull, 72ull, 73ull, 77ull };
signed long long linsys_solver_rhotoKKT[19] = { 2ull, 27ull, 30ull, 13ull, 47ull, 50ull, 8ull, 40ull, 37ull, 18ull, 15ull, 26ull, 1ull, 0ull, 6ull, 25ull, 19ull, 78ull, 14ull };
float linsys_solver_Dinv[34] = { -0x1.28b462p-3, -0x1.28b462p-3, -0x1.21c02ap7, 0x1.9d2392p-14, 0x1.4c305ap-17, -0x1.28b462p-3, -0x1.2154dap7, 0x1.9d2392p-14, -0x1.19b468p7, -0x1.28b462p-3, -0x1.21c02ap7, 0x1.c45b0ep-8, -0x1.21c02ap7, -0x1.28b462p-3, 0x1.c45b0ep-8, -0x1.28b462p-3, -0x1.28b462p-3, -0x1.21c02ap7, 0x1.9d2392p-14, -0x1.21c02ap7, 0x1.c45b0ep-8, 0x1.baeb22p-19, 0x1.a302cep-14, 0x1.47ad3ep-7, -0x1.d94d9ap5, -0x1.1d7a18p7, 0x1.14edcep-6, -0x1.19b46cp7, -0x1.0485e4p5, 0x1.7053f2p19, 0x1.250cd2p12, 0x1.074c0ap14, 0x1.87fc1p19, -0x1.28b32p-3 };
signed long long linsys_solver_KKT_i[79] = { 0ull, 1ull, 2ull, 3ull, 2ull, 4ull, 5ull, 4ull, 6ull, 7ull, 6ull, 3ull, 7ull, 8ull, 9ull, 10ull, 11ull, 10ull, 12ull, 13ull, 14ull, 12ull, 13ull, 14ull, 11ull, 15ull, 16ull, 17ull, 18ull, 17ull, 19ull, 20ull, 19ull, 21ull, 22ull, 23ull, 23ull, 24ull, 22ull, 21ull, 25ull, 26ull, 25ull, 24ull, 18ull, 20ull, 22ull, 27ull, 20ull, 26ull, 28ull, 29ull, 27ull, 28ull, 16ull, 15ull, 14ull, 30ull, 8ull, 16ull, 1ull, 15ull, 13ull, 9ull, 30ull, 31ull, 6ull, 0ull, 5ull, 9ull, 32ull, 25ull, 24ull, 0ull, 30ull, 29ull, 31ull, 32ull, 33ull };
float linsys_solver_D[34] __ATTRIBUTE_WEAK__;
float work_z[19] __ATTRIBUTE_WEAK__;
float work_delta_y[19] __ATTRIBUTE_WEAK__;
float work_Adelta_x[19] __ATTRIBUTE_WEAK__;
float work_D_temp[15] __ATTRIBUTE_WEAK__;
float linsys_solver_sol[34] __ATTRIBUTE_WEAK__;
float linsys_solver_bp[34] __ATTRIBUTE_WEAK__;
signed long long linsys_solver_etree[34] __ATTRIBUTE_WEAK__;
float ysolution[19] __ATTRIBUTE_WEAK__;
l_struct_OC_OSQPSolution solution = { ((&xsolution[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&ysolution[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])) };
float work_D_temp_A[15] __ATTRIBUTE_WEAK__;
signed long long linsys_solver_bwork[34] __ATTRIBUTE_WEAK__;
float work_Pdelta_x[15] __ATTRIBUTE_WEAK__;
float work_Ax[19] __ATTRIBUTE_WEAK__;
float work_E_temp[19] __ATTRIBUTE_WEAK__;
l_struct_OC_OSQPScaling scaling __ATTRIBUTE_WEAK__;
float work_x[15] __ATTRIBUTE_WEAK__;
float scaling_D[15] __ATTRIBUTE_WEAK__;
float linsys_solver_KKT_x[79] = { -0x1.b9c258p2, -0x1.b9c258p2, -0x1.c45458p-8, 0x1.388p13, -0x1p0, 0x1.86ap16, -0x1.b9c258p2, -0x1p0, -0x1.c45458p-8, 0x1.388p13, 0x1p0, 0x1p0, -0x1p0, -0x1.c45458p-8, -0x1.b9c258p2, -0x1.c45458p-8, 0x1.0c6f7ap-20, 0x1p0, -0x1.c45458p-8, -0x1.b9c258p2, 0x1.a79fecp-14, 0x1p0, -0x1.47ae14p-8, -0x1.0c6f7ap-17, -0x1.d5c316p-18, -0x1.b9c258p2, -0x1.b9c258p2, -0x1.c45458p-8, 0x1.388p13, -0x1p0, -0x1.c45458p-8, 0x1.0c6f7ap-20, -0x1p0, 0x1.24f8p18, 0x1.388p13, 0x1.9p6, -0x1p0, -0x1.c45458p-8, 0x1p0, -0x1p0, -0x1.c45458p-8, 0x1.0c6f7ap-20, 0x1.a36e2ep-14, 0x1p0, 0x1p0, 0x1.a36e2ep-14, -0x1p0, -0x1.c45458p-8, 0x1p0, -0x1p0, -0x1.c45458p-8, 0x1.0c6f7ap-20, 0x0p0, 0x1.a36e2ep-14, -0x1.d5c316p-18, 0x1.d5c316p-18, -0x1.a36e2ep-14, 0x1.a5870ep-13, 0x1.a36e2ep-14, -0x1.0c6f7ap-17, -0x1.47ae14p-8, 0x1.0c6f7ap-17, 0x1.47ae14p-8, -0x1.47ae14p-8, -0x1.a36e2ep-14, 0x1.a79fecp-14, 0x1.a36e2ep-14, -0x1.0c6f7ap-17, -0x1.47ae14p-8, 0x1.47ae14p-8, 0x1.0c6f7ap-20, 0x0p0, 0x1.a36e2ep-14, -0x1.d5c316p-18, -0x1.0c6f7ap-17, -0x1.d5c316p-18, 0x1.0c6f7ap-17, 0x1.d5c316p-18, -0x1.b9c258p2 };
float work_rho_vec[19] = { 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3, 0x1.28b462p-3 };
float work_y[19] __ATTRIBUTE_WEAK__;
signed long long linsys_solver_Lnz[34] __ATTRIBUTE_WEAK__;
float work_rho_inv_vec[19] = { 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.c45b0ep-8, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2, 0x1.b9c258p2 };
float work_Px[15] __ATTRIBUTE_WEAK__;
float scaling_Dinv[15] __ATTRIBUTE_WEAK__;
signed long long linsys_solver_Pdiag_idx[10] = { 0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 9ull, 11ull };
float scaling_E[19] __ATTRIBUTE_WEAK__;
signed long long linsys_solver_P[34] = { 28ull, 27ull, 15ull, 0ull, 6ull, 29ull, 21ull, 3ull, 18ull, 33ull, 25ull, 10ull, 24ull, 31ull, 9ull, 30ull, 26ull, 16ull, 1ull, 17ull, 2ull, 7ull, 4ull, 8ull, 23ull, 22ull, 5ull, 19ull, 20ull, 12ull, 11ull, 13ull, 14ull, 32ull };
signed long long work_constr_type[19] = { 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull };
float work_Atdelta_y[15] __ATTRIBUTE_WEAK__;
l_struct_OC_csc linsys_solver_KKT = { 79ull, 34ull, 34ull, ((&linsys_solver_KKT_p[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 35
#endif
])), ((&linsys_solver_KKT_i[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 79
#endif
])), ((&linsys_solver_KKT_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 79
#endif
])), 18446744073709551615ull };
l_struct_OC_qdldl linsys_solver = { 0, (&linsys_solver_L), ((&linsys_solver_Dinv[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_P[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_bp[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_sol[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_rho_inv_vec[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x1.0c6f7ap-20, 15ull, 19ull, ((&linsys_solver_Pdiag_idx[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 10
#endif
])), 10ull, (&linsys_solver_KKT), ((&linsys_solver_PtoKKT[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 12
#endif
])), ((&linsys_solver_AtoKKT[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 43
#endif
])), ((&linsys_solver_rhotoKKT[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&linsys_solver_D[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_etree[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_Lnz[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_iwork[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 102
#endif
])), ((&linsys_solver_bwork[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&linsys_solver_fwork[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])) };
l_struct_OC_OSQPWorkspace workspace = { (&data), ((l_struct_OC_linsys_solver *)(&linsys_solver)), ((&work_rho_vec[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_rho_inv_vec[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_constr_type[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_y[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_z[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_xz_tilde[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 34
#endif
])), ((&work_x_prev[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_z_prev[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_Ax[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_Px[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_Aty[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_delta_y[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_Atdelta_y[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_delta_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_Pdelta_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_Adelta_x[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), ((&work_D_temp[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_D_temp_A[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 15
#endif
])), ((&work_E_temp[(((signed int )0u))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (&settings), (&scaling), (&solution), (&info) };
float scaling_Einv[19] __ATTRIBUTE_WEAK__;


/* Function Bodies */
static inline int llvm_fcmp_ord(double X, double Y) { return X == X && Y == Y; }
static inline int llvm_fcmp_uno(double X, double Y) { return X != X || Y != Y; }
static inline int llvm_fcmp_ueq(double X, double Y) { return X == Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_une(double X, double Y) { return X != Y; }
static inline int llvm_fcmp_ult(double X, double Y) { return X <  Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_ugt(double X, double Y) { return X >  Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_ule(double X, double Y) { return X <= Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_uge(double X, double Y) { return X >= Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_oeq(double X, double Y) { return X == Y ; }
static inline int llvm_fcmp_one(double X, double Y) { return X != Y && llvm_fcmp_ord(X, Y); }
static inline int llvm_fcmp_olt(double X, double Y) { return X <  Y ; }
static inline int llvm_fcmp_ogt(double X, double Y) { return X >  Y ; }
static inline int llvm_fcmp_ole(double X, double Y) { return X <= Y ; }
static inline int llvm_fcmp_oge(double X, double Y) { return X >= Y ; }
