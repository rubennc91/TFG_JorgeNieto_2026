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
typedef struct l_struct_OC_OSQPSettings l_struct_OC_OSQPSettings;
typedef struct l_struct_OC_OSQPData l_struct_OC_OSQPData;
typedef struct l_struct_OC_csc l_struct_OC_csc;
typedef struct l_struct_OC_OSQPInfo l_struct_OC_OSQPInfo;
typedef struct l_struct_OC_OSQPScaling l_struct_OC_OSQPScaling;

/* Structure contents */
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

struct l_struct_OC_OSQPData {
  unsigned long long field0;
  unsigned long long field1;
  l_struct_OC_csc *field2;
  l_struct_OC_csc *field3;
  float *field4;
  float *field5;
  float *field6;
};

struct l_struct_OC_csc {
  unsigned long long field0;
  unsigned long long field1;
  unsigned long long field2;
  signed long long *field3;
  signed long long *field4;
  float *field5;
  unsigned long long field6;
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


/* External Global Variable Declarations */
extern l_struct_OC_OSQPSettings settings;
extern float work_x[15];
extern float work_x_prev[15];
extern l_struct_OC_OSQPData data;
extern float work_z[19];
extern float work_z_prev[19];
extern l_struct_OC_OSQPInfo info;
extern float qdata[15];
extern float scaling_D[15];
extern l_struct_OC_OSQPScaling scaling;
extern float ldata[19];
extern float udata[19];
extern float scaling_E[19];
extern signed long long Pdata_p[16];
extern float Pdata_x[12];
extern signed long long Adata_p[16];
extern float Adata_x[43];
extern signed long long work_constr_type[19];
extern float work_rho_vec[19];
extern float work_rho_inv_vec[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void osqp_set_default_settings(l_struct_OC_OSQPSettings *llvm_cbe_sets);
signed long long osqp_solve(void);
void cold_start(void);
void prea_vec_copy(float *, float *, signed long long );
void update_xz_tilde(void);
void update_x(void);
void update_z(void);
void update_y(void);
void update_info(signed long long , signed long long , signed long long );
signed long long check_termination(signed long long );
void update_status(l_struct_OC_OSQPInfo *, signed long long );
void store_solution(void);
signed long long osqp_update_lin_cost(float *llvm_cbe_q_new);
void vec_ew_prod(float *, float *, float *, signed long long );
void vec_mult_scalar(float *, float , signed long long );
void reset_info(l_struct_OC_OSQPInfo *);
signed long long osqp_update_bounds(float *llvm_cbe_l_new, float *llvm_cbe_u_new);
signed long long osqp_update_rho(float llvm_cbe_rho_new);
signed long long osqp_update_P(float *llvm_cbe_Px_new, signed long long *llvm_cbe_Px_new_idx, signed long long llvm_cbe_P_new_n);
signed long long unscale_data(void);
signed long long scale_data(void);
signed long long update_linsys_solver_matrices_qdldl(void);
signed long long osqp_update_A(float *llvm_cbe_Ax_new, signed long long *llvm_cbe_Ax_new_idx, signed long long llvm_cbe_A_new_n);
signed long long update_linsys_solver_rho_vec_qdldl(float *);


/* Global Variable Definitions and Initialization */


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

void osqp_set_default_settings(l_struct_OC_OSQPSettings *llvm_cbe_sets) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
  static  unsigned long long aesl_llvm_cbe_4_count = 0;
  static  unsigned long long aesl_llvm_cbe_5_count = 0;
  static  unsigned long long aesl_llvm_cbe_6_count = 0;
  static  unsigned long long aesl_llvm_cbe_7_count = 0;
  static  unsigned long long aesl_llvm_cbe_8_count = 0;
  static  unsigned long long aesl_llvm_cbe_9_count = 0;
  static  unsigned long long aesl_llvm_cbe_10_count = 0;
  static  unsigned long long aesl_llvm_cbe_11_count = 0;
  static  unsigned long long aesl_llvm_cbe_12_count = 0;
  static  unsigned long long aesl_llvm_cbe_13_count = 0;
  static  unsigned long long aesl_llvm_cbe_14_count = 0;
  static  unsigned long long aesl_llvm_cbe_15_count = 0;
  float *llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_16_count = 0;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  float *llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  signed long long *llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  signed long long *llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  float *llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  float *llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  float *llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  float *llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  bool *llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;
  signed long long *llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_36_count = 0;
  static  unsigned long long aesl_llvm_cbe_37_count = 0;
  signed long long *llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_38_count = 0;
  static  unsigned long long aesl_llvm_cbe_39_count = 0;
  signed long long *llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;
  static  unsigned long long aesl_llvm_cbe_41_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_set_default_settings\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 0, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_15_count);
  llvm_cbe_tmp__1 = (float *)(&llvm_cbe_sets->field0);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3FB99999A0000000, float* %%1, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_16_count);
  *llvm_cbe_tmp__1 = 0x1.99999ap-4;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.99999ap-4);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 1, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_17_count);
  llvm_cbe_tmp__2 = (float *)(&llvm_cbe_sets->field1);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3EB0C6F7A0000000, float* %%2, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_18_count);
  *llvm_cbe_tmp__2 = 0x1.0c6f7ap-20;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.0c6f7ap-20);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 2, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__3 = (signed long long *)(&llvm_cbe_sets->field2);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 10, i64* %%3, align 8, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_20_count);
  *llvm_cbe_tmp__3 = 10ull;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", 10ull);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 6, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_21_count);
  llvm_cbe_tmp__4 = (signed long long *)(&llvm_cbe_sets->field6);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 4000, i64* %%4, align 8, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_22_count);
  *llvm_cbe_tmp__4 = 4000ull;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", 4000ull);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 7, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__5 = (float *)(&llvm_cbe_sets->field7);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3F50624DE0000000, float* %%5, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_24_count);
  *llvm_cbe_tmp__5 = 0x1.0624dep-10;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.0624dep-10);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 8, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_25_count);
  llvm_cbe_tmp__6 = (float *)(&llvm_cbe_sets->field8);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3F50624DE0000000, float* %%6, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_26_count);
  *llvm_cbe_tmp__6 = 0x1.0624dep-10;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.0624dep-10);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 9, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_27_count);
  llvm_cbe_tmp__7 = (float *)(&llvm_cbe_sets->field9);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3F1A36E2E0000000, float* %%7, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_28_count);
  *llvm_cbe_tmp__7 = 0x1.a36e2ep-14;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.a36e2ep-14);
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 10, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_29_count);
  llvm_cbe_tmp__8 = (float *)(&llvm_cbe_sets->field10);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3F1A36E2E0000000, float* %%8, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_30_count);
  *llvm_cbe_tmp__8 = 0x1.a36e2ep-14;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.a36e2ep-14);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 11, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_31_count);
  llvm_cbe_tmp__9 = (float *)(&llvm_cbe_sets->field11);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x3FF99999A0000000, float* %%9, align 4, !dbg !30 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_32_count);
  *llvm_cbe_tmp__9 = 0x1.99999ap0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.99999ap0);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 12, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_33_count);
  llvm_cbe_tmp__10 = (bool *)(&llvm_cbe_sets->field12);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i1 false, i1* %%10, align 1, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_34_count);
  *llvm_cbe_tmp__10 = ((0) & 1ull);
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", 0);
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 13, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_35_count);
  llvm_cbe_tmp__11 = (signed long long *)(&llvm_cbe_sets->field13);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 0, i64* %%11, align 8, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_36_count);
  *llvm_cbe_tmp__11 = 0ull;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", 0ull);
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 14, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_37_count);
  llvm_cbe_tmp__12 = (signed long long *)(&llvm_cbe_sets->field14);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 25, i64* %%12, align 8, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_38_count);
  *llvm_cbe_tmp__12 = 25ull;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", 25ull);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds %%struct.OSQPSettings* %%sets, i64 0, i32 15, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_39_count);
  llvm_cbe_tmp__13 = (signed long long *)(&llvm_cbe_sets->field15);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 1, i64* %%13, align 8, !dbg !31 for 0x%I64xth hint within @osqp_set_default_settings  --> \n", ++aesl_llvm_cbe_40_count);
  *llvm_cbe_tmp__13 = 1ull;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", 1ull);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_set_default_settings}\n");
  return;
}


signed long long osqp_solve(void) {
  static  unsigned long long aesl_llvm_cbe_42_count = 0;
  static  unsigned long long aesl_llvm_cbe_43_count = 0;
  static  unsigned long long aesl_llvm_cbe_44_count = 0;
  unsigned long long llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_45_count = 0;
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  unsigned long long llvm_cbe_storemerge;
  unsigned long long llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_51_count = 0;
  static  unsigned long long aesl_llvm_cbe_52_count = 0;
  static  unsigned long long aesl_llvm_cbe_53_count = 0;
  static  unsigned long long aesl_llvm_cbe_54_count = 0;
  static  unsigned long long aesl_llvm_cbe_55_count = 0;
  unsigned long long llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_56_count = 0;
  static  unsigned long long aesl_llvm_cbe_57_count = 0;
  static  unsigned long long aesl_llvm_cbe_58_count = 0;
  unsigned long long llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_59_count = 0;
  static  unsigned long long aesl_llvm_cbe_60_count = 0;
  unsigned long long llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_61_count = 0;
  static  unsigned long long aesl_llvm_cbe_62_count = 0;
  static  unsigned long long aesl_llvm_cbe_63_count = 0;
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  unsigned long long llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  unsigned long long llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  unsigned long long llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  unsigned long long llvm_cbe_tmp__21;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  unsigned long long llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  unsigned long long llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_solve\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 15), align 8, !dbg !30 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_44_count);
  llvm_cbe_tmp__14 = (unsigned long long )*((&settings.field15));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__14);
  if (((llvm_cbe_tmp__14&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__24;
  } else {
    goto llvm_cbe_tmp__25;
  }

llvm_cbe_tmp__24:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @cold_start() nounwind, !dbg !30 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_47_count);
   /*tail*/ cold_start();
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__25;

llvm_cbe_tmp__25:
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
  goto llvm_cbe_tmp__26;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__26:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi i64 [ 1, %%4 ], [ %%20, %%19  for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (unsigned long long )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = 0x%I64X",llvm_cbe_storemerge);
printf("\n = 0x%I64X",1ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__21);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 6), align 8, !dbg !30 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_55_count);
  llvm_cbe_tmp__15 = (unsigned long long )*((&settings.field6));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__15);
  if ((((signed long long )llvm_cbe_storemerge) > ((signed long long )llvm_cbe_tmp__15))) {
    goto llvm_cbe_tmp__27;
  } else {
    goto llvm_cbe_tmp__28;
  }

llvm_cbe_tmp__29:
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = add nsw i64 %%storemerge, 1, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_76_count);
  llvm_cbe_tmp__21 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__21&18446744073709551615ull)));
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__21;   /* for PHI node */
  goto llvm_cbe_tmp__26;

llvm_cbe_tmp__28:
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_58_count);
  llvm_cbe_tmp__16 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), i64 %%9) nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_59_count);
   /*tail*/ prea_vec_copy((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__16);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_60_count);
  llvm_cbe_tmp__17 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__17);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_z_prev, i64 0, i64 0), i64 %%10) nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_61_count);
   /*tail*/ prea_vec_copy((float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_z_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__17);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__17);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_xz_tilde() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_62_count);
   /*tail*/ update_xz_tilde();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_x() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_63_count);
   /*tail*/ update_x();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_z() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_64_count);
   /*tail*/ update_z();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_y() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_65_count);
   /*tail*/ update_y();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 14), align 8, !dbg !30 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_66_count);
  llvm_cbe_tmp__18 = (unsigned long long )*((&settings.field14));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__18);
  if (((llvm_cbe_tmp__18&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__29;
  } else {
    goto llvm_cbe_tmp__30;
  }

llvm_cbe_tmp__30:
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = srem i64 %%storemerge, %%11, !dbg !30 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_69_count);
  llvm_cbe_tmp__19 = (unsigned long long )((signed long long )(((signed long long )llvm_cbe_storemerge) % ((signed long long )llvm_cbe_tmp__18)));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((signed long long )llvm_cbe_tmp__19));
  if (((llvm_cbe_tmp__19&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__31;
  } else {
    goto llvm_cbe_tmp__29;
  }

llvm_cbe_tmp__31:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_info(i64 %%storemerge, i64 0, i64 0) nounwind, !dbg !31 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_72_count);
   /*tail*/ update_info(llvm_cbe_storemerge, 0ull, 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument storemerge = 0x%I64X",llvm_cbe_storemerge);
printf("\nArgument  = 0x%I64X",0ull);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = tail call i64 @check_termination(i64 0) nounwind, !dbg !32 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_73_count);
  llvm_cbe_tmp__20 = (unsigned long long ) /*tail*/ check_termination(0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",0ull);
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__20);
}
  if (((llvm_cbe_tmp__20&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__29;
  } else {
    goto llvm_cbe_tmp__27;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe_tmp__27:
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = load i64* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 2), align 8, !dbg !32 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_79_count);
  llvm_cbe_tmp__22 = (unsigned long long )*((&info.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__22);
  if (((llvm_cbe_tmp__22&18446744073709551615ULL) == (18446744073709551606ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__32;
  } else {
    goto llvm_cbe_tmp__33;
  }

llvm_cbe_tmp__32:
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = tail call i64 @check_termination(i64 1) nounwind, !dbg !32 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_82_count);
  llvm_cbe_tmp__23 = (unsigned long long ) /*tail*/ check_termination(1ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",1ull);
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__23);
}
  if (((llvm_cbe_tmp__23&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__34;
  } else {
    goto llvm_cbe_tmp__35;
  }

llvm_cbe_tmp__34:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 -2) nounwind, !dbg !32 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_85_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 18446744073709551614ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",18446744073709551614ull);
}
  goto llvm_cbe_tmp__35;

llvm_cbe_tmp__35:
  goto llvm_cbe_tmp__33;

llvm_cbe_tmp__33:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @store_solution() nounwind, !dbg !32 for 0x%I64xth hint within @osqp_solve  --> \n", ++aesl_llvm_cbe_88_count);
   /*tail*/ store_solution();
if (AESL_DEBUG_TRACE) {
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_solve}\n");
  return 0ull;
}


signed long long osqp_update_lin_cost(float *llvm_cbe_q_new) {
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;
  static  unsigned long long aesl_llvm_cbe_92_count = 0;
  unsigned long long llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_93_count = 0;
  static  unsigned long long aesl_llvm_cbe_94_count = 0;
  unsigned long long llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_95_count = 0;
  static  unsigned long long aesl_llvm_cbe_96_count = 0;
  static  unsigned long long aesl_llvm_cbe_97_count = 0;
  unsigned long long llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_98_count = 0;
  static  unsigned long long aesl_llvm_cbe_99_count = 0;
  float llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_100_count = 0;
  unsigned long long llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_101_count = 0;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_update_lin_cost\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_92_count);
  llvm_cbe_tmp__36 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__36);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* %%q_new, float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), i64 %%1) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_93_count);
   /*tail*/ prea_vec_copy((float *)llvm_cbe_q_new, (float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__36);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__36);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_94_count);
  llvm_cbe_tmp__37 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__37);
  if (((llvm_cbe_tmp__37&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__41;
  } else {
    goto llvm_cbe_tmp__42;
  }

llvm_cbe_tmp__42:
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_97_count);
  llvm_cbe_tmp__38 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__38);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([15 x float]* @scaling_D, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), i64 %%5) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_98_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__38);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__38);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 0), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_99_count);
  llvm_cbe_tmp__39 = (float )*((&scaling.field0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__39, *(int*)(&llvm_cbe_tmp__39));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_100_count);
  llvm_cbe_tmp__40 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__40);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_mult_scalar(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float %%6, i64 %%7) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_101_count);
   /*tail*/ vec_mult_scalar((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__39, llvm_cbe_tmp__40);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__39, *(int*)(&llvm_cbe_tmp__39));
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__40);
}
  goto llvm_cbe_tmp__41;

llvm_cbe_tmp__41:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @reset_info(%%struct.OSQPInfo* @info) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_lin_cost  --> \n", ++aesl_llvm_cbe_103_count);
   /*tail*/ reset_info((l_struct_OC_OSQPInfo *)(&info));
if (AESL_DEBUG_TRACE) {
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_update_lin_cost}\n");
  return 0ull;
}


signed long long osqp_update_bounds(float *llvm_cbe_l_new, float *llvm_cbe_u_new) {
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  unsigned long long llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  unsigned long long llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  unsigned long long llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  static  unsigned long long aesl_llvm_cbe_115_count = 0;
  static  unsigned long long aesl_llvm_cbe_116_count = 0;
  unsigned long long llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_117_count = 0;
  static  unsigned long long aesl_llvm_cbe_118_count = 0;
  unsigned long long llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_119_count = 0;
  static  unsigned long long aesl_llvm_cbe_120_count = 0;
  static  unsigned long long aesl_llvm_cbe_121_count = 0;
  static  unsigned long long aesl_llvm_cbe_122_count = 0;
  float llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_123_count = 0;
  unsigned long long llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_124_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_update_bounds\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_109_count);
  llvm_cbe_tmp__43 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__43);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* %%l_new, float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), i64 %%1) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_110_count);
   /*tail*/ prea_vec_copy((float *)llvm_cbe_l_new, (float *)((&ldata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__43);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__43);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_111_count);
  llvm_cbe_tmp__44 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__44);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* %%u_new, float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), i64 %%2) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_112_count);
   /*tail*/ prea_vec_copy((float *)llvm_cbe_u_new, (float *)((&udata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__44);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__44);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_113_count);
  llvm_cbe_tmp__45 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__45);
  if (((llvm_cbe_tmp__45&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__50;
  } else {
    goto llvm_cbe_tmp__51;
  }

llvm_cbe_tmp__51:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_116_count);
  llvm_cbe_tmp__46 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__46);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_E, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), i64 %%6) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_117_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_E[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&ldata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&ldata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__46);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__46);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_118_count);
  llvm_cbe_tmp__47 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__47);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_E, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), i64 %%7) nounwind, !dbg !30 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_119_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_E[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&udata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&udata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__47);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__47);
}
  goto llvm_cbe_tmp__50;

llvm_cbe_tmp__50:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @reset_info(%%struct.OSQPInfo* @info) nounwind, !dbg !31 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_121_count);
   /*tail*/ reset_info((l_struct_OC_OSQPInfo *)(&info));
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 0), align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_122_count);
  llvm_cbe_tmp__48 = (float )*((&settings.field0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__48, *(int*)(&llvm_cbe_tmp__48));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = tail call i64 @osqp_update_rho(float %%9), !dbg !31 for 0x%I64xth hint within @osqp_update_bounds  --> \n", ++aesl_llvm_cbe_123_count);
  llvm_cbe_tmp__49 = (unsigned long long ) /*tail*/ osqp_update_rho(llvm_cbe_tmp__48);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__48, *(int*)(&llvm_cbe_tmp__48));
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__49);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_update_bounds}\n");
  return llvm_cbe_tmp__49;
}


signed long long osqp_update_rho(float llvm_cbe_rho_new) {
  static  unsigned long long aesl_llvm_cbe_125_count = 0;
  static  unsigned long long aesl_llvm_cbe_126_count = 0;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;
  static  unsigned long long aesl_llvm_cbe_128_count = 0;
  static  unsigned long long aesl_llvm_cbe_129_count = 0;
  static  unsigned long long aesl_llvm_cbe_130_count = 0;
  static  unsigned long long aesl_llvm_cbe_131_count = 0;
  static  unsigned long long aesl_llvm_cbe_132_count = 0;
  static  unsigned long long aesl_llvm_cbe_133_count = 0;
  double llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_134_count = 0;
  static  unsigned long long aesl_llvm_cbe_phitmp_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_135_count = 0;
  static  unsigned long long aesl_llvm_cbe_136_count = 0;
  static  unsigned long long aesl_llvm_cbe_137_count = 0;
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  double llvm_cbe_tmp__53;
  double llvm_cbe_tmp__53__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_141_count = 0;
  static  unsigned long long aesl_llvm_cbe_142_count = 0;
  static  unsigned long long aesl_llvm_cbe_143_count = 0;
  static  unsigned long long aesl_llvm_cbe_144_count = 0;
  static  unsigned long long aesl_llvm_cbe_145_count = 0;
  static  unsigned long long aesl_llvm_cbe_146_count = 0;
  static  unsigned long long aesl_llvm_cbe_147_count = 0;
  static  unsigned long long aesl_llvm_cbe_148_count = 0;
  static  unsigned long long aesl_llvm_cbe_149_count = 0;
  static  unsigned long long aesl_llvm_cbe_150_count = 0;
  static  unsigned long long aesl_llvm_cbe_151_count = 0;
  static  unsigned long long aesl_llvm_cbe_152_count = 0;
  unsigned long long llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  float llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  float llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge2_count = 0;
  unsigned long long llvm_cbe_storemerge2;
  unsigned long long llvm_cbe_storemerge2__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  signed long long *llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  unsigned long long llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  float *llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  float *llvm_cbe_tmp__62;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  float *llvm_cbe_tmp__63;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  float *llvm_cbe_tmp__64;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  static  unsigned long long aesl_llvm_cbe_175_count = 0;
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
  unsigned long long llvm_cbe_tmp__65;
  static  unsigned long long aesl_llvm_cbe_177_count = 0;
  static  unsigned long long aesl_llvm_cbe_178_count = 0;
  static  unsigned long long aesl_llvm_cbe_179_count = 0;
  static  unsigned long long aesl_llvm_cbe_180_count = 0;
  static  unsigned long long aesl_llvm_cbe_181_count = 0;
  static  unsigned long long aesl_llvm_cbe_182_count = 0;
  static  unsigned long long aesl_llvm_cbe_183_count = 0;
  static  unsigned long long aesl_llvm_cbe_184_count = 0;
  static  unsigned long long aesl_llvm_cbe_185_count = 0;
  static  unsigned long long aesl_llvm_cbe_186_count = 0;
  static  unsigned long long aesl_llvm_cbe_187_count = 0;
  static  unsigned long long aesl_llvm_cbe_188_count = 0;
  static  unsigned long long aesl_llvm_cbe_189_count = 0;
  unsigned long long llvm_cbe_tmp__66;
  static  unsigned long long aesl_llvm_cbe_190_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_191_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_update_rho\n");
  if ((llvm_fcmp_ugt(llvm_cbe_rho_new, 0x0p0))) {
    goto llvm_cbe_tmp__67;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
    goto llvm_cbe_tmp__68;
  }

llvm_cbe_tmp__67:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = fpext float %%rho_new to double, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_133_count);
  llvm_cbe_tmp__52 = (double )((double )llvm_cbe_rho_new);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__52, *(long long*)(&llvm_cbe_tmp__52));
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = or i1 %%4, %%phitmp, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )(((llvm_fcmp_ule(llvm_cbe_tmp__52, 0x1.0c6f7a0b5ed8dp-20)) | (llvm_fcmp_olt(llvm_cbe_rho_new, 0x1.e848p19)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe__2e_critedge;
  } else {
    llvm_cbe_tmp__53__PHI_TEMPORARY = (double )0x1.e848p19;   /* for PHI node */
    goto llvm_cbe_tmp__69;
  }

llvm_cbe__2e_critedge:
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__52, 0x1.0c6f7a0b5ed8dp-20))) {
    goto llvm_cbe_tmp__70;
  } else {
    llvm_cbe_tmp__53__PHI_TEMPORARY = (double )0x1.0c6f7a0b5ed8dp-20;   /* for PHI node */
    goto llvm_cbe_tmp__69;
  }

llvm_cbe_tmp__70:
  llvm_cbe_tmp__53__PHI_TEMPORARY = (double )llvm_cbe_tmp__52;   /* for PHI node */
  goto llvm_cbe_tmp__69;

llvm_cbe_tmp__69:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = phi double [ %%3, %%6 ], [ 1.000000e-06, %%.critedge ], [ 1.000000e+06, %%2 ], !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_139_count);
  llvm_cbe_tmp__53 = (double )llvm_cbe_tmp__53__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %lf",llvm_cbe_tmp__53);
printf("\n = %lf",llvm_cbe_tmp__52);
printf("\n = %lf",0x1.0c6f7a0b5ed8dp-20);
printf("\n = %lf",0x1.e848p19);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = fptrunc double %%8 to float, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_140_count);
  llvm_cbe_tmp__54 = (float )((float )llvm_cbe_tmp__53);
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 0), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_141_count);
  *((&settings.field0)) = llvm_cbe_tmp__54;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__54);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_152_count);
  llvm_cbe_tmp__55 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__55);
  if ((((signed long long )llvm_cbe_tmp__55) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = fdiv float 1.000000e+00, %%9, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_155_count);
  llvm_cbe_tmp__56 = (float )((float )(0x1p0 / llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__56, *(int*)(&llvm_cbe_tmp__56));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%9, 1.000000e+03, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_156_count);
  llvm_cbe_tmp__57 = (float )((float )(llvm_cbe_tmp__54 * 0x1.f4p9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__57, *(int*)(&llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fdiv float 1.000000e+00, %%13, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_157_count);
  llvm_cbe_tmp__58 = (float )((float )(0x1p0 / llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
  llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__71;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__71:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge2 = phi i64 [ 0, %%.lr.ph ], [ %%29, %%28  for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_storemerge2_count);
  llvm_cbe_storemerge2 = (unsigned long long )llvm_cbe_storemerge2__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",llvm_cbe_storemerge2);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__65);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [19 x i64]* @work_constr_type, i64 0, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_159_count);
  llvm_cbe_tmp__59 = (signed long long *)(&work_constr_type[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge2) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_constr_type' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load i64* %%16, align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_160_count);
  llvm_cbe_tmp__60 = (unsigned long long )*llvm_cbe_tmp__59;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__60);
  if (((llvm_cbe_tmp__60&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__72;
  } else {
    goto llvm_cbe_tmp__73;
  }

llvm_cbe_tmp__74:
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = add nsw i64 %%storemerge2, 1, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_176_count);
  llvm_cbe_tmp__65 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge2&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__65&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__65) < ((signed long long )llvm_cbe_tmp__55))) {
    llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__65;   /* for PHI node */
    goto llvm_cbe_tmp__71;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

llvm_cbe_tmp__72:
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds [19 x float]* @work_rho_vec, i64 0, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_163_count);
  llvm_cbe_tmp__61 = (float *)(&work_rho_vec[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 19 && "Write access out of array 'work_rho_vec' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%20, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_164_count);
  *llvm_cbe_tmp__61 = llvm_cbe_tmp__54;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__54);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds [19 x float]* @work_rho_inv_vec, i64 0, i64 %%storemerge2, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_165_count);
  llvm_cbe_tmp__62 = (float *)(&work_rho_inv_vec[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 19 && "Write access out of array 'work_rho_inv_vec' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%12, float* %%21, align 4, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_166_count);
  *llvm_cbe_tmp__62 = llvm_cbe_tmp__56;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__56);
  goto llvm_cbe_tmp__74;

llvm_cbe_tmp__75:
  goto llvm_cbe_tmp__74;

llvm_cbe_tmp__73:
  if (((llvm_cbe_tmp__60&18446744073709551615ULL) == (1ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__76;
  } else {
    goto llvm_cbe_tmp__75;
  }

llvm_cbe_tmp__76:
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = getelementptr inbounds [19 x float]* @work_rho_vec, i64 0, i64 %%storemerge2, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_170_count);
  llvm_cbe_tmp__63 = (float *)(&work_rho_vec[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 19 && "Write access out of array 'work_rho_vec' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%25, align 4, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_171_count);
  *llvm_cbe_tmp__63 = llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__57);
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = getelementptr inbounds [19 x float]* @work_rho_inv_vec, i64 0, i64 %%storemerge2, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_172_count);
  llvm_cbe_tmp__64 = (float *)(&work_rho_inv_vec[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 19 && "Write access out of array 'work_rho_inv_vec' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%26, align 4, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_173_count);
  *llvm_cbe_tmp__64 = llvm_cbe_tmp__58;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__58);
  goto llvm_cbe_tmp__75;

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = tail call i64 @update_linsys_solver_rho_vec_qdldl(float* getelementptr inbounds ([19 x float]* @work_rho_vec, i64 0, i64 0)) nounwind, !dbg !31 for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_189_count);
  llvm_cbe_tmp__66 = (unsigned long long ) /*tail*/ update_linsys_solver_rho_vec_qdldl((float *)((&work_rho_vec[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__66);
}
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__66;   /* for PHI node */
  goto llvm_cbe_tmp__68;

llvm_cbe_tmp__68:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%31, %%._crit_edge ], [ 1, %%0  for 0x%I64xth hint within @osqp_update_rho  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__66);
printf("\n = 0x%I64X",1ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_update_rho}\n");
  return llvm_cbe_storemerge1;
}


signed long long osqp_update_P(float *llvm_cbe_Px_new, signed long long *llvm_cbe_Px_new_idx, signed long long llvm_cbe_P_new_n) {
  static  unsigned long long aesl_llvm_cbe_192_count = 0;
  static  unsigned long long aesl_llvm_cbe_193_count = 0;
  static  unsigned long long aesl_llvm_cbe_194_count = 0;
  static  unsigned long long aesl_llvm_cbe_195_count = 0;
  static  unsigned long long aesl_llvm_cbe_196_count = 0;
  static  unsigned long long aesl_llvm_cbe_197_count = 0;
  static  unsigned long long aesl_llvm_cbe_198_count = 0;
  static  unsigned long long aesl_llvm_cbe_199_count = 0;
  static  unsigned long long aesl_llvm_cbe_200_count = 0;
  unsigned long long llvm_cbe_tmp__77;
  static  unsigned long long aesl_llvm_cbe_201_count = 0;
  signed long long *llvm_cbe_tmp__78;
  static  unsigned long long aesl_llvm_cbe_202_count = 0;
  unsigned long long llvm_cbe_tmp__79;
  static  unsigned long long aesl_llvm_cbe_203_count = 0;
  static  unsigned long long aesl_llvm_cbe_204_count = 0;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  unsigned long long llvm_cbe_tmp__80;
  static  unsigned long long aesl_llvm_cbe_206_count = 0;
  static  unsigned long long aesl_llvm_cbe_207_count = 0;
  static  unsigned long long aesl_llvm_cbe_208_count = 0;
  unsigned long long llvm_cbe_tmp__81;
  static  unsigned long long aesl_llvm_cbe_209_count = 0;
  static  unsigned long long aesl_llvm_cbe_210_count = 0;
  static  unsigned long long aesl_llvm_cbe_211_count = 0;
  static  unsigned long long aesl_llvm_cbe_212_count = 0;
  static  unsigned long long aesl_llvm_cbe_213_count = 0;
  static  unsigned long long aesl_llvm_cbe_214_count = 0;
  static  unsigned long long aesl_llvm_cbe_215_count = 0;
  static  unsigned long long aesl_llvm_cbe_216_count = 0;
  static  unsigned long long aesl_llvm_cbe_217_count = 0;
  static  unsigned long long aesl_llvm_cbe_218_count = 0;
  static  unsigned long long aesl_llvm_cbe_219_count = 0;
  static  unsigned long long aesl_llvm_cbe_220_count = 0;
  static  unsigned long long aesl_llvm_cbe_221_count = 0;
  static  unsigned long long aesl_llvm_cbe_222_count = 0;
  static  unsigned long long aesl_llvm_cbe_223_count = 0;
  static  unsigned long long aesl_llvm_cbe_224_count = 0;
  static  unsigned long long aesl_llvm_cbe_225_count = 0;
  static  unsigned long long aesl_llvm_cbe_226_count = 0;
  static  unsigned long long aesl_llvm_cbe_227_count = 0;
  static  unsigned long long aesl_llvm_cbe_228_count = 0;
  static  unsigned long long aesl_llvm_cbe_229_count = 0;
  static  unsigned long long aesl_llvm_cbe_230_count = 0;
  static  unsigned long long aesl_llvm_cbe_231_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge15_count = 0;
  unsigned long long llvm_cbe_storemerge15;
  unsigned long long llvm_cbe_storemerge15__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  float *llvm_cbe_tmp__82;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  float llvm_cbe_tmp__83;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  signed long long *llvm_cbe_tmp__84;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  unsigned long long llvm_cbe_tmp__85;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  float *llvm_cbe_tmp__86;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  unsigned long long llvm_cbe_tmp__87;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
  static  unsigned long long aesl_llvm_cbe_244_count = 0;
  static  unsigned long long aesl_llvm_cbe_245_count = 0;
  static  unsigned long long aesl_llvm_cbe_246_count = 0;
  static  unsigned long long aesl_llvm_cbe_247_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond7_count = 0;
  static  unsigned long long aesl_llvm_cbe_248_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge2_count = 0;
  unsigned long long llvm_cbe_storemerge2;
  unsigned long long llvm_cbe_storemerge2__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_249_count = 0;
  float *llvm_cbe_tmp__88;
  static  unsigned long long aesl_llvm_cbe_250_count = 0;
  float llvm_cbe_tmp__89;
  static  unsigned long long aesl_llvm_cbe_251_count = 0;
  float *llvm_cbe_tmp__90;
  static  unsigned long long aesl_llvm_cbe_252_count = 0;
  static  unsigned long long aesl_llvm_cbe_253_count = 0;
  unsigned long long llvm_cbe_tmp__91;
  static  unsigned long long aesl_llvm_cbe_254_count = 0;
  static  unsigned long long aesl_llvm_cbe_255_count = 0;
  static  unsigned long long aesl_llvm_cbe_256_count = 0;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
  static  unsigned long long aesl_llvm_cbe_258_count = 0;
  static  unsigned long long aesl_llvm_cbe_259_count = 0;
  static  unsigned long long aesl_llvm_cbe_260_count = 0;
  static  unsigned long long aesl_llvm_cbe_261_count = 0;
  static  unsigned long long aesl_llvm_cbe_262_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_263_count = 0;
  static  unsigned long long aesl_llvm_cbe_264_count = 0;
  static  unsigned long long aesl_llvm_cbe_265_count = 0;
  static  unsigned long long aesl_llvm_cbe_266_count = 0;
  unsigned long long llvm_cbe_tmp__92;
  static  unsigned long long aesl_llvm_cbe_267_count = 0;
  static  unsigned long long aesl_llvm_cbe_268_count = 0;
  static  unsigned long long aesl_llvm_cbe_269_count = 0;
  unsigned long long llvm_cbe_tmp__93;
  static  unsigned long long aesl_llvm_cbe_270_count = 0;
  static  unsigned long long aesl_llvm_cbe_271_count = 0;
  unsigned long long llvm_cbe_tmp__94;
  static  unsigned long long aesl_llvm_cbe_272_count = 0;
  static  unsigned long long aesl_llvm_cbe_273_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_update_P\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_200_count);
  llvm_cbe_tmp__77 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__77);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds [16 x i64]* @Pdata_p, i64 0, i64 %%1, !dbg !31 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_201_count);
  llvm_cbe_tmp__78 = (signed long long *)(&Pdata_p[(((signed long long )llvm_cbe_tmp__77))
#ifdef AESL_BC_SIM
 % 16
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__77));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_tmp__77) < 16)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Pdata_p' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_202_count);
  llvm_cbe_tmp__79 = (unsigned long long )*llvm_cbe_tmp__78;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__79);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_205_count);
  llvm_cbe_tmp__80 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__80);
  if (((llvm_cbe_tmp__80&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__95;
  } else {
    goto llvm_cbe_tmp__96;
  }

llvm_cbe_tmp__96:
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = tail call i64 @unscale_data() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_208_count);
   /*tail*/ unscale_data();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__81);
}
  goto llvm_cbe_tmp__95;

llvm_cbe_tmp__95:
  if (((llvm_cbe_Px_new_idx) == (((signed long long *)/*NULL*/0)))) {
    goto llvm_cbe__2e_preheader;
  } else {
    goto llvm_cbe__2e_preheader3;
  }

llvm_cbe__2e_preheader3:
  if ((((signed long long )llvm_cbe_P_new_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6;
  } else {
    goto llvm_cbe__2e_loopexit4;
  }

llvm_cbe__2e_preheader:
  if ((((signed long long )llvm_cbe_tmp__79) > ((signed long long )0ull))) {
    llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

  do {     /* Syntactic loop '.lr.ph6' to make GCC happy */
llvm_cbe__2e_lr_2e_ph6:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge15 = phi i64 [ %%17, %%.lr.ph6 ], [ 0, %%.preheader3  for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_storemerge15_count);
  llvm_cbe_storemerge15 = (unsigned long long )llvm_cbe_storemerge15__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",llvm_cbe_storemerge15);
printf("\n = 0x%I64X",llvm_cbe_tmp__87);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%Px_new, i64 %%storemerge15, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_232_count);
  llvm_cbe_tmp__82 = (float *)(&llvm_cbe_Px_new[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* %%12, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_233_count);
  llvm_cbe_tmp__83 = (float )*llvm_cbe_tmp__82;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__83, *(int*)(&llvm_cbe_tmp__83));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds i64* %%Px_new_idx, i64 %%storemerge15, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_234_count);
  llvm_cbe_tmp__84 = (signed long long *)(&llvm_cbe_Px_new_idx[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* %%14, align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_235_count);
  llvm_cbe_tmp__85 = (unsigned long long )*llvm_cbe_tmp__84;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__85);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [12 x float]* @Pdata_x, i64 0, i64 %%15, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_236_count);
  llvm_cbe_tmp__86 = (float *)(&Pdata_x[(((signed long long )llvm_cbe_tmp__85))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__85));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_tmp__85) < 12 && "Write access out of array 'Pdata_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%16, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_237_count);
  *llvm_cbe_tmp__86 = llvm_cbe_tmp__83;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__83);
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = add nsw i64 %%storemerge15, 1, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_238_count);
  llvm_cbe_tmp__87 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge15&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__87&18446744073709551615ull)));
  if (((llvm_cbe_tmp__87&18446744073709551615ULL) == (llvm_cbe_P_new_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit4;
  } else {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__87;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6;
  }

  } while (1); /* end of syntactic loop '.lr.ph6' */
  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge2 = phi i64 [ %%21, %%.lr.ph ], [ 0, %%.preheader  for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_storemerge2_count);
  llvm_cbe_storemerge2 = (unsigned long long )llvm_cbe_storemerge2__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",llvm_cbe_storemerge2);
printf("\n = 0x%I64X",llvm_cbe_tmp__91);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds float* %%Px_new, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_249_count);
  llvm_cbe_tmp__88 = (float *)(&llvm_cbe_Px_new[(((signed long long )llvm_cbe_storemerge2))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%18, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_250_count);
  llvm_cbe_tmp__89 = (float )*llvm_cbe_tmp__88;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__89, *(int*)(&llvm_cbe_tmp__89));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds [12 x float]* @Pdata_x, i64 0, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_251_count);
  llvm_cbe_tmp__90 = (float *)(&Pdata_x[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 12 && "Write access out of array 'Pdata_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%19, float* %%20, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_252_count);
  *llvm_cbe_tmp__90 = llvm_cbe_tmp__89;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__89);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = add nsw i64 %%storemerge2, 1, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_253_count);
  llvm_cbe_tmp__91 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge2&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__91&18446744073709551615ull)));
  if (((llvm_cbe_tmp__91&18446744073709551615ULL) == (llvm_cbe_tmp__79&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit;
  } else {
    llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__91;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e_loopexit:
  goto llvm_cbe_tmp__97;

llvm_cbe__2e_loopexit4:
  goto llvm_cbe_tmp__97;

llvm_cbe_tmp__97:
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_266_count);
  llvm_cbe_tmp__92 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__92);
  if (((llvm_cbe_tmp__92&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__98;
  } else {
    goto llvm_cbe_tmp__99;
  }

llvm_cbe_tmp__99:
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = tail call i64 @scale_data() nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_269_count);
   /*tail*/ scale_data();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__93);
}
  goto llvm_cbe_tmp__98;

llvm_cbe_tmp__98:
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = tail call i64 @update_linsys_solver_matrices_qdldl() nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_271_count);
   /*tail*/ update_linsys_solver_matrices_qdldl();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__94);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @reset_info(%%struct.OSQPInfo* @info) nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_P  --> \n", ++aesl_llvm_cbe_272_count);
   /*tail*/ reset_info((l_struct_OC_OSQPInfo *)(&info));
if (AESL_DEBUG_TRACE) {
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_update_P}\n");
  return 0ull;
}


signed long long osqp_update_A(float *llvm_cbe_Ax_new, signed long long *llvm_cbe_Ax_new_idx, signed long long llvm_cbe_A_new_n) {
  static  unsigned long long aesl_llvm_cbe_274_count = 0;
  static  unsigned long long aesl_llvm_cbe_275_count = 0;
  static  unsigned long long aesl_llvm_cbe_276_count = 0;
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  unsigned long long llvm_cbe_tmp__100;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  signed long long *llvm_cbe_tmp__101;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  unsigned long long llvm_cbe_tmp__102;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
  unsigned long long llvm_cbe_tmp__103;
  static  unsigned long long aesl_llvm_cbe_288_count = 0;
  static  unsigned long long aesl_llvm_cbe_289_count = 0;
  static  unsigned long long aesl_llvm_cbe_290_count = 0;
  unsigned long long llvm_cbe_tmp__104;
  static  unsigned long long aesl_llvm_cbe_291_count = 0;
  static  unsigned long long aesl_llvm_cbe_292_count = 0;
  static  unsigned long long aesl_llvm_cbe_293_count = 0;
  static  unsigned long long aesl_llvm_cbe_294_count = 0;
  static  unsigned long long aesl_llvm_cbe_295_count = 0;
  static  unsigned long long aesl_llvm_cbe_296_count = 0;
  static  unsigned long long aesl_llvm_cbe_297_count = 0;
  static  unsigned long long aesl_llvm_cbe_298_count = 0;
  static  unsigned long long aesl_llvm_cbe_299_count = 0;
  static  unsigned long long aesl_llvm_cbe_300_count = 0;
  static  unsigned long long aesl_llvm_cbe_301_count = 0;
  static  unsigned long long aesl_llvm_cbe_302_count = 0;
  static  unsigned long long aesl_llvm_cbe_303_count = 0;
  static  unsigned long long aesl_llvm_cbe_304_count = 0;
  static  unsigned long long aesl_llvm_cbe_305_count = 0;
  static  unsigned long long aesl_llvm_cbe_306_count = 0;
  static  unsigned long long aesl_llvm_cbe_307_count = 0;
  static  unsigned long long aesl_llvm_cbe_308_count = 0;
  static  unsigned long long aesl_llvm_cbe_309_count = 0;
  static  unsigned long long aesl_llvm_cbe_310_count = 0;
  static  unsigned long long aesl_llvm_cbe_311_count = 0;
  static  unsigned long long aesl_llvm_cbe_312_count = 0;
  static  unsigned long long aesl_llvm_cbe_313_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge15_count = 0;
  unsigned long long llvm_cbe_storemerge15;
  unsigned long long llvm_cbe_storemerge15__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_314_count = 0;
  float *llvm_cbe_tmp__105;
  static  unsigned long long aesl_llvm_cbe_315_count = 0;
  float llvm_cbe_tmp__106;
  static  unsigned long long aesl_llvm_cbe_316_count = 0;
  signed long long *llvm_cbe_tmp__107;
  static  unsigned long long aesl_llvm_cbe_317_count = 0;
  unsigned long long llvm_cbe_tmp__108;
  static  unsigned long long aesl_llvm_cbe_318_count = 0;
  float *llvm_cbe_tmp__109;
  static  unsigned long long aesl_llvm_cbe_319_count = 0;
  static  unsigned long long aesl_llvm_cbe_320_count = 0;
  unsigned long long llvm_cbe_tmp__110;
  static  unsigned long long aesl_llvm_cbe_321_count = 0;
  static  unsigned long long aesl_llvm_cbe_322_count = 0;
  static  unsigned long long aesl_llvm_cbe_323_count = 0;
  static  unsigned long long aesl_llvm_cbe_324_count = 0;
  static  unsigned long long aesl_llvm_cbe_325_count = 0;
  static  unsigned long long aesl_llvm_cbe_326_count = 0;
  static  unsigned long long aesl_llvm_cbe_327_count = 0;
  static  unsigned long long aesl_llvm_cbe_328_count = 0;
  static  unsigned long long aesl_llvm_cbe_329_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond7_count = 0;
  static  unsigned long long aesl_llvm_cbe_330_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge2_count = 0;
  unsigned long long llvm_cbe_storemerge2;
  unsigned long long llvm_cbe_storemerge2__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_331_count = 0;
  float *llvm_cbe_tmp__111;
  static  unsigned long long aesl_llvm_cbe_332_count = 0;
  float llvm_cbe_tmp__112;
  static  unsigned long long aesl_llvm_cbe_333_count = 0;
  float *llvm_cbe_tmp__113;
  static  unsigned long long aesl_llvm_cbe_334_count = 0;
  static  unsigned long long aesl_llvm_cbe_335_count = 0;
  unsigned long long llvm_cbe_tmp__114;
  static  unsigned long long aesl_llvm_cbe_336_count = 0;
  static  unsigned long long aesl_llvm_cbe_337_count = 0;
  static  unsigned long long aesl_llvm_cbe_338_count = 0;
  static  unsigned long long aesl_llvm_cbe_339_count = 0;
  static  unsigned long long aesl_llvm_cbe_340_count = 0;
  static  unsigned long long aesl_llvm_cbe_341_count = 0;
  static  unsigned long long aesl_llvm_cbe_342_count = 0;
  static  unsigned long long aesl_llvm_cbe_343_count = 0;
  static  unsigned long long aesl_llvm_cbe_344_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_345_count = 0;
  static  unsigned long long aesl_llvm_cbe_346_count = 0;
  static  unsigned long long aesl_llvm_cbe_347_count = 0;
  static  unsigned long long aesl_llvm_cbe_348_count = 0;
  unsigned long long llvm_cbe_tmp__115;
  static  unsigned long long aesl_llvm_cbe_349_count = 0;
  static  unsigned long long aesl_llvm_cbe_350_count = 0;
  static  unsigned long long aesl_llvm_cbe_351_count = 0;
  unsigned long long llvm_cbe_tmp__116;
  static  unsigned long long aesl_llvm_cbe_352_count = 0;
  static  unsigned long long aesl_llvm_cbe_353_count = 0;
  unsigned long long llvm_cbe_tmp__117;
  static  unsigned long long aesl_llvm_cbe_354_count = 0;
  static  unsigned long long aesl_llvm_cbe_355_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_update_A\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_282_count);
  llvm_cbe_tmp__100 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__100);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds [16 x i64]* @Adata_p, i64 0, i64 %%1, !dbg !31 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_283_count);
  llvm_cbe_tmp__101 = (signed long long *)(&Adata_p[(((signed long long )llvm_cbe_tmp__100))
#ifdef AESL_BC_SIM
 % 16
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__100));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_tmp__100) < 16)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Adata_p' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_284_count);
  llvm_cbe_tmp__102 = (unsigned long long )*llvm_cbe_tmp__101;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__102);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !31 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_287_count);
  llvm_cbe_tmp__103 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__103);
  if (((llvm_cbe_tmp__103&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__118;
  } else {
    goto llvm_cbe_tmp__119;
  }

llvm_cbe_tmp__119:
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = tail call i64 @unscale_data() nounwind, !dbg !31 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_290_count);
   /*tail*/ unscale_data();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__104);
}
  goto llvm_cbe_tmp__118;

llvm_cbe_tmp__118:
  if (((llvm_cbe_Ax_new_idx) == (((signed long long *)/*NULL*/0)))) {
    goto llvm_cbe__2e_preheader;
  } else {
    goto llvm_cbe__2e_preheader3;
  }

llvm_cbe__2e_preheader3:
  if ((((signed long long )llvm_cbe_A_new_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6;
  } else {
    goto llvm_cbe__2e_loopexit4;
  }

llvm_cbe__2e_preheader:
  if ((((signed long long )llvm_cbe_tmp__102) > ((signed long long )0ull))) {
    llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

  do {     /* Syntactic loop '.lr.ph6' to make GCC happy */
llvm_cbe__2e_lr_2e_ph6:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge15 = phi i64 [ %%17, %%.lr.ph6 ], [ 0, %%.preheader3  for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_storemerge15_count);
  llvm_cbe_storemerge15 = (unsigned long long )llvm_cbe_storemerge15__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",llvm_cbe_storemerge15);
printf("\n = 0x%I64X",llvm_cbe_tmp__110);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%Ax_new, i64 %%storemerge15, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_314_count);
  llvm_cbe_tmp__105 = (float *)(&llvm_cbe_Ax_new[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* %%12, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_315_count);
  llvm_cbe_tmp__106 = (float )*llvm_cbe_tmp__105;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__106, *(int*)(&llvm_cbe_tmp__106));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds i64* %%Ax_new_idx, i64 %%storemerge15, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_316_count);
  llvm_cbe_tmp__107 = (signed long long *)(&llvm_cbe_Ax_new_idx[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* %%14, align 8, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_317_count);
  llvm_cbe_tmp__108 = (unsigned long long )*llvm_cbe_tmp__107;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__108);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [43 x float]* @Adata_x, i64 0, i64 %%15, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_318_count);
  llvm_cbe_tmp__109 = (float *)(&Adata_x[(((signed long long )llvm_cbe_tmp__108))
#ifdef AESL_BC_SIM
 % 43
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__108));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_tmp__108) < 43 && "Write access out of array 'Adata_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%16, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_319_count);
  *llvm_cbe_tmp__109 = llvm_cbe_tmp__106;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__106);
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = add nsw i64 %%storemerge15, 1, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_320_count);
  llvm_cbe_tmp__110 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge15&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__110&18446744073709551615ull)));
  if (((llvm_cbe_tmp__110&18446744073709551615ULL) == (llvm_cbe_A_new_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit4;
  } else {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__110;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6;
  }

  } while (1); /* end of syntactic loop '.lr.ph6' */
  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge2 = phi i64 [ %%21, %%.lr.ph ], [ 0, %%.preheader  for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_storemerge2_count);
  llvm_cbe_storemerge2 = (unsigned long long )llvm_cbe_storemerge2__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",llvm_cbe_storemerge2);
printf("\n = 0x%I64X",llvm_cbe_tmp__114);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds float* %%Ax_new, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_331_count);
  llvm_cbe_tmp__111 = (float *)(&llvm_cbe_Ax_new[(((signed long long )llvm_cbe_storemerge2))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%18, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_332_count);
  llvm_cbe_tmp__112 = (float )*llvm_cbe_tmp__111;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__112, *(int*)(&llvm_cbe_tmp__112));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds [43 x float]* @Adata_x, i64 0, i64 %%storemerge2, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_333_count);
  llvm_cbe_tmp__113 = (float *)(&Adata_x[(((signed long long )llvm_cbe_storemerge2))
#ifdef AESL_BC_SIM
 % 43
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge2 = 0x%I64X",((signed long long )llvm_cbe_storemerge2));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge2) < 43 && "Write access out of array 'Adata_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%19, float* %%20, align 4, !dbg !30 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_334_count);
  *llvm_cbe_tmp__113 = llvm_cbe_tmp__112;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__112);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = add nsw i64 %%storemerge2, 1, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_335_count);
  llvm_cbe_tmp__114 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge2&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__114&18446744073709551615ull)));
  if (((llvm_cbe_tmp__114&18446744073709551615ULL) == (llvm_cbe_tmp__102&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit;
  } else {
    llvm_cbe_storemerge2__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__114;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e_loopexit:
  goto llvm_cbe_tmp__120;

llvm_cbe__2e_loopexit4:
  goto llvm_cbe_tmp__120;

llvm_cbe_tmp__120:
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_348_count);
  llvm_cbe_tmp__115 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__115);
  if (((llvm_cbe_tmp__115&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__121;
  } else {
    goto llvm_cbe_tmp__122;
  }

llvm_cbe_tmp__122:
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = tail call i64 @scale_data() nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_351_count);
   /*tail*/ scale_data();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__116);
}
  goto llvm_cbe_tmp__121;

llvm_cbe_tmp__121:
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = tail call i64 @update_linsys_solver_matrices_qdldl() nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_353_count);
   /*tail*/ update_linsys_solver_matrices_qdldl();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__117);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @reset_info(%%struct.OSQPInfo* @info) nounwind, !dbg !32 for 0x%I64xth hint within @osqp_update_A  --> \n", ++aesl_llvm_cbe_354_count);
   /*tail*/ reset_info((l_struct_OC_OSQPInfo *)(&info));
if (AESL_DEBUG_TRACE) {
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_update_A}\n");
  return 0ull;
}

