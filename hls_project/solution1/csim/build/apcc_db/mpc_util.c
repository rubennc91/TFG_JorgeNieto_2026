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

/* External Global Variable Declarations */

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void inverse_matrix_2x2(float llvm_cbe_a, float llvm_cbe_b, float llvm_cbe_c, float llvm_cbe_d, float (*llvm_cbe_inv)[2]);
void multiplyMatrixVector(float (*llvm_cbe_Ex)[2], float *llvm_cbe_u, float *llvm_cbe_result);
void referencia(float *llvm_cbe_q_new, float llvm_cbe_ref);
signed long long osqp_update_lin_cost(float *);
void calculateV(float *llvm_cbe_Ax, float (*llvm_cbe_Ex)[2], float *llvm_cbe_u, float *llvm_cbe_v);
void atualizar_restricao(float *llvm_cbe_l_new, float *llvm_cbe_u_new, float *llvm_cbe_x, float *llvm_cbe_v00);
void atualizar_restricao_v(float *llvm_cbe_l_new, float *llvm_cbe_u_new, float llvm_cbe_vdc, float (*llvm_cbe_Einv)[2], float *llvm_cbe_Ax);
signed long long osqp_update_bounds(float *, float *);
void atualizar_A(float (*llvm_cbe_Einv)[2]);
signed long long osqp_update_A(float *, signed long long *, signed long long );


/* Global Variable Definitions and Initialization */
static signed long long aesl_internal_atualizar_A_OC_A_idx[12] = { 18ull, 21ull, 23ull, 25ull, 27ull, 31ull, 32ull, 33ull, 35ull, 37ull, 41ull, 42ull };


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

void inverse_matrix_2x2(float llvm_cbe_a, float llvm_cbe_b, float llvm_cbe_c, float llvm_cbe_d, float (*llvm_cbe_inv)[2]) {
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
  static  unsigned long long aesl_llvm_cbe_16_count = 0;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  float llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  float llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  float llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  float llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  float llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  float *llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  float llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  float llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  float *llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;
  static  unsigned long long aesl_llvm_cbe_36_count = 0;
  float llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_37_count = 0;
  float llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_38_count = 0;
  float *llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_39_count = 0;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;
  float llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_41_count = 0;
  float *llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_42_count = 0;
  static  unsigned long long aesl_llvm_cbe_43_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @inverse_matrix_2x2\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = fmul float %%a, %%d, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_18_count);
  llvm_cbe_tmp__1 = (float )((float )(llvm_cbe_a * llvm_cbe_d));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__1, *(int*)(&llvm_cbe_tmp__1));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = fmul float %%b, %%c, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__2 = (float )((float )(llvm_cbe_b * llvm_cbe_c));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__2, *(int*)(&llvm_cbe_tmp__2));
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = fsub float %%1, %%2, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_20_count);
  llvm_cbe_tmp__3 = (float )((float )(llvm_cbe_tmp__1 - llvm_cbe_tmp__2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__3, *(int*)(&llvm_cbe_tmp__3));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fdiv float 1.000000e+00, %%3, !dbg !9 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__4 = (float )((float )(0x1p0 / llvm_cbe_tmp__3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__4, *(int*)(&llvm_cbe_tmp__4));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fmul float %%4, %%d, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_29_count);
  llvm_cbe_tmp__5 = (float )((float )(llvm_cbe_tmp__4 * llvm_cbe_d));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__5, *(int*)(&llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds [2 x float]* %%inv, i64 0, i64 0, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_30_count);
  llvm_cbe_tmp__6 = (float *)(&(*llvm_cbe_inv)[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'inv' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%5, float* %%6, align 4, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_31_count);
  *llvm_cbe_tmp__6 = llvm_cbe_tmp__5;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__5);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fsub float -0.000000e+00, %%b, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_32_count);
  llvm_cbe_tmp__7 = (float )((float )(-(llvm_cbe_b)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__7, *(int*)(&llvm_cbe_tmp__7));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = fmul float %%4, %%7, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_33_count);
  llvm_cbe_tmp__8 = (float )((float )(llvm_cbe_tmp__4 * llvm_cbe_tmp__7));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__8, *(int*)(&llvm_cbe_tmp__8));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds [2 x float]* %%inv, i64 0, i64 1, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_34_count);
  llvm_cbe_tmp__9 = (float *)(&(*llvm_cbe_inv)[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'inv' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%8, float* %%9, align 4, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_35_count);
  *llvm_cbe_tmp__9 = llvm_cbe_tmp__8;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__8);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fsub float -0.000000e+00, %%c, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_36_count);
  llvm_cbe_tmp__10 = (float )((float )(-(llvm_cbe_c)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__10, *(int*)(&llvm_cbe_tmp__10));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = fmul float %%4, %%10, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_37_count);
  llvm_cbe_tmp__11 = (float )((float )(llvm_cbe_tmp__4 * llvm_cbe_tmp__10));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__11, *(int*)(&llvm_cbe_tmp__11));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds [2 x float]* %%inv, i64 1, i64 0, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_38_count);
  llvm_cbe_tmp__12 = (float *)(&llvm_cbe_inv[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'inv' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%12, align 4, !dbg !8 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_39_count);
  *llvm_cbe_tmp__12 = llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__11);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%4, %%a, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_40_count);
  llvm_cbe_tmp__13 = (float )((float )(llvm_cbe_tmp__4 * llvm_cbe_a));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__13, *(int*)(&llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds [2 x float]* %%inv, i64 1, i64 1, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_41_count);
  llvm_cbe_tmp__14 = (float *)(&llvm_cbe_inv[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'inv' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%14, align 4, !dbg !7 for 0x%I64xth hint within @inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_42_count);
  *llvm_cbe_tmp__14 = llvm_cbe_tmp__13;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__13);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @inverse_matrix_2x2}\n");
  return;
}


void multiplyMatrixVector(float (*llvm_cbe_Ex)[2], float *llvm_cbe_u, float *llvm_cbe_result) {
  static  unsigned long long aesl_llvm_cbe_44_count = 0;
  static  unsigned long long aesl_llvm_cbe_45_count = 0;
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  static  unsigned long long aesl_llvm_cbe_51_count = 0;
  static  unsigned long long aesl_llvm_cbe_52_count = 0;
  static  unsigned long long aesl_llvm_cbe_53_count = 0;
  static  unsigned long long aesl_llvm_cbe_54_count = 0;
  static  unsigned long long aesl_llvm_cbe_55_count = 0;
  static  unsigned long long aesl_llvm_cbe_56_count = 0;
  static  unsigned long long aesl_llvm_cbe_57_count = 0;
  static  unsigned long long aesl_llvm_cbe_58_count = 0;
  static  unsigned long long aesl_llvm_cbe_59_count = 0;
  static  unsigned long long aesl_llvm_cbe_60_count = 0;
  static  unsigned long long aesl_llvm_cbe_61_count = 0;
  static  unsigned long long aesl_llvm_cbe_62_count = 0;
  static  unsigned long long aesl_llvm_cbe_63_count = 0;
  float *llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  float llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  float llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  float llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  float llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  float *llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  float llvm_cbe_tmp__21;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  float *llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  float llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  float llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  float llvm_cbe_tmp__25;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;
  static  unsigned long long aesl_llvm_cbe_92_count = 0;
  float *llvm_cbe_tmp__26;
  static  unsigned long long aesl_llvm_cbe_93_count = 0;
  static  unsigned long long aesl_llvm_cbe_94_count = 0;
  static  unsigned long long aesl_llvm_cbe_95_count = 0;
  static  unsigned long long aesl_llvm_cbe_96_count = 0;
  static  unsigned long long aesl_llvm_cbe_97_count = 0;
  static  unsigned long long aesl_llvm_cbe_98_count = 0;
  static  unsigned long long aesl_llvm_cbe_99_count = 0;
  float *llvm_cbe_tmp__27;
  static  unsigned long long aesl_llvm_cbe_100_count = 0;
  float llvm_cbe_tmp__28;
  static  unsigned long long aesl_llvm_cbe_101_count = 0;
  float llvm_cbe_tmp__29;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  float llvm_cbe_tmp__30;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  float llvm_cbe_tmp__31;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  float *llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  float llvm_cbe_tmp__33;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  float llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  float llvm_cbe_tmp__35;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_115_count = 0;
  static  unsigned long long aesl_llvm_cbe_116_count = 0;
  static  unsigned long long aesl_llvm_cbe_117_count = 0;
  static  unsigned long long aesl_llvm_cbe_118_count = 0;
  static  unsigned long long aesl_llvm_cbe_119_count = 0;
  static  unsigned long long aesl_llvm_cbe_120_count = 0;
  static  unsigned long long aesl_llvm_cbe_121_count = 0;
  static  unsigned long long aesl_llvm_cbe_122_count = 0;
  static  unsigned long long aesl_llvm_cbe_123_count = 0;
  static  unsigned long long aesl_llvm_cbe_124_count = 0;
  static  unsigned long long aesl_llvm_cbe_125_count = 0;
  static  unsigned long long aesl_llvm_cbe_126_count = 0;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @multiplyMatrixVector\n");
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%result, align 4, !dbg !8 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_57_count);
  *llvm_cbe_result = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds [2 x float]* %%Ex, i64 0, i64 0, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_63_count);
  llvm_cbe_tmp__15 = (float *)(&(*llvm_cbe_Ex)[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ex' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load float* %%1, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_64_count);
  llvm_cbe_tmp__16 = (float )*llvm_cbe_tmp__15;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__16, *(int*)(&llvm_cbe_tmp__16));
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%u, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_65_count);
  llvm_cbe_tmp__17 = (float )*llvm_cbe_u;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__17, *(int*)(&llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fmul float %%2, %%3, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_66_count);
  llvm_cbe_tmp__18 = (float )((float )(llvm_cbe_tmp__16 * llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fadd float %%4, 0.000000e+00, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_67_count);
  llvm_cbe_tmp__19 = (float )((float )(llvm_cbe_tmp__18 + 0x0p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__19, *(int*)(&llvm_cbe_tmp__19));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%5, float* %%result, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_68_count);
  *llvm_cbe_result = llvm_cbe_tmp__19;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__19);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds [2 x float]* %%Ex, i64 0, i64 1, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_74_count);
  llvm_cbe_tmp__20 = (float *)(&(*llvm_cbe_Ex)[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ex' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load float* %%6, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_75_count);
  llvm_cbe_tmp__21 = (float )*llvm_cbe_tmp__20;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__21, *(int*)(&llvm_cbe_tmp__21));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds float* %%u, i64 1, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_76_count);
  llvm_cbe_tmp__22 = (float *)(&llvm_cbe_u[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* %%8, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_77_count);
  llvm_cbe_tmp__23 = (float )*llvm_cbe_tmp__22;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__23, *(int*)(&llvm_cbe_tmp__23));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fmul float %%7, %%9, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_78_count);
  llvm_cbe_tmp__24 = (float )((float )(llvm_cbe_tmp__21 * llvm_cbe_tmp__23));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__24, *(int*)(&llvm_cbe_tmp__24));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = fadd float %%5, %%10, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_79_count);
  llvm_cbe_tmp__25 = (float )((float )(llvm_cbe_tmp__19 + llvm_cbe_tmp__24));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__25, *(int*)(&llvm_cbe_tmp__25));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%result, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_80_count);
  *llvm_cbe_result = llvm_cbe_tmp__25;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__25);
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%result, i64 1, !dbg !8 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_92_count);
  llvm_cbe_tmp__26 = (float *)(&llvm_cbe_result[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%12, align 4, !dbg !8 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_93_count);
  *llvm_cbe_tmp__26 = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [2 x float]* %%Ex, i64 1, i64 0, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_99_count);
  llvm_cbe_tmp__27 = (float *)(&llvm_cbe_Ex[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ex' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = load float* %%13, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_100_count);
  llvm_cbe_tmp__28 = (float )*llvm_cbe_tmp__27;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__28, *(int*)(&llvm_cbe_tmp__28));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load float* %%u, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_101_count);
  llvm_cbe_tmp__29 = (float )*llvm_cbe_u;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__29, *(int*)(&llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = fmul float %%14, %%15, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_102_count);
  llvm_cbe_tmp__30 = (float )((float )(llvm_cbe_tmp__28 * llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__30, *(int*)(&llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = fadd float %%16, 0.000000e+00, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_103_count);
  llvm_cbe_tmp__31 = (float )((float )(llvm_cbe_tmp__30 + 0x0p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__31, *(int*)(&llvm_cbe_tmp__31));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%17, float* %%12, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_104_count);
  *llvm_cbe_tmp__26 = llvm_cbe_tmp__31;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__31);
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds [2 x float]* %%Ex, i64 1, i64 1, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_110_count);
  llvm_cbe_tmp__32 = (float *)(&llvm_cbe_Ex[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ex' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%18, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_111_count);
  llvm_cbe_tmp__33 = (float )*llvm_cbe_tmp__32;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__33, *(int*)(&llvm_cbe_tmp__33));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = load float* %%8, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_112_count);
  llvm_cbe_tmp__34 = (float )*llvm_cbe_tmp__22;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__34, *(int*)(&llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = fmul float %%19, %%20, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_113_count);
  llvm_cbe_tmp__35 = (float )((float )(llvm_cbe_tmp__33 * llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__35, *(int*)(&llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fadd float %%17, %%21, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_114_count);
  llvm_cbe_tmp__36 = (float )((float )(llvm_cbe_tmp__31 + llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%22, float* %%12, align 4, !dbg !7 for 0x%I64xth hint within @multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_115_count);
  *llvm_cbe_tmp__26 = llvm_cbe_tmp__36;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__36);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @multiplyMatrixVector}\n");
  return;
}


void referencia(float *llvm_cbe_q_new, float llvm_cbe_ref) {
  static  unsigned long long aesl_llvm_cbe_128_count = 0;
  static  unsigned long long aesl_llvm_cbe_129_count = 0;
  static  unsigned long long aesl_llvm_cbe_130_count = 0;
  static  unsigned long long aesl_llvm_cbe_131_count = 0;
  static  unsigned long long aesl_llvm_cbe_132_count = 0;
  static  unsigned long long aesl_llvm_cbe_133_count = 0;
  static  unsigned long long aesl_llvm_cbe_134_count = 0;
  static  unsigned long long aesl_llvm_cbe_135_count = 0;
  static  unsigned long long aesl_llvm_cbe_136_count = 0;
  static  unsigned long long aesl_llvm_cbe_137_count = 0;
  float llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  float *llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
  float *llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_141_count = 0;
  static  unsigned long long aesl_llvm_cbe_142_count = 0;
  float llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_143_count = 0;
  float *llvm_cbe_tmp__41;
  static  unsigned long long aesl_llvm_cbe_144_count = 0;
  static  unsigned long long aesl_llvm_cbe_145_count = 0;
  unsigned long long llvm_cbe_tmp__42;
  static  unsigned long long aesl_llvm_cbe_146_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @referencia\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = fmul float %%ref, -1.000000e+04, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_137_count);
  llvm_cbe_tmp__37 = (float )((float )(llvm_cbe_ref * -0x1.388p13));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__37, *(int*)(&llvm_cbe_tmp__37));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%q_new, i64 1, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_138_count);
  llvm_cbe_tmp__38 = (float *)(&llvm_cbe_q_new[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%1, float* %%2, align 4, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_139_count);
  *llvm_cbe_tmp__38 = llvm_cbe_tmp__37;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__37);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%q_new, i64 4, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_140_count);
  llvm_cbe_tmp__39 = (float *)(&llvm_cbe_q_new[(((signed long long )4ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%1, float* %%3, align 4, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_141_count);
  *llvm_cbe_tmp__39 = llvm_cbe_tmp__37;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__37);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fmul float %%ref, -1.000000e+05, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_142_count);
  llvm_cbe_tmp__40 = (float )((float )(llvm_cbe_ref * -0x1.86ap16));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__40, *(int*)(&llvm_cbe_tmp__40));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%q_new, i64 7, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_143_count);
  llvm_cbe_tmp__41 = (float *)(&llvm_cbe_q_new[(((signed long long )7ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%5, align 4, !dbg !7 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_144_count);
  *llvm_cbe_tmp__41 = llvm_cbe_tmp__40;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__40);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = tail call i64 @osqp_update_lin_cost(float* %%q_new) nounwind, !dbg !8 for 0x%I64xth hint within @referencia  --> \n", ++aesl_llvm_cbe_145_count);
   /*tail*/ osqp_update_lin_cost((float *)llvm_cbe_q_new);
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__42);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @referencia}\n");
  return;
}


void calculateV(float *llvm_cbe_Ax, float (*llvm_cbe_Ex)[2], float *llvm_cbe_u, float *llvm_cbe_v) {
  static  unsigned long long aesl_llvm_cbe_Exu_count = 0;
  float llvm_cbe_Exu[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_147_count = 0;
  static  unsigned long long aesl_llvm_cbe_148_count = 0;
  static  unsigned long long aesl_llvm_cbe_149_count = 0;
  static  unsigned long long aesl_llvm_cbe_150_count = 0;
  static  unsigned long long aesl_llvm_cbe_151_count = 0;
  static  unsigned long long aesl_llvm_cbe_152_count = 0;
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  float *llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  float llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  float llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  float llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  float *llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_175_count = 0;
  float llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
  float *llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_177_count = 0;
  float llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_178_count = 0;
  float llvm_cbe_tmp__51;
  static  unsigned long long aesl_llvm_cbe_179_count = 0;
  float *llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_180_count = 0;
  static  unsigned long long aesl_llvm_cbe_181_count = 0;
  static  unsigned long long aesl_llvm_cbe_182_count = 0;
  static  unsigned long long aesl_llvm_cbe_183_count = 0;
  static  unsigned long long aesl_llvm_cbe_184_count = 0;
  static  unsigned long long aesl_llvm_cbe_185_count = 0;
  static  unsigned long long aesl_llvm_cbe_186_count = 0;
  static  unsigned long long aesl_llvm_cbe_187_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @calculateV\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds [2 x float]* %%Exu, i64 0, i64 0, !dbg !8 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_156_count);
  llvm_cbe_tmp__43 = (float *)(&llvm_cbe_Exu[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @multiplyMatrixVector([2 x float]* %%Ex, float* %%u, float* %%1), !dbg !8 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_157_count);
  multiplyMatrixVector(llvm_cbe_Ex, (float *)llvm_cbe_u, (float *)llvm_cbe_tmp__43);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load float* %%Ax, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_164_count);
  llvm_cbe_tmp__44 = (float )*llvm_cbe_Ax;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__44, *(int*)(&llvm_cbe_tmp__44));

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Exu' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%1, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_165_count);
  llvm_cbe_tmp__45 = (float )*llvm_cbe_tmp__43;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__45, *(int*)(&llvm_cbe_tmp__45));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fadd float %%2, %%3, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_166_count);
  llvm_cbe_tmp__46 = (float )((float )(llvm_cbe_tmp__44 + llvm_cbe_tmp__45));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__46, *(int*)(&llvm_cbe_tmp__46));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%v, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_167_count);
  *llvm_cbe_v = llvm_cbe_tmp__46;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__46);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%Ax, i64 1, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_174_count);
  llvm_cbe_tmp__47 = (float *)(&llvm_cbe_Ax[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_175_count);
  llvm_cbe_tmp__48 = (float )*llvm_cbe_tmp__47;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__48, *(int*)(&llvm_cbe_tmp__48));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [2 x float]* %%Exu, i64 0, i64 1, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_176_count);
  llvm_cbe_tmp__49 = (float *)(&llvm_cbe_Exu[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Exu' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load float* %%7, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_177_count);
  llvm_cbe_tmp__50 = (float )*llvm_cbe_tmp__49;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__50, *(int*)(&llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = fadd float %%6, %%8, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_178_count);
  llvm_cbe_tmp__51 = (float )((float )(llvm_cbe_tmp__48 + llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__51, *(int*)(&llvm_cbe_tmp__51));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds float* %%v, i64 1, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_179_count);
  llvm_cbe_tmp__52 = (float *)(&llvm_cbe_v[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%10, align 4, !dbg !7 for 0x%I64xth hint within @calculateV  --> \n", ++aesl_llvm_cbe_180_count);
  *llvm_cbe_tmp__52 = llvm_cbe_tmp__51;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__51);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @calculateV}\n");
  return;
}


void atualizar_restricao(float *llvm_cbe_l_new, float *llvm_cbe_u_new, float *llvm_cbe_x, float *llvm_cbe_v00) {
  static  unsigned long long aesl_llvm_cbe_188_count = 0;
  static  unsigned long long aesl_llvm_cbe_189_count = 0;
  static  unsigned long long aesl_llvm_cbe_190_count = 0;
  static  unsigned long long aesl_llvm_cbe_191_count = 0;
  static  unsigned long long aesl_llvm_cbe_192_count = 0;
  static  unsigned long long aesl_llvm_cbe_193_count = 0;
  static  unsigned long long aesl_llvm_cbe_194_count = 0;
  static  unsigned long long aesl_llvm_cbe_195_count = 0;
  static  unsigned long long aesl_llvm_cbe_196_count = 0;
  static  unsigned long long aesl_llvm_cbe_197_count = 0;
  static  unsigned long long aesl_llvm_cbe_198_count = 0;
  static  unsigned long long aesl_llvm_cbe_199_count = 0;
  static  unsigned long long aesl_llvm_cbe_200_count = 0;
  static  unsigned long long aesl_llvm_cbe_201_count = 0;
  static  unsigned long long aesl_llvm_cbe_202_count = 0;
  static  unsigned long long aesl_llvm_cbe_203_count = 0;
  static  unsigned long long aesl_llvm_cbe_204_count = 0;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  static  unsigned long long aesl_llvm_cbe_206_count = 0;
  static  unsigned long long aesl_llvm_cbe_207_count = 0;
  static  unsigned long long aesl_llvm_cbe_208_count = 0;
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
  float llvm_cbe_tmp__53;
  static  unsigned long long aesl_llvm_cbe_220_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_221_count = 0;
  static  unsigned long long aesl_llvm_cbe_222_count = 0;
  float llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_223_count = 0;
  float llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_224_count = 0;
  static  unsigned long long aesl_llvm_cbe_225_count = 0;
  static  unsigned long long aesl_llvm_cbe_226_count = 0;
  static  unsigned long long aesl_llvm_cbe_227_count = 0;
  static  unsigned long long aesl_llvm_cbe_228_count = 0;
  static  unsigned long long aesl_llvm_cbe_229_count = 0;
  static  unsigned long long aesl_llvm_cbe_230_count = 0;
  static  unsigned long long aesl_llvm_cbe_231_count = 0;
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  float *llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  float llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  float *llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;
  float llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  float llvm_cbe_tmp__62;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
  float *llvm_cbe_tmp__63;
  static  unsigned long long aesl_llvm_cbe_244_count = 0;
  static  unsigned long long aesl_llvm_cbe_245_count = 0;
  static  unsigned long long aesl_llvm_cbe_246_count = 0;
  static  unsigned long long aesl_llvm_cbe_247_count = 0;
  static  unsigned long long aesl_llvm_cbe_248_count = 0;
  static  unsigned long long aesl_llvm_cbe_249_count = 0;
  static  unsigned long long aesl_llvm_cbe_250_count = 0;
  static  unsigned long long aesl_llvm_cbe_251_count = 0;
  static  unsigned long long aesl_llvm_cbe_252_count = 0;
  static  unsigned long long aesl_llvm_cbe_253_count = 0;
  static  unsigned long long aesl_llvm_cbe_254_count = 0;
  static  unsigned long long aesl_llvm_cbe_255_count = 0;
  static  unsigned long long aesl_llvm_cbe_256_count = 0;
  float *llvm_cbe_tmp__64;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
  float llvm_cbe_tmp__65;
  static  unsigned long long aesl_llvm_cbe_258_count = 0;
  float llvm_cbe_tmp__66;
  static  unsigned long long aesl_llvm_cbe_259_count = 0;
  float *llvm_cbe_tmp__67;
  static  unsigned long long aesl_llvm_cbe_260_count = 0;
  static  unsigned long long aesl_llvm_cbe_261_count = 0;
  float llvm_cbe_tmp__68;
  static  unsigned long long aesl_llvm_cbe_262_count = 0;
  float llvm_cbe_tmp__69;
  static  unsigned long long aesl_llvm_cbe_263_count = 0;
  float *llvm_cbe_tmp__70;
  static  unsigned long long aesl_llvm_cbe_264_count = 0;
  static  unsigned long long aesl_llvm_cbe_265_count = 0;
  static  unsigned long long aesl_llvm_cbe_266_count = 0;
  static  unsigned long long aesl_llvm_cbe_267_count = 0;
  static  unsigned long long aesl_llvm_cbe_268_count = 0;
  static  unsigned long long aesl_llvm_cbe_269_count = 0;
  static  unsigned long long aesl_llvm_cbe_270_count = 0;
  static  unsigned long long aesl_llvm_cbe_271_count = 0;
  static  unsigned long long aesl_llvm_cbe_272_count = 0;
  static  unsigned long long aesl_llvm_cbe_273_count = 0;
  static  unsigned long long aesl_llvm_cbe_274_count = 0;
  static  unsigned long long aesl_llvm_cbe_275_count = 0;
  static  unsigned long long aesl_llvm_cbe_276_count = 0;
  float llvm_cbe_tmp__71;
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  float *llvm_cbe_tmp__72;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  float llvm_cbe_tmp__73;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  float *llvm_cbe_tmp__74;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  float *llvm_cbe_tmp__75;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  float llvm_cbe_tmp__76;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  float *llvm_cbe_tmp__77;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  float llvm_cbe_tmp__78;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
  float *llvm_cbe_tmp__79;
  static  unsigned long long aesl_llvm_cbe_288_count = 0;
  static  unsigned long long aesl_llvm_cbe_289_count = 0;
  static  unsigned long long aesl_llvm_cbe_290_count = 0;
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
  float llvm_cbe_tmp__80;
  static  unsigned long long aesl_llvm_cbe_301_count = 0;
  float llvm_cbe_tmp__81;
  static  unsigned long long aesl_llvm_cbe_302_count = 0;
  float *llvm_cbe_tmp__82;
  static  unsigned long long aesl_llvm_cbe_303_count = 0;
  static  unsigned long long aesl_llvm_cbe_304_count = 0;
  float llvm_cbe_tmp__83;
  static  unsigned long long aesl_llvm_cbe_305_count = 0;
  float llvm_cbe_tmp__84;
  static  unsigned long long aesl_llvm_cbe_306_count = 0;
  float *llvm_cbe_tmp__85;
  static  unsigned long long aesl_llvm_cbe_307_count = 0;
  static  unsigned long long aesl_llvm_cbe_308_count = 0;
  static  unsigned long long aesl_llvm_cbe_309_count = 0;
  static  unsigned long long aesl_llvm_cbe_310_count = 0;
  static  unsigned long long aesl_llvm_cbe_311_count = 0;
  static  unsigned long long aesl_llvm_cbe_312_count = 0;
  static  unsigned long long aesl_llvm_cbe_313_count = 0;
  static  unsigned long long aesl_llvm_cbe_314_count = 0;
  static  unsigned long long aesl_llvm_cbe_315_count = 0;
  static  unsigned long long aesl_llvm_cbe_316_count = 0;
  static  unsigned long long aesl_llvm_cbe_317_count = 0;
  static  unsigned long long aesl_llvm_cbe_318_count = 0;
  static  unsigned long long aesl_llvm_cbe_319_count = 0;
  float llvm_cbe_tmp__86;
  static  unsigned long long aesl_llvm_cbe_320_count = 0;
  float llvm_cbe_tmp__87;
  static  unsigned long long aesl_llvm_cbe_321_count = 0;
  float *llvm_cbe_tmp__88;
  static  unsigned long long aesl_llvm_cbe_322_count = 0;
  static  unsigned long long aesl_llvm_cbe_323_count = 0;
  float llvm_cbe_tmp__89;
  static  unsigned long long aesl_llvm_cbe_324_count = 0;
  float llvm_cbe_tmp__90;
  static  unsigned long long aesl_llvm_cbe_325_count = 0;
  float *llvm_cbe_tmp__91;
  static  unsigned long long aesl_llvm_cbe_326_count = 0;
  static  unsigned long long aesl_llvm_cbe_327_count = 0;
  static  unsigned long long aesl_llvm_cbe_328_count = 0;
  static  unsigned long long aesl_llvm_cbe_329_count = 0;
  static  unsigned long long aesl_llvm_cbe_330_count = 0;
  static  unsigned long long aesl_llvm_cbe_331_count = 0;
  static  unsigned long long aesl_llvm_cbe_332_count = 0;
  static  unsigned long long aesl_llvm_cbe_333_count = 0;
  static  unsigned long long aesl_llvm_cbe_334_count = 0;
  static  unsigned long long aesl_llvm_cbe_335_count = 0;
  static  unsigned long long aesl_llvm_cbe_336_count = 0;
  static  unsigned long long aesl_llvm_cbe_337_count = 0;
  static  unsigned long long aesl_llvm_cbe_338_count = 0;
  float llvm_cbe_tmp__92;
  static  unsigned long long aesl_llvm_cbe_339_count = 0;
  float llvm_cbe_tmp__93;
  static  unsigned long long aesl_llvm_cbe_340_count = 0;
  float *llvm_cbe_tmp__94;
  static  unsigned long long aesl_llvm_cbe_341_count = 0;
  static  unsigned long long aesl_llvm_cbe_342_count = 0;
  float llvm_cbe_tmp__95;
  static  unsigned long long aesl_llvm_cbe_343_count = 0;
  float llvm_cbe_tmp__96;
  static  unsigned long long aesl_llvm_cbe_344_count = 0;
  float *llvm_cbe_tmp__97;
  static  unsigned long long aesl_llvm_cbe_345_count = 0;
  static  unsigned long long aesl_llvm_cbe_346_count = 0;
  static  unsigned long long aesl_llvm_cbe_347_count = 0;
  static  unsigned long long aesl_llvm_cbe_348_count = 0;
  static  unsigned long long aesl_llvm_cbe_349_count = 0;
  static  unsigned long long aesl_llvm_cbe_350_count = 0;
  static  unsigned long long aesl_llvm_cbe_351_count = 0;
  static  unsigned long long aesl_llvm_cbe_352_count = 0;
  static  unsigned long long aesl_llvm_cbe_353_count = 0;
  static  unsigned long long aesl_llvm_cbe_354_count = 0;
  static  unsigned long long aesl_llvm_cbe_355_count = 0;
  static  unsigned long long aesl_llvm_cbe_356_count = 0;
  static  unsigned long long aesl_llvm_cbe_357_count = 0;
  float llvm_cbe_tmp__98;
  static  unsigned long long aesl_llvm_cbe_358_count = 0;
  float llvm_cbe_tmp__99;
  static  unsigned long long aesl_llvm_cbe_359_count = 0;
  float *llvm_cbe_tmp__100;
  static  unsigned long long aesl_llvm_cbe_360_count = 0;
  static  unsigned long long aesl_llvm_cbe_361_count = 0;
  float llvm_cbe_tmp__101;
  static  unsigned long long aesl_llvm_cbe_362_count = 0;
  float llvm_cbe_tmp__102;
  static  unsigned long long aesl_llvm_cbe_363_count = 0;
  float *llvm_cbe_tmp__103;
  static  unsigned long long aesl_llvm_cbe_364_count = 0;
  static  unsigned long long aesl_llvm_cbe_365_count = 0;
  static  unsigned long long aesl_llvm_cbe_366_count = 0;
  static  unsigned long long aesl_llvm_cbe_367_count = 0;
  static  unsigned long long aesl_llvm_cbe_368_count = 0;
  static  unsigned long long aesl_llvm_cbe_369_count = 0;
  static  unsigned long long aesl_llvm_cbe_370_count = 0;
  static  unsigned long long aesl_llvm_cbe_371_count = 0;
  static  unsigned long long aesl_llvm_cbe_372_count = 0;
  static  unsigned long long aesl_llvm_cbe_373_count = 0;
  static  unsigned long long aesl_llvm_cbe_374_count = 0;
  static  unsigned long long aesl_llvm_cbe_375_count = 0;
  static  unsigned long long aesl_llvm_cbe_376_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @atualizar_restricao\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load float* %%x, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_219_count);
  llvm_cbe_tmp__53 = (float )*llvm_cbe_x;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__53, *(int*)(&llvm_cbe_tmp__53));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = fsub float -0.000000e+00, %%1, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_220_count);
  llvm_cbe_tmp__54 = (float )((float )(-(llvm_cbe_tmp__53)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%2, float* %%l_new, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_221_count);
  *llvm_cbe_l_new = llvm_cbe_tmp__54;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__54);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%x, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_222_count);
  llvm_cbe_tmp__55 = (float )*llvm_cbe_x;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__55, *(int*)(&llvm_cbe_tmp__55));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fsub float -0.000000e+00, %%3, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_223_count);
  llvm_cbe_tmp__56 = (float )((float )(-(llvm_cbe_tmp__55)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__56, *(int*)(&llvm_cbe_tmp__56));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%u_new, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_224_count);
  *llvm_cbe_u_new = llvm_cbe_tmp__56;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__56);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%x, i64 1, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_236_count);
  llvm_cbe_tmp__57 = (float *)(&llvm_cbe_x[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_237_count);
  llvm_cbe_tmp__58 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fsub float -0.000000e+00, %%6, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_238_count);
  llvm_cbe_tmp__59 = (float )((float )(-(llvm_cbe_tmp__58)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__59, *(int*)(&llvm_cbe_tmp__59));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds float* %%l_new, i64 1, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_239_count);
  llvm_cbe_tmp__60 = (float *)(&llvm_cbe_l_new[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%7, float* %%8, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_240_count);
  *llvm_cbe_tmp__60 = llvm_cbe_tmp__59;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__59);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* %%5, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_241_count);
  llvm_cbe_tmp__61 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__61, *(int*)(&llvm_cbe_tmp__61));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fsub float -0.000000e+00, %%9, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_242_count);
  llvm_cbe_tmp__62 = (float )((float )(-(llvm_cbe_tmp__61)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__62, *(int*)(&llvm_cbe_tmp__62));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds float* %%u_new, i64 1, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_243_count);
  llvm_cbe_tmp__63 = (float *)(&llvm_cbe_u_new[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%10, float* %%11, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_244_count);
  *llvm_cbe_tmp__63 = llvm_cbe_tmp__62;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__62);
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%x, i64 2, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_256_count);
  llvm_cbe_tmp__64 = (float *)(&llvm_cbe_x[(((signed long long )2ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* %%12, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_257_count);
  llvm_cbe_tmp__65 = (float )*llvm_cbe_tmp__64;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__65, *(int*)(&llvm_cbe_tmp__65));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fsub float -0.000000e+00, %%13, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_258_count);
  llvm_cbe_tmp__66 = (float )((float )(-(llvm_cbe_tmp__65)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__66, *(int*)(&llvm_cbe_tmp__66));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds float* %%l_new, i64 2, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_259_count);
  llvm_cbe_tmp__67 = (float *)(&llvm_cbe_l_new[(((signed long long )2ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%15, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_260_count);
  *llvm_cbe_tmp__67 = llvm_cbe_tmp__66;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__66);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load float* %%12, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_261_count);
  llvm_cbe_tmp__68 = (float )*llvm_cbe_tmp__64;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__68, *(int*)(&llvm_cbe_tmp__68));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = fsub float -0.000000e+00, %%16, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_262_count);
  llvm_cbe_tmp__69 = (float )((float )(-(llvm_cbe_tmp__68)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__69, *(int*)(&llvm_cbe_tmp__69));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds float* %%u_new, i64 2, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_263_count);
  llvm_cbe_tmp__70 = (float *)(&llvm_cbe_u_new[(((signed long long )2ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%17, float* %%18, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_264_count);
  *llvm_cbe_tmp__70 = llvm_cbe_tmp__69;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__69);
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%v00, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_276_count);
  llvm_cbe_tmp__71 = (float )*llvm_cbe_v00;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__71, *(int*)(&llvm_cbe_tmp__71));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds float* %%l_new, i64 9, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_277_count);
  llvm_cbe_tmp__72 = (float *)(&llvm_cbe_l_new[(((signed long long )9ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%19, float* %%20, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_278_count);
  *llvm_cbe_tmp__72 = llvm_cbe_tmp__71;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__71);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load float* %%v00, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_279_count);
  llvm_cbe_tmp__73 = (float )*llvm_cbe_v00;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__73, *(int*)(&llvm_cbe_tmp__73));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = getelementptr inbounds float* %%u_new, i64 9, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_280_count);
  llvm_cbe_tmp__74 = (float *)(&llvm_cbe_u_new[(((signed long long )9ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%21, float* %%22, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_281_count);
  *llvm_cbe_tmp__74 = llvm_cbe_tmp__73;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__73);
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = getelementptr inbounds float* %%v00, i64 1, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_282_count);
  llvm_cbe_tmp__75 = (float *)(&llvm_cbe_v00[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = load float* %%23, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_283_count);
  llvm_cbe_tmp__76 = (float )*llvm_cbe_tmp__75;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__76, *(int*)(&llvm_cbe_tmp__76));
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = getelementptr inbounds float* %%l_new, i64 10, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_284_count);
  llvm_cbe_tmp__77 = (float *)(&llvm_cbe_l_new[(((signed long long )10ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%24, float* %%25, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_285_count);
  *llvm_cbe_tmp__77 = llvm_cbe_tmp__76;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__76);
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load float* %%23, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_286_count);
  llvm_cbe_tmp__78 = (float )*llvm_cbe_tmp__75;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__78, *(int*)(&llvm_cbe_tmp__78));
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = getelementptr inbounds float* %%u_new, i64 10, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_287_count);
  llvm_cbe_tmp__79 = (float *)(&llvm_cbe_u_new[(((signed long long )10ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%26, float* %%27, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_288_count);
  *llvm_cbe_tmp__79 = llvm_cbe_tmp__78;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__78);
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = load float* %%5, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_300_count);
  llvm_cbe_tmp__80 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__80, *(int*)(&llvm_cbe_tmp__80));
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = fmul float %%28, 0xBFD3333340000000, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_301_count);
  llvm_cbe_tmp__81 = (float )((float )(llvm_cbe_tmp__80 * -0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__81, *(int*)(&llvm_cbe_tmp__81));
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = getelementptr inbounds float* %%l_new, i64 15, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_302_count);
  llvm_cbe_tmp__82 = (float *)(&llvm_cbe_l_new[(((signed long long )15ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%29, float* %%30, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_303_count);
  *llvm_cbe_tmp__82 = llvm_cbe_tmp__81;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__81);
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%5, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_304_count);
  llvm_cbe_tmp__83 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__83, *(int*)(&llvm_cbe_tmp__83));
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = fmul float %%31, 0x3FD3333340000000, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_305_count);
  llvm_cbe_tmp__84 = (float )((float )(llvm_cbe_tmp__83 * 0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__84, *(int*)(&llvm_cbe_tmp__84));
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = getelementptr inbounds float* %%u_new, i64 15, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_306_count);
  llvm_cbe_tmp__85 = (float *)(&llvm_cbe_u_new[(((signed long long )15ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%32, float* %%33, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_307_count);
  *llvm_cbe_tmp__85 = llvm_cbe_tmp__84;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__84);
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = load float* %%5, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_319_count);
  llvm_cbe_tmp__86 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__86, *(int*)(&llvm_cbe_tmp__86));
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = fmul float %%34, 0xBFD3333340000000, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_320_count);
  llvm_cbe_tmp__87 = (float )((float )(llvm_cbe_tmp__86 * -0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__87, *(int*)(&llvm_cbe_tmp__87));
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = getelementptr inbounds float* %%l_new, i64 16, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_321_count);
  llvm_cbe_tmp__88 = (float *)(&llvm_cbe_l_new[(((signed long long )16ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%35, float* %%36, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_322_count);
  *llvm_cbe_tmp__88 = llvm_cbe_tmp__87;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__87);
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load float* %%5, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_323_count);
  llvm_cbe_tmp__89 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__89, *(int*)(&llvm_cbe_tmp__89));
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = fmul float %%37, 0x3FD3333340000000, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_324_count);
  llvm_cbe_tmp__90 = (float )((float )(llvm_cbe_tmp__89 * 0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__90, *(int*)(&llvm_cbe_tmp__90));
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = getelementptr inbounds float* %%u_new, i64 16, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_325_count);
  llvm_cbe_tmp__91 = (float *)(&llvm_cbe_u_new[(((signed long long )16ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%38, float* %%39, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_326_count);
  *llvm_cbe_tmp__91 = llvm_cbe_tmp__90;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__90);
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = load float* %%5, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_338_count);
  llvm_cbe_tmp__92 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__92, *(int*)(&llvm_cbe_tmp__92));
if (AESL_DEBUG_TRACE)
printf("\n  %%41 = fmul float %%40, 0xBFD3333340000000, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_339_count);
  llvm_cbe_tmp__93 = (float )((float )(llvm_cbe_tmp__92 * -0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__93, *(int*)(&llvm_cbe_tmp__93));
if (AESL_DEBUG_TRACE)
printf("\n  %%42 = getelementptr inbounds float* %%l_new, i64 17, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_340_count);
  llvm_cbe_tmp__94 = (float *)(&llvm_cbe_l_new[(((signed long long )17ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%41, float* %%42, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_341_count);
  *llvm_cbe_tmp__94 = llvm_cbe_tmp__93;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__93);
if (AESL_DEBUG_TRACE)
printf("\n  %%43 = load float* %%5, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_342_count);
  llvm_cbe_tmp__95 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__95, *(int*)(&llvm_cbe_tmp__95));
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = fmul float %%43, 0x3FD3333340000000, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_343_count);
  llvm_cbe_tmp__96 = (float )((float )(llvm_cbe_tmp__95 * 0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__96, *(int*)(&llvm_cbe_tmp__96));
if (AESL_DEBUG_TRACE)
printf("\n  %%45 = getelementptr inbounds float* %%u_new, i64 17, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_344_count);
  llvm_cbe_tmp__97 = (float *)(&llvm_cbe_u_new[(((signed long long )17ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%44, float* %%45, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_345_count);
  *llvm_cbe_tmp__97 = llvm_cbe_tmp__96;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__96);
if (AESL_DEBUG_TRACE)
printf("\n  %%46 = load float* %%5, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_357_count);
  llvm_cbe_tmp__98 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__98, *(int*)(&llvm_cbe_tmp__98));
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = fmul float %%46, 0xBFD3333340000000, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_358_count);
  llvm_cbe_tmp__99 = (float )((float )(llvm_cbe_tmp__98 * -0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__99, *(int*)(&llvm_cbe_tmp__99));
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = getelementptr inbounds float* %%l_new, i64 18, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_359_count);
  llvm_cbe_tmp__100 = (float *)(&llvm_cbe_l_new[(((signed long long )18ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%47, float* %%48, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_360_count);
  *llvm_cbe_tmp__100 = llvm_cbe_tmp__99;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__99);
if (AESL_DEBUG_TRACE)
printf("\n  %%49 = load float* %%5, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_361_count);
  llvm_cbe_tmp__101 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__101, *(int*)(&llvm_cbe_tmp__101));
if (AESL_DEBUG_TRACE)
printf("\n  %%50 = fmul float %%49, 0x3FD3333340000000, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_362_count);
  llvm_cbe_tmp__102 = (float )((float )(llvm_cbe_tmp__101 * 0x1.333334p-2));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__102, *(int*)(&llvm_cbe_tmp__102));
if (AESL_DEBUG_TRACE)
printf("\n  %%51 = getelementptr inbounds float* %%u_new, i64 18, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_363_count);
  llvm_cbe_tmp__103 = (float *)(&llvm_cbe_u_new[(((signed long long )18ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%50, float* %%51, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao  --> \n", ++aesl_llvm_cbe_364_count);
  *llvm_cbe_tmp__103 = llvm_cbe_tmp__102;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__102);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @atualizar_restricao}\n");
  return;
}


void atualizar_restricao_v(float *llvm_cbe_l_new, float *llvm_cbe_u_new, float llvm_cbe_vdc, float (*llvm_cbe_Einv)[2], float *llvm_cbe_Ax) {
  static  unsigned long long aesl_llvm_cbe_v_min_count = 0;
  float llvm_cbe_v_min[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_v_max_count = 0;
  float llvm_cbe_v_max[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_u_min_count = 0;
  float llvm_cbe_u_min[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_u_max_count = 0;
  float llvm_cbe_u_max[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_377_count = 0;
  static  unsigned long long aesl_llvm_cbe_378_count = 0;
  static  unsigned long long aesl_llvm_cbe_379_count = 0;
  static  unsigned long long aesl_llvm_cbe_380_count = 0;
  static  unsigned long long aesl_llvm_cbe_381_count = 0;
  static  unsigned long long aesl_llvm_cbe_382_count = 0;
  static  unsigned long long aesl_llvm_cbe_383_count = 0;
  static  unsigned long long aesl_llvm_cbe_384_count = 0;
  static  unsigned long long aesl_llvm_cbe_385_count = 0;
  static  unsigned long long aesl_llvm_cbe_386_count = 0;
  static  unsigned long long aesl_llvm_cbe_387_count = 0;
  static  unsigned long long aesl_llvm_cbe_388_count = 0;
  static  unsigned long long aesl_llvm_cbe_389_count = 0;
  static  unsigned long long aesl_llvm_cbe_390_count = 0;
  static  unsigned long long aesl_llvm_cbe_391_count = 0;
  static  unsigned long long aesl_llvm_cbe_392_count = 0;
  static  unsigned long long aesl_llvm_cbe_393_count = 0;
  static  unsigned long long aesl_llvm_cbe_394_count = 0;
  static  unsigned long long aesl_llvm_cbe_395_count = 0;
  static  unsigned long long aesl_llvm_cbe_396_count = 0;
  static  unsigned long long aesl_llvm_cbe_397_count = 0;
  static  unsigned long long aesl_llvm_cbe_398_count = 0;
  static  unsigned long long aesl_llvm_cbe_399_count = 0;
  static  unsigned long long aesl_llvm_cbe_400_count = 0;
  static  unsigned long long aesl_llvm_cbe_401_count = 0;
  static  unsigned long long aesl_llvm_cbe_402_count = 0;
  static  unsigned long long aesl_llvm_cbe_403_count = 0;
  static  unsigned long long aesl_llvm_cbe_404_count = 0;
  float llvm_cbe_tmp__104;
  static  unsigned long long aesl_llvm_cbe_405_count = 0;
  float *llvm_cbe_tmp__105;
  static  unsigned long long aesl_llvm_cbe_406_count = 0;
  static  unsigned long long aesl_llvm_cbe_407_count = 0;
  float *llvm_cbe_tmp__106;
  static  unsigned long long aesl_llvm_cbe_408_count = 0;
  static  unsigned long long aesl_llvm_cbe_409_count = 0;
  float llvm_cbe_tmp__107;
  static  unsigned long long aesl_llvm_cbe_410_count = 0;
  float *llvm_cbe_tmp__108;
  static  unsigned long long aesl_llvm_cbe_411_count = 0;
  static  unsigned long long aesl_llvm_cbe_412_count = 0;
  float *llvm_cbe_tmp__109;
  static  unsigned long long aesl_llvm_cbe_413_count = 0;
  static  unsigned long long aesl_llvm_cbe_414_count = 0;
  float *llvm_cbe_tmp__110;
  static  unsigned long long aesl_llvm_cbe_415_count = 0;
  static  unsigned long long aesl_llvm_cbe_416_count = 0;
  float *llvm_cbe_tmp__111;
  static  unsigned long long aesl_llvm_cbe_417_count = 0;
  static  unsigned long long aesl_llvm_cbe_418_count = 0;
  float llvm_cbe_tmp__112;
  static  unsigned long long aesl_llvm_cbe_419_count = 0;
  float *llvm_cbe_tmp__113;
  static  unsigned long long aesl_llvm_cbe_420_count = 0;
  static  unsigned long long aesl_llvm_cbe_421_count = 0;
  float llvm_cbe_tmp__114;
  static  unsigned long long aesl_llvm_cbe_422_count = 0;
  float *llvm_cbe_tmp__115;
  static  unsigned long long aesl_llvm_cbe_423_count = 0;
  static  unsigned long long aesl_llvm_cbe_424_count = 0;
  float *llvm_cbe_tmp__116;
  static  unsigned long long aesl_llvm_cbe_425_count = 0;
  float llvm_cbe_tmp__117;
  static  unsigned long long aesl_llvm_cbe_426_count = 0;
  float *llvm_cbe_tmp__118;
  static  unsigned long long aesl_llvm_cbe_427_count = 0;
  static  unsigned long long aesl_llvm_cbe_428_count = 0;
  float *llvm_cbe_tmp__119;
  static  unsigned long long aesl_llvm_cbe_429_count = 0;
  float llvm_cbe_tmp__120;
  static  unsigned long long aesl_llvm_cbe_430_count = 0;
  float *llvm_cbe_tmp__121;
  static  unsigned long long aesl_llvm_cbe_431_count = 0;
  static  unsigned long long aesl_llvm_cbe_432_count = 0;
  float *llvm_cbe_tmp__122;
  static  unsigned long long aesl_llvm_cbe_433_count = 0;
  static  unsigned long long aesl_llvm_cbe_434_count = 0;
  float *llvm_cbe_tmp__123;
  static  unsigned long long aesl_llvm_cbe_435_count = 0;
  static  unsigned long long aesl_llvm_cbe_436_count = 0;
  float *llvm_cbe_tmp__124;
  static  unsigned long long aesl_llvm_cbe_437_count = 0;
  static  unsigned long long aesl_llvm_cbe_438_count = 0;
  float *llvm_cbe_tmp__125;
  static  unsigned long long aesl_llvm_cbe_439_count = 0;
  static  unsigned long long aesl_llvm_cbe_440_count = 0;
  unsigned long long llvm_cbe_tmp__126;
  static  unsigned long long aesl_llvm_cbe_441_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @atualizar_restricao_v\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = fmul float %%vdc, -5.000000e-01, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_404_count);
  llvm_cbe_tmp__104 = (float )((float )(llvm_cbe_vdc * -0x1p-1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__104, *(int*)(&llvm_cbe_tmp__104));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds [2 x float]* %%u_min, i64 0, i64 0, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_405_count);
  llvm_cbe_tmp__105 = (float *)(&llvm_cbe_u_min[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'u_min' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%1, float* %%2, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_406_count);
  *llvm_cbe_tmp__105 = llvm_cbe_tmp__104;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__104);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds [2 x float]* %%u_min, i64 0, i64 1, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_407_count);
  llvm_cbe_tmp__106 = (float *)(&llvm_cbe_u_min[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'u_min' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%1, float* %%3, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_408_count);
  *llvm_cbe_tmp__106 = llvm_cbe_tmp__104;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__104);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fmul float %%vdc, 5.000000e-01, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_409_count);
  llvm_cbe_tmp__107 = (float )((float )(llvm_cbe_vdc * 0x1p-1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__107, *(int*)(&llvm_cbe_tmp__107));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds [2 x float]* %%u_max, i64 0, i64 0, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_410_count);
  llvm_cbe_tmp__108 = (float *)(&llvm_cbe_u_max[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'u_max' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%5, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_411_count);
  *llvm_cbe_tmp__108 = llvm_cbe_tmp__107;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__107);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds [2 x float]* %%u_max, i64 0, i64 1, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_412_count);
  llvm_cbe_tmp__109 = (float *)(&llvm_cbe_u_max[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'u_max' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%6, align 4, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_413_count);
  *llvm_cbe_tmp__109 = llvm_cbe_tmp__107;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__107);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [2 x float]* %%v_min, i64 0, i64 0, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_414_count);
  llvm_cbe_tmp__110 = (float *)(&llvm_cbe_v_min[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @calculateV(float* %%2, [2 x float]* %%Einv, float* %%Ax, float* %%7), !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_415_count);
  calculateV((float *)llvm_cbe_tmp__105, llvm_cbe_Einv, (float *)llvm_cbe_Ax, (float *)llvm_cbe_tmp__110);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds [2 x float]* %%v_max, i64 0, i64 0, !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_416_count);
  llvm_cbe_tmp__111 = (float *)(&llvm_cbe_v_max[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @calculateV(float* %%5, [2 x float]* %%Einv, float* %%Ax, float* %%8), !dbg !9 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_417_count);
  calculateV((float *)llvm_cbe_tmp__108, llvm_cbe_Einv, (float *)llvm_cbe_Ax, (float *)llvm_cbe_tmp__111);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'v_min' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* %%7, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_418_count);
  llvm_cbe_tmp__112 = (float )*llvm_cbe_tmp__110;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__112, *(int*)(&llvm_cbe_tmp__112));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds float* %%l_new, i64 11, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_419_count);
  llvm_cbe_tmp__113 = (float *)(&llvm_cbe_l_new[(((signed long long )11ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%10, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_420_count);
  *llvm_cbe_tmp__113 = llvm_cbe_tmp__112;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__112);

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'v_max' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load float* %%8, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_421_count);
  llvm_cbe_tmp__114 = (float )*llvm_cbe_tmp__111;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__114, *(int*)(&llvm_cbe_tmp__114));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%u_new, i64 11, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_422_count);
  llvm_cbe_tmp__115 = (float *)(&llvm_cbe_u_new[(((signed long long )11ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%12, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_423_count);
  *llvm_cbe_tmp__115 = llvm_cbe_tmp__114;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__114);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [2 x float]* %%v_min, i64 0, i64 1, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_424_count);
  llvm_cbe_tmp__116 = (float *)(&llvm_cbe_v_min[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'v_min' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = load float* %%13, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_425_count);
  llvm_cbe_tmp__117 = (float )*llvm_cbe_tmp__116;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__117, *(int*)(&llvm_cbe_tmp__117));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds float* %%l_new, i64 12, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_426_count);
  llvm_cbe_tmp__118 = (float *)(&llvm_cbe_l_new[(((signed long long )12ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%15, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_427_count);
  *llvm_cbe_tmp__118 = llvm_cbe_tmp__117;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__117);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [2 x float]* %%v_max, i64 0, i64 1, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_428_count);
  llvm_cbe_tmp__119 = (float *)(&llvm_cbe_v_max[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'v_max' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load float* %%16, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_429_count);
  llvm_cbe_tmp__120 = (float )*llvm_cbe_tmp__119;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__120, *(int*)(&llvm_cbe_tmp__120));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds float* %%u_new, i64 12, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_430_count);
  llvm_cbe_tmp__121 = (float *)(&llvm_cbe_u_new[(((signed long long )12ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%17, float* %%18, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_431_count);
  *llvm_cbe_tmp__121 = llvm_cbe_tmp__120;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__120);
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = getelementptr inbounds float* %%l_new, i64 13, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_432_count);
  llvm_cbe_tmp__122 = (float *)(&llvm_cbe_l_new[(((signed long long )13ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%19, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_433_count);
  *llvm_cbe_tmp__122 = llvm_cbe_tmp__112;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__112);
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds float* %%u_new, i64 13, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_434_count);
  llvm_cbe_tmp__123 = (float *)(&llvm_cbe_u_new[(((signed long long )13ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%20, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_435_count);
  *llvm_cbe_tmp__123 = llvm_cbe_tmp__114;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__114);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds float* %%l_new, i64 14, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_436_count);
  llvm_cbe_tmp__124 = (float *)(&llvm_cbe_l_new[(((signed long long )14ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%21, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_437_count);
  *llvm_cbe_tmp__124 = llvm_cbe_tmp__117;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__117);
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = getelementptr inbounds float* %%u_new, i64 14, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_438_count);
  llvm_cbe_tmp__125 = (float *)(&llvm_cbe_u_new[(((signed long long )14ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%17, float* %%22, align 4, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_439_count);
  *llvm_cbe_tmp__125 = llvm_cbe_tmp__120;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__120);
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = call i64 @osqp_update_bounds(float* %%l_new, float* %%u_new) nounwind, !dbg !8 for 0x%I64xth hint within @atualizar_restricao_v  --> \n", ++aesl_llvm_cbe_440_count);
  osqp_update_bounds((float *)llvm_cbe_l_new, (float *)llvm_cbe_u_new);
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__126);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @atualizar_restricao_v}\n");
  return;
}


void atualizar_A(float (*llvm_cbe_Einv)[2]) {
  static  unsigned long long aesl_llvm_cbe_A_new_count = 0;
  float llvm_cbe_A_new[12];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_A_idx_count = 0;
  signed long long llvm_cbe_A_idx[12];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_442_count = 0;
  static  unsigned long long aesl_llvm_cbe_443_count = 0;
  static  unsigned long long aesl_llvm_cbe_444_count = 0;
  static  unsigned long long aesl_llvm_cbe_445_count = 0;
  static  unsigned long long aesl_llvm_cbe_446_count = 0;
  static  unsigned long long aesl_llvm_cbe_447_count = 0;
  static  unsigned long long aesl_llvm_cbe_448_count = 0;
  static  unsigned long long aesl_llvm_cbe_449_count = 0;
  static  unsigned long long aesl_llvm_cbe_450_count = 0;
  static  unsigned long long aesl_llvm_cbe_451_count = 0;
  static  unsigned long long aesl_llvm_cbe_452_count = 0;
  static  unsigned long long aesl_llvm_cbe_453_count = 0;
  static  unsigned long long aesl_llvm_cbe_454_count = 0;
  static  unsigned long long aesl_llvm_cbe_455_count = 0;
  static  unsigned long long aesl_llvm_cbe_456_count = 0;
  float *llvm_cbe_tmp__127;
  static  unsigned long long aesl_llvm_cbe_457_count = 0;
  float *llvm_cbe_tmp__128;
  static  unsigned long long aesl_llvm_cbe_458_count = 0;
  float llvm_cbe_tmp__129;
  static  unsigned long long aesl_llvm_cbe_459_count = 0;
  static  unsigned long long aesl_llvm_cbe_460_count = 0;
  float *llvm_cbe_tmp__130;
  static  unsigned long long aesl_llvm_cbe_461_count = 0;
  float *llvm_cbe_tmp__131;
  static  unsigned long long aesl_llvm_cbe_462_count = 0;
  float llvm_cbe_tmp__132;
  static  unsigned long long aesl_llvm_cbe_463_count = 0;
  static  unsigned long long aesl_llvm_cbe_464_count = 0;
  float *llvm_cbe_tmp__133;
  static  unsigned long long aesl_llvm_cbe_465_count = 0;
  static  unsigned long long aesl_llvm_cbe_466_count = 0;
  float *llvm_cbe_tmp__134;
  static  unsigned long long aesl_llvm_cbe_467_count = 0;
  float llvm_cbe_tmp__135;
  static  unsigned long long aesl_llvm_cbe_468_count = 0;
  static  unsigned long long aesl_llvm_cbe_469_count = 0;
  float *llvm_cbe_tmp__136;
  static  unsigned long long aesl_llvm_cbe_470_count = 0;
  static  unsigned long long aesl_llvm_cbe_471_count = 0;
  float *llvm_cbe_tmp__137;
  static  unsigned long long aesl_llvm_cbe_472_count = 0;
  static  unsigned long long aesl_llvm_cbe_473_count = 0;
  float *llvm_cbe_tmp__138;
  static  unsigned long long aesl_llvm_cbe_474_count = 0;
  float llvm_cbe_tmp__139;
  static  unsigned long long aesl_llvm_cbe_475_count = 0;
  static  unsigned long long aesl_llvm_cbe_476_count = 0;
  float *llvm_cbe_tmp__140;
  static  unsigned long long aesl_llvm_cbe_477_count = 0;
  static  unsigned long long aesl_llvm_cbe_478_count = 0;
  float *llvm_cbe_tmp__141;
  static  unsigned long long aesl_llvm_cbe_479_count = 0;
  static  unsigned long long aesl_llvm_cbe_480_count = 0;
  float *llvm_cbe_tmp__142;
  static  unsigned long long aesl_llvm_cbe_481_count = 0;
  static  unsigned long long aesl_llvm_cbe_482_count = 0;
  float *llvm_cbe_tmp__143;
  static  unsigned long long aesl_llvm_cbe_483_count = 0;
  static  unsigned long long aesl_llvm_cbe_484_count = 0;
  float *llvm_cbe_tmp__144;
  static  unsigned long long aesl_llvm_cbe_485_count = 0;
  static  unsigned long long aesl_llvm_cbe_486_count = 0;
  static  unsigned long long aesl_llvm_cbe_487_count = 0;
   char *llvm_cbe_tmp__145;
  static  unsigned long long aesl_llvm_cbe_488_count = 0;
   char *llvm_cbe_tmp__146;
  static  unsigned long long aesl_llvm_cbe_489_count = 0;
  signed long long *llvm_cbe_tmp__147;
  static  unsigned long long aesl_llvm_cbe_490_count = 0;
  unsigned long long llvm_cbe_tmp__148;
  static  unsigned long long aesl_llvm_cbe_491_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @atualizar_A\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 0, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_456_count);
  llvm_cbe_tmp__127 = (float *)(&llvm_cbe_A_new[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds [2 x float]* %%Einv, i64 0, i64 0, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_457_count);
  llvm_cbe_tmp__128 = (float *)(&(*llvm_cbe_Einv)[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Einv' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_458_count);
  llvm_cbe_tmp__129 = (float )*llvm_cbe_tmp__128;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__129, *(int*)(&llvm_cbe_tmp__129));

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%1, align 16, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_459_count);
  *llvm_cbe_tmp__127 = llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__129);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 1, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_460_count);
  llvm_cbe_tmp__130 = (float *)(&llvm_cbe_A_new[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds [2 x float]* %%Einv, i64 0, i64 1, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_461_count);
  llvm_cbe_tmp__131 = (float *)(&(*llvm_cbe_Einv)[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Einv' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_462_count);
  llvm_cbe_tmp__132 = (float )*llvm_cbe_tmp__131;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__132, *(int*)(&llvm_cbe_tmp__132));

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%4, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_463_count);
  *llvm_cbe_tmp__130 = llvm_cbe_tmp__132;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__132);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 2, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_464_count);
  llvm_cbe_tmp__133 = (float *)(&llvm_cbe_A_new[(((signed long long )2ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )2ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%7, align 8, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_465_count);
  *llvm_cbe_tmp__133 = llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__129);
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 3, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_466_count);
  llvm_cbe_tmp__134 = (float *)(&llvm_cbe_A_new[(((signed long long )3ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = fsub float -0.000000e+00, %%3, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_467_count);
  llvm_cbe_tmp__135 = (float )((float )(-(llvm_cbe_tmp__129)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__135, *(int*)(&llvm_cbe_tmp__135));

#ifdef AESL_BC_SIM
  assert(((signed long long )3ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%8, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_468_count);
  *llvm_cbe_tmp__134 = llvm_cbe_tmp__135;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__135);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_469_count);
  llvm_cbe_tmp__136 = (float *)(&llvm_cbe_A_new[(((signed long long )4ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )4ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%10, align 16, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_470_count);
  *llvm_cbe_tmp__136 = llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__129);
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 5, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_471_count);
  llvm_cbe_tmp__137 = (float *)(&llvm_cbe_A_new[(((signed long long )5ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )5ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%11, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_472_count);
  *llvm_cbe_tmp__137 = llvm_cbe_tmp__132;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__132);
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 6, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_473_count);
  llvm_cbe_tmp__138 = (float *)(&llvm_cbe_A_new[(((signed long long )6ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fsub float -0.000000e+00, %%6, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_474_count);
  llvm_cbe_tmp__139 = (float )((float )(-(llvm_cbe_tmp__132)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__139, *(int*)(&llvm_cbe_tmp__139));

#ifdef AESL_BC_SIM
  assert(((signed long long )6ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%12, align 8, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_475_count);
  *llvm_cbe_tmp__138 = llvm_cbe_tmp__139;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__139);
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 7, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_476_count);
  llvm_cbe_tmp__140 = (float *)(&llvm_cbe_A_new[(((signed long long )7ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )7ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%14, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_477_count);
  *llvm_cbe_tmp__140 = llvm_cbe_tmp__132;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__132);
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 8, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_478_count);
  llvm_cbe_tmp__141 = (float *)(&llvm_cbe_A_new[(((signed long long )8ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )8ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%15, align 16, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_479_count);
  *llvm_cbe_tmp__141 = llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__129);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 9, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_480_count);
  llvm_cbe_tmp__142 = (float *)(&llvm_cbe_A_new[(((signed long long )9ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )9ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%16, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_481_count);
  *llvm_cbe_tmp__142 = llvm_cbe_tmp__135;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__135);
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 10, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_482_count);
  llvm_cbe_tmp__143 = (float *)(&llvm_cbe_A_new[(((signed long long )10ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )10ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%17, align 8, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_483_count);
  *llvm_cbe_tmp__143 = llvm_cbe_tmp__132;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__132);
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds [12 x float]* %%A_new, i64 0, i64 11, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_484_count);
  llvm_cbe_tmp__144 = (float *)(&llvm_cbe_A_new[(((signed long long )11ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )11ull) < 12 && "Write access out of array 'A_new' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%18, align 4, !dbg !7 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_485_count);
  *llvm_cbe_tmp__144 = llvm_cbe_tmp__139;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__139);
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = bitcast [12 x i64]* %%A_idx to i8*, !dbg !8 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_487_count);
  llvm_cbe_tmp__145 = ( char *)(( char *)(&llvm_cbe_A_idx));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = call i8* @memcpy(i8* %%19, i8* bitcast ([12 x i64]* @aesl_internal_atualizar_A.A_idx to i8*), i64 96 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_488_count);
  ( char *)memcpy(( char *)llvm_cbe_tmp__145, ( char *)(( char *)(&aesl_internal_atualizar_A_OC_A_idx)), 96ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",96ull);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__146);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds [12 x i64]* %%A_idx, i64 0, i64 0, !dbg !8 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_489_count);
  llvm_cbe_tmp__147 = (signed long long *)(&llvm_cbe_A_idx[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = call i64 @osqp_update_A(float* %%1, i64* %%21, i64 12) nounwind, !dbg !8 for 0x%I64xth hint within @atualizar_A  --> \n", ++aesl_llvm_cbe_490_count);
  osqp_update_A((float *)llvm_cbe_tmp__127, (signed long long *)llvm_cbe_tmp__147, 12ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",12ull);
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__148);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @atualizar_A}\n");
  return;
}

