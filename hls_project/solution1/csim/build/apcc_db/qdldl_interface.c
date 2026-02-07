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
typedef struct l_struct_OC_qdldl l_struct_OC_qdldl;
typedef struct l_struct_OC_csc l_struct_OC_csc;

/* Structure contents */
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

struct l_struct_OC_csc {
  unsigned long long field0;
  unsigned long long field1;
  unsigned long long field2;
  signed long long *field3;
  signed long long *field4;
  float *field5;
  unsigned long long field6;
};


/* External Global Variable Declarations */
extern float linsys_solver_sol[34];
extern signed long long linsys_solver_L_p[35];
extern signed long long linsys_solver_L_i[57];
extern float linsys_solver_L_x[57];
extern float linsys_solver_Dinv[34];
extern signed long long linsys_solver_P[34];
extern float linsys_solver_bp[34];
extern float linsys_solver_rho_inv_vec[19];
extern float linsys_solver_KKT_x[79];
extern float Pdata_x[12];
extern signed long long Pdata_p[16];
extern signed long long linsys_solver_PtoKKT[12];
extern l_struct_OC_qdldl linsys_solver;
extern signed long long linsys_solver_Pdiag_idx[10];
extern float Adata_x[43];
extern signed long long Adata_p[16];
extern signed long long linsys_solver_AtoKKT[43];
extern signed long long linsys_solver_KKT_p[35];
extern signed long long linsys_solver_KKT_i[79];
extern float linsys_solver_D[34];
extern signed long long linsys_solver_Lnz[34];
extern signed long long linsys_solver_etree[34];
extern signed long long linsys_solver_bwork[34];
extern signed long long linsys_solver_iwork[102];
extern float linsys_solver_fwork[34];
extern signed long long linsys_solver_rhotoKKT[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
signed long long init_linsys_solver_qdldl(void);
void permute_x(signed long long llvm_cbe_n, float *llvm_cbe_x, float *llvm_cbe_b, signed long long *llvm_cbe_P);
void permutet_x(signed long long llvm_cbe_n, float *llvm_cbe_x, float *llvm_cbe_b, signed long long *llvm_cbe_P);
signed long long solve_linsys_qdldl(float *llvm_cbe_b);
static void aesl_internal_LDLSolve(float *llvm_cbe_b);
signed long long update_linsys_solver_matrices_qdldl(void);
void update_KKT_P(float *, float *, signed long long *, signed long long , signed long long *, float , signed long long *, signed long long );
void update_KKT_A(float *, float *, signed long long *, signed long long , signed long long *);
signed long long QDLDL_factor(signed long long , signed long long *, signed long long *, float *, signed long long *, signed long long *, float *, float *, float *, signed long long *, signed long long *, signed long long *, signed long long *, float *);
signed long long update_linsys_solver_rho_vec_qdldl(float *llvm_cbe_rho_vec);
void update_KKT_param2(float *, float *, signed long long *, signed long long );
void QDLDL_solve(signed long long , signed long long *, signed long long *, float *, float *, float *);


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

signed long long init_linsys_solver_qdldl(void) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @init_linsys_solver_qdldl\n");
  if (AESL_DEBUG_TRACE)
      printf("\nEND @init_linsys_solver_qdldl}\n");
  return 0ull;
}


void permute_x(signed long long llvm_cbe_n, float *llvm_cbe_x, float *llvm_cbe_b, signed long long *llvm_cbe_P) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  signed long long *llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  unsigned long long llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  float *llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  float llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  float *llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  unsigned long long llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @permute_x\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%7, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__6);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds i64* %%P, i64 %%storemerge1, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_17_count);
  llvm_cbe_tmp__1 = (signed long long *)(&llvm_cbe_P[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_18_count);
  llvm_cbe_tmp__2 = (unsigned long long )*llvm_cbe_tmp__1;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__2);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%b, i64 %%3, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__3 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_tmp__2))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__2));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* %%4, align 4, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_20_count);
  llvm_cbe_tmp__4 = (float )*llvm_cbe_tmp__3;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__4, *(int*)(&llvm_cbe_tmp__4));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds float* %%x, i64 %%storemerge1, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_21_count);
  llvm_cbe_tmp__5 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%5, float* %%6, align 4, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_22_count);
  *llvm_cbe_tmp__5 = llvm_cbe_tmp__4;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__4);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = add nsw i64 %%storemerge1, 1, !dbg !33 for 0x%I64xth hint within @permute_x  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__6 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__6&18446744073709551615ull)));
  if (((llvm_cbe_tmp__6&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__6;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @permute_x}\n");
  return;
}


void permutet_x(signed long long llvm_cbe_n, float *llvm_cbe_x, float *llvm_cbe_b, signed long long *llvm_cbe_P) {
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;
  static  unsigned long long aesl_llvm_cbe_36_count = 0;
  static  unsigned long long aesl_llvm_cbe_37_count = 0;
  static  unsigned long long aesl_llvm_cbe_38_count = 0;
  static  unsigned long long aesl_llvm_cbe_39_count = 0;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;
  static  unsigned long long aesl_llvm_cbe_41_count = 0;
  static  unsigned long long aesl_llvm_cbe_42_count = 0;
  static  unsigned long long aesl_llvm_cbe_43_count = 0;
  static  unsigned long long aesl_llvm_cbe_44_count = 0;
  static  unsigned long long aesl_llvm_cbe_45_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  float llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  signed long long *llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  unsigned long long llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  float *llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_51_count = 0;
  static  unsigned long long aesl_llvm_cbe_52_count = 0;
  unsigned long long llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_53_count = 0;
  static  unsigned long long aesl_llvm_cbe_54_count = 0;
  static  unsigned long long aesl_llvm_cbe_55_count = 0;
  static  unsigned long long aesl_llvm_cbe_56_count = 0;
  static  unsigned long long aesl_llvm_cbe_57_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_58_count = 0;
  static  unsigned long long aesl_llvm_cbe_59_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @permutet_x\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%7, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__12);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_46_count);
  llvm_cbe_tmp__7 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_47_count);
  llvm_cbe_tmp__8 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__8, *(int*)(&llvm_cbe_tmp__8));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds i64* %%P, i64 %%storemerge1, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_48_count);
  llvm_cbe_tmp__9 = (signed long long *)(&llvm_cbe_P[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load i64* %%4, align 8, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_49_count);
  llvm_cbe_tmp__10 = (unsigned long long )*llvm_cbe_tmp__9;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds float* %%x, i64 %%5, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_50_count);
  llvm_cbe_tmp__11 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_tmp__10))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__10));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%6, align 4, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_51_count);
  *llvm_cbe_tmp__11 = llvm_cbe_tmp__8;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__8);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = add nsw i64 %%storemerge1, 1, !dbg !33 for 0x%I64xth hint within @permutet_x  --> \n", ++aesl_llvm_cbe_52_count);
  llvm_cbe_tmp__12 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__12&18446744073709551615ull)));
  if (((llvm_cbe_tmp__12&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__12;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @permutet_x}\n");
  return;
}


signed long long solve_linsys_qdldl(float *llvm_cbe_b) {
  static  unsigned long long aesl_llvm_cbe_60_count = 0;
  static  unsigned long long aesl_llvm_cbe_61_count = 0;
  static  unsigned long long aesl_llvm_cbe_62_count = 0;
  static  unsigned long long aesl_llvm_cbe_63_count = 0;
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  float llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  float llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  float *llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;
  static  unsigned long long aesl_llvm_cbe_92_count = 0;
  static  unsigned long long aesl_llvm_cbe_93_count = 0;
  static  unsigned long long aesl_llvm_cbe_94_count = 0;
  static  unsigned long long aesl_llvm_cbe_95_count = 0;
  static  unsigned long long aesl_llvm_cbe_96_count = 0;
  static  unsigned long long aesl_llvm_cbe_97_count = 0;
  static  unsigned long long aesl_llvm_cbe_98_count = 0;
  static  unsigned long long aesl_llvm_cbe_99_count = 0;
  static  unsigned long long aesl_llvm_cbe_100_count = 0;
  float llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_101_count = 0;
  float *llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  float llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  float *llvm_cbe_tmp__19;
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
  float llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;
  float *llvm_cbe_tmp__21;
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
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  float llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
  float *llvm_cbe_tmp__23;
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
  float llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  float *llvm_cbe_tmp__25;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  float llvm_cbe_tmp__26;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  float *llvm_cbe_tmp__27;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  static  unsigned long long aesl_llvm_cbe_175_count = 0;
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
  static  unsigned long long aesl_llvm_cbe_177_count = 0;
  static  unsigned long long aesl_llvm_cbe_178_count = 0;
  float llvm_cbe_tmp__28;
  static  unsigned long long aesl_llvm_cbe_179_count = 0;
  float *llvm_cbe_tmp__29;
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
  static  unsigned long long aesl_llvm_cbe_190_count = 0;
  static  unsigned long long aesl_llvm_cbe_191_count = 0;
  float llvm_cbe_tmp__30;
  static  unsigned long long aesl_llvm_cbe_192_count = 0;
  float *llvm_cbe_tmp__31;
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
  float llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  float *llvm_cbe_tmp__33;
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
  float llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_218_count = 0;
  float *llvm_cbe_tmp__35;
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
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_231_count = 0;
  float *llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
  float llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_244_count = 0;
  float *llvm_cbe_tmp__39;
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
  float llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
  float *llvm_cbe_tmp__41;
  static  unsigned long long aesl_llvm_cbe_258_count = 0;
  static  unsigned long long aesl_llvm_cbe_259_count = 0;
  static  unsigned long long aesl_llvm_cbe_260_count = 0;
  static  unsigned long long aesl_llvm_cbe_261_count = 0;
  static  unsigned long long aesl_llvm_cbe_262_count = 0;
  static  unsigned long long aesl_llvm_cbe_263_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge12_count = 0;
  unsigned long long llvm_cbe_storemerge12;
  unsigned long long llvm_cbe_storemerge12__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  float *llvm_cbe_tmp__42;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  float llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  unsigned long long llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  float *llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  float llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  float llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;
  float *llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  float llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
  float llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_288_count = 0;
  static  unsigned long long aesl_llvm_cbe_289_count = 0;
  unsigned long long llvm_cbe_tmp__51;
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
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_300_count = 0;
  static  unsigned long long aesl_llvm_cbe_301_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @solve_linsys_qdldl\n");
if (AESL_DEBUG_TRACE)
printf("\n  tail call fastcc void @aesl_internal_LDLSolve(float* %%b), !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_64_count);
   /*tail*/ aesl_internal_LDLSolve((float *)llvm_cbe_b);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%0 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 0), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_75_count);
  llvm_cbe_tmp__13 = (float )*((&linsys_solver_sol[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__13, *(int*)(&llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%0, float* %%b, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_76_count);
  *llvm_cbe_b = llvm_cbe_tmp__13;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__13);
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 1), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_87_count);
  llvm_cbe_tmp__14 = (float )*((&linsys_solver_sol[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__14, *(int*)(&llvm_cbe_tmp__14));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%b, i64 1, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_88_count);
  llvm_cbe_tmp__15 = (float *)(&llvm_cbe_b[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%1, float* %%2, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_89_count);
  *llvm_cbe_tmp__15 = llvm_cbe_tmp__14;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__14);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 2), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_100_count);
  llvm_cbe_tmp__16 = (float )*((&linsys_solver_sol[(((signed long long )2ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__16, *(int*)(&llvm_cbe_tmp__16));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%b, i64 2, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_101_count);
  llvm_cbe_tmp__17 = (float *)(&llvm_cbe_b[(((signed long long )2ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%4, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_102_count);
  *llvm_cbe_tmp__17 = llvm_cbe_tmp__16;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 3), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_113_count);
  llvm_cbe_tmp__18 = (float )*((&linsys_solver_sol[(((signed long long )3ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds float* %%b, i64 3, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_114_count);
  llvm_cbe_tmp__19 = (float *)(&llvm_cbe_b[(((signed long long )3ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%5, float* %%6, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_115_count);
  *llvm_cbe_tmp__19 = llvm_cbe_tmp__18;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__18);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 4), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_126_count);
  llvm_cbe_tmp__20 = (float )*((&linsys_solver_sol[(((signed long long )4ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__20, *(int*)(&llvm_cbe_tmp__20));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds float* %%b, i64 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_127_count);
  llvm_cbe_tmp__21 = (float *)(&llvm_cbe_b[(((signed long long )4ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%7, float* %%8, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_128_count);
  *llvm_cbe_tmp__21 = llvm_cbe_tmp__20;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__20);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 5), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_139_count);
  llvm_cbe_tmp__22 = (float )*((&linsys_solver_sol[(((signed long long )5ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__22, *(int*)(&llvm_cbe_tmp__22));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds float* %%b, i64 5, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_140_count);
  llvm_cbe_tmp__23 = (float *)(&llvm_cbe_b[(((signed long long )5ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%9, float* %%10, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_141_count);
  *llvm_cbe_tmp__23 = llvm_cbe_tmp__22;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__22);
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 6), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_152_count);
  llvm_cbe_tmp__24 = (float )*((&linsys_solver_sol[(((signed long long )6ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__24, *(int*)(&llvm_cbe_tmp__24));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%b, i64 6, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_153_count);
  llvm_cbe_tmp__25 = (float *)(&llvm_cbe_b[(((signed long long )6ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%12, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_154_count);
  *llvm_cbe_tmp__25 = llvm_cbe_tmp__24;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__24);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 7), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_165_count);
  llvm_cbe_tmp__26 = (float )*((&linsys_solver_sol[(((signed long long )7ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__26, *(int*)(&llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds float* %%b, i64 7, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_166_count);
  llvm_cbe_tmp__27 = (float *)(&llvm_cbe_b[(((signed long long )7ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%14, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_167_count);
  *llvm_cbe_tmp__27 = llvm_cbe_tmp__26;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__26);
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 8), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_178_count);
  llvm_cbe_tmp__28 = (float )*((&linsys_solver_sol[(((signed long long )8ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__28, *(int*)(&llvm_cbe_tmp__28));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds float* %%b, i64 8, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_179_count);
  llvm_cbe_tmp__29 = (float *)(&llvm_cbe_b[(((signed long long )8ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%15, float* %%16, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_180_count);
  *llvm_cbe_tmp__29 = llvm_cbe_tmp__28;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__28);
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 9), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_191_count);
  llvm_cbe_tmp__30 = (float )*((&linsys_solver_sol[(((signed long long )9ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__30, *(int*)(&llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds float* %%b, i64 9, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_192_count);
  llvm_cbe_tmp__31 = (float *)(&llvm_cbe_b[(((signed long long )9ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%17, float* %%18, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_193_count);
  *llvm_cbe_tmp__31 = llvm_cbe_tmp__30;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__30);
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 10), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_204_count);
  llvm_cbe_tmp__32 = (float )*((&linsys_solver_sol[(((signed long long )10ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__32, *(int*)(&llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds float* %%b, i64 10, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_205_count);
  llvm_cbe_tmp__33 = (float *)(&llvm_cbe_b[(((signed long long )10ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%19, float* %%20, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_206_count);
  *llvm_cbe_tmp__33 = llvm_cbe_tmp__32;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__32);
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 11), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_217_count);
  llvm_cbe_tmp__34 = (float )*((&linsys_solver_sol[(((signed long long )11ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__34, *(int*)(&llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = getelementptr inbounds float* %%b, i64 11, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_218_count);
  llvm_cbe_tmp__35 = (float *)(&llvm_cbe_b[(((signed long long )11ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%21, float* %%22, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_219_count);
  *llvm_cbe_tmp__35 = llvm_cbe_tmp__34;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__34);
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 12), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_230_count);
  llvm_cbe_tmp__36 = (float )*((&linsys_solver_sol[(((signed long long )12ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = getelementptr inbounds float* %%b, i64 12, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_231_count);
  llvm_cbe_tmp__37 = (float *)(&llvm_cbe_b[(((signed long long )12ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%23, float* %%24, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_232_count);
  *llvm_cbe_tmp__37 = llvm_cbe_tmp__36;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__36);
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 13), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_243_count);
  llvm_cbe_tmp__38 = (float )*((&linsys_solver_sol[(((signed long long )13ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__38, *(int*)(&llvm_cbe_tmp__38));
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = getelementptr inbounds float* %%b, i64 13, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_244_count);
  llvm_cbe_tmp__39 = (float *)(&llvm_cbe_b[(((signed long long )13ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%25, float* %%26, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_245_count);
  *llvm_cbe_tmp__39 = llvm_cbe_tmp__38;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__38);
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = load float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 14), align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_256_count);
  llvm_cbe_tmp__40 = (float )*((&linsys_solver_sol[(((signed long long )14ull))
#ifdef AESL_BC_SIM
 % 34
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__40, *(int*)(&llvm_cbe_tmp__40));
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = getelementptr inbounds float* %%b, i64 14, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_257_count);
  llvm_cbe_tmp__41 = (float *)(&llvm_cbe_b[(((signed long long )14ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%27, float* %%28, align 4, !dbg !32 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_258_count);
  *llvm_cbe_tmp__41 = llvm_cbe_tmp__40;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__40);
  llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__52;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__52:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge12 = phi i64 [ 0, %%.preheader ], [ %%39, %%29  for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_storemerge12_count);
  llvm_cbe_storemerge12 = (unsigned long long )llvm_cbe_storemerge12__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",llvm_cbe_storemerge12);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__51);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = getelementptr inbounds [19 x float]* @linsys_solver_rho_inv_vec, i64 0, i64 %%storemerge12, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_279_count);
  llvm_cbe_tmp__42 = (float *)(&linsys_solver_rho_inv_vec[(((signed long long )llvm_cbe_storemerge12))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge12) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'linsys_solver_rho_inv_vec' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%30, align 4, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_280_count);
  llvm_cbe_tmp__43 = (float )*llvm_cbe_tmp__42;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__43, *(int*)(&llvm_cbe_tmp__43));
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = add nsw i64 %%storemerge12, 15, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_281_count);
  llvm_cbe_tmp__44 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull)) + ((unsigned long long )(15ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__44&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = getelementptr inbounds [34 x float]* @linsys_solver_sol, i64 0, i64 %%32, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_282_count);
  llvm_cbe_tmp__45 = (float *)(&linsys_solver_sol[(((signed long long )llvm_cbe_tmp__44))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__44));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_tmp__44) < 34)) fprintf(stderr, "%s:%d: warning: Read access out of array 'linsys_solver_sol' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = load float* %%33, align 4, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_283_count);
  llvm_cbe_tmp__46 = (float )*llvm_cbe_tmp__45;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__46, *(int*)(&llvm_cbe_tmp__46));
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = fmul float %%31, %%34, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_284_count);
  llvm_cbe_tmp__47 = (float )((float )(llvm_cbe_tmp__43 * llvm_cbe_tmp__46));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__47, *(int*)(&llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = getelementptr inbounds float* %%b, i64 %%32, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_285_count);
  llvm_cbe_tmp__48 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_tmp__44))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__44));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load float* %%36, align 4, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_286_count);
  llvm_cbe_tmp__49 = (float )*llvm_cbe_tmp__48;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__49, *(int*)(&llvm_cbe_tmp__49));
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = fadd float %%37, %%35, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_287_count);
  llvm_cbe_tmp__50 = (float )((float )(llvm_cbe_tmp__49 + llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__50, *(int*)(&llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%38, float* %%36, align 4, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_288_count);
  *llvm_cbe_tmp__48 = llvm_cbe_tmp__50;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__50);
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = add nsw i64 %%storemerge12, 1, !dbg !33 for 0x%I64xth hint within @solve_linsys_qdldl  --> \n", ++aesl_llvm_cbe_289_count);
  llvm_cbe_tmp__51 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__51&18446744073709551615ull)));
  if (((llvm_cbe_tmp__51&18446744073709551615ULL) == (19ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__53;
  } else {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__51;   /* for PHI node */
    goto llvm_cbe_tmp__52;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe_tmp__53:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @solve_linsys_qdldl}\n");
  return 0ull;
}


static void aesl_internal_LDLSolve(float *llvm_cbe_b) {
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
  static  unsigned long long aesl_llvm_cbe_314_count = 0;
  static  unsigned long long aesl_llvm_cbe_315_count = 0;
  static  unsigned long long aesl_llvm_cbe_316_count = 0;
  static  unsigned long long aesl_llvm_cbe_317_count = 0;
  static  unsigned long long aesl_llvm_cbe_318_count = 0;
  static  unsigned long long aesl_llvm_cbe_319_count = 0;
  static  unsigned long long aesl_llvm_cbe_320_count = 0;
  static  unsigned long long aesl_llvm_cbe_321_count = 0;
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @aesl_internal_LDLSolve\n");
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @permute_x(i64 34, float* getelementptr inbounds ([34 x float]* @linsys_solver_bp, i64 0, i64 0), float* %%b, i64* getelementptr inbounds ([34 x i64]* @linsys_solver_P, i64 0, i64 0)), !dbg !32 for 0x%I64xth hint within @aesl_internal_LDLSolve  --> \n", ++aesl_llvm_cbe_318_count);
   /*tail*/ permute_x(34ull, (float *)((&linsys_solver_bp[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (float *)llvm_cbe_b, (signed long long *)((&linsys_solver_P[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",34ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @QDLDL_solve(i64 34, i64* getelementptr inbounds ([35 x i64]* @linsys_solver_L_p, i64 0, i64 0), i64* getelementptr inbounds ([57 x i64]* @linsys_solver_L_i, i64 0, i64 0), float* getelementptr inbounds ([57 x float]* @linsys_solver_L_x, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_Dinv, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_bp, i64 0, i64 0)) nounwind, !dbg !33 for 0x%I64xth hint within @aesl_internal_LDLSolve  --> \n", ++aesl_llvm_cbe_319_count);
   /*tail*/ QDLDL_solve(34ull, (signed long long *)((&linsys_solver_L_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 35
#endif
])), (signed long long *)((&linsys_solver_L_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_L_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (float *)((&linsys_solver_bp[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",34ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @permutet_x(i64 34, float* getelementptr inbounds ([34 x float]* @linsys_solver_sol, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_bp, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_P, i64 0, i64 0)), !dbg !32 for 0x%I64xth hint within @aesl_internal_LDLSolve  --> \n", ++aesl_llvm_cbe_320_count);
   /*tail*/ permutet_x(34ull, (float *)((&linsys_solver_sol[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (float *)((&linsys_solver_bp[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_P[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",34ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @aesl_internal_LDLSolve}\n");
  return;
}


signed long long update_linsys_solver_matrices_qdldl(void) {
  static  unsigned long long aesl_llvm_cbe_322_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_323_count = 0;
  static  unsigned long long aesl_llvm_cbe_324_count = 0;
  static  unsigned long long aesl_llvm_cbe_325_count = 0;
  unsigned long long llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_326_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_linsys_solver_matrices_qdldl\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load float* getelementptr inbounds (%%struct.qdldl* @linsys_solver, i64 0, i32 7), align 8, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_matrices_qdldl  --> \n", ++aesl_llvm_cbe_322_count);
  llvm_cbe_tmp__54 = (float )*((&linsys_solver.field7));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_KKT_P(float* getelementptr inbounds ([79 x float]* @linsys_solver_KKT_x, i64 0, i64 0), float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64 15, i64* getelementptr inbounds ([12 x i64]* @linsys_solver_PtoKKT, i64 0, i64 0), float %%1, i64* getelementptr inbounds ([10 x i64]* @linsys_solver_Pdiag_idx, i64 0, i64 0), i64 10) nounwind, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_matrices_qdldl  --> \n", ++aesl_llvm_cbe_323_count);
   /*tail*/ update_KKT_P((float *)((&linsys_solver_KKT_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (float *)((&Pdata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), (signed long long *)((&Pdata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), 15ull, (signed long long *)((&linsys_solver_PtoKKT[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), llvm_cbe_tmp__54, (signed long long *)((&linsys_solver_Pdiag_idx[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 10
#endif
])), 10ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",15ull);
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
printf("\nArgument  = 0x%I64X",10ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_KKT_A(float* getelementptr inbounds ([79 x float]* @linsys_solver_KKT_x, i64 0, i64 0), float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64 15, i64* getelementptr inbounds ([43 x i64]* @linsys_solver_AtoKKT, i64 0, i64 0)) nounwind, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_matrices_qdldl  --> \n", ++aesl_llvm_cbe_324_count);
   /*tail*/ update_KKT_A((float *)((&linsys_solver_KKT_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (float *)((&Adata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 43
#endif
])), (signed long long *)((&Adata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), 15ull, (signed long long *)((&linsys_solver_AtoKKT[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 43
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",15ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = tail call i64 @QDLDL_factor(i64 34, i64* getelementptr inbounds ([35 x i64]* @linsys_solver_KKT_p, i64 0, i64 0), i64* getelementptr inbounds ([79 x i64]* @linsys_solver_KKT_i, i64 0, i64 0), float* getelementptr inbounds ([79 x float]* @linsys_solver_KKT_x, i64 0, i64 0), i64* getelementptr inbounds ([35 x i64]* @linsys_solver_L_p, i64 0, i64 0), i64* getelementptr inbounds ([57 x i64]* @linsys_solver_L_i, i64 0, i64 0), float* getelementptr inbounds ([57 x float]* @linsys_solver_L_x, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_D, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_Dinv, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_Lnz, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_etree, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_bwork, i64 0, i64 0), i64* getelementptr inbounds ([102 x i64]* @linsys_solver_iwork, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_fwork, i64 0, i64 0)) nounwind, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_matrices_qdldl  --> \n", ++aesl_llvm_cbe_325_count);
  llvm_cbe_tmp__55 = (unsigned long long ) /*tail*/ QDLDL_factor(34ull, (signed long long *)((&linsys_solver_KKT_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 35
#endif
])), (signed long long *)((&linsys_solver_KKT_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (float *)((&linsys_solver_KKT_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (signed long long *)((&linsys_solver_L_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 35
#endif
])), (signed long long *)((&linsys_solver_L_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_L_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (float *)((&linsys_solver_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_Lnz[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_etree[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_bwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_iwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 102
#endif
])), (float *)((&linsys_solver_fwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",34ull);
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__55);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_linsys_solver_matrices_qdldl}\n");
  return llvm_cbe_tmp__55;
}


signed long long update_linsys_solver_rho_vec_qdldl(float *llvm_cbe_rho_vec) {
  static  unsigned long long aesl_llvm_cbe_327_count = 0;
  static  unsigned long long aesl_llvm_cbe_328_count = 0;
  static  unsigned long long aesl_llvm_cbe_329_count = 0;
  static  unsigned long long aesl_llvm_cbe_330_count = 0;
  static  unsigned long long aesl_llvm_cbe_331_count = 0;
  static  unsigned long long aesl_llvm_cbe_332_count = 0;
  static  unsigned long long aesl_llvm_cbe_333_count = 0;
  static  unsigned long long aesl_llvm_cbe_334_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_335_count = 0;
  float *llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_336_count = 0;
  float llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_337_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_338_count = 0;
  float *llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_339_count = 0;
  static  unsigned long long aesl_llvm_cbe_340_count = 0;
  unsigned long long llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_341_count = 0;
  static  unsigned long long aesl_llvm_cbe_342_count = 0;
  static  unsigned long long aesl_llvm_cbe_343_count = 0;
  static  unsigned long long aesl_llvm_cbe_344_count = 0;
  static  unsigned long long aesl_llvm_cbe_345_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_346_count = 0;
  static  unsigned long long aesl_llvm_cbe_347_count = 0;
  static  unsigned long long aesl_llvm_cbe_348_count = 0;
  unsigned long long llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_349_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_linsys_solver_rho_vec_qdldl\n");
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__62;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__62:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ 0, %%0 ], [ %%6, %%1  for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__60);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%rho_vec, i64 %%storemerge1, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_335_count);
  llvm_cbe_tmp__56 = (float *)(&llvm_cbe_rho_vec[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_336_count);
  llvm_cbe_tmp__57 = (float )*llvm_cbe_tmp__56;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__57, *(int*)(&llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fdiv float 1.000000e+00, %%3, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_337_count);
  llvm_cbe_tmp__58 = (float )((float )(0x1p0 / llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds [19 x float]* @linsys_solver_rho_inv_vec, i64 0, i64 %%storemerge1, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_338_count);
  llvm_cbe_tmp__59 = (float *)(&linsys_solver_rho_inv_vec[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 19 && "Write access out of array 'linsys_solver_rho_inv_vec' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%5, align 4, !dbg !32 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_339_count);
  *llvm_cbe_tmp__59 = llvm_cbe_tmp__58;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__58);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = add nsw i64 %%storemerge1, 1, !dbg !33 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_340_count);
  llvm_cbe_tmp__60 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__60&18446744073709551615ull)));
  if (((llvm_cbe_tmp__60&18446744073709551615ULL) == (19ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__63;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__60;   /* for PHI node */
    goto llvm_cbe_tmp__62;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe_tmp__63:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_KKT_param2(float* getelementptr inbounds ([79 x float]* @linsys_solver_KKT_x, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @linsys_solver_rho_inv_vec, i64 0, i64 0), i64* getelementptr inbounds ([19 x i64]* @linsys_solver_rhotoKKT, i64 0, i64 0), i64 19) nounwind, !dbg !33 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_347_count);
   /*tail*/ update_KKT_param2((float *)((&linsys_solver_KKT_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (float *)((&linsys_solver_rho_inv_vec[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (signed long long *)((&linsys_solver_rhotoKKT[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 19ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",19ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = tail call i64 @QDLDL_factor(i64 34, i64* getelementptr inbounds ([35 x i64]* @linsys_solver_KKT_p, i64 0, i64 0), i64* getelementptr inbounds ([79 x i64]* @linsys_solver_KKT_i, i64 0, i64 0), float* getelementptr inbounds ([79 x float]* @linsys_solver_KKT_x, i64 0, i64 0), i64* getelementptr inbounds ([35 x i64]* @linsys_solver_L_p, i64 0, i64 0), i64* getelementptr inbounds ([57 x i64]* @linsys_solver_L_i, i64 0, i64 0), float* getelementptr inbounds ([57 x float]* @linsys_solver_L_x, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_D, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_Dinv, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_Lnz, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_etree, i64 0, i64 0), i64* getelementptr inbounds ([34 x i64]* @linsys_solver_bwork, i64 0, i64 0), i64* getelementptr inbounds ([102 x i64]* @linsys_solver_iwork, i64 0, i64 0), float* getelementptr inbounds ([34 x float]* @linsys_solver_fwork, i64 0, i64 0)) nounwind, !dbg !33 for 0x%I64xth hint within @update_linsys_solver_rho_vec_qdldl  --> \n", ++aesl_llvm_cbe_348_count);
  llvm_cbe_tmp__61 = (unsigned long long ) /*tail*/ QDLDL_factor(34ull, (signed long long *)((&linsys_solver_KKT_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 35
#endif
])), (signed long long *)((&linsys_solver_KKT_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (float *)((&linsys_solver_KKT_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 79
#endif
])), (signed long long *)((&linsys_solver_L_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 35
#endif
])), (signed long long *)((&linsys_solver_L_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_L_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 57
#endif
])), (float *)((&linsys_solver_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (float *)((&linsys_solver_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_Lnz[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_etree[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_bwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])), (signed long long *)((&linsys_solver_iwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 102
#endif
])), (float *)((&linsys_solver_fwork[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",34ull);
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__61);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_linsys_solver_rho_vec_qdldl}\n");
  return llvm_cbe_tmp__61;
}

