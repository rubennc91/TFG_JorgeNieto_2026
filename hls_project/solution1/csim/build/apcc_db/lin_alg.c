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

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void vec_add_scaled(float *llvm_cbe_c, float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n, float llvm_cbe_sc);
float vec_scaled_norm_inf(float *llvm_cbe_S, float *llvm_cbe_v, signed long long llvm_cbe_l);
float vec_norm_inf(float *llvm_cbe_v, signed long long llvm_cbe_l);
float vec_norm_inf_diff(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_l);
float vec_mean(float *llvm_cbe_a, signed long long llvm_cbe_n);
void int_vec_set_scalar(signed long long *llvm_cbe_a, signed long long llvm_cbe_sc, signed long long llvm_cbe_n);
void vec_set_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n);
void vec_add_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n);
void vec_mult_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n);
void prea_int_vec_copy(signed long long *llvm_cbe_a, signed long long *llvm_cbe_b, signed long long llvm_cbe_n);
void prea_vec_copy(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n);
void vec_ew_recipr(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n);
float vec_prod(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n);
void vec_ew_prod(float *llvm_cbe_a, float *llvm_cbe_b, float *llvm_cbe_c, signed long long llvm_cbe_n);
void mat_mult_scalar(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long llvm_cbe_An, float llvm_cbe_sc);
void mat_premult_diag(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, float *llvm_cbe_d);
void mat_postmult_diag(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long llvm_cbe_An, float *llvm_cbe_d);
void mat_vec(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, signed long long llvm_cbe_Am, float *llvm_cbe_x, float *llvm_cbe_y, signed long long llvm_cbe_plus_eq);
void mat_tpose_vec(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, signed long long llvm_cbe_Am, float *llvm_cbe_x, float *llvm_cbe_y, signed long long llvm_cbe_plus_eq, signed long long llvm_cbe_skip_diag);
float quad_form(float *llvm_cbe_Px, signed long long *llvm_cbe_Pp, signed long long *llvm_cbe_Pi, signed long long llvm_cbe_Pn, float *llvm_cbe_x);


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

void vec_add_scaled(float *llvm_cbe_c, float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n, float llvm_cbe_sc) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  float *llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  float llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  float *llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  float llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  float llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  float llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  unsigned long long llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_add_scaled\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%9, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__8);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__1 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_20_count);
  llvm_cbe_tmp__2 = (float )*llvm_cbe_tmp__1;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__2, *(int*)(&llvm_cbe_tmp__2));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_21_count);
  llvm_cbe_tmp__3 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* %%4, align 4, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_22_count);
  llvm_cbe_tmp__4 = (float )*llvm_cbe_tmp__3;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__4, *(int*)(&llvm_cbe_tmp__4));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = fmul float %%5, %%sc, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__5 = (float )((float )(llvm_cbe_tmp__4 * llvm_cbe_sc));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__5, *(int*)(&llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fadd float %%3, %%6, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_24_count);
  llvm_cbe_tmp__6 = (float )((float )(llvm_cbe_tmp__2 + llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__6, *(int*)(&llvm_cbe_tmp__6));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds float* %%c, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_25_count);
  llvm_cbe_tmp__7 = (float *)(&llvm_cbe_c[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%7, float* %%8, align 4, !dbg !14 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_26_count);
  *llvm_cbe_tmp__7 = llvm_cbe_tmp__6;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__6);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_add_scaled  --> \n", ++aesl_llvm_cbe_27_count);
  llvm_cbe_tmp__8 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__8&18446744073709551615ull)));
  if (((llvm_cbe_tmp__8&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__8;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_add_scaled}\n");
  return;
}


float vec_scaled_norm_inf(float *llvm_cbe_S, float *llvm_cbe_v, signed long long llvm_cbe_l) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_60_count = 0;
  float llvm_cbe_tmp__9;
  float llvm_cbe_tmp__9__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_61_count = 0;
  float *llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_62_count = 0;
  float llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_63_count = 0;
  float *llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  float llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  float llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  float llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  float llvm_cbe_tmp__16;
  float llvm_cbe_tmp__16__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  float llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  unsigned long long llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_scaled_norm_inf\n");
  if ((((signed long long )llvm_cbe_l) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__9__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%15, %%11 ], [ 0, %%0  for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__18);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%14, %%11 ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_60_count);
  llvm_cbe_tmp__9 = (float )llvm_cbe_tmp__9__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__9);
printf("\n = %f",llvm_cbe_tmp__17);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%S, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_61_count);
  llvm_cbe_tmp__10 = (float *)(&llvm_cbe_S[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_62_count);
  llvm_cbe_tmp__11 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__11, *(int*)(&llvm_cbe_tmp__11));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%v, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_63_count);
  llvm_cbe_tmp__12 = (float *)(&llvm_cbe_v[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_64_count);
  llvm_cbe_tmp__13 = (float )*llvm_cbe_tmp__12;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__13, *(int*)(&llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fmul float %%4, %%6, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_65_count);
  llvm_cbe_tmp__14 = (float )((float )(llvm_cbe_tmp__11 * llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__14, *(int*)(&llvm_cbe_tmp__14));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__14, 0x0p0))) {
    goto llvm_cbe_tmp__19;
  } else {
    llvm_cbe_tmp__16__PHI_TEMPORARY = (float )llvm_cbe_tmp__14;   /* for PHI node */
    goto llvm_cbe_tmp__20;
  }

llvm_cbe_tmp__20:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = phi float [ %%10, %%9 ], [ %%7, %%.lr.ph ], !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_70_count);
  llvm_cbe_tmp__16 = (float )llvm_cbe_tmp__16__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__16);
printf("\n = %f",llvm_cbe_tmp__15);
printf("\n = %f",llvm_cbe_tmp__14);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = select i1 %%13, float %%12, float %%2, !dbg !15 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_78_count);
  llvm_cbe_tmp__17 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__16, llvm_cbe_tmp__9))) ? ((float )llvm_cbe_tmp__16) : ((float )llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__17, *(int*)(&llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_80_count);
  llvm_cbe_tmp__18 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__18&18446744073709551615ull)));
  if (((llvm_cbe_tmp__18&18446744073709551615ULL) == (llvm_cbe_l&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__17;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__18;   /* for PHI node */
    llvm_cbe_tmp__9__PHI_TEMPORARY = (float )llvm_cbe_tmp__17;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

llvm_cbe_tmp__19:
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fsub float -0.000000e+00, %%7, !dbg !14 for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe_68_count);
  llvm_cbe_tmp__15 = (float )((float )(-(llvm_cbe_tmp__14)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__15, *(int*)(&llvm_cbe_tmp__15));
  llvm_cbe_tmp__16__PHI_TEMPORARY = (float )llvm_cbe_tmp__15;   /* for PHI node */
  goto llvm_cbe_tmp__20;

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%0 ], [ %%14, %%11  for 0x%I64xth hint within @vec_scaled_norm_inf  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__17);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_scaled_norm_inf}\n");
  return llvm_cbe__2e_lcssa;
}


float vec_norm_inf(float *llvm_cbe_v, signed long long llvm_cbe_l) {
  static  unsigned long long aesl_llvm_cbe_92_count = 0;
  static  unsigned long long aesl_llvm_cbe_93_count = 0;
  static  unsigned long long aesl_llvm_cbe_94_count = 0;
  static  unsigned long long aesl_llvm_cbe_95_count = 0;
  static  unsigned long long aesl_llvm_cbe_96_count = 0;
  static  unsigned long long aesl_llvm_cbe_97_count = 0;
  static  unsigned long long aesl_llvm_cbe_98_count = 0;
  static  unsigned long long aesl_llvm_cbe_99_count = 0;
  static  unsigned long long aesl_llvm_cbe_100_count = 0;
  static  unsigned long long aesl_llvm_cbe_101_count = 0;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  float llvm_cbe_tmp__21;
  float llvm_cbe_tmp__21__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  float *llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  float llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  float llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_115_count = 0;
  static  unsigned long long aesl_llvm_cbe_116_count = 0;
  float llvm_cbe_tmp__25;
  float llvm_cbe_tmp__25__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_117_count = 0;
  static  unsigned long long aesl_llvm_cbe_118_count = 0;
  static  unsigned long long aesl_llvm_cbe_119_count = 0;
  static  unsigned long long aesl_llvm_cbe_120_count = 0;
  static  unsigned long long aesl_llvm_cbe_121_count = 0;
  static  unsigned long long aesl_llvm_cbe_122_count = 0;
  static  unsigned long long aesl_llvm_cbe_123_count = 0;
  static  unsigned long long aesl_llvm_cbe_124_count = 0;
  float llvm_cbe_tmp__26;
  static  unsigned long long aesl_llvm_cbe_125_count = 0;
  static  unsigned long long aesl_llvm_cbe_126_count = 0;
  unsigned long long llvm_cbe_tmp__27;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;
  static  unsigned long long aesl_llvm_cbe_128_count = 0;
  static  unsigned long long aesl_llvm_cbe_129_count = 0;
  static  unsigned long long aesl_llvm_cbe_130_count = 0;
  static  unsigned long long aesl_llvm_cbe_131_count = 0;
  static  unsigned long long aesl_llvm_cbe_132_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_133_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_134_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_norm_inf\n");
  if ((((signed long long )llvm_cbe_l) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__21__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%12, %%8 ], [ 0, %%0  for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__27);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%11, %%8 ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_109_count);
  llvm_cbe_tmp__21 = (float )llvm_cbe_tmp__21__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__21);
printf("\n = %f",llvm_cbe_tmp__26);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%v, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_110_count);
  llvm_cbe_tmp__22 = (float *)(&llvm_cbe_v[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !14 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_111_count);
  llvm_cbe_tmp__23 = (float )*llvm_cbe_tmp__22;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__23, *(int*)(&llvm_cbe_tmp__23));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__23, 0x0p0))) {
    goto llvm_cbe_tmp__28;
  } else {
    llvm_cbe_tmp__25__PHI_TEMPORARY = (float )llvm_cbe_tmp__23;   /* for PHI node */
    goto llvm_cbe_tmp__29;
  }

llvm_cbe_tmp__29:
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = phi float [ %%7, %%6 ], [ %%4, %%.lr.ph ], !dbg !14 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_116_count);
  llvm_cbe_tmp__25 = (float )llvm_cbe_tmp__25__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__25);
printf("\n = %f",llvm_cbe_tmp__24);
printf("\n = %f",llvm_cbe_tmp__23);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = select i1 %%10, float %%9, float %%2, !dbg !15 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_124_count);
  llvm_cbe_tmp__26 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__25, llvm_cbe_tmp__21))) ? ((float )llvm_cbe_tmp__25) : ((float )llvm_cbe_tmp__21));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__26, *(int*)(&llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_126_count);
  llvm_cbe_tmp__27 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__27&18446744073709551615ull)));
  if (((llvm_cbe_tmp__27&18446744073709551615ULL) == (llvm_cbe_l&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__26;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__27;   /* for PHI node */
    llvm_cbe_tmp__21__PHI_TEMPORARY = (float )llvm_cbe_tmp__26;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

llvm_cbe_tmp__28:
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fsub float -0.000000e+00, %%4, !dbg !14 for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe_114_count);
  llvm_cbe_tmp__24 = (float )((float )(-(llvm_cbe_tmp__23)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__24, *(int*)(&llvm_cbe_tmp__24));
  llvm_cbe_tmp__25__PHI_TEMPORARY = (float )llvm_cbe_tmp__24;   /* for PHI node */
  goto llvm_cbe_tmp__29;

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%0 ], [ %%11, %%8  for 0x%I64xth hint within @vec_norm_inf  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__26);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_norm_inf}\n");
  return llvm_cbe__2e_lcssa;
}


float vec_norm_inf_diff(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_l) {
  static  unsigned long long aesl_llvm_cbe_135_count = 0;
  static  unsigned long long aesl_llvm_cbe_136_count = 0;
  static  unsigned long long aesl_llvm_cbe_137_count = 0;
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  float llvm_cbe_tmp__30;
  float llvm_cbe_tmp__30__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  float *llvm_cbe_tmp__31;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  float llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  float *llvm_cbe_tmp__33;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  float llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  float llvm_cbe_tmp__35;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  float llvm_cbe_tmp__37;
  float llvm_cbe_tmp__37__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  static  unsigned long long aesl_llvm_cbe_175_count = 0;
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
  static  unsigned long long aesl_llvm_cbe_177_count = 0;
  float llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_178_count = 0;
  static  unsigned long long aesl_llvm_cbe_179_count = 0;
  unsigned long long llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_180_count = 0;
  static  unsigned long long aesl_llvm_cbe_181_count = 0;
  static  unsigned long long aesl_llvm_cbe_182_count = 0;
  static  unsigned long long aesl_llvm_cbe_183_count = 0;
  static  unsigned long long aesl_llvm_cbe_184_count = 0;
  static  unsigned long long aesl_llvm_cbe_185_count = 0;
  static  unsigned long long aesl_llvm_cbe_186_count = 0;
  static  unsigned long long aesl_llvm_cbe_187_count = 0;
  static  unsigned long long aesl_llvm_cbe_188_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_189_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_190_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_norm_inf_diff\n");
  if ((((signed long long )llvm_cbe_l) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__30__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%15, %%11 ], [ 0, %%0  for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__39);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%14, %%11 ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_159_count);
  llvm_cbe_tmp__30 = (float )llvm_cbe_tmp__30__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__30);
printf("\n = %f",llvm_cbe_tmp__38);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_160_count);
  llvm_cbe_tmp__31 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_161_count);
  llvm_cbe_tmp__32 = (float )*llvm_cbe_tmp__31;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__32, *(int*)(&llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_162_count);
  llvm_cbe_tmp__33 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_163_count);
  llvm_cbe_tmp__34 = (float )*llvm_cbe_tmp__33;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__34, *(int*)(&llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fsub float %%4, %%6, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_164_count);
  llvm_cbe_tmp__35 = (float )((float )(llvm_cbe_tmp__32 - llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__35, *(int*)(&llvm_cbe_tmp__35));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__35, 0x0p0))) {
    goto llvm_cbe_tmp__40;
  } else {
    llvm_cbe_tmp__37__PHI_TEMPORARY = (float )llvm_cbe_tmp__35;   /* for PHI node */
    goto llvm_cbe_tmp__41;
  }

llvm_cbe_tmp__41:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = phi float [ %%10, %%9 ], [ %%7, %%.lr.ph ], !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_169_count);
  llvm_cbe_tmp__37 = (float )llvm_cbe_tmp__37__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__37);
printf("\n = %f",llvm_cbe_tmp__36);
printf("\n = %f",llvm_cbe_tmp__35);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = select i1 %%13, float %%12, float %%2, !dbg !15 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_177_count);
  llvm_cbe_tmp__38 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__37, llvm_cbe_tmp__30))) ? ((float )llvm_cbe_tmp__37) : ((float )llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__38, *(int*)(&llvm_cbe_tmp__38));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_179_count);
  llvm_cbe_tmp__39 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__39&18446744073709551615ull)));
  if (((llvm_cbe_tmp__39&18446744073709551615ULL) == (llvm_cbe_l&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__38;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__39;   /* for PHI node */
    llvm_cbe_tmp__30__PHI_TEMPORARY = (float )llvm_cbe_tmp__38;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

llvm_cbe_tmp__40:
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fsub float -0.000000e+00, %%7, !dbg !14 for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe_167_count);
  llvm_cbe_tmp__36 = (float )((float )(-(llvm_cbe_tmp__35)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));
  llvm_cbe_tmp__37__PHI_TEMPORARY = (float )llvm_cbe_tmp__36;   /* for PHI node */
  goto llvm_cbe_tmp__41;

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%0 ], [ %%14, %%11  for 0x%I64xth hint within @vec_norm_inf_diff  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__38);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_norm_inf_diff}\n");
  return llvm_cbe__2e_lcssa;
}


float vec_mean(float *llvm_cbe_a, signed long long llvm_cbe_n) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  float llvm_cbe_tmp__42;
  float llvm_cbe_tmp__42__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_206_count = 0;
  float *llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_207_count = 0;
  float llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_208_count = 0;
  float llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_209_count = 0;
  static  unsigned long long aesl_llvm_cbe_210_count = 0;
  static  unsigned long long aesl_llvm_cbe_211_count = 0;
  static  unsigned long long aesl_llvm_cbe_212_count = 0;
  unsigned long long llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_213_count = 0;
  static  unsigned long long aesl_llvm_cbe_214_count = 0;
  static  unsigned long long aesl_llvm_cbe_215_count = 0;
  static  unsigned long long aesl_llvm_cbe_216_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_217_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_218_count = 0;
  float llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_219_count = 0;
  float llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_220_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_mean\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__42__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%6, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__46);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%5, %%.lr.ph ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_205_count);
  llvm_cbe_tmp__42 = (float )llvm_cbe_tmp__42__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__42);
printf("\n = %f",llvm_cbe_tmp__45);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_206_count);
  llvm_cbe_tmp__43 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !14 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_207_count);
  llvm_cbe_tmp__44 = (float )*llvm_cbe_tmp__43;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__44, *(int*)(&llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fadd float %%2, %%4, !dbg !14 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_208_count);
  llvm_cbe_tmp__45 = (float )((float )(llvm_cbe_tmp__42 + llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__45, *(int*)(&llvm_cbe_tmp__45));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_212_count);
  llvm_cbe_tmp__46 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__46&18446744073709551615ull)));
  if (((llvm_cbe_tmp__46&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__45;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__46;   /* for PHI node */
    llvm_cbe_tmp__42__PHI_TEMPORARY = (float )llvm_cbe_tmp__45;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%0 ], [ %%5, %%.lr.ph  for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__45);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = sitofp i64 %%n to float, !dbg !14 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_218_count);
  llvm_cbe_tmp__47 = (float )((float )(signed long long )llvm_cbe_n);
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__47, *(int*)(&llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = fdiv float %%.lcssa, %%7, !dbg !14 for 0x%I64xth hint within @vec_mean  --> \n", ++aesl_llvm_cbe_219_count);
  llvm_cbe_tmp__48 = (float )((float )(llvm_cbe__2e_lcssa / llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__48, *(int*)(&llvm_cbe_tmp__48));
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_mean}\n");
  return llvm_cbe_tmp__48;
}


void int_vec_set_scalar(signed long long *llvm_cbe_a, signed long long llvm_cbe_sc, signed long long llvm_cbe_n) {
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
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  signed long long *llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  unsigned long long llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @int_vec_set_scalar\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%3, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @int_vec_set_scalar  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__50);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds i64* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @int_vec_set_scalar  --> \n", ++aesl_llvm_cbe_233_count);
  llvm_cbe_tmp__49 = (signed long long *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 %%sc, i64* %%2, align 8, !dbg !14 for 0x%I64xth hint within @int_vec_set_scalar  --> \n", ++aesl_llvm_cbe_234_count);
  *llvm_cbe_tmp__49 = llvm_cbe_sc;
if (AESL_DEBUG_TRACE)
printf("\nsc = 0x%I64X\n", llvm_cbe_sc);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @int_vec_set_scalar  --> \n", ++aesl_llvm_cbe_235_count);
  llvm_cbe_tmp__50 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__50&18446744073709551615ull)));
  if (((llvm_cbe_tmp__50&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__50;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @int_vec_set_scalar}\n");
  return;
}


void vec_set_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n) {
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_254_count = 0;
  float *llvm_cbe_tmp__51;
  static  unsigned long long aesl_llvm_cbe_255_count = 0;
  static  unsigned long long aesl_llvm_cbe_256_count = 0;
  unsigned long long llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
  static  unsigned long long aesl_llvm_cbe_258_count = 0;
  static  unsigned long long aesl_llvm_cbe_259_count = 0;
  static  unsigned long long aesl_llvm_cbe_260_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_261_count = 0;
  static  unsigned long long aesl_llvm_cbe_262_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_set_scalar\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%3, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_set_scalar  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__52);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_set_scalar  --> \n", ++aesl_llvm_cbe_254_count);
  llvm_cbe_tmp__51 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%sc, float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_set_scalar  --> \n", ++aesl_llvm_cbe_255_count);
  *llvm_cbe_tmp__51 = llvm_cbe_sc;
if (AESL_DEBUG_TRACE)
printf("\nsc = %f\n", llvm_cbe_sc);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_set_scalar  --> \n", ++aesl_llvm_cbe_256_count);
  llvm_cbe_tmp__52 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__52&18446744073709551615ull)));
  if (((llvm_cbe_tmp__52&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__52;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_set_scalar}\n");
  return;
}


void vec_add_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_275_count = 0;
  float *llvm_cbe_tmp__53;
  static  unsigned long long aesl_llvm_cbe_276_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  float llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  unsigned long long llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_add_scalar\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%5, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__56);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_275_count);
  llvm_cbe_tmp__53 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_276_count);
  llvm_cbe_tmp__54 = (float )*llvm_cbe_tmp__53;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fadd float %%3, %%sc, !dbg !14 for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_277_count);
  llvm_cbe_tmp__55 = (float )((float )(llvm_cbe_tmp__54 + llvm_cbe_sc));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__55, *(int*)(&llvm_cbe_tmp__55));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_278_count);
  *llvm_cbe_tmp__53 = llvm_cbe_tmp__55;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__55);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_add_scalar  --> \n", ++aesl_llvm_cbe_279_count);
  llvm_cbe_tmp__56 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__56&18446744073709551615ull)));
  if (((llvm_cbe_tmp__56&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__56;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_add_scalar}\n");
  return;
}


void vec_mult_scalar(float *llvm_cbe_a, float llvm_cbe_sc, signed long long llvm_cbe_n) {
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_298_count = 0;
  float *llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_299_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_300_count = 0;
  float llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_301_count = 0;
  static  unsigned long long aesl_llvm_cbe_302_count = 0;
  unsigned long long llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_303_count = 0;
  static  unsigned long long aesl_llvm_cbe_304_count = 0;
  static  unsigned long long aesl_llvm_cbe_305_count = 0;
  static  unsigned long long aesl_llvm_cbe_306_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_307_count = 0;
  static  unsigned long long aesl_llvm_cbe_308_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_mult_scalar\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%5, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__60);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_298_count);
  llvm_cbe_tmp__57 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_299_count);
  llvm_cbe_tmp__58 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fmul float %%3, %%sc, !dbg !14 for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_300_count);
  llvm_cbe_tmp__59 = (float )((float )(llvm_cbe_tmp__58 * llvm_cbe_sc));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__59, *(int*)(&llvm_cbe_tmp__59));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_301_count);
  *llvm_cbe_tmp__57 = llvm_cbe_tmp__59;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__59);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_mult_scalar  --> \n", ++aesl_llvm_cbe_302_count);
  llvm_cbe_tmp__60 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__60&18446744073709551615ull)));
  if (((llvm_cbe_tmp__60&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__60;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_mult_scalar}\n");
  return;
}


void prea_int_vec_copy(signed long long *llvm_cbe_a, signed long long *llvm_cbe_b, signed long long llvm_cbe_n) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_322_count = 0;
  signed long long *llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_323_count = 0;
  unsigned long long llvm_cbe_tmp__62;
  static  unsigned long long aesl_llvm_cbe_324_count = 0;
  signed long long *llvm_cbe_tmp__63;
  static  unsigned long long aesl_llvm_cbe_325_count = 0;
  static  unsigned long long aesl_llvm_cbe_326_count = 0;
  unsigned long long llvm_cbe_tmp__64;
  static  unsigned long long aesl_llvm_cbe_327_count = 0;
  static  unsigned long long aesl_llvm_cbe_328_count = 0;
  static  unsigned long long aesl_llvm_cbe_329_count = 0;
  static  unsigned long long aesl_llvm_cbe_330_count = 0;
  static  unsigned long long aesl_llvm_cbe_331_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_332_count = 0;
  static  unsigned long long aesl_llvm_cbe_333_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @prea_int_vec_copy\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%5, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__64);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds i64* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_322_count);
  llvm_cbe_tmp__61 = (signed long long *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !14 for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_323_count);
  llvm_cbe_tmp__62 = (unsigned long long )*llvm_cbe_tmp__61;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__62);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds i64* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_324_count);
  llvm_cbe_tmp__63 = (signed long long *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 %%3, i64* %%4, align 8, !dbg !14 for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_325_count);
  *llvm_cbe_tmp__63 = llvm_cbe_tmp__62;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__62);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @prea_int_vec_copy  --> \n", ++aesl_llvm_cbe_326_count);
  llvm_cbe_tmp__64 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__64&18446744073709551615ull)));
  if (((llvm_cbe_tmp__64&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__64;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @prea_int_vec_copy}\n");
  return;
}


void prea_vec_copy(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n) {
  static  unsigned long long aesl_llvm_cbe_334_count = 0;
  static  unsigned long long aesl_llvm_cbe_335_count = 0;
  static  unsigned long long aesl_llvm_cbe_336_count = 0;
  static  unsigned long long aesl_llvm_cbe_337_count = 0;
  static  unsigned long long aesl_llvm_cbe_338_count = 0;
  static  unsigned long long aesl_llvm_cbe_339_count = 0;
  static  unsigned long long aesl_llvm_cbe_340_count = 0;
  static  unsigned long long aesl_llvm_cbe_341_count = 0;
  static  unsigned long long aesl_llvm_cbe_342_count = 0;
  static  unsigned long long aesl_llvm_cbe_343_count = 0;
  static  unsigned long long aesl_llvm_cbe_344_count = 0;
  static  unsigned long long aesl_llvm_cbe_345_count = 0;
  static  unsigned long long aesl_llvm_cbe_346_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_347_count = 0;
  float *llvm_cbe_tmp__65;
  static  unsigned long long aesl_llvm_cbe_348_count = 0;
  float llvm_cbe_tmp__66;
  static  unsigned long long aesl_llvm_cbe_349_count = 0;
  float *llvm_cbe_tmp__67;
  static  unsigned long long aesl_llvm_cbe_350_count = 0;
  static  unsigned long long aesl_llvm_cbe_351_count = 0;
  unsigned long long llvm_cbe_tmp__68;
  static  unsigned long long aesl_llvm_cbe_352_count = 0;
  static  unsigned long long aesl_llvm_cbe_353_count = 0;
  static  unsigned long long aesl_llvm_cbe_354_count = 0;
  static  unsigned long long aesl_llvm_cbe_355_count = 0;
  static  unsigned long long aesl_llvm_cbe_356_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_357_count = 0;
  static  unsigned long long aesl_llvm_cbe_358_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @prea_vec_copy\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%5, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__68);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_347_count);
  llvm_cbe_tmp__65 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_348_count);
  llvm_cbe_tmp__66 = (float )*llvm_cbe_tmp__65;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__66, *(int*)(&llvm_cbe_tmp__66));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_349_count);
  llvm_cbe_tmp__67 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* %%4, align 4, !dbg !14 for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_350_count);
  *llvm_cbe_tmp__67 = llvm_cbe_tmp__66;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__66);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @prea_vec_copy  --> \n", ++aesl_llvm_cbe_351_count);
  llvm_cbe_tmp__68 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__68&18446744073709551615ull)));
  if (((llvm_cbe_tmp__68&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__68;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @prea_vec_copy}\n");
  return;
}


void vec_ew_recipr(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n) {
  static  unsigned long long aesl_llvm_cbe_359_count = 0;
  static  unsigned long long aesl_llvm_cbe_360_count = 0;
  static  unsigned long long aesl_llvm_cbe_361_count = 0;
  static  unsigned long long aesl_llvm_cbe_362_count = 0;
  static  unsigned long long aesl_llvm_cbe_363_count = 0;
  static  unsigned long long aesl_llvm_cbe_364_count = 0;
  static  unsigned long long aesl_llvm_cbe_365_count = 0;
  static  unsigned long long aesl_llvm_cbe_366_count = 0;
  static  unsigned long long aesl_llvm_cbe_367_count = 0;
  static  unsigned long long aesl_llvm_cbe_368_count = 0;
  static  unsigned long long aesl_llvm_cbe_369_count = 0;
  static  unsigned long long aesl_llvm_cbe_370_count = 0;
  static  unsigned long long aesl_llvm_cbe_371_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_372_count = 0;
  float *llvm_cbe_tmp__69;
  static  unsigned long long aesl_llvm_cbe_373_count = 0;
  float llvm_cbe_tmp__70;
  static  unsigned long long aesl_llvm_cbe_374_count = 0;
  float llvm_cbe_tmp__71;
  static  unsigned long long aesl_llvm_cbe_375_count = 0;
  float *llvm_cbe_tmp__72;
  static  unsigned long long aesl_llvm_cbe_376_count = 0;
  static  unsigned long long aesl_llvm_cbe_377_count = 0;
  unsigned long long llvm_cbe_tmp__73;
  static  unsigned long long aesl_llvm_cbe_378_count = 0;
  static  unsigned long long aesl_llvm_cbe_379_count = 0;
  static  unsigned long long aesl_llvm_cbe_380_count = 0;
  static  unsigned long long aesl_llvm_cbe_381_count = 0;
  static  unsigned long long aesl_llvm_cbe_382_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_383_count = 0;
  static  unsigned long long aesl_llvm_cbe_384_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_ew_recipr\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%6, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__73);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_372_count);
  llvm_cbe_tmp__69 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_373_count);
  llvm_cbe_tmp__70 = (float )*llvm_cbe_tmp__69;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__70, *(int*)(&llvm_cbe_tmp__70));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fdiv float 1.000000e+00, %%3, !dbg !14 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_374_count);
  llvm_cbe_tmp__71 = (float )((float )(0x1p0 / llvm_cbe_tmp__70));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__71, *(int*)(&llvm_cbe_tmp__71));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_375_count);
  llvm_cbe_tmp__72 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%4, float* %%5, align 4, !dbg !14 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_376_count);
  *llvm_cbe_tmp__72 = llvm_cbe_tmp__71;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__71);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_ew_recipr  --> \n", ++aesl_llvm_cbe_377_count);
  llvm_cbe_tmp__73 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__73&18446744073709551615ull)));
  if (((llvm_cbe_tmp__73&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__73;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_ew_recipr}\n");
  return;
}


float vec_prod(float *llvm_cbe_a, float *llvm_cbe_b, signed long long llvm_cbe_n) {
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
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_401_count = 0;
  float llvm_cbe_tmp__74;
  float llvm_cbe_tmp__74__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_402_count = 0;
  float *llvm_cbe_tmp__75;
  static  unsigned long long aesl_llvm_cbe_403_count = 0;
  float llvm_cbe_tmp__76;
  static  unsigned long long aesl_llvm_cbe_404_count = 0;
  float *llvm_cbe_tmp__77;
  static  unsigned long long aesl_llvm_cbe_405_count = 0;
  float llvm_cbe_tmp__78;
  static  unsigned long long aesl_llvm_cbe_406_count = 0;
  float llvm_cbe_tmp__79;
  static  unsigned long long aesl_llvm_cbe_407_count = 0;
  float llvm_cbe_tmp__80;
  static  unsigned long long aesl_llvm_cbe_408_count = 0;
  static  unsigned long long aesl_llvm_cbe_409_count = 0;
  static  unsigned long long aesl_llvm_cbe_410_count = 0;
  static  unsigned long long aesl_llvm_cbe_411_count = 0;
  unsigned long long llvm_cbe_tmp__81;
  static  unsigned long long aesl_llvm_cbe_412_count = 0;
  static  unsigned long long aesl_llvm_cbe_413_count = 0;
  static  unsigned long long aesl_llvm_cbe_414_count = 0;
  static  unsigned long long aesl_llvm_cbe_415_count = 0;
  static  unsigned long long aesl_llvm_cbe_416_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_417_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_418_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_prod\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__74__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%9, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__81);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%8, %%.lr.ph ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_401_count);
  llvm_cbe_tmp__74 = (float )llvm_cbe_tmp__74__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__74);
printf("\n = %f",llvm_cbe_tmp__80);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_402_count);
  llvm_cbe_tmp__75 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_403_count);
  llvm_cbe_tmp__76 = (float )*llvm_cbe_tmp__75;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__76, *(int*)(&llvm_cbe_tmp__76));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_404_count);
  llvm_cbe_tmp__77 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_405_count);
  llvm_cbe_tmp__78 = (float )*llvm_cbe_tmp__77;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__78, *(int*)(&llvm_cbe_tmp__78));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fmul float %%4, %%6, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_406_count);
  llvm_cbe_tmp__79 = (float )((float )(llvm_cbe_tmp__76 * llvm_cbe_tmp__78));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__79, *(int*)(&llvm_cbe_tmp__79));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = fadd float %%2, %%7, !dbg !14 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_407_count);
  llvm_cbe_tmp__80 = (float )((float )(llvm_cbe_tmp__74 + llvm_cbe_tmp__79));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__80, *(int*)(&llvm_cbe_tmp__80));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe_411_count);
  llvm_cbe_tmp__81 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__81&18446744073709551615ull)));
  if (((llvm_cbe_tmp__81&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__80;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__81;   /* for PHI node */
    llvm_cbe_tmp__74__PHI_TEMPORARY = (float )llvm_cbe_tmp__80;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%0 ], [ %%8, %%.lr.ph  for 0x%I64xth hint within @vec_prod  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__80);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_prod}\n");
  return llvm_cbe__2e_lcssa;
}


void vec_ew_prod(float *llvm_cbe_a, float *llvm_cbe_b, float *llvm_cbe_c, signed long long llvm_cbe_n) {
  static  unsigned long long aesl_llvm_cbe_419_count = 0;
  static  unsigned long long aesl_llvm_cbe_420_count = 0;
  static  unsigned long long aesl_llvm_cbe_421_count = 0;
  static  unsigned long long aesl_llvm_cbe_422_count = 0;
  static  unsigned long long aesl_llvm_cbe_423_count = 0;
  static  unsigned long long aesl_llvm_cbe_424_count = 0;
  static  unsigned long long aesl_llvm_cbe_425_count = 0;
  static  unsigned long long aesl_llvm_cbe_426_count = 0;
  static  unsigned long long aesl_llvm_cbe_427_count = 0;
  static  unsigned long long aesl_llvm_cbe_428_count = 0;
  static  unsigned long long aesl_llvm_cbe_429_count = 0;
  static  unsigned long long aesl_llvm_cbe_430_count = 0;
  static  unsigned long long aesl_llvm_cbe_431_count = 0;
  static  unsigned long long aesl_llvm_cbe_432_count = 0;
  static  unsigned long long aesl_llvm_cbe_433_count = 0;
  static  unsigned long long aesl_llvm_cbe_434_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_435_count = 0;
  float *llvm_cbe_tmp__82;
  static  unsigned long long aesl_llvm_cbe_436_count = 0;
  float llvm_cbe_tmp__83;
  static  unsigned long long aesl_llvm_cbe_437_count = 0;
  float *llvm_cbe_tmp__84;
  static  unsigned long long aesl_llvm_cbe_438_count = 0;
  float llvm_cbe_tmp__85;
  static  unsigned long long aesl_llvm_cbe_439_count = 0;
  float llvm_cbe_tmp__86;
  static  unsigned long long aesl_llvm_cbe_440_count = 0;
  float *llvm_cbe_tmp__87;
  static  unsigned long long aesl_llvm_cbe_441_count = 0;
  static  unsigned long long aesl_llvm_cbe_442_count = 0;
  unsigned long long llvm_cbe_tmp__88;
  static  unsigned long long aesl_llvm_cbe_443_count = 0;
  static  unsigned long long aesl_llvm_cbe_444_count = 0;
  static  unsigned long long aesl_llvm_cbe_445_count = 0;
  static  unsigned long long aesl_llvm_cbe_446_count = 0;
  static  unsigned long long aesl_llvm_cbe_447_count = 0;
  static  unsigned long long aesl_llvm_cbe_448_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_449_count = 0;
  static  unsigned long long aesl_llvm_cbe_450_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @vec_ew_prod\n");
  if ((((signed long long )llvm_cbe_n) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%8, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__88);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds float* %%b, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_435_count);
  llvm_cbe_tmp__82 = (float *)(&llvm_cbe_b[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%2, align 4, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_436_count);
  llvm_cbe_tmp__83 = (float )*llvm_cbe_tmp__82;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__83, *(int*)(&llvm_cbe_tmp__83));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%a, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_437_count);
  llvm_cbe_tmp__84 = (float *)(&llvm_cbe_a[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* %%4, align 4, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_438_count);
  llvm_cbe_tmp__85 = (float )*llvm_cbe_tmp__84;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__85, *(int*)(&llvm_cbe_tmp__85));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = fmul float %%3, %%5, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_439_count);
  llvm_cbe_tmp__86 = (float )((float )(llvm_cbe_tmp__83 * llvm_cbe_tmp__85));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__86, *(int*)(&llvm_cbe_tmp__86));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds float* %%c, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_440_count);
  llvm_cbe_tmp__87 = (float *)(&llvm_cbe_c[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%7, align 4, !dbg !14 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_441_count);
  *llvm_cbe_tmp__87 = llvm_cbe_tmp__86;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__86);
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @vec_ew_prod  --> \n", ++aesl_llvm_cbe_442_count);
  llvm_cbe_tmp__88 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__88&18446744073709551615ull)));
  if (((llvm_cbe_tmp__88&18446744073709551615ULL) == (llvm_cbe_n&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__88;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @vec_ew_prod}\n");
  return;
}


void mat_mult_scalar(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long llvm_cbe_An, float llvm_cbe_sc) {
  static  unsigned long long aesl_llvm_cbe_451_count = 0;
  static  unsigned long long aesl_llvm_cbe_452_count = 0;
  static  unsigned long long aesl_llvm_cbe_453_count = 0;
  static  unsigned long long aesl_llvm_cbe_454_count = 0;
  static  unsigned long long aesl_llvm_cbe_455_count = 0;
  static  unsigned long long aesl_llvm_cbe_456_count = 0;
  static  unsigned long long aesl_llvm_cbe_457_count = 0;
  static  unsigned long long aesl_llvm_cbe_458_count = 0;
  static  unsigned long long aesl_llvm_cbe_459_count = 0;
  signed long long *llvm_cbe_tmp__89;
  static  unsigned long long aesl_llvm_cbe_460_count = 0;
  unsigned long long llvm_cbe_tmp__90;
  static  unsigned long long aesl_llvm_cbe_461_count = 0;
  static  unsigned long long aesl_llvm_cbe_462_count = 0;
  static  unsigned long long aesl_llvm_cbe_463_count = 0;
  static  unsigned long long aesl_llvm_cbe_464_count = 0;
  static  unsigned long long aesl_llvm_cbe_465_count = 0;
  static  unsigned long long aesl_llvm_cbe_466_count = 0;
  static  unsigned long long aesl_llvm_cbe_467_count = 0;
  static  unsigned long long aesl_llvm_cbe_468_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_469_count = 0;
  float *llvm_cbe_tmp__91;
  static  unsigned long long aesl_llvm_cbe_470_count = 0;
  float llvm_cbe_tmp__92;
  static  unsigned long long aesl_llvm_cbe_471_count = 0;
  float llvm_cbe_tmp__93;
  static  unsigned long long aesl_llvm_cbe_472_count = 0;
  static  unsigned long long aesl_llvm_cbe_473_count = 0;
  unsigned long long llvm_cbe_tmp__94;
  static  unsigned long long aesl_llvm_cbe_474_count = 0;
  static  unsigned long long aesl_llvm_cbe_475_count = 0;
  static  unsigned long long aesl_llvm_cbe_476_count = 0;
  static  unsigned long long aesl_llvm_cbe_477_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_478_count = 0;
  static  unsigned long long aesl_llvm_cbe_479_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @mat_mult_scalar\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds i64* %%Ap, i64 %%An, !dbg !15 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_459_count);
  llvm_cbe_tmp__89 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_An))]);
if (AESL_DEBUG_TRACE) {
printf("\nAn = 0x%I64X",((signed long long )llvm_cbe_An));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* %%1, align 8, !dbg !15 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_460_count);
  llvm_cbe_tmp__90 = (unsigned long long )*llvm_cbe_tmp__89;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__90);
  if ((((signed long long )llvm_cbe_tmp__90) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%7, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__94);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds float* %%Ax, i64 %%storemerge1, !dbg !14 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_469_count);
  llvm_cbe_tmp__91 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* %%4, align 4, !dbg !14 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_470_count);
  llvm_cbe_tmp__92 = (float )*llvm_cbe_tmp__91;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__92, *(int*)(&llvm_cbe_tmp__92));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = fmul float %%5, %%sc, !dbg !14 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_471_count);
  llvm_cbe_tmp__93 = (float )((float )(llvm_cbe_tmp__92 * llvm_cbe_sc));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__93, *(int*)(&llvm_cbe_tmp__93));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%6, float* %%4, align 4, !dbg !14 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_472_count);
  *llvm_cbe_tmp__91 = llvm_cbe_tmp__93;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__93);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = add nsw i64 %%storemerge1, 1, !dbg !15 for 0x%I64xth hint within @mat_mult_scalar  --> \n", ++aesl_llvm_cbe_473_count);
  llvm_cbe_tmp__94 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__94&18446744073709551615ull)));
  if (((llvm_cbe_tmp__94&18446744073709551615ULL) == (llvm_cbe_tmp__90&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__94;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @mat_mult_scalar}\n");
  return;
}


void mat_premult_diag(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, float *llvm_cbe_d) {
  static  unsigned long long aesl_llvm_cbe_480_count = 0;
  static  unsigned long long aesl_llvm_cbe_481_count = 0;
  static  unsigned long long aesl_llvm_cbe_482_count = 0;
  static  unsigned long long aesl_llvm_cbe_483_count = 0;
  static  unsigned long long aesl_llvm_cbe_484_count = 0;
  static  unsigned long long aesl_llvm_cbe_485_count = 0;
  static  unsigned long long aesl_llvm_cbe_486_count = 0;
  static  unsigned long long aesl_llvm_cbe_487_count = 0;
  static  unsigned long long aesl_llvm_cbe_488_count = 0;
  static  unsigned long long aesl_llvm_cbe_489_count = 0;
  static  unsigned long long aesl_llvm_cbe_490_count = 0;
  static  unsigned long long aesl_llvm_cbe_491_count = 0;
  static  unsigned long long aesl_llvm_cbe_492_count = 0;
  static  unsigned long long aesl_llvm_cbe_493_count = 0;
  static  unsigned long long aesl_llvm_cbe_494_count = 0;
  static  unsigned long long aesl_llvm_cbe_495_count = 0;
  static  unsigned long long aesl_llvm_cbe_496_count = 0;
  static  unsigned long long aesl_llvm_cbe_497_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_498_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge3_count = 0;
  unsigned long long llvm_cbe_storemerge3;
  unsigned long long llvm_cbe_storemerge3__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_499_count = 0;
  signed long long *llvm_cbe_tmp__95;
  static  unsigned long long aesl_llvm_cbe_500_count = 0;
  unsigned long long llvm_cbe_tmp__96;
  static  unsigned long long aesl_llvm_cbe_501_count = 0;
  static  unsigned long long aesl_llvm_cbe_502_count = 0;
  static  unsigned long long aesl_llvm_cbe_503_count = 0;
  static  unsigned long long aesl_llvm_cbe_504_count = 0;
  static  unsigned long long aesl_llvm_cbe_505_count = 0;
  static  unsigned long long aesl_llvm_cbe_506_count = 0;
  unsigned long long llvm_cbe_tmp__97;
  static  unsigned long long aesl_llvm_cbe_507_count = 0;
  signed long long *llvm_cbe_tmp__98;
  static  unsigned long long aesl_llvm_cbe_508_count = 0;
  unsigned long long llvm_cbe_tmp__99;
  static  unsigned long long aesl_llvm_cbe_509_count = 0;
  static  unsigned long long aesl_llvm_cbe_510_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge12_count = 0;
  unsigned long long llvm_cbe_storemerge12;
  unsigned long long llvm_cbe_storemerge12__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_511_count = 0;
  signed long long *llvm_cbe_tmp__100;
  static  unsigned long long aesl_llvm_cbe_512_count = 0;
  unsigned long long llvm_cbe_tmp__101;
  static  unsigned long long aesl_llvm_cbe_513_count = 0;
  float *llvm_cbe_tmp__102;
  static  unsigned long long aesl_llvm_cbe_514_count = 0;
  float llvm_cbe_tmp__103;
  static  unsigned long long aesl_llvm_cbe_515_count = 0;
  float *llvm_cbe_tmp__104;
  static  unsigned long long aesl_llvm_cbe_516_count = 0;
  float llvm_cbe_tmp__105;
  static  unsigned long long aesl_llvm_cbe_517_count = 0;
  float llvm_cbe_tmp__106;
  static  unsigned long long aesl_llvm_cbe_518_count = 0;
  static  unsigned long long aesl_llvm_cbe_519_count = 0;
  unsigned long long llvm_cbe_tmp__107;
  static  unsigned long long aesl_llvm_cbe_520_count = 0;
  static  unsigned long long aesl_llvm_cbe_521_count = 0;
  static  unsigned long long aesl_llvm_cbe_522_count = 0;
  static  unsigned long long aesl_llvm_cbe_523_count = 0;
  static  unsigned long long aesl_llvm_cbe_524_count = 0;
  static  unsigned long long aesl_llvm_cbe_525_count = 0;
  unsigned long long llvm_cbe_tmp__108;
  static  unsigned long long aesl_llvm_cbe_526_count = 0;
  static  unsigned long long aesl_llvm_cbe_527_count = 0;
  static  unsigned long long aesl_llvm_cbe_528_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @mat_premult_diag\n");
  if ((((signed long long )llvm_cbe_An) > ((signed long long )0ull))) {
    llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph4;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph4' to make GCC happy */
llvm_cbe__2e_lr_2e_ph4:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge3 = phi i64 [ %%4, %%.loopexit ], [ 0, %%0  for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_storemerge3_count);
  llvm_cbe_storemerge3 = (unsigned long long )llvm_cbe_storemerge3__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",llvm_cbe_storemerge3);
printf("\n = 0x%I64X",llvm_cbe_tmp__97);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds i64* %%Ap, i64 %%storemerge3, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_499_count);
  llvm_cbe_tmp__95 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_storemerge3))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_500_count);
  llvm_cbe_tmp__96 = (unsigned long long )*llvm_cbe_tmp__95;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__96);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = add nsw i64 %%storemerge3, 1, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_506_count);
  llvm_cbe_tmp__97 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge3&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__97&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds i64* %%Ap, i64 %%4, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_507_count);
  llvm_cbe_tmp__98 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_tmp__97))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__97));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* %%5, align 8, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_508_count);
  llvm_cbe_tmp__99 = (unsigned long long )*llvm_cbe_tmp__98;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__99);
  if ((((signed long long )llvm_cbe_tmp__96) < ((signed long long )llvm_cbe_tmp__99))) {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__96;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe__2e_loopexit:
  if (((llvm_cbe_tmp__97&18446744073709551615ULL) == (llvm_cbe_An&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__97;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph4;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge12 = phi i64 [ %%15, %%.lr.ph ], [ %%3, %%.lr.ph4  for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_storemerge12_count);
  llvm_cbe_storemerge12 = (unsigned long long )llvm_cbe_storemerge12__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",llvm_cbe_storemerge12);
printf("\n = 0x%I64X",llvm_cbe_tmp__107);
printf("\n = 0x%I64X",llvm_cbe_tmp__96);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds i64* %%Ai, i64 %%storemerge12, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_511_count);
  llvm_cbe_tmp__100 = (signed long long *)(&llvm_cbe_Ai[(((signed long long )llvm_cbe_storemerge12))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load i64* %%8, align 8, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_512_count);
  llvm_cbe_tmp__101 = (unsigned long long )*llvm_cbe_tmp__100;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__101);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds float* %%d, i64 %%9, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_513_count);
  llvm_cbe_tmp__102 = (float *)(&llvm_cbe_d[(((signed long long )llvm_cbe_tmp__101))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__101));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load float* %%10, align 4, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_514_count);
  llvm_cbe_tmp__103 = (float )*llvm_cbe_tmp__102;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__103, *(int*)(&llvm_cbe_tmp__103));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds float* %%Ax, i64 %%storemerge12, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_515_count);
  llvm_cbe_tmp__104 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge12))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* %%12, align 4, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_516_count);
  llvm_cbe_tmp__105 = (float )*llvm_cbe_tmp__104;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__105, *(int*)(&llvm_cbe_tmp__105));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fmul float %%13, %%11, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_517_count);
  llvm_cbe_tmp__106 = (float )((float )(llvm_cbe_tmp__105 * llvm_cbe_tmp__103));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__106, *(int*)(&llvm_cbe_tmp__106));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%12, align 4, !dbg !14 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_518_count);
  *llvm_cbe_tmp__104 = llvm_cbe_tmp__106;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__106);
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = add nsw i64 %%storemerge12, 1, !dbg !16 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_519_count);
  llvm_cbe_tmp__107 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__107&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load i64* %%5, align 8, !dbg !15 for 0x%I64xth hint within @mat_premult_diag  --> \n", ++aesl_llvm_cbe_525_count);
  llvm_cbe_tmp__108 = (unsigned long long )*llvm_cbe_tmp__98;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__108);
  if ((((signed long long )llvm_cbe_tmp__107) < ((signed long long )llvm_cbe_tmp__108))) {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__107;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
  } while (1); /* end of syntactic loop '.lr.ph4' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @mat_premult_diag}\n");
  return;
}


void mat_postmult_diag(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long llvm_cbe_An, float *llvm_cbe_d) {
  static  unsigned long long aesl_llvm_cbe_529_count = 0;
  static  unsigned long long aesl_llvm_cbe_530_count = 0;
  static  unsigned long long aesl_llvm_cbe_531_count = 0;
  static  unsigned long long aesl_llvm_cbe_532_count = 0;
  static  unsigned long long aesl_llvm_cbe_533_count = 0;
  static  unsigned long long aesl_llvm_cbe_534_count = 0;
  static  unsigned long long aesl_llvm_cbe_535_count = 0;
  static  unsigned long long aesl_llvm_cbe_536_count = 0;
  static  unsigned long long aesl_llvm_cbe_537_count = 0;
  static  unsigned long long aesl_llvm_cbe_538_count = 0;
  static  unsigned long long aesl_llvm_cbe_539_count = 0;
  static  unsigned long long aesl_llvm_cbe_540_count = 0;
  static  unsigned long long aesl_llvm_cbe_541_count = 0;
  static  unsigned long long aesl_llvm_cbe_542_count = 0;
  static  unsigned long long aesl_llvm_cbe_543_count = 0;
  static  unsigned long long aesl_llvm_cbe_544_count = 0;
  static  unsigned long long aesl_llvm_cbe_545_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_546_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge3_count = 0;
  unsigned long long llvm_cbe_storemerge3;
  unsigned long long llvm_cbe_storemerge3__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_547_count = 0;
  signed long long *llvm_cbe_tmp__109;
  static  unsigned long long aesl_llvm_cbe_548_count = 0;
  unsigned long long llvm_cbe_tmp__110;
  static  unsigned long long aesl_llvm_cbe_549_count = 0;
  static  unsigned long long aesl_llvm_cbe_550_count = 0;
  static  unsigned long long aesl_llvm_cbe_551_count = 0;
  static  unsigned long long aesl_llvm_cbe_552_count = 0;
  static  unsigned long long aesl_llvm_cbe_553_count = 0;
  unsigned long long llvm_cbe_tmp__111;
  static  unsigned long long aesl_llvm_cbe_554_count = 0;
  signed long long *llvm_cbe_tmp__112;
  static  unsigned long long aesl_llvm_cbe_555_count = 0;
  unsigned long long llvm_cbe_tmp__113;
  static  unsigned long long aesl_llvm_cbe_556_count = 0;
  static  unsigned long long aesl_llvm_cbe_557_count = 0;
  static  unsigned long long aesl_llvm_cbe_558_count = 0;
  float *llvm_cbe_tmp__114;
  static  unsigned long long aesl_llvm_cbe_559_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge12_count = 0;
  unsigned long long llvm_cbe_storemerge12;
  unsigned long long llvm_cbe_storemerge12__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_560_count = 0;
  float llvm_cbe_tmp__115;
  static  unsigned long long aesl_llvm_cbe_561_count = 0;
  float *llvm_cbe_tmp__116;
  static  unsigned long long aesl_llvm_cbe_562_count = 0;
  float llvm_cbe_tmp__117;
  static  unsigned long long aesl_llvm_cbe_563_count = 0;
  float llvm_cbe_tmp__118;
  static  unsigned long long aesl_llvm_cbe_564_count = 0;
  static  unsigned long long aesl_llvm_cbe_565_count = 0;
  unsigned long long llvm_cbe_tmp__119;
  static  unsigned long long aesl_llvm_cbe_566_count = 0;
  static  unsigned long long aesl_llvm_cbe_567_count = 0;
  static  unsigned long long aesl_llvm_cbe_568_count = 0;
  static  unsigned long long aesl_llvm_cbe_569_count = 0;
  static  unsigned long long aesl_llvm_cbe_570_count = 0;
  unsigned long long llvm_cbe_tmp__120;
  static  unsigned long long aesl_llvm_cbe_571_count = 0;
  static  unsigned long long aesl_llvm_cbe_572_count = 0;
  static  unsigned long long aesl_llvm_cbe_573_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @mat_postmult_diag\n");
  if ((((signed long long )llvm_cbe_An) > ((signed long long )0ull))) {
    llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph4;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph4' to make GCC happy */
llvm_cbe__2e_lr_2e_ph4:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge3 = phi i64 [ %%4, %%.loopexit ], [ 0, %%0  for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_storemerge3_count);
  llvm_cbe_storemerge3 = (unsigned long long )llvm_cbe_storemerge3__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",llvm_cbe_storemerge3);
printf("\n = 0x%I64X",llvm_cbe_tmp__111);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = getelementptr inbounds i64* %%Ap, i64 %%storemerge3, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_547_count);
  llvm_cbe_tmp__109 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_storemerge3))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* %%2, align 8, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_548_count);
  llvm_cbe_tmp__110 = (unsigned long long )*llvm_cbe_tmp__109;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__110);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = add nsw i64 %%storemerge3, 1, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_553_count);
  llvm_cbe_tmp__111 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge3&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__111&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds i64* %%Ap, i64 %%4, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_554_count);
  llvm_cbe_tmp__112 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_tmp__111))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__111));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* %%5, align 8, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_555_count);
  llvm_cbe_tmp__113 = (unsigned long long )*llvm_cbe_tmp__112;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__113);
  if ((((signed long long )llvm_cbe_tmp__110) < ((signed long long )llvm_cbe_tmp__113))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe__2e_loopexit:
  if (((llvm_cbe_tmp__111&18446744073709551615ULL) == (llvm_cbe_An&18446744073709551615ULL))) {
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__111;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph4;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__121:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge12 = phi i64 [ %%3, %%.lr.ph ], [ %%14, %%9  for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_storemerge12_count);
  llvm_cbe_storemerge12 = (unsigned long long )llvm_cbe_storemerge12__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",llvm_cbe_storemerge12);
printf("\n = 0x%I64X",llvm_cbe_tmp__110);
printf("\n = 0x%I64X",llvm_cbe_tmp__119);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load float* %%8, align 4, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_560_count);
  llvm_cbe_tmp__115 = (float )*llvm_cbe_tmp__114;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__115, *(int*)(&llvm_cbe_tmp__115));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds float* %%Ax, i64 %%storemerge12, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_561_count);
  llvm_cbe_tmp__116 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge12))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load float* %%11, align 4, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_562_count);
  llvm_cbe_tmp__117 = (float )*llvm_cbe_tmp__116;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__117, *(int*)(&llvm_cbe_tmp__117));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%12, %%10, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_563_count);
  llvm_cbe_tmp__118 = (float )((float )(llvm_cbe_tmp__117 * llvm_cbe_tmp__115));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__118, *(int*)(&llvm_cbe_tmp__118));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%11, align 4, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_564_count);
  *llvm_cbe_tmp__116 = llvm_cbe_tmp__118;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__118);
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = add nsw i64 %%storemerge12, 1, !dbg !16 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_565_count);
  llvm_cbe_tmp__119 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__119&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* %%5, align 8, !dbg !15 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_570_count);
  llvm_cbe_tmp__120 = (unsigned long long )*llvm_cbe_tmp__112;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__120);
  if ((((signed long long )llvm_cbe_tmp__119) < ((signed long long )llvm_cbe_tmp__120))) {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__119;   /* for PHI node */
    goto llvm_cbe_tmp__121;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds float* %%d, i64 %%storemerge3, !dbg !14 for 0x%I64xth hint within @mat_postmult_diag  --> \n", ++aesl_llvm_cbe_558_count);
  llvm_cbe_tmp__114 = (float *)(&llvm_cbe_d[(((signed long long )llvm_cbe_storemerge3))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}
  llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__110;   /* for PHI node */
  goto llvm_cbe_tmp__121;

  } while (1); /* end of syntactic loop '.lr.ph4' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @mat_postmult_diag}\n");
  return;
}


void mat_vec(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, signed long long llvm_cbe_Am, float *llvm_cbe_x, float *llvm_cbe_y, signed long long llvm_cbe_plus_eq) {
  static  unsigned long long aesl_llvm_cbe_574_count = 0;
  static  unsigned long long aesl_llvm_cbe_575_count = 0;
  static  unsigned long long aesl_llvm_cbe_576_count = 0;
  static  unsigned long long aesl_llvm_cbe_577_count = 0;
  static  unsigned long long aesl_llvm_cbe_578_count = 0;
  static  unsigned long long aesl_llvm_cbe_579_count = 0;
  static  unsigned long long aesl_llvm_cbe_580_count = 0;
  static  unsigned long long aesl_llvm_cbe_581_count = 0;
  static  unsigned long long aesl_llvm_cbe_582_count = 0;
  static  unsigned long long aesl_llvm_cbe_583_count = 0;
  static  unsigned long long aesl_llvm_cbe_584_count = 0;
  static  unsigned long long aesl_llvm_cbe_585_count = 0;
  static  unsigned long long aesl_llvm_cbe_586_count = 0;
  static  unsigned long long aesl_llvm_cbe_587_count = 0;
  static  unsigned long long aesl_llvm_cbe_588_count = 0;
  static  unsigned long long aesl_llvm_cbe_589_count = 0;
  static  unsigned long long aesl_llvm_cbe_590_count = 0;
  static  unsigned long long aesl_llvm_cbe_591_count = 0;
  static  unsigned long long aesl_llvm_cbe_592_count = 0;
  static  unsigned long long aesl_llvm_cbe_593_count = 0;
  static  unsigned long long aesl_llvm_cbe_594_count = 0;
  static  unsigned long long aesl_llvm_cbe_595_count = 0;
  static  unsigned long long aesl_llvm_cbe_596_count = 0;
  static  unsigned long long aesl_llvm_cbe_597_count = 0;
  static  unsigned long long aesl_llvm_cbe_598_count = 0;
  static  unsigned long long aesl_llvm_cbe_599_count = 0;
  static  unsigned long long aesl_llvm_cbe_600_count = 0;
  static  unsigned long long aesl_llvm_cbe_601_count = 0;
  static  unsigned long long aesl_llvm_cbe_602_count = 0;
  static  unsigned long long aesl_llvm_cbe_603_count = 0;
  static  unsigned long long aesl_llvm_cbe_604_count = 0;
  static  unsigned long long aesl_llvm_cbe_605_count = 0;
  static  unsigned long long aesl_llvm_cbe_606_count = 0;
  static  unsigned long long aesl_llvm_cbe_607_count = 0;
  static  unsigned long long aesl_llvm_cbe_608_count = 0;
  static  unsigned long long aesl_llvm_cbe_609_count = 0;
  static  unsigned long long aesl_llvm_cbe_610_count = 0;
  static  unsigned long long aesl_llvm_cbe_611_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge13_count = 0;
  unsigned long long llvm_cbe_storemerge13;
  unsigned long long llvm_cbe_storemerge13__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_612_count = 0;
  float *llvm_cbe_tmp__122;
  static  unsigned long long aesl_llvm_cbe_613_count = 0;
  static  unsigned long long aesl_llvm_cbe_614_count = 0;
  unsigned long long llvm_cbe_tmp__123;
  static  unsigned long long aesl_llvm_cbe_615_count = 0;
  static  unsigned long long aesl_llvm_cbe_616_count = 0;
  static  unsigned long long aesl_llvm_cbe_617_count = 0;
  static  unsigned long long aesl_llvm_cbe_618_count = 0;
  static  unsigned long long aesl_llvm_cbe_619_count = 0;
  static  unsigned long long aesl_llvm_cbe_620_count = 0;
  static  unsigned long long aesl_llvm_cbe_621_count = 0;
  static  unsigned long long aesl_llvm_cbe_622_count = 0;
  static  unsigned long long aesl_llvm_cbe_623_count = 0;
  static  unsigned long long aesl_llvm_cbe_624_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_625_count = 0;
  static  unsigned long long aesl_llvm_cbe_626_count = 0;
  static  unsigned long long aesl_llvm_cbe_627_count = 0;
  signed long long *llvm_cbe_tmp__124;
  static  unsigned long long aesl_llvm_cbe_628_count = 0;
  unsigned long long llvm_cbe_tmp__125;
  static  unsigned long long aesl_llvm_cbe_629_count = 0;
  static  unsigned long long aesl_llvm_cbe_630_count = 0;
  static  unsigned long long aesl_llvm_cbe_631_count = 0;
  static  unsigned long long aesl_llvm_cbe_632_count = 0;
  static  unsigned long long aesl_llvm_cbe_633_count = 0;
  static  unsigned long long aesl_llvm_cbe_634_count = 0;
  static  unsigned long long aesl_llvm_cbe_635_count = 0;
  static  unsigned long long aesl_llvm_cbe_636_count = 0;
  static  unsigned long long aesl_llvm_cbe_637_count = 0;
  static  unsigned long long aesl_llvm_cbe_638_count = 0;
  static  unsigned long long aesl_llvm_cbe_639_count = 0;
  static  unsigned long long aesl_llvm_cbe_640_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge15_2e_us_count = 0;
  unsigned long long llvm_cbe_storemerge15_2e_us;
  unsigned long long llvm_cbe_storemerge15_2e_us__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_641_count = 0;
  signed long long *llvm_cbe_tmp__126;
  static  unsigned long long aesl_llvm_cbe_642_count = 0;
  unsigned long long llvm_cbe_tmp__127;
  static  unsigned long long aesl_llvm_cbe_643_count = 0;
  static  unsigned long long aesl_llvm_cbe_644_count = 0;
  static  unsigned long long aesl_llvm_cbe_645_count = 0;
  static  unsigned long long aesl_llvm_cbe_646_count = 0;
  static  unsigned long long aesl_llvm_cbe_647_count = 0;
  static  unsigned long long aesl_llvm_cbe_648_count = 0;
  static  unsigned long long aesl_llvm_cbe_649_count = 0;
  static  unsigned long long aesl_llvm_cbe_650_count = 0;
  static  unsigned long long aesl_llvm_cbe_651_count = 0;
  static  unsigned long long aesl_llvm_cbe_652_count = 0;
  static  unsigned long long aesl_llvm_cbe_653_count = 0;
  unsigned long long llvm_cbe_tmp__128;
  static  unsigned long long aesl_llvm_cbe_654_count = 0;
  signed long long *llvm_cbe_tmp__129;
  static  unsigned long long aesl_llvm_cbe_655_count = 0;
  unsigned long long llvm_cbe_tmp__130;
  static  unsigned long long aesl_llvm_cbe_656_count = 0;
  static  unsigned long long aesl_llvm_cbe_657_count = 0;
  static  unsigned long long aesl_llvm_cbe_658_count = 0;
  static  unsigned long long aesl_llvm_cbe_659_count = 0;
  static  unsigned long long aesl_llvm_cbe_660_count = 0;
  static  unsigned long long aesl_llvm_cbe_661_count = 0;
  static  unsigned long long aesl_llvm_cbe_662_count = 0;
  static  unsigned long long aesl_llvm_cbe_663_count = 0;
  static  unsigned long long aesl_llvm_cbe_664_count = 0;
  static  unsigned long long aesl_llvm_cbe_665_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge23_2e_us_2e_us_count = 0;
  unsigned long long llvm_cbe_storemerge23_2e_us_2e_us;
  unsigned long long llvm_cbe_storemerge23_2e_us_2e_us__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_666_count = 0;
  float *llvm_cbe_tmp__131;
  static  unsigned long long aesl_llvm_cbe_667_count = 0;
  float llvm_cbe_tmp__132;
  static  unsigned long long aesl_llvm_cbe_668_count = 0;
  float llvm_cbe_tmp__133;
  static  unsigned long long aesl_llvm_cbe_669_count = 0;
  float llvm_cbe_tmp__134;
  static  unsigned long long aesl_llvm_cbe_670_count = 0;
  signed long long *llvm_cbe_tmp__135;
  static  unsigned long long aesl_llvm_cbe_671_count = 0;
  unsigned long long llvm_cbe_tmp__136;
  static  unsigned long long aesl_llvm_cbe_672_count = 0;
  float *llvm_cbe_tmp__137;
  static  unsigned long long aesl_llvm_cbe_673_count = 0;
  float llvm_cbe_tmp__138;
  static  unsigned long long aesl_llvm_cbe_674_count = 0;
  float llvm_cbe_tmp__139;
  static  unsigned long long aesl_llvm_cbe_675_count = 0;
  static  unsigned long long aesl_llvm_cbe_676_count = 0;
  unsigned long long llvm_cbe_tmp__140;
  static  unsigned long long aesl_llvm_cbe_677_count = 0;
  static  unsigned long long aesl_llvm_cbe_678_count = 0;
  static  unsigned long long aesl_llvm_cbe_679_count = 0;
  static  unsigned long long aesl_llvm_cbe_680_count = 0;
  static  unsigned long long aesl_llvm_cbe_681_count = 0;
  static  unsigned long long aesl_llvm_cbe_682_count = 0;
  static  unsigned long long aesl_llvm_cbe_683_count = 0;
  static  unsigned long long aesl_llvm_cbe_684_count = 0;
  static  unsigned long long aesl_llvm_cbe_685_count = 0;
  static  unsigned long long aesl_llvm_cbe_686_count = 0;
  static  unsigned long long aesl_llvm_cbe_687_count = 0;
  unsigned long long llvm_cbe_tmp__141;
  static  unsigned long long aesl_llvm_cbe_688_count = 0;
  static  unsigned long long aesl_llvm_cbe_689_count = 0;
  static  unsigned long long aesl_llvm_cbe_690_count = 0;
  float *llvm_cbe_tmp__142;
  static  unsigned long long aesl_llvm_cbe_691_count = 0;
  static  unsigned long long aesl_llvm_cbe_692_count = 0;
  static  unsigned long long aesl_llvm_cbe_693_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge15_count = 0;
  unsigned long long llvm_cbe_storemerge15;
  unsigned long long llvm_cbe_storemerge15__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_694_count = 0;
  signed long long *llvm_cbe_tmp__143;
  static  unsigned long long aesl_llvm_cbe_695_count = 0;
  unsigned long long llvm_cbe_tmp__144;
  static  unsigned long long aesl_llvm_cbe_696_count = 0;
  static  unsigned long long aesl_llvm_cbe_697_count = 0;
  static  unsigned long long aesl_llvm_cbe_698_count = 0;
  static  unsigned long long aesl_llvm_cbe_699_count = 0;
  static  unsigned long long aesl_llvm_cbe_700_count = 0;
  static  unsigned long long aesl_llvm_cbe_701_count = 0;
  static  unsigned long long aesl_llvm_cbe_702_count = 0;
  static  unsigned long long aesl_llvm_cbe_703_count = 0;
  static  unsigned long long aesl_llvm_cbe_704_count = 0;
  static  unsigned long long aesl_llvm_cbe_705_count = 0;
  static  unsigned long long aesl_llvm_cbe_706_count = 0;
  unsigned long long llvm_cbe_tmp__145;
  static  unsigned long long aesl_llvm_cbe_707_count = 0;
  signed long long *llvm_cbe_tmp__146;
  static  unsigned long long aesl_llvm_cbe_708_count = 0;
  unsigned long long llvm_cbe_tmp__147;
  static  unsigned long long aesl_llvm_cbe_709_count = 0;
  static  unsigned long long aesl_llvm_cbe_710_count = 0;
  static  unsigned long long aesl_llvm_cbe_711_count = 0;
  float *llvm_cbe_tmp__148;
  static  unsigned long long aesl_llvm_cbe_712_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge23_count = 0;
  unsigned long long llvm_cbe_storemerge23;
  unsigned long long llvm_cbe_storemerge23__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_713_count = 0;
  float *llvm_cbe_tmp__149;
  static  unsigned long long aesl_llvm_cbe_714_count = 0;
  float llvm_cbe_tmp__150;
  static  unsigned long long aesl_llvm_cbe_715_count = 0;
  float llvm_cbe_tmp__151;
  static  unsigned long long aesl_llvm_cbe_716_count = 0;
  float llvm_cbe_tmp__152;
  static  unsigned long long aesl_llvm_cbe_717_count = 0;
  signed long long *llvm_cbe_tmp__153;
  static  unsigned long long aesl_llvm_cbe_718_count = 0;
  unsigned long long llvm_cbe_tmp__154;
  static  unsigned long long aesl_llvm_cbe_719_count = 0;
  float *llvm_cbe_tmp__155;
  static  unsigned long long aesl_llvm_cbe_720_count = 0;
  float llvm_cbe_tmp__156;
  static  unsigned long long aesl_llvm_cbe_721_count = 0;
  float llvm_cbe_tmp__157;
  static  unsigned long long aesl_llvm_cbe_722_count = 0;
  static  unsigned long long aesl_llvm_cbe_723_count = 0;
  unsigned long long llvm_cbe_tmp__158;
  static  unsigned long long aesl_llvm_cbe_724_count = 0;
  static  unsigned long long aesl_llvm_cbe_725_count = 0;
  static  unsigned long long aesl_llvm_cbe_726_count = 0;
  static  unsigned long long aesl_llvm_cbe_727_count = 0;
  static  unsigned long long aesl_llvm_cbe_728_count = 0;
  static  unsigned long long aesl_llvm_cbe_729_count = 0;
  static  unsigned long long aesl_llvm_cbe_730_count = 0;
  static  unsigned long long aesl_llvm_cbe_731_count = 0;
  static  unsigned long long aesl_llvm_cbe_732_count = 0;
  static  unsigned long long aesl_llvm_cbe_733_count = 0;
  static  unsigned long long aesl_llvm_cbe_734_count = 0;
  unsigned long long llvm_cbe_tmp__159;
  static  unsigned long long aesl_llvm_cbe_735_count = 0;
  static  unsigned long long aesl_llvm_cbe_736_count = 0;
  static  unsigned long long aesl_llvm_cbe_737_count = 0;
  static  unsigned long long aesl_llvm_cbe_738_count = 0;
  static  unsigned long long aesl_llvm_cbe_739_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @mat_vec\n");
  if (((llvm_cbe_plus_eq&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe__2e_preheader11;
  } else {
    goto llvm_cbe_tmp__160;
  }

llvm_cbe__2e_preheader11:
  if ((((signed long long )llvm_cbe_Am) > ((signed long long )0ull))) {
    llvm_cbe_storemerge13__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph14;
  } else {
    goto llvm_cbe__2e_loopexit12;
  }

  do {     /* Syntactic loop '.lr.ph14' to make GCC happy */
llvm_cbe__2e_lr_2e_ph14:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge13 = phi i64 [ %%4, %%.lr.ph14 ], [ 0, %%.preheader11  for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_storemerge13_count);
  llvm_cbe_storemerge13 = (unsigned long long )llvm_cbe_storemerge13__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",llvm_cbe_storemerge13);
printf("\n = 0x%I64X",llvm_cbe_tmp__123);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%y, i64 %%storemerge13, !dbg !16 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_612_count);
  llvm_cbe_tmp__122 = (float *)(&llvm_cbe_y[(((signed long long )llvm_cbe_storemerge13))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",((signed long long )llvm_cbe_storemerge13));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%3, align 4, !dbg !16 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_613_count);
  *llvm_cbe_tmp__122 = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = add nsw i64 %%storemerge13, 1, !dbg !17 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_614_count);
  llvm_cbe_tmp__123 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge13&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__123&18446744073709551615ull)));
  if (((llvm_cbe_tmp__123&18446744073709551615ULL) == (llvm_cbe_Am&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit12;
  } else {
    llvm_cbe_storemerge13__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__123;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph14;
  }

  } while (1); /* end of syntactic loop '.lr.ph14' */
llvm_cbe__2e_loopexit12:
  goto llvm_cbe_tmp__160;

llvm_cbe_tmp__160:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds i64* %%Ap, i64 %%An, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_627_count);
  llvm_cbe_tmp__124 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_An))]);
if (AESL_DEBUG_TRACE) {
printf("\nAn = 0x%I64X",((signed long long )llvm_cbe_An));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* %%6, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_628_count);
  llvm_cbe_tmp__125 = (unsigned long long )*llvm_cbe_tmp__124;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__125);
  if (((llvm_cbe_tmp__125&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__161;
  } else {
    goto llvm_cbe__2e_preheader;
  }

llvm_cbe__2e_preheader:
  if ((((signed long long )llvm_cbe_An) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph6;
  } else {
    goto llvm_cbe__2e_loopexit4;
  }

llvm_cbe__2e_lr_2e_ph6:
  if (((llvm_cbe_plus_eq&18446744073709551615ULL) == (18446744073709551615ull&18446744073709551615ULL))) {
    llvm_cbe_storemerge15_2e_us__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6_2e_split_2e_us;
  } else {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6_2e__2e_lr_2e_ph6_2e_split_crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph6.split.us' to make GCC happy */
llvm_cbe__2e_lr_2e_ph6_2e_split_2e_us:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge15.us = phi i64 [ %%13, %%.loopexit.us ], [ 0, %%.lr.ph6  for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_storemerge15_2e_us_count);
  llvm_cbe_storemerge15_2e_us = (unsigned long long )llvm_cbe_storemerge15_2e_us__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15.us = 0x%I64X",llvm_cbe_storemerge15_2e_us);
printf("\n = 0x%I64X",llvm_cbe_tmp__128);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds i64* %%Ap, i64 %%storemerge15.us, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_641_count);
  llvm_cbe_tmp__126 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_storemerge15_2e_us))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15.us = 0x%I64X",((signed long long )llvm_cbe_storemerge15_2e_us));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load i64* %%11, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_642_count);
  llvm_cbe_tmp__127 = (unsigned long long )*llvm_cbe_tmp__126;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__127);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = add nsw i64 %%storemerge15.us, 1, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_653_count);
  llvm_cbe_tmp__128 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge15_2e_us&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__128&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds i64* %%Ap, i64 %%13, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_654_count);
  llvm_cbe_tmp__129 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_tmp__128))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__128));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* %%14, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_655_count);
  llvm_cbe_tmp__130 = (unsigned long long )*llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__130);
  if ((((signed long long )llvm_cbe_tmp__127) < ((signed long long )llvm_cbe_tmp__130))) {
    goto llvm_cbe__2e_lr_2e_ph_2e_split_2e_us_2e_us;
  } else {
    goto llvm_cbe__2e_loopexit_2e_us;
  }

llvm_cbe__2e_loopexit_2e_us:
  if ((((signed long long )llvm_cbe_tmp__128) < ((signed long long )llvm_cbe_An))) {
    llvm_cbe_storemerge15_2e_us__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__128;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6_2e_split_2e_us;
  } else {
    goto llvm_cbe__2e__2e_loopexit4_crit_edge;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__162:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge23.us.us = phi i64 [ %%12, %%.lr.ph.split.us.us ], [ %%28, %%18  for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_storemerge23_2e_us_2e_us_count);
  llvm_cbe_storemerge23_2e_us_2e_us = (unsigned long long )llvm_cbe_storemerge23_2e_us_2e_us__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23.us.us = 0x%I64X",llvm_cbe_storemerge23_2e_us_2e_us);
printf("\n = 0x%I64X",llvm_cbe_tmp__127);
printf("\n = 0x%I64X",llvm_cbe_tmp__140);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = getelementptr inbounds float* %%Ax, i64 %%storemerge23.us.us, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_666_count);
  llvm_cbe_tmp__131 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge23_2e_us_2e_us))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23.us.us = 0x%I64X",((signed long long )llvm_cbe_storemerge23_2e_us_2e_us));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = load float* %%19, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_667_count);
  llvm_cbe_tmp__132 = (float )*llvm_cbe_tmp__131;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__132, *(int*)(&llvm_cbe_tmp__132));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load float* %%31, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_668_count);
  llvm_cbe_tmp__133 = (float )*llvm_cbe_tmp__142;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__133, *(int*)(&llvm_cbe_tmp__133));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fmul float %%20, %%21, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_669_count);
  llvm_cbe_tmp__134 = (float )((float )(llvm_cbe_tmp__132 * llvm_cbe_tmp__133));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__134, *(int*)(&llvm_cbe_tmp__134));
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = getelementptr inbounds i64* %%Ai, i64 %%storemerge23.us.us, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_670_count);
  llvm_cbe_tmp__135 = (signed long long *)(&llvm_cbe_Ai[(((signed long long )llvm_cbe_storemerge23_2e_us_2e_us))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23.us.us = 0x%I64X",((signed long long )llvm_cbe_storemerge23_2e_us_2e_us));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = load i64* %%23, align 8, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_671_count);
  llvm_cbe_tmp__136 = (unsigned long long )*llvm_cbe_tmp__135;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__136);
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = getelementptr inbounds float* %%y, i64 %%24, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_672_count);
  llvm_cbe_tmp__137 = (float *)(&llvm_cbe_y[(((signed long long )llvm_cbe_tmp__136))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__136));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load float* %%25, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_673_count);
  llvm_cbe_tmp__138 = (float )*llvm_cbe_tmp__137;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__138, *(int*)(&llvm_cbe_tmp__138));
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = fsub float %%26, %%22, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_674_count);
  llvm_cbe_tmp__139 = (float )((float )(llvm_cbe_tmp__138 - llvm_cbe_tmp__134));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__139, *(int*)(&llvm_cbe_tmp__139));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%27, float* %%25, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_675_count);
  *llvm_cbe_tmp__137 = llvm_cbe_tmp__139;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__139);
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = add nsw i64 %%storemerge23.us.us, 1, !dbg !17 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_676_count);
  llvm_cbe_tmp__140 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge23_2e_us_2e_us&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__140&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = load i64* %%14, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_687_count);
  llvm_cbe_tmp__141 = (unsigned long long )*llvm_cbe_tmp__129;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__141);
  if ((((signed long long )llvm_cbe_tmp__140) < ((signed long long )llvm_cbe_tmp__141))) {
    llvm_cbe_storemerge23_2e_us_2e_us__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__140;   /* for PHI node */
    goto llvm_cbe_tmp__162;
  } else {
    goto llvm_cbe__2e_loopexit_2e_us;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_lr_2e_ph_2e_split_2e_us_2e_us:
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = getelementptr inbounds float* %%x, i64 %%storemerge15.us, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_690_count);
  llvm_cbe_tmp__142 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_storemerge15_2e_us))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15.us = 0x%I64X",((signed long long )llvm_cbe_storemerge15_2e_us));
}
  llvm_cbe_storemerge23_2e_us_2e_us__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__127;   /* for PHI node */
  goto llvm_cbe_tmp__162;

  } while (1); /* end of syntactic loop '.lr.ph6.split.us' */
  do {     /* Syntactic loop '.lr.ph6..lr.ph6.split_crit_edge' to make GCC happy */
llvm_cbe__2e_lr_2e_ph6_2e__2e_lr_2e_ph6_2e_split_crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge15 = phi i64 [ %%35, %%.loopexit ], [ 0, %%.lr.ph6  for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_storemerge15_count);
  llvm_cbe_storemerge15 = (unsigned long long )llvm_cbe_storemerge15__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",llvm_cbe_storemerge15);
printf("\n = 0x%I64X",llvm_cbe_tmp__145);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = getelementptr inbounds i64* %%Ap, i64 %%storemerge15, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_694_count);
  llvm_cbe_tmp__143 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = load i64* %%33, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_695_count);
  llvm_cbe_tmp__144 = (unsigned long long )*llvm_cbe_tmp__143;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__144);
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = add nsw i64 %%storemerge15, 1, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_706_count);
  llvm_cbe_tmp__145 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge15&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__145&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = getelementptr inbounds i64* %%Ap, i64 %%35, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_707_count);
  llvm_cbe_tmp__146 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_tmp__145))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__145));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load i64* %%36, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_708_count);
  llvm_cbe_tmp__147 = (unsigned long long )*llvm_cbe_tmp__146;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__147);
  if ((((signed long long )llvm_cbe_tmp__144) < ((signed long long )llvm_cbe_tmp__147))) {
    goto llvm_cbe__2e_lr_2e_ph_2e__2e_lr_2e_ph_2e_split_crit_edge;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe__2e_loopexit:
  if ((((signed long long )llvm_cbe_tmp__145) < ((signed long long )llvm_cbe_An))) {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__145;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph6_2e__2e_lr_2e_ph6_2e_split_crit_edge;
  } else {
    goto llvm_cbe__2e__2e_loopexit4_crit_edge;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__163:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge23 = phi i64 [ %%34, %%.lr.ph..lr.ph.split_crit_edge ], [ %%50, %%40  for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_storemerge23_count);
  llvm_cbe_storemerge23 = (unsigned long long )llvm_cbe_storemerge23__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",llvm_cbe_storemerge23);
printf("\n = 0x%I64X",llvm_cbe_tmp__144);
printf("\n = 0x%I64X",llvm_cbe_tmp__158);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%41 = getelementptr inbounds float* %%Ax, i64 %%storemerge23, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_713_count);
  llvm_cbe_tmp__149 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge23))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",((signed long long )llvm_cbe_storemerge23));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%42 = load float* %%41, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_714_count);
  llvm_cbe_tmp__150 = (float )*llvm_cbe_tmp__149;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__150, *(int*)(&llvm_cbe_tmp__150));
if (AESL_DEBUG_TRACE)
printf("\n  %%43 = load float* %%39, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_715_count);
  llvm_cbe_tmp__151 = (float )*llvm_cbe_tmp__148;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__151, *(int*)(&llvm_cbe_tmp__151));
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = fmul float %%42, %%43, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_716_count);
  llvm_cbe_tmp__152 = (float )((float )(llvm_cbe_tmp__150 * llvm_cbe_tmp__151));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__152, *(int*)(&llvm_cbe_tmp__152));
if (AESL_DEBUG_TRACE)
printf("\n  %%45 = getelementptr inbounds i64* %%Ai, i64 %%storemerge23, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_717_count);
  llvm_cbe_tmp__153 = (signed long long *)(&llvm_cbe_Ai[(((signed long long )llvm_cbe_storemerge23))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",((signed long long )llvm_cbe_storemerge23));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%46 = load i64* %%45, align 8, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_718_count);
  llvm_cbe_tmp__154 = (unsigned long long )*llvm_cbe_tmp__153;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__154);
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = getelementptr inbounds float* %%y, i64 %%46, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_719_count);
  llvm_cbe_tmp__155 = (float *)(&llvm_cbe_y[(((signed long long )llvm_cbe_tmp__154))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__154));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = load float* %%47, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_720_count);
  llvm_cbe_tmp__156 = (float )*llvm_cbe_tmp__155;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__156, *(int*)(&llvm_cbe_tmp__156));
if (AESL_DEBUG_TRACE)
printf("\n  %%49 = fadd float %%48, %%44, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_721_count);
  llvm_cbe_tmp__157 = (float )((float )(llvm_cbe_tmp__156 + llvm_cbe_tmp__152));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__157, *(int*)(&llvm_cbe_tmp__157));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%49, float* %%47, align 4, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_722_count);
  *llvm_cbe_tmp__155 = llvm_cbe_tmp__157;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__157);
if (AESL_DEBUG_TRACE)
printf("\n  %%50 = add nsw i64 %%storemerge23, 1, !dbg !17 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_723_count);
  llvm_cbe_tmp__158 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge23&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__158&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%51 = load i64* %%36, align 8, !dbg !15 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_734_count);
  llvm_cbe_tmp__159 = (unsigned long long )*llvm_cbe_tmp__146;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__159);
  if ((((signed long long )llvm_cbe_tmp__158) < ((signed long long )llvm_cbe_tmp__159))) {
    llvm_cbe_storemerge23__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__158;   /* for PHI node */
    goto llvm_cbe_tmp__163;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_lr_2e_ph_2e__2e_lr_2e_ph_2e_split_crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = getelementptr inbounds float* %%x, i64 %%storemerge15, !dbg !14 for 0x%I64xth hint within @mat_vec  --> \n", ++aesl_llvm_cbe_711_count);
  llvm_cbe_tmp__148 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
  llvm_cbe_storemerge23__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__144;   /* for PHI node */
  goto llvm_cbe_tmp__163;

  } while (1); /* end of syntactic loop '.lr.ph6..lr.ph6.split_crit_edge' */
llvm_cbe__2e__2e_loopexit4_crit_edge:
  goto llvm_cbe__2e_loopexit4;

llvm_cbe__2e_loopexit4:
  goto llvm_cbe_tmp__161;

llvm_cbe_tmp__161:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @mat_vec}\n");
  return;
}


void mat_tpose_vec(float *llvm_cbe_Ax, signed long long *llvm_cbe_Ap, signed long long *llvm_cbe_Ai, signed long long llvm_cbe_An, signed long long llvm_cbe_Am, float *llvm_cbe_x, float *llvm_cbe_y, signed long long llvm_cbe_plus_eq, signed long long llvm_cbe_skip_diag) {
  static  unsigned long long aesl_llvm_cbe_740_count = 0;
  static  unsigned long long aesl_llvm_cbe_741_count = 0;
  static  unsigned long long aesl_llvm_cbe_742_count = 0;
  static  unsigned long long aesl_llvm_cbe_743_count = 0;
  static  unsigned long long aesl_llvm_cbe_744_count = 0;
  static  unsigned long long aesl_llvm_cbe_745_count = 0;
  static  unsigned long long aesl_llvm_cbe_746_count = 0;
  static  unsigned long long aesl_llvm_cbe_747_count = 0;
  static  unsigned long long aesl_llvm_cbe_748_count = 0;
  static  unsigned long long aesl_llvm_cbe_749_count = 0;
  static  unsigned long long aesl_llvm_cbe_750_count = 0;
  static  unsigned long long aesl_llvm_cbe_751_count = 0;
  static  unsigned long long aesl_llvm_cbe_752_count = 0;
  static  unsigned long long aesl_llvm_cbe_753_count = 0;
  static  unsigned long long aesl_llvm_cbe_754_count = 0;
  static  unsigned long long aesl_llvm_cbe_755_count = 0;
  static  unsigned long long aesl_llvm_cbe_756_count = 0;
  static  unsigned long long aesl_llvm_cbe_757_count = 0;
  static  unsigned long long aesl_llvm_cbe_758_count = 0;
  static  unsigned long long aesl_llvm_cbe_759_count = 0;
  static  unsigned long long aesl_llvm_cbe_760_count = 0;
  static  unsigned long long aesl_llvm_cbe_761_count = 0;
  static  unsigned long long aesl_llvm_cbe_762_count = 0;
  static  unsigned long long aesl_llvm_cbe_763_count = 0;
  static  unsigned long long aesl_llvm_cbe_764_count = 0;
  static  unsigned long long aesl_llvm_cbe_765_count = 0;
  static  unsigned long long aesl_llvm_cbe_766_count = 0;
  static  unsigned long long aesl_llvm_cbe_767_count = 0;
  static  unsigned long long aesl_llvm_cbe_768_count = 0;
  static  unsigned long long aesl_llvm_cbe_769_count = 0;
  static  unsigned long long aesl_llvm_cbe_770_count = 0;
  static  unsigned long long aesl_llvm_cbe_771_count = 0;
  static  unsigned long long aesl_llvm_cbe_772_count = 0;
  static  unsigned long long aesl_llvm_cbe_773_count = 0;
  static  unsigned long long aesl_llvm_cbe_774_count = 0;
  static  unsigned long long aesl_llvm_cbe_775_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge9_count = 0;
  unsigned long long llvm_cbe_storemerge9;
  unsigned long long llvm_cbe_storemerge9__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_776_count = 0;
  float *llvm_cbe_tmp__164;
  static  unsigned long long aesl_llvm_cbe_777_count = 0;
  static  unsigned long long aesl_llvm_cbe_778_count = 0;
  unsigned long long llvm_cbe_tmp__165;
  static  unsigned long long aesl_llvm_cbe_779_count = 0;
  static  unsigned long long aesl_llvm_cbe_780_count = 0;
  static  unsigned long long aesl_llvm_cbe_781_count = 0;
  static  unsigned long long aesl_llvm_cbe_782_count = 0;
  static  unsigned long long aesl_llvm_cbe_783_count = 0;
  static  unsigned long long aesl_llvm_cbe_784_count = 0;
  static  unsigned long long aesl_llvm_cbe_785_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond11_count = 0;
  static  unsigned long long aesl_llvm_cbe_786_count = 0;
  static  unsigned long long aesl_llvm_cbe_787_count = 0;
  static  unsigned long long aesl_llvm_cbe_788_count = 0;
  signed long long *llvm_cbe_tmp__166;
  static  unsigned long long aesl_llvm_cbe_789_count = 0;
  unsigned long long llvm_cbe_tmp__167;
  static  unsigned long long aesl_llvm_cbe_790_count = 0;
  static  unsigned long long aesl_llvm_cbe_791_count = 0;
  static  unsigned long long aesl_llvm_cbe_792_count = 0;
  static  unsigned long long aesl_llvm_cbe_793_count = 0;
  static  unsigned long long aesl_llvm_cbe_794_count = 0;
  static  unsigned long long aesl_llvm_cbe_795_count = 0;
  static  unsigned long long aesl_llvm_cbe_796_count = 0;
  static  unsigned long long aesl_llvm_cbe_797_count = 0;
  static  unsigned long long aesl_llvm_cbe_798_count = 0;
  static  unsigned long long aesl_llvm_cbe_799_count = 0;
  static  unsigned long long aesl_llvm_cbe_800_count = 0;
  static  unsigned long long aesl_llvm_cbe_801_count = 0;
  static  unsigned long long aesl_llvm_cbe_802_count = 0;
  static  unsigned long long aesl_llvm_cbe_803_count = 0;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_804_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge15_count = 0;
  unsigned long long llvm_cbe_storemerge15;
  unsigned long long llvm_cbe_storemerge15__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_805_count = 0;
  signed long long *llvm_cbe_tmp__168;
  static  unsigned long long aesl_llvm_cbe_806_count = 0;
  unsigned long long llvm_cbe_tmp__169;
  static  unsigned long long aesl_llvm_cbe_807_count = 0;
  static  unsigned long long aesl_llvm_cbe_808_count = 0;
  static  unsigned long long aesl_llvm_cbe_809_count = 0;
  static  unsigned long long aesl_llvm_cbe_810_count = 0;
  static  unsigned long long aesl_llvm_cbe_811_count = 0;
  static  unsigned long long aesl_llvm_cbe_812_count = 0;
  static  unsigned long long aesl_llvm_cbe_813_count = 0;
  unsigned long long llvm_cbe_tmp__170;
  static  unsigned long long aesl_llvm_cbe_814_count = 0;
  signed long long *llvm_cbe_tmp__171;
  static  unsigned long long aesl_llvm_cbe_815_count = 0;
  unsigned long long llvm_cbe_tmp__172;
  static  unsigned long long aesl_llvm_cbe_816_count = 0;
  static  unsigned long long aesl_llvm_cbe_817_count = 0;
  static  unsigned long long aesl_llvm_cbe_818_count = 0;
  float *llvm_cbe_tmp__173;
  static  unsigned long long aesl_llvm_cbe_819_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge23_count = 0;
  unsigned long long llvm_cbe_storemerge23;
  unsigned long long llvm_cbe_storemerge23__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_820_count = 0;
  signed long long *llvm_cbe_tmp__174;
  static  unsigned long long aesl_llvm_cbe_821_count = 0;
  unsigned long long llvm_cbe_tmp__175;
  static  unsigned long long aesl_llvm_cbe_822_count = 0;
  static  unsigned long long aesl_llvm_cbe_823_count = 0;
  static  unsigned long long aesl_llvm_cbe_824_count = 0;
  static  unsigned long long aesl_llvm_cbe_825_count = 0;
  static  unsigned long long aesl_llvm_cbe_826_count = 0;
  static  unsigned long long aesl_llvm_cbe_827_count = 0;
  static  unsigned long long aesl_llvm_cbe_828_count = 0;
  static  unsigned long long aesl_llvm_cbe_829_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_830_count = 0;
  static  unsigned long long aesl_llvm_cbe_831_count = 0;
  static  unsigned long long aesl_llvm_cbe_832_count = 0;
  float *llvm_cbe_tmp__176;
  static  unsigned long long aesl_llvm_cbe_833_count = 0;
  float llvm_cbe_tmp__177;
  static  unsigned long long aesl_llvm_cbe_834_count = 0;
  float *llvm_cbe_tmp__178;
  static  unsigned long long aesl_llvm_cbe_835_count = 0;
  float llvm_cbe_tmp__179;
  static  unsigned long long aesl_llvm_cbe_836_count = 0;
  float llvm_cbe_tmp__180;
  static  unsigned long long aesl_llvm_cbe_837_count = 0;
  float llvm_cbe_tmp__181;
  static  unsigned long long aesl_llvm_cbe_838_count = 0;
  float llvm_cbe_tmp__182;
  static  unsigned long long aesl_llvm_cbe_839_count = 0;
  static  unsigned long long aesl_llvm_cbe_840_count = 0;
  float *llvm_cbe_tmp__183;
  static  unsigned long long aesl_llvm_cbe_841_count = 0;
  float llvm_cbe_tmp__184;
  static  unsigned long long aesl_llvm_cbe_842_count = 0;
  float *llvm_cbe_tmp__185;
  static  unsigned long long aesl_llvm_cbe_843_count = 0;
  float llvm_cbe_tmp__186;
  static  unsigned long long aesl_llvm_cbe_844_count = 0;
  float llvm_cbe_tmp__187;
  static  unsigned long long aesl_llvm_cbe_845_count = 0;
  float llvm_cbe_tmp__188;
  static  unsigned long long aesl_llvm_cbe_846_count = 0;
  float llvm_cbe_tmp__189;
  static  unsigned long long aesl_llvm_cbe_847_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_848_count = 0;
  static  unsigned long long aesl_llvm_cbe_849_count = 0;
  static  unsigned long long aesl_llvm_cbe_850_count = 0;
  unsigned long long llvm_cbe_tmp__190;
  static  unsigned long long aesl_llvm_cbe_851_count = 0;
  static  unsigned long long aesl_llvm_cbe_852_count = 0;
  static  unsigned long long aesl_llvm_cbe_853_count = 0;
  static  unsigned long long aesl_llvm_cbe_854_count = 0;
  static  unsigned long long aesl_llvm_cbe_855_count = 0;
  static  unsigned long long aesl_llvm_cbe_856_count = 0;
  static  unsigned long long aesl_llvm_cbe_857_count = 0;
  unsigned long long llvm_cbe_tmp__191;
  static  unsigned long long aesl_llvm_cbe_858_count = 0;
  static  unsigned long long aesl_llvm_cbe_859_count = 0;
  static  unsigned long long aesl_llvm_cbe_860_count = 0;
  static  unsigned long long aesl_llvm_cbe_861_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @mat_tpose_vec\n");
  if (((llvm_cbe_plus_eq&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe__2e_preheader7;
  } else {
    goto llvm_cbe_tmp__192;
  }

llvm_cbe__2e_preheader7:
  if ((((signed long long )llvm_cbe_An) > ((signed long long )0ull))) {
    llvm_cbe_storemerge9__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph10;
  } else {
    goto llvm_cbe__2e_loopexit8;
  }

  do {     /* Syntactic loop '.lr.ph10' to make GCC happy */
llvm_cbe__2e_lr_2e_ph10:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge9 = phi i64 [ %%4, %%.lr.ph10 ], [ 0, %%.preheader7  for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_storemerge9_count);
  llvm_cbe_storemerge9 = (unsigned long long )llvm_cbe_storemerge9__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge9 = 0x%I64X",llvm_cbe_storemerge9);
printf("\n = 0x%I64X",llvm_cbe_tmp__165);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%y, i64 %%storemerge9, !dbg !16 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_776_count);
  llvm_cbe_tmp__164 = (float *)(&llvm_cbe_y[(((signed long long )llvm_cbe_storemerge9))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge9 = 0x%I64X",((signed long long )llvm_cbe_storemerge9));
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%3, align 4, !dbg !16 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_777_count);
  *llvm_cbe_tmp__164 = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = add nsw i64 %%storemerge9, 1, !dbg !17 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_778_count);
  llvm_cbe_tmp__165 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge9&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__165&18446744073709551615ull)));
  if (((llvm_cbe_tmp__165&18446744073709551615ULL) == (llvm_cbe_An&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit8;
  } else {
    llvm_cbe_storemerge9__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__165;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph10;
  }

  } while (1); /* end of syntactic loop '.lr.ph10' */
llvm_cbe__2e_loopexit8:
  goto llvm_cbe_tmp__192;

llvm_cbe_tmp__192:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds i64* %%Ap, i64 %%An, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_788_count);
  llvm_cbe_tmp__166 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_An))]);
if (AESL_DEBUG_TRACE) {
printf("\nAn = 0x%I64X",((signed long long )llvm_cbe_An));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* %%6, align 8, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_789_count);
  llvm_cbe_tmp__167 = (unsigned long long )*llvm_cbe_tmp__166;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__167);
  if (((llvm_cbe_tmp__167&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__193;
  } else {
    goto llvm_cbe__2e_preheader;
  }

llvm_cbe__2e_preheader:
  if ((((signed long long )llvm_cbe_An) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph6;
  } else {
    goto llvm_cbe__2e_loopexit4;
  }

llvm_cbe__2e_lr_2e_ph6:
  llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__194;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__194:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge15 = phi i64 [ 0, %%.lr.ph6 ], [ %%15, %%.loopexit  for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_storemerge15_count);
  llvm_cbe_storemerge15 = (unsigned long long )llvm_cbe_storemerge15__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",llvm_cbe_storemerge15);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__170);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds i64* %%Ap, i64 %%storemerge15, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_805_count);
  llvm_cbe_tmp__168 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = load i64* %%13, align 8, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_806_count);
  llvm_cbe_tmp__169 = (unsigned long long )*llvm_cbe_tmp__168;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__169);
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = add nsw i64 %%storemerge15, 1, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_813_count);
  llvm_cbe_tmp__170 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge15&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__170&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds i64* %%Ap, i64 %%15, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_814_count);
  llvm_cbe_tmp__171 = (signed long long *)(&llvm_cbe_Ap[(((signed long long )llvm_cbe_tmp__170))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__170));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load i64* %%16, align 8, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_815_count);
  llvm_cbe_tmp__172 = (unsigned long long )*llvm_cbe_tmp__171;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__172);
  if ((((signed long long )llvm_cbe_tmp__169) < ((signed long long )llvm_cbe_tmp__172))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe__2e_loopexit:
  if (((llvm_cbe_tmp__170&18446744073709551615ULL) == (llvm_cbe_An&18446744073709551615ULL))) {
    goto llvm_cbe__2e_loopexit4;
  } else {
    llvm_cbe_storemerge15__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__170;   /* for PHI node */
    goto llvm_cbe_tmp__194;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__195:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge23 = phi i64 [ %%14, %%.lr.ph ], [ %%43, %%42  for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_storemerge23_count);
  llvm_cbe_storemerge23 = (unsigned long long )llvm_cbe_storemerge23__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",llvm_cbe_storemerge23);
printf("\n = 0x%I64X",llvm_cbe_tmp__169);
printf("\n = 0x%I64X",llvm_cbe_tmp__190);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds i64* %%Ai, i64 %%storemerge23, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_820_count);
  llvm_cbe_tmp__174 = (signed long long *)(&llvm_cbe_Ai[(((signed long long )llvm_cbe_storemerge23))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",((signed long long )llvm_cbe_storemerge23));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = load i64* %%21, align 8, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_821_count);
  llvm_cbe_tmp__175 = (unsigned long long )*llvm_cbe_tmp__174;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__175);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%10, %%23, !dbg !17 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_skip_diag&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__175&18446744073709551615ULL) == (llvm_cbe_storemerge15&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__196;
  } else {
    goto llvm_cbe_tmp__197;
  }

llvm_cbe_tmp__196:
if (AESL_DEBUG_TRACE)
printf("\n  %%43 = add nsw i64 %%storemerge23, 1, !dbg !18 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_850_count);
  llvm_cbe_tmp__190 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge23&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__190&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = load i64* %%16, align 8, !dbg !15 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_857_count);
  llvm_cbe_tmp__191 = (unsigned long long )*llvm_cbe_tmp__171;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__191);
  if ((((signed long long )llvm_cbe_tmp__190) < ((signed long long )llvm_cbe_tmp__191))) {
    llvm_cbe_storemerge23__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__190;   /* for PHI node */
    goto llvm_cbe_tmp__195;
  } else {
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe_tmp__198:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%40, %%33 ], [ %%32, %%25  for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__189);
printf("\n = %f",llvm_cbe_tmp__182);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%storemerge, float* %%19, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_848_count);
  *llvm_cbe_tmp__173 = llvm_cbe_storemerge;
if (AESL_DEBUG_TRACE)
printf("\nstoremerge = %f\n", llvm_cbe_storemerge);
  goto llvm_cbe_tmp__196;

llvm_cbe_tmp__199:
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = getelementptr inbounds float* %%Ax, i64 %%storemerge23, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_832_count);
  llvm_cbe_tmp__176 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge23))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",((signed long long )llvm_cbe_storemerge23));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = load float* %%26, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_833_count);
  llvm_cbe_tmp__177 = (float )*llvm_cbe_tmp__176;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__177, *(int*)(&llvm_cbe_tmp__177));
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = getelementptr inbounds float* %%x, i64 %%22, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_834_count);
  llvm_cbe_tmp__178 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_tmp__175))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__175));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = load float* %%28, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_835_count);
  llvm_cbe_tmp__179 = (float )*llvm_cbe_tmp__178;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__179, *(int*)(&llvm_cbe_tmp__179));
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = fmul float %%27, %%29, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_836_count);
  llvm_cbe_tmp__180 = (float )((float )(llvm_cbe_tmp__177 * llvm_cbe_tmp__179));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__180, *(int*)(&llvm_cbe_tmp__180));
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%19, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_837_count);
  llvm_cbe_tmp__181 = (float )*llvm_cbe_tmp__173;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__181, *(int*)(&llvm_cbe_tmp__181));
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = fsub float %%31, %%30, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_838_count);
  llvm_cbe_tmp__182 = (float )((float )(llvm_cbe_tmp__181 - llvm_cbe_tmp__180));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__182, *(int*)(&llvm_cbe_tmp__182));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__182;   /* for PHI node */
  goto llvm_cbe_tmp__198;

llvm_cbe_tmp__197:
  if (((llvm_cbe_plus_eq&18446744073709551615ULL) == (18446744073709551615ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__199;
  } else {
    goto llvm_cbe_tmp__200;
  }

llvm_cbe_tmp__200:
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = getelementptr inbounds float* %%Ax, i64 %%storemerge23, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_840_count);
  llvm_cbe_tmp__183 = (float *)(&llvm_cbe_Ax[(((signed long long )llvm_cbe_storemerge23))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge23 = 0x%I64X",((signed long long )llvm_cbe_storemerge23));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = load float* %%34, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_841_count);
  llvm_cbe_tmp__184 = (float )*llvm_cbe_tmp__183;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__184, *(int*)(&llvm_cbe_tmp__184));
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = getelementptr inbounds float* %%x, i64 %%22, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_842_count);
  llvm_cbe_tmp__185 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_tmp__175))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__175));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load float* %%36, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_843_count);
  llvm_cbe_tmp__186 = (float )*llvm_cbe_tmp__185;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__186, *(int*)(&llvm_cbe_tmp__186));
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = fmul float %%35, %%37, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_844_count);
  llvm_cbe_tmp__187 = (float )((float )(llvm_cbe_tmp__184 * llvm_cbe_tmp__186));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__187, *(int*)(&llvm_cbe_tmp__187));
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = load float* %%19, align 4, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_845_count);
  llvm_cbe_tmp__188 = (float )*llvm_cbe_tmp__173;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__188, *(int*)(&llvm_cbe_tmp__188));
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = fadd float %%39, %%38, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_846_count);
  llvm_cbe_tmp__189 = (float )((float )(llvm_cbe_tmp__188 + llvm_cbe_tmp__187));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__189, *(int*)(&llvm_cbe_tmp__189));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__189;   /* for PHI node */
  goto llvm_cbe_tmp__198;

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = getelementptr inbounds float* %%y, i64 %%storemerge15, !dbg !14 for 0x%I64xth hint within @mat_tpose_vec  --> \n", ++aesl_llvm_cbe_818_count);
  llvm_cbe_tmp__173 = (float *)(&llvm_cbe_y[(((signed long long )llvm_cbe_storemerge15))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge15 = 0x%I64X",((signed long long )llvm_cbe_storemerge15));
}
  llvm_cbe_storemerge23__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__169;   /* for PHI node */
  goto llvm_cbe_tmp__195;

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_loopexit4:
  goto llvm_cbe_tmp__193;

llvm_cbe_tmp__193:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @mat_tpose_vec}\n");
  return;
}


float quad_form(float *llvm_cbe_Px, signed long long *llvm_cbe_Pp, signed long long *llvm_cbe_Pi, signed long long llvm_cbe_Pn, float *llvm_cbe_x) {
  static  unsigned long long aesl_llvm_cbe_862_count = 0;
  static  unsigned long long aesl_llvm_cbe_863_count = 0;
  static  unsigned long long aesl_llvm_cbe_864_count = 0;
  static  unsigned long long aesl_llvm_cbe_865_count = 0;
  static  unsigned long long aesl_llvm_cbe_866_count = 0;
  static  unsigned long long aesl_llvm_cbe_867_count = 0;
  static  unsigned long long aesl_llvm_cbe_868_count = 0;
  static  unsigned long long aesl_llvm_cbe_869_count = 0;
  static  unsigned long long aesl_llvm_cbe_870_count = 0;
  static  unsigned long long aesl_llvm_cbe_871_count = 0;
  static  unsigned long long aesl_llvm_cbe_872_count = 0;
  static  unsigned long long aesl_llvm_cbe_873_count = 0;
  static  unsigned long long aesl_llvm_cbe_874_count = 0;
  static  unsigned long long aesl_llvm_cbe_875_count = 0;
  static  unsigned long long aesl_llvm_cbe_876_count = 0;
  static  unsigned long long aesl_llvm_cbe_877_count = 0;
  static  unsigned long long aesl_llvm_cbe_878_count = 0;
  static  unsigned long long aesl_llvm_cbe_879_count = 0;
  static  unsigned long long aesl_llvm_cbe_880_count = 0;
  static  unsigned long long aesl_llvm_cbe_881_count = 0;
  static  unsigned long long aesl_llvm_cbe_882_count = 0;
  static  unsigned long long aesl_llvm_cbe_883_count = 0;
  static  unsigned long long aesl_llvm_cbe_884_count = 0;
  static  unsigned long long aesl_llvm_cbe_885_count = 0;
  static  unsigned long long aesl_llvm_cbe_886_count = 0;
  static  unsigned long long aesl_llvm_cbe_887_count = 0;
  static  unsigned long long aesl_llvm_cbe_888_count = 0;
  static  unsigned long long aesl_llvm_cbe_889_count = 0;
  static  unsigned long long aesl_llvm_cbe_890_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_exitcond_count = 0;
  static  unsigned long long aesl_llvm_cbe_891_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge6_count = 0;
  unsigned long long llvm_cbe_storemerge6;
  unsigned long long llvm_cbe_storemerge6__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_892_count = 0;
  float llvm_cbe_tmp__201;
  float llvm_cbe_tmp__201__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_893_count = 0;
  signed long long *llvm_cbe_tmp__202;
  static  unsigned long long aesl_llvm_cbe_894_count = 0;
  unsigned long long llvm_cbe_tmp__203;
  static  unsigned long long aesl_llvm_cbe_895_count = 0;
  static  unsigned long long aesl_llvm_cbe_896_count = 0;
  static  unsigned long long aesl_llvm_cbe_897_count = 0;
  static  unsigned long long aesl_llvm_cbe_898_count = 0;
  static  unsigned long long aesl_llvm_cbe_899_count = 0;
  static  unsigned long long aesl_llvm_cbe_900_count = 0;
  static  unsigned long long aesl_llvm_cbe_901_count = 0;
  unsigned long long llvm_cbe_tmp__204;
  static  unsigned long long aesl_llvm_cbe_902_count = 0;
  signed long long *llvm_cbe_tmp__205;
  static  unsigned long long aesl_llvm_cbe_903_count = 0;
  unsigned long long llvm_cbe_tmp__206;
  static  unsigned long long aesl_llvm_cbe_904_count = 0;
  static  unsigned long long aesl_llvm_cbe_905_count = 0;
  static  unsigned long long aesl_llvm_cbe_906_count = 0;
  float *llvm_cbe_tmp__207;
  static  unsigned long long aesl_llvm_cbe_907_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge13_count = 0;
  unsigned long long llvm_cbe_storemerge13;
  unsigned long long llvm_cbe_storemerge13__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_908_count = 0;
  float llvm_cbe_tmp__208;
  float llvm_cbe_tmp__208__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_909_count = 0;
  signed long long *llvm_cbe_tmp__209;
  static  unsigned long long aesl_llvm_cbe_910_count = 0;
  unsigned long long llvm_cbe_tmp__210;
  static  unsigned long long aesl_llvm_cbe_911_count = 0;
  static  unsigned long long aesl_llvm_cbe_912_count = 0;
  static  unsigned long long aesl_llvm_cbe_913_count = 0;
  static  unsigned long long aesl_llvm_cbe_914_count = 0;
  static  unsigned long long aesl_llvm_cbe_915_count = 0;
  static  unsigned long long aesl_llvm_cbe_916_count = 0;
  static  unsigned long long aesl_llvm_cbe_917_count = 0;
  static  unsigned long long aesl_llvm_cbe_918_count = 0;
  static  unsigned long long aesl_llvm_cbe_919_count = 0;
  float *llvm_cbe_tmp__211;
  static  unsigned long long aesl_llvm_cbe_920_count = 0;
  float llvm_cbe_tmp__212;
  static  unsigned long long aesl_llvm_cbe_921_count = 0;
  float llvm_cbe_tmp__213;
  static  unsigned long long aesl_llvm_cbe_922_count = 0;
  float llvm_cbe_tmp__214;
  static  unsigned long long aesl_llvm_cbe_923_count = 0;
  float llvm_cbe_tmp__215;
  static  unsigned long long aesl_llvm_cbe_924_count = 0;
  float llvm_cbe_tmp__216;
  static  unsigned long long aesl_llvm_cbe_925_count = 0;
  float llvm_cbe_tmp__217;
  static  unsigned long long aesl_llvm_cbe_926_count = 0;
  static  unsigned long long aesl_llvm_cbe_927_count = 0;
  static  unsigned long long aesl_llvm_cbe_928_count = 0;
  static  unsigned long long aesl_llvm_cbe_929_count = 0;
  static  unsigned long long aesl_llvm_cbe_930_count = 0;
  static  unsigned long long aesl_llvm_cbe_931_count = 0;
  static  unsigned long long aesl_llvm_cbe_932_count = 0;
  static  unsigned long long aesl_llvm_cbe_933_count = 0;
  float *llvm_cbe_tmp__218;
  static  unsigned long long aesl_llvm_cbe_934_count = 0;
  float llvm_cbe_tmp__219;
  static  unsigned long long aesl_llvm_cbe_935_count = 0;
  float *llvm_cbe_tmp__220;
  static  unsigned long long aesl_llvm_cbe_936_count = 0;
  float llvm_cbe_tmp__221;
  static  unsigned long long aesl_llvm_cbe_937_count = 0;
  float llvm_cbe_tmp__222;
  static  unsigned long long aesl_llvm_cbe_938_count = 0;
  float llvm_cbe_tmp__223;
  static  unsigned long long aesl_llvm_cbe_939_count = 0;
  float llvm_cbe_tmp__224;
  static  unsigned long long aesl_llvm_cbe_940_count = 0;
  float llvm_cbe_tmp__225;
  static  unsigned long long aesl_llvm_cbe_941_count = 0;
  static  unsigned long long aesl_llvm_cbe_942_count = 0;
  static  unsigned long long aesl_llvm_cbe_943_count = 0;
  static  unsigned long long aesl_llvm_cbe_944_count = 0;
  static  unsigned long long aesl_llvm_cbe_945_count = 0;
  static  unsigned long long aesl_llvm_cbe_946_count = 0;
  float llvm_cbe_tmp__226;
  float llvm_cbe_tmp__226__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_947_count = 0;
  unsigned long long llvm_cbe_tmp__227;
  static  unsigned long long aesl_llvm_cbe_948_count = 0;
  static  unsigned long long aesl_llvm_cbe_949_count = 0;
  static  unsigned long long aesl_llvm_cbe_950_count = 0;
  static  unsigned long long aesl_llvm_cbe_951_count = 0;
  static  unsigned long long aesl_llvm_cbe_952_count = 0;
  static  unsigned long long aesl_llvm_cbe_953_count = 0;
  static  unsigned long long aesl_llvm_cbe_954_count = 0;
  static  unsigned long long aesl_llvm_cbe_955_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa5_count = 0;
  float llvm_cbe__2e_lcssa5;
  float llvm_cbe__2e_lcssa5__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_956_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @quad_form\n");
  if ((((signed long long )llvm_cbe_Pn) > ((signed long long )0ull))) {
    llvm_cbe_storemerge6__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__201__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph7;
  } else {
    llvm_cbe__2e_lcssa5__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph7' to make GCC happy */
llvm_cbe__2e_lr_2e_ph7:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge6 = phi i64 [ %%5, %%.loopexit ], [ 0, %%0  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_storemerge6_count);
  llvm_cbe_storemerge6 = (unsigned long long )llvm_cbe_storemerge6__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge6 = 0x%I64X",llvm_cbe_storemerge6);
printf("\n = 0x%I64X",llvm_cbe_tmp__204);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = phi float [ %%.lcssa, %%.loopexit ], [ 0.000000e+00, %%0  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_892_count);
  llvm_cbe_tmp__201 = (float )llvm_cbe_tmp__201__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__201);
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds i64* %%Pp, i64 %%storemerge6, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_893_count);
  llvm_cbe_tmp__202 = (signed long long *)(&llvm_cbe_Pp[(((signed long long )llvm_cbe_storemerge6))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge6 = 0x%I64X",((signed long long )llvm_cbe_storemerge6));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* %%3, align 8, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_894_count);
  llvm_cbe_tmp__203 = (unsigned long long )*llvm_cbe_tmp__202;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__203);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = add nsw i64 %%storemerge6, 1, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_901_count);
  llvm_cbe_tmp__204 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge6&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__204&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds i64* %%Pp, i64 %%5, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_902_count);
  llvm_cbe_tmp__205 = (signed long long *)(&llvm_cbe_Pp[(((signed long long )llvm_cbe_tmp__204))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__204));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* %%6, align 8, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_903_count);
  llvm_cbe_tmp__206 = (unsigned long long )*llvm_cbe_tmp__205;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__206);
  if ((((signed long long )llvm_cbe_tmp__203) < ((signed long long )llvm_cbe_tmp__206))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__201;   /* for PHI node */
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe__2e_loopexit:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ %%2, %%.lr.ph7 ], [ %%35, %%34  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",llvm_cbe_tmp__201);
printf("\n = %f",llvm_cbe_tmp__226);
}
  if (((llvm_cbe_tmp__204&18446744073709551615ULL) == (llvm_cbe_Pn&18446744073709551615ULL))) {
    llvm_cbe__2e_lcssa5__PHI_TEMPORARY = (float )llvm_cbe__2e_lcssa;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  } else {
    llvm_cbe_storemerge6__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__204;   /* for PHI node */
    llvm_cbe_tmp__201__PHI_TEMPORARY = (float )llvm_cbe__2e_lcssa;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph7;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__228:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge13 = phi i64 [ %%4, %%.lr.ph ], [ %%36, %%34  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_storemerge13_count);
  llvm_cbe_storemerge13 = (unsigned long long )llvm_cbe_storemerge13__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",llvm_cbe_storemerge13);
printf("\n = 0x%I64X",llvm_cbe_tmp__203);
printf("\n = 0x%I64X",llvm_cbe_tmp__227);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = phi float [ %%2, %%.lr.ph ], [ %%35, %%34  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_908_count);
  llvm_cbe_tmp__208 = (float )llvm_cbe_tmp__208__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__208);
printf("\n = %f",llvm_cbe_tmp__201);
printf("\n = %f",llvm_cbe_tmp__226);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds i64* %%Pi, i64 %%storemerge13, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_909_count);
  llvm_cbe_tmp__209 = (signed long long *)(&llvm_cbe_Pi[(((signed long long )llvm_cbe_storemerge13))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",((signed long long )llvm_cbe_storemerge13));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load i64* %%12, align 8, !dbg !15 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_910_count);
  llvm_cbe_tmp__210 = (unsigned long long )*llvm_cbe_tmp__209;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__210);
  if (((llvm_cbe_tmp__210&18446744073709551615ULL) == (llvm_cbe_storemerge6&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__229;
  } else {
    goto llvm_cbe_tmp__230;
  }

llvm_cbe_tmp__231:
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = phi float [ %%22, %%15 ], [ %%33, %%25 ], [ %%11, %%23  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_946_count);
  llvm_cbe_tmp__226 = (float )llvm_cbe_tmp__226__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__226);
printf("\n = %f",llvm_cbe_tmp__217);
printf("\n = %f",llvm_cbe_tmp__225);
printf("\n = %f",llvm_cbe_tmp__208);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = add nsw i64 %%storemerge13, 1, !dbg !17 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_947_count);
  llvm_cbe_tmp__227 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge13&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__227&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__227) < ((signed long long )llvm_cbe_tmp__206))) {
    llvm_cbe_storemerge13__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__227;   /* for PHI node */
    llvm_cbe_tmp__208__PHI_TEMPORARY = (float )llvm_cbe_tmp__226;   /* for PHI node */
    goto llvm_cbe_tmp__228;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__226;   /* for PHI node */
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe_tmp__229:
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds float* %%Px, i64 %%storemerge13, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_919_count);
  llvm_cbe_tmp__211 = (float *)(&llvm_cbe_Px[(((signed long long )llvm_cbe_storemerge13))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",((signed long long )llvm_cbe_storemerge13));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load float* %%16, align 4, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_920_count);
  llvm_cbe_tmp__212 = (float )*llvm_cbe_tmp__211;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__212, *(int*)(&llvm_cbe_tmp__212));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = fmul float %%17, 5.000000e-01, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_921_count);
  llvm_cbe_tmp__213 = (float )((float )(llvm_cbe_tmp__212 * 0x1p-1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__213, *(int*)(&llvm_cbe_tmp__213));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%9, align 4, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_922_count);
  llvm_cbe_tmp__214 = (float )*llvm_cbe_tmp__207;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__214, *(int*)(&llvm_cbe_tmp__214));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = fmul float %%18, %%19, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_923_count);
  llvm_cbe_tmp__215 = (float )((float )(llvm_cbe_tmp__213 * llvm_cbe_tmp__214));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__215, *(int*)(&llvm_cbe_tmp__215));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = fmul float %%20, %%19, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_924_count);
  llvm_cbe_tmp__216 = (float )((float )(llvm_cbe_tmp__215 * llvm_cbe_tmp__214));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__216, *(int*)(&llvm_cbe_tmp__216));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fadd float %%11, %%21, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_925_count);
  llvm_cbe_tmp__217 = (float )((float )(llvm_cbe_tmp__208 + llvm_cbe_tmp__216));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__217, *(int*)(&llvm_cbe_tmp__217));
  llvm_cbe_tmp__226__PHI_TEMPORARY = (float )llvm_cbe_tmp__217;   /* for PHI node */
  goto llvm_cbe_tmp__231;

llvm_cbe_tmp__230:
  if ((((signed long long )llvm_cbe_tmp__210) < ((signed long long )llvm_cbe_storemerge6))) {
    goto llvm_cbe_tmp__232;
  } else {
    llvm_cbe_tmp__226__PHI_TEMPORARY = (float )llvm_cbe_tmp__208;   /* for PHI node */
    goto llvm_cbe_tmp__231;
  }

llvm_cbe_tmp__232:
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = getelementptr inbounds float* %%Px, i64 %%storemerge13, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_933_count);
  llvm_cbe_tmp__218 = (float *)(&llvm_cbe_Px[(((signed long long )llvm_cbe_storemerge13))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge13 = 0x%I64X",((signed long long )llvm_cbe_storemerge13));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = load float* %%26, align 4, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_934_count);
  llvm_cbe_tmp__219 = (float )*llvm_cbe_tmp__218;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__219, *(int*)(&llvm_cbe_tmp__219));
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = getelementptr inbounds float* %%x, i64 %%13, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_935_count);
  llvm_cbe_tmp__220 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_tmp__210))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__210));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = load float* %%28, align 4, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_936_count);
  llvm_cbe_tmp__221 = (float )*llvm_cbe_tmp__220;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__221, *(int*)(&llvm_cbe_tmp__221));
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = fmul float %%27, %%29, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_937_count);
  llvm_cbe_tmp__222 = (float )((float )(llvm_cbe_tmp__219 * llvm_cbe_tmp__221));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__222, *(int*)(&llvm_cbe_tmp__222));
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%9, align 4, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_938_count);
  llvm_cbe_tmp__223 = (float )*llvm_cbe_tmp__207;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__223, *(int*)(&llvm_cbe_tmp__223));
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = fmul float %%30, %%31, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_939_count);
  llvm_cbe_tmp__224 = (float )((float )(llvm_cbe_tmp__222 * llvm_cbe_tmp__223));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__224, *(int*)(&llvm_cbe_tmp__224));
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = fadd float %%11, %%32, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_940_count);
  llvm_cbe_tmp__225 = (float )((float )(llvm_cbe_tmp__208 + llvm_cbe_tmp__224));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__225, *(int*)(&llvm_cbe_tmp__225));
  llvm_cbe_tmp__226__PHI_TEMPORARY = (float )llvm_cbe_tmp__225;   /* for PHI node */
  goto llvm_cbe_tmp__231;

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds float* %%x, i64 %%storemerge6, !dbg !14 for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe_906_count);
  llvm_cbe_tmp__207 = (float *)(&llvm_cbe_x[(((signed long long )llvm_cbe_storemerge6))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge6 = 0x%I64X",((signed long long )llvm_cbe_storemerge6));
}
  llvm_cbe_storemerge13__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__203;   /* for PHI node */
  llvm_cbe_tmp__208__PHI_TEMPORARY = (float )llvm_cbe_tmp__201;   /* for PHI node */
  goto llvm_cbe_tmp__228;

  } while (1); /* end of syntactic loop '.lr.ph7' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa5 = phi float [ 0.000000e+00, %%0 ], [ %%.lcssa, %%.loopexit  for 0x%I64xth hint within @quad_form  --> \n", ++aesl_llvm_cbe__2e_lcssa5_count);
  llvm_cbe__2e_lcssa5 = (float )llvm_cbe__2e_lcssa5__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa5 = %f",llvm_cbe__2e_lcssa5);
printf("\n = %f",0x0p0);
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @quad_form}\n");
  return llvm_cbe__2e_lcssa5;
}

