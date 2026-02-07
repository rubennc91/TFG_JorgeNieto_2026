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
typedef struct l_struct_OC_OSQPData l_struct_OC_OSQPData;
typedef struct l_struct_OC_csc l_struct_OC_csc;

/* Structure contents */
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


/* External Global Variable Declarations */
extern l_struct_OC_OSQPData data;
extern float ldata[19];
extern float udata[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void project(float *llvm_cbe_z);


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

void project(float *llvm_cbe_z) {
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
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  unsigned long long llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  float *llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  float llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  float *llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  float llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  float llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  float llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe__2e_not_count = 0;
  static  unsigned long long aesl_llvm_cbe_brmerge_count = 0;
  bool llvm_cbe_brmerge;
  static  unsigned long long aesl_llvm_cbe__2e_mux_count = 0;
  float llvm_cbe__2e_mux;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  float llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  unsigned long long llvm_cbe_tmp__10;
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
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  unsigned long long llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  static  unsigned long long aesl_llvm_cbe_51_count = 0;
  static  unsigned long long aesl_llvm_cbe_52_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @project\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_21_count);
  llvm_cbe_tmp__1 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__1);
  if ((((signed long long )llvm_cbe_tmp__1) > ((signed long long )0ull))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ %%12, %%.lr.ph ], [ 0, %%0  for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",llvm_cbe_tmp__10);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds float* %%z, i64 %%storemerge1, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_24_count);
  llvm_cbe_tmp__2 = (float *)(&llvm_cbe_z[(((signed long long )llvm_cbe_storemerge1))]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_25_count);
  llvm_cbe_tmp__3 = (float )*llvm_cbe_tmp__2;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__3, *(int*)(&llvm_cbe_tmp__3));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds [19 x float]* @ldata, i64 0, i64 %%storemerge1, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_26_count);
  llvm_cbe_tmp__4 = (float *)(&ldata[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'ldata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_27_count);
  llvm_cbe_tmp__5 = (float )*llvm_cbe_tmp__4;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__5, *(int*)(&llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = select i1 %%7, float %%4, float %%6, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_29_count);
  llvm_cbe_tmp__6 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__3, llvm_cbe_tmp__5))) ? ((float )llvm_cbe_tmp__3) : ((float )llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__6, *(int*)(&llvm_cbe_tmp__6));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds [19 x float]* @udata, i64 0, i64 %%storemerge1, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_30_count);
  llvm_cbe_tmp__7 = (float *)(&udata[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'udata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load float* %%9, align 4, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_31_count);
  llvm_cbe_tmp__8 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__8, *(int*)(&llvm_cbe_tmp__8));
if (AESL_DEBUG_TRACE)
printf("\n  %%brmerge = or i1 %%.not, %%7, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_brmerge_count);
  llvm_cbe_brmerge = (bool )(((llvm_fcmp_uge(llvm_cbe_tmp__6, llvm_cbe_tmp__8)) | (llvm_fcmp_ogt(llvm_cbe_tmp__3, llvm_cbe_tmp__5)))&1);
if (AESL_DEBUG_TRACE)
printf("\nbrmerge = 0x%X\n", llvm_cbe_brmerge);
if (AESL_DEBUG_TRACE)
printf("\n  %%.mux = select i1 %%.not, float %%10, float %%4, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe__2e_mux_count);
  llvm_cbe__2e_mux = (float )(((llvm_fcmp_uge(llvm_cbe_tmp__6, llvm_cbe_tmp__8))) ? ((float )llvm_cbe_tmp__8) : ((float )llvm_cbe_tmp__3));
if (AESL_DEBUG_TRACE)
printf("\n.mux = %f,  0x%x\n", llvm_cbe__2e_mux, *(int*)(&llvm_cbe__2e_mux));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = select i1 %%brmerge, float %%.mux, float %%6, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_32_count);
  llvm_cbe_tmp__9 = (float )((llvm_cbe_brmerge) ? ((float )llvm_cbe__2e_mux) : ((float )llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__9, *(int*)(&llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%3, align 4, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_33_count);
  *llvm_cbe_tmp__2 = llvm_cbe_tmp__9;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__9);
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = add nsw i64 %%storemerge1, 1, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_34_count);
  llvm_cbe_tmp__10 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__10&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !22 for 0x%I64xth hint within @project  --> \n", ++aesl_llvm_cbe_49_count);
  llvm_cbe_tmp__11 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__11);
  if ((((signed long long )llvm_cbe_tmp__10) < ((signed long long )llvm_cbe_tmp__11))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__10;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @project}\n");
  return;
}

