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
 char *osqp_version(void);
void c_strcpy( char *llvm_cbe_dest,  char *llvm_cbe_source);


/* Global Variable Definitions and Initialization */
static  char aesl_internal__OC_str[6] = "0.6.2";


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

 char *osqp_version(void) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @osqp_version\n");
  if (AESL_DEBUG_TRACE)
      printf("\nEND @osqp_version}\n");
  return ((&aesl_internal__OC_str[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 6
#endif
]));
}


void c_strcpy( char *llvm_cbe_dest,  char *llvm_cbe_source) {
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
  static  unsigned long long aesl_llvm_cbe_4_count = 0;
  static  unsigned long long aesl_llvm_cbe_5_count = 0;
  static  unsigned long long aesl_llvm_cbe_6_count = 0;
  static  unsigned long long aesl_llvm_cbe_7_count = 0;
  static  unsigned long long aesl_llvm_cbe_8_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  unsigned int llvm_cbe_storemerge;
  unsigned int llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_9_count = 0;
  static  unsigned long long aesl_llvm_cbe_10_count = 0;
  static  unsigned long long aesl_llvm_cbe_11_count = 0;
  static  unsigned long long aesl_llvm_cbe_12_count = 0;
  static  unsigned long long aesl_llvm_cbe_13_count = 0;
  unsigned long long llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_14_count = 0;
   char *llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_15_count = 0;
  unsigned char llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_16_count = 0;
   char *llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  unsigned int llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @c_strcpy\n");
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned int )0u;   /* for PHI node */
  goto llvm_cbe_tmp__6;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__6:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi i32 [ 0, %%0 ], [ %%7, %%1  for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (unsigned int )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = 0x%X",llvm_cbe_storemerge);
printf("\n = 0x%X",0u);
printf("\n = 0x%X",llvm_cbe_tmp__5);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = sext i32 %%storemerge to i64, !dbg !3 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_13_count);
  llvm_cbe_tmp__1 = (unsigned long long )((signed long long )(signed int )llvm_cbe_storemerge);
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__1);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds i8* %%source, i64 %%2, !dbg !3 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_14_count);
  llvm_cbe_tmp__2 = ( char *)(&llvm_cbe_source[(((signed long long )llvm_cbe_tmp__1))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__1));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i8* %%3, align 1, !dbg !3 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_15_count);
  llvm_cbe_tmp__3 = (unsigned char )*llvm_cbe_tmp__2;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds i8* %%dest, i64 %%2, !dbg !3 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_16_count);
  llvm_cbe_tmp__4 = ( char *)(&llvm_cbe_dest[(((signed long long )llvm_cbe_tmp__1))]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__1));
}
if (AESL_DEBUG_TRACE)
printf("\n  store i8 %%4, i8* %%5, align 1, !dbg !3 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_17_count);
  *llvm_cbe_tmp__4 = llvm_cbe_tmp__3;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = add nsw i32 %%storemerge, 1, !dbg !4 for 0x%I64xth hint within @c_strcpy  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__5 = (unsigned int )((unsigned int )(llvm_cbe_storemerge&4294967295ull)) + ((unsigned int )(1u&4294967295ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", ((unsigned int )(llvm_cbe_tmp__5&4294967295ull)));
  if (((llvm_cbe_tmp__3&255U) == (((unsigned char )0)&255U))) {
    goto llvm_cbe_tmp__7;
  } else {
    llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned int )llvm_cbe_tmp__5;   /* for PHI node */
    goto llvm_cbe_tmp__6;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe_tmp__7:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @c_strcpy}\n");
  return;
}

