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
typedef struct l_struct_OC_OSQPScaling l_struct_OC_OSQPScaling;
typedef struct l_struct_OC_OSQPData l_struct_OC_OSQPData;
typedef struct l_struct_OC_csc l_struct_OC_csc;

/* Structure contents */
struct l_struct_OC_OSQPScaling {
  float field0;
  float *field1;
  float *field2;
  float field3;
  float *field4;
  float *field5;
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


/* External Global Variable Declarations */
extern l_struct_OC_OSQPScaling scaling;
extern float scaling_D[15];
extern l_struct_OC_OSQPData data;
extern float scaling_Dinv[15];
extern float scaling_E[19];
extern float scaling_Einv[19];
extern float Pdata_x[12];
extern signed long long Pdata_p[16];
extern signed long long Pdata_i[12];
extern float qdata[15];
extern float Adata_x[43];
extern signed long long Adata_p[16];
extern signed long long Adata_i[43];
extern float ldata[19];
extern float udata[19];
extern float xsolution[15];
extern float ysolution[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
signed long long scale_data(void);
void vec_set_scalar(float *, float , signed long long );
signed long long unscale_data(void);
void mat_mult_scalar(float *, signed long long *, signed long long , float );
void mat_premult_diag(float *, signed long long *, signed long long *, signed long long , float *);
void mat_postmult_diag(float *, signed long long *, signed long long , float *);
void vec_mult_scalar(float *, float , signed long long );
void vec_ew_prod(float *, float *, float *, signed long long );
signed long long unscale_solution(void);


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

signed long long scale_data(void) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
  unsigned long long llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
  static  unsigned long long aesl_llvm_cbe_4_count = 0;
  unsigned long long llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_5_count = 0;
  static  unsigned long long aesl_llvm_cbe_6_count = 0;
  unsigned long long llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_7_count = 0;
  static  unsigned long long aesl_llvm_cbe_8_count = 0;
  unsigned long long llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_9_count = 0;
  static  unsigned long long aesl_llvm_cbe_10_count = 0;
  static  unsigned long long aesl_llvm_cbe_11_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @scale_data\n");
if (AESL_DEBUG_TRACE)
printf("\n  store float 1.000000e+00, float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 0), align 8, !dbg !25 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_1_count);
  *((&scaling.field0)) = 0x1p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_2_count);
  llvm_cbe_tmp__1 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__1);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([15 x float]* @scaling_D, i64 0, i64 0), float 1.000000e+00, i64 %%1) nounwind, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_3_count);
   /*tail*/ vec_set_scalar((float *)((&scaling_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0x1p0, llvm_cbe_tmp__1);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__1);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_4_count);
  llvm_cbe_tmp__2 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__2);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float 1.000000e+00, i64 %%2) nounwind, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_5_count);
   /*tail*/ vec_set_scalar((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0x1p0, llvm_cbe_tmp__2);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__2);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_6_count);
  llvm_cbe_tmp__3 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([19 x float]* @scaling_E, i64 0, i64 0), float 1.000000e+00, i64 %%3) nounwind, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_7_count);
   /*tail*/ vec_set_scalar((float *)((&scaling_E[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x1p0, llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__3);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_8_count);
  llvm_cbe_tmp__4 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__4);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float 1.000000e+00, i64 %%4) nounwind, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_9_count);
   /*tail*/ vec_set_scalar((float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x1p0, llvm_cbe_tmp__4);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__4);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 1.000000e+00, float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !26 for 0x%I64xth hint within @scale_data  --> \n", ++aesl_llvm_cbe_10_count);
  *((&scaling.field3)) = 0x1p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1p0);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @scale_data}\n");
  return 0ull;
}


signed long long unscale_data(void) {
  static  unsigned long long aesl_llvm_cbe_12_count = 0;
  unsigned long long llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_13_count = 0;
  float llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_14_count = 0;
  static  unsigned long long aesl_llvm_cbe_15_count = 0;
  unsigned long long llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_16_count = 0;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  unsigned long long llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  float llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  unsigned long long llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  unsigned long long llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  unsigned long long llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  unsigned long long llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  unsigned long long llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  unsigned long long llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @unscale_data\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !25 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_12_count);
  llvm_cbe_tmp__5 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__5);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !25 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_13_count);
  llvm_cbe_tmp__6 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__6, *(int*)(&llvm_cbe_tmp__6));
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_mult_scalar(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64 %%1, float %%2) nounwind, !dbg !25 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_14_count);
   /*tail*/ mat_mult_scalar((float *)((&Pdata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), (signed long long *)((&Pdata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), llvm_cbe_tmp__5, llvm_cbe_tmp__6);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__5);
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__6, *(int*)(&llvm_cbe_tmp__6));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_15_count);
  llvm_cbe_tmp__7 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__7);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_premult_diag(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%3, float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0)) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_16_count);
   /*tail*/ mat_premult_diag((float *)((&Pdata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), (signed long long *)((&Pdata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), (signed long long *)((&Pdata_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), llvm_cbe_tmp__7, (float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__7);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_17_count);
  llvm_cbe_tmp__8 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__8);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_postmult_diag(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64 %%4, float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0)) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_18_count);
   /*tail*/ mat_postmult_diag((float *)((&Pdata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])), (signed long long *)((&Pdata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), llvm_cbe_tmp__8, (float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__8);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_19_count);
  llvm_cbe_tmp__9 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__9, *(int*)(&llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_20_count);
  llvm_cbe_tmp__10 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_mult_scalar(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float %%5, i64 %%6) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_21_count);
   /*tail*/ vec_mult_scalar((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__9, llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__9, *(int*)(&llvm_cbe_tmp__9));
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__10);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_22_count);
  llvm_cbe_tmp__11 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__11);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), i64 %%7) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_23_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Dinv[(((signed long long )0ull))
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
])), llvm_cbe_tmp__11);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__11);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_24_count);
  llvm_cbe_tmp__12 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__12);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_premult_diag(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64* getelementptr inbounds ([43 x i64]* @Adata_i, i64 0, i64 0), i64 %%8, float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0)) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_25_count);
   /*tail*/ mat_premult_diag((float *)((&Adata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 43
#endif
])), (signed long long *)((&Adata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), (signed long long *)((&Adata_i[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 43
#endif
])), llvm_cbe_tmp__12, (float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__12);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_26_count);
  llvm_cbe_tmp__13 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__13);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_postmult_diag(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64 %%9, float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0)) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_27_count);
   /*tail*/ mat_postmult_diag((float *)((&Adata_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 43
#endif
])), (signed long long *)((&Adata_p[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])), llvm_cbe_tmp__13, (float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__13);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_28_count);
  llvm_cbe_tmp__14 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__14);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), i64 %%10) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_29_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Einv[(((signed long long )0ull))
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
])), llvm_cbe_tmp__14);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__14);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_30_count);
  llvm_cbe_tmp__15 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__15);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), i64 %%11) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_data  --> \n", ++aesl_llvm_cbe_31_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Einv[(((signed long long )0ull))
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
])), llvm_cbe_tmp__15);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__15);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @unscale_data}\n");
  return 0ull;
}


signed long long unscale_solution(void) {
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  unsigned long long llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;
  unsigned long long llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_36_count = 0;
  static  unsigned long long aesl_llvm_cbe_37_count = 0;
  float llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_38_count = 0;
  unsigned long long llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_39_count = 0;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @unscale_solution\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !25 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_33_count);
  llvm_cbe_tmp__16 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([15 x float]* @scaling_D, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @xsolution, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @xsolution, i64 0, i64 0), i64 %%1) nounwind, !dbg !25 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_34_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&xsolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&xsolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__16);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_35_count);
  llvm_cbe_tmp__17 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__17);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_E, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ysolution, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ysolution, i64 0, i64 0), i64 %%2) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_36_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_E[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&ysolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&ysolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__17);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__17);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !26 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_37_count);
  llvm_cbe_tmp__18 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !26 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_38_count);
  llvm_cbe_tmp__19 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__19);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_mult_scalar(float* getelementptr inbounds ([19 x float]* @ysolution, i64 0, i64 0), float %%3, i64 %%4) nounwind, !dbg !26 for 0x%I64xth hint within @unscale_solution  --> \n", ++aesl_llvm_cbe_39_count);
   /*tail*/ vec_mult_scalar((float *)((&ysolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__18, llvm_cbe_tmp__19);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__19);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @unscale_solution}\n");
  return 0ull;
}

