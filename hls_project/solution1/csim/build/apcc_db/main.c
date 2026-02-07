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
signed int main(void);
void myFunction(float *, float , float , float , float *, float *);


/* Global Variable Definitions and Initialization */
static float aesl_internal_main_OC_x_ini[3] = { 0x1.4p3, 0x1.4p2, 0x1.77p9 };
static  char aesl_internal__OC_str2[23] = "\n--- Iteracion %d ---\n";
static  char aesl_internal__OC_str3[27] = "Inputs: Vdc=%.2f, iL=%.2f\n";
static  char aesl_internal__OC_str5[15] = "  v_ref_d: %f\n";
static  char aesl_internal__OC_str6[15] = "  v_ref_q: %f\n";
static  char aesl_internal_str6[46] = "=============================================";
static  char aesl_internal_str1[46] = "   INICIO TESTBENCH HLS - OSQP SOLVER MPC    ";
static  char aesl_internal_str3[29] = "Resultados (Control Action):";
static  char aesl_internal_str4[47] = "\n=============================================";
static  char aesl_internal_str5[46] = "   TESTBENCH FINALIZADO EXITOSAMENTE (RET 0) ";


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

signed int main(void) {
  static  unsigned long long aesl_llvm_cbe_x_ini_count = 0;
  float llvm_cbe_x_ini[3];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_u00_count = 0;
  float llvm_cbe_u00[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_output_count = 0;
  float llvm_cbe_output[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_puts_count = 0;
  unsigned int llvm_cbe_puts;
  static  unsigned long long aesl_llvm_cbe_puts1_count = 0;
  unsigned int llvm_cbe_puts1;
  static  unsigned long long aesl_llvm_cbe_puts2_count = 0;
  unsigned int llvm_cbe_puts2;
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
   char *llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
   char *llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_4_count = 0;
  static  unsigned long long aesl_llvm_cbe_5_count = 0;
  static  unsigned long long aesl_llvm_cbe_6_count = 0;
  static  unsigned long long aesl_llvm_cbe_7_count = 0;
  static  unsigned long long aesl_llvm_cbe_8_count = 0;
  static  unsigned long long aesl_llvm_cbe_9_count = 0;
  static  unsigned long long aesl_llvm_cbe_10_count = 0;
  static  unsigned long long aesl_llvm_cbe_11_count = 0;
  static  unsigned long long aesl_llvm_cbe_12_count = 0;
   char *llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_13_count = 0;
   char *llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_14_count = 0;
  static  unsigned long long aesl_llvm_cbe_15_count = 0;
   char *llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_16_count = 0;
   char *llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_17_count = 0;
  static  unsigned long long aesl_llvm_cbe_18_count = 0;
  static  unsigned long long aesl_llvm_cbe_19_count = 0;
  static  unsigned long long aesl_llvm_cbe_20_count = 0;
  static  unsigned long long aesl_llvm_cbe_21_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  float *llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  float *llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  float *llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  float *llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  float *llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  unsigned int llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  unsigned int llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  static  unsigned long long aesl_llvm_cbe_puts3_count = 0;
  unsigned int llvm_cbe_puts3;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
  float llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_31_count = 0;
  double llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_32_count = 0;
  unsigned int llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_33_count = 0;
  float llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_34_count = 0;
  double llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_35_count = 0;
  unsigned int llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_36_count = 0;
  float llvm_cbe_tmp__21;
  static  unsigned long long aesl_llvm_cbe_37_count = 0;
  static  unsigned long long aesl_llvm_cbe_38_count = 0;
  float llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_39_count = 0;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;
  float llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_41_count = 0;
  float llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_42_count = 0;
  static  unsigned long long aesl_llvm_cbe_43_count = 0;
  static  unsigned long long aesl_llvm_cbe_44_count = 0;
  static  unsigned long long aesl_llvm_cbe_45_count = 0;
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  unsigned int llvm_cbe_tmp__25;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  float llvm_cbe_tmp__26;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  double llvm_cbe_tmp__27;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  unsigned int llvm_cbe_tmp__28;
  static  unsigned long long aesl_llvm_cbe_51_count = 0;
  static  unsigned long long aesl_llvm_cbe_puts3_2e_1_count = 0;
  unsigned int llvm_cbe_puts3_2e_1;
  static  unsigned long long aesl_llvm_cbe_52_count = 0;
  float llvm_cbe_tmp__29;
  static  unsigned long long aesl_llvm_cbe_53_count = 0;
  double llvm_cbe_tmp__30;
  static  unsigned long long aesl_llvm_cbe_54_count = 0;
  unsigned int llvm_cbe_tmp__31;
  static  unsigned long long aesl_llvm_cbe_55_count = 0;
  float llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_56_count = 0;
  double llvm_cbe_tmp__33;
  static  unsigned long long aesl_llvm_cbe_57_count = 0;
  unsigned int llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_58_count = 0;
  float llvm_cbe_tmp__35;
  static  unsigned long long aesl_llvm_cbe_59_count = 0;
  static  unsigned long long aesl_llvm_cbe_60_count = 0;
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_61_count = 0;
  static  unsigned long long aesl_llvm_cbe_62_count = 0;
  float llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_63_count = 0;
  float llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  unsigned int llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  float llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  double llvm_cbe_tmp__41;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  unsigned int llvm_cbe_tmp__42;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  static  unsigned long long aesl_llvm_cbe_puts3_2e_2_count = 0;
  unsigned int llvm_cbe_puts3_2e_2;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  float llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  double llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  unsigned int llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  float llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  double llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  unsigned int llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  float llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  float llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  float llvm_cbe_tmp__51;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  float llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_puts4_count = 0;
  unsigned int llvm_cbe_puts4;
  static  unsigned long long aesl_llvm_cbe_puts5_count = 0;
  unsigned int llvm_cbe_puts5;
  static  unsigned long long aesl_llvm_cbe_puts6_count = 0;
  unsigned int llvm_cbe_puts6;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;

  CODE_FOR_MAIN();
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @main\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%puts = call i32 @puts(i8* getelementptr inbounds ([46 x i8]* @aesl_internal_str6, i64 0, i64 0)), !dbg !1 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts_count);
  puts(( char *)((&aesl_internal_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 46
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts = 0x%X",llvm_cbe_puts);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts1 = call i32 @puts(i8* getelementptr inbounds ([46 x i8]* @aesl_internal_str1, i64 0, i64 0)), !dbg !2 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts1_count);
  puts(( char *)((&aesl_internal_str1[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 46
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts1 = 0x%X",llvm_cbe_puts1);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts2 = call i32 @puts(i8* getelementptr inbounds ([46 x i8]* @aesl_internal_str6, i64 0, i64 0)), !dbg !2 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts2_count);
  puts(( char *)((&aesl_internal_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 46
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts2 = 0x%X",llvm_cbe_puts2);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = bitcast [3 x float]* %%x_ini to i8*, !dbg !2 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_2_count);
  llvm_cbe_tmp__1 = ( char *)(( char *)(&llvm_cbe_x_ini));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = call i8* @memcpy(i8* %%1, i8* bitcast ([3 x float]* @aesl_internal_main.x_ini to i8*), i64 12 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_3_count);
  ( char *)memcpy(( char *)llvm_cbe_tmp__1, ( char *)(( char *)(&aesl_internal_main_OC_x_ini)), 12ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",12ull);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__2);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = bitcast [2 x float]* %%u00 to i8*, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_12_count);
  llvm_cbe_tmp__3 = ( char *)(( char *)(&llvm_cbe_u00));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = call i8* @memset(i8* %%3, i32 0, i64 8 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_13_count);
  ( char *)memset(( char *)llvm_cbe_tmp__3, 0u, 8ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%X",0u);
printf("\nArgument  = 0x%I64X",8ull);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__4);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = bitcast [2 x float]* %%output to i8*, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_15_count);
  llvm_cbe_tmp__5 = ( char *)(( char *)(&llvm_cbe_output));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = call i8* @memset(i8* %%5, i32 0, i64 8 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_16_count);
  ( char *)memset(( char *)llvm_cbe_tmp__5, 0u, 8ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%X",0u);
printf("\nArgument  = 0x%I64X",8ull);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__6);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [3 x float]* %%x_ini, i64 0, i64 2, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_21_count);
  llvm_cbe_tmp__7 = (float *)(&llvm_cbe_x_ini[(((signed long long )2ull))
#ifdef AESL_BC_SIM
 % 3
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds [3 x float]* %%x_ini, i64 0, i64 0, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_22_count);
  llvm_cbe_tmp__8 = (float *)(&llvm_cbe_x_ini[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 3
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds [2 x float]* %%u00, i64 0, i64 0, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__9 = (float *)(&llvm_cbe_u00[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds [2 x float]* %%output, i64 0, i64 0, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_24_count);
  llvm_cbe_tmp__10 = (float *)(&llvm_cbe_output[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds [2 x float]* %%output, i64 0, i64 1, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_25_count);
  llvm_cbe_tmp__11 = (float *)(&llvm_cbe_output[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds [2 x float]* %%u00, i64 0, i64 1, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_26_count);
  llvm_cbe_tmp__12 = (float *)(&llvm_cbe_u00[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([23 x i8]* @aesl_internal_.str2, i64 0, i64 0), i32 1) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_27_count);
  printf(( char *)((&aesl_internal__OC_str2[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 23
#endif
])), 1u);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%X",1u);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__13);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([27 x i8]* @aesl_internal_.str3, i64 0, i64 0), double 7.500000e+02, double 1.500000e+01) nounwind, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_28_count);
  printf(( char *)((&aesl_internal__OC_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 27
#endif
])), 0x1.77p9, 0x1.ep3);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf",0x1.77p9);
printf("\nArgument  = %lf",0x1.ep3);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__14);
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @myFunction(float* %%8, float 3.800000e+02, float 0.000000e+00, float 1.500000e+01, float* %%9, float* %%10) nounwind, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_29_count);
  myFunction((float *)llvm_cbe_tmp__8, 0x1.7cp8, 0x0p0, 0x1.ep3, (float *)llvm_cbe_tmp__9, (float *)llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.7cp8);
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = %f",0x1.ep3);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts3 = call i32 @puts(i8* getelementptr inbounds ([29 x i8]* @aesl_internal_str3, i64 0, i64 0)), !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts3_count);
  puts(( char *)((&aesl_internal_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 29
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts3 = 0x%X",llvm_cbe_puts3);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load float* %%10, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_30_count);
  llvm_cbe_tmp__15 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__15, *(int*)(&llvm_cbe_tmp__15));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = fpext float %%15 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_31_count);
  llvm_cbe_tmp__16 = (double )((double )llvm_cbe_tmp__15);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__16, *(long long*)(&llvm_cbe_tmp__16));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str5, i64 0, i64 0), double %%16) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_32_count);
  printf(( char *)((&aesl_internal__OC_str5[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__16);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__16, *(long long*)(&llvm_cbe_tmp__16));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__17);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_33_count);
  llvm_cbe_tmp__18 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fpext float %%18 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_34_count);
  llvm_cbe_tmp__19 = (double )((double )llvm_cbe_tmp__18);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__19, *(long long*)(&llvm_cbe_tmp__19));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str6, i64 0, i64 0), double %%19) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_35_count);
  printf(( char *)((&aesl_internal__OC_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__19);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__19, *(long long*)(&llvm_cbe_tmp__19));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__20);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load float* %%10, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_36_count);
  llvm_cbe_tmp__21 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__21, *(int*)(&llvm_cbe_tmp__21));

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%21, float* %%9, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_37_count);
  *llvm_cbe_tmp__9 = llvm_cbe_tmp__21;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__21);

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_38_count);
  llvm_cbe_tmp__22 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__22, *(int*)(&llvm_cbe_tmp__22));

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%22, float* %%12, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_39_count);
  *llvm_cbe_tmp__12 = llvm_cbe_tmp__22;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__22);

#ifdef AESL_BC_SIM
  if (!(((signed long long )2ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'x_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_40_count);
  llvm_cbe_tmp__23 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__23, *(int*)(&llvm_cbe_tmp__23));
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = fadd float %%23, 1.000000e+00, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_41_count);
  llvm_cbe_tmp__24 = (float )((float )(llvm_cbe_tmp__23 + 0x1p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__24, *(int*)(&llvm_cbe_tmp__24));

#ifdef AESL_BC_SIM
  assert(((signed long long )2ull) < 3 && "Write access out of array 'x_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%24, float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_42_count);
  *llvm_cbe_tmp__7 = llvm_cbe_tmp__24;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__24);
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([23 x i8]* @aesl_internal_.str2, i64 0, i64 0), i32 2) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_47_count);
  printf(( char *)((&aesl_internal__OC_str2[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 23
#endif
])), 2u);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%X",2u);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__25);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )2ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'x_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load float* %%7, align 4, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_48_count);
  llvm_cbe_tmp__26 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__26, *(int*)(&llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = fpext float %%26 to double, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_49_count);
  llvm_cbe_tmp__27 = (double )((double )llvm_cbe_tmp__26);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__27, *(long long*)(&llvm_cbe_tmp__27));
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([27 x i8]* @aesl_internal_.str3, i64 0, i64 0), double %%27, double 1.500000e+01) nounwind, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_50_count);
  printf(( char *)((&aesl_internal__OC_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 27
#endif
])), llvm_cbe_tmp__27, 0x1.ep3);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__27, *(long long*)(&llvm_cbe_tmp__27));
printf("\nArgument  = %lf",0x1.ep3);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__28);
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @myFunction(float* %%8, float 3.800000e+02, float 0.000000e+00, float 1.500000e+01, float* %%9, float* %%10) nounwind, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_51_count);
  myFunction((float *)llvm_cbe_tmp__8, 0x1.7cp8, 0x0p0, 0x1.ep3, (float *)llvm_cbe_tmp__9, (float *)llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.7cp8);
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = %f",0x1.ep3);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts3.1 = call i32 @puts(i8* getelementptr inbounds ([29 x i8]* @aesl_internal_str3, i64 0, i64 0)), !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts3_2e_1_count);
  puts(( char *)((&aesl_internal_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 29
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts3.1 = 0x%X",llvm_cbe_puts3_2e_1);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = load float* %%10, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_52_count);
  llvm_cbe_tmp__29 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__29, *(int*)(&llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = fpext float %%29 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_53_count);
  llvm_cbe_tmp__30 = (double )((double )llvm_cbe_tmp__29);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__30, *(long long*)(&llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str5, i64 0, i64 0), double %%30) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_54_count);
  printf(( char *)((&aesl_internal__OC_str5[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__30);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__30, *(long long*)(&llvm_cbe_tmp__30));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__31);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_55_count);
  llvm_cbe_tmp__32 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__32, *(int*)(&llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = fpext float %%32 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_56_count);
  llvm_cbe_tmp__33 = (double )((double )llvm_cbe_tmp__32);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__33, *(long long*)(&llvm_cbe_tmp__33));
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str6, i64 0, i64 0), double %%33) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_57_count);
  printf(( char *)((&aesl_internal__OC_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__33);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__33, *(long long*)(&llvm_cbe_tmp__33));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__34);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = load float* %%10, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_58_count);
  llvm_cbe_tmp__35 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__35, *(int*)(&llvm_cbe_tmp__35));

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%35, float* %%9, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_59_count);
  *llvm_cbe_tmp__9 = llvm_cbe_tmp__35;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__35);

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_60_count);
  llvm_cbe_tmp__36 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%36, float* %%12, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_61_count);
  *llvm_cbe_tmp__12 = llvm_cbe_tmp__36;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__36);

#ifdef AESL_BC_SIM
  if (!(((signed long long )2ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'x_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_62_count);
  llvm_cbe_tmp__37 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__37, *(int*)(&llvm_cbe_tmp__37));
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = fadd float %%37, 1.000000e+00, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_63_count);
  llvm_cbe_tmp__38 = (float )((float )(llvm_cbe_tmp__37 + 0x1p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__38, *(int*)(&llvm_cbe_tmp__38));

#ifdef AESL_BC_SIM
  assert(((signed long long )2ull) < 3 && "Write access out of array 'x_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%38, float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_64_count);
  *llvm_cbe_tmp__7 = llvm_cbe_tmp__38;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__38);
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([23 x i8]* @aesl_internal_.str2, i64 0, i64 0), i32 3) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_69_count);
  printf(( char *)((&aesl_internal__OC_str2[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 23
#endif
])), 3u);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%X",3u);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__39);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )2ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'x_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = load float* %%7, align 4, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_70_count);
  llvm_cbe_tmp__40 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__40, *(int*)(&llvm_cbe_tmp__40));
if (AESL_DEBUG_TRACE)
printf("\n  %%41 = fpext float %%40 to double, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_71_count);
  llvm_cbe_tmp__41 = (double )((double )llvm_cbe_tmp__40);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__41, *(long long*)(&llvm_cbe_tmp__41));
if (AESL_DEBUG_TRACE)
printf("\n  %%42 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([27 x i8]* @aesl_internal_.str3, i64 0, i64 0), double %%41, double 1.500000e+01) nounwind, !dbg !4 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_72_count);
  printf(( char *)((&aesl_internal__OC_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 27
#endif
])), llvm_cbe_tmp__41, 0x1.ep3);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__41, *(long long*)(&llvm_cbe_tmp__41));
printf("\nArgument  = %lf",0x1.ep3);
printf("\nReturn  = 0x%X",llvm_cbe_tmp__42);
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @myFunction(float* %%8, float 3.800000e+02, float 0.000000e+00, float 1.500000e+01, float* %%9, float* %%10) nounwind, !dbg !3 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_73_count);
  myFunction((float *)llvm_cbe_tmp__8, 0x1.7cp8, 0x0p0, 0x1.ep3, (float *)llvm_cbe_tmp__9, (float *)llvm_cbe_tmp__10);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.7cp8);
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = %f",0x1.ep3);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts3.2 = call i32 @puts(i8* getelementptr inbounds ([29 x i8]* @aesl_internal_str3, i64 0, i64 0)), !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts3_2e_2_count);
  puts(( char *)((&aesl_internal_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 29
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts3.2 = 0x%X",llvm_cbe_puts3_2e_2);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%43 = load float* %%10, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_74_count);
  llvm_cbe_tmp__43 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__43, *(int*)(&llvm_cbe_tmp__43));
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = fpext float %%43 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_75_count);
  llvm_cbe_tmp__44 = (double )((double )llvm_cbe_tmp__43);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__44, *(long long*)(&llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n  %%45 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str5, i64 0, i64 0), double %%44) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_76_count);
  printf(( char *)((&aesl_internal__OC_str5[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__44);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__44, *(long long*)(&llvm_cbe_tmp__44));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__45);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%46 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_77_count);
  llvm_cbe_tmp__46 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__46, *(int*)(&llvm_cbe_tmp__46));
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = fpext float %%46 to double, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_78_count);
  llvm_cbe_tmp__47 = (double )((double )llvm_cbe_tmp__46);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__47, *(long long*)(&llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @aesl_internal_.str6, i64 0, i64 0), double %%47) nounwind, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_79_count);
  printf(( char *)((&aesl_internal__OC_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__47);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %lf,  0x%llx",llvm_cbe_tmp__47, *(long long*)(&llvm_cbe_tmp__47));
printf("\nReturn  = 0x%X",llvm_cbe_tmp__48);
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%49 = load float* %%10, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_80_count);
  llvm_cbe_tmp__49 = (float )*llvm_cbe_tmp__10;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__49, *(int*)(&llvm_cbe_tmp__49));

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%49, float* %%9, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_81_count);
  *llvm_cbe_tmp__9 = llvm_cbe_tmp__49;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__49);

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'output' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%50 = load float* %%11, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_82_count);
  llvm_cbe_tmp__50 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__50, *(int*)(&llvm_cbe_tmp__50));

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'u00' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%50, float* %%12, align 4, !dbg !5 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_83_count);
  *llvm_cbe_tmp__12 = llvm_cbe_tmp__50;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__50);

#ifdef AESL_BC_SIM
  if (!(((signed long long )2ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'x_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%51 = load float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_84_count);
  llvm_cbe_tmp__51 = (float )*llvm_cbe_tmp__7;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__51, *(int*)(&llvm_cbe_tmp__51));
if (AESL_DEBUG_TRACE)
printf("\n  %%52 = fadd float %%51, 1.000000e+00, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_85_count);
  llvm_cbe_tmp__52 = (float )((float )(llvm_cbe_tmp__51 + 0x1p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__52, *(int*)(&llvm_cbe_tmp__52));

#ifdef AESL_BC_SIM
  assert(((signed long long )2ull) < 3 && "Write access out of array 'x_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%52, float* %%7, align 4, !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_86_count);
  *llvm_cbe_tmp__7 = llvm_cbe_tmp__52;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__52);
if (AESL_DEBUG_TRACE)
printf("\n  %%puts4 = call i32 @puts(i8* getelementptr inbounds ([47 x i8]* @aesl_internal_str4, i64 0, i64 0)), !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts4_count);
  puts(( char *)((&aesl_internal_str4[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 47
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts4 = 0x%X",llvm_cbe_puts4);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts5 = call i32 @puts(i8* getelementptr inbounds ([46 x i8]* @aesl_internal_str5, i64 0, i64 0)), !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts5_count);
  puts(( char *)((&aesl_internal_str5[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 46
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts5 = 0x%X",llvm_cbe_puts5);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%puts6 = call i32 @puts(i8* getelementptr inbounds ([46 x i8]* @aesl_internal_str6, i64 0, i64 0)), !dbg !6 for 0x%I64xth hint within @main  --> \n", ++aesl_llvm_cbe_puts6_count);
  puts(( char *)((&aesl_internal_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 46
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn puts6 = 0x%X",llvm_cbe_puts6);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @main}\n");
  return 0u;
}

