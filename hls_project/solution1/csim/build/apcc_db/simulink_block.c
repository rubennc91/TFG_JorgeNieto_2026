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

/* Structure contents */
struct l_struct_OC_OSQPScaling {
  float field0;
  float *field1;
  float *field2;
  float field3;
  float *field4;
  float *field5;
};


/* External Global Variable Declarations */
extern l_struct_OC_OSQPScaling scaling;
extern float qdata[15];
extern float ldata[19];
extern float udata[19];
extern float work_x[15];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void init_workspace_manually(void);
void myFunction(float *llvm_cbe_x_ini, float llvm_cbe_Vsd, float llvm_cbe_Vsq, float llvm_cbe_iL, float *llvm_cbe_u00, float *llvm_cbe_outputVector);
float fabsf(float );
void prea_vec_copy(float *, float *, signed long long );
static void aesl_internal_local_inverse_matrix_2x2(float llvm_cbe_a, float llvm_cbe_b, float llvm_cbe_c, float llvm_cbe_d, float (*llvm_cbe_m)[2]);
void referencia(float *, float );
void calculateV(float *, float (*)[2], float *, float *);
void atualizar_restricao(float *, float *, float *, float *);
void atualizar_restricao_v(float *, float *, float , float (*)[2], float *);
void atualizar_A(float (*)[2]);
signed long long osqp_solve(void);
static void aesl_internal_local_multiplyMatrixVector(float (*llvm_cbe_m)[2], float *llvm_cbe_v, float *llvm_cbe_res);


/* Global Variable Definitions and Initialization */
static unsigned int aesl_internal_myFunction_OC_is_initialized;


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

void init_workspace_manually(void) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @init_workspace_manually\n");
if (AESL_DEBUG_TRACE)
printf("\n  store float 1.000000e+00, float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 0), align 8, !dbg !27 for 0x%I64xth hint within @init_workspace_manually  --> \n", ++aesl_llvm_cbe_1_count);
  *((&scaling.field0)) = 0x1p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1p0);
if (AESL_DEBUG_TRACE)
printf("\n  store float 1.000000e+00, float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !27 for 0x%I64xth hint within @init_workspace_manually  --> \n", ++aesl_llvm_cbe_2_count);
  *((&scaling.field3)) = 0x1p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1p0);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @init_workspace_manually}\n");
  return;
}


void myFunction(float *llvm_cbe_x_ini, float llvm_cbe_Vsd, float llvm_cbe_Vsq, float llvm_cbe_iL, float *llvm_cbe_u00, float *llvm_cbe_outputVector) {
  static  unsigned long long aesl_llvm_cbe_Ax_count = 0;
  float llvm_cbe_Ax[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_Ex_count = 0;
  float llvm_cbe_Ex[2][2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_Einv_count = 0;
  float llvm_cbe_Einv[2][2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_z_ini_count = 0;
  float llvm_cbe_z_ini[3];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_v00_count = 0;
  float llvm_cbe_v00[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_v_count = 0;
  float llvm_cbe_v[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_u_count = 0;
  float llvm_cbe_u[2];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_q_new_count = 0;
  float llvm_cbe_q_new[15];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_l_new_count = 0;
  float llvm_cbe_l_new[19];    /* Address-exposed local */
  static  unsigned long long aesl_llvm_cbe_u_new_count = 0;
  float llvm_cbe_u_new[19];    /* Address-exposed local */
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
  static  unsigned long long aesl_llvm_cbe_22_count = 0;
  static  unsigned long long aesl_llvm_cbe_23_count = 0;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  static  unsigned long long aesl_llvm_cbe_27_count = 0;
  static  unsigned long long aesl_llvm_cbe_28_count = 0;
  static  unsigned long long aesl_llvm_cbe_29_count = 0;
  static  unsigned long long aesl_llvm_cbe_30_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  unsigned int llvm_cbe_tmp__1;
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
  static  unsigned long long aesl_llvm_cbe_64_count = 0;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  float *llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  float llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  float llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  float llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  static  unsigned long long aesl_llvm_cbe_phitmp_count = 0;
  float llvm_cbe_phitmp;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_77_count = 0;
  float *llvm_cbe_tmp__6;
  static  unsigned long long aesl_llvm_cbe_78_count = 0;
  static  unsigned long long aesl_llvm_cbe_79_count = 0;
  float *llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_80_count = 0;
  static  unsigned long long aesl_llvm_cbe_81_count = 0;
  float *llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_82_count = 0;
  static  unsigned long long aesl_llvm_cbe_83_count = 0;
  float llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_84_count = 0;
  float llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_85_count = 0;
  float *llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_86_count = 0;
  float llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_87_count = 0;
  float llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  float llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  float llvm_cbe_tmp__15;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;
  static  unsigned long long aesl_llvm_cbe_92_count = 0;
  float llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_93_count = 0;
  float llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_94_count = 0;
  float llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_95_count = 0;
  float llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_96_count = 0;
  float llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_97_count = 0;
  float llvm_cbe_tmp__21;
  static  unsigned long long aesl_llvm_cbe_98_count = 0;
  float llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_99_count = 0;
  float llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_100_count = 0;
  float llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_101_count = 0;
  float llvm_cbe_tmp__25;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  float llvm_cbe_tmp__26;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  float llvm_cbe_tmp__27;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  float llvm_cbe_tmp__28;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  float llvm_cbe_tmp__29;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  float llvm_cbe_tmp__30;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  float llvm_cbe_tmp__31;
  static  unsigned long long aesl_llvm_cbe_115_count = 0;
  float llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_116_count = 0;
  float llvm_cbe_tmp__33;
  static  unsigned long long aesl_llvm_cbe_117_count = 0;
  float llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_118_count = 0;
  float llvm_cbe_tmp__35;
  static  unsigned long long aesl_llvm_cbe_119_count = 0;
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_120_count = 0;
  float llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_121_count = 0;
  float llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_122_count = 0;
  float llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_123_count = 0;
  float llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_124_count = 0;
  float llvm_cbe_tmp__41;
  static  unsigned long long aesl_llvm_cbe_125_count = 0;
  float llvm_cbe_tmp__42;
  static  unsigned long long aesl_llvm_cbe_126_count = 0;
  float llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;
  float llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_128_count = 0;
  float llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_129_count = 0;
  float llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_130_count = 0;
  float llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_131_count = 0;
  float llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_132_count = 0;
  float llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_133_count = 0;
  float llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_134_count = 0;
  float llvm_cbe_tmp__51;
  static  unsigned long long aesl_llvm_cbe_135_count = 0;
  float llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_136_count = 0;
  float llvm_cbe_tmp__53;
  static  unsigned long long aesl_llvm_cbe_137_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
  static  unsigned long long aesl_llvm_cbe_141_count = 0;
  static  unsigned long long aesl_llvm_cbe_142_count = 0;
  static  unsigned long long aesl_llvm_cbe_143_count = 0;
  static  unsigned long long aesl_llvm_cbe_144_count = 0;
  static  unsigned long long aesl_llvm_cbe_145_count = 0;
  static  unsigned long long aesl_llvm_cbe_146_count = 0;
  float llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_147_count = 0;
  float llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_148_count = 0;
  static  unsigned long long aesl_llvm_cbe_149_count = 0;
  static  unsigned long long aesl_llvm_cbe_150_count = 0;
  static  unsigned long long aesl_llvm_cbe_151_count = 0;
  float llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_152_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  float llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  float llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  float llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  float *llvm_cbe_tmp__62;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  float *llvm_cbe_tmp__63;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  float *llvm_cbe_tmp__64;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  float *llvm_cbe_tmp__65;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  float *llvm_cbe_tmp__66;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  float *llvm_cbe_tmp__67;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  float (*llvm_cbe_tmp__68)[2];
  static  unsigned long long aesl_llvm_cbe_175_count = 0;
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
  float llvm_cbe_tmp__69;
  static  unsigned long long aesl_llvm_cbe_177_count = 0;
  float *llvm_cbe_tmp__70;
  static  unsigned long long aesl_llvm_cbe_178_count = 0;
  static  unsigned long long aesl_llvm_cbe_179_count = 0;
  float llvm_cbe_tmp__71;
  static  unsigned long long aesl_llvm_cbe_180_count = 0;
  float *llvm_cbe_tmp__72;
  static  unsigned long long aesl_llvm_cbe_181_count = 0;
  static  unsigned long long aesl_llvm_cbe_182_count = 0;
  float llvm_cbe_tmp__73;
  static  unsigned long long aesl_llvm_cbe_183_count = 0;
  float llvm_cbe_tmp__74;
  static  unsigned long long aesl_llvm_cbe_184_count = 0;
  float *llvm_cbe_tmp__75;
  static  unsigned long long aesl_llvm_cbe_185_count = 0;
  static  unsigned long long aesl_llvm_cbe_186_count = 0;
  static  unsigned long long aesl_llvm_cbe_187_count = 0;
  float (*llvm_cbe_tmp__76)[2];
  static  unsigned long long aesl_llvm_cbe_188_count = 0;
  float *llvm_cbe_tmp__77;
  static  unsigned long long aesl_llvm_cbe_189_count = 0;
  static  unsigned long long aesl_llvm_cbe_190_count = 0;
  static  unsigned long long aesl_llvm_cbe_191_count = 0;
  float llvm_cbe_tmp__78;
  static  unsigned long long aesl_llvm_cbe_192_count = 0;
  static  unsigned long long aesl_llvm_cbe_193_count = 0;
  static  unsigned long long aesl_llvm_cbe_194_count = 0;
  unsigned long long llvm_cbe_tmp__79;
  static  unsigned long long aesl_llvm_cbe_195_count = 0;
  float llvm_cbe_tmp__80;
  static  unsigned long long aesl_llvm_cbe_196_count = 0;
  float *llvm_cbe_tmp__81;
  static  unsigned long long aesl_llvm_cbe_197_count = 0;
  float llvm_cbe_tmp__82;
  static  unsigned long long aesl_llvm_cbe_198_count = 0;
  float *llvm_cbe_tmp__83;
  static  unsigned long long aesl_llvm_cbe_199_count = 0;
  float llvm_cbe_tmp__84;
  static  unsigned long long aesl_llvm_cbe_200_count = 0;
  float llvm_cbe_tmp__85;
  static  unsigned long long aesl_llvm_cbe_201_count = 0;
  static  unsigned long long aesl_llvm_cbe_202_count = 0;
  float llvm_cbe_tmp__86;
  static  unsigned long long aesl_llvm_cbe_203_count = 0;
  float llvm_cbe_tmp__87;
  static  unsigned long long aesl_llvm_cbe_204_count = 0;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  float *llvm_cbe_tmp__88;
  static  unsigned long long aesl_llvm_cbe_206_count = 0;
  static  unsigned long long aesl_llvm_cbe_207_count = 0;
  float llvm_cbe_tmp__89;
  static  unsigned long long aesl_llvm_cbe_208_count = 0;
  static  unsigned long long aesl_llvm_cbe_209_count = 0;
  float llvm_cbe_tmp__90;
  static  unsigned long long aesl_llvm_cbe_210_count = 0;
  float *llvm_cbe_tmp__91;
  static  unsigned long long aesl_llvm_cbe_211_count = 0;
  static  unsigned long long aesl_llvm_cbe_212_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @myFunction\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i32* @aesl_internal_myFunction.is_initialized, align 4, !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_47_count);
  llvm_cbe_tmp__1 = (unsigned int )*(&aesl_internal_myFunction_OC_is_initialized);
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", llvm_cbe_tmp__1);
  if (((llvm_cbe_tmp__1&4294967295U) == (0u&4294967295U))) {
    goto llvm_cbe_tmp__92;
  } else {
    goto llvm_cbe_tmp__93;
  }

llvm_cbe_tmp__92:
if (AESL_DEBUG_TRACE)
printf("\n  call void @init_workspace_manually(), !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_50_count);
  init_workspace_manually();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i32 1, i32* @aesl_internal_myFunction.is_initialized, align 4, !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_51_count);
  *(&aesl_internal_myFunction_OC_is_initialized) = 1u;
if (AESL_DEBUG_TRACE)
printf("\n = 0x%X\n", 1u);
  goto llvm_cbe_tmp__93;

llvm_cbe_tmp__93:
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds float* %%x_ini, i64 2, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_65_count);
  llvm_cbe_tmp__2 = (float *)(&llvm_cbe_x_ini[(((signed long long )2ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_66_count);
  llvm_cbe_tmp__3 = (float )*llvm_cbe_tmp__2;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__3, *(int*)(&llvm_cbe_tmp__3));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = call float @fabsf(float %%iL) nounwind, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_71_count);
  llvm_cbe_tmp__4 = (float )fabsf(llvm_cbe_iL);
if (AESL_DEBUG_TRACE) {
printf("\nArgument iL = %f,  0x%x",llvm_cbe_iL, *(int*)(&llvm_cbe_iL));
printf("\nReturn  = %f",llvm_cbe_tmp__4);
}
  if ((llvm_fcmp_olt(llvm_cbe_tmp__4, 0x1.99999ap-4))) {
    llvm_cbe_storemerge__PHI_TEMPORARY = (float )0x1.4p3;   /* for PHI node */
    goto llvm_cbe_tmp__94;
  } else {
    goto llvm_cbe_tmp__95;
  }

llvm_cbe_tmp__95:
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fdiv float %%6, %%iL, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_74_count);
  llvm_cbe_tmp__5 = (float )((float )(llvm_cbe_tmp__3 / llvm_cbe_iL));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__5, *(int*)(&llvm_cbe_tmp__5));
if (AESL_DEBUG_TRACE)
printf("\n  %%phitmp = fmul float %%10, 0x3F50624DE000000 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_phitmp_count);
  llvm_cbe_phitmp = (float )((float )(llvm_cbe_tmp__5 * 0x1.0624dep-10));
if (AESL_DEBUG_TRACE)
printf("\nphitmp = %f,  0x%x\n", llvm_cbe_phitmp, *(int*)(&llvm_cbe_phitmp));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_phitmp;   /* for PHI node */
  goto llvm_cbe_tmp__94;

llvm_cbe_tmp__94:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%phitmp, %%9 ], [ 1.000000e+01, %%4 ], !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\nphitmp = %f",llvm_cbe_phitmp);
printf("\n = %f",0x1.4p3);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds [15 x float]* %%q_new, i64 0, i64 0, !dbg !33 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_77_count);
  llvm_cbe_tmp__6 = (float *)(&llvm_cbe_q_new[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @prea_vec_copy(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* %%12, i64 15) nounwind, !dbg !33 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_78_count);
  prea_vec_copy((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)llvm_cbe_tmp__6, 15ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",15ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [19 x float]* %%l_new, i64 0, i64 0, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_79_count);
  llvm_cbe_tmp__7 = (float *)(&llvm_cbe_l_new[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @prea_vec_copy(float* getelementptr inbounds ([19 x float]* @ldata, i64 0, i64 0), float* %%13, i64 19) nounwind, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_80_count);
  prea_vec_copy((float *)((&ldata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)llvm_cbe_tmp__7, 19ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",19ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds [19 x float]* %%u_new, i64 0, i64 0, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_81_count);
  llvm_cbe_tmp__8 = (float *)(&llvm_cbe_u_new[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @prea_vec_copy(float* getelementptr inbounds ([19 x float]* @udata, i64 0, i64 0), float* %%14, i64 19) nounwind, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_82_count);
  prea_vec_copy((float *)((&udata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)llvm_cbe_tmp__8, 19ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",19ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load float* %%x_ini, align 4, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_83_count);
  llvm_cbe_tmp__9 = (float )*llvm_cbe_x_ini;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__9, *(int*)(&llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = fmul float %%15, 0xC073A28C60000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_84_count);
  llvm_cbe_tmp__10 = (float )((float )(llvm_cbe_tmp__9 * -0x1.3a28c6p8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__10, *(int*)(&llvm_cbe_tmp__10));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = getelementptr inbounds float* %%x_ini, i64 1, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_85_count);
  llvm_cbe_tmp__11 = (float *)(&llvm_cbe_x_ini[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = load float* %%17, align 4, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_86_count);
  llvm_cbe_tmp__12 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__12, *(int*)(&llvm_cbe_tmp__12));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fmul float %%18, 0x3FB99999A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_87_count);
  llvm_cbe_tmp__13 = (float )((float )(llvm_cbe_tmp__12 * 0x1.99999ap-4));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__13, *(int*)(&llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = fdiv float %%19, 0x3F747AE140000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_88_count);
  llvm_cbe_tmp__14 = (float )((float )(llvm_cbe_tmp__13 / 0x1.47ae14p-8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__14, *(int*)(&llvm_cbe_tmp__14));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = fsub float %%16, %%20, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_89_count);
  llvm_cbe_tmp__15 = (float )((float )(llvm_cbe_tmp__10 - llvm_cbe_tmp__14));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__15, *(int*)(&llvm_cbe_tmp__15));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fmul float %%15, %%Vsd, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_92_count);
  llvm_cbe_tmp__16 = (float )((float )(llvm_cbe_tmp__9 * llvm_cbe_Vsd));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__16, *(int*)(&llvm_cbe_tmp__16));
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = fmul float %%18, %%Vsq, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_93_count);
  llvm_cbe_tmp__17 = (float )((float )(llvm_cbe_tmp__12 * llvm_cbe_Vsq));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__17, *(int*)(&llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = fadd float %%22, %%23, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_94_count);
  llvm_cbe_tmp__18 = (float )((float )(llvm_cbe_tmp__16 + llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__18, *(int*)(&llvm_cbe_tmp__18));
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = fmul float %%15, %%15, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_95_count);
  llvm_cbe_tmp__19 = (float )((float )(llvm_cbe_tmp__9 * llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__19, *(int*)(&llvm_cbe_tmp__19));
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = fmul float %%18, %%18, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_96_count);
  llvm_cbe_tmp__20 = (float )((float )(llvm_cbe_tmp__12 * llvm_cbe_tmp__12));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__20, *(int*)(&llvm_cbe_tmp__20));
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = fadd float %%25, %%26, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_97_count);
  llvm_cbe_tmp__21 = (float )((float )(llvm_cbe_tmp__19 + llvm_cbe_tmp__20));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__21, *(int*)(&llvm_cbe_tmp__21));
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = fmul float %%27, 0x3FB99999A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_98_count);
  llvm_cbe_tmp__22 = (float )((float )(llvm_cbe_tmp__21 * 0x1.99999ap-4));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__22, *(int*)(&llvm_cbe_tmp__22));
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = fsub float %%24, %%28, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_99_count);
  llvm_cbe_tmp__23 = (float )((float )(llvm_cbe_tmp__18 - llvm_cbe_tmp__22));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__23, *(int*)(&llvm_cbe_tmp__23));
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = fmul float %%29, 3.000000e+00, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_100_count);
  llvm_cbe_tmp__24 = (float )((float )(llvm_cbe_tmp__23 * 0x1.8p1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__24, *(int*)(&llvm_cbe_tmp__24));
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%5, align 4, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_101_count);
  llvm_cbe_tmp__25 = (float )*llvm_cbe_tmp__2;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__25, *(int*)(&llvm_cbe_tmp__25));
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = fmul float %%31, 0x3F60624DE0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_102_count);
  llvm_cbe_tmp__26 = (float )((float )(llvm_cbe_tmp__25 * 0x1.0624dep-9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__26, *(int*)(&llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = fdiv float %%30, %%32, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_103_count);
  llvm_cbe_tmp__27 = (float )((float )(llvm_cbe_tmp__24 / llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__27, *(int*)(&llvm_cbe_tmp__27));
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = fdiv float 1.000000e+00, %%32, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_108_count);
  llvm_cbe_tmp__28 = (float )((float )(0x1p0 / llvm_cbe_tmp__26));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__28, *(int*)(&llvm_cbe_tmp__28));
if (AESL_DEBUG_TRACE)
printf("\n  %%35 = fdiv float 1.000000e+00, %%storemerge, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_112_count);
  llvm_cbe_tmp__29 = (float )((float )(0x1p0 / llvm_cbe_storemerge));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__29, *(int*)(&llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = fdiv float %%33, %%31, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_113_count);
  llvm_cbe_tmp__30 = (float )((float )(llvm_cbe_tmp__27 / llvm_cbe_tmp__25));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__30, *(int*)(&llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = fadd float %%35, %%36, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_114_count);
  llvm_cbe_tmp__31 = (float )((float )(llvm_cbe_tmp__29 + llvm_cbe_tmp__30));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__31, *(int*)(&llvm_cbe_tmp__31));
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = fdiv float %%31, %%storemerge, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_115_count);
  llvm_cbe_tmp__32 = (float )((float )(llvm_cbe_tmp__25 / llvm_cbe_storemerge));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__32, *(int*)(&llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = fsub float %%38, %%33, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_116_count);
  llvm_cbe_tmp__33 = (float )((float )(llvm_cbe_tmp__32 - llvm_cbe_tmp__27));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__33, *(int*)(&llvm_cbe_tmp__33));
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = fmul float %%37, %%39, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_117_count);
  llvm_cbe_tmp__34 = (float )((float )(llvm_cbe_tmp__31 * llvm_cbe_tmp__33));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__34, *(int*)(&llvm_cbe_tmp__34));
if (AESL_DEBUG_TRACE)
printf("\n  %%41 = fmul float %%15, 0x4073A28C60000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_118_count);
  llvm_cbe_tmp__35 = (float )((float )(llvm_cbe_tmp__9 * 0x1.3a28c6p8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__35, *(int*)(&llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n  %%42 = fadd float %%41, %%20, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_119_count);
  llvm_cbe_tmp__36 = (float )((float )(llvm_cbe_tmp__35 + llvm_cbe_tmp__14));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));
if (AESL_DEBUG_TRACE)
printf("\n  %%43 = fmul float %%42, 3.000000e+00, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_120_count);
  llvm_cbe_tmp__37 = (float )((float )(llvm_cbe_tmp__36 * 0x1.8p1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__37, *(int*)(&llvm_cbe_tmp__37));
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = fmul float %%18, 0x3FC99999A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_121_count);
  llvm_cbe_tmp__38 = (float )((float )(llvm_cbe_tmp__12 * 0x1.99999ap-3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__38, *(int*)(&llvm_cbe_tmp__38));
if (AESL_DEBUG_TRACE)
printf("\n  %%45 = fsub float %%Vsq, %%44, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_122_count);
  llvm_cbe_tmp__39 = (float )((float )(llvm_cbe_Vsq - llvm_cbe_tmp__38));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__39, *(int*)(&llvm_cbe_tmp__39));
if (AESL_DEBUG_TRACE)
printf("\n  %%46 = fmul float %%43, %%45, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_123_count);
  llvm_cbe_tmp__40 = (float )((float )(llvm_cbe_tmp__37 * llvm_cbe_tmp__39));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__40, *(int*)(&llvm_cbe_tmp__40));
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = fmul float %%46, %%34, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_124_count);
  llvm_cbe_tmp__41 = (float )((float )(llvm_cbe_tmp__40 * llvm_cbe_tmp__28));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__41, *(int*)(&llvm_cbe_tmp__41));
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = fsub float %%40, %%47, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_125_count);
  llvm_cbe_tmp__42 = (float )((float )(llvm_cbe_tmp__34 - llvm_cbe_tmp__41));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__42, *(int*)(&llvm_cbe_tmp__42));
if (AESL_DEBUG_TRACE)
printf("\n  %%49 = fmul float %%15, 0x3FC99999A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_126_count);
  llvm_cbe_tmp__43 = (float )((float )(llvm_cbe_tmp__9 * 0x1.99999ap-3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__43, *(int*)(&llvm_cbe_tmp__43));
if (AESL_DEBUG_TRACE)
printf("\n  %%50 = fsub float %%Vsd, %%49, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_127_count);
  llvm_cbe_tmp__44 = (float )((float )(llvm_cbe_Vsd - llvm_cbe_tmp__43));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__44, *(int*)(&llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n  %%51 = fmul float %%50, 3.000000e+00, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_128_count);
  llvm_cbe_tmp__45 = (float )((float )(llvm_cbe_tmp__44 * 0x1.8p1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__45, *(int*)(&llvm_cbe_tmp__45));
if (AESL_DEBUG_TRACE)
printf("\n  %%52 = fmul float %%18, 0x4073A28C60000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_129_count);
  llvm_cbe_tmp__46 = (float )((float )(llvm_cbe_tmp__12 * 0x1.3a28c6p8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__46, *(int*)(&llvm_cbe_tmp__46));
if (AESL_DEBUG_TRACE)
printf("\n  %%53 = fdiv float %%Vsd, 0x3F747AE140000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_130_count);
  llvm_cbe_tmp__47 = (float )((float )(llvm_cbe_Vsd / 0x1.47ae14p-8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__47, *(int*)(&llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n  %%54 = fadd float %%52, %%53, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_131_count);
  llvm_cbe_tmp__48 = (float )((float )(llvm_cbe_tmp__46 + llvm_cbe_tmp__47));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__48, *(int*)(&llvm_cbe_tmp__48));
if (AESL_DEBUG_TRACE)
printf("\n  %%55 = fmul float %%15, 0x3FB99999A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_132_count);
  llvm_cbe_tmp__49 = (float )((float )(llvm_cbe_tmp__9 * 0x1.99999ap-4));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__49, *(int*)(&llvm_cbe_tmp__49));
if (AESL_DEBUG_TRACE)
printf("\n  %%56 = fdiv float %%55, 0x3F747AE140000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_133_count);
  llvm_cbe_tmp__50 = (float )((float )(llvm_cbe_tmp__49 / 0x1.47ae14p-8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__50, *(int*)(&llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n  %%57 = fsub float %%54, %%56, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_134_count);
  llvm_cbe_tmp__51 = (float )((float )(llvm_cbe_tmp__48 - llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__51, *(int*)(&llvm_cbe_tmp__51));
if (AESL_DEBUG_TRACE)
printf("\n  %%58 = fmul float %%51, %%57, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_135_count);
  llvm_cbe_tmp__52 = (float )((float )(llvm_cbe_tmp__45 * llvm_cbe_tmp__51));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__52, *(int*)(&llvm_cbe_tmp__52));
if (AESL_DEBUG_TRACE)
printf("\n  %%59 = fmul float %%58, %%34, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_136_count);
  llvm_cbe_tmp__53 = (float )((float )(llvm_cbe_tmp__52 * llvm_cbe_tmp__28));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__53, *(int*)(&llvm_cbe_tmp__53));
if (AESL_DEBUG_TRACE)
printf("\n  %%60 = fadd float %%48, %%59, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_137_count);
  llvm_cbe_tmp__54 = (float )((float )(llvm_cbe_tmp__42 + llvm_cbe_tmp__53));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  %%61 = fmul float %%31, 0x3EE4F8B5A0000000, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_146_count);
  llvm_cbe_tmp__55 = (float )((float )(llvm_cbe_tmp__25 * 0x1.4f8b5ap-17));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__55, *(int*)(&llvm_cbe_tmp__55));
if (AESL_DEBUG_TRACE)
printf("\n  %%62 = fdiv float 1.000000e+00, %%61, !dbg !27 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_147_count);
  llvm_cbe_tmp__56 = (float )((float )(0x1p0 / llvm_cbe_tmp__55));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__56, *(int*)(&llvm_cbe_tmp__56));
if (AESL_DEBUG_TRACE)
printf("\n  %%63 = fsub float -0.000000e+00, %%51, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_151_count);
  llvm_cbe_tmp__57 = (float )((float )(-(llvm_cbe_tmp__45)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__57, *(int*)(&llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n  %%64 = fmul float %%62, %%63, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_152_count);
  llvm_cbe_tmp__58 = (float )((float )(llvm_cbe_tmp__56 * llvm_cbe_tmp__57));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n  %%65 = fmul float %%45, 3.000000e+00, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_156_count);
  llvm_cbe_tmp__59 = (float )((float )(llvm_cbe_tmp__39 * 0x1.8p1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__59, *(int*)(&llvm_cbe_tmp__59));
if (AESL_DEBUG_TRACE)
printf("\n  %%66 = fsub float -0.000000e+00, %%65, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_157_count);
  llvm_cbe_tmp__60 = (float )((float )(-(llvm_cbe_tmp__59)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__60, *(int*)(&llvm_cbe_tmp__60));
if (AESL_DEBUG_TRACE)
printf("\n  %%67 = fmul float %%62, %%66, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_158_count);
  llvm_cbe_tmp__61 = (float )((float )(llvm_cbe_tmp__56 * llvm_cbe_tmp__60));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__61, *(int*)(&llvm_cbe_tmp__61));
if (AESL_DEBUG_TRACE)
printf("\n  %%68 = getelementptr inbounds [2 x float]* %%Ax, i64 0, i64 0, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_162_count);
  llvm_cbe_tmp__62 = (float *)(&llvm_cbe_Ax[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'Ax' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%21, float* %%68, align 4, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_163_count);
  *llvm_cbe_tmp__62 = llvm_cbe_tmp__15;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__15);
if (AESL_DEBUG_TRACE)
printf("\n  %%69 = getelementptr inbounds [2 x float]* %%Ax, i64 0, i64 1, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_164_count);
  llvm_cbe_tmp__63 = (float *)(&llvm_cbe_Ax[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'Ax' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%60, float* %%69, align 4, !dbg !34 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_165_count);
  *llvm_cbe_tmp__63 = llvm_cbe_tmp__54;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__54);
if (AESL_DEBUG_TRACE)
printf("\n  %%70 = getelementptr inbounds [2 x [2 x float]]* %%Ex, i64 0, i64 0, i64 0, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_166_count);
  llvm_cbe_tmp__64 = (float *)(&llvm_cbe_Ex[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'Ex' bound?");
  assert(((signed long long )0ull) < 2 && "Write access out of array 'Ex' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%70, align 16, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_167_count);
  *llvm_cbe_tmp__64 = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
if (AESL_DEBUG_TRACE)
printf("\n  %%71 = getelementptr inbounds [2 x [2 x float]]* %%Ex, i64 0, i64 0, i64 1, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_168_count);
  llvm_cbe_tmp__65 = (float *)(&llvm_cbe_Ex[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'Ex' bound?");
  assert(((signed long long )1ull) < 2 && "Write access out of array 'Ex' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float -2.000000e+02, float* %%71, align 4, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_169_count);
  *llvm_cbe_tmp__65 = -0x1.9p7;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", -0x1.9p7);
if (AESL_DEBUG_TRACE)
printf("\n  %%72 = getelementptr inbounds [2 x [2 x float]]* %%Ex, i64 0, i64 1, i64 0, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_170_count);
  llvm_cbe_tmp__66 = (float *)(&llvm_cbe_Ex[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'Ex' bound?");
  assert(((signed long long )0ull) < 2 && "Write access out of array 'Ex' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%64, float* %%72, align 8, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_171_count);
  *llvm_cbe_tmp__66 = llvm_cbe_tmp__58;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__58);
if (AESL_DEBUG_TRACE)
printf("\n  %%73 = getelementptr inbounds [2 x [2 x float]]* %%Ex, i64 0, i64 1, i64 1, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_172_count);
  llvm_cbe_tmp__67 = (float *)(&llvm_cbe_Ex[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'Ex' bound?");
  assert(((signed long long )1ull) < 2 && "Write access out of array 'Ex' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%67, float* %%73, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_173_count);
  *llvm_cbe_tmp__67 = llvm_cbe_tmp__61;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__61);
if (AESL_DEBUG_TRACE)
printf("\n  %%74 = getelementptr inbounds [2 x [2 x float]]* %%Einv, i64 0, i64 0, !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_174_count);
  llvm_cbe_tmp__68 = (float (*)[2])(&llvm_cbe_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call fastcc void @aesl_internal_local_inverse_matrix_2x2(float 0.000000e+00, float -2.000000e+02, float %%64, float %%67, [2 x float]* %%74), !dbg !35 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_175_count);
  aesl_internal_local_inverse_matrix_2x2(0x0p0, -0x1.9p7, llvm_cbe_tmp__58, llvm_cbe_tmp__61, llvm_cbe_tmp__68);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = %f",-0x1.9p7);
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__61, *(int*)(&llvm_cbe_tmp__61));
}
if (AESL_DEBUG_TRACE)
printf("\n  %%75 = load float* %%17, align 4, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_176_count);
  llvm_cbe_tmp__69 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__69, *(int*)(&llvm_cbe_tmp__69));
if (AESL_DEBUG_TRACE)
printf("\n  %%76 = getelementptr inbounds [3 x float]* %%z_ini, i64 0, i64 0, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_177_count);
  llvm_cbe_tmp__70 = (float *)(&llvm_cbe_z_ini[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 3
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 3 && "Write access out of array 'z_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%75, float* %%76, align 4, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_178_count);
  *llvm_cbe_tmp__70 = llvm_cbe_tmp__69;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__69);
if (AESL_DEBUG_TRACE)
printf("\n  %%77 = load float* %%5, align 4, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_179_count);
  llvm_cbe_tmp__71 = (float )*llvm_cbe_tmp__2;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__71, *(int*)(&llvm_cbe_tmp__71));
if (AESL_DEBUG_TRACE)
printf("\n  %%78 = getelementptr inbounds [3 x float]* %%z_ini, i64 0, i64 1, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_180_count);
  llvm_cbe_tmp__72 = (float *)(&llvm_cbe_z_ini[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 3
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 3 && "Write access out of array 'z_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%77, float* %%78, align 4, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_181_count);
  *llvm_cbe_tmp__72 = llvm_cbe_tmp__71;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__71);
if (AESL_DEBUG_TRACE)
printf("\n  %%79 = fdiv float %%77, %%storemerge, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_182_count);
  llvm_cbe_tmp__73 = (float )((float )(llvm_cbe_tmp__71 / llvm_cbe_storemerge));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__73, *(int*)(&llvm_cbe_tmp__73));
if (AESL_DEBUG_TRACE)
printf("\n  %%80 = fsub float %%33, %%79, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_183_count);
  llvm_cbe_tmp__74 = (float )((float )(llvm_cbe_tmp__27 - llvm_cbe_tmp__73));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__74, *(int*)(&llvm_cbe_tmp__74));
if (AESL_DEBUG_TRACE)
printf("\n  %%81 = getelementptr inbounds [3 x float]* %%z_ini, i64 0, i64 2, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_184_count);
  llvm_cbe_tmp__75 = (float *)(&llvm_cbe_z_ini[(((signed long long )2ull))
#ifdef AESL_BC_SIM
 % 3
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )2ull) < 3 && "Write access out of array 'z_ini' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%80, float* %%81, align 4, !dbg !28 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_185_count);
  *llvm_cbe_tmp__75 = llvm_cbe_tmp__74;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__74);
if (AESL_DEBUG_TRACE)
printf("\n  call void @referencia(float* %%12, float 3.800000e+02) nounwind, !dbg !33 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_186_count);
  referencia((float *)llvm_cbe_tmp__6, 0x1.7cp8);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.7cp8);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%82 = getelementptr inbounds [2 x [2 x float]]* %%Ex, i64 0, i64 0, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_187_count);
  llvm_cbe_tmp__76 = (float (*)[2])(&llvm_cbe_Ex[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%83 = getelementptr inbounds [2 x float]* %%v00, i64 0, i64 0, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_188_count);
  llvm_cbe_tmp__77 = (float *)(&llvm_cbe_v00[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @calculateV(float* %%68, [2 x float]* %%82, float* %%u00, float* %%83) nounwind, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_189_count);
  calculateV((float *)llvm_cbe_tmp__62, llvm_cbe_tmp__76, (float *)llvm_cbe_u00, (float *)llvm_cbe_tmp__77);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @atualizar_restricao(float* %%13, float* %%14, float* %%76, float* %%83) nounwind, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_190_count);
  atualizar_restricao((float *)llvm_cbe_tmp__7, (float *)llvm_cbe_tmp__8, (float *)llvm_cbe_tmp__70, (float *)llvm_cbe_tmp__77);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 3)) fprintf(stderr, "%s:%d: warning: Read access out of array 'z_ini' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%84 = load float* %%78, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_191_count);
  llvm_cbe_tmp__78 = (float )*llvm_cbe_tmp__72;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__78, *(int*)(&llvm_cbe_tmp__78));
if (AESL_DEBUG_TRACE)
printf("\n  call void @atualizar_restricao_v(float* %%13, float* %%14, float %%84, [2 x float]* %%74, float* %%68) nounwind, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_192_count);
  atualizar_restricao_v((float *)llvm_cbe_tmp__7, (float *)llvm_cbe_tmp__8, llvm_cbe_tmp__78, llvm_cbe_tmp__68, (float *)llvm_cbe_tmp__62);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__78, *(int*)(&llvm_cbe_tmp__78));
}
if (AESL_DEBUG_TRACE)
printf("\n  call void @atualizar_A([2 x float]* %%74) nounwind, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_193_count);
  atualizar_A(llvm_cbe_tmp__68);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%85 = call i64 @osqp_solve() nounwind, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_194_count);
  osqp_solve();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__79);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%86 = load float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 11), align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_195_count);
  llvm_cbe_tmp__80 = (float )*((&work_x[(((signed long long )11ull))
#ifdef AESL_BC_SIM
 % 15
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__80, *(int*)(&llvm_cbe_tmp__80));
if (AESL_DEBUG_TRACE)
printf("\n  %%87 = getelementptr inbounds [2 x float]* %%v, i64 0, i64 0, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_196_count);
  llvm_cbe_tmp__81 = (float *)(&llvm_cbe_v[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%88 = load float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 12), align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_197_count);
  llvm_cbe_tmp__82 = (float )*((&work_x[(((signed long long )12ull))
#ifdef AESL_BC_SIM
 % 15
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__82, *(int*)(&llvm_cbe_tmp__82));
if (AESL_DEBUG_TRACE)
printf("\n  %%89 = getelementptr inbounds [2 x float]* %%v, i64 0, i64 1, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_198_count);
  llvm_cbe_tmp__83 = (float *)(&llvm_cbe_v[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ax' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%90 = load float* %%68, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_199_count);
  llvm_cbe_tmp__84 = (float )*llvm_cbe_tmp__62;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__84, *(int*)(&llvm_cbe_tmp__84));
if (AESL_DEBUG_TRACE)
printf("\n  %%91 = fsub float %%86, %%90, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_200_count);
  llvm_cbe_tmp__85 = (float )((float )(llvm_cbe_tmp__80 - llvm_cbe_tmp__84));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__85, *(int*)(&llvm_cbe_tmp__85));

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'v' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%91, float* %%87, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_201_count);
  *llvm_cbe_tmp__81 = llvm_cbe_tmp__85;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__85);

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'Ax' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%92 = load float* %%69, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_202_count);
  llvm_cbe_tmp__86 = (float )*llvm_cbe_tmp__63;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__86, *(int*)(&llvm_cbe_tmp__86));
if (AESL_DEBUG_TRACE)
printf("\n  %%93 = fsub float %%88, %%92, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_203_count);
  llvm_cbe_tmp__87 = (float )((float )(llvm_cbe_tmp__82 - llvm_cbe_tmp__86));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__87, *(int*)(&llvm_cbe_tmp__87));

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'v' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%93, float* %%89, align 4, !dbg !36 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_204_count);
  *llvm_cbe_tmp__83 = llvm_cbe_tmp__87;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__87);
if (AESL_DEBUG_TRACE)
printf("\n  %%94 = getelementptr inbounds [2 x float]* %%u, i64 0, i64 0, !dbg !37 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_205_count);
  llvm_cbe_tmp__88 = (float *)(&llvm_cbe_u[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  call fastcc void @aesl_internal_local_multiplyMatrixVector([2 x float]* %%74, float* %%87, float* %%94), !dbg !37 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_206_count);
  aesl_internal_local_multiplyMatrixVector(llvm_cbe_tmp__68, (float *)llvm_cbe_tmp__81, (float *)llvm_cbe_tmp__88);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%95 = load float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 11), align 4, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_207_count);
  llvm_cbe_tmp__89 = (float )*((&work_x[(((signed long long )11ull))
#ifdef AESL_BC_SIM
 % 15
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__89, *(int*)(&llvm_cbe_tmp__89));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%95, float* %%outputVector, align 4, !dbg !29 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_208_count);
  *llvm_cbe_outputVector = llvm_cbe_tmp__89;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__89);
if (AESL_DEBUG_TRACE)
printf("\n  %%96 = load float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 12), align 4, !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_209_count);
  llvm_cbe_tmp__90 = (float )*((&work_x[(((signed long long )12ull))
#ifdef AESL_BC_SIM
 % 15
#endif
]));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__90, *(int*)(&llvm_cbe_tmp__90));
if (AESL_DEBUG_TRACE)
printf("\n  %%97 = getelementptr inbounds float* %%outputVector, i64 1, !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_210_count);
  llvm_cbe_tmp__91 = (float *)(&llvm_cbe_outputVector[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%96, float* %%97, align 4, !dbg !30 for 0x%I64xth hint within @myFunction  --> \n", ++aesl_llvm_cbe_211_count);
  *llvm_cbe_tmp__91 = llvm_cbe_tmp__90;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__90);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @myFunction}\n");
  return;
}


static void aesl_internal_local_inverse_matrix_2x2(float llvm_cbe_a, float llvm_cbe_b, float llvm_cbe_c, float llvm_cbe_d, float (*llvm_cbe_m)[2]) {
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
  float llvm_cbe_tmp__96;
  static  unsigned long long aesl_llvm_cbe_231_count = 0;
  float llvm_cbe_tmp__97;
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  float llvm_cbe_tmp__98;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  float llvm_cbe_tmp__99;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;
  float llvm_cbe_tmp__100;
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  float *llvm_cbe_tmp__101;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
  static  unsigned long long aesl_llvm_cbe_244_count = 0;
  float llvm_cbe_tmp__102;
  static  unsigned long long aesl_llvm_cbe_245_count = 0;
  float llvm_cbe_tmp__103;
  static  unsigned long long aesl_llvm_cbe_246_count = 0;
  float *llvm_cbe_tmp__104;
  static  unsigned long long aesl_llvm_cbe_247_count = 0;
  static  unsigned long long aesl_llvm_cbe_248_count = 0;
  float llvm_cbe_tmp__105;
  static  unsigned long long aesl_llvm_cbe_249_count = 0;
  float llvm_cbe_tmp__106;
  static  unsigned long long aesl_llvm_cbe_250_count = 0;
  float *llvm_cbe_tmp__107;
  static  unsigned long long aesl_llvm_cbe_251_count = 0;
  static  unsigned long long aesl_llvm_cbe_252_count = 0;
  float llvm_cbe_tmp__108;
  static  unsigned long long aesl_llvm_cbe_253_count = 0;
  float *llvm_cbe_tmp__109;
  static  unsigned long long aesl_llvm_cbe_254_count = 0;
  static  unsigned long long aesl_llvm_cbe_255_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @aesl_internal_local_inverse_matrix_2x2\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = fmul float %%a, %%d, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_230_count);
  llvm_cbe_tmp__96 = (float )((float )(llvm_cbe_a * llvm_cbe_d));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__96, *(int*)(&llvm_cbe_tmp__96));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = fmul float %%b, %%c, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_231_count);
  llvm_cbe_tmp__97 = (float )((float )(llvm_cbe_b * llvm_cbe_c));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__97, *(int*)(&llvm_cbe_tmp__97));
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = fsub float %%1, %%2, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_232_count);
  llvm_cbe_tmp__98 = (float )((float )(llvm_cbe_tmp__96 - llvm_cbe_tmp__97));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__98, *(int*)(&llvm_cbe_tmp__98));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fdiv float 1.000000e+00, %%3, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_235_count);
  llvm_cbe_tmp__99 = (float )((float )(0x1p0 / llvm_cbe_tmp__98));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__99, *(int*)(&llvm_cbe_tmp__99));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fmul float %%4, %%d, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_241_count);
  llvm_cbe_tmp__100 = (float )((float )(llvm_cbe_tmp__99 * llvm_cbe_d));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__100, *(int*)(&llvm_cbe_tmp__100));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds [2 x float]* %%m, i64 0, i64 0, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_242_count);
  llvm_cbe_tmp__101 = (float *)(&(*llvm_cbe_m)[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'm' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%5, float* %%6, align 4, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_243_count);
  *llvm_cbe_tmp__101 = llvm_cbe_tmp__100;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__100);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = fsub float -0.000000e+00, %%b, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_244_count);
  llvm_cbe_tmp__102 = (float )((float )(-(llvm_cbe_b)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__102, *(int*)(&llvm_cbe_tmp__102));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = fmul float %%4, %%7, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_245_count);
  llvm_cbe_tmp__103 = (float )((float )(llvm_cbe_tmp__99 * llvm_cbe_tmp__102));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__103, *(int*)(&llvm_cbe_tmp__103));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds [2 x float]* %%m, i64 0, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_246_count);
  llvm_cbe_tmp__104 = (float *)(&(*llvm_cbe_m)[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'm' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%8, float* %%9, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_247_count);
  *llvm_cbe_tmp__104 = llvm_cbe_tmp__103;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__103);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fsub float -0.000000e+00, %%c, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_248_count);
  llvm_cbe_tmp__105 = (float )((float )(-(llvm_cbe_c)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__105, *(int*)(&llvm_cbe_tmp__105));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = fmul float %%4, %%10, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_249_count);
  llvm_cbe_tmp__106 = (float )((float )(llvm_cbe_tmp__99 * llvm_cbe_tmp__105));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__106, *(int*)(&llvm_cbe_tmp__106));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds [2 x float]* %%m, i64 1, i64 0, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_250_count);
  llvm_cbe_tmp__107 = (float *)(&llvm_cbe_m[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )0ull) < 2 && "Write access out of array 'm' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%11, float* %%12, align 4, !dbg !28 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_251_count);
  *llvm_cbe_tmp__107 = llvm_cbe_tmp__106;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__106);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%4, %%a, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_252_count);
  llvm_cbe_tmp__108 = (float )((float )(llvm_cbe_tmp__99 * llvm_cbe_a));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__108, *(int*)(&llvm_cbe_tmp__108));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = getelementptr inbounds [2 x float]* %%m, i64 1, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_253_count);
  llvm_cbe_tmp__109 = (float *)(&llvm_cbe_m[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  assert(((signed long long )1ull) < 2 && "Write access out of array 'm' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%13, float* %%14, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_inverse_matrix_2x2  --> \n", ++aesl_llvm_cbe_254_count);
  *llvm_cbe_tmp__109 = llvm_cbe_tmp__108;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__108);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @aesl_internal_local_inverse_matrix_2x2}\n");
  return;
}


static void aesl_internal_local_multiplyMatrixVector(float (*llvm_cbe_m)[2], float *llvm_cbe_v, float *llvm_cbe_res) {
  static  unsigned long long aesl_llvm_cbe_256_count = 0;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
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
  float *llvm_cbe_tmp__110;
  static  unsigned long long aesl_llvm_cbe_270_count = 0;
  float llvm_cbe_tmp__111;
  static  unsigned long long aesl_llvm_cbe_271_count = 0;
  float llvm_cbe_tmp__112;
  static  unsigned long long aesl_llvm_cbe_272_count = 0;
  float llvm_cbe_tmp__113;
  static  unsigned long long aesl_llvm_cbe_273_count = 0;
  float *llvm_cbe_tmp__114;
  static  unsigned long long aesl_llvm_cbe_274_count = 0;
  float llvm_cbe_tmp__115;
  static  unsigned long long aesl_llvm_cbe_275_count = 0;
  float *llvm_cbe_tmp__116;
  static  unsigned long long aesl_llvm_cbe_276_count = 0;
  float llvm_cbe_tmp__117;
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  float llvm_cbe_tmp__118;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  float llvm_cbe_tmp__119;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  float *llvm_cbe_tmp__120;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  float llvm_cbe_tmp__121;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  float llvm_cbe_tmp__122;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  float llvm_cbe_tmp__123;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  float *llvm_cbe_tmp__124;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;
  float llvm_cbe_tmp__125;
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  float llvm_cbe_tmp__126;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
  float llvm_cbe_tmp__127;
  static  unsigned long long aesl_llvm_cbe_288_count = 0;
  float llvm_cbe_tmp__128;
  static  unsigned long long aesl_llvm_cbe_289_count = 0;
  float *llvm_cbe_tmp__129;
  static  unsigned long long aesl_llvm_cbe_290_count = 0;
  static  unsigned long long aesl_llvm_cbe_291_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @aesl_internal_local_multiplyMatrixVector\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds [2 x float]* %%m, i64 0, i64 0, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_269_count);
  llvm_cbe_tmp__110 = (float *)(&(*llvm_cbe_m)[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'm' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load float* %%1, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_270_count);
  llvm_cbe_tmp__111 = (float )*llvm_cbe_tmp__110;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__111, *(int*)(&llvm_cbe_tmp__111));
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* %%v, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_271_count);
  llvm_cbe_tmp__112 = (float )*llvm_cbe_v;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__112, *(int*)(&llvm_cbe_tmp__112));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fmul float %%2, %%3, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_272_count);
  llvm_cbe_tmp__113 = (float )((float )(llvm_cbe_tmp__111 * llvm_cbe_tmp__112));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__113, *(int*)(&llvm_cbe_tmp__113));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = getelementptr inbounds [2 x float]* %%m, i64 0, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_273_count);
  llvm_cbe_tmp__114 = (float *)(&(*llvm_cbe_m)[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'm' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load float* %%5, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_274_count);
  llvm_cbe_tmp__115 = (float )*llvm_cbe_tmp__114;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__115, *(int*)(&llvm_cbe_tmp__115));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds float* %%v, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_275_count);
  llvm_cbe_tmp__116 = (float *)(&llvm_cbe_v[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load float* %%7, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_276_count);
  llvm_cbe_tmp__117 = (float )*llvm_cbe_tmp__116;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__117, *(int*)(&llvm_cbe_tmp__117));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = fmul float %%6, %%8, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_277_count);
  llvm_cbe_tmp__118 = (float )((float )(llvm_cbe_tmp__115 * llvm_cbe_tmp__117));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__118, *(int*)(&llvm_cbe_tmp__118));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fadd float %%4, %%9, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_278_count);
  llvm_cbe_tmp__119 = (float )((float )(llvm_cbe_tmp__113 + llvm_cbe_tmp__118));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__119, *(int*)(&llvm_cbe_tmp__119));
if (AESL_DEBUG_TRACE)
printf("\n  store float %%10, float* %%res, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_279_count);
  *llvm_cbe_res = llvm_cbe_tmp__119;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__119);
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds [2 x float]* %%m, i64 1, i64 0, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_280_count);
  llvm_cbe_tmp__120 = (float *)(&llvm_cbe_m[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )0ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'm' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load float* %%11, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_281_count);
  llvm_cbe_tmp__121 = (float )*llvm_cbe_tmp__120;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__121, *(int*)(&llvm_cbe_tmp__121));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load float* %%v, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_282_count);
  llvm_cbe_tmp__122 = (float )*llvm_cbe_v;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__122, *(int*)(&llvm_cbe_tmp__122));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fmul float %%12, %%13, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_283_count);
  llvm_cbe_tmp__123 = (float )((float )(llvm_cbe_tmp__121 * llvm_cbe_tmp__122));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__123, *(int*)(&llvm_cbe_tmp__123));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds [2 x float]* %%m, i64 1, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_284_count);
  llvm_cbe_tmp__124 = (float *)(&llvm_cbe_m[(((signed long long )1ull))
#ifdef AESL_BC_SIM
 % 2
#endif
][(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )1ull) < 2)) fprintf(stderr, "%s:%d: warning: Read access out of array 'm' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load float* %%15, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_285_count);
  llvm_cbe_tmp__125 = (float )*llvm_cbe_tmp__124;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__125, *(int*)(&llvm_cbe_tmp__125));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load float* %%7, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_286_count);
  llvm_cbe_tmp__126 = (float )*llvm_cbe_tmp__116;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__126, *(int*)(&llvm_cbe_tmp__126));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = fmul float %%16, %%17, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_287_count);
  llvm_cbe_tmp__127 = (float )((float )(llvm_cbe_tmp__125 * llvm_cbe_tmp__126));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__127, *(int*)(&llvm_cbe_tmp__127));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fadd float %%14, %%18, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_288_count);
  llvm_cbe_tmp__128 = (float )((float )(llvm_cbe_tmp__123 + llvm_cbe_tmp__127));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__128, *(int*)(&llvm_cbe_tmp__128));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds float* %%res, i64 1, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_289_count);
  llvm_cbe_tmp__129 = (float *)(&llvm_cbe_res[(((signed long long )1ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%19, float* %%20, align 4, !dbg !27 for 0x%I64xth hint within @aesl_internal_local_multiplyMatrixVector  --> \n", ++aesl_llvm_cbe_290_count);
  *llvm_cbe_tmp__129 = llvm_cbe_tmp__128;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__128);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @aesl_internal_local_multiplyMatrixVector}\n");
  return;
}

