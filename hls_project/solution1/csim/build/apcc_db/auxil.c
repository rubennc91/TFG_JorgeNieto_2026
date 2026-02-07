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
typedef struct l_struct_OC_OSQPSettings l_struct_OC_OSQPSettings;
typedef struct l_struct_OC_OSQPScaling l_struct_OC_OSQPScaling;
typedef struct l_struct_OC_OSQPInfo l_struct_OC_OSQPInfo;

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

struct l_struct_OC_OSQPScaling {
  float field0;
  float *field1;
  float *field2;
  float field3;
  float *field4;
  float *field5;
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


/* External Global Variable Declarations */
extern float work_x[15];
extern l_struct_OC_OSQPData data;
extern float work_z[19];
extern float work_y[19];
extern float work_xz_tilde[34];
extern l_struct_OC_OSQPSettings settings;
extern float work_x_prev[15];
extern float work_delta_x[15];
extern float work_z_prev[19];
extern float work_rho_inv_vec[19];
extern float work_rho_vec[19];
extern float work_delta_y[19];
extern float Pdata_x[12];
extern signed long long Pdata_p[16];
extern signed long long Pdata_i[12];
extern float qdata[15];
extern l_struct_OC_OSQPScaling scaling;
extern float Adata_x[43];
extern signed long long Adata_p[16];
extern signed long long Adata_i[43];
extern float work_Ax[19];
extern float scaling_Einv[19];
extern float work_Px[15];
extern float work_Aty[15];
extern float scaling_Dinv[15];
extern float udata[19];
extern float ldata[19];
extern float scaling_E[19];
extern float work_Adelta_x[19];
extern float work_Atdelta_y[15];
extern float scaling_D[15];
extern float work_Pdelta_x[15];
extern l_struct_OC_OSQPInfo info;
extern float xsolution[15];
extern float ysolution[19];

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void cold_start(void);
void vec_set_scalar(float *, float , signed long long );
void update_xz_tilde(void);
static void aesl_internal_compute_rhs(void);
signed long long solve_linsys_qdldl(float *);
void update_x(void);
void update_z(void);
void project(float *);
void update_y(void);
float compute_obj_val(float *llvm_cbe_x);
float quad_form(float *, signed long long *, signed long long *, signed long long , float *);
float vec_prod(float *, float *, signed long long );
float compute_pri_res(float *llvm_cbe_x, float *llvm_cbe_z);
void mat_vec(float *, signed long long *, signed long long *, signed long long , signed long long , float *, float *, signed long long );
void vec_add_scaled(float *, float *, float *, signed long long , float );
float vec_scaled_norm_inf(float *, float *, signed long long );
float vec_norm_inf(float *, signed long long );
float compute_pri_tol(float llvm_cbe_eps_abs, float llvm_cbe_eps_rel);
float compute_dua_res(float *llvm_cbe_x, float *llvm_cbe_y);
void prea_vec_copy(float *, float *, signed long long );
void mat_tpose_vec(float *, signed long long *, signed long long *, signed long long , signed long long , float *, float *, signed long long , signed long long );
float compute_dua_tol(float llvm_cbe_eps_abs, float llvm_cbe_eps_rel);
signed long long is_primal_infeasible(float llvm_cbe_eps_prim_inf);
void vec_ew_prod(float *, float *, float *, signed long long );
signed long long is_dual_infeasible(float llvm_cbe_eps_dual_inf);
void store_solution(void);
signed long long unscale_solution(void);
void update_info(signed long long llvm_cbe_iter, signed long long llvm_cbe_compute_objective, signed long long llvm_cbe_polish);
void reset_info(l_struct_OC_OSQPInfo *llvm_cbe_info_ptr);
void update_status(l_struct_OC_OSQPInfo *llvm_cbe_info_ptr, signed long long llvm_cbe_status_val);
void c_strcpy( char *,  char *);
signed long long check_termination(signed long long llvm_cbe_approximate);


/* Global Variable Definitions and Initialization */
static  char aesl_internal__OC_str2[18] = "primal infeasible";
static  char aesl_internal__OC_str8[12] = "interrupted";
static  char aesl_internal__OC_str9[19] = "problem non convex";
static  char aesl_internal__OC_str4[9] = "unsolved";
static  char aesl_internal__OC_str7[27] = "maximum iterations reached";
static  char aesl_internal__OC_str3[29] = "primal infeasible inaccurate";
static  char aesl_internal__OC_str[7] = "solved";
static  char aesl_internal__OC_str6[27] = "dual infeasible inaccurate";
static  char aesl_internal__OC_str5[16] = "dual infeasible";
static  char aesl_internal__OC_str1[18] = "solved inaccurate";


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

void cold_start(void) {
  static  unsigned long long aesl_llvm_cbe_1_count = 0;
  unsigned long long llvm_cbe_tmp__1;
  static  unsigned long long aesl_llvm_cbe_2_count = 0;
  static  unsigned long long aesl_llvm_cbe_3_count = 0;
  unsigned long long llvm_cbe_tmp__2;
  static  unsigned long long aesl_llvm_cbe_4_count = 0;
  static  unsigned long long aesl_llvm_cbe_5_count = 0;
  unsigned long long llvm_cbe_tmp__3;
  static  unsigned long long aesl_llvm_cbe_6_count = 0;
  static  unsigned long long aesl_llvm_cbe_7_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @cold_start\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_1_count);
  llvm_cbe_tmp__1 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__1);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0), float 0.000000e+00, i64 %%1) nounwind, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_2_count);
   /*tail*/ vec_set_scalar((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0x0p0, llvm_cbe_tmp__1);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__1);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_3_count);
  llvm_cbe_tmp__2 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__2);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0), float 0.000000e+00, i64 %%2) nounwind, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_4_count);
   /*tail*/ vec_set_scalar((float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x0p0, llvm_cbe_tmp__2);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__2);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_5_count);
  llvm_cbe_tmp__3 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([19 x float]* @work_y, i64 0, i64 0), float 0.000000e+00, i64 %%3) nounwind, !dbg !34 for 0x%I64xth hint within @cold_start  --> \n", ++aesl_llvm_cbe_6_count);
   /*tail*/ vec_set_scalar((float *)((&work_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x0p0, llvm_cbe_tmp__3);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x0p0);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__3);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @cold_start}\n");
  return;
}


void update_xz_tilde(void) {
  static  unsigned long long aesl_llvm_cbe_8_count = 0;
  static  unsigned long long aesl_llvm_cbe_9_count = 0;
  unsigned long long llvm_cbe_tmp__4;
  static  unsigned long long aesl_llvm_cbe_10_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_xz_tilde\n");
if (AESL_DEBUG_TRACE)
printf("\n  tail call fastcc void @aesl_internal_compute_rhs(), !dbg !34 for 0x%I64xth hint within @update_xz_tilde  --> \n", ++aesl_llvm_cbe_8_count);
   /*tail*/ aesl_internal_compute_rhs();
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = tail call i64 @solve_linsys_qdldl(float* getelementptr inbounds ([34 x float]* @work_xz_tilde, i64 0, i64 0)) nounwind, !dbg !34 for 0x%I64xth hint within @update_xz_tilde  --> \n", ++aesl_llvm_cbe_9_count);
   /*tail*/ solve_linsys_qdldl((float *)((&work_xz_tilde[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 34
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__4);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_xz_tilde}\n");
  return;
}


static void aesl_internal_compute_rhs(void) {
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
  unsigned long long llvm_cbe_tmp__5;
  static  unsigned long long aesl_llvm_cbe_24_count = 0;
  static  unsigned long long aesl_llvm_cbe_25_count = 0;
  static  unsigned long long aesl_llvm_cbe_26_count = 0;
  float llvm_cbe_tmp__6;
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
  unsigned long long llvm_cbe_tmp__7;
  static  unsigned long long aesl_llvm_cbe_40_count = 0;
  static  unsigned long long aesl_llvm_cbe_41_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge3_count = 0;
  unsigned long long llvm_cbe_storemerge3;
  unsigned long long llvm_cbe_storemerge3__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_42_count = 0;
  float *llvm_cbe_tmp__8;
  static  unsigned long long aesl_llvm_cbe_43_count = 0;
  float llvm_cbe_tmp__9;
  static  unsigned long long aesl_llvm_cbe_44_count = 0;
  float llvm_cbe_tmp__10;
  static  unsigned long long aesl_llvm_cbe_45_count = 0;
  float *llvm_cbe_tmp__11;
  static  unsigned long long aesl_llvm_cbe_46_count = 0;
  float llvm_cbe_tmp__12;
  static  unsigned long long aesl_llvm_cbe_47_count = 0;
  float llvm_cbe_tmp__13;
  static  unsigned long long aesl_llvm_cbe_48_count = 0;
  float *llvm_cbe_tmp__14;
  static  unsigned long long aesl_llvm_cbe_49_count = 0;
  static  unsigned long long aesl_llvm_cbe_50_count = 0;
  unsigned long long llvm_cbe_tmp__15;
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
  static  unsigned long long aesl_llvm_cbe_storemerge12_count = 0;
  unsigned long long llvm_cbe_storemerge12;
  unsigned long long llvm_cbe_storemerge12__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_65_count = 0;
  float *llvm_cbe_tmp__16;
  static  unsigned long long aesl_llvm_cbe_66_count = 0;
  float llvm_cbe_tmp__17;
  static  unsigned long long aesl_llvm_cbe_67_count = 0;
  float *llvm_cbe_tmp__18;
  static  unsigned long long aesl_llvm_cbe_68_count = 0;
  float llvm_cbe_tmp__19;
  static  unsigned long long aesl_llvm_cbe_69_count = 0;
  float *llvm_cbe_tmp__20;
  static  unsigned long long aesl_llvm_cbe_70_count = 0;
  float llvm_cbe_tmp__21;
  static  unsigned long long aesl_llvm_cbe_71_count = 0;
  float llvm_cbe_tmp__22;
  static  unsigned long long aesl_llvm_cbe_72_count = 0;
  float llvm_cbe_tmp__23;
  static  unsigned long long aesl_llvm_cbe_73_count = 0;
  unsigned long long llvm_cbe_tmp__24;
  static  unsigned long long aesl_llvm_cbe_74_count = 0;
  float *llvm_cbe_tmp__25;
  static  unsigned long long aesl_llvm_cbe_75_count = 0;
  static  unsigned long long aesl_llvm_cbe_76_count = 0;
  unsigned long long llvm_cbe_tmp__26;
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
  static  unsigned long long aesl_llvm_cbe_88_count = 0;
  static  unsigned long long aesl_llvm_cbe_89_count = 0;
  static  unsigned long long aesl_llvm_cbe_90_count = 0;
  static  unsigned long long aesl_llvm_cbe_91_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @aesl_internal_compute_rhs\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_23_count);
  llvm_cbe_tmp__5 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__5);
  if ((((signed long long )llvm_cbe_tmp__5) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph5;
  } else {
    goto llvm_cbe__2e_preheader;
  }

llvm_cbe__2e_lr_2e_ph5:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 1), align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_26_count);
  llvm_cbe_tmp__6 = (float )*((&settings.field1));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__6, *(int*)(&llvm_cbe_tmp__6));
  llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__27;

llvm_cbe__2e_preheader:
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_39_count);
  llvm_cbe_tmp__7 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__7);
  if ((((signed long long )llvm_cbe_tmp__7) > ((signed long long )0ull))) {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__27:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge3 = phi i64 [ 0, %%.lr.ph5 ], [ %%14, %%6  for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_storemerge3_count);
  llvm_cbe_storemerge3 = (unsigned long long )llvm_cbe_storemerge3__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",llvm_cbe_storemerge3);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__15);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [15 x float]* @work_x_prev, i64 0, i64 %%storemerge3, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_42_count);
  llvm_cbe_tmp__8 = (float *)(&work_x_prev[(((signed long long )llvm_cbe_storemerge3))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge3) < 15)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_x_prev' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load float* %%7, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_43_count);
  llvm_cbe_tmp__9 = (float )*llvm_cbe_tmp__8;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__9, *(int*)(&llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = fmul float %%3, %%8, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_44_count);
  llvm_cbe_tmp__10 = (float )((float )(llvm_cbe_tmp__6 * llvm_cbe_tmp__9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__10, *(int*)(&llvm_cbe_tmp__10));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds [15 x float]* @qdata, i64 0, i64 %%storemerge3, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_45_count);
  llvm_cbe_tmp__11 = (float *)(&qdata[(((signed long long )llvm_cbe_storemerge3))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge3) < 15)) fprintf(stderr, "%s:%d: warning: Read access out of array 'qdata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load float* %%10, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_46_count);
  llvm_cbe_tmp__12 = (float )*llvm_cbe_tmp__11;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__12, *(int*)(&llvm_cbe_tmp__12));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = fsub float %%9, %%11, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_47_count);
  llvm_cbe_tmp__13 = (float )((float )(llvm_cbe_tmp__10 - llvm_cbe_tmp__12));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__13, *(int*)(&llvm_cbe_tmp__13));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [34 x float]* @work_xz_tilde, i64 0, i64 %%storemerge3, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_48_count);
  llvm_cbe_tmp__14 = (float *)(&work_xz_tilde[(((signed long long )llvm_cbe_storemerge3))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",((signed long long )llvm_cbe_storemerge3));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge3) < 34 && "Write access out of array 'work_xz_tilde' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%12, float* %%13, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_49_count);
  *llvm_cbe_tmp__14 = llvm_cbe_tmp__13;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__13);
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = add nsw i64 %%storemerge3, 1, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_50_count);
  llvm_cbe_tmp__15 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge3&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__15&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__15) < ((signed long long )llvm_cbe_tmp__5))) {
    llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__15;   /* for PHI node */
    goto llvm_cbe_tmp__27;
  } else {
    goto llvm_cbe__2e_preheader;
  }

  } while (1); /* end of syntactic loop '' */
  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge12 = phi i64 [ %%26, %%.lr.ph ], [ 0, %%.preheader  for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_storemerge12_count);
  llvm_cbe_storemerge12 = (unsigned long long )llvm_cbe_storemerge12__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",llvm_cbe_storemerge12);
printf("\n = 0x%I64X",llvm_cbe_tmp__26);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds [19 x float]* @work_z_prev, i64 0, i64 %%storemerge12, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_65_count);
  llvm_cbe_tmp__16 = (float *)(&work_z_prev[(((signed long long )llvm_cbe_storemerge12))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge12) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_z_prev' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = load float* %%16, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_66_count);
  llvm_cbe_tmp__17 = (float )*llvm_cbe_tmp__16;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__17, *(int*)(&llvm_cbe_tmp__17));
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = getelementptr inbounds [19 x float]* @work_rho_inv_vec, i64 0, i64 %%storemerge12, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_67_count);
  llvm_cbe_tmp__18 = (float *)(&work_rho_inv_vec[(((signed long long )llvm_cbe_storemerge12))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge12) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_rho_inv_vec' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load float* %%18, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_68_count);
  llvm_cbe_tmp__19 = (float )*llvm_cbe_tmp__18;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__19, *(int*)(&llvm_cbe_tmp__19));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds [19 x float]* @work_y, i64 0, i64 %%storemerge12, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_69_count);
  llvm_cbe_tmp__20 = (float *)(&work_y[(((signed long long )llvm_cbe_storemerge12))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge12 = 0x%I64X",((signed long long )llvm_cbe_storemerge12));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge12) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load float* %%20, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_70_count);
  llvm_cbe_tmp__21 = (float )*llvm_cbe_tmp__20;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__21, *(int*)(&llvm_cbe_tmp__21));
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fmul float %%19, %%21, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_71_count);
  llvm_cbe_tmp__22 = (float )((float )(llvm_cbe_tmp__19 * llvm_cbe_tmp__21));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__22, *(int*)(&llvm_cbe_tmp__22));
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = fsub float %%17, %%22, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_72_count);
  llvm_cbe_tmp__23 = (float )((float )(llvm_cbe_tmp__17 - llvm_cbe_tmp__22));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__23, *(int*)(&llvm_cbe_tmp__23));
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = add nsw i64 %%1, %%storemerge12, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_73_count);
  llvm_cbe_tmp__24 = (unsigned long long )((unsigned long long )(llvm_cbe_tmp__5&18446744073709551615ull)) + ((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__24&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = getelementptr inbounds [34 x float]* @work_xz_tilde, i64 0, i64 %%24, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_74_count);
  llvm_cbe_tmp__25 = (float *)(&work_xz_tilde[(((signed long long )llvm_cbe_tmp__24))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__24));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_tmp__24) < 34 && "Write access out of array 'work_xz_tilde' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%23, float* %%25, align 4, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_75_count);
  *llvm_cbe_tmp__25 = llvm_cbe_tmp__23;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__23);
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = add nsw i64 %%storemerge12, 1, !dbg !35 for 0x%I64xth hint within @aesl_internal_compute_rhs  --> \n", ++aesl_llvm_cbe_76_count);
  llvm_cbe_tmp__26 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge12&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__26&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__26) < ((signed long long )llvm_cbe_tmp__7))) {
    llvm_cbe_storemerge12__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__26;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @aesl_internal_compute_rhs}\n");
  return;
}


void update_x(void) {
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
  unsigned long long llvm_cbe_tmp__28;
  static  unsigned long long aesl_llvm_cbe_102_count = 0;
  static  unsigned long long aesl_llvm_cbe_103_count = 0;
  static  unsigned long long aesl_llvm_cbe_104_count = 0;
  float llvm_cbe_tmp__29;
  static  unsigned long long aesl_llvm_cbe_105_count = 0;
  float llvm_cbe_tmp__30;
  static  unsigned long long aesl_llvm_cbe_106_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_107_count = 0;
  float *llvm_cbe_tmp__31;
  static  unsigned long long aesl_llvm_cbe_108_count = 0;
  float llvm_cbe_tmp__32;
  static  unsigned long long aesl_llvm_cbe_109_count = 0;
  float llvm_cbe_tmp__33;
  static  unsigned long long aesl_llvm_cbe_110_count = 0;
  float *llvm_cbe_tmp__34;
  static  unsigned long long aesl_llvm_cbe_111_count = 0;
  float llvm_cbe_tmp__35;
  static  unsigned long long aesl_llvm_cbe_112_count = 0;
  float llvm_cbe_tmp__36;
  static  unsigned long long aesl_llvm_cbe_113_count = 0;
  float llvm_cbe_tmp__37;
  static  unsigned long long aesl_llvm_cbe_114_count = 0;
  float *llvm_cbe_tmp__38;
  static  unsigned long long aesl_llvm_cbe_115_count = 0;
  static  unsigned long long aesl_llvm_cbe_116_count = 0;
  float llvm_cbe_tmp__39;
  static  unsigned long long aesl_llvm_cbe_117_count = 0;
  float *llvm_cbe_tmp__40;
  static  unsigned long long aesl_llvm_cbe_118_count = 0;
  static  unsigned long long aesl_llvm_cbe_119_count = 0;
  unsigned long long llvm_cbe_tmp__41;
  static  unsigned long long aesl_llvm_cbe_120_count = 0;
  static  unsigned long long aesl_llvm_cbe_121_count = 0;
  static  unsigned long long aesl_llvm_cbe_122_count = 0;
  static  unsigned long long aesl_llvm_cbe_123_count = 0;
  static  unsigned long long aesl_llvm_cbe_124_count = 0;
  static  unsigned long long aesl_llvm_cbe_125_count = 0;
  static  unsigned long long aesl_llvm_cbe_126_count = 0;
  static  unsigned long long aesl_llvm_cbe_127_count = 0;
  static  unsigned long long aesl_llvm_cbe_128_count = 0;
  static  unsigned long long aesl_llvm_cbe_129_count = 0;
  static  unsigned long long aesl_llvm_cbe_130_count = 0;
  static  unsigned long long aesl_llvm_cbe_131_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_x\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_101_count);
  llvm_cbe_tmp__28 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__28);
  if ((((signed long long )llvm_cbe_tmp__28) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 11), align 8, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_104_count);
  llvm_cbe_tmp__29 = (float )*((&settings.field11));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__29, *(int*)(&llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = fsub float 1.000000e+00, %%3, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_105_count);
  llvm_cbe_tmp__30 = (float )((float )(0x1p0 - llvm_cbe_tmp__29));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__30, *(int*)(&llvm_cbe_tmp__30));
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__42;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__42:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ 0, %%.lr.ph ], [ %%16, %%5  for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__41);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = getelementptr inbounds [34 x float]* @work_xz_tilde, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_107_count);
  llvm_cbe_tmp__31 = (float *)(&work_xz_tilde[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 34)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_xz_tilde' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load float* %%6, align 4, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_108_count);
  llvm_cbe_tmp__32 = (float )*llvm_cbe_tmp__31;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__32, *(int*)(&llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = fmul float %%3, %%7, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_109_count);
  llvm_cbe_tmp__33 = (float )((float )(llvm_cbe_tmp__29 * llvm_cbe_tmp__32));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__33, *(int*)(&llvm_cbe_tmp__33));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = getelementptr inbounds [15 x float]* @work_x_prev, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_110_count);
  llvm_cbe_tmp__34 = (float *)(&work_x_prev[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 15)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_x_prev' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load float* %%9, align 4, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_111_count);
  llvm_cbe_tmp__35 = (float )*llvm_cbe_tmp__34;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__35, *(int*)(&llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = fmul float %%4, %%10, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_112_count);
  llvm_cbe_tmp__36 = (float )((float )(llvm_cbe_tmp__30 * llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__36, *(int*)(&llvm_cbe_tmp__36));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = fadd float %%8, %%11, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_113_count);
  llvm_cbe_tmp__37 = (float )((float )(llvm_cbe_tmp__33 + llvm_cbe_tmp__36));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__37, *(int*)(&llvm_cbe_tmp__37));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [15 x float]* @work_x, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_114_count);
  llvm_cbe_tmp__38 = (float *)(&work_x[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 15 && "Write access out of array 'work_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%12, float* %%13, align 4, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_115_count);
  *llvm_cbe_tmp__38 = llvm_cbe_tmp__37;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__37);
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fsub float %%12, %%10, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_116_count);
  llvm_cbe_tmp__39 = (float )((float )(llvm_cbe_tmp__37 - llvm_cbe_tmp__35));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__39, *(int*)(&llvm_cbe_tmp__39));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds [15 x float]* @work_delta_x, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_117_count);
  llvm_cbe_tmp__40 = (float *)(&work_delta_x[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 15
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 15 && "Write access out of array 'work_delta_x' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%14, float* %%15, align 4, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_118_count);
  *llvm_cbe_tmp__40 = llvm_cbe_tmp__39;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__39);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = add nsw i64 %%storemerge1, 1, !dbg !35 for 0x%I64xth hint within @update_x  --> \n", ++aesl_llvm_cbe_119_count);
  llvm_cbe_tmp__41 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__41&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__41) < ((signed long long )llvm_cbe_tmp__28))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__41;   /* for PHI node */
    goto llvm_cbe_tmp__42;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_x}\n");
  return;
}


void update_z(void) {
  static  unsigned long long aesl_llvm_cbe_132_count = 0;
  static  unsigned long long aesl_llvm_cbe_133_count = 0;
  static  unsigned long long aesl_llvm_cbe_134_count = 0;
  static  unsigned long long aesl_llvm_cbe_135_count = 0;
  static  unsigned long long aesl_llvm_cbe_136_count = 0;
  static  unsigned long long aesl_llvm_cbe_137_count = 0;
  static  unsigned long long aesl_llvm_cbe_138_count = 0;
  static  unsigned long long aesl_llvm_cbe_139_count = 0;
  static  unsigned long long aesl_llvm_cbe_140_count = 0;
  unsigned long long llvm_cbe_tmp__43;
  static  unsigned long long aesl_llvm_cbe_141_count = 0;
  static  unsigned long long aesl_llvm_cbe_142_count = 0;
  static  unsigned long long aesl_llvm_cbe_143_count = 0;
  float llvm_cbe_tmp__44;
  static  unsigned long long aesl_llvm_cbe_144_count = 0;
  unsigned long long llvm_cbe_tmp__45;
  static  unsigned long long aesl_llvm_cbe_145_count = 0;
  float llvm_cbe_tmp__46;
  static  unsigned long long aesl_llvm_cbe_146_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_147_count = 0;
  unsigned long long llvm_cbe_tmp__47;
  static  unsigned long long aesl_llvm_cbe_148_count = 0;
  float *llvm_cbe_tmp__48;
  static  unsigned long long aesl_llvm_cbe_149_count = 0;
  float llvm_cbe_tmp__49;
  static  unsigned long long aesl_llvm_cbe_150_count = 0;
  float llvm_cbe_tmp__50;
  static  unsigned long long aesl_llvm_cbe_151_count = 0;
  float *llvm_cbe_tmp__51;
  static  unsigned long long aesl_llvm_cbe_152_count = 0;
  float llvm_cbe_tmp__52;
  static  unsigned long long aesl_llvm_cbe_153_count = 0;
  float llvm_cbe_tmp__53;
  static  unsigned long long aesl_llvm_cbe_154_count = 0;
  float llvm_cbe_tmp__54;
  static  unsigned long long aesl_llvm_cbe_155_count = 0;
  float *llvm_cbe_tmp__55;
  static  unsigned long long aesl_llvm_cbe_156_count = 0;
  float llvm_cbe_tmp__56;
  static  unsigned long long aesl_llvm_cbe_157_count = 0;
  float *llvm_cbe_tmp__57;
  static  unsigned long long aesl_llvm_cbe_158_count = 0;
  float llvm_cbe_tmp__58;
  static  unsigned long long aesl_llvm_cbe_159_count = 0;
  float llvm_cbe_tmp__59;
  static  unsigned long long aesl_llvm_cbe_160_count = 0;
  float llvm_cbe_tmp__60;
  static  unsigned long long aesl_llvm_cbe_161_count = 0;
  float *llvm_cbe_tmp__61;
  static  unsigned long long aesl_llvm_cbe_162_count = 0;
  static  unsigned long long aesl_llvm_cbe_163_count = 0;
  unsigned long long llvm_cbe_tmp__62;
  static  unsigned long long aesl_llvm_cbe_164_count = 0;
  static  unsigned long long aesl_llvm_cbe_165_count = 0;
  static  unsigned long long aesl_llvm_cbe_166_count = 0;
  static  unsigned long long aesl_llvm_cbe_167_count = 0;
  static  unsigned long long aesl_llvm_cbe_168_count = 0;
  static  unsigned long long aesl_llvm_cbe_169_count = 0;
  static  unsigned long long aesl_llvm_cbe_170_count = 0;
  static  unsigned long long aesl_llvm_cbe_171_count = 0;
  static  unsigned long long aesl_llvm_cbe_172_count = 0;
  static  unsigned long long aesl_llvm_cbe_173_count = 0;
  static  unsigned long long aesl_llvm_cbe_174_count = 0;
  static  unsigned long long aesl_llvm_cbe_175_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_z\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_140_count);
  llvm_cbe_tmp__43 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__43);
  if ((((signed long long )llvm_cbe_tmp__43) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 11), align 8, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_143_count);
  llvm_cbe_tmp__44 = (float )*((&settings.field11));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__44, *(int*)(&llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_144_count);
  llvm_cbe_tmp__45 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__45);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fsub float 1.000000e+00, %%3, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_145_count);
  llvm_cbe_tmp__46 = (float )((float )(0x1p0 - llvm_cbe_tmp__44));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__46, *(int*)(&llvm_cbe_tmp__46));
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__63;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__63:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ 0, %%.lr.ph ], [ %%22, %%6  for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__62);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = add nsw i64 %%4, %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_147_count);
  llvm_cbe_tmp__47 = (unsigned long long )((unsigned long long )(llvm_cbe_tmp__45&18446744073709551615ull)) + ((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__47&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds [34 x float]* @work_xz_tilde, i64 0, i64 %%7, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_148_count);
  llvm_cbe_tmp__48 = (float *)(&work_xz_tilde[(((signed long long )llvm_cbe_tmp__47))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__47));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_tmp__47) < 34)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_xz_tilde' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* %%8, align 4, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_149_count);
  llvm_cbe_tmp__49 = (float )*llvm_cbe_tmp__48;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__49, *(int*)(&llvm_cbe_tmp__49));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fmul float %%3, %%9, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_150_count);
  llvm_cbe_tmp__50 = (float )((float )(llvm_cbe_tmp__44 * llvm_cbe_tmp__49));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__50, *(int*)(&llvm_cbe_tmp__50));
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = getelementptr inbounds [19 x float]* @work_z_prev, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_151_count);
  llvm_cbe_tmp__51 = (float *)(&work_z_prev[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_z_prev' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load float* %%11, align 4, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_152_count);
  llvm_cbe_tmp__52 = (float )*llvm_cbe_tmp__51;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__52, *(int*)(&llvm_cbe_tmp__52));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%5, %%12, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_153_count);
  llvm_cbe_tmp__53 = (float )((float )(llvm_cbe_tmp__46 * llvm_cbe_tmp__52));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__53, *(int*)(&llvm_cbe_tmp__53));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fadd float %%10, %%13, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_154_count);
  llvm_cbe_tmp__54 = (float )((float )(llvm_cbe_tmp__50 + llvm_cbe_tmp__53));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__54, *(int*)(&llvm_cbe_tmp__54));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds [19 x float]* @work_rho_inv_vec, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_155_count);
  llvm_cbe_tmp__55 = (float *)(&work_rho_inv_vec[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_rho_inv_vec' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load float* %%15, align 4, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_156_count);
  llvm_cbe_tmp__56 = (float )*llvm_cbe_tmp__55;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__56, *(int*)(&llvm_cbe_tmp__56));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = getelementptr inbounds [19 x float]* @work_y, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_157_count);
  llvm_cbe_tmp__57 = (float *)(&work_y[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = load float* %%17, align 4, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_158_count);
  llvm_cbe_tmp__58 = (float )*llvm_cbe_tmp__57;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__58, *(int*)(&llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fmul float %%16, %%18, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_159_count);
  llvm_cbe_tmp__59 = (float )((float )(llvm_cbe_tmp__56 * llvm_cbe_tmp__58));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__59, *(int*)(&llvm_cbe_tmp__59));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = fadd float %%14, %%19, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_160_count);
  llvm_cbe_tmp__60 = (float )((float )(llvm_cbe_tmp__54 + llvm_cbe_tmp__59));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__60, *(int*)(&llvm_cbe_tmp__60));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds [19 x float]* @work_z, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_161_count);
  llvm_cbe_tmp__61 = (float *)(&work_z[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 19 && "Write access out of array 'work_z' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%20, float* %%21, align 4, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_162_count);
  *llvm_cbe_tmp__61 = llvm_cbe_tmp__60;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__60);
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = add nsw i64 %%storemerge1, 1, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_163_count);
  llvm_cbe_tmp__62 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__62&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__62) < ((signed long long )llvm_cbe_tmp__43))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__62;   /* for PHI node */
    goto llvm_cbe_tmp__63;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @project(float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_z  --> \n", ++aesl_llvm_cbe_174_count);
   /*tail*/ project((float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_z}\n");
  return;
}


void update_y(void) {
  static  unsigned long long aesl_llvm_cbe_176_count = 0;
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
  unsigned long long llvm_cbe_tmp__64;
  static  unsigned long long aesl_llvm_cbe_187_count = 0;
  static  unsigned long long aesl_llvm_cbe_188_count = 0;
  static  unsigned long long aesl_llvm_cbe_189_count = 0;
  float llvm_cbe_tmp__65;
  static  unsigned long long aesl_llvm_cbe_190_count = 0;
  unsigned long long llvm_cbe_tmp__66;
  static  unsigned long long aesl_llvm_cbe_191_count = 0;
  float llvm_cbe_tmp__67;
  static  unsigned long long aesl_llvm_cbe_192_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_193_count = 0;
  float *llvm_cbe_tmp__68;
  static  unsigned long long aesl_llvm_cbe_194_count = 0;
  float llvm_cbe_tmp__69;
  static  unsigned long long aesl_llvm_cbe_195_count = 0;
  unsigned long long llvm_cbe_tmp__70;
  static  unsigned long long aesl_llvm_cbe_196_count = 0;
  float *llvm_cbe_tmp__71;
  static  unsigned long long aesl_llvm_cbe_197_count = 0;
  float llvm_cbe_tmp__72;
  static  unsigned long long aesl_llvm_cbe_198_count = 0;
  float llvm_cbe_tmp__73;
  static  unsigned long long aesl_llvm_cbe_199_count = 0;
  float *llvm_cbe_tmp__74;
  static  unsigned long long aesl_llvm_cbe_200_count = 0;
  float llvm_cbe_tmp__75;
  static  unsigned long long aesl_llvm_cbe_201_count = 0;
  float llvm_cbe_tmp__76;
  static  unsigned long long aesl_llvm_cbe_202_count = 0;
  float llvm_cbe_tmp__77;
  static  unsigned long long aesl_llvm_cbe_203_count = 0;
  float *llvm_cbe_tmp__78;
  static  unsigned long long aesl_llvm_cbe_204_count = 0;
  float llvm_cbe_tmp__79;
  static  unsigned long long aesl_llvm_cbe_205_count = 0;
  float llvm_cbe_tmp__80;
  static  unsigned long long aesl_llvm_cbe_206_count = 0;
  float llvm_cbe_tmp__81;
  static  unsigned long long aesl_llvm_cbe_207_count = 0;
  float *llvm_cbe_tmp__82;
  static  unsigned long long aesl_llvm_cbe_208_count = 0;
  static  unsigned long long aesl_llvm_cbe_209_count = 0;
  float *llvm_cbe_tmp__83;
  static  unsigned long long aesl_llvm_cbe_210_count = 0;
  float llvm_cbe_tmp__84;
  static  unsigned long long aesl_llvm_cbe_211_count = 0;
  float llvm_cbe_tmp__85;
  static  unsigned long long aesl_llvm_cbe_212_count = 0;
  static  unsigned long long aesl_llvm_cbe_213_count = 0;
  unsigned long long llvm_cbe_tmp__86;
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

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_y\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_186_count);
  llvm_cbe_tmp__64 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__64);
  if ((((signed long long )llvm_cbe_tmp__64) > ((signed long long )0ull))) {
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 11), align 8, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_189_count);
  llvm_cbe_tmp__65 = (float )*((&settings.field11));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__65, *(int*)(&llvm_cbe_tmp__65));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_190_count);
  llvm_cbe_tmp__66 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__66);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fsub float 1.000000e+00, %%3, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_191_count);
  llvm_cbe_tmp__67 = (float )((float )(0x1p0 - llvm_cbe_tmp__65));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__67, *(int*)(&llvm_cbe_tmp__67));
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__87;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__87:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ 0, %%.lr.ph ], [ %%25, %%6  for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__86);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = getelementptr inbounds [19 x float]* @work_rho_vec, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_193_count);
  llvm_cbe_tmp__68 = (float *)(&work_rho_vec[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_rho_vec' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load float* %%7, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_194_count);
  llvm_cbe_tmp__69 = (float )*llvm_cbe_tmp__68;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__69, *(int*)(&llvm_cbe_tmp__69));
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = add nsw i64 %%4, %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_195_count);
  llvm_cbe_tmp__70 = (unsigned long long )((unsigned long long )(llvm_cbe_tmp__66&18446744073709551615ull)) + ((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__70&18446744073709551615ull)));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = getelementptr inbounds [34 x float]* @work_xz_tilde, i64 0, i64 %%9, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_196_count);
  llvm_cbe_tmp__71 = (float *)(&work_xz_tilde[(((signed long long )llvm_cbe_tmp__70))
#ifdef AESL_BC_SIM
 % 34
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",((signed long long )llvm_cbe_tmp__70));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_tmp__70) < 34)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_xz_tilde' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load float* %%10, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_197_count);
  llvm_cbe_tmp__72 = (float )*llvm_cbe_tmp__71;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__72, *(int*)(&llvm_cbe_tmp__72));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = fmul float %%3, %%11, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_198_count);
  llvm_cbe_tmp__73 = (float )((float )(llvm_cbe_tmp__65 * llvm_cbe_tmp__72));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__73, *(int*)(&llvm_cbe_tmp__73));
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [19 x float]* @work_z_prev, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_199_count);
  llvm_cbe_tmp__74 = (float *)(&work_z_prev[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_z_prev' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = load float* %%13, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_200_count);
  llvm_cbe_tmp__75 = (float )*llvm_cbe_tmp__74;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__75, *(int*)(&llvm_cbe_tmp__75));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = fmul float %%5, %%14, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_201_count);
  llvm_cbe_tmp__76 = (float )((float )(llvm_cbe_tmp__67 * llvm_cbe_tmp__75));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__76, *(int*)(&llvm_cbe_tmp__76));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = fadd float %%12, %%15, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_202_count);
  llvm_cbe_tmp__77 = (float )((float )(llvm_cbe_tmp__73 + llvm_cbe_tmp__76));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__77, *(int*)(&llvm_cbe_tmp__77));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = getelementptr inbounds [19 x float]* @work_z, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_203_count);
  llvm_cbe_tmp__78 = (float *)(&work_z[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_z' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = load float* %%17, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_204_count);
  llvm_cbe_tmp__79 = (float )*llvm_cbe_tmp__78;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__79, *(int*)(&llvm_cbe_tmp__79));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fsub float %%16, %%18, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_205_count);
  llvm_cbe_tmp__80 = (float )((float )(llvm_cbe_tmp__77 - llvm_cbe_tmp__79));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__80, *(int*)(&llvm_cbe_tmp__80));
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = fmul float %%8, %%19, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_206_count);
  llvm_cbe_tmp__81 = (float )((float )(llvm_cbe_tmp__69 * llvm_cbe_tmp__80));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__81, *(int*)(&llvm_cbe_tmp__81));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = getelementptr inbounds [19 x float]* @work_delta_y, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_207_count);
  llvm_cbe_tmp__82 = (float *)(&work_delta_y[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 19 && "Write access out of array 'work_delta_y' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%20, float* %%21, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_208_count);
  *llvm_cbe_tmp__82 = llvm_cbe_tmp__81;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__81);
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = getelementptr inbounds [19 x float]* @work_y, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_209_count);
  llvm_cbe_tmp__83 = (float *)(&work_y[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load float* %%22, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_210_count);
  llvm_cbe_tmp__84 = (float )*llvm_cbe_tmp__83;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__84, *(int*)(&llvm_cbe_tmp__84));
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = fadd float %%23, %%20, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_211_count);
  llvm_cbe_tmp__85 = (float )((float )(llvm_cbe_tmp__84 + llvm_cbe_tmp__81));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__85, *(int*)(&llvm_cbe_tmp__85));

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge1) < 19 && "Write access out of array 'work_y' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%24, float* %%22, align 4, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_212_count);
  *llvm_cbe_tmp__83 = llvm_cbe_tmp__85;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__85);
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = add nsw i64 %%storemerge1, 1, !dbg !35 for 0x%I64xth hint within @update_y  --> \n", ++aesl_llvm_cbe_213_count);
  llvm_cbe_tmp__86 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__86&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__86) < ((signed long long )llvm_cbe_tmp__64))) {
    llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__86;   /* for PHI node */
    goto llvm_cbe_tmp__87;
  } else {
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe__2e__crit_edge:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_y}\n");
  return;
}


float compute_obj_val(float *llvm_cbe_x) {
  static  unsigned long long aesl_llvm_cbe_227_count = 0;
  static  unsigned long long aesl_llvm_cbe_228_count = 0;
  static  unsigned long long aesl_llvm_cbe_229_count = 0;
  static  unsigned long long aesl_llvm_cbe_230_count = 0;
  unsigned long long llvm_cbe_tmp__88;
  static  unsigned long long aesl_llvm_cbe_231_count = 0;
  float llvm_cbe_tmp__89;
  static  unsigned long long aesl_llvm_cbe_232_count = 0;
  unsigned long long llvm_cbe_tmp__90;
  static  unsigned long long aesl_llvm_cbe_233_count = 0;
  float llvm_cbe_tmp__91;
  static  unsigned long long aesl_llvm_cbe_234_count = 0;
  float llvm_cbe_tmp__92;
  static  unsigned long long aesl_llvm_cbe_235_count = 0;
  static  unsigned long long aesl_llvm_cbe_236_count = 0;
  static  unsigned long long aesl_llvm_cbe_237_count = 0;
  static  unsigned long long aesl_llvm_cbe_238_count = 0;
  unsigned long long llvm_cbe_tmp__93;
  static  unsigned long long aesl_llvm_cbe_239_count = 0;
  static  unsigned long long aesl_llvm_cbe_240_count = 0;
  static  unsigned long long aesl_llvm_cbe_241_count = 0;
  float llvm_cbe_tmp__94;
  static  unsigned long long aesl_llvm_cbe_242_count = 0;
  float llvm_cbe_tmp__95;
  static  unsigned long long aesl_llvm_cbe_243_count = 0;
  static  unsigned long long aesl_llvm_cbe_244_count = 0;
  static  unsigned long long aesl_llvm_cbe_245_count = 0;
  static  unsigned long long aesl_llvm_cbe_246_count = 0;
  static  unsigned long long aesl_llvm_cbe_247_count = 0;
  float llvm_cbe_tmp__96;
  float llvm_cbe_tmp__96__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_248_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @compute_obj_val\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_230_count);
  llvm_cbe_tmp__88 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__88);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = tail call float @quad_form(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%1, float* %%x) nounwind, !dbg !34 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_231_count);
  llvm_cbe_tmp__89 = (float ) /*tail*/ quad_form((float *)((&Pdata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__88, (float *)llvm_cbe_x);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__88);
printf("\nReturn  = %f",llvm_cbe_tmp__89);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_232_count);
  llvm_cbe_tmp__90 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__90);
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = tail call float @vec_prod(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* %%x, i64 %%3) nounwind, !dbg !34 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_233_count);
  llvm_cbe_tmp__91 = (float ) /*tail*/ vec_prod((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)llvm_cbe_x, llvm_cbe_tmp__90);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__90);
printf("\nReturn  = %f",llvm_cbe_tmp__91);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fadd float %%2, %%4, !dbg !34 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_234_count);
  llvm_cbe_tmp__92 = (float )((float )(llvm_cbe_tmp__89 + llvm_cbe_tmp__91));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__92, *(int*)(&llvm_cbe_tmp__92));
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_238_count);
  llvm_cbe_tmp__93 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__93);
  if (((llvm_cbe_tmp__93&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_tmp__96__PHI_TEMPORARY = (float )llvm_cbe_tmp__92;   /* for PHI node */
    goto llvm_cbe_tmp__97;
  } else {
    goto llvm_cbe_tmp__98;
  }

llvm_cbe_tmp__98:
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !35 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_241_count);
  llvm_cbe_tmp__94 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__94, *(int*)(&llvm_cbe_tmp__94));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fmul float %%5, %%9, !dbg !35 for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_242_count);
  llvm_cbe_tmp__95 = (float )((float )(llvm_cbe_tmp__92 * llvm_cbe_tmp__94));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__95, *(int*)(&llvm_cbe_tmp__95));
  llvm_cbe_tmp__96__PHI_TEMPORARY = (float )llvm_cbe_tmp__95;   /* for PHI node */
  goto llvm_cbe_tmp__97;

llvm_cbe_tmp__97:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = phi float [ %%5, %%0 ], [ %%10, %%8  for 0x%I64xth hint within @compute_obj_val  --> \n", ++aesl_llvm_cbe_247_count);
  llvm_cbe_tmp__96 = (float )llvm_cbe_tmp__96__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__96);
printf("\n = %f",llvm_cbe_tmp__92);
printf("\n = %f",llvm_cbe_tmp__95);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @compute_obj_val}\n");
  return llvm_cbe_tmp__96;
}


float compute_pri_res(float *llvm_cbe_x, float *llvm_cbe_z) {
  static  unsigned long long aesl_llvm_cbe_249_count = 0;
  static  unsigned long long aesl_llvm_cbe_250_count = 0;
  static  unsigned long long aesl_llvm_cbe_251_count = 0;
  static  unsigned long long aesl_llvm_cbe_252_count = 0;
  unsigned long long llvm_cbe_tmp__99;
  static  unsigned long long aesl_llvm_cbe_253_count = 0;
  unsigned long long llvm_cbe_tmp__100;
  static  unsigned long long aesl_llvm_cbe_254_count = 0;
  static  unsigned long long aesl_llvm_cbe_255_count = 0;
  static  unsigned long long aesl_llvm_cbe_256_count = 0;
  unsigned long long llvm_cbe_tmp__101;
  static  unsigned long long aesl_llvm_cbe_257_count = 0;
  static  unsigned long long aesl_llvm_cbe_258_count = 0;
  unsigned long long llvm_cbe_tmp__102;
  static  unsigned long long aesl_llvm_cbe_259_count = 0;
  static  unsigned long long aesl_llvm_cbe_260_count = 0;
  unsigned long long llvm_cbe_tmp__103;
  static  unsigned long long aesl_llvm_cbe_261_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_262_count = 0;
  static  unsigned long long aesl_llvm_cbe_263_count = 0;
  unsigned long long llvm_cbe_tmp__104;
  static  unsigned long long aesl_llvm_cbe_264_count = 0;
  float llvm_cbe_tmp__105;
  static  unsigned long long aesl_llvm_cbe_265_count = 0;
  static  unsigned long long aesl_llvm_cbe_266_count = 0;
  unsigned long long llvm_cbe_tmp__106;
  static  unsigned long long aesl_llvm_cbe_267_count = 0;
  float llvm_cbe_tmp__107;
  static  unsigned long long aesl_llvm_cbe_268_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_269_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @compute_pri_res\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_252_count);
  llvm_cbe_tmp__99 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__99);
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_253_count);
  llvm_cbe_tmp__100 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__100);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_vec(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64* getelementptr inbounds ([43 x i64]* @Adata_i, i64 0, i64 0), i64 %%1, i64 %%2, float* %%x, float* getelementptr inbounds ([19 x float]* @work_Ax, i64 0, i64 0), i64 0) nounwind, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_255_count);
   /*tail*/ mat_vec((float *)((&Adata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__99, llvm_cbe_tmp__100, (float *)llvm_cbe_x, (float *)((&work_Ax[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__99);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__100);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_256_count);
  llvm_cbe_tmp__101 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__101);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_add_scaled(float* getelementptr inbounds ([19 x float]* @work_z_prev, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Ax, i64 0, i64 0), float* %%z, i64 %%3, float -1.000000e+00) nounwind, !dbg !34 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_257_count);
   /*tail*/ vec_add_scaled((float *)((&work_z_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Ax[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)llvm_cbe_z, llvm_cbe_tmp__101, -0x1p0);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__101);
printf("\nArgument  = %f",-0x1p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_258_count);
  llvm_cbe_tmp__102 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__102);
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_260_count);
  llvm_cbe_tmp__103 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__103);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%5, %%7, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__102&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__103&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__108;
  } else {
    goto llvm_cbe_tmp__109;
  }

llvm_cbe_tmp__108:
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_263_count);
  llvm_cbe_tmp__104 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__104);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_z_prev, i64 0, i64 0), i64 %%9) nounwind, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_264_count);
  llvm_cbe_tmp__105 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_z_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__104);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__104);
printf("\nReturn  = %f",llvm_cbe_tmp__105);
}
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__105;   /* for PHI node */
  goto llvm_cbe_tmp__110;

llvm_cbe_tmp__109:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_266_count);
  llvm_cbe_tmp__106 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__106);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = tail call float @vec_norm_inf(float* getelementptr inbounds ([19 x float]* @work_z_prev, i64 0, i64 0), i64 %%12) nounwind, !dbg !35 for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_267_count);
  llvm_cbe_tmp__107 = (float ) /*tail*/ vec_norm_inf((float *)((&work_z_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__106);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__106);
printf("\nReturn  = %f",llvm_cbe_tmp__107);
}
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__107;   /* for PHI node */
  goto llvm_cbe_tmp__110;

llvm_cbe_tmp__110:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%10, %%8 ], [ %%13, %%11  for 0x%I64xth hint within @compute_pri_res  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__105);
printf("\n = %f",llvm_cbe_tmp__107);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @compute_pri_res}\n");
  return llvm_cbe_storemerge;
}


float compute_pri_tol(float llvm_cbe_eps_abs, float llvm_cbe_eps_rel) {
  static  unsigned long long aesl_llvm_cbe_270_count = 0;
  static  unsigned long long aesl_llvm_cbe_271_count = 0;
  static  unsigned long long aesl_llvm_cbe_272_count = 0;
  static  unsigned long long aesl_llvm_cbe_273_count = 0;
  static  unsigned long long aesl_llvm_cbe_274_count = 0;
  unsigned long long llvm_cbe_tmp__111;
  static  unsigned long long aesl_llvm_cbe_275_count = 0;
  static  unsigned long long aesl_llvm_cbe_276_count = 0;
  unsigned long long llvm_cbe_tmp__112;
  static  unsigned long long aesl_llvm_cbe_277_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_278_count = 0;
  static  unsigned long long aesl_llvm_cbe_279_count = 0;
  unsigned long long llvm_cbe_tmp__113;
  static  unsigned long long aesl_llvm_cbe_280_count = 0;
  float llvm_cbe_tmp__114;
  static  unsigned long long aesl_llvm_cbe_281_count = 0;
  static  unsigned long long aesl_llvm_cbe_282_count = 0;
  static  unsigned long long aesl_llvm_cbe_283_count = 0;
  static  unsigned long long aesl_llvm_cbe_284_count = 0;
  static  unsigned long long aesl_llvm_cbe_285_count = 0;
  static  unsigned long long aesl_llvm_cbe_286_count = 0;
  static  unsigned long long aesl_llvm_cbe_287_count = 0;
  unsigned long long llvm_cbe_tmp__115;
  static  unsigned long long aesl_llvm_cbe_288_count = 0;
  float llvm_cbe_tmp__116;
  static  unsigned long long aesl_llvm_cbe_289_count = 0;
  static  unsigned long long aesl_llvm_cbe_290_count = 0;
  static  unsigned long long aesl_llvm_cbe_291_count = 0;
  static  unsigned long long aesl_llvm_cbe_292_count = 0;
  static  unsigned long long aesl_llvm_cbe_293_count = 0;
  static  unsigned long long aesl_llvm_cbe_294_count = 0;
  static  unsigned long long aesl_llvm_cbe_295_count = 0;
  float llvm_cbe_tmp__117;
  static  unsigned long long aesl_llvm_cbe_296_count = 0;
  static  unsigned long long aesl_llvm_cbe_297_count = 0;
  static  unsigned long long aesl_llvm_cbe_298_count = 0;
  unsigned long long llvm_cbe_tmp__118;
  static  unsigned long long aesl_llvm_cbe_299_count = 0;
  float llvm_cbe_tmp__119;
  static  unsigned long long aesl_llvm_cbe_300_count = 0;
  static  unsigned long long aesl_llvm_cbe_301_count = 0;
  static  unsigned long long aesl_llvm_cbe_302_count = 0;
  static  unsigned long long aesl_llvm_cbe_303_count = 0;
  static  unsigned long long aesl_llvm_cbe_304_count = 0;
  static  unsigned long long aesl_llvm_cbe_305_count = 0;
  static  unsigned long long aesl_llvm_cbe_306_count = 0;
  unsigned long long llvm_cbe_tmp__120;
  static  unsigned long long aesl_llvm_cbe_307_count = 0;
  float llvm_cbe_tmp__121;
  static  unsigned long long aesl_llvm_cbe_308_count = 0;
  static  unsigned long long aesl_llvm_cbe_309_count = 0;
  static  unsigned long long aesl_llvm_cbe_310_count = 0;
  static  unsigned long long aesl_llvm_cbe_311_count = 0;
  static  unsigned long long aesl_llvm_cbe_312_count = 0;
  static  unsigned long long aesl_llvm_cbe_313_count = 0;
  static  unsigned long long aesl_llvm_cbe_314_count = 0;
  float llvm_cbe_tmp__122;
  static  unsigned long long aesl_llvm_cbe_315_count = 0;
  static  unsigned long long aesl_llvm_cbe_316_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_317_count = 0;
  static  unsigned long long aesl_llvm_cbe_318_count = 0;
  static  unsigned long long aesl_llvm_cbe_319_count = 0;
  static  unsigned long long aesl_llvm_cbe_320_count = 0;
  static  unsigned long long aesl_llvm_cbe_321_count = 0;
  static  unsigned long long aesl_llvm_cbe_322_count = 0;
  float llvm_cbe_tmp__123;
  static  unsigned long long aesl_llvm_cbe_323_count = 0;
  float llvm_cbe_tmp__124;
  static  unsigned long long aesl_llvm_cbe_324_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @compute_pri_tol\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_274_count);
  llvm_cbe_tmp__111 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__111);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_276_count);
  llvm_cbe_tmp__112 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__112);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%2, %%4, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__111&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__112&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__125;
  } else {
    goto llvm_cbe_tmp__126;
  }

llvm_cbe_tmp__125:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_279_count);
  llvm_cbe_tmp__113 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__113);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0), i64 %%6) nounwind, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_280_count);
  llvm_cbe_tmp__114 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__113);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__113);
printf("\nReturn  = %f",llvm_cbe_tmp__114);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_287_count);
  llvm_cbe_tmp__115 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__115);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Ax, i64 0, i64 0), i64 %%8) nounwind, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_288_count);
  llvm_cbe_tmp__116 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Ax[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__115);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__115);
printf("\nReturn  = %f",llvm_cbe_tmp__116);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = select i1 %%10, float %%7, float %%9, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_295_count);
  llvm_cbe_tmp__117 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__114, llvm_cbe_tmp__116))) ? ((float )llvm_cbe_tmp__114) : ((float )llvm_cbe_tmp__116));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__117, *(int*)(&llvm_cbe_tmp__117));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__117;   /* for PHI node */
  goto llvm_cbe_tmp__127;

llvm_cbe_tmp__126:
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !36 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_298_count);
  llvm_cbe_tmp__118 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__118);
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = tail call float @vec_norm_inf(float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0), i64 %%13) nounwind, !dbg !36 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_299_count);
  llvm_cbe_tmp__119 = (float ) /*tail*/ vec_norm_inf((float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__118);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__118);
printf("\nReturn  = %f",llvm_cbe_tmp__119);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !36 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_306_count);
  llvm_cbe_tmp__120 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__120);
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = tail call float @vec_norm_inf(float* getelementptr inbounds ([19 x float]* @work_Ax, i64 0, i64 0), i64 %%15) nounwind, !dbg !36 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_307_count);
  llvm_cbe_tmp__121 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Ax[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__120);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__120);
printf("\nReturn  = %f",llvm_cbe_tmp__121);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = select i1 %%17, float %%14, float %%16, !dbg !35 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_314_count);
  llvm_cbe_tmp__122 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__119, llvm_cbe_tmp__121))) ? ((float )llvm_cbe_tmp__119) : ((float )llvm_cbe_tmp__121));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__122, *(int*)(&llvm_cbe_tmp__122));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__122;   /* for PHI node */
  goto llvm_cbe_tmp__127;

llvm_cbe_tmp__127:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%18, %%12 ], [ %%11, %%5  for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__122);
printf("\n = %f",llvm_cbe_tmp__117);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = fmul float %%storemerge, %%eps_rel, !dbg !34 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_322_count);
  llvm_cbe_tmp__123 = (float )((float )(llvm_cbe_storemerge * llvm_cbe_eps_rel));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__123, *(int*)(&llvm_cbe_tmp__123));
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = fadd float %%20, %%eps_abs, !dbg !34 for 0x%I64xth hint within @compute_pri_tol  --> \n", ++aesl_llvm_cbe_323_count);
  llvm_cbe_tmp__124 = (float )((float )(llvm_cbe_tmp__123 + llvm_cbe_eps_abs));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__124, *(int*)(&llvm_cbe_tmp__124));
  if (AESL_DEBUG_TRACE)
      printf("\nEND @compute_pri_tol}\n");
  return llvm_cbe_tmp__124;
}


float compute_dua_res(float *llvm_cbe_x, float *llvm_cbe_y) {
  static  unsigned long long aesl_llvm_cbe_325_count = 0;
  static  unsigned long long aesl_llvm_cbe_326_count = 0;
  static  unsigned long long aesl_llvm_cbe_327_count = 0;
  static  unsigned long long aesl_llvm_cbe_328_count = 0;
  static  unsigned long long aesl_llvm_cbe_329_count = 0;
  static  unsigned long long aesl_llvm_cbe_330_count = 0;
  unsigned long long llvm_cbe_tmp__128;
  static  unsigned long long aesl_llvm_cbe_331_count = 0;
  static  unsigned long long aesl_llvm_cbe_332_count = 0;
  unsigned long long llvm_cbe_tmp__129;
  static  unsigned long long aesl_llvm_cbe_333_count = 0;
  unsigned long long llvm_cbe_tmp__130;
  static  unsigned long long aesl_llvm_cbe_334_count = 0;
  static  unsigned long long aesl_llvm_cbe_335_count = 0;
  unsigned long long llvm_cbe_tmp__131;
  static  unsigned long long aesl_llvm_cbe_336_count = 0;
  unsigned long long llvm_cbe_tmp__132;
  static  unsigned long long aesl_llvm_cbe_337_count = 0;
  static  unsigned long long aesl_llvm_cbe_338_count = 0;
  unsigned long long llvm_cbe_tmp__133;
  static  unsigned long long aesl_llvm_cbe_339_count = 0;
  static  unsigned long long aesl_llvm_cbe_340_count = 0;
  unsigned long long llvm_cbe_tmp__134;
  static  unsigned long long aesl_llvm_cbe_341_count = 0;
  static  unsigned long long aesl_llvm_cbe_342_count = 0;
  static  unsigned long long aesl_llvm_cbe_343_count = 0;
  unsigned long long llvm_cbe_tmp__135;
  static  unsigned long long aesl_llvm_cbe_344_count = 0;
  static  unsigned long long aesl_llvm_cbe_345_count = 0;
  unsigned long long llvm_cbe_tmp__136;
  static  unsigned long long aesl_llvm_cbe_346_count = 0;
  static  unsigned long long aesl_llvm_cbe_347_count = 0;
  static  unsigned long long aesl_llvm_cbe_348_count = 0;
  unsigned long long llvm_cbe_tmp__137;
  static  unsigned long long aesl_llvm_cbe_349_count = 0;
  static  unsigned long long aesl_llvm_cbe_350_count = 0;
  unsigned long long llvm_cbe_tmp__138;
  static  unsigned long long aesl_llvm_cbe_351_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_352_count = 0;
  static  unsigned long long aesl_llvm_cbe_353_count = 0;
  float llvm_cbe_tmp__139;
  static  unsigned long long aesl_llvm_cbe_354_count = 0;
  unsigned long long llvm_cbe_tmp__140;
  static  unsigned long long aesl_llvm_cbe_355_count = 0;
  float llvm_cbe_tmp__141;
  static  unsigned long long aesl_llvm_cbe_356_count = 0;
  float llvm_cbe_tmp__142;
  static  unsigned long long aesl_llvm_cbe_357_count = 0;
  static  unsigned long long aesl_llvm_cbe_358_count = 0;
  unsigned long long llvm_cbe_tmp__143;
  static  unsigned long long aesl_llvm_cbe_359_count = 0;
  float llvm_cbe_tmp__144;
  static  unsigned long long aesl_llvm_cbe_360_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_361_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @compute_dua_res\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_330_count);
  llvm_cbe_tmp__128 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__128);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), i64 %%1) nounwind, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_331_count);
   /*tail*/ prea_vec_copy((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__128);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__128);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_332_count);
  llvm_cbe_tmp__129 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__129);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_333_count);
  llvm_cbe_tmp__130 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__130);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_vec(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%2, i64 %%3, float* %%x, float* getelementptr inbounds ([15 x float]* @work_Px, i64 0, i64 0), i64 0) nounwind, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_334_count);
   /*tail*/ mat_vec((float *)((&Pdata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__129, llvm_cbe_tmp__130, (float *)llvm_cbe_x, (float *)((&work_Px[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__129);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__130);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_335_count);
  llvm_cbe_tmp__131 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__131);
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_336_count);
  llvm_cbe_tmp__132 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__132);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_tpose_vec(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%4, i64 %%5, float* %%x, float* getelementptr inbounds ([15 x float]* @work_Px, i64 0, i64 0), i64 1, i64 1) nounwind, !dbg !34 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_337_count);
   /*tail*/ mat_tpose_vec((float *)((&Pdata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__131, llvm_cbe_tmp__132, (float *)llvm_cbe_x, (float *)((&work_Px[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 1ull, 1ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__131);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__132);
printf("\nArgument  = 0x%I64X",1ull);
printf("\nArgument  = 0x%I64X",1ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_338_count);
  llvm_cbe_tmp__133 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__133);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_add_scaled(float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Px, i64 0, i64 0), i64 %%6, float 1.000000e+00) nounwind, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_339_count);
   /*tail*/ vec_add_scaled((float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Px[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__133, 0x1p0);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__133);
printf("\nArgument  = %f",0x1p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_340_count);
  llvm_cbe_tmp__134 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__134);
  if ((((signed long long )llvm_cbe_tmp__134) > ((signed long long )0ull))) {
    goto llvm_cbe_tmp__145;
  } else {
    goto llvm_cbe_tmp__146;
  }

llvm_cbe_tmp__145:
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_343_count);
  llvm_cbe_tmp__135 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__135);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_tpose_vec(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64* getelementptr inbounds ([43 x i64]* @Adata_i, i64 0, i64 0), i64 %%10, i64 %%7, float* %%y, float* getelementptr inbounds ([15 x float]* @work_Aty, i64 0, i64 0), i64 0, i64 0) nounwind, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_344_count);
   /*tail*/ mat_tpose_vec((float *)((&Adata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__135, llvm_cbe_tmp__134, (float *)llvm_cbe_y, (float *)((&work_Aty[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0ull, 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__135);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__134);
printf("\nArgument  = 0x%I64X",0ull);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_345_count);
  llvm_cbe_tmp__136 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__136);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_add_scaled(float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Aty, i64 0, i64 0), i64 %%11, float 1.000000e+00) nounwind, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_346_count);
   /*tail*/ vec_add_scaled((float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Aty[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__136, 0x1p0);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__136);
printf("\nArgument  = %f",0x1p0);
}
  goto llvm_cbe_tmp__146;

llvm_cbe_tmp__146:
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_348_count);
  llvm_cbe_tmp__137 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__137);
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_350_count);
  llvm_cbe_tmp__138 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__138);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%14, %%16, !dbg !35 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__137&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__138&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__147;
  } else {
    goto llvm_cbe_tmp__148;
  }

llvm_cbe_tmp__147:
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_353_count);
  llvm_cbe_tmp__139 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__139, *(int*)(&llvm_cbe_tmp__139));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_354_count);
  llvm_cbe_tmp__140 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__140);
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), i64 %%19) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_355_count);
  llvm_cbe_tmp__141 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__140);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__140);
printf("\nReturn  = %f",llvm_cbe_tmp__141);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = fmul float %%18, %%20, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_356_count);
  llvm_cbe_tmp__142 = (float )((float )(llvm_cbe_tmp__139 * llvm_cbe_tmp__141));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__142, *(int*)(&llvm_cbe_tmp__142));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__142;   /* for PHI node */
  goto llvm_cbe_tmp__149;

llvm_cbe_tmp__148:
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_358_count);
  llvm_cbe_tmp__143 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__143);
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_x_prev, i64 0, i64 0), i64 %%23) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_359_count);
  llvm_cbe_tmp__144 = (float ) /*tail*/ vec_norm_inf((float *)((&work_x_prev[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__143);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__143);
printf("\nReturn  = %f",llvm_cbe_tmp__144);
}
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__144;   /* for PHI node */
  goto llvm_cbe_tmp__149;

llvm_cbe_tmp__149:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%21, %%17 ], [ %%24, %%22  for 0x%I64xth hint within @compute_dua_res  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__142);
printf("\n = %f",llvm_cbe_tmp__144);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @compute_dua_res}\n");
  return llvm_cbe_storemerge;
}


float compute_dua_tol(float llvm_cbe_eps_abs, float llvm_cbe_eps_rel) {
  static  unsigned long long aesl_llvm_cbe_362_count = 0;
  static  unsigned long long aesl_llvm_cbe_363_count = 0;
  static  unsigned long long aesl_llvm_cbe_364_count = 0;
  static  unsigned long long aesl_llvm_cbe_365_count = 0;
  static  unsigned long long aesl_llvm_cbe_366_count = 0;
  unsigned long long llvm_cbe_tmp__150;
  static  unsigned long long aesl_llvm_cbe_367_count = 0;
  static  unsigned long long aesl_llvm_cbe_368_count = 0;
  unsigned long long llvm_cbe_tmp__151;
  static  unsigned long long aesl_llvm_cbe_369_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_370_count = 0;
  static  unsigned long long aesl_llvm_cbe_371_count = 0;
  unsigned long long llvm_cbe_tmp__152;
  static  unsigned long long aesl_llvm_cbe_372_count = 0;
  float llvm_cbe_tmp__153;
  static  unsigned long long aesl_llvm_cbe_373_count = 0;
  static  unsigned long long aesl_llvm_cbe_374_count = 0;
  static  unsigned long long aesl_llvm_cbe_375_count = 0;
  static  unsigned long long aesl_llvm_cbe_376_count = 0;
  static  unsigned long long aesl_llvm_cbe_377_count = 0;
  static  unsigned long long aesl_llvm_cbe_378_count = 0;
  static  unsigned long long aesl_llvm_cbe_379_count = 0;
  static  unsigned long long aesl_llvm_cbe_380_count = 0;
  static  unsigned long long aesl_llvm_cbe_381_count = 0;
  static  unsigned long long aesl_llvm_cbe_382_count = 0;
  static  unsigned long long aesl_llvm_cbe_383_count = 0;
  static  unsigned long long aesl_llvm_cbe_384_count = 0;
  unsigned long long llvm_cbe_tmp__154;
  static  unsigned long long aesl_llvm_cbe_385_count = 0;
  float llvm_cbe_tmp__155;
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
  float llvm_cbe_tmp__156;
  static  unsigned long long aesl_llvm_cbe_397_count = 0;
  static  unsigned long long aesl_llvm_cbe_398_count = 0;
  static  unsigned long long aesl_llvm_cbe_399_count = 0;
  static  unsigned long long aesl_llvm_cbe_400_count = 0;
  static  unsigned long long aesl_llvm_cbe_401_count = 0;
  static  unsigned long long aesl_llvm_cbe_402_count = 0;
  static  unsigned long long aesl_llvm_cbe_403_count = 0;
  static  unsigned long long aesl_llvm_cbe_404_count = 0;
  static  unsigned long long aesl_llvm_cbe_405_count = 0;
  static  unsigned long long aesl_llvm_cbe_406_count = 0;
  static  unsigned long long aesl_llvm_cbe_407_count = 0;
  static  unsigned long long aesl_llvm_cbe_408_count = 0;
  unsigned long long llvm_cbe_tmp__157;
  static  unsigned long long aesl_llvm_cbe_409_count = 0;
  float llvm_cbe_tmp__158;
  static  unsigned long long aesl_llvm_cbe_410_count = 0;
  static  unsigned long long aesl_llvm_cbe_411_count = 0;
  static  unsigned long long aesl_llvm_cbe_412_count = 0;
  static  unsigned long long aesl_llvm_cbe_413_count = 0;
  static  unsigned long long aesl_llvm_cbe_414_count = 0;
  static  unsigned long long aesl_llvm_cbe_415_count = 0;
  static  unsigned long long aesl_llvm_cbe_416_count = 0;
  static  unsigned long long aesl_llvm_cbe_417_count = 0;
  static  unsigned long long aesl_llvm_cbe_418_count = 0;
  static  unsigned long long aesl_llvm_cbe_419_count = 0;
  static  unsigned long long aesl_llvm_cbe_420_count = 0;
  float llvm_cbe_tmp__159;
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
  float llvm_cbe_tmp__160;
  static  unsigned long long aesl_llvm_cbe_433_count = 0;
  float llvm_cbe_tmp__161;
  static  unsigned long long aesl_llvm_cbe_434_count = 0;
  static  unsigned long long aesl_llvm_cbe_435_count = 0;
  static  unsigned long long aesl_llvm_cbe_436_count = 0;
  unsigned long long llvm_cbe_tmp__162;
  static  unsigned long long aesl_llvm_cbe_437_count = 0;
  float llvm_cbe_tmp__163;
  static  unsigned long long aesl_llvm_cbe_438_count = 0;
  static  unsigned long long aesl_llvm_cbe_439_count = 0;
  static  unsigned long long aesl_llvm_cbe_440_count = 0;
  static  unsigned long long aesl_llvm_cbe_441_count = 0;
  static  unsigned long long aesl_llvm_cbe_442_count = 0;
  static  unsigned long long aesl_llvm_cbe_443_count = 0;
  static  unsigned long long aesl_llvm_cbe_444_count = 0;
  static  unsigned long long aesl_llvm_cbe_445_count = 0;
  static  unsigned long long aesl_llvm_cbe_446_count = 0;
  static  unsigned long long aesl_llvm_cbe_447_count = 0;
  static  unsigned long long aesl_llvm_cbe_448_count = 0;
  static  unsigned long long aesl_llvm_cbe_449_count = 0;
  unsigned long long llvm_cbe_tmp__164;
  static  unsigned long long aesl_llvm_cbe_450_count = 0;
  float llvm_cbe_tmp__165;
  static  unsigned long long aesl_llvm_cbe_451_count = 0;
  static  unsigned long long aesl_llvm_cbe_452_count = 0;
  static  unsigned long long aesl_llvm_cbe_453_count = 0;
  static  unsigned long long aesl_llvm_cbe_454_count = 0;
  static  unsigned long long aesl_llvm_cbe_455_count = 0;
  static  unsigned long long aesl_llvm_cbe_456_count = 0;
  static  unsigned long long aesl_llvm_cbe_457_count = 0;
  static  unsigned long long aesl_llvm_cbe_458_count = 0;
  static  unsigned long long aesl_llvm_cbe_459_count = 0;
  static  unsigned long long aesl_llvm_cbe_460_count = 0;
  static  unsigned long long aesl_llvm_cbe_461_count = 0;
  float llvm_cbe_tmp__166;
  static  unsigned long long aesl_llvm_cbe_462_count = 0;
  static  unsigned long long aesl_llvm_cbe_463_count = 0;
  static  unsigned long long aesl_llvm_cbe_464_count = 0;
  static  unsigned long long aesl_llvm_cbe_465_count = 0;
  static  unsigned long long aesl_llvm_cbe_466_count = 0;
  static  unsigned long long aesl_llvm_cbe_467_count = 0;
  static  unsigned long long aesl_llvm_cbe_468_count = 0;
  static  unsigned long long aesl_llvm_cbe_469_count = 0;
  static  unsigned long long aesl_llvm_cbe_470_count = 0;
  static  unsigned long long aesl_llvm_cbe_471_count = 0;
  static  unsigned long long aesl_llvm_cbe_472_count = 0;
  static  unsigned long long aesl_llvm_cbe_473_count = 0;
  unsigned long long llvm_cbe_tmp__167;
  static  unsigned long long aesl_llvm_cbe_474_count = 0;
  float llvm_cbe_tmp__168;
  static  unsigned long long aesl_llvm_cbe_475_count = 0;
  static  unsigned long long aesl_llvm_cbe_476_count = 0;
  static  unsigned long long aesl_llvm_cbe_477_count = 0;
  static  unsigned long long aesl_llvm_cbe_478_count = 0;
  static  unsigned long long aesl_llvm_cbe_479_count = 0;
  static  unsigned long long aesl_llvm_cbe_480_count = 0;
  static  unsigned long long aesl_llvm_cbe_481_count = 0;
  static  unsigned long long aesl_llvm_cbe_482_count = 0;
  static  unsigned long long aesl_llvm_cbe_483_count = 0;
  static  unsigned long long aesl_llvm_cbe_484_count = 0;
  static  unsigned long long aesl_llvm_cbe_485_count = 0;
  float llvm_cbe_tmp__169;
  static  unsigned long long aesl_llvm_cbe_486_count = 0;
  static  unsigned long long aesl_llvm_cbe_487_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
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
  static  unsigned long long aesl_llvm_cbe_498_count = 0;
  float llvm_cbe_tmp__170;
  static  unsigned long long aesl_llvm_cbe_499_count = 0;
  float llvm_cbe_tmp__171;
  static  unsigned long long aesl_llvm_cbe_500_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @compute_dua_tol\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_366_count);
  llvm_cbe_tmp__150 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__150);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_368_count);
  llvm_cbe_tmp__151 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__151);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%2, %%4, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__150&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__151&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__172;
  } else {
    goto llvm_cbe_tmp__173;
  }

llvm_cbe_tmp__172:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_371_count);
  llvm_cbe_tmp__152 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__152);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), i64 %%6) nounwind, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_372_count);
  llvm_cbe_tmp__153 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__152);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__152);
printf("\nReturn  = %f",llvm_cbe_tmp__153);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_384_count);
  llvm_cbe_tmp__154 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__154);
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Aty, i64 0, i64 0), i64 %%8) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_385_count);
  llvm_cbe_tmp__155 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Aty[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__154);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__154);
printf("\nReturn  = %f",llvm_cbe_tmp__155);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = select i1 %%10, float %%7, float %%9, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_396_count);
  llvm_cbe_tmp__156 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__153, llvm_cbe_tmp__155))) ? ((float )llvm_cbe_tmp__153) : ((float )llvm_cbe_tmp__155));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__156, *(int*)(&llvm_cbe_tmp__156));
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_408_count);
  llvm_cbe_tmp__157 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__157);
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Px, i64 0, i64 0), i64 %%12) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_409_count);
  llvm_cbe_tmp__158 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Px[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__157);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__157);
printf("\nReturn  = %f",llvm_cbe_tmp__158);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = select i1 %%14, float %%11, float %%13, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_420_count);
  llvm_cbe_tmp__159 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__156, llvm_cbe_tmp__158))) ? ((float )llvm_cbe_tmp__156) : ((float )llvm_cbe_tmp__158));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__159, *(int*)(&llvm_cbe_tmp__159));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 3), align 8, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_432_count);
  llvm_cbe_tmp__160 = (float )*((&scaling.field3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__160, *(int*)(&llvm_cbe_tmp__160));
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = fmul float %%15, %%16, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_433_count);
  llvm_cbe_tmp__161 = (float )((float )(llvm_cbe_tmp__159 * llvm_cbe_tmp__160));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__161, *(int*)(&llvm_cbe_tmp__161));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__161;   /* for PHI node */
  goto llvm_cbe_tmp__174;

llvm_cbe_tmp__173:
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_436_count);
  llvm_cbe_tmp__162 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__162);
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), i64 %%19) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_437_count);
  llvm_cbe_tmp__163 = (float ) /*tail*/ vec_norm_inf((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__162);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__162);
printf("\nReturn  = %f",llvm_cbe_tmp__163);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_449_count);
  llvm_cbe_tmp__164 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__164);
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_Aty, i64 0, i64 0), i64 %%21) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_450_count);
  llvm_cbe_tmp__165 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Aty[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__164);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__164);
printf("\nReturn  = %f",llvm_cbe_tmp__165);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = select i1 %%23, float %%20, float %%22, !dbg !35 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_461_count);
  llvm_cbe_tmp__166 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__163, llvm_cbe_tmp__165))) ? ((float )llvm_cbe_tmp__163) : ((float )llvm_cbe_tmp__165));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__166, *(int*)(&llvm_cbe_tmp__166));
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_473_count);
  llvm_cbe_tmp__167 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__167);
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_Px, i64 0, i64 0), i64 %%25) nounwind, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_474_count);
  llvm_cbe_tmp__168 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Px[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__167);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__167);
printf("\nReturn  = %f",llvm_cbe_tmp__168);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = select i1 %%27, float %%24, float %%26, !dbg !36 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_485_count);
  llvm_cbe_tmp__169 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__166, llvm_cbe_tmp__168))) ? ((float )llvm_cbe_tmp__166) : ((float )llvm_cbe_tmp__168));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__169, *(int*)(&llvm_cbe_tmp__169));
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__169;   /* for PHI node */
  goto llvm_cbe_tmp__174;

llvm_cbe_tmp__174:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%28, %%18 ], [ %%17, %%5  for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__169);
printf("\n = %f",llvm_cbe_tmp__161);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = fmul float %%storemerge, %%eps_rel, !dbg !34 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_498_count);
  llvm_cbe_tmp__170 = (float )((float )(llvm_cbe_storemerge * llvm_cbe_eps_rel));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__170, *(int*)(&llvm_cbe_tmp__170));
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = fadd float %%30, %%eps_abs, !dbg !34 for 0x%I64xth hint within @compute_dua_tol  --> \n", ++aesl_llvm_cbe_499_count);
  llvm_cbe_tmp__171 = (float )((float )(llvm_cbe_tmp__170 + llvm_cbe_eps_abs));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__171, *(int*)(&llvm_cbe_tmp__171));
  if (AESL_DEBUG_TRACE)
      printf("\nEND @compute_dua_tol}\n");
  return llvm_cbe_tmp__171;
}


signed long long is_primal_infeasible(float llvm_cbe_eps_prim_inf) {
  static  unsigned long long aesl_llvm_cbe_501_count = 0;
  static  unsigned long long aesl_llvm_cbe_502_count = 0;
  static  unsigned long long aesl_llvm_cbe_503_count = 0;
  static  unsigned long long aesl_llvm_cbe_504_count = 0;
  static  unsigned long long aesl_llvm_cbe_505_count = 0;
  static  unsigned long long aesl_llvm_cbe_506_count = 0;
  static  unsigned long long aesl_llvm_cbe_507_count = 0;
  static  unsigned long long aesl_llvm_cbe_508_count = 0;
  static  unsigned long long aesl_llvm_cbe_509_count = 0;
  static  unsigned long long aesl_llvm_cbe_510_count = 0;
  static  unsigned long long aesl_llvm_cbe_511_count = 0;
  static  unsigned long long aesl_llvm_cbe_512_count = 0;
  static  unsigned long long aesl_llvm_cbe_513_count = 0;
  static  unsigned long long aesl_llvm_cbe_514_count = 0;
  static  unsigned long long aesl_llvm_cbe_515_count = 0;
  static  unsigned long long aesl_llvm_cbe_516_count = 0;
  static  unsigned long long aesl_llvm_cbe_517_count = 0;
  static  unsigned long long aesl_llvm_cbe_518_count = 0;
  static  unsigned long long aesl_llvm_cbe_519_count = 0;
  static  unsigned long long aesl_llvm_cbe_520_count = 0;
  static  unsigned long long aesl_llvm_cbe_521_count = 0;
  static  unsigned long long aesl_llvm_cbe_522_count = 0;
  static  unsigned long long aesl_llvm_cbe_523_count = 0;
  static  unsigned long long aesl_llvm_cbe_524_count = 0;
  static  unsigned long long aesl_llvm_cbe_525_count = 0;
  static  unsigned long long aesl_llvm_cbe_526_count = 0;
  static  unsigned long long aesl_llvm_cbe_527_count = 0;
  static  unsigned long long aesl_llvm_cbe_528_count = 0;
  unsigned long long llvm_cbe_tmp__175;
  static  unsigned long long aesl_llvm_cbe_529_count = 0;
  static  unsigned long long aesl_llvm_cbe_530_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge11_count = 0;
  unsigned long long llvm_cbe_storemerge11;
  unsigned long long llvm_cbe_storemerge11__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_531_count = 0;
  float *llvm_cbe_tmp__176;
  static  unsigned long long aesl_llvm_cbe_532_count = 0;
  float llvm_cbe_tmp__177;
  static  unsigned long long aesl_llvm_cbe_533_count = 0;
  double llvm_cbe_tmp__178;
  static  unsigned long long aesl_llvm_cbe_534_count = 0;
  static  unsigned long long aesl_llvm_cbe_535_count = 0;
  static  unsigned long long aesl_llvm_cbe_536_count = 0;
  float *llvm_cbe_tmp__179;
  static  unsigned long long aesl_llvm_cbe_537_count = 0;
  float llvm_cbe_tmp__180;
  static  unsigned long long aesl_llvm_cbe_538_count = 0;
  double llvm_cbe_tmp__181;
  static  unsigned long long aesl_llvm_cbe_539_count = 0;
  static  unsigned long long aesl_llvm_cbe_540_count = 0;
  static  unsigned long long aesl_llvm_cbe_541_count = 0;
  float *llvm_cbe_tmp__182;
  static  unsigned long long aesl_llvm_cbe_542_count = 0;
  static  unsigned long long aesl_llvm_cbe_543_count = 0;
  static  unsigned long long aesl_llvm_cbe_544_count = 0;
  float *llvm_cbe_tmp__183;
  static  unsigned long long aesl_llvm_cbe_545_count = 0;
  float llvm_cbe_tmp__184;
  static  unsigned long long aesl_llvm_cbe_546_count = 0;
  static  unsigned long long aesl_llvm_cbe_547_count = 0;
  static  unsigned long long aesl_llvm_cbe_548_count = 0;
  double llvm_cbe_tmp__185;
  static  unsigned long long aesl_llvm_cbe_549_count = 0;
  static  unsigned long long aesl_llvm_cbe_550_count = 0;
  double llvm_cbe_tmp__186;
  double llvm_cbe_tmp__186__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_551_count = 0;
  float llvm_cbe_tmp__187;
  static  unsigned long long aesl_llvm_cbe_552_count = 0;
  static  unsigned long long aesl_llvm_cbe_553_count = 0;
  static  unsigned long long aesl_llvm_cbe_554_count = 0;
  static  unsigned long long aesl_llvm_cbe_555_count = 0;
  float *llvm_cbe_tmp__188;
  static  unsigned long long aesl_llvm_cbe_556_count = 0;
  float llvm_cbe_tmp__189;
  static  unsigned long long aesl_llvm_cbe_557_count = 0;
  double llvm_cbe_tmp__190;
  static  unsigned long long aesl_llvm_cbe_558_count = 0;
  static  unsigned long long aesl_llvm_cbe_559_count = 0;
  static  unsigned long long aesl_llvm_cbe_560_count = 0;
  float *llvm_cbe_tmp__191;
  static  unsigned long long aesl_llvm_cbe_561_count = 0;
  float llvm_cbe_tmp__192;
  static  unsigned long long aesl_llvm_cbe_562_count = 0;
  static  unsigned long long aesl_llvm_cbe_563_count = 0;
  static  unsigned long long aesl_llvm_cbe_564_count = 0;
  double llvm_cbe_tmp__193;
  static  unsigned long long aesl_llvm_cbe_565_count = 0;
  static  unsigned long long aesl_llvm_cbe_566_count = 0;
  double llvm_cbe_tmp__194;
  double llvm_cbe_tmp__194__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_567_count = 0;
  float llvm_cbe_tmp__195;
  static  unsigned long long aesl_llvm_cbe_568_count = 0;
  static  unsigned long long aesl_llvm_cbe_569_count = 0;
  static  unsigned long long aesl_llvm_cbe_570_count = 0;
  static  unsigned long long aesl_llvm_cbe_571_count = 0;
  unsigned long long llvm_cbe_tmp__196;
  static  unsigned long long aesl_llvm_cbe_572_count = 0;
  static  unsigned long long aesl_llvm_cbe_573_count = 0;
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
  unsigned long long llvm_cbe_tmp__197;
  static  unsigned long long aesl_llvm_cbe_596_count = 0;
  static  unsigned long long aesl_llvm_cbe_597_count = 0;
  unsigned long long llvm_cbe_tmp__198;
  static  unsigned long long aesl_llvm_cbe_598_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_599_count = 0;
  static  unsigned long long aesl_llvm_cbe_600_count = 0;
  static  unsigned long long aesl_llvm_cbe_601_count = 0;
  unsigned long long llvm_cbe_tmp__199;
  static  unsigned long long aesl_llvm_cbe_602_count = 0;
  float llvm_cbe_tmp__200;
  static  unsigned long long aesl_llvm_cbe_603_count = 0;
  static  unsigned long long aesl_llvm_cbe_604_count = 0;
  static  unsigned long long aesl_llvm_cbe_605_count = 0;
  float llvm_cbe_tmp__201;
  static  unsigned long long aesl_llvm_cbe_606_count = 0;
  static  unsigned long long aesl_llvm_cbe_607_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  float llvm_cbe_storemerge1;
  float llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_608_count = 0;
  static  unsigned long long aesl_llvm_cbe_609_count = 0;
  static  unsigned long long aesl_llvm_cbe_610_count = 0;
  static  unsigned long long aesl_llvm_cbe_611_count = 0;
  static  unsigned long long aesl_llvm_cbe_612_count = 0;
  static  unsigned long long aesl_llvm_cbe_613_count = 0;
  static  unsigned long long aesl_llvm_cbe_614_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_625_count = 0;
  static  unsigned long long aesl_llvm_cbe_626_count = 0;
  static  unsigned long long aesl_llvm_cbe_627_count = 0;
  static  unsigned long long aesl_llvm_cbe_628_count = 0;
  static  unsigned long long aesl_llvm_cbe_629_count = 0;
  static  unsigned long long aesl_llvm_cbe_630_count = 0;
  static  unsigned long long aesl_llvm_cbe_631_count = 0;
  static  unsigned long long aesl_llvm_cbe_632_count = 0;
  static  unsigned long long aesl_llvm_cbe_633_count = 0;
  unsigned long long llvm_cbe_tmp__202;
  static  unsigned long long aesl_llvm_cbe_634_count = 0;
  static  unsigned long long aesl_llvm_cbe_635_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge28_count = 0;
  unsigned long long llvm_cbe_storemerge28;
  unsigned long long llvm_cbe_storemerge28__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_636_count = 0;
  float llvm_cbe_tmp__203;
  float llvm_cbe_tmp__203__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_637_count = 0;
  float *llvm_cbe_tmp__204;
  static  unsigned long long aesl_llvm_cbe_638_count = 0;
  float llvm_cbe_tmp__205;
  static  unsigned long long aesl_llvm_cbe_639_count = 0;
  float *llvm_cbe_tmp__206;
  static  unsigned long long aesl_llvm_cbe_640_count = 0;
  float llvm_cbe_tmp__207;
  static  unsigned long long aesl_llvm_cbe_641_count = 0;
  static  unsigned long long aesl_llvm_cbe_642_count = 0;
  float llvm_cbe_tmp__208;
  static  unsigned long long aesl_llvm_cbe_643_count = 0;
  float llvm_cbe_tmp__209;
  static  unsigned long long aesl_llvm_cbe_644_count = 0;
  float *llvm_cbe_tmp__210;
  static  unsigned long long aesl_llvm_cbe_645_count = 0;
  float llvm_cbe_tmp__211;
  static  unsigned long long aesl_llvm_cbe_646_count = 0;
  static  unsigned long long aesl_llvm_cbe_647_count = 0;
  float llvm_cbe_tmp__212;
  static  unsigned long long aesl_llvm_cbe_648_count = 0;
  float llvm_cbe_tmp__213;
  static  unsigned long long aesl_llvm_cbe_649_count = 0;
  float llvm_cbe_tmp__214;
  static  unsigned long long aesl_llvm_cbe_650_count = 0;
  float llvm_cbe_tmp__215;
  static  unsigned long long aesl_llvm_cbe_651_count = 0;
  static  unsigned long long aesl_llvm_cbe_652_count = 0;
  static  unsigned long long aesl_llvm_cbe_653_count = 0;
  static  unsigned long long aesl_llvm_cbe_654_count = 0;
  unsigned long long llvm_cbe_tmp__216;
  static  unsigned long long aesl_llvm_cbe_655_count = 0;
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
  static  unsigned long long aesl_llvm_cbe_666_count = 0;
  static  unsigned long long aesl_llvm_cbe_667_count = 0;
  static  unsigned long long aesl_llvm_cbe_668_count = 0;
  static  unsigned long long aesl_llvm_cbe_669_count = 0;
  static  unsigned long long aesl_llvm_cbe_670_count = 0;
  static  unsigned long long aesl_llvm_cbe_671_count = 0;
  static  unsigned long long aesl_llvm_cbe_672_count = 0;
  static  unsigned long long aesl_llvm_cbe_673_count = 0;
  static  unsigned long long aesl_llvm_cbe_674_count = 0;
  static  unsigned long long aesl_llvm_cbe_675_count = 0;
  static  unsigned long long aesl_llvm_cbe_676_count = 0;
  static  unsigned long long aesl_llvm_cbe_677_count = 0;
  static  unsigned long long aesl_llvm_cbe__2e_lcssa_count = 0;
  float llvm_cbe__2e_lcssa;
  float llvm_cbe__2e_lcssa__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_678_count = 0;
  float llvm_cbe_tmp__217;
  static  unsigned long long aesl_llvm_cbe_679_count = 0;
  static  unsigned long long aesl_llvm_cbe_680_count = 0;
  static  unsigned long long aesl_llvm_cbe_681_count = 0;
  unsigned long long llvm_cbe_tmp__218;
  static  unsigned long long aesl_llvm_cbe_682_count = 0;
  static  unsigned long long aesl_llvm_cbe_683_count = 0;
  unsigned long long llvm_cbe_tmp__219;
  static  unsigned long long aesl_llvm_cbe_684_count = 0;
  static  unsigned long long aesl_llvm_cbe_685_count = 0;
  unsigned long long llvm_cbe_tmp__220;
  static  unsigned long long aesl_llvm_cbe_686_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond6_count = 0;
  bool llvm_cbe_or_2e_cond6;
  static  unsigned long long aesl_llvm_cbe_687_count = 0;
  static  unsigned long long aesl_llvm_cbe_688_count = 0;
  unsigned long long llvm_cbe_tmp__221;
  static  unsigned long long aesl_llvm_cbe_689_count = 0;
  static  unsigned long long aesl_llvm_cbe_690_count = 0;
  static  unsigned long long aesl_llvm_cbe_691_count = 0;
  unsigned long long llvm_cbe_tmp__222;
  static  unsigned long long aesl_llvm_cbe_692_count = 0;
  float llvm_cbe_tmp__223;
  static  unsigned long long aesl_llvm_cbe_693_count = 0;
  static  unsigned long long aesl_llvm_cbe_694_count = 0;
  unsigned long long llvm_cbe_tmp__224;
  static  unsigned long long aesl_llvm_cbe_695_count = 0;
  static  unsigned long long aesl_llvm_cbe_696_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge3_count = 0;
  unsigned long long llvm_cbe_storemerge3;
  unsigned long long llvm_cbe_storemerge3__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_697_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @is_primal_infeasible\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_528_count);
  llvm_cbe_tmp__175 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__175);
  if ((((signed long long )llvm_cbe_tmp__175) > ((signed long long )0ull))) {
    llvm_cbe_storemerge11__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph13;
  } else {
    goto llvm_cbe__2e__crit_edge14;
  }

  do {     /* Syntactic loop '.lr.ph13' to make GCC happy */
llvm_cbe__2e_lr_2e_ph13:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge11 = phi i64 [ %%40, %%39 ], [ 0, %%0  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_storemerge11_count);
  llvm_cbe_storemerge11 = (unsigned long long )llvm_cbe_storemerge11__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",llvm_cbe_storemerge11);
printf("\n = 0x%I64X",llvm_cbe_tmp__196);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = getelementptr inbounds [19 x float]* @udata, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_531_count);
  llvm_cbe_tmp__176 = (float *)(&udata[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge11) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'udata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* %%3, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_532_count);
  llvm_cbe_tmp__177 = (float )*llvm_cbe_tmp__176;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__177, *(int*)(&llvm_cbe_tmp__177));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = fpext float %%4 to double, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_533_count);
  llvm_cbe_tmp__178 = (double )((double )llvm_cbe_tmp__177);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__178, *(long long*)(&llvm_cbe_tmp__178));
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__178, 0x1.4adf4bc6a7efap86))) {
    goto llvm_cbe_tmp__225;
  } else {
    goto llvm_cbe_tmp__226;
  }

llvm_cbe_tmp__227:
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = add nsw i64 %%storemerge11, 1, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_571_count);
  llvm_cbe_tmp__196 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge11&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__196&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__196) < ((signed long long )llvm_cbe_tmp__175))) {
    llvm_cbe_storemerge11__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__196;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph13;
  } else {
    goto llvm_cbe__2e__crit_edge14;
  }

llvm_cbe_tmp__228:
  goto llvm_cbe_tmp__227;

llvm_cbe_tmp__229:
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = getelementptr inbounds [19 x float]* @work_delta_y, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_541_count);
  llvm_cbe_tmp__182 = (float *)(&work_delta_y[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge11) < 19 && "Write access out of array 'work_delta_y' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float 0.000000e+00, float* %%13, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_542_count);
  *llvm_cbe_tmp__182 = 0x0p0;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x0p0);
  goto llvm_cbe_tmp__228;

llvm_cbe_tmp__225:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds [19 x float]* @ldata, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_536_count);
  llvm_cbe_tmp__179 = (float *)(&ldata[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge11) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'ldata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%9 = load float* %%8, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_537_count);
  llvm_cbe_tmp__180 = (float )*llvm_cbe_tmp__179;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__180, *(int*)(&llvm_cbe_tmp__180));
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = fpext float %%9 to double, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_538_count);
  llvm_cbe_tmp__181 = (double )((double )llvm_cbe_tmp__180);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__181, *(long long*)(&llvm_cbe_tmp__181));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__181, -0x1.4adf4bc6a7efap86))) {
    goto llvm_cbe_tmp__229;
  } else {
    goto llvm_cbe_tmp__230;
  }

llvm_cbe_tmp__231:
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = phi double [ %%19, %%18 ], [ 0.000000e+00, %%14 ], !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_550_count);
  llvm_cbe_tmp__186 = (double )llvm_cbe_tmp__186__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %lf",llvm_cbe_tmp__186);
printf("\n = %lf",llvm_cbe_tmp__185);
printf("\n = %lf",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = fptrunc double %%21 to float, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_551_count);
  llvm_cbe_tmp__187 = (float )((float )llvm_cbe_tmp__186);
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__187, *(int*)(&llvm_cbe_tmp__187));

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge11) < 19 && "Write access out of array 'work_delta_y' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%22, float* %%15, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_552_count);
  *llvm_cbe_tmp__183 = llvm_cbe_tmp__187;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__187);
  goto llvm_cbe_tmp__228;

llvm_cbe_tmp__230:
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = getelementptr inbounds [19 x float]* @work_delta_y, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_544_count);
  llvm_cbe_tmp__183 = (float *)(&work_delta_y[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge11) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_delta_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load float* %%15, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_545_count);
  llvm_cbe_tmp__184 = (float )*llvm_cbe_tmp__183;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__184, *(int*)(&llvm_cbe_tmp__184));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__184, 0x0p0))) {
    goto llvm_cbe_tmp__232;
  } else {
    llvm_cbe_tmp__186__PHI_TEMPORARY = (double )0x0p0;   /* for PHI node */
    goto llvm_cbe_tmp__231;
  }

llvm_cbe_tmp__232:
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fpext float %%16 to double, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_548_count);
  llvm_cbe_tmp__185 = (double )((double )llvm_cbe_tmp__184);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__185, *(long long*)(&llvm_cbe_tmp__185));
  llvm_cbe_tmp__186__PHI_TEMPORARY = (double )llvm_cbe_tmp__185;   /* for PHI node */
  goto llvm_cbe_tmp__231;

llvm_cbe_tmp__233:
  goto llvm_cbe_tmp__227;

llvm_cbe_tmp__226:
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = getelementptr inbounds [19 x float]* @ldata, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_555_count);
  llvm_cbe_tmp__188 = (float *)(&ldata[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge11) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'ldata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load float* %%25, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_556_count);
  llvm_cbe_tmp__189 = (float )*llvm_cbe_tmp__188;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__189, *(int*)(&llvm_cbe_tmp__189));
if (AESL_DEBUG_TRACE)
printf("\n  %%27 = fpext float %%26 to double, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_557_count);
  llvm_cbe_tmp__190 = (double )((double )llvm_cbe_tmp__189);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__190, *(long long*)(&llvm_cbe_tmp__190));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__190, -0x1.4adf4bc6a7efap86))) {
    goto llvm_cbe_tmp__234;
  } else {
    goto llvm_cbe_tmp__233;
  }

llvm_cbe_tmp__235:
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = phi double [ %%34, %%33 ], [ 0.000000e+00, %%29 ], !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_566_count);
  llvm_cbe_tmp__194 = (double )llvm_cbe_tmp__194__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %lf",llvm_cbe_tmp__194);
printf("\n = %lf",llvm_cbe_tmp__193);
printf("\n = %lf",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = fptrunc double %%36 to float, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_567_count);
  llvm_cbe_tmp__195 = (float )((float )llvm_cbe_tmp__194);
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__195, *(int*)(&llvm_cbe_tmp__195));

#ifdef AESL_BC_SIM
  assert(((signed long long )llvm_cbe_storemerge11) < 19 && "Write access out of array 'work_delta_y' bound?");

#endif
if (AESL_DEBUG_TRACE)
printf("\n  store float %%37, float* %%30, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_568_count);
  *llvm_cbe_tmp__191 = llvm_cbe_tmp__195;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__195);
  goto llvm_cbe_tmp__233;

llvm_cbe_tmp__234:
if (AESL_DEBUG_TRACE)
printf("\n  %%30 = getelementptr inbounds [19 x float]* @work_delta_y, i64 0, i64 %%storemerge11, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_560_count);
  llvm_cbe_tmp__191 = (float *)(&work_delta_y[(((signed long long )llvm_cbe_storemerge11))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge11 = 0x%I64X",((signed long long )llvm_cbe_storemerge11));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge11) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_delta_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load float* %%30, align 4, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_561_count);
  llvm_cbe_tmp__192 = (float )*llvm_cbe_tmp__191;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__192, *(int*)(&llvm_cbe_tmp__192));
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__192, 0x0p0))) {
    goto llvm_cbe_tmp__236;
  } else {
    llvm_cbe_tmp__194__PHI_TEMPORARY = (double )0x0p0;   /* for PHI node */
    goto llvm_cbe_tmp__235;
  }

llvm_cbe_tmp__236:
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = fpext float %%31 to double, !dbg !36 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_564_count);
  llvm_cbe_tmp__193 = (double )((double )llvm_cbe_tmp__192);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__193, *(long long*)(&llvm_cbe_tmp__193));
  llvm_cbe_tmp__194__PHI_TEMPORARY = (double )llvm_cbe_tmp__193;   /* for PHI node */
  goto llvm_cbe_tmp__235;

  } while (1); /* end of syntactic loop '.lr.ph13' */
llvm_cbe__2e__crit_edge14:
if (AESL_DEBUG_TRACE)
printf("\n  %%42 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_595_count);
  llvm_cbe_tmp__197 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__197);
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_597_count);
  llvm_cbe_tmp__198 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__198);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%43, %%45, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__197&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__198&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__237;
  } else {
    goto llvm_cbe_tmp__238;
  }

llvm_cbe_tmp__237:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_E, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_delta_y, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Adelta_x, i64 0, i64 0), i64 %%1) nounwind, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_600_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_E[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_delta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Adelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__175);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__175);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_601_count);
  llvm_cbe_tmp__199 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__199);
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = tail call float @vec_norm_inf(float* getelementptr inbounds ([19 x float]* @work_Adelta_x, i64 0, i64 0), i64 %%47) nounwind, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_602_count);
  llvm_cbe_tmp__200 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Adelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__199);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__199);
printf("\nReturn  = %f",llvm_cbe_tmp__200);
}
  llvm_cbe_storemerge1__PHI_TEMPORARY = (float )llvm_cbe_tmp__200;   /* for PHI node */
  goto llvm_cbe_tmp__239;

llvm_cbe_tmp__238:
if (AESL_DEBUG_TRACE)
printf("\n  %%50 = tail call float @vec_norm_inf(float* getelementptr inbounds ([19 x float]* @work_delta_y, i64 0, i64 0), i64 %%1) nounwind, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_605_count);
  llvm_cbe_tmp__201 = (float ) /*tail*/ vec_norm_inf((float *)((&work_delta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__175);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__175);
printf("\nReturn  = %f",llvm_cbe_tmp__201);
}
  llvm_cbe_storemerge1__PHI_TEMPORARY = (float )llvm_cbe_tmp__201;   /* for PHI node */
  goto llvm_cbe_tmp__239;

llvm_cbe_tmp__239:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi float [ %%50, %%49 ], [ %%48, %%46  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (float )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = %f",llvm_cbe_storemerge1);
printf("\n = %f",llvm_cbe_tmp__201);
printf("\n = %f",llvm_cbe_tmp__200);
}
  if ((llvm_fcmp_ogt(llvm_cbe_storemerge1, 0x1.4484cp-100))) {
    goto llvm_cbe__2e_preheader;
  } else {
    goto llvm_cbe_tmp__240;
  }

llvm_cbe__2e_preheader:
if (AESL_DEBUG_TRACE)
printf("\n  %%53 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_633_count);
  llvm_cbe_tmp__202 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__202);
  if ((((signed long long )llvm_cbe_tmp__202) > ((signed long long )0ull))) {
    llvm_cbe_storemerge28__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__203__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  do {     /* Syntactic loop '.lr.ph' to make GCC happy */
llvm_cbe__2e_lr_2e_ph:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge28 = phi i64 [ %%70, %%.lr.ph ], [ 0, %%.preheader  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_storemerge28_count);
  llvm_cbe_storemerge28 = (unsigned long long )llvm_cbe_storemerge28__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge28 = 0x%I64X",llvm_cbe_storemerge28);
printf("\n = 0x%I64X",llvm_cbe_tmp__216);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%55 = phi float [ %%69, %%.lr.ph ], [ 0.000000e+00, %%.preheader  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_636_count);
  llvm_cbe_tmp__203 = (float )llvm_cbe_tmp__203__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__203);
printf("\n = %f",llvm_cbe_tmp__215);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%56 = getelementptr inbounds [19 x float]* @udata, i64 0, i64 %%storemerge28, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_637_count);
  llvm_cbe_tmp__204 = (float *)(&udata[(((signed long long )llvm_cbe_storemerge28))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge28 = 0x%I64X",((signed long long )llvm_cbe_storemerge28));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge28) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'udata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%57 = load float* %%56, align 4, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_638_count);
  llvm_cbe_tmp__205 = (float )*llvm_cbe_tmp__204;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__205, *(int*)(&llvm_cbe_tmp__205));
if (AESL_DEBUG_TRACE)
printf("\n  %%58 = getelementptr inbounds [19 x float]* @work_delta_y, i64 0, i64 %%storemerge28, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_639_count);
  llvm_cbe_tmp__206 = (float *)(&work_delta_y[(((signed long long )llvm_cbe_storemerge28))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge28 = 0x%I64X",((signed long long )llvm_cbe_storemerge28));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge28) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_delta_y' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%59 = load float* %%58, align 4, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_640_count);
  llvm_cbe_tmp__207 = (float )*llvm_cbe_tmp__206;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__207, *(int*)(&llvm_cbe_tmp__207));
if (AESL_DEBUG_TRACE)
printf("\n  %%61 = select i1 %%60, float %%59, float 0.000000e+00, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_642_count);
  llvm_cbe_tmp__208 = (float )(((llvm_fcmp_ogt(llvm_cbe_tmp__207, 0x0p0))) ? ((float )llvm_cbe_tmp__207) : ((float )0x0p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__208, *(int*)(&llvm_cbe_tmp__208));
if (AESL_DEBUG_TRACE)
printf("\n  %%62 = fmul float %%57, %%61, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_643_count);
  llvm_cbe_tmp__209 = (float )((float )(llvm_cbe_tmp__205 * llvm_cbe_tmp__208));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__209, *(int*)(&llvm_cbe_tmp__209));
if (AESL_DEBUG_TRACE)
printf("\n  %%63 = getelementptr inbounds [19 x float]* @ldata, i64 0, i64 %%storemerge28, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_644_count);
  llvm_cbe_tmp__210 = (float *)(&ldata[(((signed long long )llvm_cbe_storemerge28))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge28 = 0x%I64X",((signed long long )llvm_cbe_storemerge28));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge28) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'ldata' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%64 = load float* %%63, align 4, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_645_count);
  llvm_cbe_tmp__211 = (float )*llvm_cbe_tmp__210;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__211, *(int*)(&llvm_cbe_tmp__211));
if (AESL_DEBUG_TRACE)
printf("\n  %%66 = select i1 %%65, float %%59, float 0.000000e+00, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_647_count);
  llvm_cbe_tmp__212 = (float )(((llvm_fcmp_olt(llvm_cbe_tmp__207, 0x0p0))) ? ((float )llvm_cbe_tmp__207) : ((float )0x0p0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__212, *(int*)(&llvm_cbe_tmp__212));
if (AESL_DEBUG_TRACE)
printf("\n  %%67 = fmul float %%64, %%66, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_648_count);
  llvm_cbe_tmp__213 = (float )((float )(llvm_cbe_tmp__211 * llvm_cbe_tmp__212));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__213, *(int*)(&llvm_cbe_tmp__213));
if (AESL_DEBUG_TRACE)
printf("\n  %%68 = fadd float %%62, %%67, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_649_count);
  llvm_cbe_tmp__214 = (float )((float )(llvm_cbe_tmp__209 + llvm_cbe_tmp__213));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__214, *(int*)(&llvm_cbe_tmp__214));
if (AESL_DEBUG_TRACE)
printf("\n  %%69 = fadd float %%55, %%68, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_650_count);
  llvm_cbe_tmp__215 = (float )((float )(llvm_cbe_tmp__203 + llvm_cbe_tmp__214));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__215, *(int*)(&llvm_cbe_tmp__215));
if (AESL_DEBUG_TRACE)
printf("\n  %%70 = add nsw i64 %%storemerge28, 1, !dbg !37 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_654_count);
  llvm_cbe_tmp__216 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge28&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__216&18446744073709551615ull)));
  if ((((signed long long )llvm_cbe_tmp__216) < ((signed long long )llvm_cbe_tmp__202))) {
    llvm_cbe_storemerge28__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__216;   /* for PHI node */
    llvm_cbe_tmp__203__PHI_TEMPORARY = (float )llvm_cbe_tmp__215;   /* for PHI node */
    goto llvm_cbe__2e_lr_2e_ph;
  } else {
    llvm_cbe__2e_lcssa__PHI_TEMPORARY = (float )llvm_cbe_tmp__215;   /* for PHI node */
    goto llvm_cbe__2e__crit_edge;
  }

  } while (1); /* end of syntactic loop '.lr.ph' */
llvm_cbe__2e__crit_edge:
if (AESL_DEBUG_TRACE)
printf("\n  %%.lcssa = phi float [ 0.000000e+00, %%.preheader ], [ %%69, %%.lr.ph  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe__2e_lcssa_count);
  llvm_cbe__2e_lcssa = (float )llvm_cbe__2e_lcssa__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n.lcssa = %f",llvm_cbe__2e_lcssa);
printf("\n = %f",0x0p0);
printf("\n = %f",llvm_cbe_tmp__215);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%72 = fmul float %%storemerge1, %%eps_prim_inf, !dbg !34 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_678_count);
  llvm_cbe_tmp__217 = (float )((float )(llvm_cbe_storemerge1 * llvm_cbe_eps_prim_inf));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__217, *(int*)(&llvm_cbe_tmp__217));
  if ((llvm_fcmp_olt(llvm_cbe__2e_lcssa, llvm_cbe_tmp__217))) {
    goto llvm_cbe_tmp__241;
  } else {
    goto llvm_cbe_tmp__240;
  }

llvm_cbe_tmp__241:
if (AESL_DEBUG_TRACE)
printf("\n  %%75 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_681_count);
  llvm_cbe_tmp__218 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__218);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_tpose_vec(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64* getelementptr inbounds ([43 x i64]* @Adata_i, i64 0, i64 0), i64 %%75, i64 %%53, float* getelementptr inbounds ([19 x float]* @work_delta_y, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Atdelta_y, i64 0, i64 0), i64 0, i64 0) nounwind, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_682_count);
   /*tail*/ mat_tpose_vec((float *)((&Adata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__218, llvm_cbe_tmp__202, (float *)((&work_delta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Atdelta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0ull, 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__218);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__202);
printf("\nArgument  = 0x%I64X",0ull);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%76 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_683_count);
  llvm_cbe_tmp__219 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__219);
if (AESL_DEBUG_TRACE)
printf("\n  %%78 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_685_count);
  llvm_cbe_tmp__220 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__220);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond6 = and i1 %%77, %%79, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_or_2e_cond6_count);
  llvm_cbe_or_2e_cond6 = (bool )((((llvm_cbe_tmp__219&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__220&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond6 = 0x%X\n", llvm_cbe_or_2e_cond6);
  if (llvm_cbe_or_2e_cond6) {
    goto llvm_cbe_tmp__242;
  } else {
    goto llvm_cbe_tmp__243;
  }

llvm_cbe_tmp__242:
if (AESL_DEBUG_TRACE)
printf("\n  %%81 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_688_count);
  llvm_cbe_tmp__221 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__221);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Atdelta_y, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Atdelta_y, i64 0, i64 0), i64 %%81) nounwind, !dbg !38 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_689_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Atdelta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Atdelta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__221);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__221);
}
  goto llvm_cbe_tmp__243;

llvm_cbe_tmp__243:
if (AESL_DEBUG_TRACE)
printf("\n  %%83 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_691_count);
  llvm_cbe_tmp__222 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__222);
if (AESL_DEBUG_TRACE)
printf("\n  %%84 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_Atdelta_y, i64 0, i64 0), i64 %%83) nounwind, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_692_count);
  llvm_cbe_tmp__223 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Atdelta_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__222);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__222);
printf("\nReturn  = %f",llvm_cbe_tmp__223);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%86 = zext i1 %%85 to i64, !dbg !35 for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_694_count);
  llvm_cbe_tmp__224 = (unsigned long long )((unsigned long long )(bool )(llvm_fcmp_olt(llvm_cbe_tmp__223, llvm_cbe_tmp__217))&1U);
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__224);
  llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__224;   /* for PHI node */
  goto llvm_cbe_tmp__244;

llvm_cbe_tmp__240:
  llvm_cbe_storemerge3__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__244;

llvm_cbe_tmp__244:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge3 = phi i64 [ %%86, %%82 ], [ 0, %%87  for 0x%I64xth hint within @is_primal_infeasible  --> \n", ++aesl_llvm_cbe_storemerge3_count);
  llvm_cbe_storemerge3 = (unsigned long long )llvm_cbe_storemerge3__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge3 = 0x%I64X",llvm_cbe_storemerge3);
printf("\n = 0x%I64X",llvm_cbe_tmp__224);
printf("\n = 0x%I64X",0ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @is_primal_infeasible}\n");
  return llvm_cbe_storemerge3;
}


signed long long is_dual_infeasible(float llvm_cbe_eps_dual_inf) {
  static  unsigned long long aesl_llvm_cbe_698_count = 0;
  static  unsigned long long aesl_llvm_cbe_699_count = 0;
  static  unsigned long long aesl_llvm_cbe_700_count = 0;
  static  unsigned long long aesl_llvm_cbe_701_count = 0;
  static  unsigned long long aesl_llvm_cbe_702_count = 0;
  static  unsigned long long aesl_llvm_cbe_703_count = 0;
  unsigned long long llvm_cbe_tmp__245;
  static  unsigned long long aesl_llvm_cbe_704_count = 0;
  static  unsigned long long aesl_llvm_cbe_705_count = 0;
  unsigned long long llvm_cbe_tmp__246;
  static  unsigned long long aesl_llvm_cbe_706_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_707_count = 0;
  static  unsigned long long aesl_llvm_cbe_708_count = 0;
  unsigned long long llvm_cbe_tmp__247;
  static  unsigned long long aesl_llvm_cbe_709_count = 0;
  float llvm_cbe_tmp__248;
  static  unsigned long long aesl_llvm_cbe_710_count = 0;
  static  unsigned long long aesl_llvm_cbe_711_count = 0;
  static  unsigned long long aesl_llvm_cbe_712_count = 0;
  static  unsigned long long aesl_llvm_cbe_713_count = 0;
  static  unsigned long long aesl_llvm_cbe_714_count = 0;
  static  unsigned long long aesl_llvm_cbe_715_count = 0;
  static  unsigned long long aesl_llvm_cbe_716_count = 0;
  float llvm_cbe_tmp__249;
  static  unsigned long long aesl_llvm_cbe_717_count = 0;
  static  unsigned long long aesl_llvm_cbe_718_count = 0;
  static  unsigned long long aesl_llvm_cbe_719_count = 0;
  unsigned long long llvm_cbe_tmp__250;
  static  unsigned long long aesl_llvm_cbe_720_count = 0;
  float llvm_cbe_tmp__251;
  static  unsigned long long aesl_llvm_cbe_721_count = 0;
  static  unsigned long long aesl_llvm_cbe_722_count = 0;
  static  unsigned long long aesl_llvm_cbe_723_count = 0;
  static  unsigned long long aesl_llvm_cbe_724_count = 0;
  static  unsigned long long aesl_llvm_cbe_725_count = 0;
  static  unsigned long long aesl_llvm_cbe_726_count = 0;
  static  unsigned long long aesl_llvm_cbe_727_count = 0;
  static  unsigned long long aesl_llvm_cbe_728_count = 0;
  static  unsigned long long aesl_llvm_cbe_729_count = 0;
  float llvm_cbe_tmp__252;
  float llvm_cbe_tmp__252__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_730_count = 0;
  static  unsigned long long aesl_llvm_cbe_731_count = 0;
  static  unsigned long long aesl_llvm_cbe_732_count = 0;
  static  unsigned long long aesl_llvm_cbe_733_count = 0;
  static  unsigned long long aesl_llvm_cbe_734_count = 0;
  unsigned long long llvm_cbe_tmp__253;
  static  unsigned long long aesl_llvm_cbe_735_count = 0;
  float llvm_cbe_tmp__254;
  static  unsigned long long aesl_llvm_cbe_736_count = 0;
  float llvm_cbe_tmp__255;
  static  unsigned long long aesl_llvm_cbe_737_count = 0;
  float llvm_cbe_tmp__256;
  static  unsigned long long aesl_llvm_cbe_738_count = 0;
  static  unsigned long long aesl_llvm_cbe_739_count = 0;
  static  unsigned long long aesl_llvm_cbe_740_count = 0;
  unsigned long long llvm_cbe_tmp__257;
  static  unsigned long long aesl_llvm_cbe_741_count = 0;
  unsigned long long llvm_cbe_tmp__258;
  static  unsigned long long aesl_llvm_cbe_742_count = 0;
  static  unsigned long long aesl_llvm_cbe_743_count = 0;
  unsigned long long llvm_cbe_tmp__259;
  static  unsigned long long aesl_llvm_cbe_744_count = 0;
  unsigned long long llvm_cbe_tmp__260;
  static  unsigned long long aesl_llvm_cbe_745_count = 0;
  static  unsigned long long aesl_llvm_cbe_746_count = 0;
  unsigned long long llvm_cbe_tmp__261;
  static  unsigned long long aesl_llvm_cbe_747_count = 0;
  static  unsigned long long aesl_llvm_cbe_748_count = 0;
  unsigned long long llvm_cbe_tmp__262;
  static  unsigned long long aesl_llvm_cbe_749_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond4_count = 0;
  bool llvm_cbe_or_2e_cond4;
  static  unsigned long long aesl_llvm_cbe_750_count = 0;
  static  unsigned long long aesl_llvm_cbe_751_count = 0;
  unsigned long long llvm_cbe_tmp__263;
  static  unsigned long long aesl_llvm_cbe_752_count = 0;
  static  unsigned long long aesl_llvm_cbe_753_count = 0;
  static  unsigned long long aesl_llvm_cbe_754_count = 0;
  unsigned long long llvm_cbe_tmp__264;
  static  unsigned long long aesl_llvm_cbe_755_count = 0;
  float llvm_cbe_tmp__265;
  static  unsigned long long aesl_llvm_cbe_756_count = 0;
  static  unsigned long long aesl_llvm_cbe_757_count = 0;
  static  unsigned long long aesl_llvm_cbe_758_count = 0;
  unsigned long long llvm_cbe_tmp__266;
  static  unsigned long long aesl_llvm_cbe_759_count = 0;
  unsigned long long llvm_cbe_tmp__267;
  static  unsigned long long aesl_llvm_cbe_760_count = 0;
  static  unsigned long long aesl_llvm_cbe_761_count = 0;
  unsigned long long llvm_cbe_tmp__268;
  static  unsigned long long aesl_llvm_cbe_762_count = 0;
  static  unsigned long long aesl_llvm_cbe_763_count = 0;
  unsigned long long llvm_cbe_tmp__269;
  static  unsigned long long aesl_llvm_cbe_764_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond6_count = 0;
  bool llvm_cbe_or_2e_cond6;
  static  unsigned long long aesl_llvm_cbe_765_count = 0;
  static  unsigned long long aesl_llvm_cbe_766_count = 0;
  unsigned long long llvm_cbe_tmp__270;
  static  unsigned long long aesl_llvm_cbe_767_count = 0;
  static  unsigned long long aesl_llvm_cbe_768_count = 0;
  static  unsigned long long aesl_llvm_cbe_769_count = 0;
  static  unsigned long long aesl_llvm_cbe_770_count = 0;
  unsigned long long llvm_cbe_tmp__271;
  static  unsigned long long aesl_llvm_cbe_771_count = 0;
  float llvm_cbe_tmp__272;
  static  unsigned long long aesl_llvm_cbe_772_count = 0;
  float llvm_cbe_tmp__273;
  static  unsigned long long aesl_llvm_cbe_773_count = 0;
  float llvm_cbe_tmp__274;
  static  unsigned long long aesl_llvm_cbe_774_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge1_count = 0;
  unsigned long long llvm_cbe_storemerge1;
  unsigned long long llvm_cbe_storemerge1__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_775_count = 0;
  static  unsigned long long aesl_llvm_cbe_776_count = 0;
  static  unsigned long long aesl_llvm_cbe_777_count = 0;
  static  unsigned long long aesl_llvm_cbe_778_count = 0;
  static  unsigned long long aesl_llvm_cbe_779_count = 0;
  static  unsigned long long aesl_llvm_cbe_780_count = 0;
  static  unsigned long long aesl_llvm_cbe_781_count = 0;
  static  unsigned long long aesl_llvm_cbe_782_count = 0;
  static  unsigned long long aesl_llvm_cbe_783_count = 0;
  float *llvm_cbe_tmp__275;
  static  unsigned long long aesl_llvm_cbe_784_count = 0;
  float llvm_cbe_tmp__276;
  static  unsigned long long aesl_llvm_cbe_785_count = 0;
  double llvm_cbe_tmp__277;
  static  unsigned long long aesl_llvm_cbe_786_count = 0;
  static  unsigned long long aesl_llvm_cbe_787_count = 0;
  static  unsigned long long aesl_llvm_cbe_788_count = 0;
  float *llvm_cbe_tmp__278;
  static  unsigned long long aesl_llvm_cbe_789_count = 0;
  float llvm_cbe_tmp__279;
  static  unsigned long long aesl_llvm_cbe_790_count = 0;
  static  unsigned long long aesl_llvm_cbe_791_count = 0;
  static  unsigned long long aesl_llvm_cbe_792_count = 0;
  float *llvm_cbe_tmp__280;
  static  unsigned long long aesl_llvm_cbe_793_count = 0;
  float llvm_cbe_tmp__281;
  static  unsigned long long aesl_llvm_cbe_794_count = 0;
  double llvm_cbe_tmp__282;
  static  unsigned long long aesl_llvm_cbe_795_count = 0;
  static  unsigned long long aesl_llvm_cbe_796_count = 0;
  static  unsigned long long aesl_llvm_cbe_797_count = 0;
  float *llvm_cbe_tmp__283;
  static  unsigned long long aesl_llvm_cbe_798_count = 0;
  float llvm_cbe_tmp__284;
  static  unsigned long long aesl_llvm_cbe_799_count = 0;
  static  unsigned long long aesl_llvm_cbe_800_count = 0;
  static  unsigned long long aesl_llvm_cbe_801_count = 0;
  static  unsigned long long aesl_llvm_cbe_802_count = 0;
  unsigned long long llvm_cbe_tmp__285;
  static  unsigned long long aesl_llvm_cbe_803_count = 0;
  static  unsigned long long aesl_llvm_cbe_804_count = 0;
  static  unsigned long long aesl_llvm_cbe_805_count = 0;
  static  unsigned long long aesl_llvm_cbe_806_count = 0;
  static  unsigned long long aesl_llvm_cbe_807_count = 0;
  unsigned long long llvm_cbe_tmp__286;
  unsigned long long llvm_cbe_tmp__286__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_808_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @is_dual_infeasible\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_703_count);
  llvm_cbe_tmp__245 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__245);
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_705_count);
  llvm_cbe_tmp__246 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__246);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = and i1 %%2, %%4, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )((((llvm_cbe_tmp__245&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__246&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__287;
  } else {
    goto llvm_cbe_tmp__288;
  }

llvm_cbe_tmp__287:
if (AESL_DEBUG_TRACE)
printf("\n  %%6 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_708_count);
  llvm_cbe_tmp__247 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__247);
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = tail call float @vec_scaled_norm_inf(float* getelementptr inbounds ([15 x float]* @scaling_D, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), i64 %%6) nounwind, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_709_count);
  llvm_cbe_tmp__248 = (float ) /*tail*/ vec_scaled_norm_inf((float *)((&scaling_D[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__247);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__247);
printf("\nReturn  = %f",llvm_cbe_tmp__248);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = load float* getelementptr inbounds (%%struct.OSQPScaling* @scaling, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_716_count);
  llvm_cbe_tmp__249 = (float )*((&scaling.field0));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__249, *(int*)(&llvm_cbe_tmp__249));
  llvm_cbe_tmp__252__PHI_TEMPORARY = (float )llvm_cbe_tmp__248;   /* for PHI node */
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__249;   /* for PHI node */
  goto llvm_cbe_tmp__289;

llvm_cbe_tmp__288:
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_719_count);
  llvm_cbe_tmp__250 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__250);
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), i64 %%10) nounwind, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_720_count);
  llvm_cbe_tmp__251 = (float ) /*tail*/ vec_norm_inf((float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__250);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__250);
printf("\nReturn  = %f",llvm_cbe_tmp__251);
}
  llvm_cbe_tmp__252__PHI_TEMPORARY = (float )llvm_cbe_tmp__251;   /* for PHI node */
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )0x1p0;   /* for PHI node */
  goto llvm_cbe_tmp__289;

llvm_cbe_tmp__289:
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = phi float [ %%11, %%9 ], [ %%7, %%5  for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_729_count);
  llvm_cbe_tmp__252 = (float )llvm_cbe_tmp__252__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__252);
printf("\n = %f",llvm_cbe_tmp__251);
printf("\n = %f",llvm_cbe_tmp__248);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ 1.000000e+00, %%9 ], [ %%8, %%5  for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",0x1p0);
printf("\n = %f",llvm_cbe_tmp__249);
}
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__252, 0x1.4484cp-100))) {
    goto llvm_cbe_tmp__290;
  } else {
    goto llvm_cbe_tmp__291;
  }

llvm_cbe_tmp__290:
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_734_count);
  llvm_cbe_tmp__253 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__253);
if (AESL_DEBUG_TRACE)
printf("\n  %%17 = tail call float @vec_prod(float* getelementptr inbounds ([15 x float]* @qdata, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), i64 %%16) nounwind, !dbg !34 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_735_count);
  llvm_cbe_tmp__254 = (float ) /*tail*/ vec_prod((float *)((&qdata[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__253);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__253);
printf("\nReturn  = %f",llvm_cbe_tmp__254);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = fmul float %%storemerge, %%eps_dual_inf, !dbg !34 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_736_count);
  llvm_cbe_tmp__255 = (float )((float )(llvm_cbe_storemerge * llvm_cbe_eps_dual_inf));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__255, *(int*)(&llvm_cbe_tmp__255));
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = fmul float %%18, %%13, !dbg !34 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_737_count);
  llvm_cbe_tmp__256 = (float )((float )(llvm_cbe_tmp__255 * llvm_cbe_tmp__252));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__256, *(int*)(&llvm_cbe_tmp__256));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__254, llvm_cbe_tmp__256))) {
    goto llvm_cbe_tmp__292;
  } else {
    goto llvm_cbe_tmp__293;
  }

llvm_cbe_tmp__292:
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_740_count);
  llvm_cbe_tmp__257 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__257);
if (AESL_DEBUG_TRACE)
printf("\n  %%23 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_741_count);
  llvm_cbe_tmp__258 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__258);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_vec(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%22, i64 %%23, float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Pdelta_x, i64 0, i64 0), i64 0) nounwind, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_742_count);
   /*tail*/ mat_vec((float *)((&Pdata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__257, llvm_cbe_tmp__258, (float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Pdelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__257);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__258);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_743_count);
  llvm_cbe_tmp__259 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__259);
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_744_count);
  llvm_cbe_tmp__260 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__260);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_tpose_vec(float* getelementptr inbounds ([12 x float]* @Pdata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Pdata_p, i64 0, i64 0), i64* getelementptr inbounds ([12 x i64]* @Pdata_i, i64 0, i64 0), i64 %%24, i64 %%25, float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Pdelta_x, i64 0, i64 0), i64 1, i64 1) nounwind, !dbg !36 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_745_count);
   /*tail*/ mat_tpose_vec((float *)((&Pdata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__259, llvm_cbe_tmp__260, (float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Pdelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 1ull, 1ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__259);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__260);
printf("\nArgument  = 0x%I64X",1ull);
printf("\nArgument  = 0x%I64X",1ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_746_count);
  llvm_cbe_tmp__261 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__261);
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_748_count);
  llvm_cbe_tmp__262 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__262);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond4 = and i1 %%27, %%29, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_or_2e_cond4_count);
  llvm_cbe_or_2e_cond4 = (bool )((((llvm_cbe_tmp__261&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__262&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond4 = 0x%X\n", llvm_cbe_or_2e_cond4);
  if (llvm_cbe_or_2e_cond4) {
    goto llvm_cbe_tmp__294;
  } else {
    goto llvm_cbe_tmp__295;
  }

llvm_cbe_tmp__294:
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_751_count);
  llvm_cbe_tmp__263 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__263);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([15 x float]* @scaling_Dinv, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Pdelta_x, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @work_Pdelta_x, i64 0, i64 0), i64 %%31) nounwind, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_752_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Dinv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Pdelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Pdelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__263);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__263);
}
  goto llvm_cbe_tmp__295;

llvm_cbe_tmp__295:
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_754_count);
  llvm_cbe_tmp__264 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__264);
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = tail call float @vec_norm_inf(float* getelementptr inbounds ([15 x float]* @work_Pdelta_x, i64 0, i64 0), i64 %%33) nounwind, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_755_count);
  llvm_cbe_tmp__265 = (float ) /*tail*/ vec_norm_inf((float *)((&work_Pdelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__264);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__264);
printf("\nReturn  = %f",llvm_cbe_tmp__265);
}
  if ((llvm_fcmp_olt(llvm_cbe_tmp__265, llvm_cbe_tmp__256))) {
    goto llvm_cbe_tmp__296;
  } else {
    goto llvm_cbe_tmp__293;
  }

llvm_cbe_tmp__296:
if (AESL_DEBUG_TRACE)
printf("\n  %%37 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_758_count);
  llvm_cbe_tmp__266 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__266);
if (AESL_DEBUG_TRACE)
printf("\n  %%38 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_759_count);
  llvm_cbe_tmp__267 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__267);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @mat_vec(float* getelementptr inbounds ([43 x float]* @Adata_x, i64 0, i64 0), i64* getelementptr inbounds ([16 x i64]* @Adata_p, i64 0, i64 0), i64* getelementptr inbounds ([43 x i64]* @Adata_i, i64 0, i64 0), i64 %%37, i64 %%38, float* getelementptr inbounds ([15 x float]* @work_delta_x, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Adelta_x, i64 0, i64 0), i64 0) nounwind, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_760_count);
   /*tail*/ mat_vec((float *)((&Adata_x[(((signed long long )0ull))
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
])), llvm_cbe_tmp__266, llvm_cbe_tmp__267, (float *)((&work_delta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_Adelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__266);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__267);
printf("\nArgument  = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%39 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_761_count);
  llvm_cbe_tmp__268 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__268);
if (AESL_DEBUG_TRACE)
printf("\n  %%41 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 13), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_763_count);
  llvm_cbe_tmp__269 = (unsigned long long )*((&settings.field13));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__269);
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond6 = and i1 %%40, %%42, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_or_2e_cond6_count);
  llvm_cbe_or_2e_cond6 = (bool )((((llvm_cbe_tmp__268&18446744073709551615ULL) != (0ull&18446744073709551615ULL)) & ((llvm_cbe_tmp__269&18446744073709551615ULL) == (0ull&18446744073709551615ULL)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond6 = 0x%X\n", llvm_cbe_or_2e_cond6);
  if (llvm_cbe_or_2e_cond6) {
    goto llvm_cbe_tmp__297;
  } else {
    goto llvm_cbe_tmp__298;
  }

llvm_cbe_tmp__297:
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_766_count);
  llvm_cbe_tmp__270 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__270);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_ew_prod(float* getelementptr inbounds ([19 x float]* @scaling_Einv, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Adelta_x, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_Adelta_x, i64 0, i64 0), i64 %%44) nounwind, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_767_count);
   /*tail*/ vec_ew_prod((float *)((&scaling_Einv[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Adelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&work_Adelta_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__270);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__270);
}
  goto llvm_cbe_tmp__298;

llvm_cbe_tmp__298:
if (AESL_DEBUG_TRACE)
printf("\n  %%46 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_770_count);
  llvm_cbe_tmp__271 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__271);
if (AESL_DEBUG_TRACE)
printf("\n  %%47 = fmul float %%13, %%eps_dual_inf, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_771_count);
  llvm_cbe_tmp__272 = (float )((float )(llvm_cbe_tmp__252 * llvm_cbe_eps_dual_inf));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__272, *(int*)(&llvm_cbe_tmp__272));
if (AESL_DEBUG_TRACE)
printf("\n  %%48 = fsub float -0.000000e+00, %%eps_dual_inf, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_772_count);
  llvm_cbe_tmp__273 = (float )((float )(-(llvm_cbe_eps_dual_inf)));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__273, *(int*)(&llvm_cbe_tmp__273));
if (AESL_DEBUG_TRACE)
printf("\n  %%49 = fmul float %%13, %%48, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_773_count);
  llvm_cbe_tmp__274 = (float )((float )(llvm_cbe_tmp__252 * llvm_cbe_tmp__273));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__274, *(int*)(&llvm_cbe_tmp__274));
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__299;

  do {     /* Syntactic loop '' to make GCC happy */
llvm_cbe_tmp__299:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge1 = phi i64 [ 0, %%45 ], [ %%72, %%71  for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_storemerge1_count);
  llvm_cbe_storemerge1 = (unsigned long long )llvm_cbe_storemerge1__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",llvm_cbe_storemerge1);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__285);
}
  if ((((signed long long )llvm_cbe_storemerge1) < ((signed long long )llvm_cbe_tmp__271))) {
    goto llvm_cbe_tmp__300;
  } else {
    llvm_cbe_tmp__286__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
    goto llvm_cbe__2e_loopexit;
  }

llvm_cbe_tmp__301:
if (AESL_DEBUG_TRACE)
printf("\n  %%72 = add nsw i64 %%storemerge1, 1, !dbg !37 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_802_count);
  llvm_cbe_tmp__285 = (unsigned long long )((unsigned long long )(llvm_cbe_storemerge1&18446744073709551615ull)) + ((unsigned long long )(1ull&18446744073709551615ull));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", ((unsigned long long )(llvm_cbe_tmp__285&18446744073709551615ull)));
  llvm_cbe_storemerge1__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__285;   /* for PHI node */
  goto llvm_cbe_tmp__299;

llvm_cbe_tmp__302:
if (AESL_DEBUG_TRACE)
printf("\n  %%62 = getelementptr inbounds [19 x float]* @ldata, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_792_count);
  llvm_cbe_tmp__280 = (float *)(&ldata[(((signed long long )llvm_cbe_storemerge1))
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
printf("\n  %%63 = load float* %%62, align 4, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_793_count);
  llvm_cbe_tmp__281 = (float )*llvm_cbe_tmp__280;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__281, *(int*)(&llvm_cbe_tmp__281));
if (AESL_DEBUG_TRACE)
printf("\n  %%64 = fpext float %%63 to double, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_794_count);
  llvm_cbe_tmp__282 = (double )((double )llvm_cbe_tmp__281);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__282, *(long long*)(&llvm_cbe_tmp__282));
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__282, -0x1.4adf4bc6a7efap86))) {
    goto llvm_cbe_tmp__303;
  } else {
    goto llvm_cbe_tmp__301;
  }

llvm_cbe_tmp__300:
if (AESL_DEBUG_TRACE)
printf("\n  %%53 = getelementptr inbounds [19 x float]* @udata, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_783_count);
  llvm_cbe_tmp__275 = (float *)(&udata[(((signed long long )llvm_cbe_storemerge1))
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
printf("\n  %%54 = load float* %%53, align 4, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_784_count);
  llvm_cbe_tmp__276 = (float )*llvm_cbe_tmp__275;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__276, *(int*)(&llvm_cbe_tmp__276));
if (AESL_DEBUG_TRACE)
printf("\n  %%55 = fpext float %%54 to double, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_785_count);
  llvm_cbe_tmp__277 = (double )((double )llvm_cbe_tmp__276);
if (AESL_DEBUG_TRACE)
printf("\n = %lf,  0x%llx\n", llvm_cbe_tmp__277, *(long long*)(&llvm_cbe_tmp__277));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__277, 0x1.4adf4bc6a7efap86))) {
    goto llvm_cbe_tmp__304;
  } else {
    goto llvm_cbe_tmp__302;
  }

llvm_cbe_tmp__304:
if (AESL_DEBUG_TRACE)
printf("\n  %%58 = getelementptr inbounds [19 x float]* @work_Adelta_x, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_788_count);
  llvm_cbe_tmp__278 = (float *)(&work_Adelta_x[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_Adelta_x' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%59 = load float* %%58, align 4, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_789_count);
  llvm_cbe_tmp__279 = (float )*llvm_cbe_tmp__278;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__279, *(int*)(&llvm_cbe_tmp__279));
  if ((llvm_fcmp_ogt(llvm_cbe_tmp__279, llvm_cbe_tmp__272))) {
    goto llvm_cbe_tmp__305;
  } else {
    goto llvm_cbe_tmp__302;
  }

llvm_cbe_tmp__303:
if (AESL_DEBUG_TRACE)
printf("\n  %%67 = getelementptr inbounds [19 x float]* @work_Adelta_x, i64 0, i64 %%storemerge1, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_797_count);
  llvm_cbe_tmp__283 = (float *)(&work_Adelta_x[(((signed long long )llvm_cbe_storemerge1))
#ifdef AESL_BC_SIM
 % 19
#endif
]);
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge1 = 0x%I64X",((signed long long )llvm_cbe_storemerge1));
}

#ifdef AESL_BC_SIM
  if (!(((signed long long )llvm_cbe_storemerge1) < 19)) fprintf(stderr, "%s:%d: warning: Read access out of array 'work_Adelta_x' bound?\n", __FILE__, __LINE__);

#endif
if (AESL_DEBUG_TRACE)
printf("\n  %%68 = load float* %%67, align 4, !dbg !35 for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_798_count);
  llvm_cbe_tmp__284 = (float )*llvm_cbe_tmp__283;
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__284, *(int*)(&llvm_cbe_tmp__284));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__284, llvm_cbe_tmp__274))) {
    goto llvm_cbe_tmp__305;
  } else {
    goto llvm_cbe_tmp__301;
  }

  } while (1); /* end of syntactic loop '' */
llvm_cbe_tmp__305:
  llvm_cbe_tmp__286__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe__2e_loopexit;

llvm_cbe_tmp__293:
  goto llvm_cbe_tmp__291;

llvm_cbe_tmp__291:
  llvm_cbe_tmp__286__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe__2e_loopexit;

llvm_cbe__2e_loopexit:
if (AESL_DEBUG_TRACE)
printf("\n  %%75 = phi i64 [ 0, %%74 ], [ 0, %%70 ], [ 1, %%50  for 0x%I64xth hint within @is_dual_infeasible  --> \n", ++aesl_llvm_cbe_807_count);
  llvm_cbe_tmp__286 = (unsigned long long )llvm_cbe_tmp__286__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",llvm_cbe_tmp__286);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",1ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @is_dual_infeasible}\n");
  return llvm_cbe_tmp__286;
}


void store_solution(void) {
  static  unsigned long long aesl_llvm_cbe_809_count = 0;
  unsigned long long llvm_cbe_tmp__306;
  static  unsigned long long aesl_llvm_cbe_810_count = 0;
  static  unsigned long long aesl_llvm_cbe_811_count = 0;
  unsigned long long llvm_cbe_tmp__307;
  static  unsigned long long aesl_llvm_cbe_812_count = 0;
  static  unsigned long long aesl_llvm_cbe_813_count = 0;
  unsigned long long llvm_cbe_tmp__308;
  static  unsigned long long aesl_llvm_cbe_814_count = 0;
  static  unsigned long long aesl_llvm_cbe_815_count = 0;
  unsigned long long llvm_cbe_tmp__309;
  static  unsigned long long aesl_llvm_cbe_816_count = 0;
  static  unsigned long long aesl_llvm_cbe_817_count = 0;
  static  unsigned long long aesl_llvm_cbe_818_count = 0;
  unsigned long long llvm_cbe_tmp__310;
  static  unsigned long long aesl_llvm_cbe_819_count = 0;
  static  unsigned long long aesl_llvm_cbe_820_count = 0;
  static  unsigned long long aesl_llvm_cbe_821_count = 0;
  unsigned long long llvm_cbe_tmp__311;
  static  unsigned long long aesl_llvm_cbe_822_count = 0;
  static  unsigned long long aesl_llvm_cbe_823_count = 0;
  unsigned long long llvm_cbe_tmp__312;
  static  unsigned long long aesl_llvm_cbe_824_count = 0;
  static  unsigned long long aesl_llvm_cbe_825_count = 0;
  static  unsigned long long aesl_llvm_cbe_826_count = 0;
  static  unsigned long long aesl_llvm_cbe_827_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @store_solution\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load i64* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 2), align 8, !dbg !34 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_809_count);
  llvm_cbe_tmp__306 = (unsigned long long )*((&info.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__306);
  switch (((unsigned long long )(llvm_cbe_tmp__306&18446744073709551615ull))) {
  default:
    goto llvm_cbe_tmp__313;
;
  case ((unsigned long long )(18446744073709551613ull&18446744073709551615ull)):
    goto llvm_cbe_tmp__314;
  case ((unsigned long long )(18446744073709551612ull&18446744073709551615ull)):
    goto llvm_cbe_tmp__314;
  case ((unsigned long long )(18446744073709551609ull&18446744073709551615ull)):
    goto llvm_cbe_tmp__314;
  case ((unsigned long long )(4ull&18446744073709551615ull)):
    goto llvm_cbe_tmp__314;
  case ((unsigned long long )(3ull&18446744073709551615ull)):
    goto llvm_cbe_tmp__314;
  }
llvm_cbe_tmp__313:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !34 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_811_count);
  llvm_cbe_tmp__307 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__307);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0), float* getelementptr inbounds ([15 x float]* @xsolution, i64 0, i64 0), i64 %%3) nounwind, !dbg !34 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_812_count);
   /*tail*/ prea_vec_copy((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&xsolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), llvm_cbe_tmp__307);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__307);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !34 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_813_count);
  llvm_cbe_tmp__308 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__308);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @prea_vec_copy(float* getelementptr inbounds ([19 x float]* @work_y, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @ysolution, i64 0, i64 0), i64 %%4) nounwind, !dbg !34 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_814_count);
   /*tail*/ prea_vec_copy((float *)((&work_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), (float *)((&ysolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), llvm_cbe_tmp__308);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__308);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load i64* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 2), align 8, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_815_count);
  llvm_cbe_tmp__309 = (unsigned long long )*((&settings.field2));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__309);
  if (((llvm_cbe_tmp__309&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__315;
  } else {
    goto llvm_cbe_tmp__316;
  }

llvm_cbe_tmp__316:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = tail call i64 @unscale_solution() nounwind, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_818_count);
   /*tail*/ unscale_solution();
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__310);
}
  goto llvm_cbe_tmp__315;

llvm_cbe_tmp__315:
  goto llvm_cbe_tmp__317;

llvm_cbe_tmp__314:
if (AESL_DEBUG_TRACE)
printf("\n  %%11 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_821_count);
  llvm_cbe_tmp__311 = (unsigned long long )*((&data.field0));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__311);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([15 x float]* @xsolution, i64 0, i64 0), float 0x41DFF00000000000, i64 %%11) nounwind, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_822_count);
   /*tail*/ vec_set_scalar((float *)((&xsolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), 0x1.ffp30, llvm_cbe_tmp__311);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.ffp30);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__311);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_823_count);
  llvm_cbe_tmp__312 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__312);
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @vec_set_scalar(float* getelementptr inbounds ([19 x float]* @ysolution, i64 0, i64 0), float 0x41DFF00000000000, i64 %%12) nounwind, !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_824_count);
   /*tail*/ vec_set_scalar((float *)((&ysolution[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])), 0x1.ffp30, llvm_cbe_tmp__312);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f",0x1.ffp30);
printf("\nArgument  = 0x%I64X",llvm_cbe_tmp__312);
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @cold_start(), !dbg !35 for 0x%I64xth hint within @store_solution  --> \n", ++aesl_llvm_cbe_825_count);
   /*tail*/ cold_start();
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__317;

llvm_cbe_tmp__317:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @store_solution}\n");
  return;
}


void update_info(signed long long llvm_cbe_iter, signed long long llvm_cbe_compute_objective, signed long long llvm_cbe_polish) {
  static  unsigned long long aesl_llvm_cbe_828_count = 0;
  static  unsigned long long aesl_llvm_cbe_829_count = 0;
  static  unsigned long long aesl_llvm_cbe_830_count = 0;
  static  unsigned long long aesl_llvm_cbe_831_count = 0;
  static  unsigned long long aesl_llvm_cbe_832_count = 0;
  static  unsigned long long aesl_llvm_cbe_833_count = 0;
  static  unsigned long long aesl_llvm_cbe_834_count = 0;
  static  unsigned long long aesl_llvm_cbe_835_count = 0;
  static  unsigned long long aesl_llvm_cbe_836_count = 0;
  float llvm_cbe_tmp__318;
  static  unsigned long long aesl_llvm_cbe_837_count = 0;
  static  unsigned long long aesl_llvm_cbe_838_count = 0;
  static  unsigned long long aesl_llvm_cbe_839_count = 0;
  unsigned long long llvm_cbe_tmp__319;
  static  unsigned long long aesl_llvm_cbe_840_count = 0;
  static  unsigned long long aesl_llvm_cbe_841_count = 0;
  static  unsigned long long aesl_llvm_cbe_842_count = 0;
  float llvm_cbe_tmp__320;
  static  unsigned long long aesl_llvm_cbe_843_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  float llvm_cbe_storemerge;
  float llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_844_count = 0;
  static  unsigned long long aesl_llvm_cbe_845_count = 0;
  float llvm_cbe_tmp__321;
  static  unsigned long long aesl_llvm_cbe_846_count = 0;
  static  unsigned long long aesl_llvm_cbe_847_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_info\n");
if (AESL_DEBUG_TRACE)
printf("\n  store i64 %%iter, i64* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 0), align 8, !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_833_count);
  *((&info.field0)) = llvm_cbe_iter;
if (AESL_DEBUG_TRACE)
printf("\niter = 0x%I64X\n", llvm_cbe_iter);
  if (((llvm_cbe_compute_objective&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__322;
  } else {
    goto llvm_cbe_tmp__323;
  }

llvm_cbe_tmp__323:
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = tail call float @compute_obj_val(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0)), !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_836_count);
  llvm_cbe_tmp__318 = (float ) /*tail*/ compute_obj_val((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = %f",llvm_cbe_tmp__318);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%3, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 3), align 8, !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_837_count);
  *((&info.field3)) = llvm_cbe_tmp__318;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__318);
  goto llvm_cbe_tmp__322;

llvm_cbe_tmp__322:
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_839_count);
  llvm_cbe_tmp__319 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__319);
  if (((llvm_cbe_tmp__319&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_storemerge__PHI_TEMPORARY = (float )0x0p0;   /* for PHI node */
    goto llvm_cbe_tmp__324;
  } else {
    goto llvm_cbe_tmp__325;
  }

llvm_cbe_tmp__325:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = tail call float @compute_pri_res(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_z, i64 0, i64 0)), !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_842_count);
  llvm_cbe_tmp__320 = (float ) /*tail*/ compute_pri_res((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_z[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = %f",llvm_cbe_tmp__320);
}
  llvm_cbe_storemerge__PHI_TEMPORARY = (float )llvm_cbe_tmp__320;   /* for PHI node */
  goto llvm_cbe_tmp__324;

llvm_cbe_tmp__324:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi float [ %%8, %%7 ], [ 0.000000e+00, %%4  for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (float )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = %f",llvm_cbe_storemerge);
printf("\n = %f",llvm_cbe_tmp__320);
printf("\n = %f",0x0p0);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%storemerge, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 4), align 4, !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_844_count);
  *((&info.field4)) = llvm_cbe_storemerge;
if (AESL_DEBUG_TRACE)
printf("\nstoremerge = %f\n", llvm_cbe_storemerge);
if (AESL_DEBUG_TRACE)
printf("\n  %%10 = tail call float @compute_dua_res(float* getelementptr inbounds ([15 x float]* @work_x, i64 0, i64 0), float* getelementptr inbounds ([19 x float]* @work_y, i64 0, i64 0)), !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_845_count);
  llvm_cbe_tmp__321 = (float ) /*tail*/ compute_dua_res((float *)((&work_x[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 15
#endif
])), (float *)((&work_y[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
printf("\nReturn  = %f",llvm_cbe_tmp__321);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float %%10, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 5), align 8, !dbg !35 for 0x%I64xth hint within @update_info  --> \n", ++aesl_llvm_cbe_846_count);
  *((&info.field5)) = llvm_cbe_tmp__321;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", llvm_cbe_tmp__321);
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_info}\n");
  return;
}


void reset_info(l_struct_OC_OSQPInfo *llvm_cbe_info_ptr) {
  static  unsigned long long aesl_llvm_cbe_848_count = 0;
  static  unsigned long long aesl_llvm_cbe_849_count = 0;
  static  unsigned long long aesl_llvm_cbe_850_count = 0;
  static  unsigned long long aesl_llvm_cbe_851_count = 0;
const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @reset_info\n");
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* %%info_ptr, i64 -10), !dbg !34 for 0x%I64xth hint within @reset_info  --> \n", ++aesl_llvm_cbe_850_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)llvm_cbe_info_ptr, 18446744073709551606ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",18446744073709551606ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @reset_info}\n");
  return;
}


void update_status(l_struct_OC_OSQPInfo *llvm_cbe_info_ptr, signed long long llvm_cbe_status_val) {
  static  unsigned long long aesl_llvm_cbe_852_count = 0;
  static  unsigned long long aesl_llvm_cbe_853_count = 0;
  static  unsigned long long aesl_llvm_cbe_854_count = 0;
  static  unsigned long long aesl_llvm_cbe_855_count = 0;
  static  unsigned long long aesl_llvm_cbe_856_count = 0;
  static  unsigned long long aesl_llvm_cbe_857_count = 0;
  static  unsigned long long aesl_llvm_cbe_858_count = 0;
  static  unsigned long long aesl_llvm_cbe_859_count = 0;
  static  unsigned long long aesl_llvm_cbe_860_count = 0;
  static  unsigned long long aesl_llvm_cbe_861_count = 0;
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
  signed long long *llvm_cbe_tmp__326;
  static  unsigned long long aesl_llvm_cbe_877_count = 0;
  static  unsigned long long aesl_llvm_cbe_878_count = 0;
  static  unsigned long long aesl_llvm_cbe_879_count = 0;
  static  unsigned long long aesl_llvm_cbe_880_count = 0;
   char *llvm_cbe_tmp__327;
  static  unsigned long long aesl_llvm_cbe_881_count = 0;
  static  unsigned long long aesl_llvm_cbe_882_count = 0;
  static  unsigned long long aesl_llvm_cbe_883_count = 0;
  static  unsigned long long aesl_llvm_cbe_884_count = 0;
  static  unsigned long long aesl_llvm_cbe_885_count = 0;
   char *llvm_cbe_tmp__328;
  static  unsigned long long aesl_llvm_cbe_886_count = 0;
  static  unsigned long long aesl_llvm_cbe_887_count = 0;
  static  unsigned long long aesl_llvm_cbe_888_count = 0;
  static  unsigned long long aesl_llvm_cbe_889_count = 0;
  static  unsigned long long aesl_llvm_cbe_890_count = 0;
   char *llvm_cbe_tmp__329;
  static  unsigned long long aesl_llvm_cbe_891_count = 0;
  static  unsigned long long aesl_llvm_cbe_892_count = 0;
  static  unsigned long long aesl_llvm_cbe_893_count = 0;
  static  unsigned long long aesl_llvm_cbe_894_count = 0;
  static  unsigned long long aesl_llvm_cbe_895_count = 0;
   char *llvm_cbe_tmp__330;
  static  unsigned long long aesl_llvm_cbe_896_count = 0;
  static  unsigned long long aesl_llvm_cbe_897_count = 0;
  static  unsigned long long aesl_llvm_cbe_898_count = 0;
  static  unsigned long long aesl_llvm_cbe_899_count = 0;
  static  unsigned long long aesl_llvm_cbe_900_count = 0;
   char *llvm_cbe_tmp__331;
  static  unsigned long long aesl_llvm_cbe_901_count = 0;
  static  unsigned long long aesl_llvm_cbe_902_count = 0;
  static  unsigned long long aesl_llvm_cbe_903_count = 0;
  static  unsigned long long aesl_llvm_cbe_904_count = 0;
  static  unsigned long long aesl_llvm_cbe_905_count = 0;
   char *llvm_cbe_tmp__332;
  static  unsigned long long aesl_llvm_cbe_906_count = 0;
  static  unsigned long long aesl_llvm_cbe_907_count = 0;
  static  unsigned long long aesl_llvm_cbe_908_count = 0;
  static  unsigned long long aesl_llvm_cbe_909_count = 0;
  static  unsigned long long aesl_llvm_cbe_910_count = 0;
   char *llvm_cbe_tmp__333;
  static  unsigned long long aesl_llvm_cbe_911_count = 0;
  static  unsigned long long aesl_llvm_cbe_912_count = 0;
  static  unsigned long long aesl_llvm_cbe_913_count = 0;
  static  unsigned long long aesl_llvm_cbe_914_count = 0;
  static  unsigned long long aesl_llvm_cbe_915_count = 0;
   char *llvm_cbe_tmp__334;
  static  unsigned long long aesl_llvm_cbe_916_count = 0;
  static  unsigned long long aesl_llvm_cbe_917_count = 0;
  static  unsigned long long aesl_llvm_cbe_918_count = 0;
  static  unsigned long long aesl_llvm_cbe_919_count = 0;
  static  unsigned long long aesl_llvm_cbe_920_count = 0;
   char *llvm_cbe_tmp__335;
  static  unsigned long long aesl_llvm_cbe_921_count = 0;
  static  unsigned long long aesl_llvm_cbe_922_count = 0;
  static  unsigned long long aesl_llvm_cbe_923_count = 0;
  static  unsigned long long aesl_llvm_cbe_924_count = 0;
  static  unsigned long long aesl_llvm_cbe_925_count = 0;
   char *llvm_cbe_tmp__336;
  static  unsigned long long aesl_llvm_cbe_926_count = 0;
  static  unsigned long long aesl_llvm_cbe_927_count = 0;
  static  unsigned long long aesl_llvm_cbe_928_count = 0;
  static  unsigned long long aesl_llvm_cbe_929_count = 0;
  static  unsigned long long aesl_llvm_cbe_930_count = 0;
  static  unsigned long long aesl_llvm_cbe_931_count = 0;
  static  unsigned long long aesl_llvm_cbe_932_count = 0;
  static  unsigned long long aesl_llvm_cbe_933_count = 0;
  static  unsigned long long aesl_llvm_cbe_934_count = 0;
  static  unsigned long long aesl_llvm_cbe_935_count = 0;
  static  unsigned long long aesl_llvm_cbe_936_count = 0;
  static  unsigned long long aesl_llvm_cbe_937_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @update_status\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 2, !dbg !34 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_876_count);
  llvm_cbe_tmp__326 = (signed long long *)(&llvm_cbe_info_ptr->field2);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  store i64 %%status_val, i64* %%1, align 8, !dbg !34 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_877_count);
  *llvm_cbe_tmp__326 = llvm_cbe_status_val;
if (AESL_DEBUG_TRACE)
printf("\nstatus_val = 0x%I64X\n", llvm_cbe_status_val);
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (1ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__337;
  } else {
    goto llvm_cbe_tmp__338;
  }

llvm_cbe_tmp__337:
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !34 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_880_count);
  llvm_cbe_tmp__327 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%4, i8* getelementptr inbounds ([7 x i8]* @aesl_internal_.str, i64 0, i64 0)) nounwind, !dbg !34 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_881_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__327, ( char *)((&aesl_internal__OC_str[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 7
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__339;

llvm_cbe_tmp__338:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (2ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__340;
  } else {
    goto llvm_cbe_tmp__341;
  }

llvm_cbe_tmp__340:
if (AESL_DEBUG_TRACE)
printf("\n  %%8 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_885_count);
  llvm_cbe_tmp__328 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%8, i8* getelementptr inbounds ([18 x i8]* @aesl_internal_.str1, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_886_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__328, ( char *)((&aesl_internal__OC_str1[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 18
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__342;

llvm_cbe_tmp__341:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551613ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__343;
  } else {
    goto llvm_cbe_tmp__344;
  }

llvm_cbe_tmp__343:
if (AESL_DEBUG_TRACE)
printf("\n  %%12 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_890_count);
  llvm_cbe_tmp__329 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%12, i8* getelementptr inbounds ([18 x i8]* @aesl_internal_.str2, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_891_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__329, ( char *)((&aesl_internal__OC_str2[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 18
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__345;

llvm_cbe_tmp__344:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (3ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__346;
  } else {
    goto llvm_cbe_tmp__347;
  }

llvm_cbe_tmp__346:
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_895_count);
  llvm_cbe_tmp__330 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%16, i8* getelementptr inbounds ([29 x i8]* @aesl_internal_.str3, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_896_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__330, ( char *)((&aesl_internal__OC_str3[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 29
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__348;

llvm_cbe_tmp__347:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551606ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__349;
  } else {
    goto llvm_cbe_tmp__350;
  }

llvm_cbe_tmp__349:
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_900_count);
  llvm_cbe_tmp__331 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%20, i8* getelementptr inbounds ([9 x i8]* @aesl_internal_.str4, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_901_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__331, ( char *)((&aesl_internal__OC_str4[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 9
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__351;

llvm_cbe_tmp__350:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551612ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__352;
  } else {
    goto llvm_cbe_tmp__353;
  }

llvm_cbe_tmp__352:
if (AESL_DEBUG_TRACE)
printf("\n  %%24 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_905_count);
  llvm_cbe_tmp__332 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%24, i8* getelementptr inbounds ([16 x i8]* @aesl_internal_.str5, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_906_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__332, ( char *)((&aesl_internal__OC_str5[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 16
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__354;

llvm_cbe_tmp__353:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (4ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__355;
  } else {
    goto llvm_cbe_tmp__356;
  }

llvm_cbe_tmp__355:
if (AESL_DEBUG_TRACE)
printf("\n  %%28 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_910_count);
  llvm_cbe_tmp__333 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%28, i8* getelementptr inbounds ([27 x i8]* @aesl_internal_.str6, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_911_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__333, ( char *)((&aesl_internal__OC_str6[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 27
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__357;

llvm_cbe_tmp__356:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551614ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__358;
  } else {
    goto llvm_cbe_tmp__359;
  }

llvm_cbe_tmp__358:
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_915_count);
  llvm_cbe_tmp__334 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%32, i8* getelementptr inbounds ([27 x i8]* @aesl_internal_.str7, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_916_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__334, ( char *)((&aesl_internal__OC_str7[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 27
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__360;

llvm_cbe_tmp__359:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551611ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__361;
  } else {
    goto llvm_cbe_tmp__362;
  }

llvm_cbe_tmp__361:
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_920_count);
  llvm_cbe_tmp__335 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%36, i8* getelementptr inbounds ([12 x i8]* @aesl_internal_.str8, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_921_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__335, ( char *)((&aesl_internal__OC_str8[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 12
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__363;

llvm_cbe_tmp__362:
  if (((llvm_cbe_status_val&18446744073709551615ULL) == (18446744073709551609ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__364;
  } else {
    goto llvm_cbe_tmp__365;
  }

llvm_cbe_tmp__364:
if (AESL_DEBUG_TRACE)
printf("\n  %%40 = getelementptr inbounds %%struct.OSQPInfo* %%info_ptr, i64 0, i32 1, i64 0, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_925_count);
  llvm_cbe_tmp__336 = ( char *)(&llvm_cbe_info_ptr->field1[(((signed long long )0ull))]);
if (AESL_DEBUG_TRACE) {
}
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @c_strcpy(i8* %%40, i8* getelementptr inbounds ([19 x i8]* @aesl_internal_.str9, i64 0, i64 0)) nounwind, !dbg !35 for 0x%I64xth hint within @update_status  --> \n", ++aesl_llvm_cbe_926_count);
   /*tail*/ c_strcpy(( char *)llvm_cbe_tmp__336, ( char *)((&aesl_internal__OC_str9[(((signed long long )0ull))
#ifdef AESL_BC_SIM
 % 19
#endif
])));
if (AESL_DEBUG_TRACE) {
}
  goto llvm_cbe_tmp__365;

llvm_cbe_tmp__365:
  goto llvm_cbe_tmp__363;

llvm_cbe_tmp__363:
  goto llvm_cbe_tmp__360;

llvm_cbe_tmp__360:
  goto llvm_cbe_tmp__357;

llvm_cbe_tmp__357:
  goto llvm_cbe_tmp__354;

llvm_cbe_tmp__354:
  goto llvm_cbe_tmp__351;

llvm_cbe_tmp__351:
  goto llvm_cbe_tmp__348;

llvm_cbe_tmp__348:
  goto llvm_cbe_tmp__345;

llvm_cbe_tmp__345:
  goto llvm_cbe_tmp__342;

llvm_cbe_tmp__342:
  goto llvm_cbe_tmp__339;

llvm_cbe_tmp__339:
  if (AESL_DEBUG_TRACE)
      printf("\nEND @update_status}\n");
  return;
}


signed long long check_termination(signed long long llvm_cbe_approximate) {
  static  unsigned long long aesl_llvm_cbe_938_count = 0;
  static  unsigned long long aesl_llvm_cbe_939_count = 0;
  static  unsigned long long aesl_llvm_cbe_940_count = 0;
  static  unsigned long long aesl_llvm_cbe_941_count = 0;
  static  unsigned long long aesl_llvm_cbe_942_count = 0;
  static  unsigned long long aesl_llvm_cbe_943_count = 0;
  static  unsigned long long aesl_llvm_cbe_944_count = 0;
  static  unsigned long long aesl_llvm_cbe_945_count = 0;
  static  unsigned long long aesl_llvm_cbe_946_count = 0;
  static  unsigned long long aesl_llvm_cbe_947_count = 0;
  static  unsigned long long aesl_llvm_cbe_948_count = 0;
  static  unsigned long long aesl_llvm_cbe_949_count = 0;
  static  unsigned long long aesl_llvm_cbe_950_count = 0;
  static  unsigned long long aesl_llvm_cbe_951_count = 0;
  static  unsigned long long aesl_llvm_cbe_952_count = 0;
  static  unsigned long long aesl_llvm_cbe_953_count = 0;
  float llvm_cbe_tmp__366;
  static  unsigned long long aesl_llvm_cbe_954_count = 0;
  static  unsigned long long aesl_llvm_cbe_955_count = 0;
  static  unsigned long long aesl_llvm_cbe_956_count = 0;
  static  unsigned long long aesl_llvm_cbe_957_count = 0;
  static  unsigned long long aesl_llvm_cbe_958_count = 0;
  float llvm_cbe_tmp__367;
  static  unsigned long long aesl_llvm_cbe_959_count = 0;
  static  unsigned long long aesl_llvm_cbe_960_count = 0;
  static  unsigned long long aesl_llvm_cbe_961_count = 0;
  static  unsigned long long aesl_llvm_cbe_962_count = 0;
  static  unsigned long long aesl_llvm_cbe_963_count = 0;
  float llvm_cbe_tmp__368;
  static  unsigned long long aesl_llvm_cbe_964_count = 0;
  static  unsigned long long aesl_llvm_cbe_965_count = 0;
  static  unsigned long long aesl_llvm_cbe_966_count = 0;
  static  unsigned long long aesl_llvm_cbe_967_count = 0;
  float llvm_cbe_tmp__369;
  static  unsigned long long aesl_llvm_cbe_968_count = 0;
  static  unsigned long long aesl_llvm_cbe_969_count = 0;
  static  unsigned long long aesl_llvm_cbe_970_count = 0;
  static  unsigned long long aesl_llvm_cbe_971_count = 0;
  float llvm_cbe_tmp__370;
  static  unsigned long long aesl_llvm_cbe_972_count = 0;
  static  unsigned long long aesl_llvm_cbe_973_count = 0;
  float llvm_cbe_tmp__371;
  static  unsigned long long aesl_llvm_cbe_974_count = 0;
  static  unsigned long long aesl_llvm_cbe_or_2e_cond_count = 0;
  bool llvm_cbe_or_2e_cond;
  static  unsigned long long aesl_llvm_cbe_975_count = 0;
  static  unsigned long long aesl_llvm_cbe_976_count = 0;
  static  unsigned long long aesl_llvm_cbe_977_count = 0;
  static  unsigned long long aesl_llvm_cbe_978_count = 0;
  static  unsigned long long aesl_llvm_cbe_979_count = 0;
  static  unsigned long long aesl_llvm_cbe_980_count = 0;
  static  unsigned long long aesl_llvm_cbe_981_count = 0;
  float llvm_cbe_tmp__372;
  static  unsigned long long aesl_llvm_cbe_982_count = 0;
  static  unsigned long long aesl_llvm_cbe_983_count = 0;
  static  unsigned long long aesl_llvm_cbe_984_count = 0;
  static  unsigned long long aesl_llvm_cbe_985_count = 0;
  static  unsigned long long aesl_llvm_cbe_986_count = 0;
  float llvm_cbe_tmp__373;
  static  unsigned long long aesl_llvm_cbe_987_count = 0;
  static  unsigned long long aesl_llvm_cbe_988_count = 0;
  static  unsigned long long aesl_llvm_cbe_989_count = 0;
  static  unsigned long long aesl_llvm_cbe_990_count = 0;
  static  unsigned long long aesl_llvm_cbe_991_count = 0;
  float llvm_cbe_tmp__374;
  static  unsigned long long aesl_llvm_cbe_992_count = 0;
  static  unsigned long long aesl_llvm_cbe_993_count = 0;
  static  unsigned long long aesl_llvm_cbe_994_count = 0;
  static  unsigned long long aesl_llvm_cbe_995_count = 0;
  float llvm_cbe_tmp__375;
  static  unsigned long long aesl_llvm_cbe_996_count = 0;
  static  unsigned long long aesl_llvm_cbe_997_count = 0;
  static  unsigned long long aesl_llvm_cbe_998_count = 0;
  static  unsigned long long aesl_llvm_cbe_999_count = 0;
  static  unsigned long long aesl_llvm_cbe_1000_count = 0;
  float llvm_cbe_tmp__376;
  float llvm_cbe_tmp__376__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1001_count = 0;
  float llvm_cbe_tmp__377;
  float llvm_cbe_tmp__377__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1002_count = 0;
  float llvm_cbe_tmp__378;
  float llvm_cbe_tmp__378__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1003_count = 0;
  float llvm_cbe_tmp__379;
  float llvm_cbe_tmp__379__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1004_count = 0;
  unsigned long long llvm_cbe_tmp__380;
  static  unsigned long long aesl_llvm_cbe_1005_count = 0;
  static  unsigned long long aesl_llvm_cbe_1006_count = 0;
  static  unsigned long long aesl_llvm_cbe_1007_count = 0;
  float llvm_cbe_tmp__381;
  static  unsigned long long aesl_llvm_cbe_1008_count = 0;
  static  unsigned long long aesl_llvm_cbe_1009_count = 0;
  float llvm_cbe_tmp__382;
  static  unsigned long long aesl_llvm_cbe_1010_count = 0;
  static  unsigned long long aesl_llvm_cbe_1011_count = 0;
  static  unsigned long long aesl_llvm_cbe_1012_count = 0;
  static  unsigned long long aesl_llvm_cbe_1013_count = 0;
  unsigned long long llvm_cbe_tmp__383;
  static  unsigned long long aesl_llvm_cbe_1014_count = 0;
  static  unsigned long long aesl_llvm_cbe_1015_count = 0;
  static  unsigned long long aesl_llvm_cbe_1016_count = 0;
  static  unsigned long long aesl_llvm_cbe_1017_count = 0;
  unsigned long long llvm_cbe_tmp__384;
  unsigned long long llvm_cbe_tmp__384__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1018_count = 0;
  unsigned long long llvm_cbe_tmp__385;
  unsigned long long llvm_cbe_tmp__385__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1019_count = 0;
  float llvm_cbe_tmp__386;
  static  unsigned long long aesl_llvm_cbe_1020_count = 0;
  static  unsigned long long aesl_llvm_cbe_1021_count = 0;
  float llvm_cbe_tmp__387;
  static  unsigned long long aesl_llvm_cbe_1022_count = 0;
  static  unsigned long long aesl_llvm_cbe_1023_count = 0;
  static  unsigned long long aesl_llvm_cbe_1024_count = 0;
  static  unsigned long long aesl_llvm_cbe_1025_count = 0;
  unsigned long long llvm_cbe_tmp__388;
  static  unsigned long long aesl_llvm_cbe_1026_count = 0;
  static  unsigned long long aesl_llvm_cbe_1027_count = 0;
  static  unsigned long long aesl_llvm_cbe_1028_count = 0;
  static  unsigned long long aesl_llvm_cbe_1029_count = 0;
  static  unsigned long long aesl_llvm_cbe_1030_count = 0;
  static  unsigned long long aesl_llvm_cbe_1031_count = 0;
  static  unsigned long long aesl_llvm_cbe_1032_count = 0;
  static  unsigned long long aesl_llvm_cbe_1033_count = 0;
  static  unsigned long long aesl_llvm_cbe_1034_count = 0;
  static  unsigned long long aesl_llvm_cbe_1035_count = 0;
  static  unsigned long long aesl_llvm_cbe_1036_count = 0;
  static  unsigned long long aesl_llvm_cbe_1037_count = 0;
  static  unsigned long long aesl_llvm_cbe_1038_count = 0;
  static  unsigned long long aesl_llvm_cbe_1039_count = 0;
  unsigned long long llvm_cbe_tmp__389;
  unsigned long long llvm_cbe_tmp__389__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1040_count = 0;
  static  unsigned long long aesl_llvm_cbe_1041_count = 0;
  static  unsigned long long aesl_llvm_cbe_1042_count = 0;
  static  unsigned long long aesl_llvm_cbe_1043_count = 0;
  static  unsigned long long aesl_llvm_cbe_1044_count = 0;
  static  unsigned long long aesl_llvm_cbe_1045_count = 0;
  static  unsigned long long aesl_llvm_cbe_1046_count = 0;
  static  unsigned long long aesl_llvm_cbe_1047_count = 0;
  static  unsigned long long aesl_llvm_cbe_1048_count = 0;
  static  unsigned long long aesl_llvm_cbe_1049_count = 0;
  static  unsigned long long aesl_llvm_cbe_1050_count = 0;
  static  unsigned long long aesl_llvm_cbe_1051_count = 0;
  static  unsigned long long aesl_llvm_cbe_1052_count = 0;
  static  unsigned long long aesl_llvm_cbe_1053_count = 0;
  static  unsigned long long aesl_llvm_cbe_1054_count = 0;
  static  unsigned long long aesl_llvm_cbe_1055_count = 0;
  static  unsigned long long aesl_llvm_cbe_1056_count = 0;
  static  unsigned long long aesl_llvm_cbe_1057_count = 0;
  static  unsigned long long aesl_llvm_cbe_1058_count = 0;
  static  unsigned long long aesl_llvm_cbe_1059_count = 0;
  static  unsigned long long aesl_llvm_cbe_1060_count = 0;
  static  unsigned long long aesl_llvm_cbe_1061_count = 0;
  static  unsigned long long aesl_llvm_cbe_storemerge_count = 0;
  unsigned long long llvm_cbe_storemerge;
  unsigned long long llvm_cbe_storemerge__PHI_TEMPORARY;
  static  unsigned long long aesl_llvm_cbe_1062_count = 0;

const char* AESL_DEBUG_TRACE = getenv("DEBUG_TRACE");
if (AESL_DEBUG_TRACE)
printf("\n\{ BEGIN @check_termination\n");
if (AESL_DEBUG_TRACE)
printf("\n  %%1 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 7), align 8, !dbg !36 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_953_count);
  llvm_cbe_tmp__366 = (float )*((&settings.field7));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__366, *(int*)(&llvm_cbe_tmp__366));
if (AESL_DEBUG_TRACE)
printf("\n  %%2 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 8), align 4, !dbg !36 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_958_count);
  llvm_cbe_tmp__367 = (float )*((&settings.field8));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__367, *(int*)(&llvm_cbe_tmp__367));
if (AESL_DEBUG_TRACE)
printf("\n  %%3 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 9), align 8, !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_963_count);
  llvm_cbe_tmp__368 = (float )*((&settings.field9));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__368, *(int*)(&llvm_cbe_tmp__368));
if (AESL_DEBUG_TRACE)
printf("\n  %%4 = load float* getelementptr inbounds (%%struct.OSQPSettings* @settings, i64 0, i32 10), align 4, !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_967_count);
  llvm_cbe_tmp__369 = (float )*((&settings.field10));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__369, *(int*)(&llvm_cbe_tmp__369));
if (AESL_DEBUG_TRACE)
printf("\n  %%5 = load float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 4), align 4, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_971_count);
  llvm_cbe_tmp__370 = (float )*((&info.field4));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__370, *(int*)(&llvm_cbe_tmp__370));
if (AESL_DEBUG_TRACE)
printf("\n  %%7 = load float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 5), align 8, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_973_count);
  llvm_cbe_tmp__371 = (float )*((&info.field5));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__371, *(int*)(&llvm_cbe_tmp__371));
if (AESL_DEBUG_TRACE)
printf("\n  %%or.cond = or i1 %%6, %%8, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_or_2e_cond_count);
  llvm_cbe_or_2e_cond = (bool )(((llvm_fcmp_ogt(llvm_cbe_tmp__370, 0x1.93e594p99)) | (llvm_fcmp_ogt(llvm_cbe_tmp__371, 0x1.93e594p99)))&1);
if (AESL_DEBUG_TRACE)
printf("\nor.cond = 0x%X\n", llvm_cbe_or_2e_cond);
  if (llvm_cbe_or_2e_cond) {
    goto llvm_cbe_tmp__390;
  } else {
    goto llvm_cbe_tmp__391;
  }

llvm_cbe_tmp__390:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 -7), !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_976_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 18446744073709551609ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",18446744073709551609ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x41DFF00000000000, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 3), align 8, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_977_count);
  *((&info.field3)) = 0x1.ffp30;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.ffp30);
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
  goto llvm_cbe_tmp__392;

llvm_cbe_tmp__391:
  if (((llvm_cbe_approximate&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_tmp__376__PHI_TEMPORARY = (float )llvm_cbe_tmp__367;   /* for PHI node */
    llvm_cbe_tmp__377__PHI_TEMPORARY = (float )llvm_cbe_tmp__366;   /* for PHI node */
    llvm_cbe_tmp__378__PHI_TEMPORARY = (float )llvm_cbe_tmp__369;   /* for PHI node */
    llvm_cbe_tmp__379__PHI_TEMPORARY = (float )llvm_cbe_tmp__368;   /* for PHI node */
    goto llvm_cbe_tmp__393;
  } else {
    goto llvm_cbe_tmp__394;
  }

llvm_cbe_tmp__394:
if (AESL_DEBUG_TRACE)
printf("\n  %%13 = fmul float %%1, 1.000000e+01, !dbg !36 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_981_count);
  llvm_cbe_tmp__372 = (float )((float )(llvm_cbe_tmp__366 * 0x1.4p3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__372, *(int*)(&llvm_cbe_tmp__372));
if (AESL_DEBUG_TRACE)
printf("\n  %%14 = fmul float %%2, 1.000000e+01, !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_986_count);
  llvm_cbe_tmp__373 = (float )((float )(llvm_cbe_tmp__367 * 0x1.4p3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__373, *(int*)(&llvm_cbe_tmp__373));
if (AESL_DEBUG_TRACE)
printf("\n  %%15 = fmul float %%3, 1.000000e+01, !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_991_count);
  llvm_cbe_tmp__374 = (float )((float )(llvm_cbe_tmp__368 * 0x1.4p3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__374, *(int*)(&llvm_cbe_tmp__374));
if (AESL_DEBUG_TRACE)
printf("\n  %%16 = fmul float %%4, 1.000000e+01, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_995_count);
  llvm_cbe_tmp__375 = (float )((float )(llvm_cbe_tmp__369 * 0x1.4p3));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__375, *(int*)(&llvm_cbe_tmp__375));
  llvm_cbe_tmp__376__PHI_TEMPORARY = (float )llvm_cbe_tmp__373;   /* for PHI node */
  llvm_cbe_tmp__377__PHI_TEMPORARY = (float )llvm_cbe_tmp__372;   /* for PHI node */
  llvm_cbe_tmp__378__PHI_TEMPORARY = (float )llvm_cbe_tmp__375;   /* for PHI node */
  llvm_cbe_tmp__379__PHI_TEMPORARY = (float )llvm_cbe_tmp__374;   /* for PHI node */
  goto llvm_cbe_tmp__393;

llvm_cbe_tmp__393:
if (AESL_DEBUG_TRACE)
printf("\n  %%18 = phi float [ %%2, %%10 ], [ %%14, %%12  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1000_count);
  llvm_cbe_tmp__376 = (float )llvm_cbe_tmp__376__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__376);
printf("\n = %f",llvm_cbe_tmp__367);
printf("\n = %f",llvm_cbe_tmp__373);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%19 = phi float [ %%1, %%10 ], [ %%13, %%12  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1001_count);
  llvm_cbe_tmp__377 = (float )llvm_cbe_tmp__377__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__377);
printf("\n = %f",llvm_cbe_tmp__366);
printf("\n = %f",llvm_cbe_tmp__372);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%20 = phi float [ %%4, %%10 ], [ %%16, %%12  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1002_count);
  llvm_cbe_tmp__378 = (float )llvm_cbe_tmp__378__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__378);
printf("\n = %f",llvm_cbe_tmp__369);
printf("\n = %f",llvm_cbe_tmp__375);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%21 = phi float [ %%3, %%10 ], [ %%15, %%12  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1003_count);
  llvm_cbe_tmp__379 = (float )llvm_cbe_tmp__379__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = %f",llvm_cbe_tmp__379);
printf("\n = %f",llvm_cbe_tmp__368);
printf("\n = %f",llvm_cbe_tmp__374);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%22 = load i64* getelementptr inbounds (%%struct.OSQPData* @data, i64 0, i32 1), align 8, !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1004_count);
  llvm_cbe_tmp__380 = (unsigned long long )*((&data.field1));
if (AESL_DEBUG_TRACE)
printf("\n = 0x%I64X\n", llvm_cbe_tmp__380);
  if (((llvm_cbe_tmp__380&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_tmp__384__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__385__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
    goto llvm_cbe_tmp__395;
  } else {
    goto llvm_cbe_tmp__396;
  }

llvm_cbe_tmp__396:
if (AESL_DEBUG_TRACE)
printf("\n  %%25 = tail call float @compute_pri_tol(float %%19, float %%18), !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1007_count);
  llvm_cbe_tmp__381 = (float ) /*tail*/ compute_pri_tol(llvm_cbe_tmp__377, llvm_cbe_tmp__376);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__377, *(int*)(&llvm_cbe_tmp__377));
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__376, *(int*)(&llvm_cbe_tmp__376));
printf("\nReturn  = %f",llvm_cbe_tmp__381);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%26 = load float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 4), align 4, !dbg !39 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1009_count);
  llvm_cbe_tmp__382 = (float )*((&info.field4));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__382, *(int*)(&llvm_cbe_tmp__382));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__382, llvm_cbe_tmp__381))) {
    llvm_cbe_tmp__384__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    llvm_cbe_tmp__385__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
    goto llvm_cbe_tmp__395;
  } else {
    goto llvm_cbe_tmp__397;
  }

llvm_cbe_tmp__397:
if (AESL_DEBUG_TRACE)
printf("\n  %%29 = tail call i64 @is_primal_infeasible(float %%21), !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1013_count);
  llvm_cbe_tmp__383 = (unsigned long long ) /*tail*/ is_primal_infeasible(llvm_cbe_tmp__379);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__379, *(int*)(&llvm_cbe_tmp__379));
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__383);
}
  llvm_cbe_tmp__384__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__383;   /* for PHI node */
  llvm_cbe_tmp__385__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
  goto llvm_cbe_tmp__395;

llvm_cbe_tmp__395:
if (AESL_DEBUG_TRACE)
printf("\n  %%31 = phi i64 [ 0, %%17 ], [ %%29, %%28 ], [ 0, %%24  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1017_count);
  llvm_cbe_tmp__384 = (unsigned long long )llvm_cbe_tmp__384__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",llvm_cbe_tmp__384);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",llvm_cbe_tmp__383);
printf("\n = 0x%I64X",0ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%32 = phi i64 [ 1, %%17 ], [ 0, %%28 ], [ 1, %%24  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1018_count);
  llvm_cbe_tmp__385 = (unsigned long long )llvm_cbe_tmp__385__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",llvm_cbe_tmp__385);
printf("\n = 0x%I64X",1ull);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",1ull);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%33 = tail call float @compute_dua_tol(float %%19, float %%18), !dbg !37 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1019_count);
  llvm_cbe_tmp__386 = (float ) /*tail*/ compute_dua_tol(llvm_cbe_tmp__377, llvm_cbe_tmp__376);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__377, *(int*)(&llvm_cbe_tmp__377));
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__376, *(int*)(&llvm_cbe_tmp__376));
printf("\nReturn  = %f",llvm_cbe_tmp__386);
}
if (AESL_DEBUG_TRACE)
printf("\n  %%34 = load float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 5), align 8, !dbg !39 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1021_count);
  llvm_cbe_tmp__387 = (float )*((&info.field5));
if (AESL_DEBUG_TRACE)
printf("\n = %f,  0x%x\n", llvm_cbe_tmp__387, *(int*)(&llvm_cbe_tmp__387));
  if ((llvm_fcmp_olt(llvm_cbe_tmp__387, llvm_cbe_tmp__386))) {
    goto llvm_cbe_tmp__398;
  } else {
    goto llvm_cbe__2e_thread;
  }

llvm_cbe__2e_thread:
if (AESL_DEBUG_TRACE)
printf("\n  %%36 = tail call i64 @is_dual_infeasible(float %%20), !dbg !38 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1025_count);
  llvm_cbe_tmp__388 = (unsigned long long ) /*tail*/ is_dual_infeasible(llvm_cbe_tmp__378);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = %f,  0x%x",llvm_cbe_tmp__378, *(int*)(&llvm_cbe_tmp__378));
printf("\nReturn  = 0x%I64X",llvm_cbe_tmp__388);
}
  llvm_cbe_tmp__389__PHI_TEMPORARY = (unsigned long long )llvm_cbe_tmp__388;   /* for PHI node */
  goto llvm_cbe_tmp__399;

llvm_cbe_tmp__398:
  if (((llvm_cbe_tmp__385&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_tmp__389__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe_tmp__399;
  } else {
    goto llvm_cbe_tmp__400;
  }

llvm_cbe_tmp__400:
  if (((llvm_cbe_approximate&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__401;
  } else {
    goto llvm_cbe_tmp__402;
  }

llvm_cbe_tmp__402:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 2), !dbg !39 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1032_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 2ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",2ull);
}
  goto llvm_cbe_tmp__403;

llvm_cbe_tmp__401:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 1), !dbg !39 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1034_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 1ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",1ull);
}
  goto llvm_cbe_tmp__403;

llvm_cbe_tmp__403:
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
  goto llvm_cbe_tmp__392;

llvm_cbe_tmp__399:
if (AESL_DEBUG_TRACE)
printf("\n  %%44 = phi i64 [ %%36, %%.thread ], [ 0, %%37  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1039_count);
  llvm_cbe_tmp__389 = (unsigned long long )llvm_cbe_tmp__389__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\n = 0x%I64X",llvm_cbe_tmp__389);
printf("\n = 0x%I64X",llvm_cbe_tmp__388);
printf("\n = 0x%I64X",0ull);
}
  if (((llvm_cbe_tmp__384&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__404;
  } else {
    goto llvm_cbe_tmp__405;
  }

llvm_cbe_tmp__405:
  if (((llvm_cbe_approximate&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__406;
  } else {
    goto llvm_cbe_tmp__407;
  }

llvm_cbe_tmp__407:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 3), !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1043_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 3ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",3ull);
}
  goto llvm_cbe_tmp__408;

llvm_cbe_tmp__406:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 -3), !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1045_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 18446744073709551613ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",18446744073709551613ull);
}
  goto llvm_cbe_tmp__408;

llvm_cbe_tmp__408:
if (AESL_DEBUG_TRACE)
printf("\n  store float 0x46293E5940000000, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 3), align 8, !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1047_count);
  *((&info.field3)) = 0x1.93e594p99;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", 0x1.93e594p99);
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
  goto llvm_cbe_tmp__392;

llvm_cbe_tmp__404:
  if (((llvm_cbe_tmp__389&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )0ull;   /* for PHI node */
    goto llvm_cbe_tmp__392;
  } else {
    goto llvm_cbe_tmp__409;
  }

llvm_cbe_tmp__409:
  if (((llvm_cbe_approximate&18446744073709551615ULL) == (0ull&18446744073709551615ULL))) {
    goto llvm_cbe_tmp__410;
  } else {
    goto llvm_cbe_tmp__411;
  }

llvm_cbe_tmp__411:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 4), !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1054_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 4ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",4ull);
}
  goto llvm_cbe_tmp__412;

llvm_cbe_tmp__410:
if (AESL_DEBUG_TRACE)
printf("\n  tail call void @update_status(%%struct.OSQPInfo* @info, i64 -4), !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1056_count);
   /*tail*/ update_status((l_struct_OC_OSQPInfo *)(&info), 18446744073709551612ull);
if (AESL_DEBUG_TRACE) {
printf("\nArgument  = 0x%I64X",18446744073709551612ull);
}
  goto llvm_cbe_tmp__412;

llvm_cbe_tmp__412:
if (AESL_DEBUG_TRACE)
printf("\n  store float 0xC6293E5940000000, float* getelementptr inbounds (%%struct.OSQPInfo* @info, i64 0, i32 3), align 8, !dbg !40 for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_1058_count);
  *((&info.field3)) = -0x1.93e594p99;
if (AESL_DEBUG_TRACE)
printf("\n = %f\n", -0x1.93e594p99);
  llvm_cbe_storemerge__PHI_TEMPORARY = (unsigned long long )1ull;   /* for PHI node */
  goto llvm_cbe_tmp__392;

llvm_cbe_tmp__392:
if (AESL_DEBUG_TRACE)
printf("\n  %%storemerge = phi i64 [ 1, %%9 ], [ 1, %%42 ], [ 1, %%49 ], [ 0, %%50 ], [ 1, %%55  for 0x%I64xth hint within @check_termination  --> \n", ++aesl_llvm_cbe_storemerge_count);
  llvm_cbe_storemerge = (unsigned long long )llvm_cbe_storemerge__PHI_TEMPORARY;
if (AESL_DEBUG_TRACE) {
printf("\nstoremerge = 0x%I64X",llvm_cbe_storemerge);
printf("\n = 0x%I64X",1ull);
printf("\n = 0x%I64X",1ull);
printf("\n = 0x%I64X",1ull);
printf("\n = 0x%I64X",0ull);
printf("\n = 0x%I64X",1ull);
}
  if (AESL_DEBUG_TRACE)
      printf("\nEND @check_termination}\n");
  return llvm_cbe_storemerge;
}

