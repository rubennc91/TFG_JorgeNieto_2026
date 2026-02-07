# 1 "srcs/src/simulink_block.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 359 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "C:/Xilinx/Vitis_HLS/2020.2/common/technology/autopilot\\etc/autopilot_ssdm_op.h" 1
# 314 "C:/Xilinx/Vitis_HLS/2020.2/common/technology/autopilot\\etc/autopilot_ssdm_op.h"
    void _ssdm_op_IfRead() __attribute__ ((nothrow));
    void _ssdm_op_IfWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfNbRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfNbWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfCanRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfCanWrite() __attribute__ ((nothrow));


    void _ssdm_StreamRead() __attribute__ ((nothrow));
    void _ssdm_StreamWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamNbRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamNbWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamCanRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamCanWrite() __attribute__ ((nothrow));
    void _ssdm_op_ReadReq() __attribute__ ((nothrow));
    void _ssdm_op_Read() __attribute__ ((nothrow));
    void _ssdm_op_WriteReq() __attribute__ ((nothrow));
    void _ssdm_op_Write() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_NbReadReq() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_NbWriteReq() __attribute__ ((nothrow));




    void _ssdm_op_MemShiftRead() __attribute__ ((nothrow));

    void _ssdm_op_Wait() __attribute__ ((nothrow));
    void _ssdm_op_Poll() __attribute__ ((nothrow));

    void _ssdm_op_Return() __attribute__ ((nothrow));


    void _ssdm_op_SpecSynModule() __attribute__ ((nothrow));
    void _ssdm_op_SpecTopModule() __attribute__ ((nothrow));
    void _ssdm_op_SpecProcessDecl() __attribute__ ((nothrow));
    void _ssdm_op_SpecProcessDef() __attribute__ ((nothrow));
    void _ssdm_op_SpecPort() __attribute__ ((nothrow));
    void _ssdm_op_SpecConnection() __attribute__ ((nothrow));
    void _ssdm_op_SpecChannel() __attribute__ ((nothrow));
    void _ssdm_op_SpecSensitive() __attribute__ ((nothrow));
    void _ssdm_op_SpecModuleInst() __attribute__ ((nothrow));
    void _ssdm_op_SpecPortMap() __attribute__ ((nothrow));

    void _ssdm_op_SpecReset() __attribute__ ((nothrow));

    void _ssdm_op_SpecPlatform() __attribute__ ((nothrow));
    void _ssdm_op_SpecClockDomain() __attribute__ ((nothrow));
    void _ssdm_op_SpecPowerDomain() __attribute__ ((nothrow));

    int _ssdm_op_SpecRegionBegin() __attribute__ ((nothrow));
    int _ssdm_op_SpecRegionEnd() __attribute__ ((nothrow));

    void _ssdm_op_SpecLoopName() __attribute__ ((nothrow));

    void _ssdm_op_SpecLoopTripCount() __attribute__ ((nothrow));

    int _ssdm_op_SpecStateBegin() __attribute__ ((nothrow));
    int _ssdm_op_SpecStateEnd() __attribute__ ((nothrow));

    void _ssdm_op_SpecInterface() __attribute__ ((nothrow));

    void _ssdm_op_SpecPipeline() __attribute__ ((nothrow));
    void _ssdm_op_SpecDataflowPipeline() __attribute__ ((nothrow));


    void _ssdm_op_SpecLatency() __attribute__ ((nothrow));
    void _ssdm_op_SpecParallel() __attribute__ ((nothrow));
    void _ssdm_op_SpecProtocol() __attribute__ ((nothrow));
    void _ssdm_op_SpecOccurrence() __attribute__ ((nothrow));

    void _ssdm_op_SpecResource() __attribute__ ((nothrow));
    void _ssdm_op_SpecResourceLimit() __attribute__ ((nothrow));
    void _ssdm_op_SpecCHCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecFUCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecIFCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecIPCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecKeepValue() __attribute__ ((nothrow));
    void _ssdm_op_SpecMemCore() __attribute__ ((nothrow));

    void _ssdm_op_SpecExt() __attribute__ ((nothrow));




    void _ssdm_SpecArrayDimSize() __attribute__ ((nothrow));

    void _ssdm_RegionBegin() __attribute__ ((nothrow));
    void _ssdm_RegionEnd() __attribute__ ((nothrow));

    void _ssdm_Unroll() __attribute__ ((nothrow));
    void _ssdm_UnrollRegion() __attribute__ ((nothrow));

    void _ssdm_InlineAll() __attribute__ ((nothrow));
    void _ssdm_InlineLoop() __attribute__ ((nothrow));
    void _ssdm_Inline() __attribute__ ((nothrow));
    void _ssdm_InlineSelf() __attribute__ ((nothrow));
    void _ssdm_InlineRegion() __attribute__ ((nothrow));

    void _ssdm_SpecArrayMap() __attribute__ ((nothrow));
    void _ssdm_SpecArrayPartition() __attribute__ ((nothrow));
    void _ssdm_SpecArrayReshape() __attribute__ ((nothrow));

    void _ssdm_SpecStream() __attribute__ ((nothrow));

    void _ssdm_op_SpecStable() __attribute__ ((nothrow));
    void _ssdm_op_SpecStableContent() __attribute__ ((nothrow));

    void _ssdm_op_SpecBindPort() __attribute__ ((nothrow));

    void _ssdm_op_SpecPipoDepth() __attribute__ ((nothrow));

    void _ssdm_SpecExpr() __attribute__ ((nothrow));
    void _ssdm_SpecExprBalance() __attribute__ ((nothrow));

    void _ssdm_SpecDependence() __attribute__ ((nothrow));

    void _ssdm_SpecLoopMerge() __attribute__ ((nothrow));
    void _ssdm_SpecLoopFlatten() __attribute__ ((nothrow));
    void _ssdm_SpecLoopRewind() __attribute__ ((nothrow));

    void _ssdm_SpecFuncInstantiation() __attribute__ ((nothrow));
    void _ssdm_SpecFuncBuffer() __attribute__ ((nothrow));
    void _ssdm_SpecFuncExtract() __attribute__ ((nothrow));
    void _ssdm_SpecConstant() __attribute__ ((nothrow));

    void _ssdm_DataPack() __attribute__ ((nothrow));
    void _ssdm_SpecDataPack() __attribute__ ((nothrow));

    void _ssdm_op_SpecBitsMap() __attribute__ ((nothrow));
    void _ssdm_op_SpecLicense() __attribute__ ((nothrow));
# 2 "<built-in>" 2
# 1 "srcs/src/simulink_block.c" 2
# 1 "srcs/lib\\mpc_util.h" 1



# 1 "srcs/lib/types.h" 1







# 1 "srcs/lib/glob_opts.h" 1
# 10 "srcs/lib/glob_opts.h"
# 1 "srcs/lib/osqp_configure.h" 1
# 11 "srcs/lib/glob_opts.h" 2
# 108 "srcs/lib/glob_opts.h"
typedef long long c_int;
# 117 "srcs/lib/glob_opts.h"
typedef float c_float;
# 145 "srcs/lib/glob_opts.h"
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 1 3
# 11 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3


# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 1 3
# 10 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 1 3
# 12 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include/_mingw_mac.h" 1 3
# 13 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 2 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include/_mingw_secapi.h" 1 3
# 14 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 2 3
# 275 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 3
# 1 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\vadefs.h" 1 3
# 26 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\vadefs.h" 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\vadefs.h" 1 3








# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 1 3
# 565 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include/sdks/_mingw_directx.h" 1 3
# 566 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 2 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include/sdks/_mingw_ddk.h" 1 3
# 567 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 2 3
# 10 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\vadefs.h" 2 3




#pragma pack(push,_CRT_PACKING)
# 24 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\vadefs.h" 3
 typedef __builtin_va_list __gnuc_va_list;






  typedef __gnuc_va_list va_list;
# 103 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\vadefs.h" 3
#pragma pack(pop)
# 27 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\vadefs.h" 2 3
# 276 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 2 3
# 548 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\_mingw.h" 3
const char *__mingw_get_crt_info (void);
# 11 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 2 3




#pragma pack(push,_CRT_PACKING)
# 35 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
__extension__ typedef unsigned long size_t;
# 45 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
__extension__ typedef long ssize_t;






typedef size_t rsize_t;
# 62 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
__extension__ typedef long intptr_t;
# 75 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
__extension__ typedef unsigned long uintptr_t;
# 88 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
__extension__ typedef long ptrdiff_t;
# 98 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
typedef unsigned short wchar_t;







typedef unsigned short wint_t;
typedef unsigned short wctype_t;





typedef int errno_t;




typedef long __time32_t;




__extension__ typedef long __time64_t;
# 138 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
typedef __time64_t time_t;
# 422 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\crtdefs.h" 3
struct threadlocaleinfostruct;
struct threadmbcinfostruct;
typedef struct threadlocaleinfostruct *pthreadlocinfo;
typedef struct threadmbcinfostruct *pthreadmbcinfo;
struct __lc_time_data;

typedef struct localeinfo_struct {
  pthreadlocinfo locinfo;
  pthreadmbcinfo mbcinfo;
} _locale_tstruct,*_locale_t;



typedef struct tagLC_ID {
  unsigned short wLanguage;
  unsigned short wCountry;
  unsigned short wCodePage;
} LC_ID,*LPLC_ID;




typedef struct threadlocaleinfostruct {
  int refcount;
  unsigned int lc_codepage;
  unsigned int lc_collate_cp;
  unsigned long lc_handle[6];
  LC_ID lc_id[6];
  struct {
    char *locale;
    wchar_t *wlocale;
    int *refcount;
    int *wrefcount;
  } lc_category[6];
  int lc_clike;
  int mb_cur_max;
  int *lconv_intl_refcount;
  int *lconv_num_refcount;
  int *lconv_mon_refcount;
  struct lconv *lconv;
  int *ctype1_refcount;
  unsigned short *ctype1;
  const unsigned short *pctype;
  const unsigned char *pclmap;
  const unsigned char *pcumap;
  struct __lc_time_data *lc_time_curr;
} threadlocinfo;







#pragma pack(pop)
# 14 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 2 3

struct _exception;

#pragma pack(push,_CRT_PACKING)
# 119 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
 typedef union __mingw_dbl_type_t {
    double x;
    unsigned long long val;
    __extension__ struct {
      unsigned int low, high;
    } lh;
  } __mingw_dbl_type_t;

  typedef union __mingw_flt_type_t {
    float x;
    unsigned int val;
  } __mingw_flt_type_t;

  typedef union __mingw_ldbl_type_t
  {
    long double x;
    __extension__ struct {
      unsigned int low, high;
      int sign_exponent : 16;
      int res1 : 16;
      int res0 : 32;
    } lh;
  } __mingw_ldbl_type_t;

  typedef union __mingw_fp_types_t
  {
    long double *ld;
    double *d;
    float *f;
    __mingw_ldbl_type_t *ldt;
    __mingw_dbl_type_t *dt;
    __mingw_flt_type_t *ft;
  } __mingw_fp_types_t;




  extern double * __imp__HUGE;
# 168 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  struct _exception {
    int type;
    const char *name;
    double arg1;
    double arg2;
    double retval;
  };

  void __mingw_raise_matherr (int typ, const char *name, double a1, double a2,
         double rslt);
  void __mingw_setusermatherr (int (__attribute__((__cdecl__)) *)(struct _exception *));
  __attribute__ ((__dllimport__)) void __setusermatherr(int (__attribute__((__cdecl__)) *)(struct _exception *));



  double __attribute__((__cdecl__)) sin(double _X);
  double __attribute__((__cdecl__)) cos(double _X);
  double __attribute__((__cdecl__)) tan(double _X);
  double __attribute__((__cdecl__)) sinh(double _X);
  double __attribute__((__cdecl__)) cosh(double _X);
  double __attribute__((__cdecl__)) tanh(double _X);
  double __attribute__((__cdecl__)) asin(double _X);
  double __attribute__((__cdecl__)) acos(double _X);
  double __attribute__((__cdecl__)) atan(double _X);
  double __attribute__((__cdecl__)) atan2(double _Y,double _X);
  double __attribute__((__cdecl__)) exp(double _X);
  double __attribute__((__cdecl__)) log(double _X);
  double __attribute__((__cdecl__)) log10(double _X);
  double __attribute__((__cdecl__)) pow(double _X,double _Y);
  double __attribute__((__cdecl__)) sqrt(double _X);
  double __attribute__((__cdecl__)) ceil(double _X);
  double __attribute__((__cdecl__)) floor(double _X);


  extern float __attribute__((__cdecl__)) fabsf (float x);
  extern long double __attribute__((__cdecl__)) fabsl (long double);
  extern double __attribute__((__cdecl__)) fabs (double _X);
# 243 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  double __attribute__((__cdecl__)) ldexp(double _X,int _Y);
  double __attribute__((__cdecl__)) frexp(double _X,int *_Y);
  double __attribute__((__cdecl__)) modf(double _X,double *_Y);
  double __attribute__((__cdecl__)) fmod(double _X,double _Y);

  void __attribute__((__cdecl__)) sincos (double __x, double *p_sin, double *p_cos);
  void __attribute__((__cdecl__)) sincosl (long double __x, long double *p_sin, long double *p_cos);
  void __attribute__((__cdecl__)) sincosf (float __x, float *p_sin, float *p_cos);



  int __attribute__((__cdecl__)) abs(int _X);
  long __attribute__((__cdecl__)) labs(long _X);



  double __attribute__((__cdecl__)) atof(const char *_String);
  double __attribute__((__cdecl__)) _atof_l(const char *_String,_locale_t _Locale);
# 270 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  struct _complex {
    double x;
    double y;
  };


  double __attribute__((__cdecl__)) _cabs(struct _complex _ComplexA);
  double __attribute__((__cdecl__)) _hypot(double _X,double _Y);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _j0(double _X);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _j1(double _X);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _jn(int _X,double _Y);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _y0(double _X);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _y1(double _X);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _yn(int _X,double _Y);


  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _matherr (struct _exception *);
# 297 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _chgsign (double _X);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _copysign (double _Number,double _Sign);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _logb (double);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _nextafter (double, double);
  __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _scalb (double, long);
  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _finite (double);
  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fpclass (double);
  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _isnan (double);






__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) j0 (double) ;
__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) j1 (double) ;
__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) jn (int, double) ;
__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) y0 (double) ;
__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) y1 (double) ;
__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) yn (int, double) ;

__attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) chgsign (double);
# 327 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) finite (double);
  __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) fpclass (double);
# 372 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
typedef float float_t;
typedef double double_t;
# 407 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern int __attribute__((__cdecl__)) __fpclassifyl (long double);
  extern int __attribute__((__cdecl__)) __fpclassifyf (float);
  extern int __attribute__((__cdecl__)) __fpclassify (double);
# 520 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern int __attribute__((__cdecl__)) __isnan (double);
  extern int __attribute__((__cdecl__)) __isnanf (float);
  extern int __attribute__((__cdecl__)) __isnanl (long double);
# 607 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern int __attribute__((__cdecl__)) __signbit (double);
  extern int __attribute__((__cdecl__)) __signbitf (float);
  extern int __attribute__((__cdecl__)) __signbitl (long double);
# 664 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern float __attribute__((__cdecl__)) sinf(float _X);
  extern long double __attribute__((__cdecl__)) sinl(long double);

  extern float __attribute__((__cdecl__)) cosf(float _X);
  extern long double __attribute__((__cdecl__)) cosl(long double);

  extern float __attribute__((__cdecl__)) tanf(float _X);
  extern long double __attribute__((__cdecl__)) tanl(long double);
  extern float __attribute__((__cdecl__)) asinf(float _X);
  extern long double __attribute__((__cdecl__)) asinl(long double);

  extern float __attribute__((__cdecl__)) acosf (float);
  extern long double __attribute__((__cdecl__)) acosl (long double);

  extern float __attribute__((__cdecl__)) atanf (float);
  extern long double __attribute__((__cdecl__)) atanl (long double);

  extern float __attribute__((__cdecl__)) atan2f (float, float);
  extern long double __attribute__((__cdecl__)) atan2l (long double, long double);


  extern float __attribute__((__cdecl__)) sinhf(float _X);



  extern long double __attribute__((__cdecl__)) sinhl(long double);

  extern float __attribute__((__cdecl__)) coshf(float _X);



  extern long double __attribute__((__cdecl__)) coshl(long double);

  extern float __attribute__((__cdecl__)) tanhf(float _X);



  extern long double __attribute__((__cdecl__)) tanhl(long double);



  extern double __attribute__((__cdecl__)) acosh (double);
  extern float __attribute__((__cdecl__)) acoshf (float);
  extern long double __attribute__((__cdecl__)) acoshl (long double);


  extern double __attribute__((__cdecl__)) asinh (double);
  extern float __attribute__((__cdecl__)) asinhf (float);
  extern long double __attribute__((__cdecl__)) asinhl (long double);


  extern double __attribute__((__cdecl__)) atanh (double);
  extern float __attribute__((__cdecl__)) atanhf (float);
  extern long double __attribute__((__cdecl__)) atanhl (long double);



  extern float __attribute__((__cdecl__)) expf(float _X);



  extern long double __attribute__((__cdecl__)) expl(long double);


  extern double __attribute__((__cdecl__)) exp2(double);
  extern float __attribute__((__cdecl__)) exp2f(float);
  extern long double __attribute__((__cdecl__)) exp2l(long double);



  extern double __attribute__((__cdecl__)) expm1(double);
  extern float __attribute__((__cdecl__)) expm1f(float);
  extern long double __attribute__((__cdecl__)) expm1l(long double);


  extern float frexpf(float _X,int *_Y);



  extern long double __attribute__((__cdecl__)) frexpl(long double,int *);




  extern int __attribute__((__cdecl__)) ilogb (double);
  extern int __attribute__((__cdecl__)) ilogbf (float);
  extern int __attribute__((__cdecl__)) ilogbl (long double);


  extern float __attribute__((__cdecl__)) ldexpf(float _X,int _Y);



  extern long double __attribute__((__cdecl__)) ldexpl (long double, int);


  extern float __attribute__((__cdecl__)) logf (float);
  extern long double __attribute__((__cdecl__)) logl(long double);


  extern float __attribute__((__cdecl__)) log10f (float);
  extern long double __attribute__((__cdecl__)) log10l(long double);


  extern double __attribute__((__cdecl__)) log1p(double);
  extern float __attribute__((__cdecl__)) log1pf(float);
  extern long double __attribute__((__cdecl__)) log1pl(long double);


  extern double __attribute__((__cdecl__)) log2 (double);
  extern float __attribute__((__cdecl__)) log2f (float);
  extern long double __attribute__((__cdecl__)) log2l (long double);


  extern double __attribute__((__cdecl__)) logb (double);
  extern float __attribute__((__cdecl__)) logbf (float);
  extern long double __attribute__((__cdecl__)) logbl (long double);
# 863 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern float __attribute__((__cdecl__)) modff (float, float*);
  extern long double __attribute__((__cdecl__)) modfl (long double, long double*);


  extern double __attribute__((__cdecl__)) scalbn (double, int);
  extern float __attribute__((__cdecl__)) scalbnf (float, int);
  extern long double __attribute__((__cdecl__)) scalbnl (long double, int);

  extern double __attribute__((__cdecl__)) scalbln (double, long);
  extern float __attribute__((__cdecl__)) scalblnf (float, long);
  extern long double __attribute__((__cdecl__)) scalblnl (long double, long);



  extern double __attribute__((__cdecl__)) cbrt (double);
  extern float __attribute__((__cdecl__)) cbrtf (float);
  extern long double __attribute__((__cdecl__)) cbrtl (long double);


  extern double __attribute__((__cdecl__)) hypot (double, double) ;
  extern float __attribute__((__cdecl__)) hypotf (float x, float y);



  extern long double __attribute__((__cdecl__)) hypotl (long double, long double);


  extern float __attribute__((__cdecl__)) powf(float _X,float _Y);



  extern long double __attribute__((__cdecl__)) powl (long double, long double);


  extern float __attribute__((__cdecl__)) sqrtf (float);
  extern long double sqrtl(long double);


  extern double __attribute__((__cdecl__)) erf (double);
  extern float __attribute__((__cdecl__)) erff (float);
  extern long double __attribute__((__cdecl__)) erfl (long double);


  extern double __attribute__((__cdecl__)) erfc (double);
  extern float __attribute__((__cdecl__)) erfcf (float);
  extern long double __attribute__((__cdecl__)) erfcl (long double);


  extern double __attribute__((__cdecl__)) lgamma (double);
  extern float __attribute__((__cdecl__)) lgammaf (float);
  extern long double __attribute__((__cdecl__)) lgammal (long double);

  extern int signgam;


  extern double __attribute__((__cdecl__)) tgamma (double);
  extern float __attribute__((__cdecl__)) tgammaf (float);
  extern long double __attribute__((__cdecl__)) tgammal (long double);


  extern float __attribute__((__cdecl__)) ceilf (float);
  extern long double __attribute__((__cdecl__)) ceill (long double);


  extern float __attribute__((__cdecl__)) floorf (float);
  extern long double __attribute__((__cdecl__)) floorl (long double);


  extern double __attribute__((__cdecl__)) nearbyint ( double);
  extern float __attribute__((__cdecl__)) nearbyintf (float);
  extern long double __attribute__((__cdecl__)) nearbyintl (long double);



extern double __attribute__((__cdecl__)) rint (double);
extern float __attribute__((__cdecl__)) rintf (float);
extern long double __attribute__((__cdecl__)) rintl (long double);


extern long __attribute__((__cdecl__)) lrint (double);
extern long __attribute__((__cdecl__)) lrintf (float);
extern long __attribute__((__cdecl__)) lrintl (long double);

__extension__ long long __attribute__((__cdecl__)) llrint (double);
__extension__ long long __attribute__((__cdecl__)) llrintf (float);
__extension__ long long __attribute__((__cdecl__)) llrintl (long double);
# 1030 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern double __attribute__((__cdecl__)) round (double);
  extern float __attribute__((__cdecl__)) roundf (float);
  extern long double __attribute__((__cdecl__)) roundl (long double);


  extern long __attribute__((__cdecl__)) lround (double);
  extern long __attribute__((__cdecl__)) lroundf (float);
  extern long __attribute__((__cdecl__)) lroundl (long double);
  __extension__ long long __attribute__((__cdecl__)) llround (double);
  __extension__ long long __attribute__((__cdecl__)) llroundf (float);
  __extension__ long long __attribute__((__cdecl__)) llroundl (long double);



  extern double __attribute__((__cdecl__)) trunc (double);
  extern float __attribute__((__cdecl__)) truncf (float);
  extern long double __attribute__((__cdecl__)) truncl (long double);


  extern float __attribute__((__cdecl__)) fmodf (float, float);
  extern long double __attribute__((__cdecl__)) fmodl (long double, long double);


  extern double __attribute__((__cdecl__)) remainder (double, double);
  extern float __attribute__((__cdecl__)) remainderf (float, float);
  extern long double __attribute__((__cdecl__)) remainderl (long double, long double);


  extern double __attribute__((__cdecl__)) remquo(double, double, int *);
  extern float __attribute__((__cdecl__)) remquof(float, float, int *);
  extern long double __attribute__((__cdecl__)) remquol(long double, long double, int *);


  extern double __attribute__((__cdecl__)) copysign (double, double);
  extern float __attribute__((__cdecl__)) copysignf (float, float);
  extern long double __attribute__((__cdecl__)) copysignl (long double, long double);
# 1087 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern double __attribute__((__cdecl__)) nan(const char *tagp);
  extern float __attribute__((__cdecl__)) nanf(const char *tagp);
  extern long double __attribute__((__cdecl__)) nanl(const char *tagp);
# 1098 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
  extern double __attribute__((__cdecl__)) nextafter (double, double);
  extern float __attribute__((__cdecl__)) nextafterf (float, float);
  extern long double __attribute__((__cdecl__)) nextafterl (long double, long double);


  extern double __attribute__((__cdecl__)) nexttoward (double, long double);
  extern float __attribute__((__cdecl__)) nexttowardf (float, long double);
  extern long double __attribute__((__cdecl__)) nexttowardl (long double, long double);



  extern double __attribute__((__cdecl__)) fdim (double x, double y);
  extern float __attribute__((__cdecl__)) fdimf (float x, float y);
  extern long double __attribute__((__cdecl__)) fdiml (long double x, long double y);







  extern double __attribute__((__cdecl__)) fmax (double, double);
  extern float __attribute__((__cdecl__)) fmaxf (float, float);
  extern long double __attribute__((__cdecl__)) fmaxl (long double, long double);


  extern double __attribute__((__cdecl__)) fmin (double, double);
  extern float __attribute__((__cdecl__)) fminf (float, float);
  extern long double __attribute__((__cdecl__)) fminl (long double, long double);



  extern double __attribute__((__cdecl__)) fma (double, double, double);
  extern float __attribute__((__cdecl__)) fmaf (float, float, float);
  extern long double __attribute__((__cdecl__)) fmal (long double, long double, long double);
# 1181 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
   __attribute__ ((__dllimport__)) float __attribute__((__cdecl__)) _copysignf (float _Number,float _Sign);
   __attribute__ ((__dllimport__)) float __attribute__((__cdecl__)) _chgsignf (float _X);
   __attribute__ ((__dllimport__)) float __attribute__((__cdecl__)) _logbf(float _X);
   __attribute__ ((__dllimport__)) float __attribute__((__cdecl__)) _nextafterf(float _X,float _Y);
   __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _finitef(float _X);
   __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _isnanf(float _X);
   __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fpclassf(float _X);



   extern long double __attribute__((__cdecl__)) _chgsignl (long double);
# 1581 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\math.h" 3
#pragma pack(pop)
# 146 "srcs/lib/glob_opts.h" 2
# 9 "srcs/lib/types.h" 2
# 1 "srcs/lib/constants.h" 1
# 36 "srcs/lib/constants.h"
enum linsys_solver_type { QDLDL_SOLVER, MKL_PARDISO_SOLVER };
extern const char * LINSYS_SOLVER_NAME[];





enum osqp_error_type {
    OSQP_DATA_VALIDATION_ERROR = 1,
    OSQP_SETTINGS_VALIDATION_ERROR,
    OSQP_LINSYS_SOLVER_LOAD_ERROR,
    OSQP_LINSYS_SOLVER_INIT_ERROR,
    OSQP_NONCVX_ERROR,
    OSQP_MEM_ALLOC_ERROR,
    OSQP_WORKSPACE_NOT_INIT_ERROR,
};
extern const char * OSQP_ERROR_MESSAGE[];
# 10 "srcs/lib/types.h" 2
# 21 "srcs/lib/types.h"
typedef struct {
  c_int nzmax;
  c_int m;
  c_int n;
  c_int *p;
  c_int *i;
  c_float *x;
  c_int nz;
} csc;





typedef struct linsys_solver LinSysSolver;




typedef struct OSQP_TIMER OSQPTimer;




typedef struct {
  c_float c;
  c_float *D;
  c_float *E;
  c_float cinv;
  c_float *Dinv;
  c_float *Einv;
} OSQPScaling;




typedef struct {
  c_float *x;
  c_float *y;
} OSQPSolution;





typedef struct {
  c_int iter;
  char status[32];
  c_int status_val;





  c_float obj_val;
  c_float pri_res;
  c_float dua_res;
# 88 "srcs/lib/types.h"
  c_int rho_updates;
  c_float rho_estimate;

} OSQPInfo;
# 125 "srcs/lib/types.h"
typedef struct {
  c_int n;
  c_int m;
  csc *P;
  csc *A;
  c_float *q;
  c_float *l;
  c_float *u;
} OSQPData;





typedef struct {
  c_float rho;
  c_float sigma;
  c_int scaling;


  c_int adaptive_rho;
  c_int adaptive_rho_interval;
  c_float adaptive_rho_tolerance;





  c_int max_iter;
  c_float eps_abs;
  c_float eps_rel;
  c_float eps_prim_inf;
  c_float eps_dual_inf;
  c_float alpha;
  enum linsys_solver_type linsys_solver;
# 169 "srcs/lib/types.h"
  c_int scaled_termination;
  c_int check_termination;
  c_int warm_start;




} OSQPSettings;





typedef struct {

  OSQPData *data;


  LinSysSolver *linsys_solver;
# 198 "srcs/lib/types.h"
  c_float *rho_vec;
  c_float *rho_inv_vec;




  c_int *constr_type;






  c_float *x;
  c_float *y;
  c_float *z;
  c_float *xz_tilde;

  c_float *x_prev;


  c_float *z_prev;
# 230 "srcs/lib/types.h"
  c_float *Ax;
  c_float *Px;
  c_float *Aty;







  c_float *delta_y;
  c_float *Atdelta_y;







  c_float *delta_x;
  c_float *Pdelta_x;
  c_float *Adelta_x;
# 260 "srcs/lib/types.h"
  c_float *D_temp;
  c_float *D_temp_A;
  c_float *E_temp;




  OSQPSettings *settings;
  OSQPScaling *scaling;
  OSQPSolution *solution;
  OSQPInfo *info;
# 289 "srcs/lib/types.h"
} OSQPWorkspace;
# 298 "srcs/lib/types.h"
struct linsys_solver {
  enum linsys_solver_type type;
  c_int (*solve)(LinSysSolver *self,
                 c_float *b);






  c_int (*update_matrices)(LinSysSolver *s,
                           const csc *P,
                           const csc *A);

  c_int (*update_rho_vec)(LinSysSolver *s,
                          const c_float *rho_vec);





};
# 5 "srcs/lib\\mpc_util.h" 2

typedef struct{
    c_float a;
    c_float b;
    c_float c;
}v3ph;

void inverse_matrix_2x2(c_float a, c_float b, c_float c, c_float d, c_float inv[2][2]);
void multiplyMatrixVector(c_float Ex[2][2], c_float u[2], c_float result[2]);


void referencia(c_float* q_new, c_float ref);
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]);
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00);
void atualizar_restricao_v(c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax);
void atualizar_A(c_float Einv[2][2]);
# 2 "srcs/src/simulink_block.c" 2
# 1 "srcs/lib\\workspace.h" 1




# 1 "srcs/lib/qdldl_interface.h" 1








# 1 "srcs/lib/qdldl_types.h" 1








# 1 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\limits.h" 1 3
# 37 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\limits.h" 3
# 1 "C:/Xilinx/Vitis_HLS/2020.2/tps/mingw/6.2.0/win64.o/nt\\x86_64-w64-mingw32\\include\\limits.h" 1 3
# 38 "C:\\Xilinx\\Vitis_HLS\\2020.2\\win64\\tools\\clang-3.9-csynth\\lib\\clang\\7.0.0\\include\\limits.h" 2 3
# 10 "srcs/lib/qdldl_types.h" 2



typedef c_float QDLDL_float;
typedef c_int QDLDL_int;
typedef c_int QDLDL_bool;
# 10 "srcs/lib/qdldl_interface.h" 2




typedef struct qdldl qdldl_solver;

struct qdldl {
    enum linsys_solver_type type;
# 26 "srcs/lib/qdldl_interface.h"
    csc *L;
    c_float *Dinv;
    c_int *P;
    c_float *bp;
    c_float *sol;
    c_float *rho_inv_vec;
    c_float sigma;

    c_int n;
    c_int m;


    c_int * Pdiag_idx;
    c_int Pdiag_n;
    csc * KKT;
    c_int * PtoKKT;
    c_int * AtoKKT;
    c_int * rhotoKKT;


    QDLDL_float *D;
    QDLDL_int *etree;
    QDLDL_int *Lnz;
    QDLDL_int *iwork;
    QDLDL_bool *bwork;
    QDLDL_float *fwork;


};






c_int init_linsys_solver_qdldl(void);






c_int solve_linsys_qdldl(c_float * b);





c_int update_linsys_solver_matrices_qdldl(void);






c_int update_linsys_solver_rho_vec_qdldl(const c_float * rho_vec);
# 6 "srcs/lib\\workspace.h" 2


extern c_int Pdata_p[16];
extern c_int Pdata_i[12];
extern c_float Pdata_x[12];
extern c_int Adata_p[16];
extern c_int Adata_i[43];
extern c_float Adata_x[43];


extern csc Pdata;
extern csc Adata;
extern c_float qdata[15];
extern c_float ldata[19];
extern c_float udata[19];
extern OSQPData data;
extern OSQPSettings settings;
extern OSQPScaling scaling;


extern c_float scaling_D[15];
extern c_float scaling_Dinv[15];
extern c_float scaling_E[19];
extern c_float scaling_Einv[19];


extern c_int linsys_solver_L_p[35];
extern c_int linsys_solver_L_i[57];
extern c_float linsys_solver_L_x[57];
extern csc linsys_solver_L;


extern c_int linsys_solver_KKT_p[35];
extern c_int linsys_solver_KKT_i[79];
extern c_float linsys_solver_KKT_x[79];
extern csc linsys_solver_KKT;

extern c_float linsys_solver_Dinv[34];
extern c_int linsys_solver_P[34];
extern c_float linsys_solver_bp[34];
extern c_float linsys_solver_sol[34];
extern c_float linsys_solver_rho_inv_vec[19];

extern c_int linsys_solver_Pdiag_idx[10];
extern c_int linsys_solver_PtoKKT[12];
extern c_int linsys_solver_AtoKKT[43];
extern c_int linsys_solver_rhotoKKT[19];

extern QDLDL_float linsys_solver_D[34];
extern QDLDL_int linsys_solver_etree[34];
extern QDLDL_int linsys_solver_Lnz[34];
extern QDLDL_int linsys_solver_iwork[102];
extern QDLDL_bool linsys_solver_bwork[34];
extern QDLDL_float linsys_solver_fwork[34];

extern qdldl_solver linsys_solver;


extern c_float xsolution[15];
extern c_float ysolution[19];
extern OSQPSolution solution;
extern OSQPInfo info;


extern c_float work_rho_vec[19];
extern c_float work_rho_inv_vec[19];
extern c_int work_constr_type[19];
extern c_float work_x[15];
extern c_float work_y[19];
extern c_float work_z[19];
extern c_float work_xz_tilde[34];
extern c_float work_x_prev[15];
extern c_float work_z_prev[19];
extern c_float work_Ax[19];
extern c_float work_Px[15];
extern c_float work_Aty[15];
extern c_float work_delta_y[19];
extern c_float work_Atdelta_y[15];
extern c_float work_delta_x[15];
extern c_float work_Pdelta_x[15];
extern c_float work_Adelta_x[19];
extern c_float work_D_temp[15];
extern c_float work_D_temp_A[15];
extern c_float work_E_temp[19];

extern OSQPWorkspace workspace;
# 3 "srcs/src/simulink_block.c" 2
# 1 "srcs/lib\\osqp.h" 1
# 12 "srcs/lib\\osqp.h"
void osqp_set_default_settings(OSQPSettings *settings);



c_int osqp_solve(void);

c_int osqp_update_lin_cost(const c_float *q_new);

c_int osqp_update_bounds(const c_float *l_new,
                         const c_float *u_new);

c_int osqp_update_P(const c_float *Px_new,
                    const c_int *Px_new_idx,
                    c_int P_new_n);

c_int osqp_update_A(const c_float *Ax_new,
                    const c_int *Ax_new_idx,
                    c_int A_new_n);

c_int osqp_update_rho(c_float rho_new);


void update_xz_tilde(void);
void update_x(void);
void update_z(void);
void update_y(void);
# 4 "srcs/src/simulink_block.c" 2
# 1 "srcs/lib\\lin_alg.h" 1
# 11 "srcs/lib\\lin_alg.h"
void vec_add_scaled(c_float *c, const c_float *a, const c_float *b, c_int n, c_float sc);
c_float vec_scaled_norm_inf(const c_float *S, const c_float *v, c_int l);
c_float vec_norm_inf(const c_float *v, c_int l);
c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l);
c_float vec_mean(const c_float *a, c_int n);
void int_vec_set_scalar(c_int *a, c_int sc, c_int n);
void vec_set_scalar(c_float *a, c_float sc, c_int n);
void vec_add_scalar(c_float *a, c_float sc, c_int n);
void vec_mult_scalar(c_float *a, c_float sc, c_int n);
void prea_int_vec_copy(const c_int *a, c_int *b, c_int n);
void prea_vec_copy(const c_float *a, c_float *b, c_int n);
void vec_ew_recipr(const c_float *a, c_float *b, c_int n);
c_float vec_prod(const c_float *a, const c_float *b, c_int n);
void vec_ew_prod(const c_float *a, const c_float *b, c_float *c, c_int n);




void mat_mult_scalar(c_float *Ax, const c_int *Ap, c_int An, c_float sc);

void mat_premult_diag(c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, const c_float *d);

void mat_postmult_diag(c_float *Ax, const c_int *Ap, c_int An, const c_float *d);

void mat_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
             const c_float *x, c_float *y, c_int plus_eq);

void mat_tpose_vec(const c_float *Ax, const c_int *Ap, const c_int *Ai, c_int An, c_int Am,
                   const c_float *x, c_float *y, c_int plus_eq, c_int skip_diag);

c_float quad_form(const c_float *Px, const c_int *Pp, const c_int *Pi, c_int Pn, const c_float *x);
# 5 "srcs/src/simulink_block.c" 2
# 1 "srcs/lib\\simulink_block.h" 1
__attribute__((sdx_kernel("myFunction", 0))) void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2]);
# 6 "srcs/src/simulink_block.c" 2
# 20 "srcs/src/simulink_block.c"
extern OSQPWorkspace workspace;





static void local_inverse_matrix_2x2(float a, float b, float c, float d, float m[2][2]) {
#pragma HLS INLINE
 float det = a*d - b*c;
    float detInv = 1.0f / det;
    m[0][0] = d * detInv;
    m[0][1] = -b * detInv;
    m[1][0] = -c * detInv;
    m[1][1] = a * detInv;
}

static void local_multiplyMatrixVector(float m[2][2], float v[2], float res[2]) {
#pragma HLS INLINE
 res[0] = m[0][0]*v[0] + m[0][1]*v[1];
    res[1] = m[1][0]*v[0] + m[1][1]*v[1];
}


void referencia(c_float* q_new, c_float ref);
void calculateV(c_float Ax[2], c_float Ex[2][2], c_float u[2], c_float v[2]);
void atualizar_restricao(c_float* l_new, c_float* u_new, c_float* x, c_float* v00);


void atualizar_restricao_v(c_float* l_new, c_float* u_new, c_float vdc, c_float Einv[2][2], c_float* Ax);
void atualizar_A(c_float Einv[2][2]);


void init_workspace_manually() {

    scaling.c = 1.0;
    scaling.cinv = 1.0;
# 64 "srcs/src/simulink_block.c"
}

__attribute__((sdx_kernel("myFunction", 0))) void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2])
{_ssdm_SpecArrayDimSize(x_ini, 3);_ssdm_SpecArrayDimSize(u00, 2);_ssdm_SpecArrayDimSize(outputVector, 2);
#pragma HLS TOP name=myFunction
# 67 "srcs/src/simulink_block.c"

#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=x_ini
#pragma HLS INTERFACE s_axilite port=Vsd
#pragma HLS INTERFACE s_axilite port=Vsq
#pragma HLS INTERFACE s_axilite port=iL
#pragma HLS INTERFACE s_axilite port=u00
#pragma HLS INTERFACE s_axilite port=outputVector


#pragma HLS ARRAY_PARTITION variable=x_ini complete
#pragma HLS ARRAY_PARTITION variable=u00 complete
#pragma HLS ARRAY_PARTITION variable=outputVector complete

 static int is_initialized = 0;
    if (!is_initialized) {
        init_workspace_manually();
        is_initialized = 1;
    }

    float vdc, Ex11, Ex12, Ex21, Ex22, Ax01, Ax02, Rl;

    float Ax[2];
#pragma HLS ARRAY_PARTITION variable=Ax complete

 float Ex[2][2];
#pragma HLS ARRAY_PARTITION variable=Ex complete dim=0

 float Einv[2][2];
#pragma HLS ARRAY_PARTITION variable=Einv complete dim=0

 float z_ini[3];
    float v00[2];
    float v[2];
    float u[2];

    c_float q_new[15];
    c_float l_new[19];
    c_float u_new[19];
    float ref = 380.0f;

    vdc = x_ini[2];

    if (fabsf(iL) < 0.1f) {
        Rl = 10000.0f;
    } else {
        Rl = vdc/iL;
    }

    prea_vec_copy(qdata, q_new, 15);
    prea_vec_copy(ldata, l_new, 19);
    prea_vec_copy(udata, u_new, 19);

    Ax01 = - 314.159265f*x_ini[0] - (0.1f*x_ini[1])/0.005f;
    float term = (3.0f * (Vsd * x_ini[0] + Vsq * x_ini[1] - 0.1f * (x_ini[0] * x_ini[0] + x_ini[1] * x_ini[1]))) / (2.0f * 0.001f * x_ini[2]);
    float term2 = 1.0f / (2.0f * 0.001f * x_ini[2]);
    Ax02 = (1.0f / (0.001f * Rl) + term / x_ini[2]) * (x_ini[2] / (0.001f * Rl) - term) -
            (3.0f * (314.159265f * x_ini[0] + (0.1f * x_ini[1]) / 0.005f) * (Vsq - 2.0f * 0.1f * x_ini[1])) * term2 +
            (3.0f * (Vsd - 2.0f * 0.1f * x_ini[0]) * (314.159265f * x_ini[1] + Vsd / 0.005f - (0.1f * x_ini[0]) / 0.005f)) * term2;

    Ex11 = 0.0f; Ex12 = -1.0f/0.005f;
    float termL = 1.0f / (2.0f * 0.001f * 0.005f * x_ini[2]);
    Ex21 = -(3.0f * (Vsd - 2.0f * 0.1f * x_ini[0])) * termL;
    Ex22 = -(3.0f * (Vsq - 2.0f * 0.1f * x_ini[1])) * termL;

    Ax[0] = Ax01; Ax[1] = Ax02;
    Ex[0][0] = Ex11; Ex[0][1] = Ex12;
    Ex[1][0] = Ex21; Ex[1][1] = Ex22;


    local_inverse_matrix_2x2(Ex11, Ex12, Ex21, Ex22, Einv);

    z_ini[0] = x_ini[1]; z_ini[1] = x_ini[2]; z_ini[2] = term - x_ini[2] / (0.001f * Rl);

    referencia(q_new, ref);
    calculateV(Ax, Ex, u00, v00);
    atualizar_restricao(l_new, u_new, z_ini, v00);



    atualizar_restricao_v(l_new, u_new, z_ini[1], Einv, Ax);

    atualizar_A(Einv);

    osqp_solve();

    v[0] = work_x[11];
    v[1] = work_x[12];

    v[0] = v[0] - Ax[0];
    v[1] = v[1] - Ax[1];


    local_multiplyMatrixVector(Einv, v, u);

    u[0] = u[0]/vdc;
    u[1] = u[1]/vdc;

    outputVector[0] = work_x[11];
    outputVector[1] = work_x[12];
}
