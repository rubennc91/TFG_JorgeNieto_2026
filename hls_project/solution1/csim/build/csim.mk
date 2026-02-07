# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
# Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
# ==============================================================
CSIM_DESIGN = 1

__SIM_FPO__ = 1

__SIM_MATHHLS__ = 1

__SIM_FFT__ = 1

__SIM_FIR__ = 1

__SIM_DDS__ = 1

ObjDir = obj

HLS_SOURCES = ../../../../srcs/testbench/gold_data.c ../../../../srcs/testbench/main.c ../../../../srcs/src/auxil.c ../../../../srcs/src/error.c ../../../../srcs/src/kkt.c ../../../../srcs/src/lin_alg.c ../../../../srcs/src/mpc_util.c ../../../../srcs/src/osqp.c ../../../../srcs/src/proj.c ../../../../srcs/src/qdldl.c ../../../../srcs/src/qdldl_interface.c ../../../../srcs/src/scaling.c ../../../../srcs/src/simulink_block.c ../../../../srcs/src/util.c ../../../../srcs/src/workspace.c

override TARGET := csim.exe

AUTOPILOT_ROOT := C:/Xilinx/Vitis_HLS/2020.2
AUTOPILOT_MACH := win64
ifdef AP_GCC_M32
  AUTOPILOT_MACH := Linux_x86
  IFLAG += -m32
endif
ifndef AP_GCC_PATH
  AP_GCC_PATH := C:/Xilinx/Vitis_HLS/2020.2/tps/win64/msys64/mingw64/bin
endif
AUTOPILOT_TOOL := ${AUTOPILOT_ROOT}/${AUTOPILOT_MACH}/tools
AP_CLANG_PATH := ${AUTOPILOT_ROOT}/tps/win64/msys64/mingw64/bin
AUTOPILOT_TECH := ${AUTOPILOT_ROOT}/common/technology


IFLAG += -I "${AUTOPILOT_TOOL}/systemc/include"
IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${AUTOPILOT_ROOT}/include/ap_sysc"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC/AESL_FP_comp"
IFLAG += -I "${AUTOPILOT_TECH}/generic/SystemC/AESL_comp"
IFLAG += -I "${AUTOPILOT_TOOL}/auto_cc/include"
IFLAG += -D__VITIS_HLS__

IFLAG += -D__SIM_FPO__

IFLAG += -D__SIM_FFT__

IFLAG += -D__SIM_FIR__

IFLAG += -D__SIM_DDS__

IFLAG += -D__DSP48E1__
IFLAG += -Wno-unknown-pragmas -I../../srcs/lib 
IFLAG += -g
IFLAG += -DNT
LFLAG += -Wl,--enable-auto-import 
DFLAG += -DAUTOCC
DFLAG += -D__xilinx_ip_top= -DAESL_TB
CCFLAG += -Werror=return-type
TOOLCHAIN += 



include ./Makefile.rules

all: $(TARGET)



AUTOCC := cmd //c apcc.bat  

$(ObjDir)/gold_data.o: ../../../../srcs/testbench/gold_data.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/testbench/gold_data.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -Wno-unknown-pragmas -Wno-unknown-pragmas  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/gold_data.d

$(ObjDir)/main.o: ../../../../srcs/testbench/main.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/testbench/main.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -Wno-unknown-pragmas -Wno-unknown-pragmas  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/main.d

$(ObjDir)/auxil.o: ../../../../srcs/src/auxil.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/auxil.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/auxil.d

$(ObjDir)/error.o: ../../../../srcs/src/error.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/error.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/error.d

$(ObjDir)/kkt.o: ../../../../srcs/src/kkt.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/kkt.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/kkt.d

$(ObjDir)/lin_alg.o: ../../../../srcs/src/lin_alg.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/lin_alg.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/lin_alg.d

$(ObjDir)/mpc_util.o: ../../../../srcs/src/mpc_util.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/mpc_util.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/mpc_util.d

$(ObjDir)/osqp.o: ../../../../srcs/src/osqp.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/osqp.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/osqp.d

$(ObjDir)/proj.o: ../../../../srcs/src/proj.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/proj.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/proj.d

$(ObjDir)/qdldl.o: ../../../../srcs/src/qdldl.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/qdldl.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/qdldl.d

$(ObjDir)/qdldl_interface.o: ../../../../srcs/src/qdldl_interface.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/qdldl_interface.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/qdldl_interface.d

$(ObjDir)/scaling.o: ../../../../srcs/src/scaling.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/scaling.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/scaling.d

$(ObjDir)/simulink_block.o: ../../../../srcs/src/simulink_block.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/simulink_block.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/simulink_block.d

$(ObjDir)/util.o: ../../../../srcs/src/util.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/util.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/util.d

$(ObjDir)/workspace.o: ../../../../srcs/src/workspace.c $(ObjDir)/.dir
	$(Echo) "   Compiling(apcc) ../../../../srcs/src/workspace.c in $(BuildMode) mode" $(AVE_DIR_DLOG)
	$(Verb)  $(AUTOCC) -c -MMD -I../../../../srcs/lib  $(IFLAG) $(DFLAG) $< -o $@ ; \

-include $(ObjDir)/workspace.d
