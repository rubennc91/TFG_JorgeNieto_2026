############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls_project
set_top myFunction
add_files srcs/lib/workspace.h
add_files srcs/src/workspace.c -cflags "-Isrcs/lib"
add_files srcs/lib/util.h
add_files srcs/src/util.c -cflags "-Isrcs/lib"
add_files srcs/lib/types.h
add_files srcs/lib/spll_3ph_srf.h
add_files srcs/lib/simulink_block.h
add_files srcs/src/simulink_block.c -cflags "-Isrcs/lib"
add_files srcs/lib/scaling.h
add_files srcs/src/scaling.c -cflags "-Isrcs/lib"
add_files srcs/lib/qdldl_types.h
add_files srcs/lib/qdldl_interface.h
add_files srcs/src/qdldl_interface.c -cflags "-Isrcs/lib"
add_files srcs/lib/qdldl.h
add_files srcs/src/qdldl.c -cflags "-Isrcs/lib"
add_files srcs/lib/proj.h
add_files srcs/src/proj.c -cflags "-Isrcs/lib"
add_files srcs/lib/osqp_configure.h
add_files srcs/lib/osqp.h
add_files srcs/src/osqp.c -cflags "-Isrcs/lib"
add_files srcs/lib/mpc_util.h
add_files srcs/src/mpc_util.c -cflags "-Isrcs/lib"
add_files srcs/lib/lin_alg.h
add_files srcs/src/lin_alg.c -cflags "-Isrcs/lib"
add_files srcs/lib/kkt.h
add_files srcs/src/kkt.c -cflags "-Isrcs/lib"
add_files srcs/lib/glob_opts.h
add_files srcs/lib/error.h
add_files srcs/src/error.c -cflags "-Isrcs/lib"
add_files srcs/lib/dq0_abc.h
add_files srcs/lib/constants.h
add_files srcs/lib/auxil.h
add_files srcs/src/auxil.c -cflags "-Isrcs/lib"
add_files srcs/lib/abc_dq0_pos.h
add_files -tb srcs/testbench/main.c -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb srcs/testbench/gold_data.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb srcs/testbench/gold_data.c -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg484-1}
create_clock -period 10 -name default
source "./hls_project/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
