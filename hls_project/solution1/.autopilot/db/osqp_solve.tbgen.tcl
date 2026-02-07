set moduleName osqp_solve
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {osqp_solve}
set C_modelType { void 0 }
set C_modelArgList {
	{ work_x float 32 regular {array 15 { 2 3 } 1 1 } {global 2}  }
	{ qdata float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
	{ work_rho_inv_vec float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_L_p int 6 regular {array 35 { 1 1 } 1 1 } {global 0}  }
	{ linsys_solver_L_x float 32 regular {array 57 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_L_i int 6 regular {array 57 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_Dinv float 32 regular {array 34 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_rho_inv_vec float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ ldata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ udata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ work_rho_vec float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ Adata_x float 32 regular {array 43 { 1 3 } 1 1 } {global 0}  }
	{ info_status_val int 5 regular {pointer 2} {global 2}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "work_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "qdata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "qdata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_rho_inv_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_rho_inv_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_p", "interface" : "memory", "bitwidth" : 6, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_p","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 34,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_L_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_i", "interface" : "memory", "bitwidth" : 6, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_i","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_Dinv", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_Dinv","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_rho_inv_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_rho_inv_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ldata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "ldata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "udata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "udata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_rho_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_rho_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "Adata_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "Adata_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 42,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "info_status_val", "interface" : "wire", "bitwidth" : 5, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "info.status_val","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 78
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ work_x_address0 sc_out sc_lv 4 signal 0 } 
	{ work_x_ce0 sc_out sc_logic 1 signal 0 } 
	{ work_x_we0 sc_out sc_logic 1 signal 0 } 
	{ work_x_d0 sc_out sc_lv 32 signal 0 } 
	{ work_x_q0 sc_in sc_lv 32 signal 0 } 
	{ qdata_address0 sc_out sc_lv 4 signal 1 } 
	{ qdata_ce0 sc_out sc_logic 1 signal 1 } 
	{ qdata_q0 sc_in sc_lv 32 signal 1 } 
	{ work_rho_inv_vec_address0 sc_out sc_lv 5 signal 2 } 
	{ work_rho_inv_vec_ce0 sc_out sc_logic 1 signal 2 } 
	{ work_rho_inv_vec_q0 sc_in sc_lv 32 signal 2 } 
	{ linsys_solver_L_p_address0 sc_out sc_lv 6 signal 3 } 
	{ linsys_solver_L_p_ce0 sc_out sc_logic 1 signal 3 } 
	{ linsys_solver_L_p_q0 sc_in sc_lv 6 signal 3 } 
	{ linsys_solver_L_p_address1 sc_out sc_lv 6 signal 3 } 
	{ linsys_solver_L_p_ce1 sc_out sc_logic 1 signal 3 } 
	{ linsys_solver_L_p_q1 sc_in sc_lv 6 signal 3 } 
	{ linsys_solver_L_x_address0 sc_out sc_lv 6 signal 4 } 
	{ linsys_solver_L_x_ce0 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_L_x_q0 sc_in sc_lv 32 signal 4 } 
	{ linsys_solver_L_i_address0 sc_out sc_lv 6 signal 5 } 
	{ linsys_solver_L_i_ce0 sc_out sc_logic 1 signal 5 } 
	{ linsys_solver_L_i_q0 sc_in sc_lv 6 signal 5 } 
	{ linsys_solver_Dinv_address0 sc_out sc_lv 6 signal 6 } 
	{ linsys_solver_Dinv_ce0 sc_out sc_logic 1 signal 6 } 
	{ linsys_solver_Dinv_q0 sc_in sc_lv 32 signal 6 } 
	{ linsys_solver_rho_inv_vec_address0 sc_out sc_lv 5 signal 7 } 
	{ linsys_solver_rho_inv_vec_ce0 sc_out sc_logic 1 signal 7 } 
	{ linsys_solver_rho_inv_vec_q0 sc_in sc_lv 32 signal 7 } 
	{ ldata_address0 sc_out sc_lv 5 signal 8 } 
	{ ldata_ce0 sc_out sc_logic 1 signal 8 } 
	{ ldata_q0 sc_in sc_lv 32 signal 8 } 
	{ udata_address0 sc_out sc_lv 5 signal 9 } 
	{ udata_ce0 sc_out sc_logic 1 signal 9 } 
	{ udata_q0 sc_in sc_lv 32 signal 9 } 
	{ work_rho_vec_address0 sc_out sc_lv 5 signal 10 } 
	{ work_rho_vec_ce0 sc_out sc_logic 1 signal 10 } 
	{ work_rho_vec_q0 sc_in sc_lv 32 signal 10 } 
	{ Adata_x_address0 sc_out sc_lv 6 signal 11 } 
	{ Adata_x_ce0 sc_out sc_logic 1 signal 11 } 
	{ Adata_x_q0 sc_in sc_lv 32 signal 11 } 
	{ info_status_val_i sc_in sc_lv 5 signal 12 } 
	{ info_status_val_o sc_out sc_lv 5 signal 12 } 
	{ info_status_val_o_ap_vld sc_out sc_logic 1 outvld 12 } 
	{ grp_fu_1198_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1198_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1198_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_1198_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1198_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1202_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1202_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1202_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_1202_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1202_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1216_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1216_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1216_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1216_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1220_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1220_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1220_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1220_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1264_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1264_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1264_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_1264_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_1264_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2343_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2343_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2343_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_2343_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_2343_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "work_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_x", "role": "address0" }} , 
 	{ "name": "work_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_x", "role": "ce0" }} , 
 	{ "name": "work_x_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_x", "role": "we0" }} , 
 	{ "name": "work_x_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_x", "role": "d0" }} , 
 	{ "name": "work_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_x", "role": "q0" }} , 
 	{ "name": "qdata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "qdata", "role": "address0" }} , 
 	{ "name": "qdata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "qdata", "role": "ce0" }} , 
 	{ "name": "qdata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "qdata", "role": "q0" }} , 
 	{ "name": "work_rho_inv_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "address0" }} , 
 	{ "name": "work_rho_inv_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "ce0" }} , 
 	{ "name": "work_rho_inv_vec_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "q0" }} , 
 	{ "name": "linsys_solver_L_p_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_p_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_p_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "q0" }} , 
 	{ "name": "linsys_solver_L_p_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "address1" }} , 
 	{ "name": "linsys_solver_L_p_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "ce1" }} , 
 	{ "name": "linsys_solver_L_p_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "q1" }} , 
 	{ "name": "linsys_solver_L_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "q0" }} , 
 	{ "name": "linsys_solver_L_i_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_i_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_i_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "q0" }} , 
 	{ "name": "linsys_solver_Dinv_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "address0" }} , 
 	{ "name": "linsys_solver_Dinv_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "ce0" }} , 
 	{ "name": "linsys_solver_Dinv_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "q0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "address0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "ce0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "q0" }} , 
 	{ "name": "ldata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "ldata", "role": "address0" }} , 
 	{ "name": "ldata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ldata", "role": "ce0" }} , 
 	{ "name": "ldata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ldata", "role": "q0" }} , 
 	{ "name": "udata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "udata", "role": "address0" }} , 
 	{ "name": "udata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "udata", "role": "ce0" }} , 
 	{ "name": "udata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "udata", "role": "q0" }} , 
 	{ "name": "work_rho_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "address0" }} , 
 	{ "name": "work_rho_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "ce0" }} , 
 	{ "name": "work_rho_vec_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "q0" }} , 
 	{ "name": "Adata_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Adata_x", "role": "address0" }} , 
 	{ "name": "Adata_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Adata_x", "role": "ce0" }} , 
 	{ "name": "Adata_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Adata_x", "role": "q0" }} , 
 	{ "name": "info_status_val_i", "direction": "in", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "info_status_val", "role": "i" }} , 
 	{ "name": "info_status_val_o", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "info_status_val", "role": "o" }} , 
 	{ "name": "info_status_val_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "info_status_val", "role": "o_ap_vld" }} , 
 	{ "name": "grp_fu_1198_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1198_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1198_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1198_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1198_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_1198_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_1198_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1198_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1198_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1198_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1202_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1202_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1202_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1202_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1202_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_1202_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_1202_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1202_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1202_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1202_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1216_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1216_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1216_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1216_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1216_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1216_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1216_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1216_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1220_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1220_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1220_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1220_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1220_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1220_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1220_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1220_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1264_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1264_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_1264_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1264_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1264_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2343_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2343_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2343_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2343_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2343_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_2343_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_2343_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2343_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2343_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2343_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "29", "33", "34", "35"],
		"CDFG" : "osqp_solve",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "work_x", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "work_z", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_z"}]},
			{"Name" : "work_y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "work_x_prev", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "33", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "work_x_prev"}]},
			{"Name" : "work_z_prev", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "33", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "qdata"},
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "qdata"}]},
			{"Name" : "work_xz_tilde", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "work_xz_tilde"}]},
			{"Name" : "work_rho_inv_vec", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_P", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_P"}]},
			{"Name" : "linsys_solver_bp", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_bp"}]},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_p"}]},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_x"}]},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_i"}]},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_Dinv"}]},
			{"Name" : "linsys_solver_sol", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_sol"}]},
			{"Name" : "linsys_solver_rho_inv_vec", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_rho_inv_vec"}]},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_delta_x"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "ldata"}]},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "udata"}]},
			{"Name" : "work_rho_vec", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_delta_y"}]},
			{"Name" : "work_Ax", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Ax"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_p"}]},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_x"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_i"}]},
			{"Name" : "info_pri_res", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_pri_res"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "33", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "work_Px"},
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Px"}]},
			{"Name" : "info_dua_res", "Type" : "None", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_dua_res"}]},
			{"Name" : "info_status_val", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_status_val"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "12", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Atdelta_y"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_z_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_y_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_x_prev_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_z_prev_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_xz_tilde_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_delta_x_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_delta_y_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_Ax_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Adata_p_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Adata_i_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_Px_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939", "Parent" : "0", "Child" : ["13", "22", "24", "25", "26", "27", "28"],
		"CDFG" : "check_termination",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "approximate", "Type" : "None", "Direction" : "I"},
			{"Name" : "info_pri_res", "Type" : "None", "Direction" : "I"},
			{"Name" : "info_dua_res", "Type" : "None", "Direction" : "I"},
			{"Name" : "info_status_val", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "work_z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_Ax", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "udata"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "ldata"}]},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_delta_y"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_Atdelta_y"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_p"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_i"}]},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_x"}]},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "qdata"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "work_delta_x"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317", "Parent" : "12", "Child" : ["14", "15", "16", "17", "18", "19", "20", "21"],
		"CDFG" : "is_primal_infeasible",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "eps_prim_inf", "Type" : "None", "Direction" : "I"},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "14", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.work_Atdelta_y_U", "Parent" : "13"},
	{"ID" : "15", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.Adata_p_U", "Parent" : "13"},
	{"ID" : "16", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.Adata_i_U", "Parent" : "13"},
	{"ID" : "17", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U78", "Parent" : "13"},
	{"ID" : "18", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U79", "Parent" : "13"},
	{"ID" : "19", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U82", "Parent" : "13"},
	{"ID" : "20", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U83", "Parent" : "13"},
	{"ID" : "21", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.facc_32ns_32ns_1ns_32_6_no_dsp_1_U84", "Parent" : "13"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_dual_infeasible_fu_337", "Parent" : "12", "Child" : ["23"],
		"CDFG" : "is_dual_infeasible",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "93", "EstimateLatencyMax" : "278",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "eps_dual_inf", "Type" : "None", "Direction" : "I"},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.grp_is_dual_infeasible_fu_337.facc_32ns_32ns_1ns_32_6_no_dsp_1_U96", "Parent" : "22"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.fadd_32ns_32ns_32_5_full_dsp_1_U100", "Parent" : "12"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.fmul_32ns_32ns_32_4_max_dsp_1_U101", "Parent" : "12"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.fcmp_32ns_32ns_1_2_no_dsp_1_U102", "Parent" : "12"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.fcmp_32ns_32ns_1_2_no_dsp_1_U103", "Parent" : "12"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_check_termination_fu_939.fmul_32ns_32ns_32_4_max_dsp_1_U104", "Parent" : "12"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_solve_linsys_qdldl_fu_976", "Parent" : "0", "Child" : ["30", "31", "32"],
		"CDFG" : "solve_linsys_qdldl",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "linsys_solver_P", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_xz_tilde", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_bp", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_sol", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_rho_inv_vec", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_solve_linsys_qdldl_fu_976.linsys_solver_P_U", "Parent" : "29"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_solve_linsys_qdldl_fu_976.linsys_solver_bp_U", "Parent" : "29"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_solve_linsys_qdldl_fu_976.linsys_solver_sol_U", "Parent" : "29"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_dua_res_fu_998", "Parent" : "0",
		"CDFG" : "compute_dua_res",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_x_prev", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.faddfsub_32ns_32ns_32_5_full_dsp_1_U117", "Parent" : "0"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U120", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	osqp_solve {
		work_x {Type IO LastRead 27 FirstWrite 1}
		work_z {Type IO LastRead -1 FirstWrite -1}
		work_y {Type IO LastRead -1 FirstWrite -1}
		work_x_prev {Type IO LastRead -1 FirstWrite -1}
		work_z_prev {Type IO LastRead -1 FirstWrite -1}
		qdata {Type I LastRead 18 FirstWrite -1}
		work_xz_tilde {Type IO LastRead -1 FirstWrite -1}
		work_rho_inv_vec {Type I LastRead 24 FirstWrite -1}
		linsys_solver_P {Type I LastRead -1 FirstWrite -1}
		linsys_solver_bp {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_L_p {Type I LastRead 6 FirstWrite -1}
		linsys_solver_L_x {Type I LastRead 7 FirstWrite -1}
		linsys_solver_L_i {Type I LastRead 7 FirstWrite -1}
		linsys_solver_Dinv {Type I LastRead 3 FirstWrite -1}
		linsys_solver_sol {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_rho_inv_vec {Type I LastRead 10 FirstWrite -1}
		work_delta_x {Type IO LastRead -1 FirstWrite -1}
		ldata {Type I LastRead 21 FirstWrite -1}
		udata {Type I LastRead 23 FirstWrite -1}
		work_rho_vec {Type I LastRead 37 FirstWrite -1}
		work_delta_y {Type IO LastRead -1 FirstWrite -1}
		work_Ax {Type IO LastRead -1 FirstWrite -1}
		Adata_p {Type I LastRead -1 FirstWrite -1}
		Adata_x {Type I LastRead 29 FirstWrite -1}
		Adata_i {Type I LastRead -1 FirstWrite -1}
		info_pri_res {Type IO LastRead -1 FirstWrite -1}
		work_Px {Type IO LastRead -1 FirstWrite -1}
		info_dua_res {Type I LastRead -1 FirstWrite -1}
		info_status_val {Type IO LastRead 41 FirstWrite 2}
		work_Atdelta_y {Type IO LastRead -1 FirstWrite -1}}
	check_termination {
		approximate {Type I LastRead 1 FirstWrite -1}
		info_pri_res {Type I LastRead 0 FirstWrite -1}
		info_dua_res {Type I LastRead -1 FirstWrite -1}
		info_status_val {Type O LastRead -1 FirstWrite 2}
		work_z {Type I LastRead 2 FirstWrite -1}
		work_Ax {Type I LastRead 3 FirstWrite -1}
		udata {Type I LastRead 7 FirstWrite -1}
		ldata {Type I LastRead 7 FirstWrite -1}
		work_delta_y {Type IO LastRead 17 FirstWrite 6}
		work_Atdelta_y {Type IO LastRead -1 FirstWrite -1}
		Adata_p {Type I LastRead -1 FirstWrite -1}
		Adata_i {Type I LastRead -1 FirstWrite -1}
		Adata_x {Type I LastRead 16 FirstWrite -1}
		qdata {Type I LastRead 18 FirstWrite -1}
		work_Px {Type I LastRead 20 FirstWrite -1}
		work_delta_x {Type I LastRead 3 FirstWrite -1}}
	is_primal_infeasible {
		eps_prim_inf {Type I LastRead 0 FirstWrite -1}
		udata {Type I LastRead 7 FirstWrite -1}
		ldata {Type I LastRead 7 FirstWrite -1}
		work_delta_y {Type IO LastRead 17 FirstWrite 6}
		work_Atdelta_y {Type IO LastRead -1 FirstWrite -1}
		Adata_p {Type I LastRead -1 FirstWrite -1}
		Adata_i {Type I LastRead -1 FirstWrite -1}
		Adata_x {Type I LastRead 16 FirstWrite -1}}
	is_dual_infeasible {
		eps_dual_inf {Type I LastRead 0 FirstWrite -1}
		work_delta_x {Type I LastRead 3 FirstWrite -1}
		qdata {Type I LastRead 3 FirstWrite -1}}
	solve_linsys_qdldl {
		linsys_solver_P {Type I LastRead -1 FirstWrite -1}
		work_xz_tilde {Type IO LastRead 14 FirstWrite 9}
		linsys_solver_bp {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_L_p {Type I LastRead 6 FirstWrite -1}
		linsys_solver_L_x {Type I LastRead 7 FirstWrite -1}
		linsys_solver_L_i {Type I LastRead 7 FirstWrite -1}
		linsys_solver_Dinv {Type I LastRead 3 FirstWrite -1}
		linsys_solver_sol {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_rho_inv_vec {Type I LastRead 10 FirstWrite -1}}
	compute_dua_res {
		qdata {Type I LastRead 1 FirstWrite -1}
		work_x_prev {Type O LastRead -1 FirstWrite 2}
		work_Px {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "3", "EnableSignal" : "ap_enable_pp3"}
	{"Pipeline" : "4", "EnableSignal" : "ap_enable_pp4"}
	{"Pipeline" : "5", "EnableSignal" : "ap_enable_pp5"}
	{"Pipeline" : "6", "EnableSignal" : "ap_enable_pp6"}
	{"Pipeline" : "7", "EnableSignal" : "ap_enable_pp7"}
	{"Pipeline" : "8", "EnableSignal" : "ap_enable_pp8"}
	{"Pipeline" : "9", "EnableSignal" : "ap_enable_pp9"}
	{"Pipeline" : "10", "EnableSignal" : "ap_enable_pp10"}
	{"Pipeline" : "12", "EnableSignal" : "ap_enable_pp12"}
	{"Pipeline" : "13", "EnableSignal" : "ap_enable_pp13"}
	{"Pipeline" : "14", "EnableSignal" : "ap_enable_pp14"}
]}

set Spec2ImplPortList { 
	work_x { ap_memory {  { work_x_address0 mem_address 1 4 }  { work_x_ce0 mem_ce 1 1 }  { work_x_we0 mem_we 1 1 }  { work_x_d0 mem_din 1 32 }  { work_x_q0 mem_dout 0 32 } } }
	qdata { ap_memory {  { qdata_address0 mem_address 1 4 }  { qdata_ce0 mem_ce 1 1 }  { qdata_q0 mem_dout 0 32 } } }
	work_rho_inv_vec { ap_memory {  { work_rho_inv_vec_address0 mem_address 1 5 }  { work_rho_inv_vec_ce0 mem_ce 1 1 }  { work_rho_inv_vec_q0 mem_dout 0 32 } } }
	linsys_solver_L_p { ap_memory {  { linsys_solver_L_p_address0 mem_address 1 6 }  { linsys_solver_L_p_ce0 mem_ce 1 1 }  { linsys_solver_L_p_q0 mem_dout 0 6 }  { linsys_solver_L_p_address1 MemPortADDR2 1 6 }  { linsys_solver_L_p_ce1 MemPortCE2 1 1 }  { linsys_solver_L_p_q1 MemPortDOUT2 0 6 } } }
	linsys_solver_L_x { ap_memory {  { linsys_solver_L_x_address0 mem_address 1 6 }  { linsys_solver_L_x_ce0 mem_ce 1 1 }  { linsys_solver_L_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_i { ap_memory {  { linsys_solver_L_i_address0 mem_address 1 6 }  { linsys_solver_L_i_ce0 mem_ce 1 1 }  { linsys_solver_L_i_q0 mem_dout 0 6 } } }
	linsys_solver_Dinv { ap_memory {  { linsys_solver_Dinv_address0 mem_address 1 6 }  { linsys_solver_Dinv_ce0 mem_ce 1 1 }  { linsys_solver_Dinv_q0 mem_dout 0 32 } } }
	linsys_solver_rho_inv_vec { ap_memory {  { linsys_solver_rho_inv_vec_address0 mem_address 1 5 }  { linsys_solver_rho_inv_vec_ce0 mem_ce 1 1 }  { linsys_solver_rho_inv_vec_q0 mem_dout 0 32 } } }
	ldata { ap_memory {  { ldata_address0 mem_address 1 5 }  { ldata_ce0 mem_ce 1 1 }  { ldata_q0 mem_dout 0 32 } } }
	udata { ap_memory {  { udata_address0 mem_address 1 5 }  { udata_ce0 mem_ce 1 1 }  { udata_q0 mem_dout 0 32 } } }
	work_rho_vec { ap_memory {  { work_rho_vec_address0 mem_address 1 5 }  { work_rho_vec_ce0 mem_ce 1 1 }  { work_rho_vec_q0 mem_dout 0 32 } } }
	Adata_x { ap_memory {  { Adata_x_address0 mem_address 1 6 }  { Adata_x_ce0 mem_ce 1 1 }  { Adata_x_q0 mem_dout 0 32 } } }
	info_status_val { ap_ovld {  { info_status_val_i in_data 0 5 }  { info_status_val_o out_data 1 5 }  { info_status_val_o_ap_vld out_vld 1 1 } } }
}
