set moduleName atualizar_restricao_v
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
set C_modelName {atualizar_restricao_v}
set C_modelType { void 0 }
set C_modelArgList {
	{ l_new float 32 regular {array 19 { 2 0 } 1 1 }  }
	{ u_new float 32 regular {array 19 { 2 0 } 1 1 }  }
	{ vdc float 32 regular  }
	{ Einv_0_0_read float 32 regular  }
	{ Einv_0_1_read float 32 regular  }
	{ Einv_1_0_read float 32 regular  }
	{ Einv_1_1_read float 32 regular  }
	{ Ax_0_read float 32 regular  }
	{ Ax_1_read float 32 regular  }
	{ ldata float 32 regular {array 19 { 0 3 } 0 1 } {global 1}  }
	{ udata float 32 regular {array 19 { 0 3 } 0 1 } {global 1}  }
	{ info_status_val int 5 regular {pointer 1} {global 1}  }
	{ work_rho_vec float 32 regular {array 19 { 2 3 } 1 1 } {global 2}  }
	{ work_rho_inv_vec float 32 regular {array 19 { 0 3 } 0 1 } {global 1}  }
	{ linsys_solver_rho_inv_vec float 32 regular {array 19 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_KKT_x float 32 regular {array 79 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_L_p int 6 regular {array 35 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_bwork int 1 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_fwork float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_D float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_iwork int 6 regular {array 102 { 0 2 } 1 1 } {global 2}  }
	{ linsys_solver_L_x float 32 regular {array 57 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_L_i int 6 regular {array 57 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_Dinv float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "l_new", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "u_new", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "vdc", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Einv_0_0_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Einv_0_1_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Einv_1_0_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Einv_1_1_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Ax_0_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Ax_1_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "ldata", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "ldata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "udata", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "udata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "info_status_val", "interface" : "wire", "bitwidth" : 5, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "info.status_val","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_rho_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_rho_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_rho_inv_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_rho_inv_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_rho_inv_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_rho_inv_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_KKT_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_KKT_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 78,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_p", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_p","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 34,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_bwork", "interface" : "memory", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_bwork","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_fwork", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_fwork","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_D", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_D","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_iwork", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_iwork","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 101,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_L_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_i", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_i","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_Dinv", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_Dinv","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 135
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ l_new_address0 sc_out sc_lv 5 signal 0 } 
	{ l_new_ce0 sc_out sc_logic 1 signal 0 } 
	{ l_new_we0 sc_out sc_logic 1 signal 0 } 
	{ l_new_d0 sc_out sc_lv 32 signal 0 } 
	{ l_new_q0 sc_in sc_lv 32 signal 0 } 
	{ l_new_address1 sc_out sc_lv 5 signal 0 } 
	{ l_new_ce1 sc_out sc_logic 1 signal 0 } 
	{ l_new_we1 sc_out sc_logic 1 signal 0 } 
	{ l_new_d1 sc_out sc_lv 32 signal 0 } 
	{ u_new_address0 sc_out sc_lv 5 signal 1 } 
	{ u_new_ce0 sc_out sc_logic 1 signal 1 } 
	{ u_new_we0 sc_out sc_logic 1 signal 1 } 
	{ u_new_d0 sc_out sc_lv 32 signal 1 } 
	{ u_new_q0 sc_in sc_lv 32 signal 1 } 
	{ u_new_address1 sc_out sc_lv 5 signal 1 } 
	{ u_new_ce1 sc_out sc_logic 1 signal 1 } 
	{ u_new_we1 sc_out sc_logic 1 signal 1 } 
	{ u_new_d1 sc_out sc_lv 32 signal 1 } 
	{ vdc sc_in sc_lv 32 signal 2 } 
	{ Einv_0_0_read sc_in sc_lv 32 signal 3 } 
	{ Einv_0_1_read sc_in sc_lv 32 signal 4 } 
	{ Einv_1_0_read sc_in sc_lv 32 signal 5 } 
	{ Einv_1_1_read sc_in sc_lv 32 signal 6 } 
	{ Ax_0_read sc_in sc_lv 32 signal 7 } 
	{ Ax_1_read sc_in sc_lv 32 signal 8 } 
	{ ldata_address0 sc_out sc_lv 5 signal 9 } 
	{ ldata_ce0 sc_out sc_logic 1 signal 9 } 
	{ ldata_we0 sc_out sc_logic 1 signal 9 } 
	{ ldata_d0 sc_out sc_lv 32 signal 9 } 
	{ udata_address0 sc_out sc_lv 5 signal 10 } 
	{ udata_ce0 sc_out sc_logic 1 signal 10 } 
	{ udata_we0 sc_out sc_logic 1 signal 10 } 
	{ udata_d0 sc_out sc_lv 32 signal 10 } 
	{ info_status_val sc_out sc_lv 5 signal 11 } 
	{ info_status_val_ap_vld sc_out sc_logic 1 outvld 11 } 
	{ work_rho_vec_address0 sc_out sc_lv 5 signal 12 } 
	{ work_rho_vec_ce0 sc_out sc_logic 1 signal 12 } 
	{ work_rho_vec_we0 sc_out sc_logic 1 signal 12 } 
	{ work_rho_vec_d0 sc_out sc_lv 32 signal 12 } 
	{ work_rho_vec_q0 sc_in sc_lv 32 signal 12 } 
	{ work_rho_inv_vec_address0 sc_out sc_lv 5 signal 13 } 
	{ work_rho_inv_vec_ce0 sc_out sc_logic 1 signal 13 } 
	{ work_rho_inv_vec_we0 sc_out sc_logic 1 signal 13 } 
	{ work_rho_inv_vec_d0 sc_out sc_lv 32 signal 13 } 
	{ linsys_solver_rho_inv_vec_address0 sc_out sc_lv 5 signal 14 } 
	{ linsys_solver_rho_inv_vec_ce0 sc_out sc_logic 1 signal 14 } 
	{ linsys_solver_rho_inv_vec_we0 sc_out sc_logic 1 signal 14 } 
	{ linsys_solver_rho_inv_vec_d0 sc_out sc_lv 32 signal 14 } 
	{ linsys_solver_rho_inv_vec_q0 sc_in sc_lv 32 signal 14 } 
	{ linsys_solver_KKT_x_address0 sc_out sc_lv 7 signal 15 } 
	{ linsys_solver_KKT_x_ce0 sc_out sc_logic 1 signal 15 } 
	{ linsys_solver_KKT_x_we0 sc_out sc_logic 1 signal 15 } 
	{ linsys_solver_KKT_x_d0 sc_out sc_lv 32 signal 15 } 
	{ linsys_solver_KKT_x_q0 sc_in sc_lv 32 signal 15 } 
	{ linsys_solver_L_p_address0 sc_out sc_lv 6 signal 16 } 
	{ linsys_solver_L_p_ce0 sc_out sc_logic 1 signal 16 } 
	{ linsys_solver_L_p_we0 sc_out sc_logic 1 signal 16 } 
	{ linsys_solver_L_p_d0 sc_out sc_lv 6 signal 16 } 
	{ linsys_solver_L_p_q0 sc_in sc_lv 6 signal 16 } 
	{ linsys_solver_bwork_address0 sc_out sc_lv 6 signal 17 } 
	{ linsys_solver_bwork_ce0 sc_out sc_logic 1 signal 17 } 
	{ linsys_solver_bwork_we0 sc_out sc_logic 1 signal 17 } 
	{ linsys_solver_bwork_d0 sc_out sc_lv 1 signal 17 } 
	{ linsys_solver_bwork_q0 sc_in sc_lv 1 signal 17 } 
	{ linsys_solver_fwork_address0 sc_out sc_lv 6 signal 18 } 
	{ linsys_solver_fwork_ce0 sc_out sc_logic 1 signal 18 } 
	{ linsys_solver_fwork_we0 sc_out sc_logic 1 signal 18 } 
	{ linsys_solver_fwork_d0 sc_out sc_lv 32 signal 18 } 
	{ linsys_solver_fwork_q0 sc_in sc_lv 32 signal 18 } 
	{ linsys_solver_D_address0 sc_out sc_lv 6 signal 19 } 
	{ linsys_solver_D_ce0 sc_out sc_logic 1 signal 19 } 
	{ linsys_solver_D_we0 sc_out sc_logic 1 signal 19 } 
	{ linsys_solver_D_d0 sc_out sc_lv 32 signal 19 } 
	{ linsys_solver_D_q0 sc_in sc_lv 32 signal 19 } 
	{ linsys_solver_iwork_address0 sc_out sc_lv 7 signal 20 } 
	{ linsys_solver_iwork_ce0 sc_out sc_logic 1 signal 20 } 
	{ linsys_solver_iwork_we0 sc_out sc_logic 1 signal 20 } 
	{ linsys_solver_iwork_d0 sc_out sc_lv 6 signal 20 } 
	{ linsys_solver_iwork_address1 sc_out sc_lv 7 signal 20 } 
	{ linsys_solver_iwork_ce1 sc_out sc_logic 1 signal 20 } 
	{ linsys_solver_iwork_we1 sc_out sc_logic 1 signal 20 } 
	{ linsys_solver_iwork_d1 sc_out sc_lv 6 signal 20 } 
	{ linsys_solver_iwork_q1 sc_in sc_lv 6 signal 20 } 
	{ linsys_solver_L_x_address0 sc_out sc_lv 6 signal 21 } 
	{ linsys_solver_L_x_ce0 sc_out sc_logic 1 signal 21 } 
	{ linsys_solver_L_x_we0 sc_out sc_logic 1 signal 21 } 
	{ linsys_solver_L_x_d0 sc_out sc_lv 32 signal 21 } 
	{ linsys_solver_L_x_q0 sc_in sc_lv 32 signal 21 } 
	{ linsys_solver_L_i_address0 sc_out sc_lv 6 signal 22 } 
	{ linsys_solver_L_i_ce0 sc_out sc_logic 1 signal 22 } 
	{ linsys_solver_L_i_we0 sc_out sc_logic 1 signal 22 } 
	{ linsys_solver_L_i_d0 sc_out sc_lv 6 signal 22 } 
	{ linsys_solver_L_i_q0 sc_in sc_lv 6 signal 22 } 
	{ linsys_solver_Dinv_address0 sc_out sc_lv 6 signal 23 } 
	{ linsys_solver_Dinv_ce0 sc_out sc_logic 1 signal 23 } 
	{ linsys_solver_Dinv_we0 sc_out sc_logic 1 signal 23 } 
	{ linsys_solver_Dinv_d0 sc_out sc_lv 32 signal 23 } 
	{ linsys_solver_Dinv_q0 sc_in sc_lv 32 signal 23 } 
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
	{ grp_fu_1250_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1250_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1250_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1250_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1256_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1256_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1256_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1256_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1264_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1264_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1264_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_1264_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_1264_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "l_new_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "l_new", "role": "address0" }} , 
 	{ "name": "l_new_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "l_new", "role": "ce0" }} , 
 	{ "name": "l_new_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "l_new", "role": "we0" }} , 
 	{ "name": "l_new_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "l_new", "role": "d0" }} , 
 	{ "name": "l_new_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "l_new", "role": "q0" }} , 
 	{ "name": "l_new_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "l_new", "role": "address1" }} , 
 	{ "name": "l_new_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "l_new", "role": "ce1" }} , 
 	{ "name": "l_new_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "l_new", "role": "we1" }} , 
 	{ "name": "l_new_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "l_new", "role": "d1" }} , 
 	{ "name": "u_new_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "u_new", "role": "address0" }} , 
 	{ "name": "u_new_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "u_new", "role": "ce0" }} , 
 	{ "name": "u_new_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "u_new", "role": "we0" }} , 
 	{ "name": "u_new_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "u_new", "role": "d0" }} , 
 	{ "name": "u_new_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "u_new", "role": "q0" }} , 
 	{ "name": "u_new_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "u_new", "role": "address1" }} , 
 	{ "name": "u_new_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "u_new", "role": "ce1" }} , 
 	{ "name": "u_new_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "u_new", "role": "we1" }} , 
 	{ "name": "u_new_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "u_new", "role": "d1" }} , 
 	{ "name": "vdc", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "vdc", "role": "default" }} , 
 	{ "name": "Einv_0_0_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Einv_0_0_read", "role": "default" }} , 
 	{ "name": "Einv_0_1_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Einv_0_1_read", "role": "default" }} , 
 	{ "name": "Einv_1_0_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Einv_1_0_read", "role": "default" }} , 
 	{ "name": "Einv_1_1_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Einv_1_1_read", "role": "default" }} , 
 	{ "name": "Ax_0_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Ax_0_read", "role": "default" }} , 
 	{ "name": "Ax_1_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Ax_1_read", "role": "default" }} , 
 	{ "name": "ldata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "ldata", "role": "address0" }} , 
 	{ "name": "ldata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ldata", "role": "ce0" }} , 
 	{ "name": "ldata_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ldata", "role": "we0" }} , 
 	{ "name": "ldata_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ldata", "role": "d0" }} , 
 	{ "name": "udata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "udata", "role": "address0" }} , 
 	{ "name": "udata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "udata", "role": "ce0" }} , 
 	{ "name": "udata_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "udata", "role": "we0" }} , 
 	{ "name": "udata_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "udata", "role": "d0" }} , 
 	{ "name": "info_status_val", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "info_status_val", "role": "default" }} , 
 	{ "name": "info_status_val_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "info_status_val", "role": "ap_vld" }} , 
 	{ "name": "work_rho_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "address0" }} , 
 	{ "name": "work_rho_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "ce0" }} , 
 	{ "name": "work_rho_vec_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "we0" }} , 
 	{ "name": "work_rho_vec_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "d0" }} , 
 	{ "name": "work_rho_vec_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_rho_vec", "role": "q0" }} , 
 	{ "name": "work_rho_inv_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "address0" }} , 
 	{ "name": "work_rho_inv_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "ce0" }} , 
 	{ "name": "work_rho_inv_vec_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "we0" }} , 
 	{ "name": "work_rho_inv_vec_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_rho_inv_vec", "role": "d0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "address0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "ce0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "we0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "d0" }} , 
 	{ "name": "linsys_solver_rho_inv_vec_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_rho_inv_vec", "role": "q0" }} , 
 	{ "name": "linsys_solver_KKT_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "address0" }} , 
 	{ "name": "linsys_solver_KKT_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "ce0" }} , 
 	{ "name": "linsys_solver_KKT_x_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "we0" }} , 
 	{ "name": "linsys_solver_KKT_x_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "d0" }} , 
 	{ "name": "linsys_solver_KKT_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "q0" }} , 
 	{ "name": "linsys_solver_L_p_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_p_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_p_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "we0" }} , 
 	{ "name": "linsys_solver_L_p_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "d0" }} , 
 	{ "name": "linsys_solver_L_p_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_p", "role": "q0" }} , 
 	{ "name": "linsys_solver_bwork_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_bwork", "role": "address0" }} , 
 	{ "name": "linsys_solver_bwork_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_bwork", "role": "ce0" }} , 
 	{ "name": "linsys_solver_bwork_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_bwork", "role": "we0" }} , 
 	{ "name": "linsys_solver_bwork_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_bwork", "role": "d0" }} , 
 	{ "name": "linsys_solver_bwork_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_bwork", "role": "q0" }} , 
 	{ "name": "linsys_solver_fwork_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_fwork", "role": "address0" }} , 
 	{ "name": "linsys_solver_fwork_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_fwork", "role": "ce0" }} , 
 	{ "name": "linsys_solver_fwork_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_fwork", "role": "we0" }} , 
 	{ "name": "linsys_solver_fwork_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_fwork", "role": "d0" }} , 
 	{ "name": "linsys_solver_fwork_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_fwork", "role": "q0" }} , 
 	{ "name": "linsys_solver_D_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_D", "role": "address0" }} , 
 	{ "name": "linsys_solver_D_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_D", "role": "ce0" }} , 
 	{ "name": "linsys_solver_D_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_D", "role": "we0" }} , 
 	{ "name": "linsys_solver_D_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_D", "role": "d0" }} , 
 	{ "name": "linsys_solver_D_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_D", "role": "q0" }} , 
 	{ "name": "linsys_solver_iwork_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "address0" }} , 
 	{ "name": "linsys_solver_iwork_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "ce0" }} , 
 	{ "name": "linsys_solver_iwork_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "we0" }} , 
 	{ "name": "linsys_solver_iwork_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "d0" }} , 
 	{ "name": "linsys_solver_iwork_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "address1" }} , 
 	{ "name": "linsys_solver_iwork_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "ce1" }} , 
 	{ "name": "linsys_solver_iwork_we1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "we1" }} , 
 	{ "name": "linsys_solver_iwork_d1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "d1" }} , 
 	{ "name": "linsys_solver_iwork_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_iwork", "role": "q1" }} , 
 	{ "name": "linsys_solver_L_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_x_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "we0" }} , 
 	{ "name": "linsys_solver_L_x_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "d0" }} , 
 	{ "name": "linsys_solver_L_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_L_x", "role": "q0" }} , 
 	{ "name": "linsys_solver_L_i_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "address0" }} , 
 	{ "name": "linsys_solver_L_i_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "ce0" }} , 
 	{ "name": "linsys_solver_L_i_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "we0" }} , 
 	{ "name": "linsys_solver_L_i_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "d0" }} , 
 	{ "name": "linsys_solver_L_i_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_L_i", "role": "q0" }} , 
 	{ "name": "linsys_solver_Dinv_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "address0" }} , 
 	{ "name": "linsys_solver_Dinv_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "ce0" }} , 
 	{ "name": "linsys_solver_Dinv_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "we0" }} , 
 	{ "name": "linsys_solver_Dinv_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "d0" }} , 
 	{ "name": "linsys_solver_Dinv_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "q0" }} , 
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
 	{ "name": "grp_fu_1250_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1250_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1250_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1250_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1250_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1250_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1250_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1250_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1256_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1256_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1256_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1256_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1256_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1256_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1256_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1256_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1264_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1264_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_1264_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1264_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1264_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1264_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "10", "11", "12"],
		"CDFG" : "atualizar_restricao_v",
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
			{"Name" : "l_new", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "u_new", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "vdc", "Type" : "None", "Direction" : "I"},
			{"Name" : "Einv_0_0_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "Einv_0_1_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "Einv_1_0_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "Einv_1_1_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "Ax_0_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "Ax_1_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "info_status_val", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "settings_rho", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "work_constr_type", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_rho_vec", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "work_rho_inv_vec", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "linsys_solver_rho_inv_vec", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_rhotoKKT", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_KKT_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_x"}]},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_p"}]},
			{"Name" : "linsys_solver_bwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_bwork"}]},
			{"Name" : "linsys_solver_fwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_fwork"}]},
			{"Name" : "linsys_solver_D", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_D"}]},
			{"Name" : "linsys_solver_iwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_iwork"}]},
			{"Name" : "linsys_solver_KKT_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_p"}]},
			{"Name" : "linsys_solver_KKT_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_i"}]},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_x"}]},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_i"}]},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_Dinv"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_constr_type_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_rhotoKKT_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647", "Parent" : "0", "Child" : ["4", "5", "6", "7", "8", "9"],
		"CDFG" : "QDLDL_factor",
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
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_bwork", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_fwork", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_D", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_iwork", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_KKT_p", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_KKT_i", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_KKT_x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "IO"}]},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.linsys_solver_KKT_p_U", "Parent" : "3"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.linsys_solver_KKT_i_U", "Parent" : "3"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.fsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "3"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "3"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.fdiv_32ns_32ns_32_16_no_dsp_1_U3", "Parent" : "3"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_647.fcmp_32ns_32ns_1_2_no_dsp_1_U4", "Parent" : "3"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fptrunc_64ns_32_2_no_dsp_1_U26", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_2_no_dsp_1_U27", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dcmp_64ns_64ns_1_2_no_dsp_1_U29", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	atualizar_restricao_v {
		l_new {Type IO LastRead 13 FirstWrite 11}
		u_new {Type IO LastRead 15 FirstWrite 11}
		vdc {Type I LastRead 0 FirstWrite -1}
		Einv_0_0_read {Type I LastRead 3 FirstWrite -1}
		Einv_0_1_read {Type I LastRead 3 FirstWrite -1}
		Einv_1_0_read {Type I LastRead 3 FirstWrite -1}
		Einv_1_1_read {Type I LastRead 3 FirstWrite -1}
		Ax_0_read {Type I LastRead 3 FirstWrite -1}
		Ax_1_read {Type I LastRead 3 FirstWrite -1}
		ldata {Type O LastRead -1 FirstWrite 14}
		udata {Type O LastRead -1 FirstWrite 16}
		info_status_val {Type O LastRead -1 FirstWrite 17}
		settings_rho {Type IO LastRead -1 FirstWrite -1}
		work_constr_type {Type I LastRead -1 FirstWrite -1}
		work_rho_vec {Type IO LastRead 45 FirstWrite 44}
		work_rho_inv_vec {Type O LastRead -1 FirstWrite 44}
		linsys_solver_rho_inv_vec {Type IO LastRead 47 FirstWrite 63}
		linsys_solver_rhotoKKT {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_x {Type IO LastRead 24 FirstWrite -1}
		linsys_solver_L_p {Type IO LastRead 27 FirstWrite 0}
		linsys_solver_bwork {Type IO LastRead 27 FirstWrite 1}
		linsys_solver_fwork {Type IO LastRead 30 FirstWrite 1}
		linsys_solver_D {Type IO LastRead 27 FirstWrite 1}
		linsys_solver_iwork {Type IO LastRead 30 FirstWrite 1}
		linsys_solver_KKT_p {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_i {Type I LastRead -1 FirstWrite -1}
		linsys_solver_L_x {Type IO LastRead 29 FirstWrite 36}
		linsys_solver_L_i {Type IO LastRead 29 FirstWrite 30}
		linsys_solver_Dinv {Type IO LastRead 30 FirstWrite 21}}
	QDLDL_factor {
		linsys_solver_L_p {Type IO LastRead 27 FirstWrite 0}
		linsys_solver_bwork {Type IO LastRead 27 FirstWrite 1}
		linsys_solver_fwork {Type IO LastRead 30 FirstWrite 1}
		linsys_solver_D {Type IO LastRead 27 FirstWrite 1}
		linsys_solver_iwork {Type IO LastRead 30 FirstWrite 1}
		linsys_solver_KKT_p {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_i {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_x {Type I LastRead 24 FirstWrite -1}
		linsys_solver_L_x {Type IO LastRead 29 FirstWrite 36}
		linsys_solver_L_i {Type IO LastRead 29 FirstWrite 30}
		linsys_solver_Dinv {Type IO LastRead 30 FirstWrite 21}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
	{"Pipeline" : "1", "EnableSignal" : "ap_enable_pp1"}
	{"Pipeline" : "2", "EnableSignal" : "ap_enable_pp2"}
	{"Pipeline" : "3", "EnableSignal" : "ap_enable_pp3"}
	{"Pipeline" : "4", "EnableSignal" : "ap_enable_pp4"}
	{"Pipeline" : "5", "EnableSignal" : "ap_enable_pp5"}
	{"Pipeline" : "6", "EnableSignal" : "ap_enable_pp6"}
	{"Pipeline" : "7", "EnableSignal" : "ap_enable_pp7"}
	{"Pipeline" : "8", "EnableSignal" : "ap_enable_pp8"}
]}

set Spec2ImplPortList { 
	l_new { ap_memory {  { l_new_address0 mem_address 1 5 }  { l_new_ce0 mem_ce 1 1 }  { l_new_we0 mem_we 1 1 }  { l_new_d0 mem_din 1 32 }  { l_new_q0 mem_dout 0 32 }  { l_new_address1 MemPortADDR2 1 5 }  { l_new_ce1 MemPortCE2 1 1 }  { l_new_we1 MemPortWE2 1 1 }  { l_new_d1 MemPortDIN2 1 32 } } }
	u_new { ap_memory {  { u_new_address0 mem_address 1 5 }  { u_new_ce0 mem_ce 1 1 }  { u_new_we0 mem_we 1 1 }  { u_new_d0 mem_din 1 32 }  { u_new_q0 mem_dout 0 32 }  { u_new_address1 MemPortADDR2 1 5 }  { u_new_ce1 MemPortCE2 1 1 }  { u_new_we1 MemPortWE2 1 1 }  { u_new_d1 MemPortDIN2 1 32 } } }
	vdc { ap_none {  { vdc in_data 0 32 } } }
	Einv_0_0_read { ap_none {  { Einv_0_0_read in_data 0 32 } } }
	Einv_0_1_read { ap_none {  { Einv_0_1_read in_data 0 32 } } }
	Einv_1_0_read { ap_none {  { Einv_1_0_read in_data 0 32 } } }
	Einv_1_1_read { ap_none {  { Einv_1_1_read in_data 0 32 } } }
	Ax_0_read { ap_none {  { Ax_0_read in_data 0 32 } } }
	Ax_1_read { ap_none {  { Ax_1_read in_data 0 32 } } }
	ldata { ap_memory {  { ldata_address0 mem_address 1 5 }  { ldata_ce0 mem_ce 1 1 }  { ldata_we0 mem_we 1 1 }  { ldata_d0 mem_din 1 32 } } }
	udata { ap_memory {  { udata_address0 mem_address 1 5 }  { udata_ce0 mem_ce 1 1 }  { udata_we0 mem_we 1 1 }  { udata_d0 mem_din 1 32 } } }
	info_status_val { ap_vld {  { info_status_val out_data 1 5 }  { info_status_val_ap_vld out_vld 1 1 } } }
	work_rho_vec { ap_memory {  { work_rho_vec_address0 mem_address 1 5 }  { work_rho_vec_ce0 mem_ce 1 1 }  { work_rho_vec_we0 mem_we 1 1 }  { work_rho_vec_d0 mem_din 1 32 }  { work_rho_vec_q0 mem_dout 0 32 } } }
	work_rho_inv_vec { ap_memory {  { work_rho_inv_vec_address0 mem_address 1 5 }  { work_rho_inv_vec_ce0 mem_ce 1 1 }  { work_rho_inv_vec_we0 mem_we 1 1 }  { work_rho_inv_vec_d0 mem_din 1 32 } } }
	linsys_solver_rho_inv_vec { ap_memory {  { linsys_solver_rho_inv_vec_address0 mem_address 1 5 }  { linsys_solver_rho_inv_vec_ce0 mem_ce 1 1 }  { linsys_solver_rho_inv_vec_we0 mem_we 1 1 }  { linsys_solver_rho_inv_vec_d0 mem_din 1 32 }  { linsys_solver_rho_inv_vec_q0 mem_dout 0 32 } } }
	linsys_solver_KKT_x { ap_memory {  { linsys_solver_KKT_x_address0 mem_address 1 7 }  { linsys_solver_KKT_x_ce0 mem_ce 1 1 }  { linsys_solver_KKT_x_we0 mem_we 1 1 }  { linsys_solver_KKT_x_d0 mem_din 1 32 }  { linsys_solver_KKT_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_p { ap_memory {  { linsys_solver_L_p_address0 mem_address 1 6 }  { linsys_solver_L_p_ce0 mem_ce 1 1 }  { linsys_solver_L_p_we0 mem_we 1 1 }  { linsys_solver_L_p_d0 mem_din 1 6 }  { linsys_solver_L_p_q0 mem_dout 0 6 } } }
	linsys_solver_bwork { ap_memory {  { linsys_solver_bwork_address0 mem_address 1 6 }  { linsys_solver_bwork_ce0 mem_ce 1 1 }  { linsys_solver_bwork_we0 mem_we 1 1 }  { linsys_solver_bwork_d0 mem_din 1 1 }  { linsys_solver_bwork_q0 mem_dout 0 1 } } }
	linsys_solver_fwork { ap_memory {  { linsys_solver_fwork_address0 mem_address 1 6 }  { linsys_solver_fwork_ce0 mem_ce 1 1 }  { linsys_solver_fwork_we0 mem_we 1 1 }  { linsys_solver_fwork_d0 mem_din 1 32 }  { linsys_solver_fwork_q0 mem_dout 0 32 } } }
	linsys_solver_D { ap_memory {  { linsys_solver_D_address0 mem_address 1 6 }  { linsys_solver_D_ce0 mem_ce 1 1 }  { linsys_solver_D_we0 mem_we 1 1 }  { linsys_solver_D_d0 mem_din 1 32 }  { linsys_solver_D_q0 mem_dout 0 32 } } }
	linsys_solver_iwork { ap_memory {  { linsys_solver_iwork_address0 mem_address 1 7 }  { linsys_solver_iwork_ce0 mem_ce 1 1 }  { linsys_solver_iwork_we0 mem_we 1 1 }  { linsys_solver_iwork_d0 mem_din 1 6 }  { linsys_solver_iwork_address1 MemPortADDR2 1 7 }  { linsys_solver_iwork_ce1 MemPortCE2 1 1 }  { linsys_solver_iwork_we1 MemPortWE2 1 1 }  { linsys_solver_iwork_d1 MemPortDIN2 1 6 }  { linsys_solver_iwork_q1 MemPortDOUT2 0 6 } } }
	linsys_solver_L_x { ap_memory {  { linsys_solver_L_x_address0 mem_address 1 6 }  { linsys_solver_L_x_ce0 mem_ce 1 1 }  { linsys_solver_L_x_we0 mem_we 1 1 }  { linsys_solver_L_x_d0 mem_din 1 32 }  { linsys_solver_L_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_i { ap_memory {  { linsys_solver_L_i_address0 mem_address 1 6 }  { linsys_solver_L_i_ce0 mem_ce 1 1 }  { linsys_solver_L_i_we0 mem_we 1 1 }  { linsys_solver_L_i_d0 mem_din 1 6 }  { linsys_solver_L_i_q0 mem_dout 0 6 } } }
	linsys_solver_Dinv { ap_memory {  { linsys_solver_Dinv_address0 mem_address 1 6 }  { linsys_solver_Dinv_ce0 mem_ce 1 1 }  { linsys_solver_Dinv_we0 mem_we 1 1 }  { linsys_solver_Dinv_d0 mem_din 1 32 }  { linsys_solver_Dinv_q0 mem_dout 0 32 } } }
}
