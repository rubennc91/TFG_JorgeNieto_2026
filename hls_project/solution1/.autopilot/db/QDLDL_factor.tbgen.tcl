set moduleName QDLDL_factor
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
set C_modelName {QDLDL_factor}
set C_modelType { void 0 }
set C_modelArgList {
	{ linsys_solver_L_p int 6 regular {array 35 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_bwork int 1 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_fwork float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_D float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_iwork int 6 regular {array 102 { 0 2 } 1 1 } {global 2}  }
	{ linsys_solver_KKT_x float 32 regular {array 79 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_L_x float 32 regular {array 57 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_L_i int 6 regular {array 57 { 2 3 } 1 1 } {global 2}  }
	{ linsys_solver_Dinv float 32 regular {array 34 { 2 3 } 1 1 } {global 2}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "linsys_solver_L_p", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_p","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 34,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_bwork", "interface" : "memory", "bitwidth" : 1, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_bwork","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_fwork", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_fwork","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_D", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_D","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_iwork", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_iwork","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 101,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_KKT_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_KKT_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 78,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_L_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_i", "interface" : "memory", "bitwidth" : 6, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_i","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_Dinv", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_Dinv","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 53
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ linsys_solver_L_p_address0 sc_out sc_lv 6 signal 0 } 
	{ linsys_solver_L_p_ce0 sc_out sc_logic 1 signal 0 } 
	{ linsys_solver_L_p_we0 sc_out sc_logic 1 signal 0 } 
	{ linsys_solver_L_p_d0 sc_out sc_lv 6 signal 0 } 
	{ linsys_solver_L_p_q0 sc_in sc_lv 6 signal 0 } 
	{ linsys_solver_bwork_address0 sc_out sc_lv 6 signal 1 } 
	{ linsys_solver_bwork_ce0 sc_out sc_logic 1 signal 1 } 
	{ linsys_solver_bwork_we0 sc_out sc_logic 1 signal 1 } 
	{ linsys_solver_bwork_d0 sc_out sc_lv 1 signal 1 } 
	{ linsys_solver_bwork_q0 sc_in sc_lv 1 signal 1 } 
	{ linsys_solver_fwork_address0 sc_out sc_lv 6 signal 2 } 
	{ linsys_solver_fwork_ce0 sc_out sc_logic 1 signal 2 } 
	{ linsys_solver_fwork_we0 sc_out sc_logic 1 signal 2 } 
	{ linsys_solver_fwork_d0 sc_out sc_lv 32 signal 2 } 
	{ linsys_solver_fwork_q0 sc_in sc_lv 32 signal 2 } 
	{ linsys_solver_D_address0 sc_out sc_lv 6 signal 3 } 
	{ linsys_solver_D_ce0 sc_out sc_logic 1 signal 3 } 
	{ linsys_solver_D_we0 sc_out sc_logic 1 signal 3 } 
	{ linsys_solver_D_d0 sc_out sc_lv 32 signal 3 } 
	{ linsys_solver_D_q0 sc_in sc_lv 32 signal 3 } 
	{ linsys_solver_iwork_address0 sc_out sc_lv 7 signal 4 } 
	{ linsys_solver_iwork_ce0 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_iwork_we0 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_iwork_d0 sc_out sc_lv 6 signal 4 } 
	{ linsys_solver_iwork_address1 sc_out sc_lv 7 signal 4 } 
	{ linsys_solver_iwork_ce1 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_iwork_we1 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_iwork_d1 sc_out sc_lv 6 signal 4 } 
	{ linsys_solver_iwork_q1 sc_in sc_lv 6 signal 4 } 
	{ linsys_solver_KKT_x_address0 sc_out sc_lv 7 signal 5 } 
	{ linsys_solver_KKT_x_ce0 sc_out sc_logic 1 signal 5 } 
	{ linsys_solver_KKT_x_q0 sc_in sc_lv 32 signal 5 } 
	{ linsys_solver_L_x_address0 sc_out sc_lv 6 signal 6 } 
	{ linsys_solver_L_x_ce0 sc_out sc_logic 1 signal 6 } 
	{ linsys_solver_L_x_we0 sc_out sc_logic 1 signal 6 } 
	{ linsys_solver_L_x_d0 sc_out sc_lv 32 signal 6 } 
	{ linsys_solver_L_x_q0 sc_in sc_lv 32 signal 6 } 
	{ linsys_solver_L_i_address0 sc_out sc_lv 6 signal 7 } 
	{ linsys_solver_L_i_ce0 sc_out sc_logic 1 signal 7 } 
	{ linsys_solver_L_i_we0 sc_out sc_logic 1 signal 7 } 
	{ linsys_solver_L_i_d0 sc_out sc_lv 6 signal 7 } 
	{ linsys_solver_L_i_q0 sc_in sc_lv 6 signal 7 } 
	{ linsys_solver_Dinv_address0 sc_out sc_lv 6 signal 8 } 
	{ linsys_solver_Dinv_ce0 sc_out sc_logic 1 signal 8 } 
	{ linsys_solver_Dinv_we0 sc_out sc_logic 1 signal 8 } 
	{ linsys_solver_Dinv_d0 sc_out sc_lv 32 signal 8 } 
	{ linsys_solver_Dinv_q0 sc_in sc_lv 32 signal 8 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
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
 	{ "name": "linsys_solver_KKT_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "address0" }} , 
 	{ "name": "linsys_solver_KKT_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "ce0" }} , 
 	{ "name": "linsys_solver_KKT_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_KKT_x", "role": "q0" }} , 
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
 	{ "name": "linsys_solver_Dinv_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "linsys_solver_Dinv", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_KKT_p_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_KKT_i_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U3", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fcmp_32ns_32ns_1_2_no_dsp_1_U4", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Pipeline" : "2", "EnableSignal" : "ap_enable_pp2"}
	{"Pipeline" : "3", "EnableSignal" : "ap_enable_pp3"}
]}

set Spec2ImplPortList { 
	linsys_solver_L_p { ap_memory {  { linsys_solver_L_p_address0 mem_address 1 6 }  { linsys_solver_L_p_ce0 mem_ce 1 1 }  { linsys_solver_L_p_we0 mem_we 1 1 }  { linsys_solver_L_p_d0 mem_din 1 6 }  { linsys_solver_L_p_q0 mem_dout 0 6 } } }
	linsys_solver_bwork { ap_memory {  { linsys_solver_bwork_address0 mem_address 1 6 }  { linsys_solver_bwork_ce0 mem_ce 1 1 }  { linsys_solver_bwork_we0 mem_we 1 1 }  { linsys_solver_bwork_d0 mem_din 1 1 }  { linsys_solver_bwork_q0 mem_dout 0 1 } } }
	linsys_solver_fwork { ap_memory {  { linsys_solver_fwork_address0 mem_address 1 6 }  { linsys_solver_fwork_ce0 mem_ce 1 1 }  { linsys_solver_fwork_we0 mem_we 1 1 }  { linsys_solver_fwork_d0 mem_din 1 32 }  { linsys_solver_fwork_q0 mem_dout 0 32 } } }
	linsys_solver_D { ap_memory {  { linsys_solver_D_address0 mem_address 1 6 }  { linsys_solver_D_ce0 mem_ce 1 1 }  { linsys_solver_D_we0 mem_we 1 1 }  { linsys_solver_D_d0 mem_din 1 32 }  { linsys_solver_D_q0 mem_dout 0 32 } } }
	linsys_solver_iwork { ap_memory {  { linsys_solver_iwork_address0 mem_address 1 7 }  { linsys_solver_iwork_ce0 mem_ce 1 1 }  { linsys_solver_iwork_we0 mem_we 1 1 }  { linsys_solver_iwork_d0 mem_din 1 6 }  { linsys_solver_iwork_address1 MemPortADDR2 1 7 }  { linsys_solver_iwork_ce1 MemPortCE2 1 1 }  { linsys_solver_iwork_we1 MemPortWE2 1 1 }  { linsys_solver_iwork_d1 MemPortDIN2 1 6 }  { linsys_solver_iwork_q1 MemPortDOUT2 0 6 } } }
	linsys_solver_KKT_x { ap_memory {  { linsys_solver_KKT_x_address0 mem_address 1 7 }  { linsys_solver_KKT_x_ce0 mem_ce 1 1 }  { linsys_solver_KKT_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_x { ap_memory {  { linsys_solver_L_x_address0 mem_address 1 6 }  { linsys_solver_L_x_ce0 mem_ce 1 1 }  { linsys_solver_L_x_we0 mem_we 1 1 }  { linsys_solver_L_x_d0 mem_din 1 32 }  { linsys_solver_L_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_i { ap_memory {  { linsys_solver_L_i_address0 mem_address 1 6 }  { linsys_solver_L_i_ce0 mem_ce 1 1 }  { linsys_solver_L_i_we0 mem_we 1 1 }  { linsys_solver_L_i_d0 mem_din 1 6 }  { linsys_solver_L_i_q0 mem_dout 0 6 } } }
	linsys_solver_Dinv { ap_memory {  { linsys_solver_Dinv_address0 mem_address 1 6 }  { linsys_solver_Dinv_ce0 mem_ce 1 1 }  { linsys_solver_Dinv_we0 mem_we 1 1 }  { linsys_solver_Dinv_d0 mem_din 1 32 }  { linsys_solver_Dinv_q0 mem_dout 0 32 } } }
}
