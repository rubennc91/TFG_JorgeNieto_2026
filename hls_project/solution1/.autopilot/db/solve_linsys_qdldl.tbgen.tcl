set moduleName solve_linsys_qdldl
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
set C_modelName {solve_linsys_qdldl}
set C_modelType { void 0 }
set C_modelArgList {
	{ work_xz_tilde float 32 regular {array 34 { 2 1 } 1 1 } {global 2}  }
	{ linsys_solver_L_p int 6 regular {array 35 { 1 1 } 1 1 } {global 0}  }
	{ linsys_solver_L_x float 32 regular {array 57 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_L_i int 6 regular {array 57 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_Dinv float 32 regular {array 34 { 1 3 } 1 1 } {global 0}  }
	{ linsys_solver_rho_inv_vec float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "work_xz_tilde", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_xz_tilde","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_p", "interface" : "memory", "bitwidth" : 6, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_p","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 34,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_L_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_L_i", "interface" : "memory", "bitwidth" : 6, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "linsys_solver_L_i","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 56,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_Dinv", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_Dinv","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 33,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "linsys_solver_rho_inv_vec", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "linsys_solver_rho_inv_vec","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 41
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ work_xz_tilde_address0 sc_out sc_lv 6 signal 0 } 
	{ work_xz_tilde_ce0 sc_out sc_logic 1 signal 0 } 
	{ work_xz_tilde_we0 sc_out sc_logic 1 signal 0 } 
	{ work_xz_tilde_d0 sc_out sc_lv 32 signal 0 } 
	{ work_xz_tilde_q0 sc_in sc_lv 32 signal 0 } 
	{ work_xz_tilde_address1 sc_out sc_lv 6 signal 0 } 
	{ work_xz_tilde_ce1 sc_out sc_logic 1 signal 0 } 
	{ work_xz_tilde_q1 sc_in sc_lv 32 signal 0 } 
	{ linsys_solver_L_p_address0 sc_out sc_lv 6 signal 1 } 
	{ linsys_solver_L_p_ce0 sc_out sc_logic 1 signal 1 } 
	{ linsys_solver_L_p_q0 sc_in sc_lv 6 signal 1 } 
	{ linsys_solver_L_p_address1 sc_out sc_lv 6 signal 1 } 
	{ linsys_solver_L_p_ce1 sc_out sc_logic 1 signal 1 } 
	{ linsys_solver_L_p_q1 sc_in sc_lv 6 signal 1 } 
	{ linsys_solver_L_x_address0 sc_out sc_lv 6 signal 2 } 
	{ linsys_solver_L_x_ce0 sc_out sc_logic 1 signal 2 } 
	{ linsys_solver_L_x_q0 sc_in sc_lv 32 signal 2 } 
	{ linsys_solver_L_i_address0 sc_out sc_lv 6 signal 3 } 
	{ linsys_solver_L_i_ce0 sc_out sc_logic 1 signal 3 } 
	{ linsys_solver_L_i_q0 sc_in sc_lv 6 signal 3 } 
	{ linsys_solver_Dinv_address0 sc_out sc_lv 6 signal 4 } 
	{ linsys_solver_Dinv_ce0 sc_out sc_logic 1 signal 4 } 
	{ linsys_solver_Dinv_q0 sc_in sc_lv 32 signal 4 } 
	{ linsys_solver_rho_inv_vec_address0 sc_out sc_lv 5 signal 5 } 
	{ linsys_solver_rho_inv_vec_ce0 sc_out sc_logic 1 signal 5 } 
	{ linsys_solver_rho_inv_vec_q0 sc_in sc_lv 32 signal 5 } 
	{ grp_fu_1008_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1008_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1008_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_1008_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1008_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1020_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1020_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1020_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1020_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "work_xz_tilde_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "address0" }} , 
 	{ "name": "work_xz_tilde_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "ce0" }} , 
 	{ "name": "work_xz_tilde_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "we0" }} , 
 	{ "name": "work_xz_tilde_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "d0" }} , 
 	{ "name": "work_xz_tilde_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "q0" }} , 
 	{ "name": "work_xz_tilde_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "address1" }} , 
 	{ "name": "work_xz_tilde_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "ce1" }} , 
 	{ "name": "work_xz_tilde_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_xz_tilde", "role": "q1" }} , 
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
 	{ "name": "grp_fu_1008_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1008_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1008_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1008_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1008_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_1008_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_1008_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1008_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1008_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1008_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1020_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1020_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1020_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1020_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1020_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1020_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1020_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1020_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_P_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_bp_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_sol_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	solve_linsys_qdldl {
		linsys_solver_P {Type I LastRead -1 FirstWrite -1}
		work_xz_tilde {Type IO LastRead 14 FirstWrite 9}
		linsys_solver_bp {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_L_p {Type I LastRead 6 FirstWrite -1}
		linsys_solver_L_x {Type I LastRead 7 FirstWrite -1}
		linsys_solver_L_i {Type I LastRead 7 FirstWrite -1}
		linsys_solver_Dinv {Type I LastRead 3 FirstWrite -1}
		linsys_solver_sol {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_rho_inv_vec {Type I LastRead 10 FirstWrite -1}}}

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
]}

set Spec2ImplPortList { 
	work_xz_tilde { ap_memory {  { work_xz_tilde_address0 mem_address 1 6 }  { work_xz_tilde_ce0 mem_ce 1 1 }  { work_xz_tilde_we0 mem_we 1 1 }  { work_xz_tilde_d0 mem_din 1 32 }  { work_xz_tilde_q0 mem_dout 0 32 }  { work_xz_tilde_address1 MemPortADDR2 1 6 }  { work_xz_tilde_ce1 MemPortCE2 1 1 }  { work_xz_tilde_q1 MemPortDOUT2 0 32 } } }
	linsys_solver_L_p { ap_memory {  { linsys_solver_L_p_address0 mem_address 1 6 }  { linsys_solver_L_p_ce0 mem_ce 1 1 }  { linsys_solver_L_p_q0 mem_dout 0 6 }  { linsys_solver_L_p_address1 MemPortADDR2 1 6 }  { linsys_solver_L_p_ce1 MemPortCE2 1 1 }  { linsys_solver_L_p_q1 MemPortDOUT2 0 6 } } }
	linsys_solver_L_x { ap_memory {  { linsys_solver_L_x_address0 mem_address 1 6 }  { linsys_solver_L_x_ce0 mem_ce 1 1 }  { linsys_solver_L_x_q0 mem_dout 0 32 } } }
	linsys_solver_L_i { ap_memory {  { linsys_solver_L_i_address0 mem_address 1 6 }  { linsys_solver_L_i_ce0 mem_ce 1 1 }  { linsys_solver_L_i_q0 mem_dout 0 6 } } }
	linsys_solver_Dinv { ap_memory {  { linsys_solver_Dinv_address0 mem_address 1 6 }  { linsys_solver_Dinv_ce0 mem_ce 1 1 }  { linsys_solver_Dinv_q0 mem_dout 0 32 } } }
	linsys_solver_rho_inv_vec { ap_memory {  { linsys_solver_rho_inv_vec_address0 mem_address 1 5 }  { linsys_solver_rho_inv_vec_ce0 mem_ce 1 1 }  { linsys_solver_rho_inv_vec_q0 mem_dout 0 32 } } }
}
