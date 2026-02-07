set moduleName check_termination
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
set C_modelName {check_termination}
set C_modelType { int 1 }
set C_modelArgList {
	{ approximate int 1 regular  }
	{ info_pri_res float 32 regular {pointer 0} {global 0}  }
	{ info_status_val int 5 regular {pointer 1} {global 1}  }
	{ work_z float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ work_Ax float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ udata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ ldata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ work_delta_y float 32 regular {array 19 { 2 3 } 1 1 } {global 2}  }
	{ Adata_x float 32 regular {array 43 { 1 3 } 1 1 } {global 0}  }
	{ qdata float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
	{ work_Px float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
	{ work_delta_x float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "approximate", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "info_pri_res", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "info.pri_res","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "info_status_val", "interface" : "wire", "bitwidth" : 5, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "info.status_val","cData": "long long int","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_z", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_z","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_Ax", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_Ax","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "udata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "udata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ldata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "ldata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_delta_y", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_delta_y","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "Adata_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "Adata_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 42,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "qdata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "qdata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_Px", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_Px","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_delta_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_delta_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 1} ]}
# RTL Port declarations: 
set portNum 40
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ approximate sc_in sc_lv 1 signal 0 } 
	{ info_pri_res sc_in sc_lv 32 signal 1 } 
	{ info_status_val sc_out sc_lv 5 signal 2 } 
	{ info_status_val_ap_vld sc_out sc_logic 1 outvld 2 } 
	{ work_z_address0 sc_out sc_lv 5 signal 3 } 
	{ work_z_ce0 sc_out sc_logic 1 signal 3 } 
	{ work_z_q0 sc_in sc_lv 32 signal 3 } 
	{ work_Ax_address0 sc_out sc_lv 5 signal 4 } 
	{ work_Ax_ce0 sc_out sc_logic 1 signal 4 } 
	{ work_Ax_q0 sc_in sc_lv 32 signal 4 } 
	{ udata_address0 sc_out sc_lv 5 signal 5 } 
	{ udata_ce0 sc_out sc_logic 1 signal 5 } 
	{ udata_q0 sc_in sc_lv 32 signal 5 } 
	{ ldata_address0 sc_out sc_lv 5 signal 6 } 
	{ ldata_ce0 sc_out sc_logic 1 signal 6 } 
	{ ldata_q0 sc_in sc_lv 32 signal 6 } 
	{ work_delta_y_address0 sc_out sc_lv 5 signal 7 } 
	{ work_delta_y_ce0 sc_out sc_logic 1 signal 7 } 
	{ work_delta_y_we0 sc_out sc_logic 1 signal 7 } 
	{ work_delta_y_d0 sc_out sc_lv 32 signal 7 } 
	{ work_delta_y_q0 sc_in sc_lv 32 signal 7 } 
	{ Adata_x_address0 sc_out sc_lv 6 signal 8 } 
	{ Adata_x_ce0 sc_out sc_logic 1 signal 8 } 
	{ Adata_x_q0 sc_in sc_lv 32 signal 8 } 
	{ qdata_address0 sc_out sc_lv 4 signal 9 } 
	{ qdata_ce0 sc_out sc_logic 1 signal 9 } 
	{ qdata_q0 sc_in sc_lv 32 signal 9 } 
	{ work_Px_address0 sc_out sc_lv 4 signal 10 } 
	{ work_Px_ce0 sc_out sc_logic 1 signal 10 } 
	{ work_Px_q0 sc_in sc_lv 32 signal 10 } 
	{ work_delta_x_address0 sc_out sc_lv 4 signal 11 } 
	{ work_delta_x_ce0 sc_out sc_logic 1 signal 11 } 
	{ work_delta_x_q0 sc_in sc_lv 32 signal 11 } 
	{ ap_return sc_out sc_lv 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "approximate", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "approximate", "role": "default" }} , 
 	{ "name": "info_pri_res", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "info_pri_res", "role": "default" }} , 
 	{ "name": "info_status_val", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "info_status_val", "role": "default" }} , 
 	{ "name": "info_status_val_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "info_status_val", "role": "ap_vld" }} , 
 	{ "name": "work_z_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_z", "role": "address0" }} , 
 	{ "name": "work_z_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_z", "role": "ce0" }} , 
 	{ "name": "work_z_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_z", "role": "q0" }} , 
 	{ "name": "work_Ax_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_Ax", "role": "address0" }} , 
 	{ "name": "work_Ax_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_Ax", "role": "ce0" }} , 
 	{ "name": "work_Ax_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_Ax", "role": "q0" }} , 
 	{ "name": "udata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "udata", "role": "address0" }} , 
 	{ "name": "udata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "udata", "role": "ce0" }} , 
 	{ "name": "udata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "udata", "role": "q0" }} , 
 	{ "name": "ldata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "ldata", "role": "address0" }} , 
 	{ "name": "ldata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ldata", "role": "ce0" }} , 
 	{ "name": "ldata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ldata", "role": "q0" }} , 
 	{ "name": "work_delta_y_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "work_delta_y", "role": "address0" }} , 
 	{ "name": "work_delta_y_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_delta_y", "role": "ce0" }} , 
 	{ "name": "work_delta_y_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_delta_y", "role": "we0" }} , 
 	{ "name": "work_delta_y_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_delta_y", "role": "d0" }} , 
 	{ "name": "work_delta_y_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_delta_y", "role": "q0" }} , 
 	{ "name": "Adata_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Adata_x", "role": "address0" }} , 
 	{ "name": "Adata_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Adata_x", "role": "ce0" }} , 
 	{ "name": "Adata_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Adata_x", "role": "q0" }} , 
 	{ "name": "qdata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "qdata", "role": "address0" }} , 
 	{ "name": "qdata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "qdata", "role": "ce0" }} , 
 	{ "name": "qdata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "qdata", "role": "q0" }} , 
 	{ "name": "work_Px_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_Px", "role": "address0" }} , 
 	{ "name": "work_Px_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_Px", "role": "ce0" }} , 
 	{ "name": "work_Px_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_Px", "role": "q0" }} , 
 	{ "name": "work_delta_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_delta_x", "role": "address0" }} , 
 	{ "name": "work_delta_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_delta_x", "role": "ce0" }} , 
 	{ "name": "work_delta_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_delta_x", "role": "q0" }} , 
 	{ "name": "ap_return", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_return", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "10", "12", "13", "14", "15", "16"],
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
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "udata"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "ldata"}]},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_delta_y"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_Atdelta_y"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_p"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_i"}]},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_x"}]},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "qdata"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "work_delta_x"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9"],
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
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.work_Atdelta_y_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.Adata_p_U", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.Adata_i_U", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U78", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U79", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U82", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U83", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_primal_infeasible_fu_317.facc_32ns_32ns_1ns_32_6_no_dsp_1_U84", "Parent" : "1"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_is_dual_infeasible_fu_337", "Parent" : "0", "Child" : ["11"],
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_is_dual_infeasible_fu_337.facc_32ns_32ns_1ns_32_6_no_dsp_1_U96", "Parent" : "10"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U100", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U101", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fcmp_32ns_32ns_1_2_no_dsp_1_U102", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fcmp_32ns_32ns_1_2_no_dsp_1_U103", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U104", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		qdata {Type I LastRead 3 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	approximate { ap_none {  { approximate in_data 0 1 } } }
	info_pri_res { ap_none {  { info_pri_res in_data 0 32 } } }
	info_status_val { ap_vld {  { info_status_val out_data 1 5 }  { info_status_val_ap_vld out_vld 1 1 } } }
	work_z { ap_memory {  { work_z_address0 mem_address 1 5 }  { work_z_ce0 mem_ce 1 1 }  { work_z_q0 mem_dout 0 32 } } }
	work_Ax { ap_memory {  { work_Ax_address0 mem_address 1 5 }  { work_Ax_ce0 mem_ce 1 1 }  { work_Ax_q0 mem_dout 0 32 } } }
	udata { ap_memory {  { udata_address0 mem_address 1 5 }  { udata_ce0 mem_ce 1 1 }  { udata_q0 mem_dout 0 32 } } }
	ldata { ap_memory {  { ldata_address0 mem_address 1 5 }  { ldata_ce0 mem_ce 1 1 }  { ldata_q0 mem_dout 0 32 } } }
	work_delta_y { ap_memory {  { work_delta_y_address0 mem_address 1 5 }  { work_delta_y_ce0 mem_ce 1 1 }  { work_delta_y_we0 mem_we 1 1 }  { work_delta_y_d0 mem_din 1 32 }  { work_delta_y_q0 mem_dout 0 32 } } }
	Adata_x { ap_memory {  { Adata_x_address0 mem_address 1 6 }  { Adata_x_ce0 mem_ce 1 1 }  { Adata_x_q0 mem_dout 0 32 } } }
	qdata { ap_memory {  { qdata_address0 mem_address 1 4 }  { qdata_ce0 mem_ce 1 1 }  { qdata_q0 mem_dout 0 32 } } }
	work_Px { ap_memory {  { work_Px_address0 mem_address 1 4 }  { work_Px_ce0 mem_ce 1 1 }  { work_Px_q0 mem_dout 0 32 } } }
	work_delta_x { ap_memory {  { work_delta_x_address0 mem_address 1 4 }  { work_delta_x_ce0 mem_ce 1 1 }  { work_delta_x_q0 mem_dout 0 32 } } }
}
