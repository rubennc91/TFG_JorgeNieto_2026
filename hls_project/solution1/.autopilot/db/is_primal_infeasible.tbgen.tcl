set moduleName is_primal_infeasible
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
set C_modelName {is_primal_infeasible}
set C_modelType { int 1 }
set C_modelArgList {
	{ eps_prim_inf float 32 regular  }
	{ udata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ ldata float 32 regular {array 19 { 1 3 } 1 1 } {global 0}  }
	{ work_delta_y float 32 regular {array 19 { 2 3 } 1 1 } {global 2}  }
	{ Adata_x float 32 regular {array 43 { 1 3 } 1 1 } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "eps_prim_inf", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "udata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "udata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ldata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "ldata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_delta_y", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_delta_y","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 18,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "Adata_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "Adata_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 42,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 1} ]}
# RTL Port declarations: 
set portNum 45
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ eps_prim_inf sc_in sc_lv 32 signal 0 } 
	{ udata_address0 sc_out sc_lv 5 signal 1 } 
	{ udata_ce0 sc_out sc_logic 1 signal 1 } 
	{ udata_q0 sc_in sc_lv 32 signal 1 } 
	{ ldata_address0 sc_out sc_lv 5 signal 2 } 
	{ ldata_ce0 sc_out sc_logic 1 signal 2 } 
	{ ldata_q0 sc_in sc_lv 32 signal 2 } 
	{ work_delta_y_address0 sc_out sc_lv 5 signal 3 } 
	{ work_delta_y_ce0 sc_out sc_logic 1 signal 3 } 
	{ work_delta_y_we0 sc_out sc_logic 1 signal 3 } 
	{ work_delta_y_d0 sc_out sc_lv 32 signal 3 } 
	{ work_delta_y_q0 sc_in sc_lv 32 signal 3 } 
	{ Adata_x_address0 sc_out sc_lv 6 signal 4 } 
	{ Adata_x_ce0 sc_out sc_logic 1 signal 4 } 
	{ Adata_x_q0 sc_in sc_lv 32 signal 4 } 
	{ ap_return sc_out sc_lv 1 signal -1 } 
	{ grp_fu_346_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_346_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_346_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_346_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_346_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_350_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_350_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_350_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_350_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_1845_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1845_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_1845_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_1845_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_354_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_354_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_354_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_354_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_354_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_359_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_359_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_359_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_359_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_359_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "eps_prim_inf", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "eps_prim_inf", "role": "default" }} , 
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
 	{ "name": "ap_return", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_return", "role": "default" }} , 
 	{ "name": "grp_fu_346_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_346_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_346_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_346_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_346_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_346_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_346_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_346_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_346_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_346_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_350_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_1845_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1845_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_1845_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1845_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_1845_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_1845_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_1845_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_1845_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_354_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_354_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_354_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_354_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_354_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_354_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_354_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_354_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_354_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_354_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_359_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_359_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_359_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_359_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_359_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_Atdelta_y_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Adata_p_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Adata_i_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_2_no_dsp_1_U78", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_2_no_dsp_1_U79", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dcmp_64ns_64ns_1_2_no_dsp_1_U82", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dcmp_64ns_64ns_1_2_no_dsp_1_U83", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.facc_32ns_32ns_1ns_32_6_no_dsp_1_U84", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	is_primal_infeasible {
		eps_prim_inf {Type I LastRead 0 FirstWrite -1}
		udata {Type I LastRead 7 FirstWrite -1}
		ldata {Type I LastRead 7 FirstWrite -1}
		work_delta_y {Type IO LastRead 17 FirstWrite 6}
		work_Atdelta_y {Type IO LastRead -1 FirstWrite -1}
		Adata_p {Type I LastRead -1 FirstWrite -1}
		Adata_i {Type I LastRead -1 FirstWrite -1}
		Adata_x {Type I LastRead 16 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
	{"Pipeline" : "1", "EnableSignal" : "ap_enable_pp1"}
]}

set Spec2ImplPortList { 
	eps_prim_inf { ap_none {  { eps_prim_inf in_data 0 32 } } }
	udata { ap_memory {  { udata_address0 mem_address 1 5 }  { udata_ce0 mem_ce 1 1 }  { udata_q0 mem_dout 0 32 } } }
	ldata { ap_memory {  { ldata_address0 mem_address 1 5 }  { ldata_ce0 mem_ce 1 1 }  { ldata_q0 mem_dout 0 32 } } }
	work_delta_y { ap_memory {  { work_delta_y_address0 mem_address 1 5 }  { work_delta_y_ce0 mem_ce 1 1 }  { work_delta_y_we0 mem_we 1 1 }  { work_delta_y_d0 mem_din 1 32 }  { work_delta_y_q0 mem_dout 0 32 } } }
	Adata_x { ap_memory {  { Adata_x_address0 mem_address 1 6 }  { Adata_x_ce0 mem_ce 1 1 }  { Adata_x_q0 mem_dout 0 32 } } }
}
