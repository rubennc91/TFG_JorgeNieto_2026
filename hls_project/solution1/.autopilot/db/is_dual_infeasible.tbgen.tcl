set moduleName is_dual_infeasible
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
set C_modelName {is_dual_infeasible}
set C_modelType { void 0 }
set C_modelArgList {
	{ eps_dual_inf float 32 regular  }
	{ work_delta_x float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
	{ qdata float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "eps_dual_inf", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "work_delta_x", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_delta_x","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "qdata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "qdata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 22
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ eps_dual_inf sc_in sc_lv 32 signal 0 } 
	{ work_delta_x_address0 sc_out sc_lv 4 signal 1 } 
	{ work_delta_x_ce0 sc_out sc_logic 1 signal 1 } 
	{ work_delta_x_q0 sc_in sc_lv 32 signal 1 } 
	{ qdata_address0 sc_out sc_lv 4 signal 2 } 
	{ qdata_ce0 sc_out sc_logic 1 signal 2 } 
	{ qdata_q0 sc_in sc_lv 32 signal 2 } 
	{ grp_fu_350_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_350_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_350_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_350_p_ce sc_out sc_logic 1 signal -1 } 
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
 	{ "name": "eps_dual_inf", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "eps_dual_inf", "role": "default" }} , 
 	{ "name": "work_delta_x_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_delta_x", "role": "address0" }} , 
 	{ "name": "work_delta_x_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_delta_x", "role": "ce0" }} , 
 	{ "name": "work_delta_x_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_delta_x", "role": "q0" }} , 
 	{ "name": "qdata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "qdata", "role": "address0" }} , 
 	{ "name": "qdata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "qdata", "role": "ce0" }} , 
 	{ "name": "qdata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "qdata", "role": "q0" }} , 
 	{ "name": "grp_fu_350_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_350_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_350_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_350_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_359_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_359_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_359_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_359_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_359_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.facc_32ns_32ns_1ns_32_6_no_dsp_1_U96", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	is_dual_infeasible {
		eps_dual_inf {Type I LastRead 0 FirstWrite -1}
		work_delta_x {Type I LastRead 3 FirstWrite -1}
		qdata {Type I LastRead 3 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "93", "Max" : "278"}
	, {"Name" : "Interval", "Min" : "93", "Max" : "278"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	eps_dual_inf { ap_none {  { eps_dual_inf in_data 0 32 } } }
	work_delta_x { ap_memory {  { work_delta_x_address0 mem_address 1 4 }  { work_delta_x_ce0 mem_ce 1 1 }  { work_delta_x_q0 mem_dout 0 32 } } }
	qdata { ap_memory {  { qdata_address0 mem_address 1 4 }  { qdata_ce0 mem_ce 1 1 }  { qdata_q0 mem_dout 0 32 } } }
}
