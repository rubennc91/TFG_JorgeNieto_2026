set moduleName compute_dua_res
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
set C_modelName {compute_dua_res}
set C_modelType { void 0 }
set C_modelArgList {
	{ qdata float 32 regular {array 15 { 1 3 } 1 1 } {global 0}  }
	{ work_x_prev float 32 regular {array 15 { 0 3 } 0 1 } {global 1}  }
	{ work_Px float 32 regular {array 15 { 0 3 } 0 1 } {global 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "qdata", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "qdata","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_x_prev", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_x_prev","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} , 
 	{ "Name" : "work_Px", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "work_Px","cData": "float","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 14,"step" : 1}]}]}], "extern" : 0} ]}
# RTL Port declarations: 
set portNum 17
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ qdata_address0 sc_out sc_lv 4 signal 0 } 
	{ qdata_ce0 sc_out sc_logic 1 signal 0 } 
	{ qdata_q0 sc_in sc_lv 32 signal 0 } 
	{ work_x_prev_address0 sc_out sc_lv 4 signal 1 } 
	{ work_x_prev_ce0 sc_out sc_logic 1 signal 1 } 
	{ work_x_prev_we0 sc_out sc_logic 1 signal 1 } 
	{ work_x_prev_d0 sc_out sc_lv 32 signal 1 } 
	{ work_Px_address0 sc_out sc_lv 4 signal 2 } 
	{ work_Px_ce0 sc_out sc_logic 1 signal 2 } 
	{ work_Px_we0 sc_out sc_logic 1 signal 2 } 
	{ work_Px_d0 sc_out sc_lv 32 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "qdata_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "qdata", "role": "address0" }} , 
 	{ "name": "qdata_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "qdata", "role": "ce0" }} , 
 	{ "name": "qdata_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "qdata", "role": "q0" }} , 
 	{ "name": "work_x_prev_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_x_prev", "role": "address0" }} , 
 	{ "name": "work_x_prev_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_x_prev", "role": "ce0" }} , 
 	{ "name": "work_x_prev_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_x_prev", "role": "we0" }} , 
 	{ "name": "work_x_prev_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_x_prev", "role": "d0" }} , 
 	{ "name": "work_Px_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "work_Px", "role": "address0" }} , 
 	{ "name": "work_Px_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_Px", "role": "ce0" }} , 
 	{ "name": "work_Px_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "work_Px", "role": "we0" }} , 
 	{ "name": "work_Px_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "work_Px", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
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
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "O"}]}]}


set ArgLastReadFirstWriteLatency {
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
]}

set Spec2ImplPortList { 
	qdata { ap_memory {  { qdata_address0 mem_address 1 4 }  { qdata_ce0 mem_ce 1 1 }  { qdata_q0 mem_dout 0 32 } } }
	work_x_prev { ap_memory {  { work_x_prev_address0 mem_address 1 4 }  { work_x_prev_ce0 mem_ce 1 1 }  { work_x_prev_we0 mem_we 1 1 }  { work_x_prev_d0 mem_din 1 32 } } }
	work_Px { ap_memory {  { work_Px_address0 mem_address 1 4 }  { work_Px_ce0 mem_ce 1 1 }  { work_Px_we0 mem_we 1 1 }  { work_Px_d0 mem_din 1 32 } } }
}
