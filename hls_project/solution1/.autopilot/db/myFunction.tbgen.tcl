set moduleName myFunction
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {myFunction}
set C_modelType { void 0 }
set C_modelArgList {
	{ x_ini_0 int 32 regular {axi_slave 0}  }
	{ x_ini_1 int 32 regular {axi_slave 0}  }
	{ x_ini_2 int 32 regular {axi_slave 0}  }
	{ Vsd float 32 regular {axi_slave 0}  }
	{ Vsq float 32 regular {axi_slave 0}  }
	{ iL float 32 regular {axi_slave 0}  }
	{ u00_0 int 32 regular {axi_slave 0}  }
	{ u00_1 int 32 regular {axi_slave 0}  }
	{ outputVector_0 int 32 regular {axi_slave 1}  }
	{ outputVector_1 int 32 regular {axi_slave 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "x_ini_0", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "x_ini_0","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "x_ini_1", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "x_ini_1","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "x_ini_2", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "x_ini_2","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":32}, "offset_end" : {"in":39}} , 
 	{ "Name" : "Vsd", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "Vsd","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":40}, "offset_end" : {"in":47}} , 
 	{ "Name" : "Vsq", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "Vsq","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":48}, "offset_end" : {"in":55}} , 
 	{ "Name" : "iL", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "iL","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":56}, "offset_end" : {"in":63}} , 
 	{ "Name" : "u00_0", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "u00_0","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":64}, "offset_end" : {"in":71}} , 
 	{ "Name" : "u00_1", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "u00_1","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"in":72}, "offset_end" : {"in":79}} , 
 	{ "Name" : "outputVector_0", "interface" : "axi_slave", "bundle":"control","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "outputVector_0","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"out":80}, "offset_end" : {"out":87}} , 
 	{ "Name" : "outputVector_1", "interface" : "axi_slave", "bundle":"control","type":"ap_vld","bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "outputVector_1","cData": "int","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 0}]}]}], "offset" : {"out":96}, "offset_end" : {"out":103}} ]}
# RTL Port declarations: 
set portNum 20
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"myFunction","role":"start","value":"0","valid_bit":"0"},{"name":"myFunction","role":"continue","value":"0","valid_bit":"4"},{"name":"myFunction","role":"auto_start","value":"0","valid_bit":"7"},{"name":"x_ini_0","role":"data","value":"16"},{"name":"x_ini_1","role":"data","value":"24"},{"name":"x_ini_2","role":"data","value":"32"},{"name":"Vsd","role":"data","value":"40"},{"name":"Vsq","role":"data","value":"48"},{"name":"iL","role":"data","value":"56"},{"name":"u00_0","role":"data","value":"64"},{"name":"u00_1","role":"data","value":"72"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"myFunction","role":"start","value":"0","valid_bit":"0"},{"name":"myFunction","role":"done","value":"0","valid_bit":"1"},{"name":"myFunction","role":"idle","value":"0","valid_bit":"2"},{"name":"myFunction","role":"ready","value":"0","valid_bit":"3"},{"name":"myFunction","role":"auto_start","value":"0","valid_bit":"7"},{"name":"outputVector_0","role":"data","value":"80"}, {"name":"outputVector_0","role":"valid","value":"84","valid_bit":"0"},{"name":"outputVector_1","role":"data","value":"96"}, {"name":"outputVector_1","role":"valid","value":"100","valid_bit":"0"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "64", "77", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97"],
		"CDFG" : "myFunction",
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
			{"Name" : "x_ini_0", "Type" : "None", "Direction" : "I"},
			{"Name" : "x_ini_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "x_ini_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "Vsd", "Type" : "None", "Direction" : "I"},
			{"Name" : "Vsq", "Type" : "None", "Direction" : "I"},
			{"Name" : "iL", "Type" : "None", "Direction" : "I"},
			{"Name" : "u00_0", "Type" : "None", "Direction" : "I"},
			{"Name" : "u00_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "outputVector_0", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "outputVector_1", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "is_initialized", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "qdata"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "ldata"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "ldata"}]},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "udata"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "udata"}]},
			{"Name" : "info_status_val", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "info_status_val"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "info_status_val"}]},
			{"Name" : "settings_rho", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "settings_rho"}]},
			{"Name" : "work_constr_type", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "work_constr_type"}]},
			{"Name" : "work_rho_vec", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_rho_vec"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "work_rho_vec"}]},
			{"Name" : "work_rho_inv_vec", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_rho_inv_vec"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "work_rho_inv_vec"}]},
			{"Name" : "linsys_solver_rho_inv_vec", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_rho_inv_vec"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_rho_inv_vec"}]},
			{"Name" : "linsys_solver_rhotoKKT", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_rhotoKKT"}]},
			{"Name" : "linsys_solver_KKT_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_KKT_x"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_KKT_x"}]},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_L_p"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_L_p"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_L_p"}]},
			{"Name" : "linsys_solver_bwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_bwork"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_bwork"}]},
			{"Name" : "linsys_solver_fwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_fwork"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_fwork"}]},
			{"Name" : "linsys_solver_D", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_D"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_D"}]},
			{"Name" : "linsys_solver_iwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_iwork"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_iwork"}]},
			{"Name" : "linsys_solver_KKT_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_KKT_p"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_KKT_p"}]},
			{"Name" : "linsys_solver_KKT_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_KKT_i"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_KKT_i"}]},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_L_x"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_L_x"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_L_x"}]},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_L_i"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_L_i"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_L_i"}]},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_Dinv"},
					{"ID" : "64", "SubInstance" : "grp_atualizar_restricao_v_fu_1119", "Port" : "linsys_solver_Dinv"},
					{"ID" : "77", "SubInstance" : "grp_QDLDL_factor_fu_1172", "Port" : "linsys_solver_Dinv"}]},
			{"Name" : "atualizar_A_A_idx", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "Adata_x"}]},
			{"Name" : "Pdata_x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_PtoKKT", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_Pdiag_idx", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_AtoKKT", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_x"}]},
			{"Name" : "work_z", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_z"}]},
			{"Name" : "work_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_y"}]},
			{"Name" : "work_x_prev", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_x_prev"}]},
			{"Name" : "work_z_prev", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_z_prev"}]},
			{"Name" : "work_xz_tilde", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_xz_tilde"}]},
			{"Name" : "linsys_solver_P", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_P"}]},
			{"Name" : "linsys_solver_bp", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_bp"}]},
			{"Name" : "linsys_solver_sol", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "linsys_solver_sol"}]},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_delta_x"}]},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_delta_y"}]},
			{"Name" : "work_Ax", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_Ax"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "Adata_p"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "Adata_i"}]},
			{"Name" : "info_pri_res", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "info_pri_res"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_Px"}]},
			{"Name" : "info_dua_res", "Type" : "None", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "info_dua_res"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "28", "SubInstance" : "grp_osqp_solve_fu_1055", "Port" : "work_Atdelta_y"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.qdata_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ldata_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.udata_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_rho_vec_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_rho_inv_vec_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_rho_inv_vec_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_KKT_x_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_L_p_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_bwork_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_fwork_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_D_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_iwork_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_L_x_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_L_i_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_Dinv_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.atualizar_A_A_idx_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Adata_x_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Pdata_x_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_PtoKKT_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_Pdiag_idx_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.linsys_solver_AtoKKT_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.work_x_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_new_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.q_new_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.l_new_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.u_new_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055", "Parent" : "0", "Child" : ["29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "57", "61", "62", "63"],
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
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_z"}]},
			{"Name" : "work_y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "work_x_prev", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "61", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "work_x_prev"}]},
			{"Name" : "work_z_prev", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "61", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "qdata"},
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "qdata"}]},
			{"Name" : "work_xz_tilde", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "work_xz_tilde"}]},
			{"Name" : "work_rho_inv_vec", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "linsys_solver_P", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_P"}]},
			{"Name" : "linsys_solver_bp", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_bp"}]},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_p"}]},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_x"}]},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_L_i"}]},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_Dinv"}]},
			{"Name" : "linsys_solver_sol", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_sol"}]},
			{"Name" : "linsys_solver_rho_inv_vec", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "57", "SubInstance" : "grp_solve_linsys_qdldl_fu_976", "Port" : "linsys_solver_rho_inv_vec"}]},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_delta_x"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "ldata"}]},
			{"Name" : "udata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "udata"}]},
			{"Name" : "work_rho_vec", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_delta_y"}]},
			{"Name" : "work_Ax", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Ax"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_p"}]},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_x"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "Adata_i"}]},
			{"Name" : "info_pri_res", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_pri_res"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "61", "SubInstance" : "grp_compute_dua_res_fu_998", "Port" : "work_Px"},
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Px"}]},
			{"Name" : "info_dua_res", "Type" : "None", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_dua_res"}]},
			{"Name" : "info_status_val", "Type" : "OVld", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "info_status_val"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "40", "SubInstance" : "grp_check_termination_fu_939", "Port" : "work_Atdelta_y"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_z_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_y_U", "Parent" : "28"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_x_prev_U", "Parent" : "28"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_z_prev_U", "Parent" : "28"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_xz_tilde_U", "Parent" : "28"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_delta_x_U", "Parent" : "28"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_delta_y_U", "Parent" : "28"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_Ax_U", "Parent" : "28"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.Adata_p_U", "Parent" : "28"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.Adata_i_U", "Parent" : "28"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.work_Px_U", "Parent" : "28"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939", "Parent" : "28", "Child" : ["41", "50", "52", "53", "54", "55", "56"],
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
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "udata"}]},
			{"Name" : "ldata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "ldata"}]},
			{"Name" : "work_delta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_delta_y"}]},
			{"Name" : "work_Atdelta_y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "work_Atdelta_y"}]},
			{"Name" : "Adata_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_p"}]},
			{"Name" : "Adata_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_i"}]},
			{"Name" : "Adata_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "41", "SubInstance" : "grp_is_primal_infeasible_fu_317", "Port" : "Adata_x"}]},
			{"Name" : "qdata", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "qdata"}]},
			{"Name" : "work_Px", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "work_delta_x", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_is_dual_infeasible_fu_337", "Port" : "work_delta_x"}]}]},
	{"ID" : "41", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317", "Parent" : "40", "Child" : ["42", "43", "44", "45", "46", "47", "48", "49"],
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
	{"ID" : "42", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.work_Atdelta_y_U", "Parent" : "41"},
	{"ID" : "43", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.Adata_p_U", "Parent" : "41"},
	{"ID" : "44", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.Adata_i_U", "Parent" : "41"},
	{"ID" : "45", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U78", "Parent" : "41"},
	{"ID" : "46", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.fpext_32ns_64_2_no_dsp_1_U79", "Parent" : "41"},
	{"ID" : "47", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U82", "Parent" : "41"},
	{"ID" : "48", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.dcmp_64ns_64ns_1_2_no_dsp_1_U83", "Parent" : "41"},
	{"ID" : "49", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_primal_infeasible_fu_317.facc_32ns_32ns_1ns_32_6_no_dsp_1_U84", "Parent" : "41"},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_dual_infeasible_fu_337", "Parent" : "40", "Child" : ["51"],
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
	{"ID" : "51", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.grp_is_dual_infeasible_fu_337.facc_32ns_32ns_1ns_32_6_no_dsp_1_U96", "Parent" : "50"},
	{"ID" : "52", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.fadd_32ns_32ns_32_5_full_dsp_1_U100", "Parent" : "40"},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.fmul_32ns_32ns_32_4_max_dsp_1_U101", "Parent" : "40"},
	{"ID" : "54", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.fcmp_32ns_32ns_1_2_no_dsp_1_U102", "Parent" : "40"},
	{"ID" : "55", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.fcmp_32ns_32ns_1_2_no_dsp_1_U103", "Parent" : "40"},
	{"ID" : "56", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_check_termination_fu_939.fmul_32ns_32ns_32_4_max_dsp_1_U104", "Parent" : "40"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_solve_linsys_qdldl_fu_976", "Parent" : "28", "Child" : ["58", "59", "60"],
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
	{"ID" : "58", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_solve_linsys_qdldl_fu_976.linsys_solver_P_U", "Parent" : "57"},
	{"ID" : "59", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_solve_linsys_qdldl_fu_976.linsys_solver_bp_U", "Parent" : "57"},
	{"ID" : "60", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_solve_linsys_qdldl_fu_976.linsys_solver_sol_U", "Parent" : "57"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.grp_compute_dua_res_fu_998", "Parent" : "28",
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
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.faddfsub_32ns_32ns_32_5_full_dsp_1_U117", "Parent" : "28"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_osqp_solve_fu_1055.fmul_32ns_32ns_32_4_max_dsp_1_U120", "Parent" : "28"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119", "Parent" : "0", "Child" : ["65", "66", "67", "74", "75", "76"],
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
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_x"}]},
			{"Name" : "linsys_solver_L_p", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_p"}]},
			{"Name" : "linsys_solver_bwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_bwork"}]},
			{"Name" : "linsys_solver_fwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_fwork"}]},
			{"Name" : "linsys_solver_D", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_D"}]},
			{"Name" : "linsys_solver_iwork", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_iwork"}]},
			{"Name" : "linsys_solver_KKT_p", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_p"}]},
			{"Name" : "linsys_solver_KKT_i", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_KKT_i"}]},
			{"Name" : "linsys_solver_L_x", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_x"}]},
			{"Name" : "linsys_solver_L_i", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_L_i"}]},
			{"Name" : "linsys_solver_Dinv", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "67", "SubInstance" : "grp_QDLDL_factor_fu_647", "Port" : "linsys_solver_Dinv"}]}]},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.work_constr_type_U", "Parent" : "64"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.linsys_solver_rhotoKKT_U", "Parent" : "64"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647", "Parent" : "64", "Child" : ["68", "69", "70", "71", "72", "73"],
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
	{"ID" : "68", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.linsys_solver_KKT_p_U", "Parent" : "67"},
	{"ID" : "69", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.linsys_solver_KKT_i_U", "Parent" : "67"},
	{"ID" : "70", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.fsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "67"},
	{"ID" : "71", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "67"},
	{"ID" : "72", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.fdiv_32ns_32ns_32_16_no_dsp_1_U3", "Parent" : "67"},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.grp_QDLDL_factor_fu_647.fcmp_32ns_32ns_1_2_no_dsp_1_U4", "Parent" : "67"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.fptrunc_64ns_32_2_no_dsp_1_U26", "Parent" : "64"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.fpext_32ns_64_2_no_dsp_1_U27", "Parent" : "64"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_atualizar_restricao_v_fu_1119.dcmp_64ns_64ns_1_2_no_dsp_1_U29", "Parent" : "64"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172", "Parent" : "0", "Child" : ["78", "79", "80", "81", "82", "83"],
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
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.linsys_solver_KKT_p_U", "Parent" : "77"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.linsys_solver_KKT_i_U", "Parent" : "77"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.fsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "77"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "77"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.fdiv_32ns_32ns_32_16_no_dsp_1_U3", "Parent" : "77"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_QDLDL_factor_fu_1172.fcmp_32ns_32ns_1_2_no_dsp_1_U4", "Parent" : "77"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.faddfsub_32ns_32ns_32_5_full_dsp_1_U141", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.faddfsub_32ns_32ns_32_5_full_dsp_1_U142", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fsub_32ns_32ns_32_5_full_dsp_1_U143", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fsub_32ns_32ns_32_5_full_dsp_1_U144", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U145", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U146", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U147", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U148", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U149", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U150", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U151", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fcmp_32ns_32ns_1_2_no_dsp_1_U152", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_32_32_1_1_U153", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fcmp_32ns_32ns_1_2_no_dsp_1_U154", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	myFunction {
		x_ini_0 {Type I LastRead 6 FirstWrite -1}
		x_ini_1 {Type I LastRead 6 FirstWrite -1}
		x_ini_2 {Type I LastRead 0 FirstWrite -1}
		Vsd {Type I LastRead 0 FirstWrite -1}
		Vsq {Type I LastRead 0 FirstWrite -1}
		iL {Type I LastRead 0 FirstWrite -1}
		u00_0 {Type I LastRead 80 FirstWrite -1}
		u00_1 {Type I LastRead 80 FirstWrite -1}
		outputVector_0 {Type O LastRead -1 FirstWrite 109}
		outputVector_1 {Type O LastRead -1 FirstWrite 109}
		is_initialized {Type IO LastRead -1 FirstWrite -1}
		qdata {Type IO LastRead -1 FirstWrite -1}
		ldata {Type IO LastRead -1 FirstWrite -1}
		udata {Type IO LastRead -1 FirstWrite -1}
		info_status_val {Type IO LastRead -1 FirstWrite -1}
		settings_rho {Type IO LastRead -1 FirstWrite -1}
		work_constr_type {Type I LastRead -1 FirstWrite -1}
		work_rho_vec {Type IO LastRead -1 FirstWrite -1}
		work_rho_inv_vec {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_rho_inv_vec {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_rhotoKKT {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_x {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_L_p {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_bwork {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_fwork {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_D {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_iwork {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_KKT_p {Type I LastRead -1 FirstWrite -1}
		linsys_solver_KKT_i {Type I LastRead -1 FirstWrite -1}
		linsys_solver_L_x {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_L_i {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_Dinv {Type IO LastRead -1 FirstWrite -1}
		atualizar_A_A_idx {Type I LastRead -1 FirstWrite -1}
		Adata_x {Type IO LastRead -1 FirstWrite -1}
		Pdata_x {Type I LastRead -1 FirstWrite -1}
		linsys_solver_PtoKKT {Type I LastRead -1 FirstWrite -1}
		linsys_solver_Pdiag_idx {Type I LastRead -1 FirstWrite -1}
		linsys_solver_AtoKKT {Type I LastRead -1 FirstWrite -1}
		work_x {Type IO LastRead -1 FirstWrite -1}
		work_z {Type IO LastRead -1 FirstWrite -1}
		work_y {Type IO LastRead -1 FirstWrite -1}
		work_x_prev {Type IO LastRead -1 FirstWrite -1}
		work_z_prev {Type IO LastRead -1 FirstWrite -1}
		work_xz_tilde {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_P {Type I LastRead -1 FirstWrite -1}
		linsys_solver_bp {Type IO LastRead -1 FirstWrite -1}
		linsys_solver_sol {Type IO LastRead -1 FirstWrite -1}
		work_delta_x {Type IO LastRead -1 FirstWrite -1}
		work_delta_y {Type IO LastRead -1 FirstWrite -1}
		work_Ax {Type IO LastRead -1 FirstWrite -1}
		Adata_p {Type I LastRead -1 FirstWrite -1}
		Adata_i {Type I LastRead -1 FirstWrite -1}
		info_pri_res {Type IO LastRead -1 FirstWrite -1}
		work_Px {Type IO LastRead -1 FirstWrite -1}
		info_dua_res {Type I LastRead -1 FirstWrite -1}
		work_Atdelta_y {Type IO LastRead -1 FirstWrite -1}}
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
		work_Px {Type O LastRead -1 FirstWrite 2}}
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
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
	{"Pipeline" : "1", "EnableSignal" : "ap_enable_pp1"}
	{"Pipeline" : "2", "EnableSignal" : "ap_enable_pp2"}
	{"Pipeline" : "3", "EnableSignal" : "ap_enable_pp3"}
	{"Pipeline" : "4", "EnableSignal" : "ap_enable_pp4"}
	{"Pipeline" : "5", "EnableSignal" : "ap_enable_pp5"}
	{"Pipeline" : "8", "EnableSignal" : "ap_enable_pp8"}
	{"Pipeline" : "9", "EnableSignal" : "ap_enable_pp9"}
	{"Pipeline" : "10", "EnableSignal" : "ap_enable_pp10"}
	{"Pipeline" : "11", "EnableSignal" : "ap_enable_pp11"}
]}

set Spec2ImplPortList { 
}

set busDeadlockParameterList { 
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
