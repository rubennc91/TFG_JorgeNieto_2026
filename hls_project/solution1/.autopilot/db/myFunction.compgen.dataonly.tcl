# This script segment is generated automatically by AutoPilot

set axilite_register_dict [dict create]
set port_control {
x_ini_0 { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 16
	offset_end 23
}
x_ini_1 { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 24
	offset_end 31
}
x_ini_2 { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 32
	offset_end 39
}
Vsd { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 40
	offset_end 47
}
Vsq { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 48
	offset_end 55
}
iL { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 56
	offset_end 63
}
u00_0 { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 64
	offset_end 71
}
u00_1 { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 72
	offset_end 79
}
outputVector_0 { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 80
	offset_end 87
}
outputVector_1 { 
	dir O
	width 32
	depth 1
	mode ap_vld
	offset 96
	offset_end 103
}
ap_start { }
ap_done { }
ap_ready { }
ap_idle { }
}
dict set axilite_register_dict control $port_control


