// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xmyfunction.h"

extern XMyfunction_Config XMyfunction_ConfigTable[];

XMyfunction_Config *XMyfunction_LookupConfig(u16 DeviceId) {
	XMyfunction_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XMYFUNCTION_NUM_INSTANCES; Index++) {
		if (XMyfunction_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XMyfunction_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XMyfunction_Initialize(XMyfunction *InstancePtr, u16 DeviceId) {
	XMyfunction_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XMyfunction_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XMyfunction_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

