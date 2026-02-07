// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xmyfunction.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XMyfunction_CfgInitialize(XMyfunction *InstancePtr, XMyfunction_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XMyfunction_Start(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL) & 0x80;
    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XMyfunction_IsDone(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XMyfunction_IsIdle(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XMyfunction_IsReady(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XMyfunction_EnableAutoRestart(XMyfunction *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XMyfunction_DisableAutoRestart(XMyfunction *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_AP_CTRL, 0);
}

void XMyfunction_Set_x_ini_0(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_0_DATA, Data);
}

u32 XMyfunction_Get_x_ini_0(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_0_DATA);
    return Data;
}

void XMyfunction_Set_x_ini_1(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_1_DATA, Data);
}

u32 XMyfunction_Get_x_ini_1(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_1_DATA);
    return Data;
}

void XMyfunction_Set_x_ini_2(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_2_DATA, Data);
}

u32 XMyfunction_Get_x_ini_2(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_X_INI_2_DATA);
    return Data;
}

void XMyfunction_Set_Vsd(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_VSD_DATA, Data);
}

u32 XMyfunction_Get_Vsd(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_VSD_DATA);
    return Data;
}

void XMyfunction_Set_Vsq(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_VSQ_DATA, Data);
}

u32 XMyfunction_Get_Vsq(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_VSQ_DATA);
    return Data;
}

void XMyfunction_Set_iL(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IL_DATA, Data);
}

u32 XMyfunction_Get_iL(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IL_DATA);
    return Data;
}

void XMyfunction_Set_u00_0(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_U00_0_DATA, Data);
}

u32 XMyfunction_Get_u00_0(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_U00_0_DATA);
    return Data;
}

void XMyfunction_Set_u00_1(XMyfunction *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_U00_1_DATA, Data);
}

u32 XMyfunction_Get_u00_1(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_U00_1_DATA);
    return Data;
}

u32 XMyfunction_Get_outputVector_0(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_0_DATA);
    return Data;
}

u32 XMyfunction_Get_outputVector_0_vld(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_0_CTRL);
    return Data & 0x1;
}

u32 XMyfunction_Get_outputVector_1(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_1_DATA);
    return Data;
}

u32 XMyfunction_Get_outputVector_1_vld(XMyfunction *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_1_CTRL);
    return Data & 0x1;
}

void XMyfunction_InterruptGlobalEnable(XMyfunction *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_GIE, 1);
}

void XMyfunction_InterruptGlobalDisable(XMyfunction *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_GIE, 0);
}

void XMyfunction_InterruptEnable(XMyfunction *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IER);
    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IER, Register | Mask);
}

void XMyfunction_InterruptDisable(XMyfunction *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IER);
    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IER, Register & (~Mask));
}

void XMyfunction_InterruptClear(XMyfunction *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMyfunction_WriteReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_ISR, Mask);
}

u32 XMyfunction_InterruptGetEnabled(XMyfunction *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_IER);
}

u32 XMyfunction_InterruptGetStatus(XMyfunction *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMyfunction_ReadReg(InstancePtr->Control_BaseAddress, XMYFUNCTION_CONTROL_ADDR_ISR);
}

