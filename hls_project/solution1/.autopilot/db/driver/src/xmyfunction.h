// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XMYFUNCTION_H
#define XMYFUNCTION_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xmyfunction_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u32 Control_BaseAddress;
} XMyfunction_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XMyfunction;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XMyfunction_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XMyfunction_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XMyfunction_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XMyfunction_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XMyfunction_Initialize(XMyfunction *InstancePtr, u16 DeviceId);
XMyfunction_Config* XMyfunction_LookupConfig(u16 DeviceId);
int XMyfunction_CfgInitialize(XMyfunction *InstancePtr, XMyfunction_Config *ConfigPtr);
#else
int XMyfunction_Initialize(XMyfunction *InstancePtr, const char* InstanceName);
int XMyfunction_Release(XMyfunction *InstancePtr);
#endif

void XMyfunction_Start(XMyfunction *InstancePtr);
u32 XMyfunction_IsDone(XMyfunction *InstancePtr);
u32 XMyfunction_IsIdle(XMyfunction *InstancePtr);
u32 XMyfunction_IsReady(XMyfunction *InstancePtr);
void XMyfunction_EnableAutoRestart(XMyfunction *InstancePtr);
void XMyfunction_DisableAutoRestart(XMyfunction *InstancePtr);

void XMyfunction_Set_x_ini_0(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_x_ini_0(XMyfunction *InstancePtr);
void XMyfunction_Set_x_ini_1(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_x_ini_1(XMyfunction *InstancePtr);
void XMyfunction_Set_x_ini_2(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_x_ini_2(XMyfunction *InstancePtr);
void XMyfunction_Set_Vsd(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_Vsd(XMyfunction *InstancePtr);
void XMyfunction_Set_Vsq(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_Vsq(XMyfunction *InstancePtr);
void XMyfunction_Set_iL(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_iL(XMyfunction *InstancePtr);
void XMyfunction_Set_u00_0(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_u00_0(XMyfunction *InstancePtr);
void XMyfunction_Set_u00_1(XMyfunction *InstancePtr, u32 Data);
u32 XMyfunction_Get_u00_1(XMyfunction *InstancePtr);
u32 XMyfunction_Get_outputVector_0(XMyfunction *InstancePtr);
u32 XMyfunction_Get_outputVector_0_vld(XMyfunction *InstancePtr);
u32 XMyfunction_Get_outputVector_1(XMyfunction *InstancePtr);
u32 XMyfunction_Get_outputVector_1_vld(XMyfunction *InstancePtr);

void XMyfunction_InterruptGlobalEnable(XMyfunction *InstancePtr);
void XMyfunction_InterruptGlobalDisable(XMyfunction *InstancePtr);
void XMyfunction_InterruptEnable(XMyfunction *InstancePtr, u32 Mask);
void XMyfunction_InterruptDisable(XMyfunction *InstancePtr, u32 Mask);
void XMyfunction_InterruptClear(XMyfunction *InstancePtr, u32 Mask);
u32 XMyfunction_InterruptGetEnabled(XMyfunction *InstancePtr);
u32 XMyfunction_InterruptGetStatus(XMyfunction *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
