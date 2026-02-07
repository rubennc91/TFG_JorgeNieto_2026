// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - enable ap_done interrupt (Read/Write)
//        bit 1  - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - ap_done (COR/TOW)
//        bit 1  - ap_ready (COR/TOW)
//        others - reserved
// 0x10 : Data signal of x_ini_0
//        bit 31~0 - x_ini_0[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of x_ini_1
//        bit 31~0 - x_ini_1[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of x_ini_2
//        bit 31~0 - x_ini_2[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of Vsd
//        bit 31~0 - Vsd[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of Vsq
//        bit 31~0 - Vsq[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of iL
//        bit 31~0 - iL[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of u00_0
//        bit 31~0 - u00_0[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of u00_1
//        bit 31~0 - u00_1[31:0] (Read/Write)
// 0x4c : reserved
// 0x50 : Data signal of outputVector_0
//        bit 31~0 - outputVector_0[31:0] (Read)
// 0x54 : Control signal of outputVector_0
//        bit 0  - outputVector_0_ap_vld (Read/COR)
//        others - reserved
// 0x60 : Data signal of outputVector_1
//        bit 31~0 - outputVector_1[31:0] (Read)
// 0x64 : Control signal of outputVector_1
//        bit 0  - outputVector_1_ap_vld (Read/COR)
//        others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XMYFUNCTION_CONTROL_ADDR_AP_CTRL             0x00
#define XMYFUNCTION_CONTROL_ADDR_GIE                 0x04
#define XMYFUNCTION_CONTROL_ADDR_IER                 0x08
#define XMYFUNCTION_CONTROL_ADDR_ISR                 0x0c
#define XMYFUNCTION_CONTROL_ADDR_X_INI_0_DATA        0x10
#define XMYFUNCTION_CONTROL_BITS_X_INI_0_DATA        32
#define XMYFUNCTION_CONTROL_ADDR_X_INI_1_DATA        0x18
#define XMYFUNCTION_CONTROL_BITS_X_INI_1_DATA        32
#define XMYFUNCTION_CONTROL_ADDR_X_INI_2_DATA        0x20
#define XMYFUNCTION_CONTROL_BITS_X_INI_2_DATA        32
#define XMYFUNCTION_CONTROL_ADDR_VSD_DATA            0x28
#define XMYFUNCTION_CONTROL_BITS_VSD_DATA            32
#define XMYFUNCTION_CONTROL_ADDR_VSQ_DATA            0x30
#define XMYFUNCTION_CONTROL_BITS_VSQ_DATA            32
#define XMYFUNCTION_CONTROL_ADDR_IL_DATA             0x38
#define XMYFUNCTION_CONTROL_BITS_IL_DATA             32
#define XMYFUNCTION_CONTROL_ADDR_U00_0_DATA          0x40
#define XMYFUNCTION_CONTROL_BITS_U00_0_DATA          32
#define XMYFUNCTION_CONTROL_ADDR_U00_1_DATA          0x48
#define XMYFUNCTION_CONTROL_BITS_U00_1_DATA          32
#define XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_0_DATA 0x50
#define XMYFUNCTION_CONTROL_BITS_OUTPUTVECTOR_0_DATA 32
#define XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_0_CTRL 0x54
#define XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_1_DATA 0x60
#define XMYFUNCTION_CONTROL_BITS_OUTPUTVECTOR_1_DATA 32
#define XMYFUNCTION_CONTROL_ADDR_OUTPUTVECTOR_1_CTRL 0x64

