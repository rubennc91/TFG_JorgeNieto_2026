# HLS Implementation of OSQP Solver for FPGA (TFG Jorge Nieto 2026)

This repository contains a **High-Level Synthesis (HLS)** implementation of the [OSQP (Operator Splitting Quadratic Program)](https://osqp.org/) solver, optimized for Xilinx Vitis HLS.

This project is part of a Bachelor's Thesis (Trabajo de Fin de Grado - TFG) developed by **[Jorge Nieto]** and supervised by Rubén Nieto.

## 🎯 Project Overview

The goal of this project is to embed the OSQP solver into an FPGA (specifically targeting the **Xilinx XC7Z010**, e.g., Zybo board) to solve Model Predictive Control (MPC) problems in real-time.

Standard C code for solvers often relies on dynamic memory allocation (`malloc`) and complex pointer-to-pointer structures, which are not synthesizable by HLS tools. This project refactors the OSQP C code to be fully static and hardware-friendly.

## 🚀 Key Features & Modifications

To achieve HLS synthesis, the original OSQP source code was heavily modified using a **Flat Array Strategy**:

1.  **Static Memory Only:** All dynamic memory allocation (`malloc`, `free`) has been removed.
2.  **Flattened Data Structures:** - Compressed Sparse Column (CSC) matrices (`P`, `A`) are "exploded" into global static arrays (e.g., `Pdata_x`, `Pdata_i`, `Pdata_p`).
    - Removed pointer-to-pointer references in function arguments to satisfy Vitis HLS constraints.
3.  **QDLDL Interface Adaptation:** The linear system solver interface was rewritten to access global data structures directly, removing function pointers and polymorphism.
4.  **Floating Point:** Configured to use `float` (32-bit) instead of `double` to optimize DSP slice usage on the FPGA.

## 📂 Repository Structure

* **`srcs/src/`**: Modified source code (`osqp.c`, `auxil.c`, `scaling.c`, etc.).
* **`srcs/lib/`**: Header files defining the static structures and flattened arrays.
* **`srcs/src/simulink_block.c`**: Contains the **Top-Level Function** (`myFunction`) that wraps the solver for the control loop.
* **`srcs/src/workspace.c`**: Contains the hardcoded problem data (matrices and vectors) generated from MATLAB/Python.

## 🛠️ How to Build (Vitis HLS)

This project is designed for **Xilinx Vitis HLS 2020.2** (or compatible versions).

1.  Open Vitis HLS.
2.  Create a new project.
3.  **Add Source Files:** Add all `.c` files from `srcs/src/` (excluding `main.c` if used for C-simulation only).
4.  **Add CFLAGS:** For every source file, add the include path:
    `-I/path/to/project/srcs/lib`
5.  **Set Top Function:** `myFunction` (located in `simulink_block.c`).
6.  **Select Part:** `xc7z010-clg400-1` (or your target FPGA).
7.  Run **C Synthesis**.

## 🔌 Top-Level Interface

The hardware block `myFunction` exposes the following interface for the MPC controller:

```c
void myFunction(
    float x_ini[3],      // Initial state [ia, ib, vdc]
    float Vsd,           // Grid Voltage d-axis
    float Vsq,           // Grid Voltage q-axis
    float iL,            // Load current
    float u00[2],        // Previous control input
    float outputVector[2]// Computed control action (v_ref)
);
```

## ✅ To-Do List / Future Work

- [x] Port OSQP C code to static C compatible with Vitis HLS.
- [x] Verify C-Synthesis and remove all pointer-to-pointer errors.
- [ ] **Cosimulation:** Verify numerical accuracy against the original C code using a Testbench.
- [ ] **IP Export:** Export the RTL design as an IP Core for Vivado.
- [ ] **System Integration:** Integrate the IP Core into a Zynq Block Design (Connect via AXI4-Lite or AXI-Stream).
- [ ] **Hardware Validation:** Run the solver on the physical FPGA (Zybo Z7-10) and benchmark execution time.

## 📊 Resource Usage (Example)

* **Target:** xc7z010-clg400-1
* **Latency:** [-] cycles
* **BRAM:** [14 %]
* **DSP:** [53 %]
* **FF:** [50 %]
* **LUT:** [157 %]

## 📜 Credits & License

* **Original OSQP Solver:** Stellato et al.
* **QDLDL Linear Solver:** Paul Goulart et al.
* **HLS Porting & Adaptation:** [Jorge Nieto / Rubén Nieto]

This project follows the **Apache 2.0 License** (same as OSQP).
