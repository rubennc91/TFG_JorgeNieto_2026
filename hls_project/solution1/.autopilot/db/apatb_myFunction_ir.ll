; ModuleID = 'C:/Users/ruben.nieto.PERSONAL/Documents/GitHub/TFG_JorgeNieto_2026/hls_project/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

@Pdata_i = global [12 x i64] [i64 0, i64 1, i64 3, i64 4, i64 6, i64 7, i64 8, i64 9, i64 9, i64 11, i64 11, i64 13], align 8, !dbg !0
@Pdata_p = global [16 x i64] [i64 0, i64 1, i64 2, i64 2, i64 3, i64 4, i64 4, i64 5, i64 6, i64 7, i64 8, i64 8, i64 10, i64 10, i64 12, i64 12], align 8, !dbg !11
@Pdata_x = global [12 x float] [float 1.000000e+06, float 1.000000e+04, float 1.000000e+06, float 1.000000e+04, float 1.000000e+05, float 1.000000e+05, float 1.000000e+02, float 0x3F1A36E2E0000000, float 0xBF1A36E2E0000000, float 0x3F2A36E2E0000000, float 0xBF1A36E2E0000000, float 0x3F1A36E2E0000000], align 4, !dbg !20
@Adata_i = global [43 x i64] [i64 0, i64 3, i64 1, i64 4, i64 2, i64 4, i64 5, i64 3, i64 6, i64 4, i64 7, i64 5, i64 7, i64 8, i64 6, i64 7, i64 8, i64 9, i64 15, i64 16, i64 10, i64 15, i64 3, i64 11, i64 12, i64 15, i64 16, i64 17, i64 18, i64 4, i64 5, i64 11, i64 15, i64 17, i64 6, i64 13, i64 14, i64 17, i64 18, i64 7, i64 8, i64 13, i64 17], align 8, !dbg !42
@Adata_p = global [16 x i64] [i64 0, i64 2, i64 4, i64 7, i64 9, i64 11, i64 14, i64 15, i64 16, i64 17, i64 20, i64 22, i64 29, i64 34, i64 39, i64 43], align 8, !dbg !47
@Adata_x = global [43 x float] [float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0x3F1A36E2E0000000, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0x3F1A36E2E0000000, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0xBEE05E9BC0000000, float 0xBF747AE140000000, float 1.000000e+00, float 0xBEDF4DDA80000000, float 0x3F1A36E2E0000000, float 0xBEE05E9BC0000000, float 0xBF747AE140000000, float 0x3EE05E9BC0000000, float 0x3F747AE140000000, float 0xBEE05E9BC0000000, float 0xBF747AE140000000, float 0x3E35798EE0000000, float 0x3F1A36E2E0000000, float 0xBEDF4DDA80000000, float 0x3EDF4DDA80000000, float 0xBEDF4DDA80000000, float 0x3F1A36E2E0000000, float 0xBEE05E9BC0000000, float 0xBF747AE140000000, float 0x3EE05E9BC0000000, float 0x3F747AE140000000, float 0x3E35798EE0000000, float 0x3F1A36E2E0000000, float 0xBEDF4DDA80000000, float 0x3EDF4DDA80000000], align 4, !dbg !49
@linsys_solver_L_i = global [57 x i64] [i64 31, i64 32, i64 30, i64 3, i64 8, i64 6, i64 31, i64 7, i64 31, i64 8, i64 31, i64 30, i64 31, i64 30, i64 31, i64 11, i64 15, i64 14, i64 14, i64 30, i64 15, i64 30, i64 29, i64 30, i64 29, i64 30, i64 18, i64 27, i64 20, i64 27, i64 28, i64 25, i64 25, i64 27, i64 24, i64 26, i64 32, i64 26, i64 27, i64 32, i64 27, i64 28, i64 32, i64 28, i64 29, i64 32, i64 29, i64 32, i64 30, i64 32, i64 33, i64 31, i64 32, i64 33, i64 32, i64 33, i64 33], align 8, !dbg !116
@linsys_solver_L_p = global [35 x i64] [i64 0, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 9, i64 11, i64 13, i64 15, i64 16, i64 17, i64 18, i64 20, i64 22, i64 24, i64 26, i64 27, i64 28, i64 29, i64 31, i64 32, i64 34, i64 35, i64 37, i64 40, i64 43, i64 46, i64 48, i64 51, i64 54, i64 56, i64 57, i64 57], align 8, !dbg !121
@linsys_solver_L_x = global [57 x float] [float 0x3EB2F55020000000, float 0x3EB21E9080000000, float 0x3F47BC74C0000000, float 0x40621C02A0000000, float 0x3F19D6EBC0000000, float 0xBEE4F35720000000, float 0x3F47BC74C0000000, float 0xC062154DC0000000, float 0xBF8DA0A620000000, float 0xBF19D6EBC0000000, float 0x3EB7D2DA80000000, float 0xBF8CD8B7A0000000, float 0xBF2A536860000000, float 0x3F47BC74C0000000, float 0xBF47BC74C0000000, float 0xC0621C02A0000000, float 0xBE6AD7F2A0000000, float 0xC0621C02A0000000, float 0x3F47BC74C0000000, float 0xBF47BC74C0000000, float 0xBE6AD7F2A0000000, float 0xBEA7D2DA80000000, float 0xBEB21E9080000000, float 0xBEB2F55020000000, float 0x3EB21E9080000000, float 0x3EB2F55020000000, float 0x40621C02A0000000, float 0x3F19D6EBC0000000, float 0x40621C02A0000000, float 0x3EA7270E00000000, float 0x3F7C45C660000000, float 0xBECBEF1EE0000000, float 0x3F1A363720000000, float 0xBF1A363720000000, float 0xBF847AE000000000, float 0xC04D94D9A0000000, float 0xBF783BAB60000000, float 0xBF8D3B9A60000000, float 0xBF8D3B9A60000000, float 0xBEA7D2DA80000000, float 0x3E55798EE0000000, float 0xBF914EE240000000, float 0x3F1A363720000000, float 0x3F18982760000000, float 0xBEA77CF440000000, float 0x3E45798EE0000000, float 0xBF6AAD7040000000, float 0xBF6AAD7040000000, float 0x3EEAB25DE0000000, float 0x3FCF718820000000, float 0xC016851700000000, float 0xBFDF16D3C0000000, float 0xBE45798EE0000000, float 0xBFA2BD0B40000000, float 0x3E7AD7F2A0000000, float 0x3FB150C9A0000000, float 0x401DDA8820000000], align 4, !dbg !126
@linsys_solver_KKT_i = global [79 x i64] [i64 0, i64 1, i64 2, i64 3, i64 2, i64 4, i64 5, i64 4, i64 6, i64 7, i64 6, i64 3, i64 7, i64 8, i64 9, i64 10, i64 11, i64 10, i64 12, i64 13, i64 14, i64 12, i64 13, i64 14, i64 11, i64 15, i64 16, i64 17, i64 18, i64 17, i64 19, i64 20, i64 19, i64 21, i64 22, i64 23, i64 23, i64 24, i64 22, i64 21, i64 25, i64 26, i64 25, i64 24, i64 18, i64 20, i64 22, i64 27, i64 20, i64 26, i64 28, i64 29, i64 27, i64 28, i64 16, i64 15, i64 14, i64 30, i64 8, i64 16, i64 1, i64 15, i64 13, i64 9, i64 30, i64 31, i64 6, i64 0, i64 5, i64 9, i64 32, i64 25, i64 24, i64 0, i64 30, i64 29, i64 31, i64 32, i64 33], align 8, !dbg !146
@linsys_solver_KKT_p = global [35 x i64] [i64 0, i64 1, i64 2, i64 3, i64 5, i64 6, i64 7, i64 9, i64 11, i64 14, i64 15, i64 16, i64 18, i64 19, i64 20, i64 23, i64 26, i64 27, i64 28, i64 30, i64 31, i64 33, i64 34, i64 35, i64 36, i64 38, i64 41, i64 44, i64 48, i64 51, i64 56, i64 64, i64 70, i64 74, i64 79], align 8, !dbg !151
@linsys_solver_KKT_x = global [79 x float] [float 0xC01B9C2580000000, float 0xC01B9C2580000000, float 0xBF7C454580000000, float 1.000000e+04, float -1.000000e+00, float 1.000000e+05, float 0xC01B9C2580000000, float -1.000000e+00, float 0xBF7C454580000000, float 1.000000e+04, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0xBF7C454580000000, float 0xC01B9C2580000000, float 0xBF7C454580000000, float 0x3EB0C6F7A0000000, float 1.000000e+00, float 0xBF7C454580000000, float 0xC01B9C2580000000, float 0x3F1A79FEC0000000, float 1.000000e+00, float 0xBF747AE140000000, float 0xBEE0C6F7A0000000, float 0xBEDD5C3160000000, float 0xC01B9C2580000000, float 0xC01B9C2580000000, float 0xBF7C454580000000, float 1.000000e+04, float -1.000000e+00, float 0xBF7C454580000000, float 0x3EB0C6F7A0000000, float -1.000000e+00, float 3.000000e+05, float 1.000000e+04, float 1.000000e+02, float -1.000000e+00, float 0xBF7C454580000000, float 1.000000e+00, float -1.000000e+00, float 0xBF7C454580000000, float 0x3EB0C6F7A0000000, float 0x3F1A36E2E0000000, float 1.000000e+00, float 1.000000e+00, float 0x3F1A36E2E0000000, float -1.000000e+00, float 0xBF7C454580000000, float 1.000000e+00, float -1.000000e+00, float 0xBF7C454580000000, float 0x3EB0C6F7A0000000, float 0.000000e+00, float 0x3F1A36E2E0000000, float 0xBEDD5C3160000000, float 0x3EDD5C3160000000, float 0xBF1A36E2E0000000, float 0x3F2A5870E0000000, float 0x3F1A36E2E0000000, float 0xBEE0C6F7A0000000, float 0xBF747AE140000000, float 0x3EE0C6F7A0000000, float 0x3F747AE140000000, float 0xBF747AE140000000, float 0xBF1A36E2E0000000, float 0x3F1A79FEC0000000, float 0x3F1A36E2E0000000, float 0xBEE0C6F7A0000000, float 0xBF747AE140000000, float 0x3F747AE140000000, float 0x3EB0C6F7A0000000, float 0.000000e+00, float 0x3F1A36E2E0000000, float 0xBEDD5C3160000000, float 0xBEE0C6F7A0000000, float 0xBEDD5C3160000000, float 0x3EE0C6F7A0000000, float 0x3EDD5C3160000000, float 0xC01B9C2580000000], align 4, !dbg !153

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: noinline
define void @apatb_myFunction_ir(float* %x_ini, float %Vsd, float %Vsq, float %iL, float* %u00, float* %outputVector) local_unnamed_addr #1 {
entry:
  %x_ini_copy_0 = alloca float, align 512
  %x_ini_copy_1 = alloca float, align 512
  %x_ini_copy_2 = alloca float, align 512
  %u00_copy_0 = alloca float, align 512
  %u00_copy_1 = alloca float, align 512
  %outputVector_copy_0 = alloca float, align 512
  %outputVector_copy_1 = alloca float, align 512
  %0 = bitcast float* %x_ini to [3 x float]*
  %1 = bitcast float* %u00 to [2 x float]*
  %2 = bitcast float* %outputVector to [2 x float]*
  call void @copy_in([3 x float]* %0, float* nonnull align 512 %x_ini_copy_0, float* nonnull align 512 %x_ini_copy_1, float* nonnull align 512 %x_ini_copy_2, [2 x float]* %1, float* nonnull align 512 %u00_copy_0, float* nonnull align 512 %u00_copy_1, [2 x float]* %2, float* nonnull align 512 %outputVector_copy_0, float* nonnull align 512 %outputVector_copy_1)
  call void @apatb_myFunction_hw(float* %x_ini_copy_0, float* %x_ini_copy_1, float* %x_ini_copy_2, float %Vsd, float %Vsq, float %iL, float* %u00_copy_0, float* %u00_copy_1, float* %outputVector_copy_0, float* %outputVector_copy_1)
  call void @copy_out([3 x float]* %0, float* nonnull align 512 %x_ini_copy_0, float* nonnull align 512 %x_ini_copy_1, float* nonnull align 512 %x_ini_copy_2, [2 x float]* %1, float* nonnull align 512 %u00_copy_0, float* nonnull align 512 %u00_copy_1, [2 x float]* %2, float* nonnull align 512 %outputVector_copy_0, float* nonnull align 512 %outputVector_copy_1)
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @onebyonecpy_hls.p0a3f32.84.85(float* noalias align 512 "orig.arg.no"="0" %_0, float* noalias align 512 "orig.arg.no"="0" %_1, float* noalias align 512 "orig.arg.no"="0" %_2, [3 x float]* noalias readonly "orig.arg.no"="1") #2 {
entry:
  %1 = icmp eq float* %_0, null
  %2 = icmp eq [3 x float]* %0, null
  %3 = or i1 %1, %2
  br i1 %3, label %ret, label %copy

copy:                                             ; preds = %entry
  %_01 = bitcast float* %_0 to i8*
  %_12 = bitcast float* %_1 to i8*
  %_23 = bitcast float* %_2 to i8*
  br label %for.loop

for.loop:                                         ; preds = %.exit, %copy
  %for.loop.idx3 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %.exit ]
  %src.addr.gep2 = getelementptr [3 x float], [3 x float]* %0, i64 0, i64 %for.loop.idx3
  %4 = bitcast float* %src.addr.gep2 to i8*
  switch i64 %for.loop.idx3, label %.default [
    i64 0, label %.case.0
    i64 1, label %.case.1
    i64 2, label %.case.2
  ]

.default:                                         ; preds = %for.loop
  unreachable

.case.0:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %_01, i8* align 1 %4, i64 4, i1 false)
  br label %.exit

.case.1:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %_12, i8* align 1 %4, i64 4, i1 false)
  br label %.exit

.case.2:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %_23, i8* align 1 %4, i64 4, i1 false)
  br label %.exit

.exit:                                            ; preds = %.case.2, %.case.1, %.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx3, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 3
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @onebyonecpy_hls.p0a2f32.86.87(float* noalias align 512 "orig.arg.no"="0" %_0, float* noalias align 512 "orig.arg.no"="0" %_1, [2 x float]* noalias readonly "orig.arg.no"="1") #2 {
entry:
  %1 = icmp eq float* %_0, null
  %2 = icmp eq [2 x float]* %0, null
  %3 = or i1 %1, %2
  br i1 %3, label %ret, label %copy

copy:                                             ; preds = %entry
  %_01 = bitcast float* %_0 to i8*
  %_12 = bitcast float* %_1 to i8*
  br label %for.loop

for.loop:                                         ; preds = %.exit, %copy
  %for.loop.idx3 = phi i64 [ 0, %copy ], [ 1, %.exit ]
  %src.addr.gep2 = getelementptr [2 x float], [2 x float]* %0, i64 0, i64 %for.loop.idx3
  %4 = bitcast float* %src.addr.gep2 to i8*
  %switch = icmp ult i64 %for.loop.idx3, 1
  br i1 %switch, label %.case.0, label %.case.1

.case.0:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %_01, i8* align 1 %4, i64 4, i1 false)
  br label %.exit

.case.1:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %_12, i8* align 1 %4, i64 4, i1 false)
  br label %.exit

.exit:                                            ; preds = %.case.1, %.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx3, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 2
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @copy_in([3 x float]* readonly "orig.arg.no"="0", float* noalias align 512 "orig.arg.no"="1" %_0, float* noalias align 512 "orig.arg.no"="1" %_1, float* noalias align 512 "orig.arg.no"="1" %_2, [2 x float]* readonly "orig.arg.no"="2", float* noalias align 512 "orig.arg.no"="3" %_01, float* noalias align 512 "orig.arg.no"="3" %_12, [2 x float]* readonly "orig.arg.no"="4", float* noalias align 512 "orig.arg.no"="5" %_03, float* noalias align 512 "orig.arg.no"="5" %_14) #3 {
entry:
  call void @onebyonecpy_hls.p0a3f32.84.85(float* align 512 %_0, float* align 512 %_1, float* align 512 %_2, [3 x float]* %0)
  call void @onebyonecpy_hls.p0a2f32.86.87(float* align 512 %_01, float* align 512 %_12, [2 x float]* %1)
  call void @onebyonecpy_hls.p0a2f32.86.87(float* align 512 %_03, float* align 512 %_14, [2 x float]* %2)
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @onebyonecpy_hls.p0a3f32.92.93([3 x float]* noalias "orig.arg.no"="0", float* noalias readonly align 512 "orig.arg.no"="1" %_0, float* noalias readonly align 512 "orig.arg.no"="1" %_1, float* noalias readonly align 512 "orig.arg.no"="1" %_2) #2 {
entry:
  %1 = icmp eq [3 x float]* %0, null
  %2 = icmp eq float* %_0, null
  %3 = or i1 %1, %2
  br i1 %3, label %ret, label %copy

copy:                                             ; preds = %entry
  %_01 = bitcast float* %_0 to i8*
  %_12 = bitcast float* %_1 to i8*
  %_23 = bitcast float* %_2 to i8*
  br label %for.loop

for.loop:                                         ; preds = %.exit, %copy
  %for.loop.idx3 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %.exit ]
  %dst.addr.gep1 = getelementptr [3 x float], [3 x float]* %0, i64 0, i64 %for.loop.idx3
  %4 = bitcast float* %dst.addr.gep1 to i8*
  switch i64 %for.loop.idx3, label %.default [
    i64 0, label %.case.0
    i64 1, label %.case.1
    i64 2, label %.case.2
  ]

.default:                                         ; preds = %for.loop
  unreachable

.case.0:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %4, i8* align 1 %_01, i64 4, i1 false)
  br label %.exit

.case.1:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %4, i8* align 1 %_12, i64 4, i1 false)
  br label %.exit

.case.2:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %4, i8* align 1 %_23, i64 4, i1 false)
  br label %.exit

.exit:                                            ; preds = %.case.2, %.case.1, %.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx3, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 3
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @onebyonecpy_hls.p0a2f32.94.95([2 x float]* noalias "orig.arg.no"="0", float* noalias readonly align 512 "orig.arg.no"="1" %_0, float* noalias readonly align 512 "orig.arg.no"="1" %_1) #2 {
entry:
  %1 = icmp eq [2 x float]* %0, null
  %2 = icmp eq float* %_0, null
  %3 = or i1 %1, %2
  br i1 %3, label %ret, label %copy

copy:                                             ; preds = %entry
  %_01 = bitcast float* %_0 to i8*
  %_12 = bitcast float* %_1 to i8*
  br label %for.loop

for.loop:                                         ; preds = %.exit, %copy
  %for.loop.idx3 = phi i64 [ 0, %copy ], [ 1, %.exit ]
  %dst.addr.gep1 = getelementptr [2 x float], [2 x float]* %0, i64 0, i64 %for.loop.idx3
  %4 = bitcast float* %dst.addr.gep1 to i8*
  %switch = icmp ult i64 %for.loop.idx3, 1
  br i1 %switch, label %.case.0, label %.case.1

.case.0:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %4, i8* align 1 %_01, i64 4, i1 false)
  br label %.exit

.case.1:                                          ; preds = %for.loop
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %4, i8* align 1 %_12, i64 4, i1 false)
  br label %.exit

.exit:                                            ; preds = %.case.1, %.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx3, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 2
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline
define internal void @copy_out([3 x float]* "orig.arg.no"="0", float* noalias readonly align 512 "orig.arg.no"="1" %_0, float* noalias readonly align 512 "orig.arg.no"="1" %_1, float* noalias readonly align 512 "orig.arg.no"="1" %_2, [2 x float]* "orig.arg.no"="2", float* noalias readonly align 512 "orig.arg.no"="3" %_01, float* noalias readonly align 512 "orig.arg.no"="3" %_12, [2 x float]* "orig.arg.no"="4", float* noalias readonly align 512 "orig.arg.no"="5" %_03, float* noalias readonly align 512 "orig.arg.no"="5" %_14) #4 {
entry:
  call void @onebyonecpy_hls.p0a3f32.92.93([3 x float]* %0, float* align 512 %_0, float* align 512 %_1, float* align 512 %_2)
  call void @onebyonecpy_hls.p0a2f32.94.95([2 x float]* %1, float* align 512 %_01, float* align 512 %_12)
  call void @onebyonecpy_hls.p0a2f32.94.95([2 x float]* %2, float* align 512 %_03, float* align 512 %_14)
  ret void
}

declare void @apatb_myFunction_hw(float*, float*, float*, float, float, float, float*, float*, float*, float*)

define void @myFunction_hw_stub_wrapper(float*, float*, float*, float, float, float, float*, float*, float*, float*) #5 {
entry:
  %10 = alloca [3 x float]
  %11 = alloca [2 x float]
  %12 = alloca [2 x float]
  call void @copy_out([3 x float]* %10, float* %0, float* %1, float* %2, [2 x float]* %11, float* %6, float* %7, [2 x float]* %12, float* %8, float* %9)
  %13 = bitcast [3 x float]* %10 to float*
  %14 = bitcast [2 x float]* %11 to float*
  %15 = bitcast [2 x float]* %12 to float*
  call void @myFunction_hw_stub(float* %13, float %3, float %4, float %5, float* %14, float* %15)
  call void @copy_in([3 x float]* %10, float* %0, float* %1, float* %2, [2 x float]* %11, float* %6, float* %7, [2 x float]* %12, float* %8, float* %9)
  ret void
}

declare void @myFunction_hw_stub(float*, float, float, float, float*, float*)

attributes #0 = { argmemonly nounwind }
attributes #1 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #2 = { argmemonly noinline "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline "fpga.wrapper.func"="copyin" }
attributes #4 = { argmemonly noinline "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441, !441}
!llvm.module.flags = !{!442, !443, !444}
!blackbox_cfg = !{!445}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Pdata_i", scope: !2, file: !13, line: 17, type: !160, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10)
!3 = !DIFile(filename: "C:/Users/ruben.nieto.PERSONAL/Documents/GitHub/TFG_JorgeNieto_2026/hls_project/solution1/.autopilot/db\5Cworkspace.pp.0.c", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "linsys_solver_type", file: !6, line: 36, size: 32, elements: !7)
!6 = !DIFile(filename: "srcs/lib/constants.h", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "QDLDL_SOLVER", value: 0)
!9 = !DIEnumerator(name: "MKL_PARDISO_SOLVER", value: 1)
!10 = !{!0, !11, !20, !27, !42, !47, !49, !52, !54, !59, !64, !66, !68, !70, !74, !76, !79, !81, !83, !85, !87, !89, !91, !93, !95, !97, !99, !101, !103, !105, !107, !110, !112, !114, !116, !121, !126, !129, !131, !136, !139, !141, !146, !151, !153, !156, !158, !161, !163, !166, !168, !170, !172, !175, !177, !179, !181, !183, !185, !187, !189, !191, !193, !195, !197, !199, !205, !210, !212, !214, !219, !221, !228, !230, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !361, !363, !365, !367, !369, !371, !373, !375, !377, !379, !381, !383, !385, !388, !391, !393, !398, !401, !403, !405, !407, !409, !411, !413, !415, !417, !419, !421, !423, !425, !427, !429, !431, !433, !435, !437, !439}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "Pdata_p", scope: !2, file: !13, line: 20, type: !14, isLocal: false, isDefinition: true)
!13 = !DIFile(filename: "srcs/src/workspace.c", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 1024, elements: !18)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "c_int", file: !16, line: 108, baseType: !17)
!16 = !DIFile(filename: "srcs/lib/glob_opts.h", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!17 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!18 = !{!19}
!19 = !DISubrange(count: 16)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "Pdata_x", scope: !2, file: !13, line: 23, type: !22, isLocal: false, isDefinition: true)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 384, elements: !25)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "c_float", file: !16, line: 117, baseType: !24)
!24 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!25 = !{!26}
!26 = !DISubrange(count: 12)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "Pdata", scope: !2, file: !13, line: 28, type: !29, isLocal: false, isDefinition: true)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "csc", file: !30, line: 29, baseType: !31)
!30 = !DIFile(filename: "srcs/lib\5Ctypes.h", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 21, size: 448, elements: !32)
!32 = !{!33, !34, !35, !36, !38, !39, !41}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "nzmax", scope: !31, file: !30, line: 22, baseType: !15, size: 64)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !31, file: !30, line: 23, baseType: !15, size: 64, offset: 64)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !31, file: !30, line: 24, baseType: !15, size: 64, offset: 128)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !31, file: !30, line: 25, baseType: !37, size: 64, offset: 192)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !31, file: !30, line: 26, baseType: !37, size: 64, offset: 256)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !31, file: !30, line: 27, baseType: !40, size: 64, offset: 320)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "nz", scope: !31, file: !30, line: 28, baseType: !15, size: 64, offset: 384)
!42 = !DIGlobalVariableExpression(var: !43, expr: !DIExpression())
!43 = distinct !DIGlobalVariable(name: "Adata_i", scope: !2, file: !13, line: 30, type: !44, isLocal: false, isDefinition: true)
!44 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 2752, elements: !45)
!45 = !{!46}
!46 = !DISubrange(count: 43)
!47 = !DIGlobalVariableExpression(var: !48, expr: !DIExpression())
!48 = distinct !DIGlobalVariable(name: "Adata_p", scope: !2, file: !13, line: 34, type: !14, isLocal: false, isDefinition: true)
!49 = !DIGlobalVariableExpression(var: !50, expr: !DIExpression())
!50 = distinct !DIGlobalVariable(name: "Adata_x", scope: !2, file: !13, line: 37, type: !51, isLocal: false, isDefinition: true)
!51 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1376, elements: !45)
!52 = !DIGlobalVariableExpression(var: !53, expr: !DIExpression())
!53 = distinct !DIGlobalVariable(name: "Adata", scope: !2, file: !13, line: 45, type: !29, isLocal: false, isDefinition: true)
!54 = !DIGlobalVariableExpression(var: !55, expr: !DIExpression())
!55 = distinct !DIGlobalVariable(name: "qdata", scope: !2, file: !13, line: 47, type: !56, isLocal: false, isDefinition: true)
!56 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 480, elements: !57)
!57 = !{!58}
!58 = !DISubrange(count: 15)
!59 = !DIGlobalVariableExpression(var: !60, expr: !DIExpression())
!60 = distinct !DIGlobalVariable(name: "ldata", scope: !2, file: !13, line: 51, type: !61, isLocal: false, isDefinition: true)
!61 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 608, elements: !62)
!62 = !{!63}
!63 = !DISubrange(count: 19)
!64 = !DIGlobalVariableExpression(var: !65, expr: !DIExpression())
!65 = distinct !DIGlobalVariable(name: "udata", scope: !2, file: !13, line: 56, type: !61, isLocal: false, isDefinition: true)
!66 = !DIGlobalVariableExpression(var: !67, expr: !DIExpression())
!67 = distinct !DIGlobalVariable(name: "data.n", linkageName: "data.n", scope: !2, file: !13, line: 61, type: !37, isLocal: false, isDefinition: true)
!68 = !DIGlobalVariableExpression(var: !69, expr: !DIExpression())
!69 = distinct !DIGlobalVariable(name: "data.m", linkageName: "data.m", scope: !2, file: !13, line: 61, type: !37, isLocal: false, isDefinition: true)
!70 = !DIGlobalVariableExpression(var: !71, expr: !DIExpression())
!71 = distinct !DIGlobalVariable(name: "data.P", linkageName: "data.P", scope: !2, file: !13, line: 61, type: !72, isLocal: false, isDefinition: true)
!72 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !73, size: 64)
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!74 = !DIGlobalVariableExpression(var: !75, expr: !DIExpression())
!75 = distinct !DIGlobalVariable(name: "data.A", linkageName: "data.A", scope: !2, file: !13, line: 61, type: !72, isLocal: false, isDefinition: true)
!76 = !DIGlobalVariableExpression(var: !77, expr: !DIExpression())
!77 = distinct !DIGlobalVariable(name: "data.q", linkageName: "data.q", scope: !2, file: !13, line: 61, type: !78, isLocal: false, isDefinition: true)
!78 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !40, size: 64)
!79 = !DIGlobalVariableExpression(var: !80, expr: !DIExpression())
!80 = distinct !DIGlobalVariable(name: "data.l", linkageName: "data.l", scope: !2, file: !13, line: 61, type: !78, isLocal: false, isDefinition: true)
!81 = !DIGlobalVariableExpression(var: !82, expr: !DIExpression())
!82 = distinct !DIGlobalVariable(name: "data.u", linkageName: "data.u", scope: !2, file: !13, line: 61, type: !78, isLocal: false, isDefinition: true)
!83 = !DIGlobalVariableExpression(var: !84, expr: !DIExpression())
!84 = distinct !DIGlobalVariable(name: "settings.rho", linkageName: "settings.rho", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!85 = !DIGlobalVariableExpression(var: !86, expr: !DIExpression())
!86 = distinct !DIGlobalVariable(name: "settings.sigma", linkageName: "settings.sigma", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!87 = !DIGlobalVariableExpression(var: !88, expr: !DIExpression())
!88 = distinct !DIGlobalVariable(name: "settings.scaling", linkageName: "settings.scaling", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!89 = !DIGlobalVariableExpression(var: !90, expr: !DIExpression())
!90 = distinct !DIGlobalVariable(name: "settings.adaptive_rho", linkageName: "settings.adaptive_rho", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!91 = !DIGlobalVariableExpression(var: !92, expr: !DIExpression())
!92 = distinct !DIGlobalVariable(name: "settings.adaptive_rho_interval", linkageName: "settings.adaptive_rho_interval", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!93 = !DIGlobalVariableExpression(var: !94, expr: !DIExpression())
!94 = distinct !DIGlobalVariable(name: "settings.adaptive_rho_tolerance", linkageName: "settings.adaptive_rho_tolerance", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!95 = !DIGlobalVariableExpression(var: !96, expr: !DIExpression())
!96 = distinct !DIGlobalVariable(name: "settings.max_iter", linkageName: "settings.max_iter", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!97 = !DIGlobalVariableExpression(var: !98, expr: !DIExpression())
!98 = distinct !DIGlobalVariable(name: "settings.eps_abs", linkageName: "settings.eps_abs", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!99 = !DIGlobalVariableExpression(var: !100, expr: !DIExpression())
!100 = distinct !DIGlobalVariable(name: "settings.eps_rel", linkageName: "settings.eps_rel", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!101 = !DIGlobalVariableExpression(var: !102, expr: !DIExpression())
!102 = distinct !DIGlobalVariable(name: "settings.eps_prim_inf", linkageName: "settings.eps_prim_inf", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!103 = !DIGlobalVariableExpression(var: !104, expr: !DIExpression())
!104 = distinct !DIGlobalVariable(name: "settings.eps_dual_inf", linkageName: "settings.eps_dual_inf", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!105 = !DIGlobalVariableExpression(var: !106, expr: !DIExpression())
!106 = distinct !DIGlobalVariable(name: "settings.alpha", linkageName: "settings.alpha", scope: !2, file: !13, line: 64, type: !40, isLocal: false, isDefinition: true)
!107 = !DIGlobalVariableExpression(var: !108, expr: !DIExpression())
!108 = distinct !DIGlobalVariable(name: "settings.linsys_solver", linkageName: "settings.linsys_solver", scope: !2, file: !13, line: 64, type: !109, isLocal: false, isDefinition: true)
!109 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!110 = !DIGlobalVariableExpression(var: !111, expr: !DIExpression())
!111 = distinct !DIGlobalVariable(name: "settings.scaled_termination", linkageName: "settings.scaled_termination", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!112 = !DIGlobalVariableExpression(var: !113, expr: !DIExpression())
!113 = distinct !DIGlobalVariable(name: "settings.check_termination", linkageName: "settings.check_termination", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!114 = !DIGlobalVariableExpression(var: !115, expr: !DIExpression())
!115 = distinct !DIGlobalVariable(name: "settings.warm_start", linkageName: "settings.warm_start", scope: !2, file: !13, line: 64, type: !37, isLocal: false, isDefinition: true)
!116 = !DIGlobalVariableExpression(var: !117, expr: !DIExpression())
!117 = distinct !DIGlobalVariable(name: "linsys_solver_L_i", scope: !2, file: !13, line: 70, type: !118, isLocal: false, isDefinition: true)
!118 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 3648, elements: !119)
!119 = !{!120}
!120 = !DISubrange(count: 57)
!121 = !DIGlobalVariableExpression(var: !122, expr: !DIExpression())
!122 = distinct !DIGlobalVariable(name: "linsys_solver_L_p", scope: !2, file: !13, line: 75, type: !123, isLocal: false, isDefinition: true)
!123 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 2240, elements: !124)
!124 = !{!125}
!125 = !DISubrange(count: 35)
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = distinct !DIGlobalVariable(name: "linsys_solver_L_x", scope: !2, file: !13, line: 79, type: !128, isLocal: false, isDefinition: true)
!128 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1824, elements: !119)
!129 = !DIGlobalVariableExpression(var: !130, expr: !DIExpression())
!130 = distinct !DIGlobalVariable(name: "linsys_solver_L", scope: !2, file: !13, line: 91, type: !29, isLocal: false, isDefinition: true)
!131 = !DIGlobalVariableExpression(var: !132, expr: !DIExpression())
!132 = distinct !DIGlobalVariable(name: "linsys_solver_Dinv", scope: !2, file: !13, line: 93, type: !133, isLocal: false, isDefinition: true)
!133 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 1088, elements: !134)
!134 = !{!135}
!135 = !DISubrange(count: 34)
!136 = !DIGlobalVariableExpression(var: !137, expr: !DIExpression())
!137 = distinct !DIGlobalVariable(name: "linsys_solver_P", scope: !2, file: !13, line: 101, type: !138, isLocal: false, isDefinition: true)
!138 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 2176, elements: !134)
!139 = !DIGlobalVariableExpression(var: !140, expr: !DIExpression())
!140 = distinct !DIGlobalVariable(name: "linsys_solver_rho_inv_vec", scope: !2, file: !13, line: 107, type: !61, isLocal: false, isDefinition: true)
!141 = !DIGlobalVariableExpression(var: !142, expr: !DIExpression())
!142 = distinct !DIGlobalVariable(name: "linsys_solver_Pdiag_idx", scope: !2, file: !13, line: 112, type: !143, isLocal: false, isDefinition: true)
!143 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 640, elements: !144)
!144 = !{!145}
!145 = !DISubrange(count: 10)
!146 = !DIGlobalVariableExpression(var: !147, expr: !DIExpression())
!147 = distinct !DIGlobalVariable(name: "linsys_solver_KKT_i", scope: !2, file: !13, line: 115, type: !148, isLocal: false, isDefinition: true)
!148 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 5056, elements: !149)
!149 = !{!150}
!150 = !DISubrange(count: 79)
!151 = !DIGlobalVariableExpression(var: !152, expr: !DIExpression())
!152 = distinct !DIGlobalVariable(name: "linsys_solver_KKT_p", scope: !2, file: !13, line: 121, type: !123, isLocal: false, isDefinition: true)
!153 = !DIGlobalVariableExpression(var: !154, expr: !DIExpression())
!154 = distinct !DIGlobalVariable(name: "linsys_solver_KKT_x", scope: !2, file: !13, line: 125, type: !155, isLocal: false, isDefinition: true)
!155 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 2528, elements: !149)
!156 = !DIGlobalVariableExpression(var: !157, expr: !DIExpression())
!157 = distinct !DIGlobalVariable(name: "linsys_solver_KKT", scope: !2, file: !13, line: 141, type: !29, isLocal: false, isDefinition: true)
!158 = !DIGlobalVariableExpression(var: !159, expr: !DIExpression())
!159 = distinct !DIGlobalVariable(name: "linsys_solver_PtoKKT", scope: !2, file: !13, line: 144, type: !160, isLocal: false, isDefinition: true)
!160 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 768, elements: !25)
!161 = !DIGlobalVariableExpression(var: !162, expr: !DIExpression())
!162 = distinct !DIGlobalVariable(name: "linsys_solver_AtoKKT", scope: !2, file: !13, line: 145, type: !44, isLocal: false, isDefinition: true)
!163 = !DIGlobalVariableExpression(var: !164, expr: !DIExpression())
!164 = distinct !DIGlobalVariable(name: "linsys_solver_rhotoKKT", scope: !2, file: !13, line: 146, type: !165, isLocal: false, isDefinition: true)
!165 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 1216, elements: !62)
!166 = !DIGlobalVariableExpression(var: !167, expr: !DIExpression())
!167 = distinct !DIGlobalVariable(name: "linsys_solver.type", linkageName: "linsys_solver.type", scope: !2, file: !13, line: 156, type: !109, isLocal: false, isDefinition: true)
!168 = !DIGlobalVariableExpression(var: !169, expr: !DIExpression())
!169 = distinct !DIGlobalVariable(name: "linsys_solver.L", linkageName: "linsys_solver.L", scope: !2, file: !13, line: 156, type: !72, isLocal: false, isDefinition: true)
!170 = !DIGlobalVariableExpression(var: !171, expr: !DIExpression())
!171 = distinct !DIGlobalVariable(name: "linsys_solver.Dinv", linkageName: "linsys_solver.Dinv", scope: !2, file: !13, line: 156, type: !78, isLocal: false, isDefinition: true)
!172 = !DIGlobalVariableExpression(var: !173, expr: !DIExpression())
!173 = distinct !DIGlobalVariable(name: "linsys_solver.P", linkageName: "linsys_solver.P", scope: !2, file: !13, line: 156, type: !174, isLocal: false, isDefinition: true)
!174 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!175 = !DIGlobalVariableExpression(var: !176, expr: !DIExpression())
!176 = distinct !DIGlobalVariable(name: "linsys_solver.bp", linkageName: "linsys_solver.bp", scope: !2, file: !13, line: 156, type: !78, isLocal: false, isDefinition: true)
!177 = !DIGlobalVariableExpression(var: !178, expr: !DIExpression())
!178 = distinct !DIGlobalVariable(name: "linsys_solver.sol", linkageName: "linsys_solver.sol", scope: !2, file: !13, line: 156, type: !78, isLocal: false, isDefinition: true)
!179 = !DIGlobalVariableExpression(var: !180, expr: !DIExpression())
!180 = distinct !DIGlobalVariable(name: "linsys_solver.rho_inv_vec", linkageName: "linsys_solver.rho_inv_vec", scope: !2, file: !13, line: 156, type: !78, isLocal: false, isDefinition: true)
!181 = !DIGlobalVariableExpression(var: !182, expr: !DIExpression())
!182 = distinct !DIGlobalVariable(name: "linsys_solver.sigma", linkageName: "linsys_solver.sigma", scope: !2, file: !13, line: 156, type: !40, isLocal: false, isDefinition: true)
!183 = !DIGlobalVariableExpression(var: !184, expr: !DIExpression())
!184 = distinct !DIGlobalVariable(name: "linsys_solver.n", linkageName: "linsys_solver.n", scope: !2, file: !13, line: 156, type: !37, isLocal: false, isDefinition: true)
!185 = !DIGlobalVariableExpression(var: !186, expr: !DIExpression())
!186 = distinct !DIGlobalVariable(name: "linsys_solver.m", linkageName: "linsys_solver.m", scope: !2, file: !13, line: 156, type: !37, isLocal: false, isDefinition: true)
!187 = !DIGlobalVariableExpression(var: !188, expr: !DIExpression())
!188 = distinct !DIGlobalVariable(name: "linsys_solver.Pdiag_idx", linkageName: "linsys_solver.Pdiag_idx", scope: !2, file: !13, line: 156, type: !174, isLocal: false, isDefinition: true)
!189 = !DIGlobalVariableExpression(var: !190, expr: !DIExpression())
!190 = distinct !DIGlobalVariable(name: "linsys_solver.Pdiag_n", linkageName: "linsys_solver.Pdiag_n", scope: !2, file: !13, line: 156, type: !37, isLocal: false, isDefinition: true)
!191 = !DIGlobalVariableExpression(var: !192, expr: !DIExpression())
!192 = distinct !DIGlobalVariable(name: "linsys_solver.KKT", linkageName: "linsys_solver.KKT", scope: !2, file: !13, line: 156, type: !72, isLocal: false, isDefinition: true)
!193 = !DIGlobalVariableExpression(var: !194, expr: !DIExpression())
!194 = distinct !DIGlobalVariable(name: "linsys_solver.PtoKKT", linkageName: "linsys_solver.PtoKKT", scope: !2, file: !13, line: 156, type: !174, isLocal: false, isDefinition: true)
!195 = !DIGlobalVariableExpression(var: !196, expr: !DIExpression())
!196 = distinct !DIGlobalVariable(name: "linsys_solver.AtoKKT", linkageName: "linsys_solver.AtoKKT", scope: !2, file: !13, line: 156, type: !174, isLocal: false, isDefinition: true)
!197 = !DIGlobalVariableExpression(var: !198, expr: !DIExpression())
!198 = distinct !DIGlobalVariable(name: "linsys_solver.rhotoKKT", linkageName: "linsys_solver.rhotoKKT", scope: !2, file: !13, line: 156, type: !174, isLocal: false, isDefinition: true)
!199 = !DIGlobalVariableExpression(var: !200, expr: !DIExpression())
!200 = distinct !DIGlobalVariable(name: "linsys_solver.D", linkageName: "linsys_solver.D", scope: !2, file: !13, line: 156, type: !201, isLocal: false, isDefinition: true)
!201 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !202, size: 64)
!202 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !203, size: 64)
!203 = !DIDerivedType(tag: DW_TAG_typedef, name: "QDLDL_float", file: !204, line: 13, baseType: !23)
!204 = !DIFile(filename: "srcs/lib/qdldl_types.h", directory: "C:\5CUsers\5Cruben.nieto.PERSONAL\5CDocuments\5CGitHub\5CTFG_JorgeNieto_2026")
!205 = !DIGlobalVariableExpression(var: !206, expr: !DIExpression())
!206 = distinct !DIGlobalVariable(name: "linsys_solver.etree", linkageName: "linsys_solver.etree", scope: !2, file: !13, line: 156, type: !207, isLocal: false, isDefinition: true)
!207 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !208, size: 64)
!208 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !209, size: 64)
!209 = !DIDerivedType(tag: DW_TAG_typedef, name: "QDLDL_int", file: !204, line: 14, baseType: !15)
!210 = !DIGlobalVariableExpression(var: !211, expr: !DIExpression())
!211 = distinct !DIGlobalVariable(name: "linsys_solver.Lnz", linkageName: "linsys_solver.Lnz", scope: !2, file: !13, line: 156, type: !207, isLocal: false, isDefinition: true)
!212 = !DIGlobalVariableExpression(var: !213, expr: !DIExpression())
!213 = distinct !DIGlobalVariable(name: "linsys_solver.iwork", linkageName: "linsys_solver.iwork", scope: !2, file: !13, line: 156, type: !207, isLocal: false, isDefinition: true)
!214 = !DIGlobalVariableExpression(var: !215, expr: !DIExpression())
!215 = distinct !DIGlobalVariable(name: "linsys_solver.bwork", linkageName: "linsys_solver.bwork", scope: !2, file: !13, line: 156, type: !216, isLocal: false, isDefinition: true)
!216 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !217, size: 64)
!217 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !218, size: 64)
!218 = !DIDerivedType(tag: DW_TAG_typedef, name: "QDLDL_bool", file: !204, line: 15, baseType: !15)
!219 = !DIGlobalVariableExpression(var: !220, expr: !DIExpression())
!220 = distinct !DIGlobalVariable(name: "linsys_solver.fwork", linkageName: "linsys_solver.fwork", scope: !2, file: !13, line: 156, type: !201, isLocal: false, isDefinition: true)
!221 = !DIGlobalVariableExpression(var: !222, expr: !DIExpression())
!222 = distinct !DIGlobalVariable(name: "solution", scope: !2, file: !13, line: 184, type: !223, isLocal: false, isDefinition: true)
!223 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPSolution", file: !30, line: 60, baseType: !224)
!224 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 57, size: 128, elements: !225)
!225 = !{!226, !227}
!226 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !224, file: !30, line: 58, baseType: !40, size: 64)
!227 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !224, file: !30, line: 59, baseType: !40, size: 64, offset: 64)
!228 = !DIGlobalVariableExpression(var: !229, expr: !DIExpression())
!229 = distinct !DIGlobalVariable(name: "info.iter", linkageName: "info.iter", scope: !2, file: !13, line: 186, type: !37, isLocal: false, isDefinition: true)
!230 = !DIGlobalVariableExpression(var: !231, expr: !DIExpression())
!231 = distinct !DIGlobalVariable(name: "info.status", linkageName: "info.status", scope: !2, file: !13, line: 186, type: !232, isLocal: false, isDefinition: true)
!232 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !233, size: 64)
!233 = !DICompositeType(tag: DW_TAG_array_type, baseType: !234, size: 256, elements: !235)
!234 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!235 = !{!236}
!236 = !DISubrange(count: 32)
!237 = !DIGlobalVariableExpression(var: !238, expr: !DIExpression())
!238 = distinct !DIGlobalVariable(name: "info.status_val", linkageName: "info.status_val", scope: !2, file: !13, line: 186, type: !37, isLocal: false, isDefinition: true)
!239 = !DIGlobalVariableExpression(var: !240, expr: !DIExpression())
!240 = distinct !DIGlobalVariable(name: "info.obj_val", linkageName: "info.obj_val", scope: !2, file: !13, line: 186, type: !40, isLocal: false, isDefinition: true)
!241 = !DIGlobalVariableExpression(var: !242, expr: !DIExpression())
!242 = distinct !DIGlobalVariable(name: "info.pri_res", linkageName: "info.pri_res", scope: !2, file: !13, line: 186, type: !40, isLocal: false, isDefinition: true)
!243 = !DIGlobalVariableExpression(var: !244, expr: !DIExpression())
!244 = distinct !DIGlobalVariable(name: "info.dua_res", linkageName: "info.dua_res", scope: !2, file: !13, line: 186, type: !40, isLocal: false, isDefinition: true)
!245 = !DIGlobalVariableExpression(var: !246, expr: !DIExpression())
!246 = distinct !DIGlobalVariable(name: "info.rho_updates", linkageName: "info.rho_updates", scope: !2, file: !13, line: 186, type: !37, isLocal: false, isDefinition: true)
!247 = !DIGlobalVariableExpression(var: !248, expr: !DIExpression())
!248 = distinct !DIGlobalVariable(name: "info.rho_estimate", linkageName: "info.rho_estimate", scope: !2, file: !13, line: 186, type: !40, isLocal: false, isDefinition: true)
!249 = !DIGlobalVariableExpression(var: !250, expr: !DIExpression())
!250 = distinct !DIGlobalVariable(name: "work_rho_vec", scope: !2, file: !13, line: 188, type: !61, isLocal: false, isDefinition: true)
!251 = !DIGlobalVariableExpression(var: !252, expr: !DIExpression())
!252 = distinct !DIGlobalVariable(name: "work_rho_inv_vec", scope: !2, file: !13, line: 193, type: !61, isLocal: false, isDefinition: true)
!253 = !DIGlobalVariableExpression(var: !254, expr: !DIExpression())
!254 = distinct !DIGlobalVariable(name: "work_constr_type", scope: !2, file: !13, line: 198, type: !165, isLocal: false, isDefinition: true)
!255 = !DIGlobalVariableExpression(var: !256, expr: !DIExpression())
!256 = distinct !DIGlobalVariable(name: "workspace", scope: !2, file: !13, line: 217, type: !257, isLocal: false, isDefinition: true)
!257 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPWorkspace", file: !30, line: 289, baseType: !258)
!258 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 182, size: 1664, elements: !259)
!259 = !{!260, !272, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314, !335, !346, !348}
!260 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !258, file: !30, line: 184, baseType: !261, size: 64)
!261 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !262, size: 64)
!262 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPData", file: !30, line: 133, baseType: !263)
!263 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 125, size: 448, elements: !264)
!264 = !{!265, !266, !267, !268, !269, !270, !271}
!265 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !263, file: !30, line: 126, baseType: !15, size: 64)
!266 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !263, file: !30, line: 127, baseType: !15, size: 64, offset: 64)
!267 = !DIDerivedType(tag: DW_TAG_member, name: "P", scope: !263, file: !30, line: 128, baseType: !73, size: 64, offset: 128)
!268 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !263, file: !30, line: 129, baseType: !73, size: 64, offset: 192)
!269 = !DIDerivedType(tag: DW_TAG_member, name: "q", scope: !263, file: !30, line: 130, baseType: !40, size: 64, offset: 256)
!270 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !263, file: !30, line: 131, baseType: !40, size: 64, offset: 320)
!271 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !263, file: !30, line: 132, baseType: !40, size: 64, offset: 384)
!272 = !DIDerivedType(tag: DW_TAG_member, name: "linsys_solver", scope: !258, file: !30, line: 187, baseType: !273, size: 64, offset: 64)
!273 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !274, size: 64)
!274 = !DIDerivedType(tag: DW_TAG_typedef, name: "LinSysSolver", file: !30, line: 35, baseType: !275)
!275 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "linsys_solver", file: !30, line: 298, size: 256, elements: !276)
!276 = !{!277, !278, !282, !288}
!277 = !DIDerivedType(tag: DW_TAG_member, name: "type", scope: !275, file: !30, line: 299, baseType: !5, size: 32)
!278 = !DIDerivedType(tag: DW_TAG_member, name: "solve", scope: !275, file: !30, line: 300, baseType: !279, size: 64, offset: 64)
!279 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !280, size: 64)
!280 = !DISubroutineType(types: !281)
!281 = !{!15, !273, !40}
!282 = !DIDerivedType(tag: DW_TAG_member, name: "update_matrices", scope: !275, file: !30, line: 308, baseType: !283, size: 64, offset: 128)
!283 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !284, size: 64)
!284 = !DISubroutineType(types: !285)
!285 = !{!15, !273, !286, !286}
!286 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !287, size: 64)
!287 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !29)
!288 = !DIDerivedType(tag: DW_TAG_member, name: "update_rho_vec", scope: !275, file: !30, line: 312, baseType: !289, size: 64, offset: 192)
!289 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !290, size: 64)
!290 = !DISubroutineType(types: !291)
!291 = !{!15, !273, !292}
!292 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !293, size: 64)
!293 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!294 = !DIDerivedType(tag: DW_TAG_member, name: "rho_vec", scope: !258, file: !30, line: 198, baseType: !40, size: 64, offset: 128)
!295 = !DIDerivedType(tag: DW_TAG_member, name: "rho_inv_vec", scope: !258, file: !30, line: 199, baseType: !40, size: 64, offset: 192)
!296 = !DIDerivedType(tag: DW_TAG_member, name: "constr_type", scope: !258, file: !30, line: 204, baseType: !37, size: 64, offset: 256)
!297 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !258, file: !30, line: 211, baseType: !40, size: 64, offset: 320)
!298 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !258, file: !30, line: 212, baseType: !40, size: 64, offset: 384)
!299 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !258, file: !30, line: 213, baseType: !40, size: 64, offset: 448)
!300 = !DIDerivedType(tag: DW_TAG_member, name: "xz_tilde", scope: !258, file: !30, line: 214, baseType: !40, size: 64, offset: 512)
!301 = !DIDerivedType(tag: DW_TAG_member, name: "x_prev", scope: !258, file: !30, line: 216, baseType: !40, size: 64, offset: 576)
!302 = !DIDerivedType(tag: DW_TAG_member, name: "z_prev", scope: !258, file: !30, line: 219, baseType: !40, size: 64, offset: 640)
!303 = !DIDerivedType(tag: DW_TAG_member, name: "Ax", scope: !258, file: !30, line: 230, baseType: !40, size: 64, offset: 704)
!304 = !DIDerivedType(tag: DW_TAG_member, name: "Px", scope: !258, file: !30, line: 231, baseType: !40, size: 64, offset: 768)
!305 = !DIDerivedType(tag: DW_TAG_member, name: "Aty", scope: !258, file: !30, line: 232, baseType: !40, size: 64, offset: 832)
!306 = !DIDerivedType(tag: DW_TAG_member, name: "delta_y", scope: !258, file: !30, line: 240, baseType: !40, size: 64, offset: 896)
!307 = !DIDerivedType(tag: DW_TAG_member, name: "Atdelta_y", scope: !258, file: !30, line: 241, baseType: !40, size: 64, offset: 960)
!308 = !DIDerivedType(tag: DW_TAG_member, name: "delta_x", scope: !258, file: !30, line: 249, baseType: !40, size: 64, offset: 1024)
!309 = !DIDerivedType(tag: DW_TAG_member, name: "Pdelta_x", scope: !258, file: !30, line: 250, baseType: !40, size: 64, offset: 1088)
!310 = !DIDerivedType(tag: DW_TAG_member, name: "Adelta_x", scope: !258, file: !30, line: 251, baseType: !40, size: 64, offset: 1152)
!311 = !DIDerivedType(tag: DW_TAG_member, name: "D_temp", scope: !258, file: !30, line: 260, baseType: !40, size: 64, offset: 1216)
!312 = !DIDerivedType(tag: DW_TAG_member, name: "D_temp_A", scope: !258, file: !30, line: 261, baseType: !40, size: 64, offset: 1280)
!313 = !DIDerivedType(tag: DW_TAG_member, name: "E_temp", scope: !258, file: !30, line: 262, baseType: !40, size: 64, offset: 1344)
!314 = !DIDerivedType(tag: DW_TAG_member, name: "settings", scope: !258, file: !30, line: 267, baseType: !315, size: 64, offset: 1408)
!315 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !316, size: 64)
!316 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPSettings", file: !30, line: 176, baseType: !317)
!317 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 139, size: 768, elements: !318)
!318 = !{!319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334}
!319 = !DIDerivedType(tag: DW_TAG_member, name: "rho", scope: !317, file: !30, line: 140, baseType: !23, size: 32)
!320 = !DIDerivedType(tag: DW_TAG_member, name: "sigma", scope: !317, file: !30, line: 141, baseType: !23, size: 32, offset: 32)
!321 = !DIDerivedType(tag: DW_TAG_member, name: "scaling", scope: !317, file: !30, line: 142, baseType: !15, size: 64, offset: 64)
!322 = !DIDerivedType(tag: DW_TAG_member, name: "adaptive_rho", scope: !317, file: !30, line: 145, baseType: !15, size: 64, offset: 128)
!323 = !DIDerivedType(tag: DW_TAG_member, name: "adaptive_rho_interval", scope: !317, file: !30, line: 146, baseType: !15, size: 64, offset: 192)
!324 = !DIDerivedType(tag: DW_TAG_member, name: "adaptive_rho_tolerance", scope: !317, file: !30, line: 147, baseType: !23, size: 32, offset: 256)
!325 = !DIDerivedType(tag: DW_TAG_member, name: "max_iter", scope: !317, file: !30, line: 153, baseType: !15, size: 64, offset: 320)
!326 = !DIDerivedType(tag: DW_TAG_member, name: "eps_abs", scope: !317, file: !30, line: 154, baseType: !23, size: 32, offset: 384)
!327 = !DIDerivedType(tag: DW_TAG_member, name: "eps_rel", scope: !317, file: !30, line: 155, baseType: !23, size: 32, offset: 416)
!328 = !DIDerivedType(tag: DW_TAG_member, name: "eps_prim_inf", scope: !317, file: !30, line: 156, baseType: !23, size: 32, offset: 448)
!329 = !DIDerivedType(tag: DW_TAG_member, name: "eps_dual_inf", scope: !317, file: !30, line: 157, baseType: !23, size: 32, offset: 480)
!330 = !DIDerivedType(tag: DW_TAG_member, name: "alpha", scope: !317, file: !30, line: 158, baseType: !23, size: 32, offset: 512)
!331 = !DIDerivedType(tag: DW_TAG_member, name: "linsys_solver", scope: !317, file: !30, line: 159, baseType: !5, size: 32, offset: 544)
!332 = !DIDerivedType(tag: DW_TAG_member, name: "scaled_termination", scope: !317, file: !30, line: 169, baseType: !15, size: 64, offset: 576)
!333 = !DIDerivedType(tag: DW_TAG_member, name: "check_termination", scope: !317, file: !30, line: 170, baseType: !15, size: 64, offset: 640)
!334 = !DIDerivedType(tag: DW_TAG_member, name: "warm_start", scope: !317, file: !30, line: 171, baseType: !15, size: 64, offset: 704)
!335 = !DIDerivedType(tag: DW_TAG_member, name: "scaling", scope: !258, file: !30, line: 268, baseType: !336, size: 64, offset: 1472)
!336 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !337, size: 64)
!337 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPScaling", file: !30, line: 52, baseType: !338)
!338 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 45, size: 384, elements: !339)
!339 = !{!340, !341, !342, !343, !344, !345}
!340 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !338, file: !30, line: 46, baseType: !23, size: 32)
!341 = !DIDerivedType(tag: DW_TAG_member, name: "D", scope: !338, file: !30, line: 47, baseType: !40, size: 64, offset: 64)
!342 = !DIDerivedType(tag: DW_TAG_member, name: "E", scope: !338, file: !30, line: 48, baseType: !40, size: 64, offset: 128)
!343 = !DIDerivedType(tag: DW_TAG_member, name: "cinv", scope: !338, file: !30, line: 49, baseType: !23, size: 32, offset: 192)
!344 = !DIDerivedType(tag: DW_TAG_member, name: "Dinv", scope: !338, file: !30, line: 50, baseType: !40, size: 64, offset: 256)
!345 = !DIDerivedType(tag: DW_TAG_member, name: "Einv", scope: !338, file: !30, line: 51, baseType: !40, size: 64, offset: 320)
!346 = !DIDerivedType(tag: DW_TAG_member, name: "solution", scope: !258, file: !30, line: 269, baseType: !347, size: 64, offset: 1536)
!347 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !223, size: 64)
!348 = !DIDerivedType(tag: DW_TAG_member, name: "info", scope: !258, file: !30, line: 270, baseType: !349, size: 64, offset: 1600)
!349 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !350, size: 64)
!350 = !DIDerivedType(tag: DW_TAG_typedef, name: "OSQPInfo", file: !30, line: 91, baseType: !351)
!351 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !30, line: 66, size: 640, elements: !352)
!352 = !{!353, !354, !355, !356, !357, !358, !359, !360}
!353 = !DIDerivedType(tag: DW_TAG_member, name: "iter", scope: !351, file: !30, line: 67, baseType: !15, size: 64)
!354 = !DIDerivedType(tag: DW_TAG_member, name: "status", scope: !351, file: !30, line: 68, baseType: !233, size: 256, offset: 64)
!355 = !DIDerivedType(tag: DW_TAG_member, name: "status_val", scope: !351, file: !30, line: 69, baseType: !15, size: 64, offset: 320)
!356 = !DIDerivedType(tag: DW_TAG_member, name: "obj_val", scope: !351, file: !30, line: 75, baseType: !23, size: 32, offset: 384)
!357 = !DIDerivedType(tag: DW_TAG_member, name: "pri_res", scope: !351, file: !30, line: 76, baseType: !23, size: 32, offset: 416)
!358 = !DIDerivedType(tag: DW_TAG_member, name: "dua_res", scope: !351, file: !30, line: 77, baseType: !23, size: 32, offset: 448)
!359 = !DIDerivedType(tag: DW_TAG_member, name: "rho_updates", scope: !351, file: !30, line: 88, baseType: !15, size: 64, offset: 512)
!360 = !DIDerivedType(tag: DW_TAG_member, name: "rho_estimate", scope: !351, file: !30, line: 89, baseType: !23, size: 32, offset: 576)
!361 = !DIGlobalVariableExpression(var: !362, expr: !DIExpression())
!362 = distinct !DIGlobalVariable(name: "scaling_D", scope: !2, file: !13, line: 10, type: !56, isLocal: false, isDefinition: true)
!363 = !DIGlobalVariableExpression(var: !364, expr: !DIExpression())
!364 = distinct !DIGlobalVariable(name: "scaling_Dinv", scope: !2, file: !13, line: 11, type: !56, isLocal: false, isDefinition: true)
!365 = !DIGlobalVariableExpression(var: !366, expr: !DIExpression())
!366 = distinct !DIGlobalVariable(name: "scaling_E", scope: !2, file: !13, line: 12, type: !61, isLocal: false, isDefinition: true)
!367 = !DIGlobalVariableExpression(var: !368, expr: !DIExpression())
!368 = distinct !DIGlobalVariable(name: "scaling_Einv", scope: !2, file: !13, line: 13, type: !61, isLocal: false, isDefinition: true)
!369 = !DIGlobalVariableExpression(var: !370, expr: !DIExpression())
!370 = distinct !DIGlobalVariable(name: "scaling.c", linkageName: "scaling.c", scope: !2, file: !13, line: 67, type: !40, isLocal: false, isDefinition: true)
!371 = !DIGlobalVariableExpression(var: !372, expr: !DIExpression())
!372 = distinct !DIGlobalVariable(name: "scaling.D", linkageName: "scaling.D", scope: !2, file: !13, line: 67, type: !78, isLocal: false, isDefinition: true)
!373 = !DIGlobalVariableExpression(var: !374, expr: !DIExpression())
!374 = distinct !DIGlobalVariable(name: "scaling.E", linkageName: "scaling.E", scope: !2, file: !13, line: 67, type: !78, isLocal: false, isDefinition: true)
!375 = !DIGlobalVariableExpression(var: !376, expr: !DIExpression())
!376 = distinct !DIGlobalVariable(name: "scaling.cinv", linkageName: "scaling.cinv", scope: !2, file: !13, line: 67, type: !40, isLocal: false, isDefinition: true)
!377 = !DIGlobalVariableExpression(var: !378, expr: !DIExpression())
!378 = distinct !DIGlobalVariable(name: "scaling.Dinv", linkageName: "scaling.Dinv", scope: !2, file: !13, line: 67, type: !78, isLocal: false, isDefinition: true)
!379 = !DIGlobalVariableExpression(var: !380, expr: !DIExpression())
!380 = distinct !DIGlobalVariable(name: "scaling.Einv", linkageName: "scaling.Einv", scope: !2, file: !13, line: 67, type: !78, isLocal: false, isDefinition: true)
!381 = !DIGlobalVariableExpression(var: !382, expr: !DIExpression())
!382 = distinct !DIGlobalVariable(name: "linsys_solver_bp", scope: !2, file: !13, line: 105, type: !133, isLocal: false, isDefinition: true)
!383 = !DIGlobalVariableExpression(var: !384, expr: !DIExpression())
!384 = distinct !DIGlobalVariable(name: "linsys_solver_sol", scope: !2, file: !13, line: 106, type: !133, isLocal: false, isDefinition: true)
!385 = !DIGlobalVariableExpression(var: !386, expr: !DIExpression())
!386 = distinct !DIGlobalVariable(name: "linsys_solver_D", scope: !2, file: !13, line: 148, type: !387, isLocal: false, isDefinition: true)
!387 = !DICompositeType(tag: DW_TAG_array_type, baseType: !203, size: 1088, elements: !134)
!388 = !DIGlobalVariableExpression(var: !389, expr: !DIExpression())
!389 = distinct !DIGlobalVariable(name: "linsys_solver_etree", scope: !2, file: !13, line: 149, type: !390, isLocal: false, isDefinition: true)
!390 = !DICompositeType(tag: DW_TAG_array_type, baseType: !209, size: 2176, elements: !134)
!391 = !DIGlobalVariableExpression(var: !392, expr: !DIExpression())
!392 = distinct !DIGlobalVariable(name: "linsys_solver_Lnz", scope: !2, file: !13, line: 150, type: !390, isLocal: false, isDefinition: true)
!393 = !DIGlobalVariableExpression(var: !394, expr: !DIExpression())
!394 = distinct !DIGlobalVariable(name: "linsys_solver_iwork", scope: !2, file: !13, line: 151, type: !395, isLocal: false, isDefinition: true)
!395 = !DICompositeType(tag: DW_TAG_array_type, baseType: !209, size: 6528, elements: !396)
!396 = !{!397}
!397 = !DISubrange(count: 102)
!398 = !DIGlobalVariableExpression(var: !399, expr: !DIExpression())
!399 = distinct !DIGlobalVariable(name: "linsys_solver_bwork", scope: !2, file: !13, line: 152, type: !400, isLocal: false, isDefinition: true)
!400 = !DICompositeType(tag: DW_TAG_array_type, baseType: !218, size: 2176, elements: !134)
!401 = !DIGlobalVariableExpression(var: !402, expr: !DIExpression())
!402 = distinct !DIGlobalVariable(name: "linsys_solver_fwork", scope: !2, file: !13, line: 153, type: !387, isLocal: false, isDefinition: true)
!403 = !DIGlobalVariableExpression(var: !404, expr: !DIExpression())
!404 = distinct !DIGlobalVariable(name: "xsolution", scope: !2, file: !13, line: 182, type: !56, isLocal: false, isDefinition: true)
!405 = !DIGlobalVariableExpression(var: !406, expr: !DIExpression())
!406 = distinct !DIGlobalVariable(name: "ysolution", scope: !2, file: !13, line: 183, type: !61, isLocal: false, isDefinition: true)
!407 = !DIGlobalVariableExpression(var: !408, expr: !DIExpression())
!408 = distinct !DIGlobalVariable(name: "work_x", scope: !2, file: !13, line: 199, type: !56, isLocal: false, isDefinition: true)
!409 = !DIGlobalVariableExpression(var: !410, expr: !DIExpression())
!410 = distinct !DIGlobalVariable(name: "work_y", scope: !2, file: !13, line: 200, type: !61, isLocal: false, isDefinition: true)
!411 = !DIGlobalVariableExpression(var: !412, expr: !DIExpression())
!412 = distinct !DIGlobalVariable(name: "work_z", scope: !2, file: !13, line: 201, type: !61, isLocal: false, isDefinition: true)
!413 = !DIGlobalVariableExpression(var: !414, expr: !DIExpression())
!414 = distinct !DIGlobalVariable(name: "work_xz_tilde", scope: !2, file: !13, line: 202, type: !133, isLocal: false, isDefinition: true)
!415 = !DIGlobalVariableExpression(var: !416, expr: !DIExpression())
!416 = distinct !DIGlobalVariable(name: "work_x_prev", scope: !2, file: !13, line: 203, type: !56, isLocal: false, isDefinition: true)
!417 = !DIGlobalVariableExpression(var: !418, expr: !DIExpression())
!418 = distinct !DIGlobalVariable(name: "work_z_prev", scope: !2, file: !13, line: 204, type: !61, isLocal: false, isDefinition: true)
!419 = !DIGlobalVariableExpression(var: !420, expr: !DIExpression())
!420 = distinct !DIGlobalVariable(name: "work_Ax", scope: !2, file: !13, line: 205, type: !61, isLocal: false, isDefinition: true)
!421 = !DIGlobalVariableExpression(var: !422, expr: !DIExpression())
!422 = distinct !DIGlobalVariable(name: "work_Px", scope: !2, file: !13, line: 206, type: !56, isLocal: false, isDefinition: true)
!423 = !DIGlobalVariableExpression(var: !424, expr: !DIExpression())
!424 = distinct !DIGlobalVariable(name: "work_Aty", scope: !2, file: !13, line: 207, type: !56, isLocal: false, isDefinition: true)
!425 = !DIGlobalVariableExpression(var: !426, expr: !DIExpression())
!426 = distinct !DIGlobalVariable(name: "work_delta_y", scope: !2, file: !13, line: 208, type: !61, isLocal: false, isDefinition: true)
!427 = !DIGlobalVariableExpression(var: !428, expr: !DIExpression())
!428 = distinct !DIGlobalVariable(name: "work_Atdelta_y", scope: !2, file: !13, line: 209, type: !56, isLocal: false, isDefinition: true)
!429 = !DIGlobalVariableExpression(var: !430, expr: !DIExpression())
!430 = distinct !DIGlobalVariable(name: "work_delta_x", scope: !2, file: !13, line: 210, type: !56, isLocal: false, isDefinition: true)
!431 = !DIGlobalVariableExpression(var: !432, expr: !DIExpression())
!432 = distinct !DIGlobalVariable(name: "work_Pdelta_x", scope: !2, file: !13, line: 211, type: !56, isLocal: false, isDefinition: true)
!433 = !DIGlobalVariableExpression(var: !434, expr: !DIExpression())
!434 = distinct !DIGlobalVariable(name: "work_Adelta_x", scope: !2, file: !13, line: 212, type: !61, isLocal: false, isDefinition: true)
!435 = !DIGlobalVariableExpression(var: !436, expr: !DIExpression())
!436 = distinct !DIGlobalVariable(name: "work_D_temp", scope: !2, file: !13, line: 213, type: !56, isLocal: false, isDefinition: true)
!437 = !DIGlobalVariableExpression(var: !438, expr: !DIExpression())
!438 = distinct !DIGlobalVariable(name: "work_D_temp_A", scope: !2, file: !13, line: 214, type: !56, isLocal: false, isDefinition: true)
!439 = !DIGlobalVariableExpression(var: !440, expr: !DIExpression())
!440 = distinct !DIGlobalVariable(name: "work_E_temp", scope: !2, file: !13, line: 215, type: !61, isLocal: false, isDefinition: true)
!441 = !{!"clang version 7.0.0 "}
!442 = !{i32 2, !"Dwarf Version", i32 4}
!443 = !{i32 2, !"Debug Info Version", i32 3}
!444 = !{i32 1, !"wchar_size", i32 4}
!445 = !{}
