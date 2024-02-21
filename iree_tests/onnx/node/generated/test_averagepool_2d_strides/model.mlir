module {
  func.func @test_averagepool_2d_strides(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,10,10],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [5 : si64, 5 : si64], torch.onnx.strides = [3 : si64, 3 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,10,10],f32>
    return %0 : !torch.vtensor<[1,3,10,10],f32>
  }
}

