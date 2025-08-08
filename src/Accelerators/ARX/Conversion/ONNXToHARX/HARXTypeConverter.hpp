#define DEBUG_TYPE "harx-to-larx"

using namespace mlir;
using namespace onnx_mlir::harx;



namespace onnx_mlir {
namespace harx {


struct MyTypeConverter : public mlir::TypeConverter {
  MyTypeConverter() {

    // (B) 다른 타입은 그대로 둠
    addConversion([](mlir::Type t) { return t; });
    // (C) tensor<…>·memref<…>처럼 Shape + Element 로 된 타입 처리
    addConversion([&](mlir::ShapedType st) -> std::optional<mlir::Type> {
      // 1) element type 먼저 변환
      mlir::Type newElem = convertType(st.getElementType());
      if (!newElem)                        // 실패 시 전체 변환 실패
        return std::nullopt;
      // 2) 모양(shape)은 그대로 두고 element 만 교체
      return st.clone(newElem);
    });

    // (선택) 필요하면 캐스트 materialization 도 추가
    addSourceMaterialization([](mlir::OpBuilder &b, mlir::Type dstTy,
                                mlir::ValueRange inputs, mlir::Location loc)
        -> std::optional<mlir::Value> {
      // 예: f32 → f16 로 변환된 값에 arith.fpext / fptrunc 생성
      return std::optional<mlir::Value>{
        b.create<mlir::arith::TruncFOp>(loc, dstTy, inputs[0]).getResult()
      };
    });
  }
};

}
}
