#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;


namespace mlir {

  void registerToPythonTranslation();
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  registerToPythonTranslation();

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
