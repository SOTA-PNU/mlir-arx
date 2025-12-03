#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <stack>
#include <utility>
#include <list>


#define DEBUG_TYPE "translate-to-python"

using namespace mlir;
using llvm::formatv;


namespace mlir {




    struct PythonEmitter {
        explicit PythonEmitter(raw_ostream& os, bool declareVariablesAtTop,
            StringRef fileId);

        LogicalResult printValue(Location loc, Attribute attr);
        LogicalResult emitAttribute(Location loc, Attribute attr);
        LogicalResult emitOperation(Operation& op, bool trailingSemicolon);

        LogicalResult emitType(Location loc, Type type, bool isreturn);
        LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
        LogicalResult emitTupleType(Location loc, ArrayRef<Type> types, bool isreturn);

        LogicalResult emitVariableDeclaration(OpResult result,
            bool trailingSemicolon);

        LogicalResult emitVariableDeclaration(Location loc, Type type,
            StringRef name);

        LogicalResult emitOuter();

        StringRef getOrCreateName(Value val);
        StringRef getOrCreateName(Block& block);

        StringRef getOrCreateNameArg(Value val);
		StringRef getLastVar() { return last_var; }


        LogicalResult emitAssignPrefix(Operation& op);


        void printHexString(StringRef str) {
            os << "\"0x" << llvm::toHex(str) << "\"";
        }
        void printHexString(ArrayRef<char> data) {
            printHexString(StringRef(data.data(), data.size()));
        }


        raw_indented_ostream& ostream() { return os; };
        bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };
        bool hasValueInScope(Value val);



        struct Scope {
            ~Scope() { emitter.labelInScopeCount.pop(); }

        private:
            llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
            llvm::ScopedHashTableScope<Block*, std::string> blockMapperScope;

        protected:
            Scope(PythonEmitter& emitter)
                : valueMapperScope(emitter.valueMapper),
                blockMapperScope(emitter.blockMapper), emitter(emitter){
                emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
            }
            PythonEmitter& emitter;
        };

        struct FunctionScope : Scope {
            FunctionScope(PythonEmitter& emitter) : Scope(emitter) {
                // Re-use value names.
                emitter.resetValueCounter();
            }
        };

        void resetValueCounter();
		void addVariableOpDeclaration(Value decl) {

			emitType(decl.getLoc(), decl.getType(), false);
			os << "bld.add_output('" << getOrCreateName(decl) << "', shape, dtype)" << "\n";


            //StringRef ref = getOrCreateName(decl);
			//outputMapper.insert(std::pair<Value, StringRef>(decl, ref));
            //variableOpDeclarations.push_back(ref);
        }


    private:

        using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
        using BlockMapper = llvm::ScopedHashTable<Block*, std::string>;

        using OutMapper = std::map<Value, StringRef>;



        raw_indented_ostream os;
        bool declareVariablesAtTop;
        std::string fileId;
        std::stack<int64_t> valueInScopeCount;
        std::stack<int64_t> labelInScopeCount;

		std::list<StringRef> variableOpDeclarations;


        ValueMapper valueMapper;
        ValueMapper argMapper;
        BlockMapper blockMapper;

        OutMapper outputMapper;

        StringRef last_var;


        uint64_t loopNestingLevel{ 0 };
        unsigned int valueCount{ 0 };
        unsigned int argCount{ 0 };
    };


    template <typename ForwardIterator, typename UnaryFunctor,
        typename NullaryFunctor>
    static inline LogicalResult
        interleaveWithError(ForwardIterator begin, ForwardIterator end,
            UnaryFunctor eachFn, NullaryFunctor betweenFn) {
        if (begin == end)
            return success();
        if (failed(eachFn(*begin)))
            return failure();
        ++begin;
        for (; begin != end; ++begin) {
            betweenFn();
            if (failed(eachFn(*begin)))
                return failure();
        }
        return success();
    }

    template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
    static inline LogicalResult interleaveWithError(const Container& c,
        UnaryFunctor eachFn,
        NullaryFunctor betweenFn) {
        return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
    }

    template <typename Container, typename UnaryFunctor>
    static inline LogicalResult interleaveCommaWithError(const Container& c,
        raw_ostream& os,
        UnaryFunctor eachFn) {
        return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
    }


    static LogicalResult printONNXBuilder(raw_ostream& os) {
       
		std::string s = R"(

def dtype_to_onnx(dtype: str) -> int:
    m = {
        'f32': TensorProto.FLOAT,
        'f16': TensorProto.FLOAT16,
        'f64': TensorProto.DOUBLE,
        'i64': TensorProto.INT64,
        'i32': TensorProto.INT32,
        'i16': TensorProto.INT16,
        'i8' : TensorProto.INT8,
        'u8' : TensorProto.UINT8,
        'bool': TensorProto.BOOL,
        'none': TensorProto.BOOL
    }
    return m.get(dtype, TensorProto.FLOAT)

def parse_dense_payload(content: str, shape: List[int], dtype: str) -> np.ndarray:
    content = content.strip()
    # hex blob
    if content.startswith('"0x'):
        hexstr = content.strip('"')[2:]
        hexstr = re.sub(r'\s+', '', hexstr)
        buf = bytes.fromhex(hexstr)
        npdt = np.float32 if dtype.startswith('f') else (np.int64 if dtype.endswith('64') else np.int32)
        arr = np.frombuffer(buf, dtype=npdt)
        expect = int(np.prod([d for d in shape if d>=0]) or 1)
        assert arr.size == expect, f"hex size mismatch: {arr.size} vs {shape}"
        return arr.reshape(shape if shape else [arr.size])
    #     Ʈ    ͷ 
    if content.startswith('['):
        py = ast.literal_eval(content)
        npdt = np.float32 if dtype.startswith('f') else (np.int64 if dtype.endswith('64') else np.int32)
        arr = np.array(py, dtype=npdt)
        if shape:
            arr = arr.reshape(shape)
        return arr
    #   Į  
    try:
        val = float(content) if dtype.startswith('f') else int(content)
        npdt = np.float32 if dtype.startswith('f') else (np.int64 if dtype.endswith('64') else np.int32)
        if shape:
            return np.full(shape, val, dtype=npdt)
        return np.array(val, dtype=npdt)
    except:
        raise ValueError(f"Unsupported dense content: {content[:80]}...")

class ONNXBuilder:
    def __init__(self):
        self.nodes = []
        self.inputs_info = []
        self.outputs_info = []
        self.initializers = []
        self.return_types = []
        self.none_values = []
        self.value_map: Dict[str, str] = {}  # SSA -> ONNX name
        self.value_types: Dict[str, Tuple[List[int], str]] = {}  # SSA -> (shape,dtype)

    def ensure_value_name(self, ssa: str) -> str:
        n = to_valid_name(ssa)
        self.value_map.setdefault(ssa, n)
        return self.value_map[ssa]

    def add_initializer_from_array(self, name: str, arr: np.ndarray, dtype_str: str):
        onnx_dtype = dtype_to_onnx(dtype_str)
        self.initializers.append(helper.make_tensor(
            name, onnx_dtype, list(arr.shape), arr.astype({
                TensorProto.FLOAT: np.float32,
                TensorProto.FLOAT16: np.float16,
                TensorProto.DOUBLE: np.float64,
                TensorProto.INT64: np.int64,
                TensorProto.INT32: np.int32,
                TensorProto.INT16: np.int16,
                TensorProto.INT8:  np.int8,
                TensorProto.UINT8: np.uint8,
                TensorProto.BOOL:  np.bool_,
            }[onnx_dtype]).ravel()
        ))

    def add_input(self, name: str, shape: List[int], dtype: str):
        self.inputs_info.append(
            helper.make_tensor_value_info(name, dtype_to_onnx(dtype), [d if d>=0 else None for d in shape])
        )

    def add_output(self, name: str, shape: List[int], dtype: str):
        self.outputs_info.append(
            helper.make_tensor_value_info(name, dtype_to_onnx(dtype), [d if d>=0 else None for d in shape])
        )

    def add_node(self, op_type: str, inputs: List[str], outputs: List[str], **attrs):
        #  Ӽ         ONNX ǥ   Ÿ     θ 
        for input in inputs:
            if input in self.none_values:
                inputs.remove(input)

        clean_attrs = {}
        for k,v in attrs.items():
            if v is None: continue
            clean_attrs[k] = v
        if 'onnx_node_name' in clean_attrs:
            clean_attrs['name'] = clean_attrs['onnx_node_name']
            clean_attrs.pop('onnx_node_name')

        self.nodes.append(helper.make_node(op_type, inputs, outputs, **clean_attrs))

    def make_model(self, name="MLIR2ONNX"):
        # Validate that all outputs are produced by nodes or mapped to produced nodes
        output_names = set(o.name for o in self.outputs_info)
        node_outputs = set()
        for node in self.nodes:
            # Convert node outputs to set of strings and check value_map
            for out in node.output:
                node_outputs.add(str(out))
                if str(out) in self.value_map:
                    node_outputs.add(self.value_map[str(out)])
        
        # Debug: print what we have
        print(f"DEBUG: output_names = {output_names}")
        print(f"DEBUG: node_outputs = {node_outputs}")
        print(f"DEBUG: value_map = {self.value_map}")
        
        if not output_names.issubset(node_outputs):
            missing = output_names - node_outputs
            print(f"DEBUG: missing = {missing}")
            # Check if missing outputs are mapped SSA values that exist in node outputs
            for out in list(missing):
                # Find original SSA value that maps to this output name
                ssa_vals = [k for k,v in self.value_map.items() if v == out]
                print(f"DEBUG: for output {out}, ssa_vals = {ssa_vals}")
                # Check if the SSA value (without % prefix) exists in node_outputs
                if ssa_vals:
                    ssa_val = ssa_vals[0]
                    # Remove % prefix if present for comparison
                    ssa_val_clean = ssa_val[1:] if ssa_val.startswith('%') else ssa_val
                    if ssa_val_clean in node_outputs:
                        print(f"DEBUG: removing {out} from missing")
                        missing.remove(out)
            if missing:
                raise ValueError(f"Graph outputs {missing} are not produced by any node. Available outputs: {node_outputs}. Value map: {self.value_map}")

        g = helper.make_graph(
            self.nodes, name,
            self.inputs_info, self.outputs_info, 
            self.initializers
        )
        m = helper.make_model(g, producer_name="mlir2onnx-generic")
        onnx.checker.check_model(m)
        return m

)";
		os << s << "\n";

		os << "bld = ONNXBuilder()\n";

		return success();
	}


    static LogicalResult printHeader(raw_ostream& os) {
        os << "import re, ast\n";
        os << "from typing import Dict, List, Tuple, Any\n";
		os << "import numpy as np\n";
		os << "import onnx\n";
		os << "from onnx import helper, TensorProto\n";
		os << "import onnxruntime as ort\n";

        std::string s = R"(
import sys

if len(sys.argv) < 2:
    print("Usage: python test.py <save.onnx>")
    sys.exit(1)
save_filename = sys.argv[1]        

    )";
        os << s << "\n";

        return success();
	}

    

    static LogicalResult printOperation(PythonEmitter& emitter, ModuleOp moduleOp) {
        raw_ostream& os = emitter.ostream();

        os << "#onnx to python (zento)\n";

        if (failed(printHeader(os)))
        {
            return failure();
        }

        if (failed(printONNXBuilder(os)))
        {
            return failure();
		}
    
        for (Operation& op : moduleOp) {
            if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
                return failure();
        }
        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, cf::BranchOp moduleOp) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python BranchOp\n";
        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, cf::CondBranchOp moduleOp) {
       
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python CondBranchOp\n";

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, func::CallOp callOp) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python CallOp\n";

        return success();
    }

    static LogicalResult printFunctionArgs(PythonEmitter& emitter,
        Operation* functionOp,
        ArrayRef<Type> arguments) {
        raw_indented_ostream& os = emitter.ostream();

        return (
            interleaveCommaWithError(arguments, os, [&](Type arg) -> LogicalResult {
                return emitter.emitType(functionOp->getLoc(), arg, false);
                }));
    }

    static LogicalResult printFunctionArgs(PythonEmitter& emitter,
        Operation* functionOp,
        Region::BlockArgListType arguments) {
        raw_indented_ostream& os = emitter.ostream();

        return (interleaveCommaWithError(
            arguments, os, [&](BlockArgument arg) -> LogicalResult {
                emitter.emitVariableDeclaration(
                    functionOp->getLoc(), arg.getType(), emitter.getOrCreateNameArg(arg));
                os << "bld.add_input('" << emitter.getOrCreateNameArg(arg) << "', shape, dtype)\n";
                return success();
            }));
    }


    static LogicalResult printFunctionBody(PythonEmitter& emitter,
        Operation* functionOp,
        Region::BlockListType& blocks) {

        raw_indented_ostream& os = emitter.ostream();
        //os.indent();

		os << "#Function body with " << blocks.size() << " blocks\n";

        if (emitter.shouldDeclareVariablesAtTop()) {
            WalkResult result =
                functionOp->walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
                    return WalkResult::skip();
                for (OpResult result : op->getResults()) {
                    if (failed(emitter.emitVariableDeclaration(
                        result, /*trailingSemicolon=*/true))) {
                        return WalkResult(
                            op->emitError("unable to declare result variable for op"));
                    }
                }
                return WalkResult::advance();
                    });
            if (result.wasInterrupted())
                return failure();
        }

        // Create label names for basic blocks.
        for (Block& block : blocks) {
            emitter.getOrCreateName(block);
        }

        // Declare variables for basic block arguments.
        for (Block& block : llvm::drop_begin(blocks)) {
            for (BlockArgument& arg : block.getArguments()) {
                if (emitter.hasValueInScope(arg))
                    return functionOp->emitOpError(" block argument #")
                    << arg.getArgNumber() << " is out of scope";
                if (failed(
                    emitter.emitType(block.getParentOp()->getLoc(), arg.getType(), false))) {
                    return failure();
                }
                os << " " << emitter.getOrCreateName(arg) << ";\n";
            }
        }

        for (Block& block : blocks) {
            for (Operation& op : block.getOperations()) {
                if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
                    return failure();
            }
        }

        //os.unindent();

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, func::FuncOp funcOp) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python funcOp\n";

        if (!emitter.shouldDeclareVariablesAtTop() &&
            funcOp.getBlocks().size() > 1) {
            return funcOp.emitOpError(
                "with multiple blocks needs variables declared at top");
        }



        PythonEmitter::FunctionScope scope(emitter);

       
        if (failed(emitter.emitTypes(funcOp.getLoc(), funcOp.getFunctionType().getResults())))
            return failure();

        os << "bld.function_name='" << funcOp.getName() << "'\n";

        //os << "(";
        Operation* operation = funcOp.getOperation();
        if (failed(printFunctionArgs(emitter, operation, funcOp.getArguments())))
            return failure();
        //os << ") {\n";
        if (failed(printFunctionBody(emitter, operation, funcOp.getBlocks())))
            return failure();
        //os << "}";

        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, func::ReturnOp returnOp) {
      
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python ReturnOp\n";


        if (failed(emitter.emitOuter()))
			return failure();


        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXAddOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";

            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();

            os << "\n";
        }

        os << "bld.add_node('Add', ['";

        for (unsigned int i = 0; i < op.getNumOperands(); i++)
        {
            os << emitter.getOrCreateName(op.getOperand(i));
            
            if (i < op.getNumOperands() - 1)
                os << "', '";
        }
        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";


        emitter.addVariableOpDeclaration(op.getResult());

        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, ONNXMulOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('Mul', ['";

        for(unsigned int i = 0; i < op.getNumOperands(); i++)
        {
			os << emitter.getOrCreateName(op.getOperand(i));
            
            if(i < op.getNumOperands() - 1)
				os << "', '"; 
		}
        os << "'], ['";
		os	<< emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";


        emitter.addVariableOpDeclaration(op.getResult());

        os << "\n";

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXLogOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('Log', ['";

        os << emitter.getOrCreateName(op.getOperand());

        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());
        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, ONNXTanhOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
			if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('Tanh', ['";

        os << emitter.getOrCreateName(op.getOperand());

        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";


        emitter.addVariableOpDeclaration(op.getResult());

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXReturnOp onnxReturnOp) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python ONNXReturnOp\n";


        return success();
    }


    static LogicalResult printConstantOp(PythonEmitter& emitter, Operation* operation,
        Attribute value) {
        
        OpResult result = operation->getResult(0);


        return emitter.emitAttribute(operation->getLoc(), value);
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXConstantOp onnxConstantOp) {
        raw_ostream& os = emitter.ostream();
        //os << "#Call to Python ONNXConstantOp\n";

        Type resultType = onnxConstantOp.getResult().getType();
        Type type = onnxConstantOp.getType();
        Operation* operation = onnxConstantOp.getOperation();
        Attribute value = onnxConstantOp.getValue().value();

        if (failed(emitter.emitAssignPrefix(*operation)))
            return failure();

        os << "";

        if (failed(printConstantOp(emitter, operation, value)))
			return failure();

		os << "\n";

		os << "arr = parse_dense_payload(" << emitter.getLastVar() << ", shape, dtype)\n";
        os << "bld.add_initializer_from_array('"<< emitter.getLastVar()  <<"', arr, dtype)\n";

		return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXNoneOp op) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python ONNXNoneOp\n";

        
        for (auto r : op->getResults())
        {
            os << "bld.none_values.append('" << emitter.getOrCreateName(r) << "')\n";
        }

     

        //op.getOperand()

        //bld.none_values.append(name)
        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, ONNXConvOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('Conv', ['";

        for (unsigned int i = 0; i < op.getNumOperands(); i++)
        {
            os << emitter.getOrCreateName(op.getOperand(i));

            if (i < op.getNumOperands() - 1)
                os << "', '";
        }
        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());

        return success();
    }

    


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXMaxPoolSingleOutOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('MaxPool', ['";

        os << emitter.getOrCreateName(op.getOperand());

        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());
        return success();
    }

    static LogicalResult printOperation(PythonEmitter& emitter, ONNXReduceMeanV13Op op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();

        /*
        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }
        */

        os << "bld.add_node('GlobalAveragePool', ['";

        os << emitter.getOrCreateName(op.getOperand());
        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXReshapeOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();
            os << "\n";
        }

        os << "bld.add_node('Reshape', ['";

        for (unsigned int i = 0; i < op.getNumOperands(); i++)
        {
            os << emitter.getOrCreateName(op.getOperand(i));

            if (i < op.getNumOperands() - 1)
                os << "', '";
        }

        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());
        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXGemmOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=";
            if (failed(emitter.printValue(op->getLoc(), attr.getValue())))
				return failure();

            os <<  "\n";
        }

        os << "bld.add_node('Gemm', ['";

        for (unsigned int i = 0; i < op.getNumOperands(); i++)
        {
            os << emitter.getOrCreateName(op.getOperand(i));

            if (i < op.getNumOperands() - 1)
                os << "', '";
        }

        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());

        return success();
    }


    static LogicalResult printOperation(PythonEmitter& emitter, ONNXReluOp op) {
        raw_ostream& os = emitter.ostream();

        os << "node_attrs = {}\n";

        auto namedattrs = op->getPropertiesStorage()
            ? llvm::to_vector(op->getDiscardableAttrs())
            : op->getAttrs();


        for (auto attr : namedattrs)
        {
            os << "node_attrs['" << attr.getName().strref() << "']" << "=" << attr.getValue() << "\n";
        }

        os << "bld.add_node('Relu', ['";

        os << emitter.getOrCreateName(op.getOperand());
        os << "'], ['";
        os << emitter.getOrCreateName(op.getResult()) << "'], **node_attrs)\n";

        os << "\n";

        emitter.addVariableOpDeclaration(op.getResult());
        return success();
    }



    static LogicalResult printOperation(PythonEmitter& emitter, ONNXEntryPointOp onnxEntryPointOp) {
        raw_ostream& os = emitter.ostream();
        os << "#Call to Python ONNXEntryPointOp\n";

        std::string s = R"(

def demo_infer(onnx_path: str, feed: Dict[str, np.ndarray]):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = [o.name for o in sess.get_outputs()]
    res = sess.run(outputs, feed)
    return dict(zip(outputs, res))


model = bld.make_model(name="MLIR_ONNX_Generic");
onnx.save(model, save_filename)
g = model.graph
feeds = {}
for inp in g.input:
    # initializer         (initializer는 모델에 포함된 데이터)
    if any(ini.name == inp.name for ini in g.initializer):
        continue
    shape = [d.dim_value if (d.dim_value>0) else 1 for d in inp.type.tensor_type.shape.dim]
    dt = inp.type.tensor_type.elem_type
    npdt = {TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.FLOAT16: np.float16,
            TensorProto.INT64: np.int64,
            TensorProto.INT32: np.int32,
            TensorProto.INT16: np.int16,
            TensorProto.INT8:  np.int8,
            TensorProto.UINT8: np.uint8,
            TensorProto.BOOL:  np.bool_}.get(dt, np.float32)
    feeds[inp.name] = np.random.rand(*[s if s>0 else 1 for s in shape]).astype(npdt)


out = demo_infer('test.onnx', feeds)
for k,v in out.items():
    print(f"[out] {k}: shape={v.shape}, dtype={v.dtype}")
    print("min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
                    np.min(v),
                    np.max(v),
                    np.mean(v),
                    np.std(v, dtype=np.float64)
          ))


        )";



        os << s << "\n";


        return success();
    }


    PythonEmitter::PythonEmitter(raw_ostream& os, bool declareVariablesAtTop,
        StringRef fileId)
        : os(os), declareVariablesAtTop(declareVariablesAtTop),
        fileId(fileId.str()) {
        valueInScopeCount.push(0);
        labelInScopeCount.push(0);
    }



    LogicalResult PythonEmitter::emitOperation(Operation& op, bool trailingSemicolon) {
        
        //os << "#Zento Python operation: " << op.getName().getStringRef() << "\n";


        LogicalResult status =
            llvm::TypeSwitch<Operation*, LogicalResult>(&op)
            // Builtin ops.
            .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
            // CF ops.
            .Case<cf::BranchOp, cf::CondBranchOp>(
                [&](auto op) { return printOperation(*this, op); })
            // EmitC ops.
            .Case<  ONNXAddOp, ONNXMulOp, ONNXLogOp, ONNXTanhOp, ONNXReturnOp, ONNXConstantOp, ONNXNoneOp,
                    ONNXConvOp, ONNXReluOp, ONNXMaxPoolSingleOutOp, ONNXReduceMeanV13Op, ONNXReshapeOp, ONNXGemmOp,
                    ONNXEntryPointOp
                   
                   /*ONNXAbsOp, ONNXAcosOp, ONNXAcoshOp, ONNXAddOp,

                   ONNXAndOp, ONNXArgMaxOp, ONNXArgMinOp, ONNXAsinOp, ONNXAsinhOp,
                   ONNXAtanOp, ONNXAtanhOp, ONNXAveragePoolOp, ONNXBatchNormalizationOp,
                   ONNXBernoulliOp, ONNXBitShiftOp, ONNXBitwiseAndOp, ONNXBitwiseNotOp, ONNXBitwiseOrOp,
                    ONNXBitwiseXorOp, ONNXBlackmanWindowOp, ONNXCastOp, ONNXCastLikeOp, ONNXCeilOp, ONNXCeluOp,
                    ONNXCenterCropPadOp, ONNXClipOp, ONNXClipV12Op, ONNXClipV11Op, ONNXClipV6Op, ONNXCol2ImOp,
                    ONNXCol2ImOp, ONNXCompressOp, ONNXConcatOp, ONNXConcatFromSequenceOp, ONNXConstantOp, ONNXConstantOfShapeOp,
                    ONNXConvOp, ONNXConvIntegerOp, ONNXConvTransposeOp, ONNXCosOp, ONNXCoshOp, ONNXCumSumOp, ONNXDFTOp,
                    ONNXDFTV17Op, ONNXDeformConvOp, ONNXDepthToSpaceOp, ONNXDequantizeLinearOp, ONNXDetOp, ONNXDivOp, ONNXDropoutOp,
                    ONNXDynamicQuantizeLinearOp, ONNXEinsumOp, ONNXEluOp, ONNXEqualOp, ONNXErfOp, ONNXExpOp, ONNXExpandOp, ONNXEyeLikeOp,
                    ONNXFlattenOp, ONNXFloorOp, ONNXGRUOp, ONNXGatherOp, ONNXGatherElementsOp, ONNXGatherNDOp, ONNXGeluOp, ONNXGemmOp,
                    ONNXGlobalAveragePoolOp, ONNXGlobalLpPoolOp, ONNXGlobalMaxPoolOp, ONNXGreaterOp, ONNXGreaterOrEqualOp, ONNXGridSampleOp,
                    ONNXGridSampleV16Op, ONNXGroupNormalizationOp, ONNXGroupNormalizationV18Op, ONNXHammingWindowOp, ONNXHannWindowOp,
                    ONNXHardSigmoidOp, ONNXHardSwishOp, ONNXHardmaxOp, ONNXIdentityOp, ONNXIfOp, ONNXInstanceNormalizationOp, ONNXIsInfOp, 
                    ONNXIsNaNOp, ONNXLRNOp, ONNXLSTMOp, ONNXLayerNormalizationOp, ONNXLeakyReluOp, ONNXLessOp, ONNXLessOrEqualOp, ONNXLogOp, ONNXLogSoftmaxOp,
                    ONNXLoopOp, ONNXLpNormalizationOp, ONNXLpPoolOp, ONNXMatMulOp, ONNXMatMulIntegerOp, ONNXMaxOp, ONNXMaxPoolOp, ONNXMaxRoiPoolOp, ONNXMaxUnpoolOp,
                    ONNXMeanOp, ONNXMeanVarianceNormalizationOp, ONNXMelWeightMatrixOp, ONNXMinOp, ONNXMishOp, ONNXModOp, ONNXMulOp, ONNXMultinomialOp, ONNXNegOp,
                    ONNXNegativeLogLikelihoodLossOp, ONNXNonMaxSuppressionOp, ONNXNonZeroOp, ONNXNotOp, ONNXOneHotOp, ONNXOptionalOp, ONNXOptionalGetElementOp,
                    ONNXOptionalHasElementOp, ONNXOrOp, ONNXPReluOp, ONNXPadOp, ONNXPadV18Op, ONNXPadV13Op, ONNXPadV11Op, ONNXPadV2Op, ONNXPowOp, ONNXQLinearConvOp,
                    ONNXQLinearMatMulOp, ONNXQuantizeLinearOp, ONNXRNNOp, ONNXRandomNormalOp, ONNXRandomNormalLikeOp, ONNXRandomUniformOp, ONNXRandomUniformLikeOp,
                    ONNXRangeOp, ONNXReciprocalOp, ONNXReduceL1Op, ONNXReduceL1V13Op, ONNXReduceL2Op, ONNXReduceL2V13Op, ONNXReduceLogSumOp, ONNXReduceLogSumV13Op,
                    ONNXReduceLogSumExpOp, ONNXReduceLogSumExpV13Op, ONNXReduceMaxOp, ONNXReduceMaxV18Op, ONNXReduceMaxV13Op, ONNXReduceMeanOp, ONNXReduceMeanV13Op,
                    ONNXReduceMinOp, ONNXReduceMinV18Op, ONNXReduceMinV13Op, ONNXReduceProdOp, ONNXReduceProdV13Op, ONNXReduceSumOp, ONNXReduceSumV11Op
            */
            // next time
            
                >(

                [&](auto op) { return printOperation(*this, op); })
            // Func ops.
            .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Default([&](Operation*) {
            return op.emitOpError("unable to find printer for op");
                });

        if (failed(status))
            return failure();

        /*

        if (hasDeferredEmission(&op))
            return success();

        if (getEmittedExpression() ||
            (isa<emitc::ExpressionOp>(op) &&
                shouldBeInlined(cast<emitc::ExpressionOp>(op))))
            return success();

        os << (trailingSemicolon ? ";\n" : "\n");
        */

        return success();
    }

    
    LogicalResult PythonEmitter::emitType(Location loc, Type type, bool isreturn) {
		//os << "#emitType is not implemented"<< type << "\n";

        if (auto tType = dyn_cast<TensorType>(type)) {

            if (!tType.hasRank())
                return emitError(loc, "cannot emit unranked tensor type");

            os << "shape = [";
            auto shape = tType.getShape();
            int i = 0;
            for (auto dimSize : shape) {
                
                if (ShapedType::isDynamic(dimSize))
                {
                    os << "-1";
                }
                else {
                    os << dimSize;
                }
                i++;
                if(i != shape.size())
                    os << ", ";
            }
            os << "]\n";

			os << "dtype = '" << tType.getElementType() << "'\n";

            if (isreturn)
            {
                os << "bld.return_types.append((shape, dtype))\n";
            }

            return success();
        }

        os << "#emitType is not implemented" << type << "\n";

		return success();
    }


    LogicalResult PythonEmitter::emitTypes(Location loc, ArrayRef<Type> types) {

        os << "#emitTypes is not implemented\n";

        switch (types.size()) {
        case 0:
            return success();
        case 1:
            return emitType(loc, types.front(), true);
        default:
            return emitTupleType(loc, types, true);
        }
    }


    LogicalResult PythonEmitter::emitTupleType(Location loc, ArrayRef<Type> types, bool isreturn) {
       
        if (failed(interleaveCommaWithError(
            types, os, [&](Type type) { return emitType(loc, type, isreturn); })))
            return failure();
        return success();
    }


    LogicalResult PythonEmitter::emitVariableDeclaration(OpResult result,
        bool assingvariable) {

        StringRef var;

        if (hasValueInScope(result)) {
            return result.getDefiningOp()->emitError(
                "result variable for the operation already declared");
        }
        if (failed(emitVariableDeclaration(result.getOwner()->getLoc(),
            result.getType(),
            var = getOrCreateName(result))))
            return failure();
        if (assingvariable)
        {
            last_var = var;
            os << var;
        }
           

        return success();
    }

    LogicalResult PythonEmitter::emitVariableDeclaration(Location loc, Type type,
        StringRef name) {

        if (failed(emitType(loc, type, false)))
            return failure();
        os << "bld.value_types['"<< name << "'] = (shape, dtype)\n";
        //os << "bld.add_input('" << name << "', shape, dtype)\n";
        return success();
    }


    StringRef PythonEmitter::getOrCreateName(Value val) {
        //os << val << "\n";

        if (!valueMapper.count(val)) {
            assert(
                "cacheDeferredOpResult should have been called on this value, "
                "update the emitOperation function.");

            valueMapper.insert(val, formatv("v{0}", ++valueCount));
        }
        return *valueMapper.begin(val);
    }


    StringRef PythonEmitter::getOrCreateNameArg(Value val) {
        //os << val << "\n";

        if (!valueMapper.count(val)) {
            valueMapper.insert(val, formatv("arg{0}", argCount++));
        }
        return *valueMapper.begin(val);
    }

    StringRef PythonEmitter::getOrCreateName(Block& block) {
        if (!blockMapper.count(&block))
            blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
        return *blockMapper.begin(&block);
    }


    LogicalResult PythonEmitter::emitAttribute(Location loc, Attribute attr) {
        auto printInt = [&](const APInt& val, bool isUnsigned) {
            if (val.getBitWidth() == 1) {
                if (val.getBoolValue())
                    os << "True";
                else
                    os << "False";
            }
            else {
                SmallString<128> strValue;
                val.toString(strValue, 10, !isUnsigned, false);
                os << strValue;
            }
        };


        auto printFloat = [&](const APFloat& val) {
            if (val.isFinite()) {
                SmallString<128> strValue;
                // Use default values of toString except don't truncate zeros.
                val.toString(strValue, 0, 0, false);
                os << strValue;
                switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
                case llvm::APFloatBase::S_IEEEhalf:
                    os << "f16";
                    break;
                case llvm::APFloatBase::S_BFloat:
                    os << "bf16";
                    break;
                case llvm::APFloatBase::S_IEEEsingle:
                    os << "f";
                    break;
                case llvm::APFloatBase::S_IEEEdouble:
                    break;
                default:
                    llvm_unreachable("unsupported floating point type");
                };
            }
            else if (val.isNaN()) {
                os << "NAN";
            }
            else if (val.isInfinity()) {
                if (val.isNegative())
                    os << "-";
                os << "INFINITY";
            }
            };

        // Print floating point attributes.
        if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
            if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(
                fAttr.getType())) {
                return emitError(
                    loc, "expected floating point attribute to be f16, bf16, f32 or f64");
            }
            printFloat(fAttr.getValue());
            return success();
        }
        if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
            if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(
                dense.getElementType())) {
                return emitError(
                    loc, "expected floating point attribute to be f16, bf16, f32 or f64");
            }

            ArrayRef<char> rawData = dense.getRawData();
            os << "'";
            printHexString(rawData);
            os << "'";
            return success();
        }

        // Print integer attributes.
        if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
            if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
                //printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
                return success();
            }
            if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
                printInt(iAttr.getValue(), false);
                return success();
            }
        }
        if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
            if (auto iType = dyn_cast<IntegerType>(
                cast<TensorType>(dense.getType()).getElementType())) {
                os << "'[";
                interleaveComma(dense, os, [&](const APInt& val) {
                    printInt(val, false);
                    });
                os << "]'";
                return success();
            }
            if (auto iType = dyn_cast<IndexType>(
                cast<TensorType>(dense.getType()).getElementType())) {
                os << "'[";
                interleaveComma(dense, os,
                    [&](const APInt& val) { printInt(val, false); });
                os << "]'";
                return success();
            }
        }

        // Print symbolic reference attributes.
        if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
            if (sAttr.getNestedReferences().size() > 1)
                return emitError(loc, "attribute has more than 1 nested reference");
            os << sAttr.getRootReference().getValue();
            return success();
        }

        /*
        if (auto sAttr = dyn_cast<StringAttr>(attr)) {
            os << "'\"" << sAttr.getValue() << "\"'";
            return success();
        }*/

        // Print type attributes.
        if (auto type = dyn_cast<TypeAttr>(attr))
            return emitType(loc, type.getValue(), false);

        return emitError(loc, "cannot emit attribute: ") << attr;
    }


    LogicalResult PythonEmitter::emitAssignPrefix(Operation& op) {


        switch (op.getNumResults()) {
        case 0:
            break;
        case 1: {
            OpResult result = op.getResult(0);
            if (failed(emitVariableDeclaration(result, true)))
                return failure();
            os << " = ";
            break;
        }
        default:
            os << "std::tie(";
            interleaveComma(op.getResults(), os,
                [&](Value result) { os << getOrCreateName(result); });
            os << ") = ";
        }
        return success();
    }


    LogicalResult PythonEmitter::printValue(Location loc, Attribute attr)
    {
        auto printInt = [&](const APInt& val, bool isUnsigned) {
            if (val.getBitWidth() == 1) {
                if (val.getBoolValue())
                    os << "True";
                else
                    os << "False";
            }
            else {
                SmallString<128> strValue;
                val.toString(strValue, 10, !isUnsigned, false);
                os << strValue;
            }
            };


        auto printFloat = [&](const APFloat& val) {
            if (val.isFinite()) {
                SmallString<128> strValue;
                // Use default values of toString except don't truncate zeros.
                val.toString(strValue, 0, 0, false);
                os << strValue;
            }
            else if (val.isNaN()) {
                os << "np.nan";
            }
            else if (val.isInfinity()) {
                if (val.isNegative())
                    os << "-";
                os << "np.infinity";
            }
            };

        // Print floating point attributes.
        if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
            if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(
                fAttr.getType())) {
                return emitError(
                    loc, "expected floating point attribute to be f16, bf16, f32 or f64");
            }
            printFloat(fAttr.getValue());
            return success();
        }
        if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
            if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(
                dense.getElementType())) {
                return emitError(
                    loc, "expected floating point attribute to be f16, bf16, f32 or f64");
            }

            ArrayRef<char> rawData = dense.getRawData();

            printHexString(rawData);
            return success();
        }

        // Print integer attributes.
        if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
            if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
                printInt(iAttr.getValue(),false);
                return success();
            }
            if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
                printInt(iAttr.getValue(), false);
                return success();
            }
        }
        if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
            if (auto iType = dyn_cast<IntegerType>(
                cast<TensorType>(dense.getType()).getElementType())) {
                os << '[';
                interleaveComma(dense, os, [&](const APInt& val) {
                       printInt(val, false);
                    });
                os << ']';
                return success();
            }
            if (auto iType = dyn_cast<IndexType>(
                cast<TensorType>(dense.getType()).getElementType())) {
                os << '[';
                interleaveComma(dense, os,
                    [&](const APInt& val) { printInt(val, false); });
                os << ']';
                return success();
            }
        }

        // Print symbolic reference attributes.
        if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
            if (sAttr.getNestedReferences().size() > 1)
                return emitError(loc, "attribute has more than 1 nested reference");
            os << sAttr.getRootReference().getValue();
            return success();
        }


        if (auto sAttr = dyn_cast<StringAttr>(attr)) {
            os << "'" << sAttr.getValue() << "'";
            return success();
        }

        // Print type attributes.
        if (auto type = dyn_cast<TypeAttr>(attr))
            return emitType(loc, type.getValue(), false);


        os << attr;

        return success();
    }



    LogicalResult PythonEmitter::emitOuter()
    {
        for (auto iter : outputMapper) {
			Value val = iter.first;
			StringRef name = iter.second;

			if (failed(emitType(val.getLoc(), val.getType(), false)))
				return failure();

            os << iter.first << " " << iter.second << "\n";
        }
        return success();
    }

    void PythonEmitter::resetValueCounter() { valueCount = 0; }
    bool PythonEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }


    LogicalResult translateToPython(Operation* op, raw_ostream& os,
        bool declareVariablesAtTop,
        StringRef fileId) {
        PythonEmitter emitter(os, declareVariablesAtTop, fileId);
        return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
    }

}
