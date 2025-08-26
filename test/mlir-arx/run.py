#%%
import os
import random
import math
import difflib

def generate_random_string(length):
    text = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ''.join(random.choice(text) for _ in range(length))


test_list = [   
    ("MNIST : C++ to ARX C++", f'{os.getcwd()}/code_to_arx.py ', '-i', "./code/mnist_onnx.h", "./arx/mnist_onnx.h"),
    ("MNIST : EmitC to C++", "mlir-translate -mlir-to-cpp ", '', "./emitc/onnx_to_emitc.mlir", "./code/mnist_onnx.h"),
    ("MNIST : Onnx to EmitC", "onnx-mlir --maccel=ARX --EmitEmitCIR --EmitMLIR", '', "./onnx/mnist-12-int8.onnx", "./emitc/onnx_to_emitc.mlir"),
    ("MNIST : Onnx to ARX", "onnx-mlir --maccel=ARX --EmitARXIR --EmitMLIR", '', "./onnx/mnist-12-int8.onnx", "./harx/onnx_to_arx.mlir"),
    ("MNIST : Onnx to ONNX-MLIR", "onnx-mlir --maccel=ARX --EmitMLIR ", '', "./onnx/mnist-12-int8.onnx", "./mlir/onnx_to_mlir.mlir"),
]

pass_test = 0 
error_test = 0
run_test = 0
fail_list = []
success_list = []
for (name, test, input_param, input_file, pred_file) in test_list:
    pass_test += 1
    tmp_name = generate_random_string(30)
    tmp_file_path = os.path.join('output', tmp_name)
    
    shell_cmd = "{} {} {} -o {}".format(test, input_param, input_file, tmp_file_path)
    print(shell_cmd)
    os.system(shell_cmd)
    
    output_file_name = ""
    if os.path.exists(tmp_file_path + ".onnx.mlir"):
        output_file_name = tmp_file_path + ".onnx.mlir"
    elif os.path.exists(tmp_file_path):
        output_file_name = tmp_file_path
    else:
        raise Exception("File not found: " + test)

    with open(os.path.join(input_file), 'rb') as f1:
        input = f1.readlines()
        
    with open(os.path.join(pred_file), 'rb') as f1:
        pred = f1.readlines()[1:]
        pred = [str(line, 'utf-8') for line in pred]

    with open(os.path.join(output_file_name), 'rb') as f1:
        output = f1.readlines()[1:]
        output = [str(line, 'utf-8') for line in output]

    diff = list(difflib.context_diff(pred, output))
    if len(diff) > 0:
        print("❌ Test failed for: " + input_file)
        print("Differences found:")
        for line in diff:
            print(line, end='')
        error_test += 1
        fail_list.append(name)
    else:
        print("✅ Test passed for: " + input_file)
        run_test += 1
        success_list.append(name)
    
    if os.path.exists(tmp_file_path + ".tmp"):
        os.remove(tmp_file_path + ".tmp")
    if os.path.exists(tmp_file_path + ".onnx.mlir"):
        os.remove(tmp_file_path + ".onnx.mlir")
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

print()
print("Summary:")
print(f"Total tests run: {pass_test}")
print(f"✅ Total tests passed: {run_test}")
for success in success_list:
    print(f"\tPassed test: {success}")
print(f"❌ Total tests failed: {error_test}")
for fail in fail_list:
    print(f"\tFailed test: {fail}")

if error_test > 0:
    raise Exception("Some tests failed. Please check the output above.")
    os._exit(1)
else:
    os._exit(0)  # Exit the script cleanly
    
# %%
