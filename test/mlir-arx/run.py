#%%
import os
import random
import math
import difflib

def generate_random_string(length):
    text = "1234567890_=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ''.join(random.choice(text) for _ in range(length)) + '.mlir'

list_of_test_onnx_to_arx = [
    "onnx-mlir-opt -maccel=ARX --shape-inference --constprop-onnx --convert-onnx-to-harx --canonicalize",
]

list_of_file_for_onnx_to_arx = [ 
    ("./mlir/uint_model_1x10x10.mlir", "./harx/uint_model_1x10x10.mlir"),
]

pass_test = 0 
error_test = 0
run_test = 0
for test in list_of_test_onnx_to_arx:
    for (input_file, pred_file) in list_of_file_for_onnx_to_arx:
        pass_test += 1
        tmp_name = generate_random_string(30)
        tmp_file_path = os.path.join('output', tmp_name)
        
        shell_cmd = "{} {} -o {}".format(test, input_file, tmp_file_path)
        print(shell_cmd)
        os.system(shell_cmd)
        if not os.path.exists(tmp_file_path):
            raise Exception("File not found: " + test)
        
        with open(os.path.join(input_file), 'r') as f1:
            input = f1.readlines()
            
        with open(os.path.join(pred_file), 'r') as f1:
            pred = f1.readlines()
                
        with open(os.path.join(tmp_file_path), 'r') as f1:
            output = f1.readlines()

        diff = list(difflib.context_diff(pred, output))
        if len(diff) > 0:
            print("❌ Test failed for: " + input_file)
            print("Differences found:")
            for line in diff:
                print(line, end='')
            error_test += 1
        else:
            print("✅ Test passed for: " + input_file)
            run_test += 1
        
        # Clean up the temporary file
        os.remove(tmp_file_path)

print()
print("Summary:")
print(f"Total tests run: {pass_test}")
print(f"✅ Total tests passed: {run_test}")
print(f"❌ Total tests failed: {error_test}")

if error_test > 0:
    raise Exception(f"Some tests failed. Total failed: {run_test}/{pass_test}")
else:
    print(f"All tests passed successfully! {run_test}/{pass_test}")
    
# %%
