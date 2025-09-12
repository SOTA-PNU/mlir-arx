#%%
import os
import random
import math
import difflib
import xml.etree.ElementTree as ET
from datetime import datetime

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

total_tests = 0 
failed_tests = 0
passed_tests = 0
fail_list = []
success_list = []
for (name, test, input_param, input_file, pred_file) in test_list:
    total_tests += 1
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
        failed_tests += 1
        fail_list.append(name)
    else:
        print("✅ Test passed for: " + input_file)
        passed_tests += 1
        success_list.append(name)
    
    # Keep artifacts for CI collection instead of deleting
    # Move files to artifacts directory for CI collection
    # Use CI_PROJECT_DIR if available (GitLab CI), otherwise use relative path
    project_root = os.environ.get('CI_PROJECT_DIR', os.path.join(os.getcwd(), '..', '..'))
    artifacts_dir = os.path.join(project_root, 'artifacts', 'test_results')
    print(f"DEBUG: Creating artifacts directory at: {artifacts_dir}")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create unique filename with test index and random suffix to avoid conflicts
    safe_name = f"{total_tests:02d}_{name.replace(' ', '_').replace(':', '_')}_{tmp_name[:8]}"
    
    if os.path.exists(tmp_file_path + ".tmp"):
        os.rename(tmp_file_path + ".tmp", os.path.join(artifacts_dir, f"{safe_name}.tmp"))
    if os.path.exists(tmp_file_path + ".onnx.mlir"):
        os.rename(tmp_file_path + ".onnx.mlir", os.path.join(artifacts_dir, f"{safe_name}.onnx.mlir"))
    if os.path.exists(tmp_file_path):
        os.rename(tmp_file_path, os.path.join(artifacts_dir, f"{safe_name}.out"))

print()
print("Summary:")
print(f"Total tests run: {total_tests}")
print(f"✅ Total tests passed: {passed_tests}")
for success in success_list:
    print(f"\tPassed test: {success}")
print(f"❌ Total tests failed: {failed_tests}")
for fail in fail_list:
    print(f"\tFailed test: {fail}")

def generate_junit_xml(artifacts_dir, success_list, fail_list, total_tests):
    """Generate JUnit XML format for GitLab CI integration"""
    # Create testsuite element
    testsuite = ET.Element('testsuite')
    testsuite.set('name', 'MNIST ARX Tests')
    testsuite.set('tests', str(total_tests))
    testsuite.set('failures', str(len(fail_list)))
    testsuite.set('errors', '0')
    testsuite.set('time', '0')
    testsuite.set('timestamp', datetime.now().isoformat())
    
    # Add successful test cases
    for test_name in success_list:
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', test_name)
        testcase.set('classname', 'ARXTests')
    
    # Add failed test cases
    for test_name in fail_list:
        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', test_name)
        testcase.set('classname', 'ARXTests')
        failure = ET.SubElement(testcase, 'failure')
        failure.set('message', f'Test {test_name} failed')
        failure.text = f'Output mismatch detected in {test_name}'
    
    # Write XML file
    junit_file = os.path.join(artifacts_dir, 'test_results.xml')
    tree = ET.ElementTree(testsuite)
    ET.indent(tree, space="  ", level=0)
    tree.write(junit_file, encoding='utf-8', xml_declaration=True)
    return junit_file

# Save test summary to artifacts
project_root = os.environ.get('CI_PROJECT_DIR', os.path.join(os.getcwd(), '..', '..'))
artifacts_dir = os.path.join(project_root, 'artifacts', 'test_results')
print(f"DEBUG: Final artifacts directory: {artifacts_dir}")
os.makedirs(artifacts_dir, exist_ok=True)
print(f"DEBUG: Directory created successfully, contents: {os.listdir(artifacts_dir) if os.path.exists(artifacts_dir) else 'Does not exist'}")

# Generate text summary
summary_file = os.path.join(artifacts_dir, 'test_summary.txt')
with open(summary_file, 'w') as f:
    f.write("ARX Test Results Summary\n")
    f.write("========================\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total tests run: {total_tests}\n")
    f.write(f"✅ Total tests passed: {passed_tests}\n")
    f.write(f"❌ Total tests failed: {failed_tests}\n")
    f.write(f"Success rate: {(passed_tests/total_tests*100):.1f}%\n\n")
    
    if success_list:
        f.write("Passed Tests:\n")
        for success in success_list:
            f.write(f"  ✅ {success}\n")
        f.write("\n")
    
    if fail_list:
        f.write("Failed Tests:\n")
        for fail in fail_list:
            f.write(f"  ❌ {fail}\n")

# Generate JUnit XML for GitLab CI
junit_file = generate_junit_xml(artifacts_dir, success_list, fail_list, total_tests)
print(f"JUnit XML report generated: {junit_file}")

if failed_tests > 0:
    print(f"\n❌ {failed_tests} test(s) failed. Check artifacts for detailed results.")
    raise Exception("Some tests failed. Please check the output above.")
else:
    print(f"\n✅ All {total_tests} tests passed successfully!")
    os._exit(0)  # Exit the script cleanly
    
# %%
