# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import subprocess
import sys

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent

# Test directory generated by iree_tests/onnx/scripts/generate_onnx_tests.py
# TODO(scotttodd): flag to run across any compatible test suite folder
TEST_SOURCE_ROOT = REPO_ROOT / "iree_tests/onnx/node/generated"

# Write lists of tests that passed/failed to compile.
COMPILE_SUCCESSES_FILE = REPO_ROOT / "iree_tests/onnx/node/compile_successes.txt"
COMPILE_FAILURES_FILE = REPO_ROOT / "iree_tests/onnx/node/compile_failures.txt"


def find_tests(root_dir_path):
    test_dir_paths = []
    for test_dir_path in root_dir_path.iterdir():
        if not test_dir_path.is_dir():
            continue
        model_file_path = test_dir_path / "model.mlir"
        test_data_flagfile_path = test_dir_path / "test_data_flags.txt"
        if model_file_path.exists() and test_data_flagfile_path.exists():
            test_dir_paths.append(test_dir_path)

    print(f"Found {len(test_dir_paths)} tests")
    return sorted(test_dir_paths)


def compile_test(test_dir_path):
    test_model_path = test_dir_path / "model.mlir"

    # TODO(scotttodd): compile for other configurations
    #   (this should be parameterized with a test runner - pytest or ctest)
    compiled_model_path = test_dir_path / "model_cpu.vmfb"
    exec_args = [
        "iree-compile",
        str(test_model_path),
        "--iree-hal-target-backends=llvm-cpu",
        "-o",
        str(compiled_model_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        # print(
        #     f"  {test_dir_path.name[5:]} compile failed,\n    stdout: {ret.stdout},\n    stderr: {ret.stderr}",
        #     file=sys.stderr,
        # )
        # print(f"  {test_dir_path.name[5:]} compile failed", file=sys.stderr)
        return False

    config_flagfile_path = test_dir_path / "config_cpu_flags.txt"
    config_flagfile_lines = []
    config_flagfile_lines.append("--device=local-task\n")
    config_flagfile_lines.append(f"--module={compiled_model_path.name}\n")
    with open(config_flagfile_path, "wt") as f:
        f.writelines(config_flagfile_lines)

    return True


if __name__ == "__main__":
    test_dir_paths = find_tests(TEST_SOURCE_ROOT)

    # TODO(scotttodd): clear .vmfb files from source/build dir?

    print(f"Compiling tests in '{TEST_SOURCE_ROOT}'")

    print("******************************************************************")
    passed_compiles = []
    failed_compiles = []
    # TODO(scotttodd): parallelize this (or move into a test runner like pytest)
    # num_tests_to_run = len(test_dir_paths)
    num_tests_to_run = min(20, len(test_dir_paths))
    for i in range(num_tests_to_run):
        test_dir_path = test_dir_paths[i]
        test_name = test_dir_path.name

        current_number = str(i).rjust(4, "0")
        progress_str = f"[{current_number}/{num_tests_to_run}]"
        print(f"{progress_str}: Compiling {test_name}")

        test_dir_path = Path(TEST_SOURCE_ROOT) / test_name
        result = compile_test(test_dir_path)
        if result:
            passed_compiles.append(test_name)
        else:
            failed_compiles.append(test_name)
    print("******************************************************************")

    passed_compiles.sort()
    failed_compiles.sort()

    with open(COMPILE_SUCCESSES_FILE, "wt") as f:
        f.write("\n".join(passed_compiles))
    with open(COMPILE_FAILURES_FILE, "wt") as f:
        f.write("\n".join(failed_compiles))

    print(f"Compile pass count: {len(passed_compiles)}")
    print(f"Compile fail count: {len(failed_compiles)}")