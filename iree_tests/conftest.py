# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
import json
import os
import pytest
import subprocess


@dataclass(frozen=True)
class IreeCompileAndRunTestSpec:
    """Spec for an IREE compilation test."""

    input_mlir_path: Path
    data_flagfile_path: Path

    config_name: str
    iree_compile_flags: str
    iree_run_module_flags: str

    expect_compile_success: bool
    expect_run_success: bool
    skip_run: bool


def pytest_collect_file(parent, file_path):
    if file_path.name.endswith("model.mlir"):
        return MlirFile.from_parent(parent, path=file_path)


class MlirFile(pytest.File):

    def collect(self):
        test_name = self.path.parent.name

        # Note: this could be a glob() if we want to support multiple
        # input/output test cases per test folder.
        test_data_flagfile_path = self.path.parent / "test_data_flags.txt"
        if not test_data_flagfile_path.exists():
            print(f"Missing test_data_flags.txt for test '{test_name}'")
            return []

        global _iree_test_configs
        for config in _iree_test_configs:
            if test_name in config["skip_compile_tests"]:
                continue

            expect_compile_success = (
                test_name not in config["expected_compile_failures"]
            )
            expect_run_success = test_name not in config["expected_run_failures"]
            skip_run = test_name in config["skip_run_tests"]

            spec = IreeCompileAndRunTestSpec(
                input_mlir_path=self.path,
                data_flagfile_path=test_data_flagfile_path,
                config_name=config["config_name"],
                iree_compile_flags=config["iree_compile_flags"],
                iree_run_module_flags=config["iree_run_module_flags"],
                expect_compile_success=expect_compile_success,
                expect_run_success=expect_run_success,
                skip_run=skip_run,
            )
            yield IreeCompileRunItem.from_parent(self, name=test_name, spec=spec)


class IreeCompileRunItem(pytest.Item):
    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

        # TODO(scotttodd): output to a temp path
        vmfb_name = f"model_{self.spec.config_name}.vmfb"
        self.compiled_model_path = self.spec.input_mlir_path.parent / vmfb_name

    def runtest(self):
        # First test compilation...
        if not self.spec.expect_compile_success:
            pytest.xfail("Expected compilation to fail")
        self.test_compile()
        if not self.spec.expect_compile_success:
            return

        if self.spec.skip_run:
            return

        # ... then test runtime execution
        if not self.spec.expect_run_success:
            pytest.xfail("Expected run to fail")
        self.test_run()

    def test_compile(self):
        exec_args = [
            "iree-compile",
            str(self.spec.input_mlir_path),
            "-o",
            str(self.compiled_model_path),
        ]
        exec_args.extend(self.spec.iree_compile_flags)
        process = subprocess.run(exec_args, capture_output=True)
        if process.returncode != 0:
            raise IreeCompileException(process)

    def test_run(self):
        exec_args = [
            "iree-run-module",
            f"--module={self.compiled_model_path}",
            f"--flagfile={self.spec.data_flagfile_path}",
        ]
        exec_args.extend(self.spec.iree_run_module_flags)
        # TODO(scotttodd): swap cwd for a temp path?
        process = subprocess.run(
            exec_args, capture_output=True, cwd=self.spec.data_flagfile_path.parent
        )
        if process.returncode != 0:
            raise IreeRunException(process)

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, IreeCompileException):
            return "\n".join(excinfo.value.args)
        if isinstance(excinfo.value, IreeRunException):
            return "\n".join(excinfo.value.args)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"IREE compile: {self.name}"


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.

        super().__init__(
            f"Error invoking iree-compile\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n {' '.join(process.args)}\n\n"
        )


class IreeRunException(Exception):
    """Runtime exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.

        # TODO(scotttodd): log CWD as part of the reproducer
        #   (flagfiles use relative paths like `@input_0.npy`)
        super().__init__(
            f"Error invoking iree-run-module\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n {' '.join(process.args)}\n\n"
        )


# TODO(scotttodd): move this setup code into a (scoped) function?
#   Is there some way to share state across pytest functions?

# Load a list of configuration files following this schema:
#   {
#     "config_name": str,
#     "iree_compile_flags": list of str,
#     "iree_run_module_flags": list of str,
#     "skip_compile_tests": list of str,
#     "skip_run_tests": list of str,
#     "expected_compile_failures": list of str,
#     "expected_run_failures": list of str
#   }
#
# For example, to test the on CPU with the `llvm-cpu`` backend on the `local-task` device:
#   {
#     "config_name": "cpu",
#     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
#     "iree_run_module_flags": ["--device=local-task"],
#     "skip_compile_tests": [],
#     "skip_run_tests": [],
#     "expected_compile_failures": ["test_abs"],
#     "expected_run_failures": ["test_add"],
#   }
#
# First check for the `IREE_TEST_CONFIG_FILES` environment variable. If defined,
# this should point to a semicolon-delimited list of config file paths, e.g.
# `export IREE_TEST_CONFIG_FILES=~/iree/config_cpu.json;~/iree/config_gpu.json`.
_iree_test_configs = []
_iree_test_config_files = [
    config for config in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if config
]

# If no config files were specified via the environment variable, default to in-tree config files.
if not _iree_test_config_files:
    THIS_DIR = Path(__file__).parent
    REPO_ROOT = THIS_DIR.parent
    _iree_test_config_files = [
        REPO_ROOT / "iree_tests/configs/config_cpu.json",
    ]

for config_file in _iree_test_config_files:
    with open(config_file) as f:
        _iree_test_configs.append(json.load(f))
