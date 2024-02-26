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

from pprint import pprint  # for debugging


# TODO(scotttodd): split into 'compile' -> 'run', with a dependency?
# TODO(scotttodd): or have the config files choose what to do?
@dataclass(frozen=True)
class IreeCompileAndRunTestSpec:
    """Spec for an IREE compilation test."""

    input_mlir_path: Path
    data_flagfile_path: Path

    config_name: str
    iree_compile_flags: str
    iree_run_module_flags: str

    expect_compilation_success: bool
    expect_run_success: bool


def pytest_collect_file(parent, file_path):
    if file_path.name.endswith("model.mlir"):
        # print(f"Found model.mlir at '{file_path.parent.name}'")
        return MlirFile.from_parent(parent, path=file_path)


class MlirFile(pytest.File):
    def collect(self):
        # print(f"  MlirFile with path: '{self.path}'")

        test_name = self.path.parent.name

        # Note: this could be a glob() if we want to support multiple
        # input/output test cases per test folder.
        test_data_flagfile_path = self.path.parent / "test_data_flags.txt"
        if not test_data_flagfile_path.exists():
            print(f"Missing test_data_flags.txt for test '{test_name}'")
            return []

        global _iree_test_configs
        for config in _iree_test_configs:

            # TODO(scotttodd): load config file(s), populate one spec per config
            #   (then join the config name with the test name)

            expect_compilation_success = True
            if test_name in config["expected_compile_failures"]:
                expect_compilation_success = False
            expect_run_success = True
            if test_name in config["expected_run_failures"]:
                expect_run_success = False

            spec = IreeCompileAndRunTestSpec(
                input_mlir_path=self.path,
                data_flagfile_path=test_data_flagfile_path,
                config_name=config["config_name"],
                iree_compile_flags=config["iree_compile_flags"],
                iree_run_module_flags=config["iree_run_module_flags"],
                expect_compilation_success=expect_compilation_success,
                expect_run_success=expect_run_success,
            )
            yield IreeCompileRunItem.from_parent(self, name=test_name, spec=spec)


class IreeCompileRunItem(pytest.Item):
    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

    def runtest(self):
        # print("Running test with spec:")
        # pprint(self.spec)

        if not self.spec.expect_compilation_success:
            pytest.xfail("Expected compilation to file")

        # TODO(scotttodd): output to a temp path
        # TODO(scotttodd): parameterize this name
        vmfb_name = f"model_{self.spec.config_name}.vmfb"
        compiled_model_path = self.spec.input_mlir_path.parent / vmfb_name
        exec_args = [
            "iree-compile",
            str(self.spec.input_mlir_path),
            "-o",
            str(compiled_model_path),
        ]
        exec_args.extend(self.spec.iree_compile_flags)
        process = subprocess.run(exec_args, capture_output=True)
        if process.returncode != 0:
            raise IreeCompileException(process)

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, IreeCompileException):
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


# TODO(scotttodd): move this into a function? Some way to share state across
#     pytest functions?
_iree_test_configs = []
_iree_test_config_files = [
    config for config in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if config
]

if not _iree_test_config_files:
    # Default to in-tree config files.
    THIS_DIR = Path(__file__).parent
    REPO_ROOT = THIS_DIR.parent
    _iree_test_config_files = [
        REPO_ROOT / "iree_tests/configs/config_cpu.json",
    ]

for config_file in _iree_test_config_files:
    with open(config_file) as f:
        _iree_test_configs.append(json.load(f))
