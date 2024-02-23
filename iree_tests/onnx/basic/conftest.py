# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
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

    compile_target_backend: str
    run_module_device: str
    expect_compilation_success: bool
    expect_run_success: bool


def pytest_collect_file(parent, file_path):
    if file_path.name.endswith("model.mlir"):
        print(f"Found model.mlir at '{file_path.parent.name}'")
        return MlirFile.from_parent(parent, path=file_path)


class MlirFile(pytest.File):
    def collect(self):
        print(f"  MlirFile with path: '{self.path}'")

        test_name = self.path.parent.name

        # Note: this could be a glob() if we want to support multiple
        # input/output test cases per test folder.
        test_data_flagfile_path = self.path.parent / "test_data_flags.txt"
        if not test_data_flagfile_path.exists():
            print(f"Missing test_data_flags.txt for test '{test_name}'")
            return []

        # TODO(scotttodd): load config file(s), populate one spec per config
        #   (then join the config name with the test name)
        # * environment variable(s) for defining configs?
        #   - list of directories?
        #   - direct list of files?

        spec = IreeCompileAndRunTestSpec(
            input_mlir_path=self.path,
            data_flagfile_path=test_data_flagfile_path,
            compile_target_backend="llvm-cpu",
            run_module_device="local-task",
            expect_compilation_success=True,
            expect_run_success=True,
        )
        yield IreeCompileRunItem.from_parent(self, name=test_name, spec=spec)


class IreeCompileRunItem(pytest.Item):
    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

    def runtest(self):
        print("Running test with spec:")
        pprint(self.spec)

        # TODO(scotttodd): output to a temp path
        # TODO(scotttodd): parameterize this name
        # TODO(scotttodd): log a reproducer / command called
        compiled_model_path = self.spec.input_mlir_path.parent / "model_cpu.vmfb"
        exec_args = [
            "iree-compile",
            str(self.spec.input_mlir_path),
            f"--iree-hal-target-backends={self.spec.compile_target_backend}",
            "-o",
            str(compiled_model_path),
        ]
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


# Newline after "collecting ..." in output (remove when done debugging?)
print("")
