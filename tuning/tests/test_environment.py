# SPDX-License-Identifier: Apache-2.0
"""The environment record's CUDA toolkit: the loaded library's build_info first, nvcc only when the
library names none (the same rule as the benchmark's record)."""
import eldpc

from eldpctune import env, report

GPU_BUILD_INFO = ("eldpc 0.1.0; core=compiler=GNU 13.3.0; flags=-O3; build_type=Release; "
                  "isa=avx2; lib=fastbpgpu6g; cuda=NVIDIA 13.3.73; cuda_arch=sm_89,sm_120")
CPU_BUILD_INFO = "eldpc 0.1.0; core=compiler=GNU 13.3.0; flags=-O3; isa=avx2; lib=fastbpcpu6g"
NVCC_TEXT = ("nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.2, "
             "V13.2.86\nBuild cuda_13.2")


def test_the_toolkit_comes_from_build_info_and_falls_back_to_nvcc():
    assert env.toolkit_record(GPU_BUILD_INFO, []) == {
        "version": "13.3", "nvcc": "V13.3.73", "first_line": None, "source": "build_info"}
    # an nvcc that disagrees is ignored when the library names its toolkit
    commands = [{"command": "nvcc --version", "exit_code": 0, "stdout": NVCC_TEXT, "stderr": ""}]
    assert env.toolkit_record(GPU_BUILD_INFO, commands)["source"] == "build_info"
    assert env.toolkit_record(CPU_BUILD_INFO, commands) == {
        "version": "13.2", "nvcc": "V13.2.86", "source": "nvcc", "command": "nvcc --version",
        "first_line": "nvcc: NVIDIA (R) Cuda compiler driver"}
    missing = [{"command": "nvcc --version", "exit_code": None, "stdout": "", "stderr": "not found"}]
    assert env.toolkit_record(CPU_BUILD_INFO, missing) is None
    assert env.toolkit_record("unavailable: no library", []) is None


def _capture(monkeypatch, build_info):
    ran = []

    def run_command(argv, timeout=30.0, display=None):
        ran.append(display or " ".join(argv))
        text = NVCC_TEXT if "nvcc" in argv[0] or display == "$CUDACXX --version" else "GPU, 1, 8.9"
        return {"command": display or " ".join(argv), "exit_code": 0, "stdout": text, "stderr": ""}

    monkeypatch.setattr(env, "run_command", run_command)
    monkeypatch.setattr(eldpc, "build_info", lambda: build_info)
    monkeypatch.setattr(eldpc, "isa", lambda: "avx2")
    monkeypatch.setattr(eldpc, "devices", lambda: [])
    monkeypatch.delenv("CUDACXX", raising=False)
    return env.capture_environment(), ran


def test_capture_environment_asks_nvcc_only_without_a_build_info_toolkit(monkeypatch):
    record, ran = _capture(monkeypatch, GPU_BUILD_INFO)
    assert record["toolkit"]["source"] == "build_info" and record["toolkit"]["version"] == "13.3"
    assert not any("nvcc" in command for command in ran)
    assert [c["command"] for c in record["commands"]] == [" ".join(env.GPU_QUERY)]
    record, ran = _capture(monkeypatch, CPU_BUILD_INFO)
    assert record["toolkit"]["source"] == "nvcc" and record["toolkit"]["version"] == "13.2"
    assert "nvcc --version" in ran


def test_pareto_md_names_the_toolkit_and_its_source():
    assert report.toolkit_text({"version": "13.3", "nvcc": "V13.3.73", "source": "build_info"}) \
        == "13.3 (V13.3.73, from the library's build_info)"
    assert report.toolkit_text({"version": "13.2", "nvcc": "V13.2.86", "source": "nvcc",
                                "command": "nvcc --version"}) \
        == "13.2 (V13.2.86, from `nvcc --version`)"
    assert report.toolkit_text(None) == "not recorded"


def test_the_environment_section_of_pareto_md_has_the_toolkit_line():
    from test_report import _results

    results = _results(True)
    results["environment"]["toolkit"] = env.toolkit_record(GPU_BUILD_INFO, [])
    text = report.render_pareto_md(results)
    assert "- CUDA toolkit: 13.3 (V13.3.73, from the library's build_info)" in text
