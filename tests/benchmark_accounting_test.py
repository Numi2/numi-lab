#!/usr/bin/env python3
"""Exercise benchmark accounting with controllable native API outcomes, without a GPU."""
import os, pathlib, shlex, subprocess, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
mock = r"""
#include "metalrobo/c_api.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <chrono>
#include <vector>
static unsigned envs, calls;
static const char* lastError="injected runtime failure";
static std::vector<MRTaskTransitionC> transitions;
static const char* mode() { const char* m = std::getenv("NUMI_BENCH_TEST_MODE"); return m ? m : ""; }
extern "C" {
MRTaskRolloutHandle* mr_create_task_rollout(const MRRunManifestC* m) {
    envs=m->profile.environment_count; calls=0; return reinterpret_cast<MRTaskRolloutHandle*>(1);
}
void mr_task_rollout_destroy(MRTaskRolloutHandle*) {}
const char* mr_last_error() { return lastError; }
const char* mr_task_rollout_device_name(const MRTaskRolloutHandle*) {
    return !std::strcmp(mode(),"null_device") ? nullptr : "mock";
}
MRTaskRolloutLayoutC mr_task_rollout_layout(const MRTaskRolloutHandle*) {
    if (calls==2) lastError="clobbered by later API";
    MRTaskRolloutLayoutC l{}; l.action_count=1; l.run_fingerprint=17; l.robot_fingerprint=23; return l;
}
int mr_task_rollout_advance(MRTaskRolloutHandle*, const float*, size_t, const uint32_t*, size_t,
    uint32_t steps, uint64_t, uint32_t, MRTaskRolloutAdvanceC* a) {
    ++calls;
    if (calls==2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    *a={}; a->successful_environment_steps=steps*envs;
    a->first_failing_environment=UINT32_MAX; a->first_failing_control_step=UINT32_MAX;
    if (calls==1 && !std::strcmp(mode(),"warmup")) {a->failed_environment_steps=1; --a->successful_environment_steps;}
    if (calls==2 && (!std::strcmp(mode(),"incomplete") || !std::strcmp(mode(),"error"))) {
        a->successful_environment_steps=2; a->failed_environment_steps=1;
        a->first_failing_environment=1; a->first_failing_control_step=1; a->first_gpu_status_code=11;
    }
    transitions.assign(steps*envs,{}); for (auto& t:transitions) t.reward=1;
    if (calls==1 && !std::strcmp(mode(),"warmup_physics_error")) transitions[0].physics_error=1;
    if (calls==1 && !std::strcmp(mode(),"warmup_nan")) transitions[0].reward=std::numeric_limits<float>::quiet_NaN();
    if (calls==2 && !std::strcmp(mode(),"physics_error")) transitions[0].physics_error=1;
    if (calls==2 && !std::strcmp(mode(),"nan")) transitions[0].reward=std::numeric_limits<float>::quiet_NaN();
    return (calls==2 && !std::strcmp(mode(),"error")) ||
           (calls==1 && !std::strcmp(mode(),"warmup_error")) ? -1 : 0;
}
const MRTaskTransitionC* mr_task_rollout_transitions(const MRTaskRolloutHandle*) {return transitions.data();}
}
"""
with tempfile.TemporaryDirectory() as temp:
    temp = pathlib.Path(temp)
    (temp/"mock.cpp").write_text(mock)
    binary = temp/"benchmark"
    subprocess.run(["clang++","-std=c++23","-I",str(root/"include"),str(root/"apps/benchmark.cpp"),
                    str(temp/"mock.cpp"),"-o",str(binary)],check=True)
    def run(mode="",*args):
        return subprocess.run([str(binary),"--envs","2","--steps","3",*args],
                              env=dict(os.environ,NUMI_BENCH_TEST_MODE=mode),capture_output=True,text=True)
    ok=run(); assert ok.returncode==0,ok.stderr
    fields=dict(part.split("=",1) for part in shlex.split(ok.stdout))
    assert fields["requested_env_steps"]=="6" and fields["accepted_env_steps"]=="6"
    assert fields["failed_env_steps"]=="0" and fields["accepted_physics_steps"]=="24"
    assert fields["completed"]=="1" and fields["run_fingerprint"]=="17"
    expected = 6 * (1.0 / 60.0) / float(fields["wall_s"])
    assert abs(float(fields["accepted_simulated_s_per_wall_s"]) / expected - 1) < 0.002
    null_device=run("null_device")
    assert null_device.returncode==0 and 'device="unknown"' in null_device.stdout
    for mode in ["incomplete","error","physics_error","nan"]:
        trial=run(mode); assert trial.returncode!=0,(mode,trial.stdout)
        f=dict(part.split("=",1) for part in shlex.split(trial.stdout))
        assert f["completed"]=="0", (mode,f)
        if mode in ("incomplete","error"):
            assert f["accepted_env_steps"]=="2" and f["failed_env_steps"]=="1"
            assert f["first_gpu_status"]=="11" and "mean_reward" not in f
        if mode in ("physics_error","nan"):
            assert "mean_reward" not in f
        if mode=="error":
            assert "injected runtime failure" in trial.stderr and "clobbered" not in trial.stderr
    assert run("warmup").returncode!=0
    assert run("warmup_error").returncode!=0
    assert run("warmup_physics_error").returncode!=0
    assert run("warmup_nan").returncode!=0
    for opt in [("--envs","-1"),("--steps","4294967296"),("--seed","18446744073709551616"),("--seed"," -1")]:
        assert run("",*opt).returncode!=0,opt
print("benchmark accounting rejects incomplete, failed, invalid and oversized workloads")
