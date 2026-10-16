from pathlib import Path
import difflib
import hashlib
import json

O = Path(__file__).resolve().parents[1]
B = O.parents[1]
base = B / 'snapshots/source-cgls-block-right-preconditioned-149/matter/src/runtime.mm'
target = O / 'candidate/matter/src/runtime.mm'
text = base.read_text()

def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)

replace('''            if (const char* selector = std::getenv("NM_FEM_TRUE_RESIDUAL_ROOT")) {
                char* end = nullptr;
                const unsigned long parsed = std::strtoul(selector, &end, 10);
                if (end == selector || *end != '\\0' ||''', '''            if (const char* selector = std::getenv("NM_FEM_TRUE_RESIDUAL_ROOT")) {
                const bool everyControlStep = std::strcmp(selector, "all") == 0;
                char* end = nullptr;
                const unsigned long parsed = everyControlStep
                    ? request.controlStep : std::strtoul(selector, &end, 10);
                if ((!everyControlStep && (end == selector || *end != '\\0')) ||''')

replace('''                char* rootEnd = nullptr;
                char* iterationEnd = nullptr;
                const unsigned long root = std::strtoul(
                    baseCauchySelector, &rootEnd, 10);
                const bool allNewtonIterations = separator != nullptr &&''', '''                char* rootEnd = nullptr;
                char* iterationEnd = nullptr;
                const bool everyControlStep = separator != nullptr &&
                    separator - baseCauchySelector == 3 &&
                    std::strncmp(baseCauchySelector, "all", 3) == 0;
                const unsigned long root = everyControlStep
                    ? request.controlStep : std::strtoul(
                        baseCauchySelector, &rootEnd, 10);
                const bool allNewtonIterations = separator != nullptr &&''')

replace('''                if (separator == nullptr || rootEnd != separator ||
                    (!allNewtonIterations && iterationEnd != end) ||
                    !digitsOnly(baseCauchySelector, separator) ||''', '''                if (separator == nullptr ||
                    (!everyControlStep && rootEnd != separator) ||
                    (!allNewtonIterations && iterationEnd != end) ||
                    (!everyControlStep &&
                     !digitsOnly(baseCauchySelector, separator)) ||''')

replace('''                    rejectBaseCauchy(
                        "selector_must_be_control_step_colon_all_or_newton_0_to_256");
                } else {
                    femBaseCauchyFallbackRoot =''', '''                    rejectBaseCauchy(
                        "selector_must_be_control_step_colon_all_or_newton_0_to_256_or_exact_all_colon_all");
                } else if (everyControlStep &&
                           (!allNewtonIterations ||
                            femBaseCGLSIterations == 0u ||
                            !femBaseCGLSPrimaryPolicyRequested)) {
                    // All-control scope is admitted only for the existing
                    // bounded CGLS primary policy. Every selected control still
                    // passes the complete original capability/certificate gate.
                    rejectBaseCauchy(
                        "all_control_steps_requires_cgls_primary_policy_and_all_newton_suffix");
                } else {
                    femBaseCauchyFallbackRoot =''')

replace('''                        // The numerical fallback remains bounded to one
                        // explicit control step. The suffix may select one
                        // Newton state or every Newton state in that step.''', '''                        // A numeric selector remains bounded to its explicit
                        // control step. Exact all:all resolves to this control
                        // and repeats the full bounded capability gate below.''')

replace('''                                "fem_base_cauchy_all_newton_enabled={\\"schema\\":1,\\"root\\":%u,\\"scope\\":\\"all_newton_iterations_on_one_control_step\\",\\"maximum_trace_records\\":512,\\"physical_operator_changed\\":false}\\n",
                                femBaseCauchyFallbackRoot);''', '''                                "fem_base_cauchy_all_newton_enabled={\\"schema\\":1,\\"root\\":%u,\\"scope\\":\\"%s\\",\\"maximum_trace_records\\":512,\\"physical_operator_changed\\":false}\\n",
                                femBaseCauchyFallbackRoot,
                                everyControlStep
                                    ? "all_newton_iterations_on_all_control_steps"
                                    : "all_newton_iterations_on_one_control_step");''')

target.write_text(text)
patch = ''.join(difflib.unified_diff(base.read_text().splitlines(True),
    text.splitlines(True), fromfile='a/matter/src/runtime.mm',
    tofile='b/matter/src/runtime.mm'))
(O/'runtime.patch').write_text(patch)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(O/'preparation.json').write_text(json.dumps({
    'schema': 'numi.cardboard.all_controls_selector_preparation.v1',
    'base': str(base), 'base_sha256': sha(base),
    'candidate_sha256': sha(target), 'patch_sha256': sha(O/'runtime.patch'),
    'scope': 'opt-in exact all:all selector; existing per-control capability, private update, true residual, acceptance and rollback unchanged',
    'native_execution': False,
}, indent=2)+'\n')
print(json.dumps({'candidate': str(target), 'sha256': sha(target)}))
