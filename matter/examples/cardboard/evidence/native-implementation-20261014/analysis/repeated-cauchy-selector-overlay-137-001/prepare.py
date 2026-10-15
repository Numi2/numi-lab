from pathlib import Path
import difflib, hashlib, json

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/repeated-cauchy-selector-overlay-137-001'
S = B / 'snapshots/source-cache-projection-reuse-runtime-136'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p = 'matter/src/runtime.mm'
before = (S / p).read_text()
after = before

def replace(old, new):
    global after
    assert after.count(old) == 1, old
    after = after.replace(old, new)

replace('''            std::uint32_t femBaseCauchyFallbackIteration = 0u;
''', '''            std::uint32_t femBaseCauchyFallbackIteration = 0u;
            bool femBaseCauchyFallbackAllNewtonIterations = false;
''')
replace('''                const unsigned long iteration = separator == nullptr
                    ? 0ul : std::strtoul(separator + 1, &iterationEnd, 10);
''', '''                const bool allNewtonIterations = separator != nullptr &&
                    std::strcmp(separator + 1, "all") == 0;
                const unsigned long iteration = separator == nullptr ||
                    allNewtonIterations ? 0ul :
                    std::strtoul(separator + 1, &iterationEnd, 10);
''')
replace('''                    iterationEnd != end ||
                    !digitsOnly(baseCauchySelector, separator) ||
                    !digitsOnly(separator + 1, end) ||
''', '''                    (!allNewtonIterations && iterationEnd != end) ||
                    !digitsOnly(baseCauchySelector, separator) ||
                    (!allNewtonIterations &&
                     !digitsOnly(separator + 1, end)) ||
''')
replace('''                        "selector_must_be_control_step_colon_newton_0_to_256");
''', '''                        "selector_must_be_control_step_colon_all_or_newton_0_to_256");
''')
replace('''                    femBaseCauchyFallbackIteration =
                        static_cast<std::uint32_t>(iteration);
''', '''                    femBaseCauchyFallbackIteration =
                        static_cast<std::uint32_t>(iteration);
                    femBaseCauchyFallbackAllNewtonIterations =
                        allNewtonIterations;
''')
replace('''                        // This bounded experimental selector changes one
                        // exact control/Newton state only.
''', '''                        // The numerical fallback remains bounded to one
                        // explicit control step. The suffix may select one
                        // Newton state or every Newton state in that step.
''')
replace('''                                (!femTrueResidualNewtonSelectorValid ||
                                 femTrueResidualNewtonIteration !=
''', '''                                (!femTrueResidualNewtonSelectorValid ||
                                 femBaseCauchyFallbackAllNewtonIterations ||
                                 femTrueResidualNewtonIteration !=
''')
replace('''                request.controlStep == femBaseCauchyFallbackRoot &&
                !femBaseCauchyFallbackConfigValid) {
''', '''                !femBaseCauchyFallbackConfigValid) {
''')
replace('''                femBaseCauchyFallbackSelectedRoot &&
                nonlinearIteration == femBaseCauchyFallbackIteration;
''', '''                femBaseCauchyFallbackSelectedRoot &&
                (femBaseCauchyFallbackAllNewtonIterations ||
                 nonlinearIteration == femBaseCauchyFallbackIteration);
''')
# Keep the original exact-state event unchanged. An additional explicit event
# identifies the broader requested scope without mislabelling iteration zero.
replace('''                        femBaseCauchyFallbackSelectedRoot = true;
                        std::fprintf(stderr,
''', '''                        femBaseCauchyFallbackSelectedRoot = true;
                        if (femBaseCauchyFallbackAllNewtonIterations) {
                            std::fprintf(stderr,
                                "fem_base_cauchy_all_newton_enabled={\\\"schema\\\":1,\\\"root\\\":%u,\\\"scope\\\":\\\"all_newton_iterations_on_one_control_step\\\",\\\"physical_operator_changed\\\":false}\\n",
                                femBaseCauchyFallbackRoot);
                        }
                        std::fprintf(stderr,
''')

assert after != before
O.mkdir()
(O / 'base').mkdir()
(O / 'candidate').mkdir()
(O / 'base/runtime.mm').write_text(before)
(O / 'candidate/runtime.mm').write_text(after)
(O / 'selector.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True),
    fromfile='a/' + p, tofile='b/' + p)))
(O / 'prepare.py').write_bytes(Path(__file__).read_bytes())
(O / 'integration.json').write_text(json.dumps(dict(
    schema='numi.cardboard.repeated_cauchy_selector.v1',
    source_snapshot=str(S), path=p,
    base_sha256=sha(S / p), candidate_sha256=sha(O / 'candidate/runtime.mm'),
    scope='Adds control:all at the same physical operator and numerical gates; requires true-residual certification at every selected Newton state. Invalid selectors fail before every step. No native execution yet.',
    physical_steps=0), indent=2) + '\n')
print('prepared137-selector', sha(O / 'candidate/runtime.mm'))
