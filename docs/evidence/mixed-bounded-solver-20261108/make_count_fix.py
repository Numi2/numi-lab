from pathlib import Path
source=Path('candidate-2.metal').read_text();start=source.index('kernel void mr_world_reduce_wave32_status(');end=source.index('// Ordered sparse generalized blocks',start);body=source[start:end]
a='''        if (source.code != MR_STEP_SUCCESS) {'''
b='''        if (source.iterations != MR_INVALID_INDEX) {
            iterations = max(iterations, source.iterations);
        }
        if (source.code != MR_STEP_SUCCESS) {'''
assert body.count(a)==1;body=body.replace(a,b)
a='''        residuals = max(residuals, source.residuals);
        iterations = max(iterations, source.iterations);''';b='''        residuals = max(residuals, source.residuals);''';assert body.count(a)==1;body=body.replace(a,b)
Path('candidate-2-count-fixed.metal').write_text(source[:start]+body+source[end:])
