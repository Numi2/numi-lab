from pathlib import Path
for label in ('2e-4','2e-5'):
 source=Path('candidate-'+label+'.metal').read_text();lo=source.index('kernel void mr_world_reduce_wave32_status(');hi=source.index('// Ordered sparse generalized blocks',lo);body=source[lo:hi]
 a='''        if (source.code != MR_STEP_SUCCESS) {''';b='''        if (source.iterations != MR_INVALID_INDEX) {
            iterations = max(iterations, source.iterations);
        }
        if (source.code != MR_STEP_SUCCESS) {''';assert body.count(a)==1;body=body.replace(a,b)
 a='''        residuals = max(residuals, source.residuals);
        iterations = max(iterations, source.iterations);''';b='''        residuals = max(residuals, source.residuals);''';assert body.count(a)==1;body=body.replace(a,b)
 Path('candidate-'+label+'-reporting.metal').write_text(source[:lo]+body+source[hi:])
p=Path('sustained.mm');s=p.read_text();a='''auto &s=status[i];bool stepRestored=''';b='''auto &s=status[i];const float* prevQ=i?q+(i-1)*cap.nq:hw.model.defaultQ.data();const float* prevV=i?v+(i-1)*cap.nv:hw.model.defaultV.data();const MRRodNodeStateGPU* prevNodes=i?nodes+(i-1)*cap.nrod:world.defaultRodNodes().data();bool unchangedFromPrevious=std::memcmp(q+i*cap.nq,prevQ,cap.nq*sizeof(float))==0&&std::memcmp(v+i*cap.nv,prevV,cap.nv*sizeof(float))==0&&std::memcmp(nodes+i*cap.nrod,prevNodes,cap.nrod*sizeof(MRRodNodeStateGPU))==0;bool stepRestored=''';assert s.count(a)==1;s=s.replace(a,b)
a='''std::cout<<"{\\\"step_restored\\\":"<<(stepRestored?"true":"false")''';b='''std::cout<<"{\\\"unchanged_from_previous\\\":"<<(unchangedFromPrevious?"true":"false")<<",\\\"failure_tangent_diagnostic\\\":"<<s.residuals.z<<",\\\"step_restored\\\":"<<(stepRestored?"true":"false")''';assert s.count(a)==1;s=s.replace(a,b);Path('sustained-v3.mm').write_text(s)
