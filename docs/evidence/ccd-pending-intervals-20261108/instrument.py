from pathlib import Path
s=Path('Collision.before.metal').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a)
 s=s.replace(a,b)
replace('    thread uint& work) {\n    const bool planeA=', '    thread uint& work,thread float2& diagnostic) {\n    const bool planeA=')
replace('        if(!vertexDone){clear=false;lower=-1;return true;}', '''        if(!vertexDone){
            clear=false;lower=-1;
#if CCD_CAPTURE_SLOT < 5
            diagnostic=CCD_CAPTURE_SLOT<count ? float2(pending[CCD_CAPTURE_SLOT].lower,pending[CCD_CAPTURE_SLOT].upper) : float2(-1.f,-1.f);
#elif CCD_CAPTURE_SLOT == 5
            diagnostic=float2(box0.centerAndRadius.y,box1.centerAndRadius.y);
#elif CCD_CAPTURE_SLOT == 6
            diagnostic=box0.rotation.xy;
#elif CCD_CAPTURE_SLOT == 7
            diagnostic=box0.rotation.zw;
#elif CCD_CAPTURE_SLOT == 8
            diagnostic=end.xy;
#elif CCD_CAPTURE_SLOT == 9
            diagnostic=end.zw;
#elif CCD_CAPTURE_SLOT == 10
            diagnostic=float2(guard,tolerance);
#elif CCD_CAPTURE_SLOT == 11
            diagnostic=float2(n.y,plane0.centerAndRadius.y);
#elif CCD_CAPTURE_SLOT == 12
            diagnostic=float2(n.x,n.z);
#else
#error invalid capture slot
#endif
            return true;
        }''')
replace('    bool polynomialClear=false;float polynomialLower=0,polynomialUpper=0;uint polynomialWork=0;\n    if(certifiedBoxPlaneSweep(', '    bool polynomialClear=false;float polynomialLower=0,polynomialUpper=0;uint polynomialWork=0;float2 diagnostic=float2(0);\n    if(certifiedBoxPlaneSweep(')
replace('            dispatch.maxConservativeAdvancementIterations,polynomialClear,polynomialLower,polynomialUpper,polynomialWork)) {', '            dispatch.maxConservativeAdvancementIterations,polynomialClear,polynomialLower,polynomialUpper,polynomialWork,diagnostic)) {')
replace('        if(polynomialLower<0) {\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;', '        if(polynomialLower<0) {\n            record.intervalAndDistance.xy=diagnostic;\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;')
Path('Collision.instrumented.metal').write_text(s)
p=Path('probe.before.cpp').read_text(); old='for(int fixture:{1,2,12,13,14,15})'; assert p.count(old)==1;p=p.replace(old,'for(int fixture:{12})');Path('probe.cpp').write_text(p)
