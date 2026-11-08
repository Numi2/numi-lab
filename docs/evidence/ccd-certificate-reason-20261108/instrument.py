from pathlib import Path
p=Path('Collision.before.metal');s=p.read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a)
 s=s.replace(a,b)
replace('        bool vertexDone=false;\n        for(uint iteration=0;iteration<budget;++iteration) {', '        bool vertexDone=false;\n        uint diagnosticReason=1u; float diagnosticLower=0.f;\n        for(uint iteration=0;iteration<budget;++iteration) {')
replace('            const auto node=pending[--count];\n            const float tl=', '            const auto node=pending[--count];\n            diagnosticLower=node.lower;\n            const float tl=')
replace('            if(count+2>32||left.upper==node.lower||left.upper==node.upper)break;', '            if(count+2>32){diagnosticReason=2u;break;}\n            if(left.upper==node.lower||left.upper==node.upper){diagnosticReason=3u;break;}')
replace('        if(!vertexDone){clear=false;lower=-1;return true;}', '        if(!vertexDone){clear=false;lower=-1.f-diagnosticLower;upper=float(diagnosticReason*16u+corner);return true;}')
replace('        if(polynomialLower<0) {\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;', '        if(polynomialLower<0) {\n            record.intervalAndDistance.x=polynomialUpper;\n            record.intervalAndDistance.y=-1.f-polynomialLower;\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;')
Path('Collision.instrumented.metal').write_text(s)
