from pathlib import Path
s=Path('Collision.before.metal').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a)
 s=s.replace(a,b)
replace('        bool vertexDone=false;\n        for(uint iteration=0;iteration<budget;++iteration) {', '        bool vertexDone=false;\n        uint diagnosticReason=1u,diagnosticBits=0u; float diagnosticLower=0.f;\n        for(uint iteration=0;iteration<budget;++iteration) {')
replace('            const auto node=pending[--count];\n            const float tl=', '            const auto node=pending[--count];\n            diagnosticLower=node.lower; diagnosticBits=0u;\n            const float tl=')
replace('            if(positive) { if(count==0){vertexDone=true;break;} continue; }', '            if(positive) { diagnosticBits|=1u; if(count==0){vertexDone=true;break;} continue; }')
replace('            if(negativeWitness&&tl>0.f&&leftValue.lo>0&&leftValue.hi<band*denominatorLower) {', '            diagnosticBits|=(negativeWitness?2u:0u)|(tl>0.f?4u:0u)|(leftValue.lo>0?8u:0u)|(leftValue.hi<band*denominatorLower?16u:0u);\n            if(negativeWitness&&tl>0.f&&leftValue.lo>0&&leftValue.hi<band*denominatorLower) {')
replace('            if(count+2>32||left.upper==node.lower||left.upper==node.upper)break;', '            if(count+2>32){diagnosticReason=2u;break;}\n            if(left.upper==node.lower||left.upper==node.upper){diagnosticReason=3u;break;}')
replace('        if(!vertexDone){clear=false;lower=-1;return true;}', '        if(!vertexDone){clear=false;lower=-1.f-diagnosticLower;upper=float(diagnosticReason*65536u+corner*8192u+diagnosticBits*64u+count);return true;}')
replace('        if(polynomialLower<0) {\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;', '        if(polynomialLower<0) {\n            record.intervalAndDistance.x=polynomialUpper;\n            record.intervalAndDistance.y=-1.f-polynomialLower;\n            record.flags|=MR_CCD_PAIR_UNRESOLVED;record.status=MR_STEP_DID_NOT_CONVERGE;')
Path('Collision.instrumented.metal').write_text(s)
