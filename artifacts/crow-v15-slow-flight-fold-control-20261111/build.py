from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-slow-flight-fold-control-20261111';B=R/'build/crow-v15-slow-flight-fold-control-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/"artifacts/crow-v15-contact-gated-flight-toes-20261111/curl-gait.metal").read_text()
flight=(R/"artifacts/crow-v15-contact-gated-flight-toes-20261111/curl-flight.metal").read_text()
needle="        // Keep the training-only teacher active through the final target."
assert base.count(needle)==1 and flight.count(needle)==1
extra="""        // Experiment: physical joint targets share the liftoff/toe ramp and release before the final approach.
        if (fixedFlightController && state.episode.z==10u && dispatch.counts.w==40u && action>=7u && action<=12u) {
            const uint joint=(action-7u)%3u;
            const float blend=float(((state.status.w>>16u)&15u)|(((state.status.w>>28u)&7u)<<4u))/100.0f;
            const float tuck=joint==0u ? 0.40f : joint==1u ? 0.50f : 0.30f;
            navigationTeacherRequested=mix(navigationTeacherRequested,tuck,clamp(blend,0.0f,1.0f));
        }
"""
counter_needle="            state.status.w=(state.status.w & ~(31u<<23u)) | (nextToeFlight<<23u);"
counter_extra="""            // Separate 100-tick physical leg transition; release after waypoint four.
            const uint foldMask=(15u<<16u)|(7u<<28u);
            const uint oldFold=((state.status.w>>16u)&15u)|(((state.status.w>>28u)&7u)<<4u);
            const uint nextFold=curlFlight ? min(oldFold+1u,100u) : (oldFold>0u ? oldFold-1u : 0u);
            state.status.w=(state.status.w & ~foldMask)|((nextFold&15u)<<16u)|((nextFold>>4u)<<28u);
"""
assert base.count(counter_needle)==1 and flight.count(counter_needle)==1
variants={"tuck-gait":base.replace(needle,extra+needle).replace(counter_needle,counter_needle+"\n"+counter_extra),"tuck-flight":flight.replace(needle,extra+needle).replace(counter_needle,counter_needle+"\n"+counter_extra)}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
for name,s in variants.items():
 source=A/(name+'.metal');source.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(source),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);out[name]={'source_sha256':h(source),'metallib_sha256':h(ml)}
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
