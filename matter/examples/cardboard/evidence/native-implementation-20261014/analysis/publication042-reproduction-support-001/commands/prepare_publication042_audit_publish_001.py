from pathlib import Path
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
p=q/'archive_publication042_001.py';s=p.read_text().replace('publication-binding-reference041','publication-binding-reference042');p.write_text(s)
s=(q/'audit_publication041_001.py').read_text()
for a,b in [('publication041','publication042'),('publication-binding-reference041','publication-binding-reference042'),('summary-reference040','summary-reference041'),('2144','2234'),('source-integrated-measured-coupon-contact-254-002','source-integrated-globalization-contact-memory-259-003'),("len(f['source_sha256'])==849","len(f['source_sha256'])==852"),("binding['source254']","binding['source259']"),('publication042-reproduction-002','publication042-reproduction-001')]:s=s.replace(a,b)
s=s.replace("['native602','diagnostic-602','source255','source257']","['native609','strict-full-coupon-609','source261','source262','source265']")
a=s.index("cpu1=json.loads");b=s.index("\npkg=",a)
s=s[:a]+"""cpu2=json.loads((q/'analysis/source259-build-003/report.json').read_text());assert cpu2['all_pass'] and cpu2['all_inputs_unchanged'];assert sha(q/'build-source259-integrated-003/source-build.json')==binding['source259']['source_build_sha256']==cpu2['source_build_sha256']
for n,h in [('native611-integrated259-default-off-history-adjudication-001','native611_adjudication_sha256'),('native612-best-of-seven-phase-root-adjudication-002','native612_adjudication_sha256')]:
 assert sha(q/'analysis'/n/'report.json')==binding[h]
 assert json.loads((q/'analysis'/n/'report.json').read_text())['verdict'].startswith('PASS')
"""+s[b:]
s=s.replace("'canonical_native_trajectory_pending':True,'active602_255_257_inventory_excluded':True","'canonical_tiny_native_trajectory_qualified':True,'full_target_trajectory_qualified':False,'active609_260_261_262_265_excluded':True")
s=s.replace("['apps/NumiHumanRestingVisual.hpp','apps/NumiHumanRestingIntercostalRibAttachment.hpp','tools/test_intercostal_rib_motion_attachment.mm']","json.loads((q/'analysis/publication042-upstream-001/report.json').read_text())['paths']")
# Upstream report uses preserved_upstream_paths or changes; inspect at execution rather than assume its schema.
s=s.replace("for n in json.loads((q/'analysis/publication042-upstream-001/report.json').read_text())['paths']:assert (r/n).read_bytes()==git('show',parent+':'+n)","for n in subprocess.check_output(['git','diff','--name-only','89da10e5dd7d620db154c3cea279803a112a302a',parent],cwd=r,text=True).splitlines():assert (r/n).read_bytes()==git('show',parent+':'+n)")
p=q/'audit_publication042_001.py';assert not p.exists();p.write_text(s)
s=(q/'publish_publication041_001.py').read_text().replace('publication041','publication042').replace('publication-binding-reference041','publication-binding-reference042')
s=s.replace("'canonical_native_trajectory_pending':True,'active602_255_257_inventory_excluded':True","'canonical_tiny_native_trajectory_qualified':True,'full_target_trajectory_qualified':False,'active609_260_261_262_265_excluded':True")
a=s.index('msg.write_text("""');b=s.index('""")',a)+4
s=s[:a]+'''msg.write_text("""Repair cardboard globalization registration and preserve source-bound evidence

Integrate the default-off best-of CGLS path, exact mortar-query observer and checked17-buffer contact plan. Repair five missing Runtime pipeline registrations after the preserved enabled608 crash. Ten targets build and14 CPU checks pass. Default-off611 reproduces32 production files; enabled612 completes all seven tiny stages under original physical acceptance criteria, selecting11 strictly lower CGLS trials with exact tested fractions.

Capture actual cached failing-state action602, retain its failed control19 history, and independently replay two exact mortar threshold crossings. Preserve trace field errata, failed builds and unavailable historical source reconstruction bytes. Whole-runtime shipping memory admission, conservative CCD repair, compact trajectory parity and the fresh full strict609 run remain open or active.

Publish only bounded solver/component progress. Full solver correctness, mesh/time convergence, material calibration, analogue validation, physical box assembly and crush/impact prediction remain unqualified. Preserve concurrent Human changes.
""")'''+s[b:]
p=q/'publish_publication042_001.py';assert not p.exists();p.write_text(s)
# Syntax-check these unexecuted commands without generating bytecode.
for n in ['archive_publication042_001.py','audit_publication042_001.py','publish_publication042_001.py','prepare_publication042_support_001.py']:compile((q/n).read_text(),str(q/n),'exec')
print('publication042 commands syntax PASS')
