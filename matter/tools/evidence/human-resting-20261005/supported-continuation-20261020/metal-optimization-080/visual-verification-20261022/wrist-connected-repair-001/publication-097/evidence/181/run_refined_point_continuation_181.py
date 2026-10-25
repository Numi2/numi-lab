#!/usr/bin/env python3
"""Bounded orchestration of existing connected skin preparation tools."""
from pathlib import Path
import ast, argparse, hashlib, importlib.util, json, shutil, subprocess, sys, time
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=D/'full-connected-clearance-150'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
SLOT='/tmp/numi-human-geometry-slot.py'
DRIVER=Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/refined-point-181-template/point_driver.py')
TARGET=S/'run-035-current-point-exit-targets-001/build_run034_first_union_exits.py'
OWNER=D/'publication-089/canonical/matter/tools/human_skin_fat_bone_audit.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,j): Path(p).write_text(json.dumps(j,indent=2)+'\n')
def owner():
    s=importlib.util.spec_from_file_location('construction_audit_owner',OWNER)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def literal(value):
    if isinstance(value,Path): return ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(value))],keywords=[])
    return ast.parse(repr(value),mode='eval').body
def emit(template,dest,overrides,kind):
    tree=ast.parse(template.read_text())
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self,node):
            if kind=='driver' and type(node.value)is int and node.value in(1239,4156):
                return ast.copy_location(ast.Name(id='TARGET_NODE_COUNT' if node.value==1239 else 'SURFACE_COUNT',ctx=ast.Load()),node)
            return node
        def visit_BinOp(self,node):
            if kind=='driver' and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id=='R' and isinstance(node.right,ast.Constant) and node.right.value=='run-032-partial-iterate-exact-gated-scp-001/attempt-001/fit-report.json':
                return ast.copy_location(ast.Name(id='PARENT_FIT',ctx=ast.Load()),node)
            return self.generic_visit(node)
        def visit_BoolOp(self,node):
            # Linearized barriers guide proposal construction. Exact nonlinear
            # self intersection, not a sufficient linear surrogate, gates it.
            if kind=='driver' and isinstance(node.op,ast.And):
                node.values=[x for x in node.values if not(isinstance(x,ast.Name)and x.id=='self_linear_ok')]
                if len(node.values)==1:return self.visit(node.values[0])
            return self.generic_visit(node)
        def visit_Call(self,node):
            if kind=='driver' and isinstance(node.func,ast.Name) and node.func.id=='relax_pair_rows_for_projection':
                node.args[2]=ast.parse("min(PAIR_NUMERICAL_RELAXATION_MM, .25*min(float(r['preserved_margin_mm']) for r in meta))",mode='eval').body
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if kind=='target' and any(isinstance(x,ast.Name) and x.id=='expected'for x in node.targets):
                node.value=ast.Name(id='EXPECTED_QUERY_HASHES',ctx=ast.Load())
                return node
            return self.generic_visit(node)
    tree=Adapt().visit(tree)
    assert isinstance(tree.body[-1],ast.If), 'missing explicit main guard'
    additions=[ast.Assign(targets=[ast.Name(id=k,ctx=ast.Store())],value=literal(v))for k,v in overrides.items()]
    tree.body[-1:-1]=additions
    ast.fix_missing_locations(tree)
    text=ast.unparse(tree)+'\n'
    compile(text,str(dest),'exec')
    dest.write_text(text)
    for f in ('area_scp_support_023.py','self_separation_026.py'):
        if kind=='driver':shutil.copy2(DRIVER.with_name(f),dest.with_name(f))
    return sha(dest)
def run(command,log):
    start=time.monotonic()
    with Path(log).open('x')as out:
        r=subprocess.run([PY,SLOT,*map(str,command)],stdout=out,stderr=subprocess.STDOUT,timeout=900)
    write(str(log)+'.receipt.json',dict(command=[PY,SLOT,*map(str,command)],returncode=r.returncode,wall_seconds=time.monotonic()-start,log_sha256=sha(log)))
    if r.returncode:raise RuntimeError('stage failed: '+str(log))
def last_point(fit,candidate):
    for stage in reversed(fit['solver']['stages']):
        for trial in stage.get('trials',[]):
            if trial.get('accepted'):
                path=Path(trial['exact_point_gate']['command'][-1])/'summary.json'
                if read(path)['candidate_sha256']==sha(candidate):return path
    raise ValueError('no complete point audit matches final candidate')
def prepare(candidate,point,surface,parent,out,execute):
    o=owner();ip=read(point);sr=read(surface);previous=read(parent)
    assert ip['candidate_sha256']==sr['candidate_sha256']==sha(candidate)
    assert ip['complete_surface_audit']is True and ip['exact_PLC_surface_pairs']==0
    count=o.summarize_surface_crossing_report(sr)
    if not ip['inside_nodes']:
        out.mkdir(parents=True,exist_ok=False)
        write(out/'result.json',dict(status='POINT_CLEARANCE_COMPLETE_SURFACE_METHOD_REQUIRED',surface_pairs=count['total_crossing_pairs'],candidate=str(candidate),native_admission=False))
        return
    out.mkdir(parents=True,exist_ok=False)
    query=point.parent/'query'
    expected=dict(candidate=sha(candidate),**{'query_'+k:sha(query/(k+'.csv'))for k in ('points','pairs','nodes')})
    target=out/'targets';target.mkdir()
    targetpy=target/'build_targets.py'
    emit(TARGET,targetpy,dict(CAND=candidate,QUERY=query,OUT=target,RPT=target/'nearest-union-exits.json',NPZ=target/'nearest-union-exits.npz',EXPECTED_QUERY_HASHES=expected),'target')
    pins=dict(candidate_sha256=sha(candidate),point_summary_sha256=sha(point),surface_report_sha256=sha(surface),parent_fit_sha256=sha(parent),driver_template_sha256=sha(DRIVER),target_template_sha256=sha(TARGET))
    write(out/'source-inputs.json',pins)
    if not execute:return
    run([PY,targetpy],out/'targets.log')
    tj=read(target/'nearest-union-exits.json')
    assert tj['status']=='PASS_FIRST_UNION_EXIT_TARGETS_ONLY' and tj['passing_targets']==ip['inside_nodes']
    fitdir=out/'fit';fitdir.mkdir()
    receipt=out/'surface-count.json';write(receipt,count)
    # Parse the frozen literal PIN dictionary without executing the template.
    t=ast.parse(DRIVER.read_text())
    oldpins=next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='PIN'for x in n.targets))
    paths=dict(start=candidate,target_start=candidate,inside_report=point,target_json=target/'nearest-union-exits.json',target_npz=target/'nearest-union-exits.npz',old_trial=candidate,crossing_report=surface,crossing_count=receipt,parent_fit=parent)
    pins={**oldpins,**{k:sha(v)for k,v in paths.items()}}
    overrides=dict(START=candidate,TARGET_START=candidate,INSIDE_REPORT=point,TARGET_JSON=paths['target_json'],TARGET_NPZ=paths['target_npz'],OLD_TRIAL=candidate,CROSSING_REPORT=surface,CROSSING_COUNT=receipt,OUT=fitdir/'attempt-001',PARENT_FIT=parent,PIN=pins,TARGET_NODE_COUNT=ip['inside_nodes'],SURFACE_COUNT=count['total_crossing_pairs'],STEP_MM=1.0,MAX_SOLVE_S=20.0)
    driver=fitdir/'run_connected_fit.py';emit(DRIVER,driver,overrides,'driver')
    run([PY,driver],out/'fit.log')
    fp=fitdir/'attempt-001/fit-report.json';fj=read(fp);cp=fitdir/'attempt-001/candidate.npz'
    if fj['solver']['accepted_steps']==0:
        write(out/'result.json',dict(status='NO_ACCEPTED_CONSTRUCTION_PROGRESS',fit_report_sha256=sha(fp),candidate_sha256=sha(cp),native_admission=False));return
    pp=last_point(fj,cp)
    merit=o.assess_point_clearance_construction_progress(ip,read(pp))
    assert merit['construction_progress']
    audit=out/'audit-159'
    run([PY,D/'audit_skin_bone_surfaces_159.py',cp,audit,'--max-seconds','600'],out/'audit159.log')
    after=o.summarize_surface_crossing_report(read(audit/'report.json'),expected_bank_keys={(b['body'],b['stable_bone_id'])for b in sr['banks']})
    result=dict(status='CONSTRUCTION_PROGRESS_ONLY',candidate=str(cp),candidate_sha256=sha(cp),point_summary=str(pp),surface_report=str(audit/'report.json'),parent_fit=str(fp),before_surface_pairs=count['total_crossing_pairs'],after_surface_pairs=after['total_crossing_pairs'],point_merit=merit,native_admission=False,accepted_physical_steps=0)
    write(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='point_merit'}),flush=True)
    if after['total_crossing_pairs']>=count['total_crossing_pairs']:return
    return cp,pp,audit/'report.json',fp
def main():
    a=argparse.ArgumentParser()
    for k in ('candidate','point','surface','parent','out'):a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--rounds',type=int,default=4);a.add_argument('--prepare-only',action='store_true')
    args=a.parse_args()
    assert 1<=args.rounds<=8
    args.out.mkdir(parents=True,exist_ok=False)
    state=(args.candidate,args.point,args.surface,args.parent)
    for r in range(args.rounds):
        result=prepare(*state,args.out/f'round-{r+1:03d}',not args.prepare_only)
        if not result:break
        state=result
if __name__=='__main__':main()
