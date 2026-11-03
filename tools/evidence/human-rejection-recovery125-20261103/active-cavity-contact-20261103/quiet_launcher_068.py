def configure(d,p,pin):
    runner=d/'run_native_plan.py'
    launch=runner.read_text()
    launch=launch.replace('if not sampled_gpu_entry and ', 'if False and not sampled_gpu_entry and ')
    launch=launch.replace('if early_sample is None and elapsed>20:', 'if False and early_sample is None and elapsed>20:')
    old="log=(D/(stage_name+'.stdout.log')).read_text()"
    new="""from collections import deque
log_tail=deque(maxlen=200); source_import_pass=False
with (D/(stage_name+'.stdout.log')).open() as streamed:
 for line in streamed:
  source_import_pass=source_import_pass or 'registered_volume_source_only=PASS' in line
  log_tail.append(line)
log=''.join(log_tail)[-6500:]"""
    assert old in launch
    launch=launch.replace(old,new).replace("'source_import_pass':'registered_volume_source_only=PASS' in log","'source_import_pass':source_import_pass")
    runner.write_text(launch)
    compile(launch,str(runner),'exec')
    p['runtime_launcher']=pin(runner)
    p['routine_observation_policy']={'full_stdout_readback':'bounded streaming tail and source marker scan','early_cpu_sample':False,'gpu_entry_vmmap':False,'disk_and_swap_guard':'unchanged'}
