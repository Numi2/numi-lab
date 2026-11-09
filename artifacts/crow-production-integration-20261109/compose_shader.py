from pathlib import Path
import subprocess
R=Path.cwd()
base=subprocess.check_output(['git','show','aef51b91:src/metal/LocomotionTask.metal'],text=True)
s=(R/'build/crow-hierarchical-goal-memory-20261109/actor.metal').read_text()
# Preserve the previous journey teacher implementation byte-for-byte.
a='inline float birdFlowJourneyTeacherAction(';b='inline uint taskActionHoldPeriod('
x=base.index(a);y=base.index(b,x);i=s.index(a);j=s.index(b,i)
s=s[:i]+base[x:y]+'#include "CrowFlightControllerV1.metal"\n\n'+s[j:]
s=s.replace('#define HIERARCHY_ORACLE false\n','').replace('#define HIERARCHY_RECOVER false\n','')
old='''        const bool fixedFlightController =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u;'''
new='''        const bool fixedFlightController =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('HIERARCHY_ORACLE || (dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f)','(dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f)')
old='''        float navigationTeacherRequested = avianNavigationTeacher && action >= 15u
            ? studentRequested : teacherRequested;'''
orig_start=base.index('        const float navigationTeacherRequested = !avianNavigationTeacher')
orig_end=base.index('\n        // Keep the training-only teacher',orig_start)
legacy=base[orig_start:orig_end].replace('const float navigationTeacherRequested =','const float legacyNavigationTeacherRequested =',1)
new=legacy+'''
        float navigationTeacherRequested = fixedFlightController
            ? (avianNavigationTeacher && action >= 15u ? studentRequested : teacherRequested)
            : legacyNavigationTeacherRequested;'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        const float navigationTeacherBlend = avianNavigationTeacher
            ? 1.0f
            : 1.0f;'''
new='''        const float navigationTeacherBlend = fixedFlightController
            ? 1.0f
            : avianNavigationTeacher
            ? (state.navigation.z >= 5.0f ? 0.0f : 1.0f)
            : 1.0f;'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        if (avianJourneyTeacher) {
            const float4 label = float4('''
new='''        if (avianJourneyTeacher && fixedFlightController) {
            const float4 label = float4('''
assert s.count(old)==1;s=s.replace(old,new)
old='''    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_FLIGHT_PHASE) != 0u &&
        (dispatch.sampling.w != 0u || HIERARCHY_ORACLE)) {'''
new='''    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        dispatch.sampling.w != 0u) {'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('(1u << 28u)','MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION')
old='''        const float4 actorLatent = clamp(float4(
            actionStream[actionBase+0u],actionStream[actionBase+1u],
            actionStream[actionBase+2u],actionStream[actionBase+3u]),-1.0f,1.0f);'''
new='''        const float4 actorLatent = fixedFlightController
            ? clamp(float4(actionStream[actionBase+0u],actionStream[actionBase+1u],
                           actionStream[actionBase+2u],actionStream[actionBase+3u]),-1.0f,1.0f)
            : float4(0.0f);'''
assert s.count(old)==1;s=s.replace(old,new)
assert 'HIERARCHY_' not in s
(R/'src/metal/LocomotionTask.metal').write_text(s)
