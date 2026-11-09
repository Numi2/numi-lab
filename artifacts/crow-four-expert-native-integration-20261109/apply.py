from pathlib import Path
import re
oldid='american_crow_v15_dual_waypoint_expert_navigation_development_v1'
newid='american_crow_v15_triple_waypoint_expert_navigation_development_v1'
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def edit(path,fn):
 p=Path(path);a=p.read_text();b=fn(a);assert a!=b,path;p.write_text(b)
def header(s):
 return once(s,'    MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION = 11,','    MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION = 11,\n    // Separate deterministic experts at waypoint indices two through four.\n    MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION = 12,')
def flags(s):
 return once(s,'    MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION = 1u << 30u,','    MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION = 1u << 30u,\n    // Stage four uses actor slots 12..15; physical head slot 15 is zero before landing.\n    MR_TASK_PROGRAM_AVIAN_CROW_TRIPLE_WAYPOINT_EXPERT_NAVIGATION = 1u << 31u,')
def binding(s):
 s=once(s,'    case v15DualWaypointExpertNavigation = 11','    case v15DualWaypointExpertNavigation = 11\n    case v15TripleWaypointExpertNavigation = 12')
 s=once(s,'        case .v15DualWaypointExpertNavigation: "v15-dual-waypoint-expert-navigation"','        case .v15DualWaypointExpertNavigation: "v15-dual-waypoint-expert-navigation"\n        case .v15TripleWaypointExpertNavigation: "v15-triple-waypoint-expert-navigation"')
 return once(s,'        case .v15DualWaypointExpertNavigation: "'+oldid+'"','        case .v15DualWaypointExpertNavigation: "'+oldid+'"\n        case .v15TripleWaypointExpertNavigation: "'+newid+'"')
def app(s):
 s=once(s,'                case "v15-dual-waypoint-expert-navigation":\n                    birdFlowJourneyVariant = .v15DualWaypointExpertNavigation','                case "v15-dual-waypoint-expert-navigation":\n                    birdFlowJourneyVariant = .v15DualWaypointExpertNavigation\n                case "v15-triple-waypoint-expert-navigation":\n                    birdFlowJourneyVariant = .v15TripleWaypointExpertNavigation')
 s=s.replace('or v15-dual-waypoint-expert-navigation.','v15-dual-waypoint-expert-navigation, or v15-triple-waypoint-expert-navigation.')
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant != \.v15DualWaypointExpertNavigation\)\)$',r'\1\2birdFlowJourneyVariant != .v15DualWaypointExpertNavigation &&\n\1\2birdFlowJourneyVariant != .v15TripleWaypointExpertNavigation))',s)
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant == \.v15DualWaypointExpertNavigation \|\|$',r'\1\2birdFlowJourneyVariant == .v15DualWaypointExpertNavigation ||\n\1\2birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation ||',s)
 s=s.replace('''                        : options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation
                        ? "v15_dual_waypoint_expert_goal_memory_rgbd_history"''','''                        : options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation
                        ? "v15_triple_waypoint_expert_goal_memory_rgbd_history"
                        : options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation
                        ? "v15_dual_waypoint_expert_goal_memory_rgbd_history"''')
 s=s.replace('''       options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation {''','''       options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation ||
       options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation {''')
 return s
def capi(s):
 s=once(s,'MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION) {','MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION) {')
 s=once(s,'''        const bool dualWaypointExpertNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION;''','''        const bool tripleWaypointExpertNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION;
        const bool dualWaypointExpertNavigation = tripleWaypointExpertNavigation || (journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION);''')
 return once(s,'''            manifest.task.id = dualWaypointExpertNavigation
                ? "'''+oldid+'''"''','''            manifest.task.id = tripleWaypointExpertNavigation
                ? "'''+newid+'''"
                : dualWaypointExpertNavigation
                ? "'''+oldid+'''"''')
def task(s):
 needle='pack.id == "'+oldid+'"'
 assert s.count(needle)==8,s.count(needle)
 s=s.replace(needle,'(pack.id == "'+oldid+'" || pack.id == "'+newid+'")')
 needle='''    if (threatGroup != MR_INVALID_INDEX) {'''
 return once(s,needle,'''    if (pack.id == "'''+newid+'''") {
        staged->header.schedule.w |= MR_TASK_PROGRAM_AVIAN_CROW_TRIPLE_WAYPOINT_EXPERT_NAVIGATION;
    }
'''+needle)
def shader(s):
 s=once(s,'''        const float studentRequested = actionStream[actionBase + action];''','''        // Slot 15 carries stage-four command in the triple-expert contract.
        // The physical head action is held at zero until the landing actor takes over.
        const float studentRequested =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_TRIPLE_WAYPOINT_EXPERT_NAVIGATION) != 0u &&
            fixedFlightController && state.navigation.z < 5.0f && action == 15u
            ? 0.0f : actionStream[actionBase + action];''')
 return once(s,'''             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u : 0u);''','''             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u :
             ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_TRIPLE_WAYPOINT_EXPERT_NAVIGATION) != 0u &&
              state.navigation.z >= 4.0f && state.navigation.z < 5.0f ? 12u : 0u));''')
for p,f in [('include/metalrobo/c_api.h',header),('include/metalrobo/task_program_types.h',flags),('bindings/swift/MetalRoboTaskRollout.swift',binding),('apps/task_rollout.swift',app),('apps/task_train.swift',app),('src/c_api.cpp',capi),('src/core/TaskProgram.cpp',task),('src/metal/LocomotionTask.metal',shader)]:edit(p,f)
