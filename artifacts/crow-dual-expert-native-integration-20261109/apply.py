from pathlib import Path
import re
oldid='american_crow_v15_waypoint_expert_navigation_development_v1'
newid='american_crow_v15_dual_waypoint_expert_navigation_development_v1'
def change(path,fn):
 p=Path(path);a=p.read_text();b=fn(a);assert a!=b,path;p.write_text(b)
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def header(s):return once(s,'    MR_BIRDFLOW_JOURNEY_V15_WAYPOINT_EXPERT_NAVIGATION = 10,','    MR_BIRDFLOW_JOURNEY_V15_WAYPOINT_EXPERT_NAVIGATION = 10,\n    // Distinct deterministic stage-two and stage-three command heads.\n    MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION = 11,')
def flags(s):return once(s,'    MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION = 1u << 29u,','    MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION = 1u << 29u,\n    // Route outputs 8..11 are active only at waypoint index three.\n    MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION = 1u << 30u,')
def binding(s):
 s=once(s,'    case v15WaypointExpertNavigation = 10','    case v15WaypointExpertNavigation = 10\n    case v15DualWaypointExpertNavigation = 11')
 s=once(s,'        case .v15WaypointExpertNavigation: "v15-waypoint-expert-navigation"','        case .v15WaypointExpertNavigation: "v15-waypoint-expert-navigation"\n        case .v15DualWaypointExpertNavigation: "v15-dual-waypoint-expert-navigation"')
 s=once(s,'        case .v15WaypointExpertNavigation: "'+oldid+'"','        case .v15WaypointExpertNavigation: "'+oldid+'"\n        case .v15DualWaypointExpertNavigation: "'+newid+'"')
 return s
def app(s):
 s=once(s,'''                case "v15-waypoint-expert-navigation":
                    birdFlowJourneyVariant = .v15WaypointExpertNavigation''','''                case "v15-waypoint-expert-navigation":
                    birdFlowJourneyVariant = .v15WaypointExpertNavigation
                case "v15-dual-waypoint-expert-navigation":
                    birdFlowJourneyVariant = .v15DualWaypointExpertNavigation''')
 s=s.replace('or v15-waypoint-expert-navigation.','v15-waypoint-expert-navigation, or v15-dual-waypoint-expert-navigation.')
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant != \.v15WaypointExpertNavigation\)\)$',r'\1\2birdFlowJourneyVariant != .v15WaypointExpertNavigation &&\n\1\2birdFlowJourneyVariant != .v15DualWaypointExpertNavigation))',s)
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant == \.v15WaypointExpertNavigation \|\|$',r'\1\2birdFlowJourneyVariant == .v15WaypointExpertNavigation ||\n\1\2birdFlowJourneyVariant == .v15DualWaypointExpertNavigation ||',s)
 s=s.replace('''                        : options.birdFlowJourneyVariant == .v15WaypointExpertNavigation
                        ? "v15_waypoint_expert_goal_memory_rgbd_history"''','''                        : options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation
                        ? "v15_dual_waypoint_expert_goal_memory_rgbd_history"
                        : options.birdFlowJourneyVariant == .v15WaypointExpertNavigation
                        ? "v15_waypoint_expert_goal_memory_rgbd_history"''')
 s=s.replace('if options.birdFlowJourneyVariant == .v15WaypointExpertNavigation {','if options.birdFlowJourneyVariant == .v15WaypointExpertNavigation ||\n       options.birdFlowJourneyVariant == .v15DualWaypointExpertNavigation {')
 return s
def capi(s):
 s=once(s,'MR_BIRDFLOW_JOURNEY_V15_WAYPOINT_EXPERT_NAVIGATION) {','MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION) {')
 s=once(s,'''        const bool waypointExpertNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_WAYPOINT_EXPERT_NAVIGATION;''','''        const bool dualWaypointExpertNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_DUAL_WAYPOINT_EXPERT_NAVIGATION;
        const bool waypointExpertNavigation = dualWaypointExpertNavigation || (journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_WAYPOINT_EXPERT_NAVIGATION);''')
 s=once(s,'''            manifest.task.id = waypointExpertNavigation
                ? "'''+oldid+'''"
                : "american_crow_v15_goal_memory_navigation_development_v1";''','''            manifest.task.id = dualWaypointExpertNavigation
                ? "'''+newid+'''"
                : waypointExpertNavigation
                ? "'''+oldid+'''"
                : "american_crow_v15_goal_memory_navigation_development_v1";''')
 return s
def task(s):
 needle='pack.id == "'+oldid+'"'
 assert s.count(needle)==7,s.count(needle)
 s=s.replace(needle,'(pack.id == "'+oldid+'" || pack.id == "'+newid+'")')
 needle='''    if (threatGroup != MR_INVALID_INDEX) {'''
 assert s.count(needle)==1
 return s.replace(needle,'''    if (pack.id == "'''+newid+'''") {
        staged->header.schedule.w |= MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION;
    }
'''+needle)
def shader(s):
 s=once(s,'''            state.navigation.z >= 2.0f && state.navigation.z < 3.0f ? 4u : 0u;''','''            state.navigation.z >= 2.0f && state.navigation.z < 3.0f ? 4u :
            ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_DUAL_WAYPOINT_EXPERT_NAVIGATION) != 0u &&
             state.navigation.z >= 3.0f && state.navigation.z < 4.0f ? 8u : 0u);''')
 s=once(s,'''            const bool expertLabel = commandOffset == 4u && action >= 4u && action < 8u;
            teacherActions[actionBase + action] = action < 4u
                ? label[action] : expertLabel ? label[action - 4u] : 0.0f;''','''            const bool expertLabel = commandOffset != 0u &&
                action >= commandOffset && action < commandOffset + 4u;
            teacherActions[actionBase + action] = action < 4u
                ? label[action] : expertLabel ? label[action - commandOffset] : 0.0f;''')
 return s
for path,fn in [('include/metalrobo/c_api.h',header),('include/metalrobo/task_program_types.h',flags),('bindings/swift/MetalRoboTaskRollout.swift',binding),('apps/task_rollout.swift',app),('apps/task_train.swift',app),('src/c_api.cpp',capi),('src/core/TaskProgram.cpp',task),('src/metal/LocomotionTask.metal',shader)]:change(path,fn)
