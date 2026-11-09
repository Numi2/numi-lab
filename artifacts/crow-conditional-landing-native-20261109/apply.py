from pathlib import Path
import re
oldid='american_crow_v15_triple_waypoint_expert_navigation_development_v1'
newid='american_crow_v15_conditional_landing_navigation_development_v1'
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def edit(path,fn):
 p=Path(path);a=p.read_text();b=fn(a);assert a!=b,path;p.write_text(b)
def header(s):
 return once(s,'    MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION = 12,','    MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION = 12,\n    // Separately fingerprinted learned late-landing recovery head.\n    MR_BIRDFLOW_JOURNEY_V15_CONDITIONAL_LANDING_NAVIGATION = 13,')
def binding(s):
 s=once(s,'    case v15TripleWaypointExpertNavigation = 12','    case v15TripleWaypointExpertNavigation = 12\n    case v15ConditionalLandingNavigation = 13')
 s=once(s,'        case .v15TripleWaypointExpertNavigation: "v15-triple-waypoint-expert-navigation"','        case .v15TripleWaypointExpertNavigation: "v15-triple-waypoint-expert-navigation"\n        case .v15ConditionalLandingNavigation: "v15-conditional-landing-navigation"')
 return once(s,'        case .v15TripleWaypointExpertNavigation: "'+oldid+'"','        case .v15TripleWaypointExpertNavigation: "'+oldid+'"\n        case .v15ConditionalLandingNavigation: "'+newid+'"')
def app(s):
 s=once(s,'                case "v15-triple-waypoint-expert-navigation":\n                    birdFlowJourneyVariant = .v15TripleWaypointExpertNavigation','                case "v15-triple-waypoint-expert-navigation":\n                    birdFlowJourneyVariant = .v15TripleWaypointExpertNavigation\n                case "v15-conditional-landing-navigation":\n                    birdFlowJourneyVariant = .v15ConditionalLandingNavigation')
 s=s.replace('or v15-triple-waypoint-expert-navigation.','v15-triple-waypoint-expert-navigation, or v15-conditional-landing-navigation.')
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant != \.v15TripleWaypointExpertNavigation\)\)$',r'\1\2birdFlowJourneyVariant != .v15TripleWaypointExpertNavigation &&\n\1\2birdFlowJourneyVariant != .v15ConditionalLandingNavigation))',s)
 s=re.sub(r'(?m)^([ \t]*)(options\.)?birdFlowJourneyVariant == \.v15TripleWaypointExpertNavigation \|\|$',r'\1\2birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation ||\n\1\2birdFlowJourneyVariant == .v15ConditionalLandingNavigation ||',s)
 s=s.replace('''                        : options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation
                        ? "v15_triple_waypoint_expert_goal_memory_rgbd_history"''','''                        : options.birdFlowJourneyVariant == .v15ConditionalLandingNavigation
                        ? "v15_conditional_landing_goal_memory_rgbd_history"
                        : options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation
                        ? "v15_triple_waypoint_expert_goal_memory_rgbd_history"''')
 s=s.replace('''       options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation {''','''       options.birdFlowJourneyVariant == .v15TripleWaypointExpertNavigation ||
       options.birdFlowJourneyVariant == .v15ConditionalLandingNavigation {''')
 return s
def capi(s):
 s=once(s,'MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION) {','MR_BIRDFLOW_JOURNEY_V15_CONDITIONAL_LANDING_NAVIGATION) {')
 s=once(s,'''        const bool tripleWaypointExpertNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION;''','''        const bool conditionalLandingNavigation = journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_CONDITIONAL_LANDING_NAVIGATION;
        const bool tripleWaypointExpertNavigation = conditionalLandingNavigation || (journey &&
            config->birdflow_journey_variant == MR_BIRDFLOW_JOURNEY_V15_TRIPLE_WAYPOINT_EXPERT_NAVIGATION);''')
 return once(s,'''            manifest.task.id = tripleWaypointExpertNavigation
                ? "'''+oldid+'''"''','''            manifest.task.id = conditionalLandingNavigation
                ? "'''+newid+'''"
                : tripleWaypointExpertNavigation
                ? "'''+oldid+'''"''')
def task(s):
 needle='pack.id == "'+oldid+'"';assert s.count(needle)==9,s.count(needle)
 s=s.replace(needle,'(pack.id == "'+oldid+'" || pack.id == "'+newid+'")')
 needle='''    staged->header.interactionCurriculum = {
        interactionClip == nullptr ? 0.0f : resetProbability,
        interactionClip == nullptr ? 0.0f : maximumResetPhase,
        0.0f,
        0.0f,
    };'''
 replacement='''    staged->header.interactionCurriculum = {
        interactionClip == nullptr ? 0.0f : resetProbability,
        interactionClip == nullptr ? 0.0f : maximumResetPhase,
        pack.id == "'''+newid+'''" ? 2600.0f : 0.0f,
        0.0f,
    };'''
 return once(s,needle,replacement)
def types(s):
 return once(s,'    // Reset probability, maximum normalized reset phase, reserved, reserved.','    // Reset probability, maximum normalized reset phase, late landing recovery\n    // start step for the explicitly fingerprinted crow task (zero disables), reserved.')
def shader(s):
 return once(s,'''        const uint commandOffset =
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION)''','''        // The separately fingerprinted task may invoke a learned recovery
        // head after prolonged unsupported landing. The frozen flight law is unchanged.
        const uint commandOffset =
            fixedFlightController && program.interactionCurriculum.z > 0.0f &&
            state.navigation.z >= 5.0f &&
            state.episode.x >= uint(program.interactionCurriculum.z) &&
            state.threatTeacher.w < 1.5f ? 4u :
            (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_WAYPOINT_EXPERT_NAVIGATION)''')
for p,f in [('include/metalrobo/c_api.h',header),('include/metalrobo/task_program_types.h',types),('bindings/swift/MetalRoboTaskRollout.swift',binding),('apps/task_rollout.swift',app),('apps/task_train.swift',app),('src/c_api.cpp',capi),('src/core/TaskProgram.cpp',task),('src/metal/LocomotionTask.metal',shader)]:edit(p,f)
