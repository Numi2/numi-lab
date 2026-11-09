from pathlib import Path
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def edit(path,fn):
 p=Path(path);a=p.read_text();b=fn(a);assert a!=b;p.write_text(b)
def head(s):
 s=once(s,'    // Untouched layouts reserved for the next assessment.\n    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_G = 8,','    // Previously assessed layouts, now development only.\n    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_G = 8,')
 return once(s,'    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_H = 9,','    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_H = 9,\n    // Untouched layouts reserved for a new assessment.\n    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_I = 10,\n    MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_J = 11,')
def bind(s):
 return once(s,'    case reservedFinalH = 9','    case reservedFinalH = 9\n    case reservedFinalI = 10\n    case reservedFinalJ = 11')
def capi(s):
 s=once(s,'constexpr std::array<std::array<mr_float4, 5u>, 9u> positions{{','constexpr std::array<std::array<mr_float4, 5u>, 11u> positions{{')
 s=once(s,'        // Untouched assessment layouts; never use for fitting or selection.','        // Former assessment layouts, now development only.')
 needle='''          {5.08f, -0.17f, 0.78f, 1.0f}}},
    }};'''
 replacement='''          {5.08f, -0.17f, 0.78f, 1.0f}}},
        // Untouched assessment layouts; never use for fitting or selection.
        {{{1.42f, 0.66f, 0.80f, 1.0f},
          {1.42f, -0.72f, 0.80f, 1.0f},
          {2.70f, -0.38f, 0.65f, 1.0f},
          {3.86f, 0.35f, 0.65f, 1.0f},
          {5.12f, 0.10f, 0.81f, 1.0f}}},
        {{{1.35f, 0.73f, 0.80f, 1.0f},
          {1.35f, -0.65f, 0.80f, 1.0f},
          {2.61f, 0.36f, 0.65f, 1.0f},
          {3.88f, -0.40f, 0.65f, 1.0f},
          {5.00f, -0.06f, 0.79f, 1.0f}}},
    }};'''
 s=once(s,needle,replacement)
 return once(s,'MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_H) {','MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_J) {')
def app(s):
 s=once(s,'''                case "reserved-final-h":
                    birdFlowNavigationCourse = .reservedFinalH''','''                case "reserved-final-h":
                    birdFlowNavigationCourse = .reservedFinalH
                case "reserved-final-i":
                    birdFlowNavigationCourse = .reservedFinalI
                case "reserved-final-j":
                    birdFlowNavigationCourse = .reservedFinalJ''')
 s=s.replace('reserved-final-g, or reserved-final-h.','reserved-final-g, reserved-final-h, reserved-final-i, or reserved-final-j.')
 import re
 s=re.sub(r'(?m)^(\s*)options.birdFlowNavigationCourse == \.reservedFinalG\n\1\? "reserved-final-g"',r'\1options.birdFlowNavigationCourse == .reservedFinalI\n\1? "reserved-final-i"\n\1: options.birdFlowNavigationCourse == .reservedFinalJ\n\1? "reserved-final-j"\n\1: options.birdFlowNavigationCourse == .reservedFinalG\n\1? "reserved-final-g"',s)
 assert s.count('options.birdFlowNavigationCourse == .reservedFinalI')==3
 s=s.replace('options.birdFlowNavigationCourse == .reservedFinalG ||\n                               options.birdFlowNavigationCourse == .reservedFinalH','options.birdFlowNavigationCourse == .reservedFinalI ||\n                               options.birdFlowNavigationCourse == .reservedFinalJ')
 s=s.replace('options.birdFlowNavigationCourse == .reservedFinalG ||\n                       options.birdFlowNavigationCourse == .reservedFinalH','options.birdFlowNavigationCourse == .reservedFinalI ||\n                       options.birdFlowNavigationCourse == .reservedFinalJ')
 return s
for p,f in [('include/metalrobo/c_api.h',head),('bindings/swift/MetalRoboTaskRollout.swift',bind),('src/c_api.cpp',capi),('apps/task_rollout.swift',app)]:edit(p,f)
