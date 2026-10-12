import Foundation
import Metal
let device=MTLCreateSystemDefaultDevice()!
let source=try String(contentsOfFile:CommandLine.arguments[1],encoding:.utf8)
let options=MTLCompileOptions()
options.mathMode = .safe
options.languageVersion = .version4_0
let start=ProcessInfo.processInfo.systemUptime
let lib=try device.makeLibrary(source:source,options:options)
guard lib.functionNames.contains("nm_human_resting_skin") else {fatalError("skin kernel absent")}
let r:[String:Any]=["device":device.name,"compiled":true,"function_count":lib.functionNames.count,"wall_seconds":ProcessInfo.processInfo.systemUptime-start,"math_mode":"safe","metal_language":"4.0","dispatch":false]
print(String(data:try JSONSerialization.data(withJSONObject:r,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
