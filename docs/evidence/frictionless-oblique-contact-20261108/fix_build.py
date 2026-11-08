from pathlib import Path
p=Path('/Users/numi/ccd-frictionless-oblique-20261108/build.py')
s=p.read_text().replace('import json,shlex,subprocess,hashlib','import json,shlex,subprocess,hashlib,re')
a=s.index('for indent in (12,8):');b=s.index("source=p/'candidate.metal'",a)
s=s[:a]+'''pattern=r"(?m)^( +)float inverse\\[3\\]\\[3\\];\\n\\1if \\(!invert3x3\\(effective, inverse\\)\\) \\{"
matches=list(re.finditer(pattern,s));assert len(matches)==4,len(matches)
for match in reversed(matches[:3]):
 indent=match.group(1)
 s=s[:match.start()]+indent+'frictionlessNormalBlock(effective, evaluatedCones[constraintBase + localConstraint]);\\n'+s[match.start():]
'''+s[b:]
p.write_text(s)
