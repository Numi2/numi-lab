import pathlib,runpy
p=pathlib.Path(__file__).resolve().parent
g=runpy.run_path(str(p/"prepare.py"));s=g["s"];pos=g["pos"];ins=g["instrument"]
def remove_block(s,start):
 brace=s.index("{",start);level=0
 for i in range(brace,len(s)):
  if s[i]=="{":level+=1
  elif s[i]=="}":
   level-=1
   if level==0:return s[:start]+s[i+1:]
 raise RuntimeError("unclosed block")
fnstart=s.index("inline float typedNormalCrossContactResponse(")
fnend=s.index("\n}\n",fnstart)+2
fn=s[fnstart:fnend].replace("typedNormalCrossContactResponse(","typedRigidOnlyCrossResponse(",1)
fn=remove_block(fn,fn.index("    if (targetRod && sourceRod) {"))
ins=ins.replace("typedNormalCrossContactResponse(","typedRigidOnlyCrossResponse(")
ins=remove_block(ins,ins.index("                    if (typedRodConstraint(contact) && typedRodConstraint(source)) {"))
for axis in range(3):
 out=s[:fnend]+"\n\n"+fn+s[fnend:pos]+ins.replace("AXIS_VALUE",str(axis)+"u")+s[pos:]
 (p/("direct%d.metal"%axis)).write_text(out)
