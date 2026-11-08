from pathlib import Path
root=Path(__file__).resolve().parent
earlier=root.parent/"tracker-observation-output-rollback-20261108"/"prepare_world.py"
source=root.parent/"tracker-masked-policy-20261108"/"probe.cpp"
s=earlier.read_text().replace("world-original.cpp",str(source)).replace("world_probe.cpp",str(root/"probe.cpp"))
exec(compile(s,str(earlier),"exec"),{})
