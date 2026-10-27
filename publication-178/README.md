# P178: Hand-region skin/bone construction checkpoint

P178 follows P177. The unselected static construction checkpoint moves 249 skin nodes by at most 0.03930 mm from P177's 675-pair candidate. Exact Float32 checks report 673 surface pairs, zero skin self-pairs and zero sampled contained node–bone pairs across all 185 typed bone banks; only bank 43/16 changed (−2). The 192 nonpoint payload arrays, topology and inherited source ancestry are bitwise unchanged.

The full 640 axis-filtered proposal is retained as failed evidence: its exact skin test found 80 self-pairs. Its isolated 249-node component has 15 self-pairs. The five fraction trials are retained with their audits; the 0.05 trial is the 673 checkpoint. These are construction experiments, not physical simulation steps. A prior 001 line-search failed before generating trials because NaNs were compared without NaN-aware equality; the retained 002 source fixes that guard.

This candidate is not admitted to the native scene and accepted no physical steps. Sampled node containment is not a proof of closed-solid containment. Skin/fat volume validity, mass and force transfer, motion across the breathing cycle, the remaining skin/bone intersections, existing head boundary loops and integrated physiology remain unqualified. Mixed-source anatomy is a reference assembly, not one measured individual.
