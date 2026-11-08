from pathlib import Path
for variant in ('main','four'):
 p=Path(variant+'.before.metal');s=p.read_text();a='''                maximumNormalResidual,
                maximumConeViolation,
                0.0f
            );
        }
        waveStatuses[islandBase + islandIndex] = result;''';b='''                maximumNormalResidual,
                maximumConeViolation,
                ((work.flags & MR_ISLAND_WORK_HAS_ROD)==0u && maximumNormalResidual>0.002f) ? 1.0f : 0.0f
            );
        }
        waveStatuses[islandBase + islandIndex] = result;''';assert s.count(a)==1,(variant,s.count(a));Path(variant+'.candidate.metal').write_text(s.replace(a,b))
