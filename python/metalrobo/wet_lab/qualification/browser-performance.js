(async()=>{
 const waitFor=async(fn)=>{const end=performance.now()+20000;while(!fn()){if(performance.now()>end)throw Error('Specimen load timeout');await new Promise(r=>setTimeout(r,40));}};
 await waitFor(()=>learnedLab.state.geometry?.cells.length===123921&&learnedLab.state.field&&!document.querySelector('#learned-root').classList.contains('working'));
 const genes=['Gfap','Clu','Mbp','C1qa'],input=document.querySelector('#learned-gene');
 for(const gene of genes){input.value=gene;await learnedLab.loadField();}
 const switches=[];for(let i=0;i<40;i++){input.value=genes[i%4];const t=performance.now();await learnedLab.loadField();switches.push(performance.now()-t);}
 const picks=[];for(let i=0;i<500;i++){const cell=learnedLab.state.geometry.cells[(i*197)%123921],t=performance.now(),p=learnedLab.pickAt(cell[1],cell[2],30);picks.push(performance.now()-t);if(!p||Math.hypot(p.xy[0]-cell[1],p.xy[1]-cell[2])>.0001)throw Error('Indexed pick not nearest to actual cell');}
 const frames=[],draws=[];let previous;for(let i=0;i<90;i++){await new Promise(r=>requestAnimationFrame(t=>{if(previous!==undefined)frames.push(t-previous);previous=t;learnedLab.state.camera.x=Math.sin(i/12)*50;const start=performance.now();learnedLab.draw();draws.push(performance.now()-start);r();}));}
 function percentile(a,p=.95){return a.slice().sort((a,b)=>a-b)[Math.min(a.length-1,Math.ceil(a.length*p)-1)];}
 const result={format:'numilab-spatial-browser-performance/v1',sourceCells:123921,sourceSHA256:learnedLab.state.geometry.sourceSHA256,viewport:[innerWidth,innerHeight],browser:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,canvasCount:4,fullGeometryIndexed:true,lod:'2048-square full-specimen raster, all source cells available to spatial picking',samples:{feature:switches.length,pick:picks.length,pan:frames.length},p95:{warmFeatureSwitchMs:percentile(switches),spatialPickMs:percentile(picks),panFrameMs:percentile(frames),drawScriptMs:percentile(draws)},targets:{warmFeatureSwitchMs:100,spatialPickMs:20,panFrameMs:33.4},biologicalValidation:false};result.passed=Object.keys(result.targets).every(k=>result.p95[k]<=result.targets[k]);return result;
})()
