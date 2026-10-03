(async()=>{
 const root=document.querySelector('#learned-root'),wait=async(fn)=>{const end=performance.now()+30000;while(!fn()){if(performance.now()>end)throw Error('Workflow timeout: '+document.querySelector('#learned-status').textContent);await new Promise(r=>setTimeout(r,50));}},click=async(id,fn)=>{document.querySelector(id).click();await wait(fn);};
 await wait(()=>learnedLab.state.geometry?.specimen==='chip1'&&learnedLab.state.field&&!root.classList.contains('working'));
 document.querySelector('#learned-population').value='neuron|0';document.querySelector('#learned-population').dispatchEvent(new Event('change',{bubbles:true}));await wait(()=>!root.classList.contains('working'));
 await click('#learned-seal',()=>learnedLab.state.run&&!root.classList.contains('working'));
 const before=learnedLab.state;const id=before.run.id;if(before.run.revealed||before.field.arms.some(a=>a.observed!==null))throw Error('Held-out observation exposed before reveal');if(new Set(before.field.arms.map(a=>a.control)).size!==1)throw Error('Comparison control references differ');
 await click('#learned-reveal',()=>learnedLab.state.run?.revealed&&!root.classList.contains('working'));
 const after=learnedLab.state;if(!after.field.arms.some(a=>a.observed!==null))throw Error('No measured observations');if(!after.field.arms.every(a=>a.residual===null||Math.abs(a.residual-(a.predicted-a.observed))<1e-5))throw Error('Incorrect residual sign');
 document.querySelector('#learned-role').value='neighbor';document.querySelector('#learned-role').dispatchEvent(new Event('change',{bubbles:true}));if(learnedLab.state.role!=='neighbor')throw Error('Neighbor response selection failed');
 document.querySelector('#learned-failure-mode').open=true;
 await click('#learned-replay',()=>document.querySelector('#learned-status').textContent.includes('bit-exactly')&&!root.classList.contains('working'));
 return {format:'numilab-spatial-browser-workflow/v1',run:id,selectedInterventions:after.run.registration.plan.targets,arms:after.run.arms.length,observationsHiddenBeforeReveal:true,commonControlReference:true,signedResidualsCorrect:true,directAndNeighborSelectable:true,computedFailureDiagnostics:after.run.comparison.arms.every(a=>a.diagnostics.length>0),nativeReplay:true,biologicalPromotion:after.run.comparison.biologicalPromotion,viewport:[innerWidth,innerHeight],sourceCells:after.geometry.cells.length,status:'passed'};
})()
