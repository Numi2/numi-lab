import {node,fmt,badge} from '../dom.js';
import type {Readout,OwnerData} from '../types.js';
export function populationView(host:HTMLElement,readout:Readout,target:string|undefined,sharedMax?:number){
 host.replaceChildren();host.append(badge(String(typeof readout.access==='string'?readout.access:readout.access?.status||'OWNER-CONTROLLED ACCESS')),node('p',readout.units,'muted'));
 const arms=(readout.arms||[]).filter(a=>!target||a.target===target);
 const values=arms.flatMap(a=>[a.control,a.predicted,a.observed]).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v));const max=sharedMax??Math.max(1e-9,...values.map(Math.abs));
 for(const a of arms){const box=node('section');box.append(node('h2',a.target+' · '+(a.population||'')));
  for(const [label,key] of [['Control reference','control'],['Predicted mean','predicted'],['Observed mean','observed'],['Residual: pred − obs','residual']]){
   if((key==='predicted'||key==='residual')&&a.predicted==null&&a.evidence==='MEASURED')continue;
   const row=node('div',undefined,'bar-row'),track=node('div',undefined,'bar-track'),bar=node('div',undefined,'bar '+key);
   const value=a[key];bar.style.width=typeof value==='number'?Math.min(100,Math.abs(value)/(key==='residual'?Math.max(max,Math.abs(value)):max)*100)+'%':'0';if(key==='residual'){track.classList.add('signed');const width=typeof value==='number'?Math.min(50,Math.abs(value)/Math.max(max,Math.abs(value))*50):0;bar.style.width=width+'%';bar.style.marginLeft=(value>=0?50:50-width)+'%';bar.style.background=value<0?'#287f9c':'#b56249'}track.append(bar);row.append(node('span',label),track,node('strong',fmt(value)));box.append(row);
  }
  if(a.matchedBaseline!==undefined)box.append(node('p','Matched baseline '+fmt(a.matchedBaseline)+' · matched residual '+fmt(a.matchedResidual)+' · no-change residual '+fmt(a.noChangeResidual),'muted'));
  box.append(node('p',`${a.controlCells??'Unknown'} distinct controls · ${a.outcomeCells??'Unknown'} outcome cells · ${a.accessibleOutcomeCells??(a.observed!=null?'Population mean opened; cells unavailable':'Not opened')} accessible · ${a.reservedOutcomeCells??'Owner controlled'} reserved`,'muted'));
  const distribution=a.controlValues||a.referenceValues;const observed=a.observedValues;
  if(Array.isArray(distribution)&&distribution.length)box.append(distributionPlot(distribution,Array.isArray(observed)?observed:[],max,readout.units));
  box.append(node('p',a.missingReason||a.reason||readout.distributionStatus||'Uncertainty about transfer is unqualified. Cell counts are not independent biological units.','muted'));
  host.append(box);
 }
 if(!arms.length)host.append(node('div','Select a supported intervention to request its bounded molecular readout.','empty'));
 host.append(node('p','Fixed measured endpoint. Population predictions do not establish individual cell trajectories.','muted'));
}
export function distributionPlot(control:number[],observed:number[],max:number,units:string){
 const figure=node('figure');figure.style.margin='20px 0';const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 520 120');svg.classList.add('plot');svg.setAttribute('role','img');svg.setAttribute('aria-label','Bounded measured cell values: controls above, accessible outcomes below. '+units);
 for(const [values,y,color] of [[control,32,'#16866f'],[observed,72,'#7962ba']] as [number[],number,string][]){values.forEach((value,i)=>{const c=document.createElementNS(svg.namespaceURI,'circle');c.setAttribute('cx',String(25+Math.min(1,value/Math.max(max,1e-9))*470));c.setAttribute('cy',String(y+(i%5-2)*3));c.setAttribute('r','2.5');c.setAttribute('fill',color);const t=document.createElementNS(svg.namespaceURI,'title');t.textContent=fmt(value);c.append(t);svg.append(c)})}
 figure.append(svg,node('figcaption',`${control.length} requested control values · ${observed.length} accessible outcome values. This bounded display is not a replicate-level accuracy limit.`,'muted'));return figure;
}
export function supportView(data:OwnerData|undefined){const host=node('div',undefined,'support-grid');if(!data)return host;
 const counts=data.reference||{};const controls=counts.referenceCells??counts.controlCells??data.controlCells;
 for(const [label,value] of [['Selected controls',controls],['Biological units',data.biologicalUnits==null?'Unresolved':data.biologicalUnits],['Objective features',(data.genes||[]).length],['Recommendation','Unqualified']]){const d=node('div');d.append(node('b',Array.isArray(value)?value.join(', '):String(value??'Unknown')),node('small',String(label)));host.append(d)}return host}
