import {ServiceClient,ServiceError} from './client.js';
import type {SharedState,OwnerData} from './types.js';
export class AcknowledgedState {
 value:SharedState|null=null; pending=false; rejected:OwnerData|null=null;
 onChange:()=>void=()=>{};onError:(error:Error)=>void=()=>{};
 constructor(public client:ServiceClient){}
 async refresh(){const state=await this.client.request<SharedState>('shared');if(!this.value||state.revision>=this.value.revision){this.value=state;this.onChange()}return state}
 async mutate(intent:OwnerData){
  if(this.pending)throw new Error('An edit is awaiting acknowledgement. Keep your intent and try after it completes.');
  if(!this.value)throw new Error('Reconnect to the laboratory first.');
  this.pending=true;this.onChange();
  try {const state=await this.client.request<SharedState>('shared',{...intent,expectedRevision:this.value.revision,actor:'human'});if(state.revision<this.value.revision)throw new Error('Stale acknowledgement rejected');this.value=state;this.rejected=null;return state}
  catch(error){this.rejected=structuredClone(intent);if(error instanceof ServiceError&&error.status===409)await this.refresh();throw error}
  finally{this.pending=false;this.onChange()}
 }
}
export interface Preferences {appearance:string;navigation:boolean;drawer:string|null;activeDraft:string|null;comparison:boolean;normalization:string;role:string}
export class Layout {
 value:Preferences={appearance:'system',navigation:false,drawer:null,activeDraft:null,comparison:false,normalization:'raw_counts',role:'direct'};
 constructor(){try{this.value={...this.value,...JSON.parse(localStorage.getItem('numilab-workbench-layout/v1')||'{}')}}catch{}}
 update(p:Partial<Preferences>){Object.assign(this.value,p);localStorage.setItem('numilab-workbench-layout/v1',JSON.stringify(this.value));document.documentElement.dataset.appearance=this.value.appearance}
}
