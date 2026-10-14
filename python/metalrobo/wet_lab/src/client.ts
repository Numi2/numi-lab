export class ServiceError extends Error {constructor(message:string,public status:number,public details:unknown){super(message)}}
export interface Timing {path:string;ms:number;bytes:number;cached:boolean;channel?:string}
export class ServiceClient {
 private pending=new Map<string,AbortController>(); private cache=new Map<string,unknown>();
 readonly timings:Timing[]=[];
 constructor(private token:string){}
 cancel(channel:string){this.pending.get(channel)?.abort();this.pending.delete(channel)}
 clear(){this.cache.clear();for(const controller of this.pending.values())controller.abort();this.pending.clear()}
 async request<T>(path:string,body?:unknown,options:{channel?:string;cacheKey?:string}={}):Promise<T>{
  if(options.channel)this.cancel(options.channel);
  if(options.cacheKey&&this.cache.has(options.cacheKey)){const value=this.cache.get(options.cacheKey) as T;this.timings.push({path,ms:0,bytes:0,cached:true,channel:options.channel});if(this.timings.length>300)this.timings.shift();return structuredClone(value)}
  const controller=new AbortController();if(options.channel){this.pending.set(options.channel,controller)}
  const start=performance.now();try {
   const response=await fetch('/api/'+path,{method:body===undefined?'GET':'POST',headers:{'X-Wet-Lab-Token':this.token,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal});
   const raw=await response.text();const result=JSON.parse(raw);this.timings.push({path,ms:performance.now()-start,bytes:new TextEncoder().encode(raw).length,cached:false,channel:options.channel});if(this.timings.length>300)this.timings.shift();
   if(!response.ok)throw new ServiceError(result.error||'Request failed',response.status,result);
   if(controller.signal.aborted)throw new DOMException('Obsolete response','AbortError');
   if(options.cacheKey){if(this.cache.size>=32)this.cache.delete(this.cache.keys().next().value!);this.cache.set(options.cacheKey,result)}
   return result as T;
  } finally {if(options.channel&&this.pending.get(options.channel)===controller)this.pending.delete(options.channel)}
 }
}
// Identity includes the full owner binding and access state. No preview survives a scientific-input edit.
export function identity(...parts:unknown[]):string{return JSON.stringify(parts)}
