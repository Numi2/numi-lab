export function node<K extends keyof HTMLElementTagNameMap>(tag:K,text?:string,cls?:string):HTMLElementTagNameMap[K]{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el}
export function button(text:string,action:()=>void,cls?:string){const b=node('button',text,cls);b.type='button';b.onclick=action;return b}
export function fmt(value:unknown){return typeof value==='number'&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumSignificantDigits:5}):'Unavailable'}
export function labeled(label:string,input:HTMLElement){const l=node('label');l.append(node('span',label),input);return l}
export function badge(text:string){return node('span',text,'badge')}
export function notice(text:string){const e=document.getElementById('notice')!;e.textContent=text;e.hidden=false}
export function errorMessage(error:unknown){return error instanceof Error?error.message:String(error)}
export function select(items:{id:string;title?:string;cellType?:string}[],value:string,change:(value:string)=>void){const el=node('select');for(const item of items){const o=node('option',item.title||item.cellType||item.id);o.value=item.id;el.append(o)}el.value=value;el.onchange=()=>change(el.value);return el}
export function detail(label:string,data:unknown){const d=node('details');d.append(node('summary',label),node('pre',JSON.stringify(data,null,2)));return d}
