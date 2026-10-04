// Handler unit test only: synthetic arguments are not touch-device usability evidence.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../metalrobo/wet_lab/learned.js'),'utf8');
const start=source.indexOf('  function setupCanvas(mode)'),end=source.indexOf('\n  async function mount',start);
assert(end>start);const canvas={focus(){},setPointerCapture(){},setAttribute(){},clientWidth:600,clientHeight:400,getBoundingClientRect:()=>({width:600,height:400})};
const ctx={el:()=>canvas,camera:{x:0,y:0,scale:1},draw(){},index:null,Map,Math};vm.createContext(ctx);vm.runInContext(source.slice(start,end)+';setupCanvas("control");',ctx);
const event=(id,x,y)=>({pointerId:id,offsetX:x,offsetY:y});canvas.onpointerdown(event(1,100,100));canvas.onpointermove(event(1,130,120));assert.equal(ctx.camera.x,30);assert.equal(ctx.camera.y,20);canvas.onpointerup(event(1,130,120));
canvas.onpointerdown(event(1,100,100));canvas.onpointerdown(event(2,200,100));canvas.onpointermove(event(2,300,100));assert.equal(ctx.camera.scale,2);canvas.onpointercancel(event(1,100,100));canvas.onpointerup(event(2,300,100));canvas.onpointermove(event(2,500,100));assert.equal(ctx.camera.scale,2);
canvas.onkeydown({key:'ArrowRight',preventDefault(){}});assert.equal(ctx.camera.x,54);canvas.onkeydown({key:'0',preventDefault(){}});assert.equal(ctx.camera.scale,1);assert.equal(ctx.camera.x,0);
console.log('Pointer pan, two-pointer pinch, cancellation, keyboard and reset handler unit checks passed; physical touch remains pending.');
