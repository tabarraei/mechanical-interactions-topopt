// Headless DOM/canvas stubs test standalone control logic, not browser rendering.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map(),timers=[];
const ctx=new Proxy({}, {get:(target,key)=>target[key]||(()=>{}),set:(target,key,value)=>{target[key]=value;return true}});
const document={getElementById(id){if(!elements.has(id))elements.set(id,{value:id==='field'?'rho':'0',textContent:'',max:0,appendChild(){},getContext(){return ctx}});return elements.get(id)},createElement(){return {}}};
class ImageMock{set src(value){this._src=value;if(this.onload)this.onload()}}
const sandbox={document,Image:ImageMock,console,assert,setInterval(fn){timers.push(fn);return timers.length-1},clearInterval(id){timers[id]=null}};
vm.createContext(sandbox);vm.runInContext(code,sandbox);
vm.runInContext(`
assert.equal(cases.length,12);
assert.equal(cases.reduce((n,c)=>n+c.T,0),3709);
for(const c of cases){assert.equal(c.rho.length,c.T);assert.equal(c.x.length,c.T);assert.equal(c.metrics.length,c.T)}
assert.equal(Number(element('step',0).max),191);
assert.equal(Number(element('step',1).max),192);
element('case',0).value='6';element('case',0).onchange();
assert.equal(Number(element('step',0).max),500);assert.equal(current(0),0);
element('step',0).value='500';element('step',0).oninput();
assert(element('metrics',0).textContent.includes('none (terminal)'));
element('field').value='x';element('field').onchange();
assert(element('metrics',0).textContent.includes('Raw design variable x'));
element('play',0).onclick();assert.equal(current(0),0);assert.equal(element('play',0).textContent,'Pause');
`,sandbox);
timers[timers.length-1]();assert.equal(Number(document.getElementById('step0').value),1);
vm.runInContext(`element('play',0).onclick();assert.equal(element('play',0).textContent,'Play');element('case',0).value='4';element('case',0).onchange();assert.equal(Number(element('step',0).max),367);assert.equal(current(0),0);`,sandbox);
console.log('PASS: all12/3709 previews, variable slider bounds, selection reset, terminal timing, rho/x labels, play/pause and independent comparison controls');
