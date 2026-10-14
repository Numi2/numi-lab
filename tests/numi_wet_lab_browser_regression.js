// Executed through playwright-cli run-code; engineering regression, not usability.
async (page) => {
  const results = {surface:'Automated Chromium regression; actual Codex interactions qualified separately',checks:[]};
  await page.locator('#connection').filter({hasText:'Connected'}).waitFor();
  const revision = await page.locator('#connection').textContent();
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.getByRole('button',{name:'Search / commands',exact:false}).click();
  await page.locator('#command-query').fill('Performance');
  await page.locator('#command-query').press('Enter');
  await page.locator('#drawer').waitFor({state:'visible'});
  if(!await page.locator('#drawer-title').textContent().then(t=>t.includes('Performance')))throw Error('Keyboard command failed');
  await page.keyboard.press('Escape');
  await page.keyboard.press('Tab');
  const focus=await page.evaluate(()=>({visible:document.activeElement.matches(':focus-visible'),outline:getComputedStyle(document.activeElement).outlineStyle}));
  if(!focus.visible||focus.outline==='none')throw Error('Keyboard focus is not visible');results.focus=focus;
  await page.emulateMedia({reducedMotion:'reduce'});
  for(const appearance of ['light','dark']){
    await page.emulateMedia({colorScheme:appearance});
    for(const width of [1440,1024,768]){
      await page.setViewportSize({width,height:960});
      const layout=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,labels:[...document.querySelectorAll('input,textarea,select')].filter(e=>!e.hidden&&e.offsetParent!==null).every(e=>!!(e.closest('label')||e.getAttribute('aria-label'))),unnamedButtons:[...document.querySelectorAll('button')].filter(e=>!e.textContent.trim()&&!e.getAttribute('aria-label')).length,reduced:matchMedia('(prefers-reduced-motion:reduce)').matches}));
      if(layout.scrollWidth>width+1||!layout.labels||layout.unnamedButtons||!layout.reduced)throw Error('Layout/accessibility regression '+JSON.stringify(layout));
      const contrast=await page.evaluate(()=>{const style=getComputedStyle(document.documentElement);const luminance=(hex)=>{const h=hex.trim().slice(1);const full=h.length===3?h.split('').map(c=>c+c).join(''):h;const rgb=full.match(/../g).map(x=>parseInt(x,16)/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4);return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]};const bg=luminance(style.getPropertyValue('--surface'));return Object.fromEntries(['--ink','--muted','--accent'].map(key=>{const l=luminance(style.getPropertyValue(key));return [key,(Math.max(l,bg)+.05)/(Math.min(l,bg)+.05)]}));});
      if(Object.values(contrast).some(ratio=>!Number.isFinite(ratio)||ratio<4.5))throw Error('Text contrast regression '+JSON.stringify(contrast));
      results.checks.push({appearance,...layout,contrast});
    }
  }
  await page.emulateMedia({colorScheme:'light',reducedMotion:'no-preference'});await page.setViewportSize({width:1440,height:1000});
  if((await page.locator('#connection').textContent())!==revision)throw Error('Presentation preference mutated scientific revision');
  if(errors.length)throw Error('Browser errors '+JSON.stringify(errors));
  results.revisionUnchanged=true;results.errors=errors;
  return results;
}
