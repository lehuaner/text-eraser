import fs from 'fs';
const wasm = fs.readFileSync(new URL('../build/textcore.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;
const mem = () => new Uint8Array(ex.memory.buffer);

function runBlock(hh, ww, dir, bg){
  bg = bg || [200,190,170];
  const H=200,W=200;
  const rgb=new Float32Array(H*W*3);
  for(let i=0;i<H*W;i++){rgb[i*3]=bg[0];rgb[i*3+1]=bg[1];rgb[i*3+2]=bg[2];}
  // add a mild vertical gradient so it's not perfectly uniform
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){const i=(y*W+x)*3;rgb[i]=Math.min(255,rgb[i]+y*0.2);rgb[i+1]=Math.min(255,rgb[i+1]+y*0.2);}
  const mask=new Uint8Array(H*W);
  const y0=Math.floor((H-hh)/2), x0=Math.floor((W-ww)/2);
  for(let y=y0;y<y0+hh;y++)for(let x=x0;x<x0+ww;x++){mask[y*W+x]=255;rgb[(y*W+x)*3]=255;rgb[(y*W+x)*3+1]=255;rgb[(y*W+x)*3+2]=255;}
  const rb=new Uint8Array(rgb.buffer);
  const pRgb=ex.alloc(rb.length),pMask=ex.alloc(mask.length),pOut=ex.alloc(rb.length);
  mem().set(rb,pRgb);mem().set(mask,pMask);
  ex.patchmatch_inpaint(pRgb,H,W,pMask,0,0,7,dir,0,pOut);
  const o=new Float32Array(ex.memory.buffer.slice(pOut,pOut+rb.length));
  let white=0,nan=0,oor=0,tot=0,sum=0,mn=999,mx=-999;
  for(let y=y0;y<y0+hh;y++)for(let x=x0;x<x0+ww;x++){tot++;const i=(y*W+x)*3;
    const r=o[i],g=o[i+1],b=o[i+2];
    if([r,g,b].some(v=>!isFinite(v)))nan++;
    else if(r<0||r>255||g<0||g>255||b<0||b>255)oor++;
    else if(Math.abs(r-255)<1&&Math.abs(g-255)<1&&Math.abs(b-255)<1)white++;
    else {const m=(r+g+b)/3;sum+=m;mn=Math.min(mn,m);mx=Math.max(mx,m);}
  }
  return {hh,ww,dir,white,nan,oor,pct:(100*white/tot).toFixed(1),mean:(sum/Math.max(1,(tot-white-nan-oor))).toFixed(1),mn:mn.toFixed(1),mx:mx.toFixed(1)};
}
console.log('=== default (no direction) ===');
for(const [hh,ww] of [[10,10],[20,60],[40,120],[80,80],[120,40],[200,200]]){
  const r=runBlock(hh,ww,-1.0);
  console.log(`  ${hh}x${ww}: white=${r.white} nan=${r.nan} oor=${r.oor} fillMean=${r.mean} [${r.mn},${r.mx}]`);
}
console.log('=== genuine direction mode (should also be clean now) ===');
for(const dir of [0.0, 45.0, 90.0]){
  const r=runBlock(60,120,dir);
  console.log(`  dir=${dir}: white=${r.white} nan=${r.nan} oor=${r.oor} fillMean=${r.mean} [${r.mn},${r.mx}]`);
}
