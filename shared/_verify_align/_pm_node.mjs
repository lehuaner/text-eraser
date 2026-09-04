// Node side of the patchmatch_inpaint cross-end check.
// Input file layout: rgb f32 (H*W*3) | mask u8 (H*W) | sample u8 (H*W) |
//                    header f32 [H, W, P, has_sample, direction_deg, seed]
import fs from 'fs';
const [fin, fout] = process.argv.slice(2);
const buf = fs.readFileSync(fin);
const H = buf.readFloatLE(buf.length - 24);
const W = buf.readFloatLE(buf.length - 20);
const P = buf.readFloatLE(buf.length - 16);
const hasSample = buf.readFloatLE(buf.length - 12);
const dirDeg = buf.readFloatLE(buf.length - 8);
const seed = buf.readFloatLE(buf.length - 4);
const n = H * W;
const rgbF = new Float32Array(buf.buffer.slice(0, n * 12));
const mask = buf.subarray(n * 12, n * 13);
const sample = buf.subarray(n * 13, n * 14);

const wasm = fs.readFileSync(new URL('../build/textcore.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;
const mem = () => new Uint8Array(ex.memory.buffer);

const pIn = ex.alloc(n * 12), pMask = ex.alloc(n), pS = ex.alloc(n), pOut = ex.alloc(n * 12);
mem().set(new Uint8Array(rgbF.buffer), pIn);
mem().set(mask, pMask);
if (hasSample) mem().set(sample, pS);
ex.patchmatch_inpaint(pIn, H, W, pMask, hasSample ? pS : 0, hasSample, P, dirDeg, seed, pOut);
const out = new Uint8Array(ex.memory.buffer.slice(pOut, pOut + n * 12));
fs.writeFileSync(fout, out);
ex.dealloc(pIn, n * 12); ex.dealloc(pMask, n); ex.dealloc(pS, n); ex.dealloc(pOut, n * 12);
