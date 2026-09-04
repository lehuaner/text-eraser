// Browser-side consumer of the shared WASM core (Node stand-in for the browser).
// Loads build/textcore.wasm via WebAssembly, runs distance_transform_edt on a
// mask file, writes <maskfile>.out_node.bin for the cross-consumer identity check.
//
// NOTE: wasm memory may GROW during alloc / during the kernel call, which
// invalidates the `memory.buffer` reference. We re-fetch `ex.memory.buffer`
// right before every read/write to stay correct (the browser must do the same).
const fs = require("fs");
const path = require("path");

const wasmPath = path.join(__dirname, "..", "build", "textcore.wasm");
const maskFile = process.argv[2];
if (!maskFile) {
  console.error("usage: node node-edt.cjs <maskfile.bin>");
  process.exit(1);
}

const buf = fs.readFileSync(maskFile);
const h = buf.readInt32LE(0);
const w = buf.readInt32LE(4);
const n = h * w;
const mask = buf.subarray(8, 8 + n);

WebAssembly.instantiate(fs.readFileSync(wasmPath), {}).then(({ instance }) => {
  const ex = instance.exports;

  const pMask = ex.alloc(n);
  const pOut = ex.alloc(n * 4);

  // Fresh buffer AFTER allocations (alloc may have grown memory).
  let m = ex.memory.buffer;
  new Uint8Array(m, pMask, n).set(mask);

  ex.distance_transform_edt(pMask, h, w, pOut);

  // Fresh buffer AGAIN (the kernel may have grown memory internally).
  m = ex.memory.buffer;
  // Read n*4 raw bytes of the f32 output directly (avoids typed-array/buffer quirks).
  const out = Buffer.allocUnsafe(n * 4);
  out.set(new Uint8Array(m, pOut, n * 4));
  if (out.length !== n * 4) {
    throw new Error(`output size ${out.length} != expected ${n * 4}`);
  }
  fs.writeFileSync(maskFile + ".out_node.bin", out);
  ex.dealloc(pMask, n);
  ex.dealloc(pOut, n * 4);
  console.log(
    `node: ${path.basename(maskFile)} ${h}x${w} n=${n} wrote ${
      path.basename(maskFile)
    }.out_node.bin (${out.length} bytes)`
  );
});
