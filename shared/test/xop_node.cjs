// Node runner: executes one shared-core operator via the Node (WebAssembly) binding.
// Used by xop_test.py to compare Node vs Python(wasmtime) byte-for-byte.
// Usage: node xop_node.cjs <op> <in.bin> <out.bin> [args...]
const fs = require("fs");
const tc = require("../bindings/textcore.js");

function readIn(file) {
  const buf = fs.readFileSync(file);
  const h = buf.readInt32LE(0);
  const w = buf.readInt32LE(4);
  const data = buf.subarray(8);
  return { h, w, data };
}
function writeTA(file, ta) {
  fs.writeFileSync(file, Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength));
}

async function main() {
  const op = process.argv[2];
  const infile = process.argv[3];
  const outfile = process.argv[4];
  const rest = process.argv.slice(5).map((x) => parseInt(x, 10));
  const { h, w, data } = readIn(infile);
  await tc.ready();
  let out;
  if (op === "rgb_to_gray") {
    // input payload is uint8 RGB bytes; convert to float32 BY VALUE (matches the
    // Python binding's astype(float32)), not a bit reinterpretation.
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    const f32 = new Float32Array(u8.length);
    for (let i = 0; i < u8.length; i++) f32[i] = u8[i];
    out = await tc.rgbToGray(f32, h, w);
    writeTA(outfile, out);
  } else if (op === "distance") {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    out = await tc.distanceTransformEdt(u8, h, w);
    writeTA(outfile, out);
  } else if (op === "otsu") {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    const r = await tc.thresholdOtsu(u8, h, w);
    writeTA(outfile, r.bin);
  } else if (op === "morph") {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    const ksize = rest[0];
    const shape = rest[1] ? "rect" : "ellipse";
    // rest[2] convention matches the Python driver: 1 = dilate, 0 = erode.
    const opn = rest[2] ? "dilate" : "erode";
    // Build the structuring-element bitmap. For rect it's all ones; for ellipse it's the
    // disk approximation. (The exact cv2 rasterized ellipse is validated separately by the
    // backend smoke test, which forwards cv2.getStructuringElement directly.)
    const kh = ksize, kw = ksize;
    const kern = new Uint8Array(kh * kw);
    if (shape === "rect") {
      kern.fill(1);
    } else {
      const anchor = Math.floor(ksize / 2), ca = (ksize - 1) / 2;
      for (let ky = 0; ky < ksize; ky++)
        for (let kx = 0; kx < ksize; kx++) {
          const dx = kx - anchor, dy = ky - anchor;
          if (dx * dx + dy * dy <= ca * ca + 1e-6) kern[ky * ksize + kx] = 1;
        }
    }
    out = await tc.morphology(u8, h, w, kern, kh, kw, opn);
    writeTA(outfile, out);
  } else if (op === "cc") {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    const r = await tc.connectedComponents(u8, h, w);
    const stats = new Int32Array(r.n * 5);
    for (let i = 0; i < r.n; i++) {
      const s = r.stats[i];
      const b = i * 5;
      stats[b] = s.left; stats[b + 1] = s.top; stats[b + 2] = s.width;
      stats[b + 3] = s.height; stats[b + 4] = s.area;
    }
    const head = Buffer.alloc(4);
    head.writeInt32LE(r.n, 0);
    fs.writeFileSync(outfile, Buffer.concat([head, Buffer.from(r.labels.buffer), Buffer.from(stats.buffer)]));
  } else if (op === "resize_cubic") {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.length);
    out = await tc.resizeGrayCubic(u8, h, w, rest[0], rest[1]);
    writeTA(outfile, out);
  } else if (op === "resize_linear") {
    const f32 = new Float32Array(data.buffer, data.byteOffset, data.length / 4);
    out = await tc.resizeFloatLinear(f32, h, w, rest[0], rest[1]);
    writeTA(outfile, out);
  } else {
    throw new Error("unknown op " + op);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
