// Validate the BROWSER binding glue (textcore.browser.js) in Node by polyfilling
// fetch -> local wasm file. This exercises the exact call path the Web Worker uses
// (fetch + WebAssembly.instantiate), proving the browser glue produces the same
// bytes as the Python (wasmtime) and Node (WebAssembly.instantiate) runners.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const WASM = path.join(__dirname, "..", "build", "textcore.wasm");
const { ensure, eraseTextGlyphs } = require(path.join(__dirname, "..", "bindings", "textcore.browser.js"));

// Polyfill fetch so the browser binding can load the local .wasm in Node.
globalThis.fetch = async (url) => {
  const buf = fs.readFileSync(WASM);
  return { ok: true, status: 200, arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

// CRITICAL: md5 MUST read the file at the path, never hash the path string itself.
function md5(p) { return crypto.createHash("md5").update(fs.readFileSync(p)).digest("hex"); }

(async () => {
  const H = 64, W = 80;
  const rgbBuf = fs.readFileSync(path.join(__dirname, "rgb.bin"));
  const tmBuf = fs.readFileSync(path.join(__dirname, "tm.bin"));
  const rgbF32 = new Float32Array(rgbBuf.buffer, rgbBuf.byteOffset, rgbBuf.byteLength / 4);
  const tmU8 = new Uint8Array(tmBuf.buffer, tmBuf.byteOffset, tmBuf.byteLength);

  await ensure(); // load textcore.wasm through the browser binding's fetch path

  // mirror index.js defaults: chromaKeep=1 (deglowChromaKeep default true)
  const [result, fill, clean, zone] = eraseTextGlyphs(
    rgbF32, H, W, tmU8, null, 1.15, 0.6, 10, 1, 1, 1, -1.0, 0);

  fs.writeFileSync(path.join(__dirname, "brw_result.bin"), Buffer.from(result.buffer));
  fs.writeFileSync(path.join(__dirname, "brw_fill.bin"), Buffer.from(fill.buffer));
  fs.writeFileSync(path.join(__dirname, "brw_clean.bin"), Buffer.from(clean.buffer));
  fs.writeFileSync(path.join(__dirname, "brw_zone.bin"), Buffer.from(zone.buffer));

  for (const name of ["brw_result.bin", "brw_fill.bin", "brw_clean.bin", "brw_zone.bin"]) {
    console.log(name.padEnd(16), md5(path.join(__dirname, name)));
  }
  console.log("n_text_px =", fill.reduce((a, v) => a + (v > 0 ? 1 : 0), 0));
})();
