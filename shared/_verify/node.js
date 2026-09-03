// Node cross-end runner for erase_text_glyphs.
// Loads the SAME input binaries (rgb.bin / tm.bin) the Python runner used, runs
// them through WebAssembly.instantiate of the identical .wasm, and dumps outputs
// for md5 comparison. Any divergence vs the Python (wasmtime) run is a bug.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { eraseTextGlyphs } = require(path.join(__dirname, "..", "bindings", "textcore.js"));

const OUT = __dirname;

function md5(p) {
  return crypto.createHash("md5").update(fs.readFileSync(p)).digest("hex");
}

(async () => {
  const H = 64, W = 80;
  const rgbBuf = fs.readFileSync(path.join(OUT, "rgb.bin"));
  const tmBuf = fs.readFileSync(path.join(OUT, "tm.bin"));
  const rgbF32 = new Float32Array(rgbBuf.buffer, rgbBuf.byteOffset, rgbBuf.byteLength / 4);
  const tmU8 = new Uint8Array(tmBuf.buffer, tmBuf.byteOffset, tmBuf.byteLength);

  const [result, fill, clean, zone] = await eraseTextGlyphs(
    rgbF32, H, W, tmU8, null, 1.15, 0.6, 10, 1, 1, 1, -1.0, 0);

  fs.writeFileSync(path.join(OUT, "node_result.bin"), Buffer.from(result.buffer));
  fs.writeFileSync(path.join(OUT, "node_fill.bin"), Buffer.from(fill.buffer));
  fs.writeFileSync(path.join(OUT, "node_clean.bin"), Buffer.from(clean.buffer));
  fs.writeFileSync(path.join(OUT, "node_zone.bin"), Buffer.from(zone.buffer));

  for (const name of ["node_result.bin", "node_fill.bin", "node_clean.bin", "node_zone.bin"]) {
    console.log(name.padEnd(16), md5(path.join(OUT, name)));
  }
  console.log("n_text_px =", fill.reduce((a, v) => a + (v > 0 ? 1 : 0), 0));
})();
