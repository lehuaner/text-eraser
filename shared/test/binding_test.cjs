// Binding-level cross-host test (browser side).
// Uses shared/bindings/textcore.js (WebAssembly) — the real integration layer
// the browser will use. Writes <f>.bind_node.bin for identity comparison.
const fs = require("fs");
const path = require("path");
const { distanceTransformEdt } = require("../bindings/textcore.js");

const f = process.argv[2];
const buf = fs.readFileSync(f);
const h = buf.readInt32LE(0);
const w = buf.readInt32LE(4);
const n = h * w;
const mask = new Uint8Array(buf.subarray(8, 8 + n));

distanceTransformEdt(mask, h, w).then((out) => {
  fs.writeFileSync(f + ".bind_node.bin", Buffer.from(out.buffer));
  console.log(`node-binding: ${path.basename(f)} ${h}x${w} wrote .bind_node.bin`);
});
