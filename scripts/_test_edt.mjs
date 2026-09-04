// Read mask.raw from each scripts/_cmp_work/_edt/<name>/ dir, run the pure-JS
// distanceFromZeros (exact Euclidean EDT), and write js.raw for comparison.
import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { pathToFileURL } from 'url';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORK = join(__dirname, '_cmp_work', '_edt');
const { distanceFromZeros } = await import(pathToFileURL(join(__dirname, '..', 'browser', 'src', 'cv-bridge.js')).href);

for (const name of readdirSync(WORK)) {
  const d = join(WORK, name);
  const [H, W] = readFileSync(join(d, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
  const mask = new Uint8Array(readFileSync(join(d, 'mask.raw')));
  const out = distanceFromZeros(mask, H, W, 3);
  writeFileSync(join(d, 'js.raw'), Buffer.from(out.buffer, out.byteOffset, out.byteLength));
  console.log(`${name}: js dist written ${H}x${W}`);
}
console.log('EDT TEST DONE');
