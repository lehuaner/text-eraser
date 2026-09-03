// Bundle the pure-browser text-eraser ESM port into a single classic IIFE so it can
// be loaded inside a *classic* Web Worker via importScripts (opencv.js UMD only attaches
// `cv` in a classic worker; a module worker cannot load it). The IIFE exposes `self.TE`.
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

await build({
  entryPoints: [resolve(__dirname, 'src/index.js')],
  bundle: true,
  format: 'iife',
  globalName: 'TE',
  outfile: resolve(__dirname, 'dist/te-bundle.js'),
  // onnxruntime-web is loaded by the worker via a direct URL import; keep it external
  // so the bundle never tries to resolve the bare specifier.
  external: ['onnxruntime-web'],
  target: ['es2020'],
  logLevel: 'info',
});

console.log('built browser/dist/te-bundle.js');
