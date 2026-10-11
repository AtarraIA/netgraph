// Builds the app into one self-contained HTML file (it opens straight from disk, no server needed):
//   node build.mjs
// The engine goes in <script id="engine">; the page also hands that same text to the training worker.
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
import {ENGINE, UI} from './src/sources.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = f => fs.readFileSync(path.join(here, 'src', f), 'utf8');
const join = files => files.map(f => `/* ===== ${f} ===== */\n${src(f).trimEnd()}\n`).join('\n');
export const OUT = path.join(here, 'Netgraph_ compose neural networks.html');

export function build(){
  const html = src('index.html')
    .replace('{{STYLE}}', () => src('style.css').trimEnd())
    .replace('{{BODY}}', () => src('body.html').trimEnd())
    .replace('{{ENGINE}}', () => join(ENGINE))
    .replace('{{UI}}', () => join(UI));
  fs.writeFileSync(OUT, html);
  return OUT;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)){
  const out = build();
  console.log(`Built ${path.basename(out)} (${(fs.statSync(out).size/1024).toFixed(0)} KB)`);
}
