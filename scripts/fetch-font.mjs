// Download the Unica One webfont so the deployed site doesn't depend on
// the Google Fonts CDN at runtime. The latin-subset woff2 is fetched from
// whatever URL Google currently serves, then written into the Vite public
// dir so it ships at /fonts/unica-one.woff2.
//
// Run:  node scripts/fetch-font.mjs

import { writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(REPO_ROOT, 'src', 'public', 'fonts');
const OUT_FILE = join(OUT_DIR, 'unica-one.woff2');

const CSS_URL = 'https://fonts.googleapis.com/css2?family=Unica+One&display=swap';
// Modern UA → Google serves woff2 (otherwise we'd get woff or ttf).
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15';

async function main() {
  const cssRes = await fetch(CSS_URL, { headers: { 'User-Agent': UA } });
  if (!cssRes.ok) throw new Error(`Fetch CSS failed: ${cssRes.status}`);
  const css = await cssRes.text();

  // The CSS contains a /* latin */ block with a url(...) for the woff2.
  const m = css.match(/\/\*\s*latin\s*\*\/[\s\S]*?url\((https?:\/\/[^)]+)\)\s*format\(['"]?woff2['"]?\)/);
  if (!m) throw new Error(`Couldn't find latin woff2 URL in CSS:\n${css}`);
  const fontUrl = m[1];

  const fontRes = await fetch(fontUrl);
  if (!fontRes.ok) throw new Error(`Fetch font failed: ${fontRes.status}`);
  const buf = Buffer.from(await fontRes.arrayBuffer());

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(OUT_FILE, buf);
  console.log(`wrote ${OUT_FILE} (${(buf.length / 1024).toFixed(1)} KiB) from ${fontUrl}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
