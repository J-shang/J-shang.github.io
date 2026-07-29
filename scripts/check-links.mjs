import { readdir, readFile, access } from 'node:fs/promises';
import { resolve, relative } from 'node:path';

const dist = resolve('dist');

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const path = resolve(directory, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  }));
  return nested.flat();
}

const htmlFiles = (await walk(dist)).filter((file) => (
  file.endsWith('.html')
  && !/ \d+\.html$/.test(file)
));
const failures = [];

for (const file of htmlFiles) {
  const html = await readFile(file, 'utf8');
  const currentPath = `/${relative(dist, file).replace(/index\.html$/, '').replaceAll('\\', '/')}`;
  const references = [
    ...[...html.matchAll(/href="([^"]+)"/g)].map((match) => ({ attribute: 'href', value: match[1] })),
    ...[...html.matchAll(/src="([^"]+)"/g)].map((match) => ({ attribute: 'src', value: match[1] })),
  ];
  for (const { attribute, value } of references) {
    if (/^(https?:|mailto:|#|data:)/.test(value)) continue;
    const url = new URL(value, `https://j-shang.github.io${currentPath}`);
    let target = url.pathname;
    if (target.endsWith('/')) target += 'index.html';
    const targetFile = resolve(dist, `.${target}`);
    try { await access(targetFile); }
    catch { failures.push(`${currentPath} -> ${attribute}="${value}"`); }
  }
}

if (failures.length) {
  console.error(`Found ${failures.length} broken local references:\n${failures.join('\n')}`);
  process.exitCode = 1;
} else {
  console.log(`Checked ${htmlFiles.length} HTML pages: no broken local links or assets.`);
}
