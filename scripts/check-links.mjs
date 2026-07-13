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

const htmlFiles = (await walk(dist)).filter((file) => file.endsWith('.html'));
const failures = [];

for (const file of htmlFiles) {
  const html = await readFile(file, 'utf8');
  const currentPath = `/${relative(dist, file).replace(/index\.html$/, '').replaceAll('\\', '/')}`;
  const links = [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
  for (const href of links) {
    if (/^(https?:|mailto:|#|data:)/.test(href)) continue;
    const url = new URL(href, `https://j-shang.github.io${currentPath}`);
    let target = url.pathname;
    if (target.endsWith('/')) target += 'index.html';
    const targetFile = resolve(dist, `.${target}`);
    try { await access(targetFile); }
    catch { failures.push(`${currentPath} -> ${href}`); }
  }
}

if (failures.length) {
  console.error(`Found ${failures.length} broken local links:\n${failures.join('\n')}`);
  process.exitCode = 1;
} else {
  console.log(`Checked ${htmlFiles.length} HTML pages: no broken local links.`);
}
