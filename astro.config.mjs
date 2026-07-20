import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlides from './src/lib/rehype-slides.mjs';

export default defineConfig({
  site: 'https://j-shang.github.io',
  output: 'static',
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        [rehypeKatex, { strict: false, throwOnError: false }],
        rehypeSlides,
      ],
    }),
    shikiConfig: {
      theme: 'github-dark-default',
      wrap: true,
    },
  },
});
