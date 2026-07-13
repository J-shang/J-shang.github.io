import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const notes = (await getCollection('notes')).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'J. Shang · Technical Notes',
    description: '关于 AI 系统、优化器与规模化训练的技术笔记。',
    site: context.site,
    items: notes.map((note) => ({
      title: note.data.title,
      description: note.data.description,
      pubDate: note.data.date,
      link: `/notes/${note.id}/`,
      categories: [note.data.category],
    })),
    customData: '<language>zh-CN</language>',
  });
}
