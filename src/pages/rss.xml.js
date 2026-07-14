import rss from '@astrojs/rss';
import { loadSiteContent, noteHref, sectionTitle } from '../lib/content';

export async function GET(context) {
  const { notes, topicById } = await loadSiteContent();
  notes.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'J. Shang · Technical Notes',
    description: '关于 AI 系统、优化器、数据处理与规模化训练的技术笔记。',
    site: context.site,
    items: notes.map((note) => {
      const topic = topicById.get(note.data.topic);
      return {
        title: note.data.title,
        description: note.data.description,
        pubDate: note.data.date,
        link: noteHref(note),
        categories: topic ? [topic.data.title, sectionTitle(topic, note.data.section)] : [note.data.topic],
      };
    }),
    customData: '<language>zh-CN</language>',
  });
}
