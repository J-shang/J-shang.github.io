import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const notes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/data/notes' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.enum(['优化基础', '线性代数', '数值计算', '深度学习工程', 'LLM 实验方法', 'Muon 专题']),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    cutoff: z.coerce.date().optional(),
    order: z.number().default(99),
    featured: z.boolean().default(false),
    readtime: z.number().optional(),
    source: z.url().optional(),
  }),
});

export const collections = { notes };
