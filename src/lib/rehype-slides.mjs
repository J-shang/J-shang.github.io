const layoutPattern = /^<!--\s*layout:\s*([a-z-]+)\s*-->$/;
const notesPattern = /^<!--\s*notes:\s*([\s\S]*?)\s*-->$/;
const incrementalPattern = /^<!--\s*incremental\s*-->$/;
const allowedLayouts = new Set(['title', 'statement', 'split', 'figure', 'code', 'comparison']);

function isRawComment(node) {
  return node?.type === 'raw' && typeof node.value === 'string';
}

function addClass(node, className) {
  node.properties ??= {};
  const current = node.properties.className;
  const classes = Array.isArray(current) ? current : current ? [current] : [];
  if (!classes.includes(className)) node.properties.className = [...classes, className];
}

function classNames(node) {
  const value = node?.properties?.className;
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === 'string') return value.split(/\s+/).filter(Boolean);
  return [];
}

function isElement(node, tagName) {
  return node?.type === 'element' && (!tagName || node.tagName === tagName);
}

function isWhitespace(node) {
  return node?.type === 'text' && node.value.trim().length === 0;
}

function readableText(node) {
  if (!node) return '';
  if (node.type === 'text') return node.value;
  if (!Array.isArray(node.children)) return '';
  if (node.tagName === 'annotation' || node.properties?.ariaHidden === 'true') return '';
  if (classNames(node).includes('katex-html')) return '';
  return node.children.map(readableText).join('');
}

function countElements(node, predicate) {
  if (!node) return 0;
  const current = isElement(node) && predicate(node) ? 1 : 0;
  return current + (node.children ?? []).reduce((total, child) => (
    total + countElements(child, predicate)
  ), 0);
}

function containsImage(node) {
  if (isElement(node, 'img')) return true;
  return (node?.children ?? []).some(containsImage);
}

function isImageParagraph(node) {
  if (!isElement(node, 'p')) return false;
  const meaningful = (node.children ?? []).filter((child) => !isWhitespace(child));
  return meaningful.length === 1 && containsImage(meaningful[0]);
}

function titleLength(title) {
  return Array.from(title.replace(/\s+/g, '')).length;
}

function figureSurface(children, heading) {
  const media = children.find(isImageParagraph);
  if (!media) return { children, hasMedia: false, hasCopy: false };

  const copy = children.filter((node) => node !== heading && node !== media);
  const hasCopy = copy.some((node) => !isWhitespace(node));
  const figureGrid = {
    type: 'element',
    tagName: 'div',
    properties: { className: ['slide__figure-grid'] },
    children: [
      {
        type: 'element',
        tagName: 'div',
        properties: { className: ['slide__media'] },
        children: [media],
      },
      ...(hasCopy ? [{
        type: 'element',
        tagName: 'div',
        properties: { className: ['slide__copy'] },
        children: copy,
      }] : []),
    ],
  };

  return {
    children: heading ? [heading, figureGrid] : [figureGrid],
    hasMedia: true,
    hasCopy,
  };
}

function slideTraits(layout, children, heading) {
  const title = readableText(heading).replace(/\s+/g, ' ').trim();
  const length = titleLength(title);
  const content = children.filter((node) => node !== heading);
  const readableContent = content.map(readableText).join(' ').replace(/\s+/g, ' ').trim();
  const characters = titleLength(readableContent);
  const listItems = content.reduce((total, node) => (
    total + countElements(node, (element) => element.tagName === 'li')
  ), 0);
  const tableRows = content.reduce((total, node) => (
    total + countElements(node, (element) => element.tagName === 'tr')
  ), 0);
  const displayMath = content.reduce((total, node) => (
    total + countElements(node, (element) => classNames(element).includes('katex-display'))
  ), 0);
  const codeBlocks = content.reduce((total, node) => (
    total + countElements(node, (element) => element.tagName === 'pre')
  ), 0);
  const density = characters + listItems * 34 + tableRows * 42 + displayMath * 48 + codeBlocks * 120;
  const meaningfulContent = content.filter((node) => !isWhitespace(node));
  const listFocus = meaningfulContent.length === 1
    && (isElement(meaningfulContent[0], 'ul') || isElement(meaningfulContent[0], 'ol'))
    && listItems >= 4;
  const appendix = /^Appendix\b/i.test(title);
  const dense = appendix || density >= 650 || tableRows >= 6 || (listItems >= 6 && characters >= 380);
  const extraDense = density >= 900 || (appendix && density >= 700);

  return {
    title,
    length,
    density,
    classes: [
      ...(length >= (layout === 'title' ? 18 : 30) ? ['slide--long-title'] : []),
      ...(length >= (layout === 'title' ? 28 : 46) ? ['slide--extra-long-title'] : []),
      ...(appendix ? ['slide--appendix'] : []),
      ...(dense ? ['slide--dense'] : []),
      ...(extraDense ? ['slide--extra-dense'] : []),
      ...(listFocus ? ['slide--list-focus'] : []),
    ],
  };
}

function normalizeSlide(nodes) {
  let layout = 'default';
  let incremental = false;
  const children = [];
  const notes = [];

  for (const node of nodes) {
    if (isRawComment(node)) {
      const value = node.value.trim();
      const layoutMatch = value.match(layoutPattern);
      if (layoutMatch) {
        if (!allowedLayouts.has(layoutMatch[1])) {
          throw new Error(`Unsupported slide layout: ${layoutMatch[1]}`);
        }
        layout = layoutMatch[1];
        continue;
      }
      if (incrementalPattern.test(value)) {
        incremental = true;
        continue;
      }
      const notesMatch = value.match(notesPattern);
      if (notesMatch) {
        notes.push({
          type: 'element',
          tagName: 'aside',
          properties: { className: ['slide-notes'], 'aria-label': '演讲者备注', hidden: true },
          children: [{ type: 'text', value: notesMatch[1].trim() }],
        });
        continue;
      }
    }

    if (incremental && node.type === 'element' && (node.tagName === 'ul' || node.tagName === 'ol')) {
      addClass(node, 'incremental');
      incremental = false;
    }
    children.push(node);
  }

  return { layout, children, notes };
}

function slideId(index, total) {
  return `slide-${String(index + 1).padStart(String(total).length, '0')}`;
}

function isSlideBreak(node, file) {
  if (node.type !== 'element' || node.tagName !== 'hr') return false;
  const start = node.position?.start?.offset;
  const end = node.position?.end?.offset;
  if (Number.isInteger(start) && Number.isInteger(end)) {
    return String(file.value ?? '').slice(start, end).trim() === '---';
  }
  return false;
}

export default function rehypeSlides() {
  return (tree, file) => {
    const sourcePath = String(file.path ?? file.history?.[0] ?? '').replaceAll('\\', '/');
    const hasSlideDirective = tree.children?.some((node) => (
      isRawComment(node) && (layoutPattern.test(node.value.trim()) || notesPattern.test(node.value.trim()))
    ));
    if (!sourcePath.includes('/data/slides/') && !hasSlideDirective) return;

    const groups = [[]];
    for (const node of tree.children ?? []) {
      if (isSlideBreak(node, file)) groups.push([]);
      else groups.at(-1).push(node);
    }
    const emptyIndex = groups.findIndex((group) => !group.some((node) => (
      node.type !== 'text' || node.value.trim().length > 0
    )));
    if (emptyIndex >= 0) throw new Error(`Slide ${emptyIndex + 1} is empty.`);
    const total = groups.length;

    tree.children = groups.map((group, index) => {
      const { layout, children, notes } = normalizeSlide(group);
      const id = slideId(index, total);
      const numberId = `${id}-number`;
      const heading = children.find((node) => (
        node.type === 'element' && (node.tagName === 'h1' || node.tagName === 'h2')
      ));
      if (heading) {
        heading.properties ??= {};
        heading.properties.id ??= `${id}-title`;
      }
      const traits = slideTraits(layout, children, heading);
      const figure = layout === 'figure'
        ? figureSurface(children, heading)
        : { children, hasMedia: false, hasCopy: false };
      const labelledBy = [numberId, heading?.properties?.id].filter(Boolean).join(' ');
      return {
        type: 'element',
        tagName: 'section',
        properties: {
          id,
          className: [
            'slide',
            `slide--${layout}`,
            ...traits.classes,
            ...(figure.hasMedia ? ['slide--figure-has-media'] : []),
            ...(figure.hasCopy ? ['slide--figure-has-copy'] : []),
          ],
          tabIndex: -1,
          role: 'group',
          'aria-roledescription': '幻灯片',
          'aria-labelledby': labelledBy,
          'data-slide-index': String(index + 1),
          'data-slide-total': String(total),
          'data-layout': layout,
          'data-title-length': String(traits.length),
          'data-density': String(traits.density),
        },
        children: [
          {
            type: 'element',
            tagName: 'span',
            properties: { id: numberId, className: ['sr-only'] },
            children: [{ type: 'text', value: `第 ${index + 1} 页，共 ${total} 页` }],
          },
          {
            type: 'element',
            tagName: 'div',
            properties: { className: ['slide__surface'] },
            children: figure.children,
          },
          ...notes,
        ],
      };
    });
  };
}
