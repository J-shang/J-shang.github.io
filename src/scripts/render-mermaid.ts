interface MermaidDiagram {
  canvas: HTMLDivElement;
  figure: HTMLElement;
  source: string;
  status: HTMLParagraphElement;
}

const palette = (dark: boolean) => dark
  ? {
      paper: '#101310',
      raised: '#171b17',
      ink: '#edf0e9',
      softInk: '#a8afa5',
      line: '#596159',
      signal: '#83a7ff',
      signalSoft: '#1c315e',
      warm: '#ff8a57',
    }
  : {
      paper: '#f5f3ed',
      raised: '#fffdf8',
      ink: '#171816',
      softInk: '#565a55',
      line: '#aaa99f',
      signal: '#1258ff',
      signalSoft: '#dce6ff',
      warm: '#ef6c36',
    };

function createDiagram(block: HTMLPreElement, index: number): MermaidDiagram {
  const source = block.textContent?.trim() ?? '';
  const precedingLabel = block.previousElementSibling?.textContent?.trim();
  const label = precedingLabel || `Mermaid 图 ${index + 1}`;

  const figure = document.createElement('figure');
  figure.className = 'mermaid-diagram';
  figure.setAttribute('aria-label', label);

  const canvas = document.createElement('div');
  canvas.className = 'mermaid-diagram__canvas';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', label);
  canvas.setAttribute('aria-busy', 'true');

  const status = document.createElement('p');
  status.className = 'mermaid-diagram__status';
  status.textContent = '正在渲染图表…';

  const details = document.createElement('details');
  details.className = 'mermaid-diagram__source';
  const summary = document.createElement('summary');
  summary.textContent = '查看 Mermaid 源码';
  details.append(summary, block.cloneNode(true));

  figure.append(canvas, status, details);
  block.replaceWith(figure);
  return { canvas, figure, source, status };
}

export async function renderMermaidDiagrams(root: ParentNode = document) {
  const blocks = [...root.querySelectorAll<HTMLPreElement>('pre[data-language="mermaid"]')];
  if (blocks.length === 0) return;

  let mermaid;
  try {
    ({ default: mermaid } = await import('mermaid'));
  } catch (error) {
    console.error('Mermaid failed to load; keeping source code fallback.', error);
    return;
  }

  const diagrams = blocks.map(createDiagram);
  let renderVersion = 0;

  const renderAll = async () => {
    const version = ++renderVersion;
    const dark = document.documentElement.dataset.theme === 'dark';
    const colors = palette(dark);

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      suppressErrorRendering: true,
      theme: 'base',
      fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      themeVariables: {
        background: colors.raised,
        primaryColor: colors.signalSoft,
        primaryTextColor: colors.ink,
        primaryBorderColor: colors.signal,
        secondaryColor: colors.raised,
        secondaryTextColor: colors.ink,
        secondaryBorderColor: colors.line,
        tertiaryColor: colors.paper,
        tertiaryTextColor: colors.ink,
        tertiaryBorderColor: colors.line,
        lineColor: colors.softInk,
        textColor: colors.ink,
        edgeLabelBackground: colors.paper,
        clusterBkg: colors.raised,
        clusterBorder: colors.line,
        noteBkgColor: colors.raised,
        noteTextColor: colors.ink,
        noteBorderColor: colors.warm,
      },
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true,
      },
    });

    for (const [index, diagram] of diagrams.entries()) {
      diagram.canvas.setAttribute('aria-busy', 'true');
      diagram.figure.classList.remove('is-error');
      diagram.status.hidden = false;
      diagram.status.textContent = '正在渲染图表…';

      try {
        const id = `mermaid-${version}-${index}`;
        const { svg, bindFunctions } = await mermaid.render(id, diagram.source);
        if (version !== renderVersion) return;
        diagram.canvas.innerHTML = svg;
        diagram.canvas.querySelector('svg')?.setAttribute('aria-hidden', 'true');
        bindFunctions?.(diagram.canvas);
        diagram.canvas.setAttribute('aria-busy', 'false');
        diagram.status.hidden = true;
      } catch (error) {
        if (version !== renderVersion) return;
        diagram.canvas.replaceChildren();
        diagram.canvas.setAttribute('aria-busy', 'false');
        diagram.figure.classList.add('is-error');
        diagram.status.hidden = false;
        diagram.status.textContent = '图表渲染失败，请展开下方源码查看。';
        const details = diagram.figure.querySelector('details');
        if (details instanceof HTMLDetailsElement) details.open = true;
        console.error('Mermaid diagram failed to render.', error);
      }
    }
  };

  await renderAll();

  const themeObserver = new MutationObserver((mutations) => {
    if (mutations.some((mutation) => mutation.attributeName === 'data-theme')) {
      void renderAll();
    }
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });
}
