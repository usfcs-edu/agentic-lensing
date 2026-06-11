window.MathJax = {
  // textmacros: \texttt{...} in math otherwise typesets \_ with a literal
  // backslash (pandoc escapes underscores inside \texttt).
  loader: { load: ["[tex]/textmacros"] },
  tex: {
    packages: { "[+]": ["textmacros"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    // Number \begin{equation} blocks and resolve \label/\ref/\eqref in-page.
    tags: "ams",
    // Pandoc macro expansion can leave \ensuremath inside math strings.
    macros: { ensuremath: ["#1", 1] }
  },
  options: {
    ignoreHtmlClass: ".*|",
    // "math": pandoc emits <span class="math inline"> for math inside raw
    // HTML tables/figcaptions; arithmatex covers the markdown-body math.
    // "md-ellipsis": material wraps nav/TOC label text in this span, which
    // would re-match ignoreHtmlClass and leave heading math raw in the TOC.
    processHtmlClass: "arithmatex|math|md-ellipsis"
  }
};

document$.subscribe(() => {
  // Reset AMS labels/numbering on each (instant) navigation, otherwise
  // MathJax throws duplicate-label errors when revisiting a page.
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
