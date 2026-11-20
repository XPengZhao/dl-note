(() => {
  const SKIP_TAGS = new Set([
    "CODE",
    "KBD",
    "PRE",
    "SAMP",
    "SCRIPT",
    "STYLE",
    "SVG",
    "TEXTAREA",
  ]);
  const SKIP_CLASSES = ["arithmatex", "mermaid", "mermaid-modal"];
  const INLINE_CODE_TAGS = new Set(["CODE", "KBD", "SAMP"]);
  const STRUCTURAL_TAGS = new Set([
    "P",
    "DIV",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "UL",
    "OL",
    "LI",
    "TABLE",
    "THEAD",
    "TBODY",
    "TFOOT",
    "TR",
    "TD",
    "TH",
    "PRE",
    "BLOCKQUOTE",
    "HR",
    "SECTION",
    "ARTICLE",
    "FIGURE",
    "FIGCAPTION",
    "DL",
    "DT",
    "DD",
  ]);
  const CJK =
    "\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3100-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff";
  const AN = "A-Za-z0-9";
  const GAP_CLASS = "cjk-latin-space";
  const GAP_SPACES = "[ \\t\\u00a0\\u2005]*";
  const boundary = new RegExp(
    `([${CJK}])${GAP_SPACES}([${AN}])|([${AN}])${GAP_SPACES}([${CJK}])`,
    "g",
  );
  const trimEndGap = new RegExp(`[ \\t\\u00a0\\u2005]+$`);
  const trimStartGap = new RegExp(`^[ \\t\\u00a0\\u2005]+`);
  const onlyGapSpaces = new RegExp(`^[ \\t\\u00a0\\u2005]*$`);
  const endsWithCjk = new RegExp(`[${CJK}]$`);
  const endsWithAn = new RegExp(`[${AN}]$`);
  const startsWithCjk = new RegExp(`^[${CJK}]`);
  const startsWithAn = new RegExp(`^[${AN}]`);

  function isSkipped(el) {
    if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
    if (SKIP_TAGS.has(el.tagName)) return true;
    return SKIP_CLASSES.some((name) => el.classList.contains(name));
  }

  function isGap(node) {
    return node && node.nodeType === Node.ELEMENT_NODE && node.classList.contains(GAP_CLASS);
  }

  function isIgnorableSpace(node) {
    return node && node.nodeType === Node.TEXT_NODE && onlyGapSpaces.test(node.nodeValue || "");
  }

  function isStructural(node) {
    return node && node.nodeType === Node.ELEMENT_NODE && STRUCTURAL_TAGS.has(node.tagName);
  }

  function makeGap() {
    const span = document.createElement("span");
    span.className = GAP_CLASS;
    span.setAttribute("aria-hidden", "true");
    span.textContent = "\u200b";
    return span;
  }

  function edgeText(node, fromEnd) {
    if (!node) return "";
    if (isGap(node)) return "";
    if (node.nodeType === Node.TEXT_NODE) return node.nodeValue || "";
    if (node.nodeType !== Node.ELEMENT_NODE) return "";
    if (node.tagName === "BR") return "\n";
    if (isSkipped(node)) {
      return INLINE_CODE_TAGS.has(node.tagName) ? node.textContent || "" : "";
    }
    const kids = node.childNodes;
    if (fromEnd) {
      for (let i = kids.length - 1; i >= 0; i -= 1) {
        const text = edgeText(kids[i], true);
        if (text) return text;
      }
    } else {
      for (let i = 0; i < kids.length; i += 1) {
        const text = edgeText(kids[i], false);
        if (text) return text;
      }
    }
    return "";
  }

  function needsSpace(left, right) {
    if (!left || !right) return false;
    const l = left.replace(trimEndGap, "");
    const r = right.replace(trimStartGap, "");
    if (!l || !r) return false;
    return (
      (endsWithCjk.test(l) && startsWithAn.test(r)) ||
      (endsWithAn.test(l) && startsWithCjk.test(r))
    );
  }

  function splitTextNode(node) {
    const value = node.nodeValue;
    if (!value) return;
    boundary.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    let changed = false;
    while ((match = boundary.exec(value))) {
      changed = true;
      if (match.index > last) {
        frag.appendChild(document.createTextNode(value.slice(last, match.index)));
      }
      frag.appendChild(document.createTextNode(match[1] || match[3]));
      frag.appendChild(makeGap());
      frag.appendChild(document.createTextNode(match[2] || match[4]));
      last = match.index + match[0].length;
    }
    if (!changed) return;
    if (last < value.length) {
      frag.appendChild(document.createTextNode(value.slice(last)));
    }
    node.replaceWith(frag);
  }

  function processSiblings(el) {
    if (el.tagName === "TR" || el.tagName === "TABLE" || el.tagName === "THEAD" || el.tagName === "TBODY" || el.tagName === "TFOOT") {
      return;
    }
    const nodes = [...el.childNodes];
    let i = 0;
    while (i < nodes.length) {
      const left = nodes[i];
      if (isGap(left) || isIgnorableSpace(left) || (left.nodeType !== Node.TEXT_NODE && left.nodeType !== Node.ELEMENT_NODE)) {
        i += 1;
        continue;
      }
      let j = i + 1;
      while (j < nodes.length && (isGap(nodes[j]) || isIgnorableSpace(nodes[j]))) j += 1;
      if (j >= nodes.length) break;
      const right = nodes[j];
      if (isStructural(left) || isStructural(right)) {
        i = j;
        continue;
      }
      if (needsSpace(edgeText(left, true), edgeText(right, false))) {
        const alreadyGapped = nodes.slice(i + 1, j).some(isGap);
        if (!alreadyGapped) {
          for (let k = i + 1; k < j; k += 1) {
            if (isIgnorableSpace(nodes[k])) nodes[k].remove();
          }
          el.insertBefore(makeGap(), right);
        }
      }
      i = j;
    }
  }

  function process(el) {
    if (isSkipped(el)) return;
    for (const child of [...el.childNodes]) {
      if (child.nodeType === Node.ELEMENT_NODE) process(child);
    }
    for (const child of [...el.childNodes]) {
      if (child.nodeType === Node.TEXT_NODE) splitTextNode(child);
    }
    processSiblings(el);
  }

  function init() {
    document.querySelectorAll(".md-typeset").forEach(process);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
