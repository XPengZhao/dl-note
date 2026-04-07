const MERMAID_MODAL_ID = "mermaid-modal";
const MERMAID_SCALE_STEP = 0.15;
const MERMAID_MIN_SCALE = 0.5;
const MERMAID_MAX_SCALE = 2.5;

function ensureMermaidModal() {
  let modal = document.getElementById(MERMAID_MODAL_ID);
  if (modal) return modal;

  modal = document.createElement("div");
  modal.id = MERMAID_MODAL_ID;
  modal.className = "mermaid-modal";
  modal.innerHTML = `
    <div class="mermaid-modal__backdrop"></div>
    <div class="mermaid-modal__panel" role="dialog" aria-modal="true" aria-label="Diagram viewer">
      <div class="mermaid-modal__toolbar">
        <button class="mermaid-modal__button" data-action="zoom-out" type="button" aria-label="Zoom out">-</button>
        <span class="mermaid-modal__label">100%</span>
        <button class="mermaid-modal__button" data-action="zoom-in" type="button" aria-label="Zoom in">+</button>
        <button class="mermaid-modal__button" data-action="reset" type="button" aria-label="Reset zoom">Reset</button>
        <button class="mermaid-modal__button mermaid-modal__close" data-action="close" type="button" aria-label="Close diagram">Close</button>
      </div>
      <div class="mermaid-modal__viewport">
        <div class="mermaid-modal__content"></div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  const content = modal.querySelector(".mermaid-modal__content");
  const label = modal.querySelector(".mermaid-modal__label");

  function applyScale(scale) {
    const svg = content.querySelector("svg");
    if (!svg) return;
    modal.dataset.scale = String(scale);
    svg.style.transform = `scale(${scale})`;
    svg.style.transformOrigin = "top left";
    label.textContent = `${Math.round(scale * 100)}%`;
  }

  function closeModal() {
    modal.classList.remove("is-open");
    content.innerHTML = "";
    modal.dataset.scale = "1";
    label.textContent = "100%";
  }

  function updateScale(delta) {
    const current = Number(modal.dataset.scale || "1");
    applyScale(Math.max(MERMAID_MIN_SCALE, Math.min(MERMAID_MAX_SCALE, current + delta)));
  }

  modal.addEventListener("click", (event) => {
    const action = event.target?.dataset?.action;
    if (event.target.classList.contains("mermaid-modal__backdrop") || action === "close") {
      closeModal();
      return;
    }
    if (action === "zoom-in") updateScale(MERMAID_SCALE_STEP);
    if (action === "zoom-out") updateScale(-MERMAID_SCALE_STEP);
    if (action === "reset") applyScale(1);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.classList.contains("is-open")) {
      closeModal();
    }
  });

  modal.openWith = (sourceNode) => {
    const svg = sourceNode.querySelector("svg");
    if (!svg) return;
    content.innerHTML = "";
    const clone = svg.cloneNode(true);
    clone.removeAttribute("style");
    content.appendChild(clone);
    modal.classList.add("is-open");
    applyScale(1);
  };

  return modal;
}

function enhanceMermaid(node) {
  if (node.dataset.enhanced === "true") return;

  const wrapper = document.createElement("div");
  wrapper.className = "mermaid-figure";

  const toolbar = document.createElement("div");
  toolbar.className = "mermaid-figure__toolbar";

  const button = document.createElement("button");
  button.className = "mermaid-figure__button";
  button.type = "button";
  button.innerHTML = `
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3M5 5h6v2H7v10h10v-4h2v6H5V5Z"></path>
    </svg>
  `;
  button.setAttribute("aria-label", "Open diagram in modal viewer");
  button.title = "Open diagram";

  toolbar.appendChild(button);

  node.parentNode.insertBefore(wrapper, node);
  wrapper.append(toolbar, node);

  const modal = ensureMermaidModal();
  const open = () => modal.openWith(node);

  button.addEventListener("click", (event) => {
    event.stopPropagation();
    open();
  });

  node.addEventListener("click", open);
  node.setAttribute("title", "Click to open diagram");
  node.dataset.enhanced = "true";
}

document$.subscribe(async () => {
  if (typeof mermaid === "undefined") return;

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
  });

  const nodes = Array.from(
    document.querySelectorAll('.mermaid:not([data-rendered="true"])'),
  );

  if (nodes.length === 0) return;

  await mermaid.run({ nodes });

  nodes.forEach((node) => {
    node.dataset.rendered = "true";
    enhanceMermaid(node);
  });
});
