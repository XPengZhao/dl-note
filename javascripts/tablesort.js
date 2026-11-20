function parseMetric(item) {
  const raw = String(item ?? "")
    .replace(/\u00a0/g, " ")
    .trim();
  if (
    raw === "" ||
    raw === "—" ||
    raw === "–" ||
    raw === "−" ||
    raw === "-"
  ) {
    return NaN;
  }
  const n = parseFloat(raw.replace(/%/g, "").replace(/,/g, ""));
  return Number.isFinite(n) ? n : NaN;
}

function isMetric(item) {
  const raw = String(item ?? "").trim();
  if (raw === "—" || raw === "–" || raw === "−") return true;
  return /^[+-]?\d+(\.\d+)?%?$/.test(raw.replace(/,/g, ""));
}

function compareMetric(a, b) {
  const na = parseMetric(a);
  const nb = parseMetric(b);
  const va = Number.isNaN(na) ? Number.NEGATIVE_INFINITY : na;
  const vb = Number.isNaN(nb) ? Number.NEGATIVE_INFINITY : nb;
  return vb - va;
}

// Epoch rows carry conditional rates computed from the original evaluation summary.
function addRateSwitch(table, labels, controls, onChange) {
  const first = labels.indexOf("pos-1");
  if (first < 0 || !Array.from({length: 7}, (_, i) => labels[first + i] === `pos-${i + 1}`).every(Boolean)) return false;
  const estimate = table.closest(".estimate-position-rates");
  const entries = [];
  for (const row of table.tBodies[0].rows) {
    const rates = row.cells[0].dataset.conditional?.split(" ");
    if (!estimate && rates?.length !== 7) return false;
    let previous = 100;
    for (let i = 0; i < 7; i++) {
      const cell = row.cells[first + i];
      const cumulative = cell.textContent;
      let conditional = rates?.[i];
      if (conditional === undefined) {
        const value = /^\d+(?:\.\d+)?%$/.test(cumulative.trim()) ? parseFloat(cumulative) : NaN;
        conditional = previous > 0 && Number.isFinite(value)
          ? (100 * value / previous).toFixed(2) + "%" : "—";
        previous = value;
      }
      entries.push({ cell, cumulative, conditional });
    }
  }
  const group = document.createElement("div");
  group.className = "position-rate-switch";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "位置接受率显示方式");
  for (const [mode, label] of [["cumulative", "累计接受率"], ["conditional", "条件接受率"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    if (estimate && mode === "conditional") {
      button.title = "按表中已舍入的累计接受率估算";
      button.setAttribute("aria-label", "条件接受率（按表中累计值估算）");
    }
    button.dataset.mode = mode;
    button.setAttribute("aria-pressed", String(mode === "cumulative"));
    button.addEventListener("click", () => {
      if (button.getAttribute("aria-pressed") === "true") return;
      document.querySelectorAll(".position-rate-switch").forEach(peer => {
        peer.dispatchEvent(new CustomEvent("ratechange", { detail: mode }));
      });
    });
    group.append(button);
  }
  group.addEventListener("ratechange", ({ detail: mode }) => {
    entries.forEach(entry => { entry.cell.textContent = entry[mode]; });
    for (const button of group.children) {
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    }
    onChange(first);
  });
  controls.append(group);
  return true;
}

// Each selected header is a sort key, in the order it was selected.
// Cycling a key's direction keeps its priority; removing it promotes later keys.
function enableMultiSort(table) {
  if (table.dataset.multiSortReady === "true" || !table.tHead) return;
  table.dataset.multiSortReady = "true";
  const wrapper = table.closest(".js-multisort");
  const headers = Array.from(table.tHead.rows[table.tHead.rows.length - 1].cells);
  const collator = new Intl.Collator(document.documentElement.lang || "zh", {
    numeric: true, sensitivity: "base",
  });
  let keys = [];
  let baseRows = Array.from(table.tBodies, (body) => Array.from(body.rows));
  const controls = document.createElement("div");
  controls.className = "table-sort-controls";
  controls.hidden = true;
  const status = document.createElement("span");
  status.className = "table-sort-status";
  status.setAttribute("aria-live", "polite");
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "table-sort-reset";
  reset.textContent = "清除排序";
  controls.append(status, reset);
  wrapper.prepend(controls);

  const labels = headers.map((header) => header.textContent.trim());
  const buttons = headers.map((header, column) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "table-sort-header";
    while (header.firstChild) button.append(header.firstChild);
    const badge = document.createElement("span");
    badge.className = "table-sort-badge";
    badge.setAttribute("aria-hidden", "true");
    badge.hidden = true;
    button.append(badge);
    header.append(button);
    button.addEventListener("click", () => {
      if (table.dataset.rowDragging === "true") return;
      const index = keys.findIndex((key) => key.column === column);
      if (index === -1) {
        keys.push({ column, direction: "ascending" });
      } else if (keys[index].direction === "ascending") {
        keys[index].direction = "descending";
      } else {
        keys.splice(index, 1);
      }
      applySort();
    });
    return { button, badge };
  });

  const updateHeaders = (manual = false) => {
    headers.forEach((header, column) => {
      const priority = keys.findIndex((key) => key.column === column);
      const { button, badge } = buttons[column];
      header.removeAttribute("aria-sort");
      badge.hidden = priority === -1;
      if (priority === -1) {
        delete header.dataset.sortPriority;
        delete header.dataset.sortDirection;
        button.setAttribute("aria-label", `${labels[column]}，未排序；点击设为升序`);
      } else {
        const key = keys[priority];
        const ascending = key.direction === "ascending";
        header.dataset.sortPriority = String(priority + 1);
        header.dataset.sortDirection = key.direction;
        // aria-sort belongs to the primary column; badges show all priorities.
        if (priority === 0) header.setAttribute("aria-sort", key.direction);
        badge.textContent = `${priority + 1} ${ascending ? "↑" : "↓"}`;
        button.setAttribute("aria-label", `${labels[column]}，${ascending ? "升序" : "降序"}，第 ${priority + 1} 优先级；点击${ascending ? "切换为降序" : "取消排序"}`);
      }
      button.title = button.getAttribute("aria-label");
    });
    controls.hidden = !wrapper.classList.contains("has-table-copy") && !hasRateSwitch && keys.length === 0 && !manual;
    reset.hidden = hasRateSwitch && keys.length === 0;
    reset.disabled = keys.length === 0;
    status.textContent = keys.map((key, index) =>
      `${index + 1}. ${labels[key.column]} ${key.direction === "ascending" ? "↑" : "↓"}`
    ).join(" → ") || (manual ? "手动顺序" : "");
  };
  const missing = (value) => /^(?:[—–−-])?$/.test(value.trim());
  const compareValues = (left, right, direction) => {
    const leftMissing = missing(left);
    const rightMissing = missing(right);
    // Missing measurements stay at the end in either direction.
    if (leftMissing || rightMissing) return Number(leftMissing) - Number(rightMissing);
    const comparison = isMetric(left) && isMetric(right)
      ? parseMetric(left) - parseMetric(right) : collator.compare(left, right);
    return direction === "ascending" ? comparison : -comparison;
  };
  const applySort = () => {
    Array.from(table.tBodies).forEach((body, bodyIndex) => {
      const original = baseRows[bodyIndex];
      const order = new Map(original.map((row, index) => [row, index]));
      const rows = Array.from(body.rows).sort((left, right) => {
        for (const key of keys) {
          const value = (row) => String(row.cells[key.column]?.dataset.sort ||
            row.cells[key.column]?.textContent || "").trim();
          const comparison = compareValues(value(left), value(right), key.direction);
          if (comparison) return comparison;
        }
        return order.get(left) - order.get(right);
      });
      rows.forEach((row) => body.append(row));
    });
    updateHeaders();
  };
  const hasRateSwitch = addRateSwitch(table, labels, controls, first => {
    if (keys.some(key => key.column >= first && key.column < first + 7)) applySort();
  });
  if (hasRateSwitch) wrapper.classList.add("has-position-rates");
  reset.addEventListener("click", () => {
    if (table.dataset.rowDragging === "true") return;
    keys = [];
    applySort();
  });
  table.addEventListener("roworderchange", () => {
    keys = [];
    baseRows = Array.from(table.tBodies, (body) => Array.from(body.rows));
    updateHeaders(true);
  });
  updateHeaders();
}

// Opt-in row ordering; handles stay inside the first cell so column sorting
// and the table's numeric alignment keep their original column indexes.
function enableRowReorder(table) {
  if (table.dataset.rowReorderReady === "true") return;
  table.dataset.rowReorderReady = "true";

  const clearSort = () => {
    table.dispatchEvent(new Event("roworderchange"));
    table.querySelectorAll("th[aria-sort]").forEach((header) => {
      header.removeAttribute("aria-sort");
    });
  };

  Array.from(table.tBodies).forEach((body) => {
    Array.from(body.rows).forEach((row) => {
      if (!row.cells.length) return;
      const handle = document.createElement("button");
      handle.type = "button";
      handle.className = "table-row-handle";
      const label = Array.from(row.cells).slice(0, 4)
        .map((cell) => cell.textContent.trim()).join("，");
      handle.setAttribute("aria-label", `移动行：${label}`);
      handle.title = "拖动调整行顺序；Esc 取消；也可用上下方向键移动";
      row.cells[0].prepend(handle);

      handle.addEventListener("keydown", (event) => {
        if (table.dataset.rowDragging === "true") return;
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        const neighbor = event.key === "ArrowUp"
          ? row.previousElementSibling : row.nextElementSibling;
        if (!neighbor) return;
        body.insertBefore(row, event.key === "ArrowUp"
          ? neighbor : neighbor.nextElementSibling);
        clearSort();
        handle.focus({ preventScroll: true });
      });

      handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || !event.isPrimary ||
            table.dataset.rowDragging === "true") return;
        table.dataset.rowDragging = "true";
        event.preventDefault();
        const pointerId = event.pointerId;
        const startX = event.clientX;
        const startY = event.clientY;
        const originalRows = Array.from(body.rows);
        const wrapper = table.closest(".js-row-reorder");
        const reducedMotion = window.matchMedia(
          "(prefers-reduced-motion: reduce)"
        ).matches;
        const animations = new Map();
        let preview = null;
        let previewLeft = 0;
        let pointerOffset = 0;
        let lastX = startX;
        let lastY = startY;
        let scrollFrame = 0;
        let changed = false;
        let finished = false;
        handle.focus({ preventScroll: true });
        table.setPointerCapture(pointerId);

        // Use layout positions, excluding the temporary slide animation.
        const layoutTop = (item) => {
          const transform = getComputedStyle(item).transform;
          const offset = transform === "none" ? 0
            : new DOMMatrixReadOnly(transform).m42;
          return item.getBoundingClientRect().top - offset;
        };
        const startPreview = () => {
          const bounds = row.getBoundingClientRect();
          const clip = wrapper.getBoundingClientRect();
          previewLeft = Math.max(bounds.left, clip.left);
          pointerOffset = startY - bounds.top;
          preview = document.createElement("div");
          preview.className = "md-typeset table-row-preview";
          preview.setAttribute("aria-hidden", "true");
          preview.style.width = `${Math.min(bounds.right, clip.right) - previewLeft}px`;
          preview.style.font = getComputedStyle(table).font;
          const frame = document.createElement("div");
          frame.className = wrapper.className;
          const cloneTable = table.cloneNode(false);
          cloneTable.removeAttribute("id");
          cloneTable.style.width = `${bounds.width}px`;
          cloneTable.style.marginLeft = `${bounds.left - previewLeft}px`;
          const cloneBody = document.createElement("tbody");
          const cloneRow = row.cloneNode(true);
          cloneRow.querySelectorAll("[id]").forEach((item) => item.removeAttribute("id"));
          cloneRow.querySelectorAll("button, a, input").forEach((item) => {
            item.setAttribute("tabindex", "-1");
          });
          Array.from(cloneRow.cells).forEach((cell, index) => {
            const width = `${row.cells[index].getBoundingClientRect().width}px`;
            cell.style.width = width;
            cell.style.minWidth = width;
            cell.style.maxWidth = width;
            cell.style.boxSizing = "border-box";
          });
          cloneBody.append(cloneRow);
          cloneTable.append(cloneBody);
          frame.append(cloneTable);
          preview.append(frame);
          document.body.append(preview);
          row.classList.add("table-row-placeholder");
          wrapper.classList.add("table-reordering");
        };
        const positionPreview = () => {
          preview.style.transform = `translate3d(${previewLeft + lastX - startX}px, ${lastY - pointerOffset}px, 0)`;
        };
        const updateSlot = () => {
          const candidates = Array.from(body.rows).filter((item) => item !== row);
          const reference = candidates.find((item) =>
            lastY < layoutTop(item) + item.getBoundingClientRect().height / 2
          ) || null;
          if (reference === row.nextElementSibling) return;
          const positions = new Map(candidates.map((item) => [item, item.getBoundingClientRect().top]));
          animations.forEach((animation) => animation.cancel());
          animations.clear();
          body.insertBefore(row, reference);
          changed = true;
          if (!reducedMotion) {
            candidates.forEach((item) => {
              const delta = positions.get(item) - item.getBoundingClientRect().top;
              if (Math.abs(delta) < 1) return;
              animations.set(item, item.animate([
                { transform: `translateY(${delta}px)` },
                { transform: "translateY(0)" },
              ], { duration: 160, easing: "ease-out" }));
            });
          }
        };
        const autoScroll = () => {
          if (finished) return;
          const edge = 48;
          const viewportHeight = document.documentElement.clientHeight;
          const speed = lastY < edge ? -Math.min(12, (edge - lastY) / 3)
            : lastY > viewportHeight - edge
              ? Math.min(12, (lastY - viewportHeight + edge) / 3) : 0;
          if (speed) {
            window.scrollBy(0, speed);
            updateSlot();
          }
          scrollFrame = requestAnimationFrame(autoScroll);
        };
        const move = (nextEvent) => {
          if (nextEvent.pointerId !== pointerId || finished) return;
          lastX = nextEvent.clientX;
          lastY = nextEvent.clientY;
          if (!preview) {
            if (Math.hypot(lastX - startX, lastY - startY) < 4) return;
            startPreview();
            scrollFrame = requestAnimationFrame(autoScroll);
          }
          positionPreview();
          updateSlot();
        };
        const finish = (endEvent) => {
          if (endEvent.pointerId !== pointerId || finished) return;
          finished = true;
          cancelAnimationFrame(scrollFrame);
          table.removeEventListener("pointermove", move);
          table.removeEventListener("pointerup", finish);
          table.removeEventListener("pointercancel", finish);
          table.removeEventListener("lostpointercapture", finish);
          document.removeEventListener("keydown", escape);
          const cancelled = endEvent.type !== "pointerup";
          if (cancelled) {
            originalRows.forEach((originalRow) => body.append(originalRow));
          } else if (changed) {
            clearSort();
          }
          if (table.hasPointerCapture(pointerId)) {
            table.releasePointerCapture(pointerId);
          }
          const cleanup = () => {
            table.dataset.rowDragging = "false";
            animations.forEach((animation) => animation.cancel());
            preview?.remove();
            row.classList.remove("table-row-placeholder");
            wrapper.classList.remove("table-reordering");
            handle.focus({ preventScroll: true });
          };
          if (preview && !cancelled && !reducedMotion) {
            const landing = row.getBoundingClientRect();
            preview.animate([
              { transform: preview.style.transform },
              { transform: `translate3d(${previewLeft}px, ${landing.top}px, 0)`, opacity: 0.5 },
            ], { duration: 120, easing: "ease-out", fill: "forwards" })
              .finished.then(cleanup, cleanup);
          } else {
            cleanup();
          }
        };
        const escape = (keyEvent) => {
          if (keyEvent.key !== "Escape") return;
          keyEvent.preventDefault();
          finish({ pointerId, type: "pointercancel" });
        };
        table.addEventListener("pointermove", move);
        table.addEventListener("pointerup", finish);
        table.addEventListener("pointercancel", finish);
        table.addEventListener("lostpointercapture", finish);
        document.addEventListener("keydown", escape);
      });
    });
  });
}

function enableTableCopy(figure) {
  const table = figure.querySelector("table");
  if (!table || figure.querySelector(".table-copy")) return;
  const button = document.createElement("button");
  button.className = "table-copy";
  button.type = "button";
  button.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path/></svg>';
  const icon = button.querySelector("path");
  const copyPath = "M8 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v3 M10 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z";
  const show = (label, path) => {
    button.title = label;
    button.setAttribute("aria-label", label);
    icon.setAttribute("d", path);
  };
  show("复制表格", copyPath);
  let timer;
  button.addEventListener("click", async () => {
    const text = Array.from(table.rows, row => Array.from(row.cells, cell => {
      const copy = cell.cloneNode(true);
      copy.querySelectorAll(".table-row-handle, .table-sort-badge").forEach(el => el.remove());
      return copy.textContent.trim().replace(/\s+/g, " ");
    }).join("\t")).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      show("已复制", "m4 12 5 5L20 6");
    } catch {
      show("复制失败", "M12 4v10m0 4v1");
    }
    clearTimeout(timer);
    timer = setTimeout(() => show("复制表格", copyPath), 1800);
  });
  const controls = figure.querySelector(".table-sort-controls");
  if (controls) {
    controls.append(button);
    controls.hidden = false;
    controls.parentElement.classList.add("has-table-copy");
  } else {
    figure.prepend(button);
  }
}

document$.subscribe(() => {
  document.querySelectorAll("figure.dspark-table-figure > figcaption").forEach((caption, index) => {
    let label = caption.querySelector(".table-caption-number");
    if (!label) {
      label = document.createElement("span");
      label.className = "table-caption-number";
      caption.prepend(label);
    }
    label.textContent = `Table ${index + 1}: `;
  });
  // Pair colors follow sampling settings even after sorting or dragging rows.
  document.querySelectorAll(".dspark-paired table").forEach((table) => {
    const groups = new Map();
    Array.from(table.tBodies).forEach((body) => {
      Array.from(body.rows).forEach((row) => {
        if (row.dataset.pairTone !== undefined) return;
        const key = JSON.stringify(Array.from(row.cells).slice(0, 4)
          .map((cell) => cell.textContent.trim()));
        if (!groups.has(key)) groups.set(key, groups.size % 2);
        row.dataset.pairTone = String(groups.get(key));
      });
    });
  });
  document.querySelectorAll(".js-multisort table").forEach(enableMultiSort);
  document.querySelectorAll(".js-row-reorder table").forEach(enableRowReorder);
  document.querySelectorAll("figure.dspark-table-figure").forEach(enableTableCopy);
  if (typeof Tablesort === "undefined") return;
  if (!Tablesort._metricExtended) {
    Tablesort.extend("metric", isMetric, compareMetric);
    Tablesort._metricExtended = true;
  }
  document.querySelectorAll(".js-sortable-table table").forEach((table) => {
    if (table.closest(".js-multisort") || table.dataset.tablesortReady === "true") return;
    table.dataset.tablesortReady = "true";
    new Tablesort(table);
  });
});
