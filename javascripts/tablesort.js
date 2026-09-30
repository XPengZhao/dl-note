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

document$.subscribe(() => {
  if (typeof Tablesort === "undefined") return;
  if (!Tablesort._metricExtended) {
    Tablesort.extend("metric", isMetric, compareMetric);
    Tablesort._metricExtended = true;
  }
  document.querySelectorAll(".js-sortable-table table").forEach((table) => {
    if (table.dataset.tablesortReady === "true") return;
    table.dataset.tablesortReady = "true";
    new Tablesort(table);
  });
});
