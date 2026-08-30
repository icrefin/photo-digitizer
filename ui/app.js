/* Photo Digitizer frontend. Vanilla JS against the Tauri global API. */

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

const state = {
  dir: null,
  files: [],
  photos: new Map(),   // id -> meta
  sheets: new Map(),   // file -> { b64, pageW, pageH }
  editing: null,       // { id, file, quad: [[x,y]x4] in preview px }
  selected: new Set(), // ids
  detected: new Set(), // file names processed
  activeFile: null,
  busy: false,
};

/* ---------- helpers ---------- */

function toast(msg, ms = 3200) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), ms);
}

function setProgress(text, frac) {
  const wrap = $("progressWrap");
  if (text == null) {
    wrap.classList.add("hidden");
    $("btnCancel").classList.add("hidden");
    state.busy = false;
    return;
  }
  wrap.classList.remove("hidden");
  $("progressText").textContent = text;
  $("progressPct").textContent = frac == null ? "" : `${Math.round(frac * 100)}%`;
  $("progressFill").style.width = `${Math.round((frac ?? 0) * 100)}%`;
}

function showCancel(show) {
  $("btnCancel").classList.toggle("hidden", !show);
  state.busy = show;
}

function dims(m) {
  return `${m.width}×${m.height}`;
}

function dimsAfter(m) {
  return m.enh_width ? `${m.enh_width}×${m.enh_height}` : null;
}

/* ---------- file list ---------- */

async function loadFolder(dir) {
  state.dir = dir;
  const files = await invoke("list_scans", { dir });
  state.files = files;
  state.photos.clear();
  state.sheets.clear();
  state.selected.clear();
  state.detected.clear();
  state.activeFile = null;
  $("fileCount").textContent = files.length ? `${files.length}` : "";
  $("currentFile").textContent = dir;
  $("btnDetectAll").disabled = !files.length;
  renderFileList();
  renderSplit();
}

function renderFileList() {
  const ul = $("fileList");
  ul.innerHTML = "";
  for (const f of state.files) {
    const li = document.createElement("li");
    li.dataset.file = f;
    if (state.detected.has(f)) li.classList.add("done");
    if (f === state.activeFile) li.classList.add("active");
    const dot = document.createElement("span");
    dot.className = "dot";
    const name = document.createElement("span");
    name.textContent = f;
    name.title = f;
    li.append(dot, name);
    li.onclick = () => selectFile(f, true);
    ul.appendChild(li);
  }
}

/* ---------- detection ---------- */

function selectFile(file, detectIfMissing) {
  state.activeFile = file;
  renderFileList();
  $("currentFile").textContent = file;
  renderSplit();
  renderGrid();
  if (detectIfMissing && !state.detected.has(file)) {
    detectOne(file);
  }
}

function renderSplit() {
  const sheet = state.activeFile ? state.sheets.get(state.activeFile) : null;
  if (sheet) {
    $("sheetImg").src = `data:image/jpeg;base64,${sheet.b64}`;
    $("sheetImg").classList.remove("hidden");
    $("sheetWrap").classList.remove("hidden");
    $("sheetHint").classList.add("hidden");
    if (state.editing) drawQuadOverlay();
  } else {
    $("sheetImg").removeAttribute("src");
    $("sheetImg").classList.add("hidden");
    $("sheetWrap").classList.add("hidden");
    $("sheetHint").classList.remove("hidden");
  }
}

async function detectOne(file) {
  try {
    setProgress(`Detecting photos in ${file}…`, null);
    const { sheet_b64, page_w, page_h, photos } = await invoke("detect_one", { dir: state.dir, file });
    state.detected.add(file);
    state.sheets.set(file, { b64: sheet_b64, pageW: page_w, pageH: page_h });
    for (const m of photos) state.photos.set(m.id, m);
    setProgress(null);
    renderFileList();
    renderSplit();
    renderGrid();
    if (!photos.length) toast(`${file}: no photos found`);
  } catch (e) {
    setProgress(null);
    toast(`Detection failed: ${e}`);
  }
}

async function detectAll() {
  $("btnDetectAll").disabled = true;
  showCancel(true);
  try {
    const unlisten = await listen("file-detected", (ev) => {
      const { file, done, total, sheet_b64, page_w, page_h, photos } = ev.payload;
      state.detected.add(file);
      state.sheets.set(file, { b64: sheet_b64, pageW: page_w, pageH: page_h });
      for (const m of photos) state.photos.set(m.id, m);
      setProgress(`Detecting ${done}/${total} — ${file}`, done / total);
      renderFileList();
      if (!state.activeFile) {
        selectFile(file, false);
      } else {
        renderSplit();
        renderGrid();
      }
    });
    const n = await invoke("detect_all", { dir: state.dir });
    unlisten();
    setProgress(null);
    toast(`Detected ${state.photos.size} photos across ${n} sheets`);
  } catch (e) {
    setProgress(null);
    toast(`Detection failed: ${e}`);
  } finally {
    $("btnDetectAll").disabled = false;
  }
}

/* ---------- grid ---------- */

function photosInView() {
  return [...state.photos.values()].filter(
    (m) => !state.activeFile || m.source_file === state.activeFile
  );
}

function renderGrid() {
  const grid = $("grid");
  const list = photosInView();
  $("emptyState").classList.toggle("hidden", list.length > 0);
  $("btnSelectAll").disabled = !list.length;
  $("btnEnhance").disabled = !list.length;
  $("btnSave").disabled = !list.length;
  grid.innerHTML = "";

  for (const m of list) {
    const card = document.createElement("div");
    card.className = "card" + (state.selected.has(m.id) ? " selected" : "");
    card.dataset.id = m.id;

    const wrap = document.createElement("div");
    wrap.className = "thumb-wrap";
    const img = document.createElement("img");
    img.className = "thumb";
    img.src = m.enh_thumb
      ? `data:image/jpeg;base64,${m.enh_thumb}`
      : `data:image/jpeg;base64,${m.thumb}`;
    img.title = "Click to compare";
    wrap.appendChild(img);
    wrap.onclick = () => openCompare(m.id);

    if (m.enhanced) {
      const b = document.createElement("span");
      b.className = "badge ai";
      b.textContent = "AI ✦";
      card.appendChild(b);
    } else if (m.rotation || m.method !== "none") {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = m.rotation ? `${m.rotation}° ${m.method}` : m.method;
      card.appendChild(b);
    }

    const check = document.createElement("input");
    check.type = "checkbox";
    check.className = "check";
    check.checked = state.selected.has(m.id);
    check.onchange = () => {
      check.checked ? state.selected.add(m.id) : state.selected.delete(m.id);
      card.classList.toggle("selected", check.checked);
      $("btnEnhance").disabled = !state.selected.size;
      $("btnSave").disabled = !state.selected.size;
    };
    card.appendChild(check);

    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent =
      m.source_file.replace(/\.jpe?g$/i, "").slice(-14) + ` #${m.id.split("#")[1]}`;
    name.title = `${m.source_file} · ${m.id}`;
    const rot = document.createElement("div");
    rot.className = "rot";
    const bl = document.createElement("button");
    bl.textContent = "⟲";
    bl.title = "Rotate 90° counter-clockwise";
    bl.onclick = (e) => { e.stopPropagation(); rotate(m.id, -90); };
    const br = document.createElement("button");
    br.textContent = "⟳";
    br.title = "Rotate 90° clockwise";
    br.onclick = (e) => { e.stopPropagation(); rotate(m.id, 90); };
    const be = document.createElement("button");
    be.textContent = "✎";
    be.title = "Adjust the crop on the sheet";
    be.onclick = (e) => { e.stopPropagation(); enterCropEdit(m.id); };
    rot.append(bl, br, be);
    meta.append(name, rot);
    card.append(wrap, meta);
    grid.appendChild(card);
  }

  if (list.length) {
    $("currentFile").textContent =
      state.activeFile || `${state.detected.size} sheets · ${list.length} photos`;
  }
}

async function rotate(id, deg) {
  try {
    const meta = await invoke("rotate_photo", { id, deg });
    state.photos.set(id, meta);
    renderGrid();
  } catch (e) {
    toast(`Rotate failed: ${e}`);
  }
}

/* ---------- enhance ---------- */

async function enhance() {
  if (state.busy) return;
  if (!state.selected.size) return toast("Select photos first");
  const ids = [...state.selected];
  const upscale = $("optUpscale").checked;
  const colorize = $("optColorize").checked;
  const faces = $("optFaces").checked;
  if (!upscale && !colorize && !faces) return toast("Pick at least one enhancement");
  $("btnEnhance").disabled = true;
  showCancel(true);
  try {
    const unlisten = await listen("enhance-progress", (ev) => {
      const { id, done, total, stage } = ev.payload;
      const label = `[${done}/${total}] ${id.split("#")[0].slice(-10)}#${id.split("#")[1]} — ${stage}`;
      $("enhStatus").textContent = label;
      setProgress(label, total ? done / total : null);
      showCancel(true);
    });
    const unlisten2 = await listen("photo-enhanced", (ev) => {
      state.photos.set(ev.payload.id, ev.payload.meta);
      renderGrid();
    });
    const [, cancelled] = await invoke("enhance_photos", { ids, upscale, colorize, faces });
    unlisten();
    unlisten2();
    setProgress(null);
    if (cancelled) {
      $("enhStatus").textContent = "Cancelled.";
      toast("Enhancement cancelled");
    } else {
      $("enhStatus").textContent = `Done — ${ids.length} photo(s) enhanced.`;
      const last = state.photos.get(ids[ids.length - 1]);
      if (last?.enhanced) openCompare(last.id);
    }
    renderGrid();
  } catch (e) {
    setProgress(null);
    $("enhStatus").textContent = "";
    toast(`Enhance failed: ${e}`);
  } finally {
    $("btnEnhance").disabled = false;
  }
}

/* ---------- compare modal: fixed columns, wheel zoom ---------- */

const panes = {
  before: { el: null, img: null, z: 1, tx: 0, ty: 0 },
  after: { el: null, img: null, z: 1, tx: 0, ty: 0 },
};

function applyZoom(pane) {
  const p = panes[pane];
  p.img.style.transform = `translate(${p.tx}px, ${p.ty}px) scale(${p.z})`;
}

function resetZoom(pane) {
  const p = panes[pane];
  p.z = 1;
  p.tx = 0;
  p.ty = 0;
  applyZoom(pane);
}

function setupPane(key) {
  const p = panes[key];
  p.el = $(key === "before" ? "paneBefore" : "paneAfter");
  p.img = key === "before" ? $("imgBefore") : $("imgAfter");

  p.el.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = p.el.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    // fine steps: ~1% per trackpad tick, ~8% per mouse notch, capped
    const k = Math.min(1.08, Math.max(1 / 1.08, Math.exp(-e.deltaY * 0.0008)));
    const nz = Math.min(16, Math.max(0.5, p.z * k));
    const kk = nz / p.z;
    // keep the point under the cursor stationary
    p.tx = cx - (cx - p.tx) * kk;
    p.ty = cy - (cy - p.ty) * kk;
    p.z = nz;
    applyZoom(key);
  }, { passive: false });

  let dragging = false;
  let lx = 0;
  let ly = 0;
  p.el.addEventListener("mousedown", (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    p.el.style.cursor = "grabbing";
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    p.tx += e.clientX - lx;
    p.ty += e.clientY - ly;
    lx = e.clientX;
    ly = e.clientY;
    applyZoom(key);
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    p.el.style.cursor = "grab";
  });
  p.el.addEventListener("dblclick", () => resetZoom(key));
}

function openCompare(id) {
  const m = state.photos.get(id);
  if (!m) return;
  $("modalTitle").textContent = m.id;
  const afterDims = dimsAfter(m);
  $("modalInfo").textContent = afterDims
    ? `original ${dims(m)}  →  enhanced ${afterDims}`
    : `${dims(m)} · run Enhance to see AI output here`;

  $("imgBefore").src = `data:image/jpeg;base64,${m.thumb}`;
  if (m.enh_thumb) {
    $("imgAfter").src = `data:image/jpeg;base64,${m.enh_thumb}`;
    $("afterEmpty").classList.add("hidden");
  } else {
    $("imgAfter").removeAttribute("src");
    $("afterEmpty").classList.remove("hidden");
  }
  resetZoom("before");
  resetZoom("after");
  $("modal").classList.remove("hidden");
}

/* ---------- manual crop editor ---------- */

let dragCorner = -1;

function enterCropEdit(id) {
  const m = state.photos.get(id);
  if (!m || !m.quad || m.quad.length !== 8) return toast("No quad available for this photo");
  if (state.activeFile !== m.source_file) {
    selectFile(m.source_file, false);
  }
  if (!state.detected.has(m.source_file)) return toast("Detect the sheet first");
  const sheet = state.sheets.get(m.source_file);
  const img = $("sheetImg");
  const applyMapping = () => {
    const { pageW, pageH } = sheet;
    const sx = img.naturalWidth / pageW;
    const sy = img.naturalHeight / pageH;
    state.editing = {
      id,
      file: m.source_file,
      quad: [
        [m.quad[0] * sx, m.quad[1] * sy],
        [m.quad[2] * sx, m.quad[3] * sy],
        [m.quad[4] * sx, m.quad[5] * sy],
        [m.quad[6] * sx, m.quad[7] * sy],
      ],
    };
    overlayNodes = null;
    $("cropText").textContent = `Drag the corners of #${id.split("#")[1]} to adjust the crop, then apply`;
    $("cropTools").classList.remove("hidden");
    drawQuadOverlay();
  };
  if (img.complete && img.naturalWidth) applyMapping();
  else img.onload = applyMapping;
  const reposition = () => {
    if (state.editing) {
      positionQuadOverlay();
      drawQuadOverlay();
    }
  };
  img.addEventListener("load", reposition, { once: true });
}

function exitCropEdit() {
  state.editing = null;
  overlayNodes = null;
  $("cropTools").classList.add("hidden");
  $("quadOverlay").innerHTML = "";
}

let overlayNodes = null; // { svg, polygon, handles: [circle×4] }

// Pin the SVG to the img's live bounding box: immune to any wrapper layout
// offsets, so pointer→viewBox mapping is exact at every edge.
function positionQuadOverlay() {
  const img = $("sheetImg");
  const svg = $("quadOverlay");
  const wrap = $("sheetWrap").getBoundingClientRect();
  const r = img.getBoundingClientRect();
  // the SVG's containing block is #sheetWrap (position: relative), so
  // offsets must be computed against the wrap's rect, not the pane's
  svg.style.left = `${r.left - wrap.left}px`;
  svg.style.top = `${r.top - wrap.top}px`;
  svg.style.width = `${r.width}px`;
  svg.style.height = `${r.height}px`;
}
window.addEventListener("resize", () => {
  if (state.editing) positionQuadOverlay();
});

function drawQuadOverlay() {
  const e = state.editing;
  const img = $("sheetImg");
  const svg = $("quadOverlay");
  if (!e || !img.naturalWidth) return;
  positionQuadOverlay();
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  if (!overlayNodes) {
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("fill", "rgba(0,180,255,.15)");
    poly.setAttribute("stroke", "#00c8ff");
    poly.setAttribute("stroke-width", "3");
    svg.appendChild(poly);
    const handles = [];
    for (let i = 0; i < 4; i++) {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("r", "16");
      c.setAttribute("fill", "#00c8ff");
      c.setAttribute("stroke", "#fff");
      c.setAttribute("stroke-width", "3");
      c.setAttribute("class", "handle");
      c.dataset.corner = String(i);
      c.addEventListener("pointerdown", onHandleDown);
      svg.appendChild(c);
      handles.push(c);
    }
    overlayNodes = { svg, polygon: poly, handles };
  }
  updateOverlayPositions();
}

function updateOverlayPositions() {
  const e = state.editing;
  if (!e || !overlayNodes) return;
  overlayNodes.polygon.setAttribute(
    "points",
    e.quad.map((p) => p.join(",")).join(" ")
  );
  overlayNodes.handles.forEach((h, i) => {
    h.setAttribute("cx", e.quad[i][0]);
    h.setAttribute("cy", e.quad[i][1]);
  });
}

function onHandleDown(e) {
  e.preventDefault();
  const corner = Number(e.currentTarget.dataset.corner);
  const svg = $("quadOverlay");
  svg.setPointerCapture(e.pointerId);
  const move = (ev) => {
    if (!state.editing) return;
    const rect = svg.getBoundingClientRect();
    const vx = ((ev.clientX - rect.left) / rect.width) * imgNaturalW();
    const vy = ((ev.clientY - rect.top) / rect.height) * imgNaturalH();
    state.editing.quad[corner] = [
      Math.max(0, Math.min(imgNaturalW(), vx)),
      Math.max(0, Math.min(imgNaturalH(), vy)),
    ];
    updateOverlayPositions();
  };
  const up = () => {
    svg.removeEventListener("pointermove", move);
    svg.removeEventListener("pointerup", up);
    regenPreviewDebounced();
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerup", up);
}

// After a drag ends, refresh the baked preview so the drawn box reflects the
// adjusted crop boundary.
let regenTimer = null;
function regenPreviewDebounced() {
  clearTimeout(regenTimer);
  regenTimer = setTimeout(regenPreviewNow, 250);
}

async function regenPreviewNow() {
  const e = state.editing;
  if (!e) return;
  const quads = quadsForFile(e.file, e.id, e.quad);
  try {
    const b64 = await invoke("sheet_preview_with_quads", {
      dir: state.dir, file: e.file, quads,
    });
    if (state.editing && state.editing.file === e.file) {
      $("sheetImg").src = `data:image/jpeg;base64,${b64}`;
    } else if (state.activeFile === e.file) {
      const sheet = state.sheets.get(e.file);
      state.sheets.set(e.file, { ...sheet, b64 });
      renderSplit();
    }
  } catch (_) { /* preview refresh is best-effort */ }
}

/// All quads for a file sorted by photo index, with `editId`'s quad replaced.
function quadsForFile(file, editId, editQuad) {
  const metas = [...state.photos.values()]
    .filter((m) => m.source_file === file && m.quad && m.quad.length === 8)
    .sort((a, b) => Number(a.id.split("#")[1]) - Number(b.id.split("#")[1]));
  return metas.map((m) =>
    m.id === editId ? editQuad.flatMap((p) => [p[0], p[1]]) : m.quad
  );
}

function imgNaturalW() {
  return $("sheetImg").naturalWidth;
}
function imgNaturalH() {
  return $("sheetImg").naturalHeight;
}

async function applyCrop() {
  const e = state.editing;
  if (!e) return;
  const sheet = state.sheets.get(e.file);
  const k = sheet.pageW / imgNaturalW();
  const quad = e.quad.flatMap((p) => [p[0] * k, p[1] * k]);
  try {
    setProgress("Re-cropping…", null);
    const meta = await invoke("re_extract_photo", {
      dir: state.dir, file: e.file, id: e.id, quad,
    });
    state.photos.set(e.id, meta);
    // refresh the baked preview: the drawn box now reflects the new crop
    try {
      const quads = quadsForFile(e.file, e.id, meta.quad);
      const b64 = await invoke("sheet_preview_with_quads", {
        dir: state.dir, file: e.file, quads,
      });
      const sheet = state.sheets.get(e.file);
      state.sheets.set(e.file, { ...sheet, b64 });
      state.editing = null;
      overlayNodes = null;
      renderSplit();
    } catch (_) { /* best effort */ }
    setProgress(null);
    toast("Crop updated");
    exitCropEdit();
    renderGrid();
  } catch (err) {
    setProgress(null);
    toast(`Re-crop failed: ${err}`);
  }
}

/* ---------- save ---------- */

async function save() {
  if (!state.selected.size) return toast("Select photos first");
  const ids = [...state.selected];
  const format = $("saveFormat").value;
  let outDir = null;
  try {
    outDir = await invoke("pick_folder");
  } catch (e) {
    toast(`Folder picker failed: ${e}`);
    return;
  }
  if (!outDir) return;
  try {
    const saved = await invoke("save_photos", { ids, outDir, format });
    $("saveStatus").textContent = `Saved ${saved.length} file(s) to ${outDir}`;
    toast(`Saved ${saved.length} photo(s) → ${outDir}`, 5000);
  } catch (e) {
    toast(`Save failed: ${e}`);
  }
}

/* ---------- wiring ---------- */

window.addEventListener("DOMContentLoaded", async () => {
  $("btnBrowse").onclick = async () => {
    const dir = await invoke("pick_folder");
    if (dir) {
      $("scanDir").value = dir;
      await loadFolder(dir);
    }
  };
  $("btnLoad").onclick = async () => {
    const dir = $("scanDir").value.trim();
    if (!dir) return toast("Choose a folder first");
    try {
      await loadFolder(dir);
    } catch (e) {
      toast(`Load failed: ${e}`);
    }
  };
  $("btnDetectAll").onclick = detectAll;
  $("btnSelectAll").onclick = () => {
    const list = photosInView();
    const allSelected = list.every((m) => state.selected.has(m.id));
    if (allSelected) state.selected.clear();
    else list.forEach((m) => state.selected.add(m.id));
    renderGrid();
  };
  $("btnEnhance").onclick = enhance;
  $("btnCancel").onclick = () => invoke("cancel_enhance").catch(() => {});
  $("btnCropApply").onclick = applyCrop;
  $("btnCropCancel").onclick = exitCropEdit;
  $("btnSave").onclick = save;
  $("modalClose").onclick = () => $("modal").classList.add("hidden");
  $("modal").onclick = (e) => {
    if (e.target === $("modal")) $("modal").classList.add("hidden");
  };
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("modal").classList.add("hidden");
  });

  setupPane("before");
  setupPane("after");

  try {
    const def = await invoke("default_dir");
    if (def) {
      $("scanDir").value = def;
      await loadFolder(def);
    }
  } catch (_) { /* ignore */ }
});
