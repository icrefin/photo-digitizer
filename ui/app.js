/* Photo Digitizer frontend. Vanilla JS against the Tauri global API. */

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

function selectedRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value;
}

const state = {
  dir: null,
  files: [],
  photos: new Map(),   // id -> meta
  sheets: new Map(),   // file -> { b64, pageW, pageH }
  editing: null,       // { id, file, quad: [[x,y]x4] in preview px }
  selected: new Set(), // ids
  detected: new Set(), // file names processed
  activeFile: null,
  compareId: null,     // id shown in the compare modal
  busy: false,
};

/* ---------- language (English / 中文) ---------- */

let lang = localStorage.getItem("lang")
  || (navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en");

const I18N = {
  en: {
    folderPh: "Folder containing scanned sheets…",
    browse: "Browse…",
    loadEmpty: "Choose a folder first", loadFailed: "Load failed: {0}",
    scanSheets: "Scan sheets",
    selectAll: "Select all", detectAll: "Detect all sheets",
    cancel: "✕ Cancel",
    originalSheet: "Original sheet",
    applyCrop: "Apply crop", cancelCrop: "Cancel",
    sheetHint: "Select a sheet on the left to detect its photos.",
    empty1: "Detect a sheet to see its photos here.",
    empty2: "Each sheet is split into individual photos, straightened and trimmed.",
    enhance: "Enhance",
    panelDesc: "Runs locally — Real-ESRGAN ×4 for resolution, Zhang colorization for faded color.",
    optUpscale: "AI upscale", optUpscaleSub: "Real-ESRGAN ×4",
    optEngineFast: "Standard (Real-ESRGAN, seconds)",
    optEnginePhantom: "Phantom AI (diffusion, minutes)",
    optEngineUnavailable: "Phantom AI not installed — see sidecar/phantom/README.md",
    optColorize: "AI color", optColorizeSub: "colorization / restore",
    optFaces: "AI faces", optFacesSub: "GFPGAN v1.4 restoration",
    enhanceSelected: "Enhance selected",
    save: "Save",
    jpgOpt: "JPEG (small)", pngOpt: "PNG (lossless)",
    saveSelected: "Save selected…",
    tip: "Click a photo to compare original ↔ enhanced. Scroll to zoom each pane, drag to pan, double-click to reset. Use ⟲/⟳ to fix rotation and ✎ to adjust the crop by dragging its corners on the sheet — a manually cropped photo keeps its crop when sheets are re-detected, and can be reset to its original from the compare window. If two photos were detected as one, right-click the sheet and drag over one of them to extract it as a separate photo.",
    resetEnh: "Reset to original",
    compare: "Compare",
    tagOriginal: "original", tagEnhanced: "enhanced",
    afterEmpty: "No enhancement yet — select and run Enhance.",
    modalFoot: "scroll = zoom · drag = pan · double-click = reset",
    detectProgress: "Detecting photos in {0}…",
    noPhotos: "{0}: no photos found",
    detectFailed: "Detection failed: {0}",
    detectDone: "Detected {0} photos across {1} sheets",
    selectFirst: "Select photos first",
    noEnhOpt: "Pick at least one enhancement",
    cancelled: "Enhancement cancelled",
    enhancedDone: "Done — {0} photo(s) enhanced.",
    enhanceFailed: "Enhance failed: {0}",
    rotateFailed: "Rotate failed: {0}",
    needDetect: "Detect the sheet first",
    noQuad: "No quad available for this photo",
    cropTextEdit: "Drag the corners of #{0} to adjust the crop, then apply",
    recrop: "Re-cropping…",
    cropUpdated: "Crop updated",
    recropFailed: "Re-crop failed: {0}",
    drawHint: "Drag on the sheet to draw a crop box for a new photo",
    drawAdjust: "Drag the corners to adjust the new crop, then apply",
    addingPhoto: "Adding photo…",
    photoAdded: "Photo added",
    addFailed: "Add photo failed: {0}",
    finishCropFirst: "Finish the current crop edit first",
    selectSheetFirst: "Select a sheet first",
    noSheet: "No sheet shown",
    savedTo: "Saved {0} photo(s) → {1}",
    saveFailed: "Save failed: {0}",
    pickFailed: "Folder picker failed: {0}",
    resetDone: "Reset to original — you can enhance again anytime",
    resetFailed: "Reset failed: {0}",
    aboutBtn: "About",
    aboutVersion: "Version", aboutBuilt: "Built",
    aboutPlatform: "Platform", aboutClose: "OK",
    aboutOpenSource: "Open source",
    aboutCopyright: "Copyright",
    aboutFailed: "About failed: {0}",
    modalInfo: "original {0}  →  enhanced {1}",
    modalInfoNoEnh: "{0} · run Enhance to see AI output here",
    stageLoad: "loading AI models…",
    stageStartUpscale: "starting upscale…",
    stageUpscale: "upscaling {0}",
    stageColorize: "colorizing…",
    stageFaces: "restoring faces {0}",
    stagePhantom: "Phantom AI {0}",
    stageDone: "done",
    mFace: "face", mObject: "object", mNone: "none", mManual: "manual",
    rotCcw: "Rotate 90° counter-clockwise",
    rotCw: "Rotate 90° clockwise",
    adjustCrop: "Adjust the crop on the sheet",
    manualBadgeTip: "Manually cropped — auto-detection keeps this crop",
    clickCompare: "Click to compare",
  },
  zh: {
    folderPh: "包含扫描页的文件夹…",
    browse: "浏览…",
    loadEmpty: "请先选择文件夹", loadFailed: "加载失败：{0}",
    scanSheets: "扫描页",
    selectAll: "全选", detectAll: "检测全部页",
    cancel: "✕ 取消",
    originalSheet: "原始扫描页",
    applyCrop: "应用裁剪", cancelCrop: "取消",
    sheetHint: "在左侧选择扫描页以检测其中的照片。",
    empty1: "检测扫描页后可在此查看照片。",
    empty2: "每页将被拆分为独立的照片，并自动摆正和修边。",
    enhance: "增强",
    panelDesc: "本地运行 — Real-ESRGAN ×4 提升分辨率，Zhang 方法为褪色照片修复色彩。",
    optUpscale: "AI 放大", optUpscaleSub: "Real-ESRGAN ×4",
    optEngineFast: "标准（Real-ESRGAN，秒级）",
    optEnginePhantom: "Phantom AI（扩散模型，分钟级）",
    optEngineUnavailable: "Phantom AI 未安装 — 见 sidecar/phantom/README.md",
    optColorize: "AI 上色", optColorizeSub: "上色 / 修复",
    optFaces: "AI 人脸", optFacesSub: "GFPGAN v1.4 修复",
    enhanceSelected: "增强选中项",
    save: "保存",
    jpgOpt: "JPEG（小）", pngOpt: "PNG（无损）",
    saveSelected: "保存选中…",
    tip: "点击照片可对比原图与增强效果。滚动缩放，拖动平移，双击复位。使用 ⟲/⟳ 修正旋转，✎ 在扫描页上拖动角点调整裁剪 — 手动裁剪过的照片在重新检测时会保留该裁剪，并可在对比窗口中恢复原图。若自动检测将两张照片误合为一张，可在扫描页上右键并拖动框选其中一张，将其单独提取出来。",
    resetEnh: "恢复原图",
    compare: "对比",
    tagOriginal: "原图", tagEnhanced: "增强后",
    afterEmpty: "尚未增强 — 请选择照片并运行增强。",
    modalFoot: "滚动 = 缩放 · 拖动 = 平移 · 双击 = 复位",
    detectProgress: "正在检测 {0} 中的照片…",
    noPhotos: "{0}：未找到照片",
    detectFailed: "检测失败：{0}",
    detectDone: "共检测到 {0} 张照片，来自 {1} 个扫描页",
    selectFirst: "请先选择照片",
    noEnhOpt: "请至少选择一种增强方式",
    cancelled: "增强已取消",
    enhancedDone: "完成 — 已增强 {0} 张照片。",
    enhanceFailed: "增强失败：{0}",
    rotateFailed: "旋转失败：{0}",
    needDetect: "请先检测该扫描页",
    noQuad: "该照片没有可调整的裁剪区域",
    cropTextEdit: "拖动 # {0} 的角点调整裁剪范围，然后应用",
    recrop: "正在重新裁剪…",
    cropUpdated: "裁剪已更新",
    recropFailed: "重新裁剪失败：{0}",
    drawHint: "在扫描页上拖动，为新照片绘制裁剪框",
    drawAdjust: "拖动角点调整新裁剪，然后应用",
    addingPhoto: "正在添加照片…",
    photoAdded: "照片已添加",
    addFailed: "添加照片失败：{0}",
    finishCropFirst: "请先完成当前的裁剪操作",
    selectSheetFirst: "请先选择扫描页",
    noSheet: "没有显示扫描页",
    savedTo: "已保存 {0} 张照片 → {1}",
    saveFailed: "保存失败：{0}",
    pickFailed: "选择文件夹失败：{0}",
    resetDone: "已恢复原图 — 可随时再次增强",
    resetFailed: "恢复失败：{0}",
    aboutBtn: "关于",
    aboutVersion: "版本", aboutBuilt: "构建时间",
    aboutPlatform: "平台", aboutClose: "好的",
    aboutOpenSource: "开源组件",
    aboutCopyright: "版权",
    aboutFailed: "关于信息获取失败：{0}",
    modalInfo: "原图 {0}  →  增强后 {1}",
    modalInfoNoEnh: "{0} · 运行增强后可在此查看 AI 输出",
    stageLoad: "正在加载 AI 模型…",
    stageStartUpscale: "开始放大…",
    stageUpscale: "放大中 {0}",
    stageColorize: "上色中…",
    stageFaces: "修复人脸 {0}",
    stagePhantom: "Phantom AI {0}",
    stageDone: "完成",
    mFace: "人脸", mObject: "物体", mNone: "无", mManual: "手动",
    rotCcw: "逆时针旋转 90°",
    rotCw: "顺时针旋转 90°",
    adjustCrop: "在扫描页上调整裁剪",
    manualBadgeTip: "手动裁剪 — 自动检测将保留此裁剪",
    clickCompare: "点击对比",
  },
};

function t(key, ...args) {
  let s = (I18N[lang] && I18N[lang][key]) ?? I18N.en[key] ?? key;
  args.forEach((a, i) => { s = s.replace(new RegExp(`\\{${i}\\}`), a); });
  return s;
}

function tStage(stage) {
  if (stage == null) return "";
  if (stage.startsWith("loading AI models")) return t("stageLoad");
  if (stage.startsWith("starting upscale")) return t("stageStartUpscale");
  if (stage.startsWith("upscaling")) return t("stageUpscale", stage.replace(/^upscaling\s*/, "").trim());
  if (stage.startsWith("colorizing")) return t("stageColorize");
  if (stage.startsWith("restoring faces")) return t("stageFaces", stage.replace(/^restoring faces\s*/, "").trim());
  if (stage.startsWith("Phantom")) return t("stagePhantom", stage.replace(/^Phantom\s*/, "").trim());
  if (stage === "done") return t("stageDone");
  return stage;
}

function tMethod(m) {
  if (!m) return "";
  if (m === "manual") return t("mManual");
  const inner = /^manual\s*\(\s*(\w+)\s*\)$/.exec(m);
  const core = inner ? inner[1] : m;
  return (inner ? t("mManual") + " (" : "") + (t("m" + core[0].toUpperCase() + core.slice(1)) ?? core) + (inner ? ")" : "");
}

function applyLang() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  $("langSel").value = lang;
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of document.querySelectorAll("[data-i18n-ph]")) {
    el.placeholder = t(el.dataset.i18nPh);
  }
  if ($("modal").classList.contains("hidden")) $("modalTitle").textContent = t("compare");
  renderFileList();
  renderSplit();
  renderGrid();
  if (state.editing) {
    $("cropText").textContent = state.editing.id === null
      ? t("drawAdjust")
      : t("cropTextEdit", state.editing.id.split("#")[1]);
  }
}

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
    setProgress(t("detectProgress", file), null);
    const { sheet_b64, page_w, page_h, photos } = await invoke("detect_one", { dir: state.dir, file });
    state.detected.add(file);
    state.sheets.set(file, { b64: sheet_b64, pageW: page_w, pageH: page_h });
    for (const m of photos) state.photos.set(m.id, m);
    setProgress(null);
    renderFileList();
    renderSplit();
    renderGrid();
    if (!photos.length) toast(t("noPhotos", file));
  } catch (e) {
    setProgress(null);
    toast(t("detectFailed", e));
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
    toast(t("detectDone", state.photos.size, n));
  } catch (e) {
    setProgress(null);
    toast(t("detectFailed", e));
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
    img.title = t("clickCompare");
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
      b.textContent = m.rotation ? `${m.rotation}° ${tMethod(m.method)}` : tMethod(m.method);
      card.appendChild(b);
    }
    if (m.manual) {
      const b = document.createElement("span");
      b.className = "badge manual";
      b.textContent = "✎ crop";
      b.title = t("manualBadgeTip");
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
    bl.title = t("rotCcw");
    bl.onclick = (e) => { e.stopPropagation(); rotate(m.id, -90); };
    const br = document.createElement("button");
    br.textContent = "⟳";
    br.title = t("rotCw");
    br.onclick = (e) => { e.stopPropagation(); rotate(m.id, 90); };
    const be = document.createElement("button");
    be.textContent = "✎";
    be.title = t("adjustCrop");
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
    toast(t("rotateFailed", e));
  }
}

/* ---------- enhance engine (Phantom sidecar) ---------- */

async function initPhantomEngine() {
  const row = $("engineRow");
  const phantomRadio = document.querySelector('input[name="engine"][value="phantom"]');
  const fastRadio = document.querySelector('input[name="engine"][value="esrgan"]');
  const saved = localStorage.getItem("engine");
  if (saved === "phantom" && phantomRadio) {
    phantomRadio.checked = true;
    if (fastRadio) fastRadio.checked = false;
  }
  let available = false;
  try {
    const status = await invoke("phantom_status");
    available = !!status.available;
    if (!available) $("engineNote").textContent = t("optEngineUnavailable");
  } catch (_) { /* keep hidden */ }
  if (!available || !phantomRadio) {
    if (phantomRadio) phantomRadio.disabled = true;
    if (localStorage.getItem("engine") === "phantom") localStorage.setItem("engine", "esrgan");
    row.classList.add("hidden");
    return;
  }
  // Show the engine row only when upscaling is on.
  const syncRow = () => row.classList.toggle("hidden", !$("optUpscale").checked);
  $("optUpscale").addEventListener("change", syncRow);
  syncRow();
  document.querySelectorAll('input[name="engine"]').forEach((r) => {
    r.addEventListener("change", () => localStorage.setItem("engine", r.value));
  });
}

/* ---------- about & version ---------- */

// Projects this app is built on (licenses verified from each project's
// repository). Additional per-project license terms apply.
const OPEN_SOURCE = [
  ["Tauri", "Apache-2.0 / MIT"],
  ["OpenCV", "Apache-2.0"],
  ["ONNX Runtime", "MIT"],
  ["Real-ESRGAN", "BSD-3-Clause"],
  ["GFPGAN", "Apache-2.0"],
  ["YuNet (OpenCV face detection)", "Apache-2.0"],
  ["Zhang et al. colorization", "BSD-2-Clause"],
  ["Dreamoving Phantom (PASD)", "Apache-2.0"],
  ["Diffusers (PASD runtime)", "Apache-2.0"],
  ["Stable Diffusion 1.5 (model weights)", "CreativeML Open RAIL-M"],
];

let aboutInfo = null;

function fmtTs(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

async function openAbout() {
  if (!aboutInfo) {
    try {
      aboutInfo = await invoke("about_info");
    } catch (e) {
      toast(t("aboutFailed", e));
      return;
    }
  }
  $("aboutDlg").showModal();
}

async function initAbout() {
  try {
    aboutInfo = await invoke("about_info");
  } catch (_) { /* footer keeps bare name */ }
  if (aboutInfo) {
    $("appVerLabel").textContent = `${aboutInfo.name} · ${fmtTs(aboutInfo.build_ts)}`;
    $("aboutVersion").textContent = fmtTs(aboutInfo.build_ts);
    $("aboutBuilt").textContent = fmtTs(aboutInfo.build_ts);
    $("aboutPlatform").textContent = `${aboutInfo.platform} / ${aboutInfo.arch}`;
  }
  $("aboutLicenses").innerHTML = OPEN_SOURCE.map(
    ([name, lic]) => `<div class="lic-row"><span>${name}</span><b>${lic}</b></div>`
  ).join("");
  $("btnAbout").onclick = openAbout;
  $("aboutClose").onclick = () => $("aboutDlg").close();
  // Menu bar → Photo Digitizer → About Photo Digitizer opens the same dialog.
  try {
    await listen("show-about", openAbout);
  } catch (_) { /* not a Tauri context (tests) */ }
}

/* ---------- enhance ---------- */

async function enhance() {
  if (state.busy) return;
  if (!state.selected.size) return toast(t("selectFirst"));
  const ids = [...state.selected];
  const upscale = $("optUpscale").checked;
  const colorize = $("optColorize").checked;
  const faces = $("optFaces").checked;
  const engine = selectedRadio("engine") || "esrgan";
  if (!upscale && !colorize && !faces) return toast(t("noEnhOpt"));
  $("btnEnhance").disabled = true;
  showCancel(true);
  try {
    const unlisten = await listen("enhance-progress", (ev) => {
      const { id, done, total, stage } = ev.payload;
      const label = `[${done}/${total}] ${id.split("#")[0].slice(-10)}#${id.split("#")[1]} — ${tStage(stage)}`;
      $("enhStatus").textContent = label;
      setProgress(label, total ? done / total : null);
      showCancel(true);
    });
    const unlisten2 = await listen("photo-enhanced", (ev) => {
      state.photos.set(ev.payload.id, ev.payload.meta);
      renderGrid();
    });
    const [, cancelled] = await invoke("enhance_photos", { ids, upscale, colorize, faces, engine });
    unlisten();
    unlisten2();
    setProgress(null);
    if (cancelled) {
      $("enhStatus").textContent = t("cancelled");
      toast(t("cancelled"));
    } else {
      $("enhStatus").textContent = t("enhancedDone", ids.length);
      const last = state.photos.get(ids[ids.length - 1]);
      if (last?.enhanced) openCompare(last.id);
    }
    renderGrid();
  } catch (e) {
    setProgress(null);
    $("enhStatus").textContent = "";
    toast(t("enhanceFailed", e));
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
  state.compareId = id;
  $("modalTitle").textContent = m.id;
  const afterDims = dimsAfter(m);
  $("modalInfo").textContent = afterDims
    ? t("modalInfo", dims(m), afterDims)
    : t("modalInfoNoEnh", dims(m));
  $("btnResetEnh").classList.toggle("hidden", !m.enhanced);

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

/* ---------- reset enhancement ---------- */

async function resetEnhancement() {
  const id = state.compareId;
  if (!id) return;
  try {
    const meta = await invoke("reset_enhancement", { id });
    state.photos.set(id, meta);
    renderGrid();
    openCompare(id); // refresh the modal: after-pane falls back to empty
    toast(t("resetDone"));
  } catch (e) {
    toast(t("resetFailed", e));
  }
}

/* ---------- manual crop editor ---------- */

let dragCorner = -1;

function enterCropEdit(id) {
  const m = state.photos.get(id);
  if (!m || !m.quad || m.quad.length !== 8) return toast(t("noQuad"));
  if (state.activeFile !== m.source_file) {
    selectFile(m.source_file, false);
  }
  if (!state.detected.has(m.source_file)) return toast(t("needDetect"));
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
    $("cropText").textContent = t("cropTextEdit", id.split("#")[1]);
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
  previewGen++; // drop any in-flight editing preview
  drawing = null;
  if (drawPoly) { drawPoly.remove(); drawPoly = null; }
  state.editing = null;
  overlayNodes = null;
  $("cropTools").classList.add("hidden");
  $("quadOverlay").innerHTML = "";
}

/* ---------- new photo crop (right-click on the sheet) ----------
   Use case: auto-detection merged two photos into one. Right-click above
   the sheet, drag a box around one photo, then apply — it is extracted as
   a new photo and appears in the grid as a manual (green-box) crop. */

let drawing = null; // { anchor: [x,y] } while the user drags out the box
let drawPoly = null;

function toImgCoords(cx, cy) {
  const img = $("sheetImg");
  const r = img.getBoundingClientRect();
  return [
    Math.max(0, Math.min(imgNaturalW(), ((cx - r.left) / r.width) * imgNaturalW())),
    Math.max(0, Math.min(imgNaturalH(), ((cy - r.top) / r.height) * imgNaturalH())),
  ];
}

function startDrawNewCrop() {
  const file = state.activeFile;
  if (!file) return toast(t("selectSheetFirst"));
  if (!state.detected.has(file)) return toast(t("needDetect"));
  if (state.editing || drawing) return toast(t("finishCropFirst"));
  if (!imgNaturalW()) return toast(t("noSheet"));
  drawing = { anchor: null };
  const svg = $("quadOverlay");
  svg.setAttribute("viewBox", `0 0 ${imgNaturalW()} ${imgNaturalH()}`);
  positionQuadOverlay();
  drawPoly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  drawPoly.setAttribute("fill", "rgba(0,180,255,.12)");
  drawPoly.setAttribute("stroke", "#00c8ff");
  drawPoly.setAttribute("stroke-width", "3");
  drawPoly.setAttribute("stroke-dasharray", "8 6");
  svg.appendChild(drawPoly);
  $("cropText").textContent = t("drawHint");
  $("cropTools").classList.remove("hidden");
}

function onDrawDown(e) {
  if (!drawing || e.button !== 0) return;
  e.preventDefault();
  drawing.anchor = toImgCoords(e.clientX, e.clientY);
  const svg = $("quadOverlay");
  svg.setPointerCapture(e.pointerId);
  const move = (ev) => {
    if (!drawing || !drawing.anchor) return;
    const q = toImgCoords(ev.clientX, ev.clientY);
    const x0 = Math.min(drawing.anchor[0], q[0]);
    const y0 = Math.min(drawing.anchor[1], q[1]);
    const x1 = Math.max(drawing.anchor[0], q[0]);
    const y1 = Math.max(drawing.anchor[1], q[1]);
    drawPoly.setAttribute("points", `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1}`);
  };
  const up = (ev) => {
    svg.removeEventListener("pointermove", move);
    svg.removeEventListener("pointerup", up);
    if (!drawing || !drawing.anchor) return;
    const q = toImgCoords(ev.clientX, ev.clientY);
    const x0 = Math.min(drawing.anchor[0], q[0]);
    const y0 = Math.min(drawing.anchor[1], q[1]);
    const x1 = Math.max(drawing.anchor[0], q[0]);
    const y1 = Math.max(drawing.anchor[1], q[1]);
    finishDraw([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerup", up);
}

function finishDraw(quad) {
  drawing = null;
  if (drawPoly) { drawPoly.remove(); drawPoly = null; }
  state.editing = { id: null, file: state.activeFile, quad };
  overlayNodes = null;
  $("cropText").textContent = t("drawAdjust");
  drawQuadOverlay();
}

async function applyNewCrop(e) {
  const sheet = state.sheets.get(e.file);
  const k = sheet.pageW / imgNaturalW();
  const quad = e.quad.flatMap((p) => [p[0] * k, p[1] * k]);
  try {
    setProgress(t("addingPhoto"), null);
    const meta = await invoke("add_manual_photo", {
      dir: state.dir, file: e.file, quad,
    });
    state.photos.set(meta.id, meta);
    // refresh the baked preview so the new crop box is drawn (green)
    try {
      previewGen++;
      const { quads, manual } = quadsForFile(e.file, meta.id, meta.quad);
      const b64 = await invoke("sheet_preview_with_quads", {
        dir: state.dir, file: e.file, quads, manual,
      });
      const sheet = state.sheets.get(e.file);
      state.sheets.set(e.file, { ...sheet, b64 });
      state.editing = null;
      overlayNodes = null;
      renderSplit();
    } catch (_) { /* best effort */ }
    setProgress(null);
    toast(t("photoAdded"));
    exitCropEdit();
    renderGrid();
  } catch (err) {
    setProgress(null);
    toast(t("addFailed", err));
  }
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
// adjusted crop boundary. A generation counter discards stale responses so an
// in-flight preview can never overwrite a newer one (e.g. after Apply crop).
let regenTimer = null;
let previewGen = 0;
function regenPreviewDebounced() {
  clearTimeout(regenTimer);
  regenTimer = setTimeout(regenPreviewNow, 250);
}

async function regenPreviewNow() {
  const e = state.editing;
  if (!e) return;
  const gen = ++previewGen;
  // quadsForFile speaks page coordinates; the editing quad is in preview px
  const sheet = state.sheets.get(e.file);
  const k = sheet ? sheet.pageW / imgNaturalW() : 1;
  const quadPage = e.quad.flatMap((p) => [p[0] * k, p[1] * k]);
  const { quads, manual } = quadsForFile(e.file, e.id, quadPage);
  try {
    const b64 = await invoke("sheet_preview_with_quads", {
      dir: state.dir, file: e.file, quads, manual,
    });
    if (gen !== previewGen) return; // superseded by a newer edit or Apply
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
/// Returns { quads, manual } in page coordinates (flat, 8 numbers each);
/// `manual` parallel-marks user-adjusted crops so the preview draws them green.
/// `editQuad` may be flat (as returned by the backend) or [[x,y]×4].
function quadsForFile(file, editId, editQuad) {
  const flat =
    editQuad.length === 8 && typeof editQuad[0] === "number"
      ? editQuad
      : editQuad.flatMap((p) => [p[0], p[1]]);
  const metas = [...state.photos.values()]
    .filter((m) => m.source_file === file && m.quad && m.quad.length === 8)
    .sort((a, b) => Number(a.id.split("#")[1]) - Number(b.id.split("#")[1]));
  const quads = [];
  const manual = [];
  for (const m of metas) {
    const editing = m.id === editId;
    quads.push(editing ? flat : m.quad);
    manual.push(editing || !!m.manual);
  }
  return { quads, manual };
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
  if (!e.id) return applyNewCrop(e);
  const sheet = state.sheets.get(e.file);
  const k = sheet.pageW / imgNaturalW();
  const quad = e.quad.flatMap((p) => [p[0] * k, p[1] * k]);
  try {
    setProgress(t("recrop"), null);
    const meta = await invoke("re_extract_photo", {
      dir: state.dir, file: e.file, id: e.id, quad,
    });
    state.photos.set(e.id, meta);
    // refresh the baked preview: the drawn box now reflects the new crop
    try {
      previewGen++; // invalidate any in-flight editing preview
      const { quads, manual } = quadsForFile(e.file, e.id, meta.quad);
      const b64 = await invoke("sheet_preview_with_quads", {
        dir: state.dir, file: e.file, quads, manual,
      });
      const sheet = state.sheets.get(e.file);
      state.sheets.set(e.file, { ...sheet, b64 });
      state.editing = null;
      overlayNodes = null;
      renderSplit();
    } catch (_) { /* best effort */ }
    setProgress(null);
    toast(t("cropUpdated"));
    exitCropEdit();
    renderGrid();
  } catch (err) {
    setProgress(null);
    toast(t("recropFailed", err));
  }
}

/* ---------- save ---------- */

async function save() {
  if (!state.selected.size) return toast(t("selectFirst"));
  const ids = [...state.selected];
  const format = $("saveFormat").value;
  let outDir = null;
  try {
    outDir = await invoke("pick_folder", { save: true });
  } catch (e) {
    toast(t("pickFailed", e));
    return;
  }
  if (!outDir) return;
  try {
    const saved = await invoke("save_photos", { ids, outDir, format });
    $("saveStatus").textContent = t("savedTo", saved.length, outDir);
    toast(t("savedTo", saved.length, outDir), 5000);
  } catch (e) {
    toast(t("saveFailed", e));
  }
}

/* ---------- wiring ---------- */

window.addEventListener("DOMContentLoaded", async () => {
  $("langSel").onchange = (e) => {
    lang = e.target.value;
    localStorage.setItem("lang", lang);
    applyLang();
  };
  $("btnBrowse").onclick = async () => {
    const dir = await invoke("pick_folder", { save: false });
    if (dir) {
      $("scanDir").value = dir;
      await loadFolder(dir);
    }
  };
  // Typed/pasted paths load on Enter (the Load button was removed).
  $("scanDir").addEventListener("keydown", async (e) => {
    if (e.key !== "Enter" || e.isComposing) return;
    const dir = e.target.value.trim();
    if (!dir) return toast(t("loadEmpty"));
    try {
      await loadFolder(dir);
    } catch (err) {
      toast(t("loadFailed", err));
    }
  });
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
  $("sheetWrap").addEventListener("contextmenu", (e) => {
    e.preventDefault(); // suppress the webview menu for our crop flow
    startDrawNewCrop();
  });
  $("sheetWrap").addEventListener("pointerdown", onDrawDown);
  $("btnSave").onclick = save;
  $("modalClose").onclick = () => $("modal").classList.add("hidden");
  $("btnResetEnh").onclick = resetEnhancement;
  $("modal").onclick = (e) => {
    if (e.target === $("modal")) $("modal").classList.add("hidden");
  };
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("modal").classList.add("hidden");
  });

  setupPane("before");
  setupPane("after");
  applyLang();
  await initPhantomEngine();
  await initAbout();

  try {
    const def = await invoke("default_dir");
    if (def) {
      $("scanDir").value = def;
      await loadFolder(def);
    }
  } catch (_) { /* ignore */ }
});
