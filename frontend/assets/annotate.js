// Ground-truth annotation page for the Review-2 challenge benchmark.
// Shares $, $$, escapeHtml and formatTime with app.js.

const HYPOTHESIS_OPTIONS = [
  ["visible_action", "visible action"],
  ["spoken_mention", "spoken word"],
  ["visible_object", "visible object"],
  ["visible_text", "on-screen text"],
];
const INTENT_OPTIONS = [
  "transcript_search",
  "visual_search",
  "action_search",
  "relationship_search",
  "speaker_search",
  "ocr_search",
  "multimodal_search",
];
const CATEGORY_PREFIX = { timestamp: "ts", ambiguous: "amb", action_relation: "ar", negative: "neg" };
const CATEGORY_LABEL = { timestamp: "timestamp", ambiguous: "ambiguous", action_relation: "action / relation", negative: "negative" };
const annotation = { videos: [], queries: [], summary: null, moments: [], editingId: null };

const round1 = (value) => Math.round(Number(value) * 10) / 10;
// Same rounding as the stored marks, so the clock shows exactly what "Mark" saves.
const formatMark = (seconds) => {
  const tenths = Math.round(Math.max(0, Number(seconds) || 0) * 10);
  return `${formatTime(Math.floor(tenths / 10))}.${tenths % 10}`;
};
const COUNT_WARNING = /protocol asks for/;
const annotatePlayer = $("#annotate-player");
const currentVideo = () => annotation.videos.find((video) => video.video_id === $("#annotate-video").value);

$("#ann-hypotheses").innerHTML = HYPOTHESIS_OPTIONS.map(([value, label]) => `<label><input type="checkbox" name="ann-hypothesis" value="${value}" /> ${label}</label>`).join("");
$("#ann-intents").innerHTML = INTENT_OPTIONS.map((value) => `<label><input type="checkbox" name="ann-intent" value="${value}" /> ${value.replace("_search", "").replaceAll("_", " ")}</label>`).join("");
const savedAnnotator = localStorage.getItem("vidquery-annotator");
if (savedAnnotator) $("#ann-annotator").value = savedAnnotator;

function annotationStatus(message, isError = false) {
  const panel = $("#ann-status");
  panel.className = `state-panel${isError ? " error" : ""}`;
  panel.textContent = message;
}

async function annotationRequest(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join("; ") : payload.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

async function loadAnnotations() {
  try {
    const payload = await annotationRequest("/api/annotations");
    annotation.videos = payload.videos.filter((video) => video.state === "READY");
    annotation.queries = payload.queries;
    annotation.summary = payload.summary;
    const select = $("#annotate-video");
    const previous = select.value;
    select.innerHTML = annotation.videos.length
      ? annotation.videos.map((video) => `<option value="${escapeHtml(video.video_id)}">${escapeHtml(video.display_name)}</option>`).join("")
      : '<option value="">No READY videos yet</option>';
    if (annotation.videos.some((video) => video.video_id === previous)) select.value = previous;
    selectAnnotationVideo(select.value !== previous);
    renderAnnotationProgress();
  } catch (error) {
    annotationStatus(error.message, true);
  }
}

function selectAnnotationVideo(changed) {
  const video = currentVideo();
  $("#annotate-split").value = video?.split || "";
  if (video && changed) {
    annotatePlayer.src = video.stream_url;
    clearAnnotationForm();
  }
  renderAnnotationList();
}

function renderAnnotationProgress() {
  const summary = annotation.summary;
  if (!summary) return;
  const categories = Object.entries(summary.category_targets).map(([name, target]) => {
    const count = summary.category_counts[name] || 0;
    return `<span class="progress-chip${count >= target ? " done" : ""}">${escapeHtml(CATEGORY_LABEL[name] || name)} ${count}/${target}</span>`;
  }).join("");
  const splits = Object.entries(summary.split_counts).map(([name, count]) => `<span class="progress-chip">${escapeHtml(name)} ${count}</span>`).join("");
  const shown = summary.warnings.filter((warning) => !COUNT_WARNING.test(warning) && warning !== "no queries labelled yet");
  const warnings = shown.length ? `<p class="progress-warn">${shown.map(escapeHtml).join(" · ")}</p>` : "";
  $("#annotation-progress").innerHTML = `<div><strong>${summary.query_count}/${summary.target_query_count}</strong> queries labelled${summary.runnable ? " · benchmark runnable" : ""}</div><div class="progress-row">${categories}${splits}</div>${warnings}`;
}

function renderMoments() {
  $("#ann-moments").innerHTML = annotation.moments.map((moment, index) => `<li><button type="button" class="seek" data-time="${moment.peak_time}">${formatMark(moment.start_time)} → <strong>${formatMark(moment.peak_time)}</strong> → ${formatMark(moment.end_time)}</button><button type="button" class="remove" data-index="${index}" aria-label="Remove moment">&times;</button></li>`).join("");
  $$("#ann-moments .seek").forEach((button) => button.addEventListener("click", () => { annotatePlayer.currentTime = Number(button.dataset.time); }));
  $$("#ann-moments .remove").forEach((button) => button.addEventListener("click", () => {
    annotation.moments.splice(Number(button.dataset.index), 1);
    renderMoments();
  }));
}

function renderAnnotationList() {
  const video = currentVideo();
  const list = $("#ann-list");
  const rows = video ? annotation.queries.filter((item) => item.source_video_id === video.source_video_id) : [];
  $("#ann-count").textContent = `${rows.length} for this video`;
  if (!rows.length) {
    list.innerHTML = '<div class="state-panel">No queries labelled for this video yet.</div>';
    return;
  }
  list.innerHTML = rows.map((item) => {
    const moments = item.relevant.map((moment) => `<button type="button" class="seek" data-time="${moment.peak_time}">${formatMark(moment.peak_time)}</button>`).join("") || '<span class="muted">does not occur</span>';
    return `<article class="ann-row"><div><p class="eyebrow">${escapeHtml(item.id)} · ${escapeHtml(CATEGORY_LABEL[item.category] || item.category)}</p><h3>${escapeHtml(item.query)}</h3><div class="ann-row-moments">${moments}</div></div><div class="ann-row-actions"><button type="button" class="edit" data-id="${escapeHtml(item.id)}">Edit</button><button type="button" class="delete" data-id="${escapeHtml(item.id)}">Delete</button></div></article>`;
  }).join("");
  $$("#ann-list .seek").forEach((button) => button.addEventListener("click", () => { annotatePlayer.currentTime = Number(button.dataset.time); }));
  $$("#ann-list .edit").forEach((button) => button.addEventListener("click", () => editAnnotation(button.dataset.id)));
  $$("#ann-list .delete").forEach((button) => button.addEventListener("click", () => deleteAnnotation(button.dataset.id)));
}

function suggestAnnotationId() {
  const video = currentVideo();
  if (!video) return "";
  const prefix = CATEGORY_PREFIX[$("#ann-category").value] || "q";
  const slug = video.source_video_id.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 24) || "video";
  const taken = new Set(annotation.queries.map((item) => item.id));
  for (let index = 1; index < 1000; index += 1) {
    const candidate = `${prefix}-${slug}-${String(index).padStart(2, "0")}`;
    if (!taken.has(candidate)) return candidate;
  }
  return "";
}

function updateCategoryState() {
  const negative = $("#ann-category").value === "negative";
  $("#ann-moment-fields").disabled = negative;
  if (negative) {
    annotation.moments = [];
    renderMoments();
  }
  if (!annotation.editingId) $("#ann-id").value = suggestAnnotationId();
}

function clearAnnotationForm() {
  annotation.editingId = null;
  annotation.moments = [];
  ["#ann-query", "#ann-start", "#ann-peak", "#ann-end", "#ann-notes"].forEach((selector) => { $(selector).value = ""; });
  $("#ann-tolerance").value = "2";
  $$("input[name='ann-hypothesis'], input[name='ann-intent']").forEach((input) => { input.checked = false; });
  $("#ann-id").readOnly = false;
  renderMoments();
  updateCategoryState();
}

function editAnnotation(id) {
  const item = annotation.queries.find((query) => query.id === id);
  if (!item) return;
  annotation.editingId = id;
  $("#ann-id").value = item.id;
  $("#ann-id").readOnly = true;
  $("#ann-query").value = item.query;
  $("#ann-category").value = item.category;
  $("#ann-tolerance").value = item.tolerance_seconds ?? 2;
  $("#ann-notes").value = item.notes || "";
  $$("input[name='ann-hypothesis']").forEach((input) => { input.checked = (item.expected_hypothesis_ids || []).includes(input.value); });
  $$("input[name='ann-intent']").forEach((input) => { input.checked = (item.expected_intents || []).includes(input.value); });
  annotation.moments = item.relevant.map(({ peak_time, start_time, end_time }) => ({ peak_time, start_time, end_time }));
  $("#ann-moment-fields").disabled = item.category === "negative";
  renderMoments();
  $("#ann-query").focus();
}

async function deleteAnnotation(id) {
  if (!confirm(`Delete ${id}?`)) return;
  try {
    await annotationRequest(`/api/annotations/queries/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (annotation.editingId === id) clearAnnotationForm();
    annotationStatus(`Deleted ${id}.`);
    await loadAnnotations();
  } catch (error) {
    annotationStatus(error.message, true);
  }
}

function addMomentFromInputs() {
  const start = $("#ann-start").value;
  const peak = $("#ann-peak").value;
  const end = $("#ann-end").value;
  if (start === "" || peak === "" || end === "") {
    annotationStatus("Mark start, peak and end before adding the moment.", true);
    return false;
  }
  const moment = { start_time: round1(start), peak_time: round1(peak), end_time: round1(end) };
  if (!(moment.start_time <= moment.peak_time && moment.peak_time <= moment.end_time)) {
    annotationStatus("The peak must lie between start and end.", true);
    return false;
  }
  annotation.moments.push(moment);
  ["#ann-start", "#ann-peak", "#ann-end"].forEach((selector) => { $(selector).value = ""; });
  renderMoments();
  $("#ann-status").classList.add("hidden");
  return true;
}

function markTime(which) {
  if ($("#ann-moment-fields").disabled) return;
  $(`#ann-${which}`).value = round1(annotatePlayer.currentTime).toFixed(1);
}

$("#annotate-video").addEventListener("change", () => selectAnnotationVideo(true));
$("#annotate-split").addEventListener("change", async (event) => {
  const video = currentVideo();
  if (!video || !event.target.value) return;
  try {
    await annotationRequest(`/api/annotations/videos/${encodeURIComponent(video.source_video_id)}/split`, { method: "PUT", body: JSON.stringify({ split: event.target.value }) });
    annotationStatus(`${video.display_name} is in the ${event.target.value} split.`);
    await loadAnnotations();
  } catch (error) {
    annotationStatus(error.message, true);
  }
});
$("#ann-category").addEventListener("change", updateCategoryState);
$("#ann-add-moment").addEventListener("click", addMomentFromInputs);
$("#ann-clear").addEventListener("click", clearAnnotationForm);
$$(".annotate-buttons [data-mark]").forEach((button) => button.addEventListener("click", () => markTime(button.dataset.mark)));
$$(".annotate-buttons [data-step]").forEach((button) => button.addEventListener("click", () => {
  annotatePlayer.currentTime = Math.max(0, annotatePlayer.currentTime + Number(button.dataset.step));
}));
annotatePlayer.addEventListener("timeupdate", () => { $("#annotate-now").textContent = formatMark(annotatePlayer.currentTime); });
annotatePlayer.addEventListener("seeked", () => { $("#annotate-now").textContent = formatMark(annotatePlayer.currentTime); });

document.addEventListener("keydown", (event) => {
  if (!$("#annotate-view").classList.contains("active")) return;
  if (event.target.closest("input, textarea, select") || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.target === annotatePlayer && [" ", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const step = event.shiftKey ? 1 : 0.1;
  const actions = {
    "[": () => markTime("start"),
    "p": () => markTime("peak"),
    "P": () => markTime("peak"),
    "]": () => markTime("end"),
    " ": () => (annotatePlayer.paused ? annotatePlayer.play() : annotatePlayer.pause()),
    "ArrowLeft": () => { annotatePlayer.currentTime = Math.max(0, annotatePlayer.currentTime - step); },
    "ArrowRight": () => { annotatePlayer.currentTime += step; },
  };
  const action = actions[event.key];
  if (!action) return;
  event.preventDefault();
  action();
});

$("#annotation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const video = currentVideo();
  if (!video) {
    annotationStatus("Pick a READY video first.", true);
    return;
  }
  const category = $("#ann-category").value;
  const pending = ["#ann-start", "#ann-peak", "#ann-end"].some((selector) => $(selector).value !== "");
  if (category !== "negative" && pending && !addMomentFromInputs()) return;
  if (category !== "negative" && !annotation.moments.length) {
    annotationStatus("Add at least one relevant moment, or choose the negative category.", true);
    return;
  }
  const annotator = $("#ann-annotator").value.trim();
  localStorage.setItem("vidquery-annotator", annotator);
  const body = {
    id: $("#ann-id").value.trim(),
    query: $("#ann-query").value.trim(),
    category,
    source_video_id: video.source_video_id,
    expected_intents: $$("input[name='ann-intent']:checked").map((input) => input.value),
    expected_hypothesis_ids: $$("input[name='ann-hypothesis']:checked").map((input) => input.value),
    tolerance_seconds: Number($("#ann-tolerance").value) || 2,
    relevant: category === "negative" ? [] : annotation.moments,
    annotator,
    notes: $("#ann-notes").value.trim(),
  };
  if (!annotation.editingId && annotation.queries.some((item) => item.id === body.id)) {
    annotationStatus(`ID ${body.id} already exists. Use Edit to change it, or pick another ID.`, true);
    return;
  }
  try {
    const payload = await annotationRequest(`/api/annotations/queries/${encodeURIComponent(body.id)}`, { method: "PUT", body: JSON.stringify(body) });
    annotationStatus(`${payload.replaced ? "Updated" : "Saved"} ${body.id}.`);
    await loadAnnotations();
    clearAnnotationForm();
  } catch (error) {
    annotationStatus(error.message, true);
  }
});

if ($("#annotate-view").classList.contains("active")) loadAnnotations();
