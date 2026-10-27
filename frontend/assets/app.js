const views = ["search", "library", "upload", "annotate"];
const state = { videos: [], poller: null, lastQuery: "", pendingRemoval: null, toastTimer: null };
const REMOVABLE_STATES = new Set(["UPLOADED", "READY", "FAILED"]);

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const formatTime = (seconds) => {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
};
// Sub-second display for localized peaks: 00:42.3
const formatPeak = (seconds) => {
  const value = Math.max(0, Number(seconds) || 0);
  const whole = Math.floor(value);
  const tenths = Math.floor((value - whole) * 10);
  return `${formatTime(whole)}.${tenths}`;
};
const seekTimeFor = (item) => (item.localization ? item.localization.peak_time : item.start_time);
// Shown whenever Groq was not used as configured, so users know which path produced the result.
const PLANNER_NOTICES = {
  fallback_provider_unavailable: "Local query planner used: Groq unavailable (rate limit or network)",
  fallback_provider_unconfigured: "Local query planner used: Groq not configured",
  fallback_low_confidence: "Local query planner used: Groq plan below the confidence threshold",
};
const ANSWER_NOTICES = {
  provider_unavailable: "No generated answer: Groq unavailable (rate limit or network)",
  rejected_unverified_claims: "Generated answer rejected: it made claims the evidence does not back",
  rejected_invalid_evidence_ids: "Generated answer rejected: it cited evidence that was not retrieved",
  downgraded_weak_evidence: "Generated answer downgraded: the system's verdicts are weak",
  downgraded_insufficient_evidence: "Generated answer downgraded: the evidence is insufficient",
};
const PRECISION_LABEL = {
  word: "word timing",
  frame: "frame observation",
  interval: "evidence interval",
  utterance_interpolated: "utterance (interpolated)",
  utterance: "utterance",
  segment: "5 s segment only",
};

function showView(name) {
  views.forEach((view) => {
    $(`#${view}-view`).classList.toggle("active", view === name);
    $(`.nav-link[data-view='${view}']`).classList.toggle("active", view === name);
  });
  if (name === "library") refreshLibrary();
  if (name === "annotate" && typeof loadAnnotations === "function") loadAnnotations();
  history.replaceState(null, "", `#${name}`);
}

$$('.nav-link').forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));

async function checkHealth() {
  const health = $("#health");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("health unavailable");
    const payload = await response.json();
    const modelLabel = (name, status) => `${name} ${String(status).startsWith("available_") ? "ready" : status}`;
    const modelSummary = [
      modelLabel("YOLO", payload.models?.yolo || "unknown"),
      modelLabel("Whisper", payload.models?.whisper || "unknown"),
    ].join(" · ");
    health.className = "health ok";
    health.title = Object.entries(payload.models || {}).map(([name, status]) => `${name}: ${status}`).join("\n");
    health.innerHTML = `<span></span>${escapeHtml(modelSummary)}`;
  } catch (_) {
    health.className = "health error";
    health.innerHTML = "<span></span>System unavailable";
  }
}

$$('.examples button').forEach((button) => button.addEventListener("click", () => {
  $("#query").value = button.textContent;
  $("#query").focus();
}));

async function runSearch(query, hypothesisId = null) {
  const panel = $("#search-state");
  const section = $("#results-section");
  panel.className = "state-panel";
  panel.textContent = "Ranking transcript, entity, action, speaker, and relationship evidence…";
  section.classList.add("hidden");
  state.lastQuery = query;
  try {
    const selectedVideo = $("#video-filter").value;
    const body = {
      query,
      video_ids: selectedVideo ? [selectedVideo] : [],
      limit: 10,
      ranking: "reliability",
      localize: true,
      collapse_duplicate_evidence: true,
    };
    if (hypothesisId) body.hypothesis_id = hypothesisId;
    const response = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Search failed");
    panel.classList.add("hidden");
    renderResults(payload);
  } catch (error) {
    panel.className = "state-panel error";
    panel.textContent = error.message;
  }
}

$("#search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("#query").value.trim();
  if (query) runSearch(query);
});

function renderInterpretation(payload) {
  const card = $("#interpretation");
  const hypotheses = payload.hypotheses || [];
  const status = payload.interpretation || "single";
  if (status === "single" && !hypotheses.length) {
    card.classList.add("hidden");
    card.innerHTML = "";
    return;
  }
  const chips = hypotheses.map((item) => {
    const selected = item.hypothesis_id === payload.selected_hypothesis_id;
    return `<button class="hypothesis-chip${selected ? " selected" : ""}" data-hypothesis="${escapeHtml(item.hypothesis_id)}" title="${escapeHtml(item.description)}">${escapeHtml(item.label)} · ${item.result_count} hit${item.result_count === 1 ? "" : "s"} · ${Math.round(item.posterior * 100)}%</button>`;
  }).join("");
  let headline = "";
  if (status === "clarification_suggested") headline = payload.clarification_prompt || "Did you mean…?";
  else if (status === "insufficient_evidence") headline = "Not enough evidence for a confident answer. The closest moments are shown below with their confidence.";
  else if (status === "resolved") headline = "Ambiguous query resolved from the evidence. Pick another reading to override.";
  card.className = `interpretation-card ${status}`;
  card.innerHTML = `<p class="eyebrow">QUERY INTERPRETATION</p><p class="interpretation-text">${escapeHtml(headline)}</p><div class="hypothesis-row">${chips}</div>`;
  card.classList.remove("hidden");
  $$('.hypothesis-chip').forEach((button) => button.addEventListener("click", () => runSearch(state.lastQuery, button.dataset.hypothesis)));
}

function confidenceBadge(item) {
  if (!item.evidence) return "";
  const verdict = item.evidence.verdict;
  const label = verdict === "supported" ? "supported" : verdict === "weak" ? "weak evidence" : "insufficient";
  return `<span class="verdict ${verdict}" title="${escapeHtml(item.evidence.explanation || "")}">${label} · ${Math.round(item.evidence.confidence * 100)}%</span>`;
}

function renderResults(payload) {
  const section = $("#results-section");
  const results = $("#results");
  const plan = payload.parsed_query;
  section.classList.remove("hidden");
  $("#result-count").textContent = `${payload.results.length} result${payload.results.length === 1 ? "" : "s"}`;
  renderInterpretation(payload);
  const chips = [
    ...(plan.intent ? [`intent · ${plan.intent.replaceAll("_", " ")}`] : []),
    ...plan.visual_entities.map((value) => `visual · ${value}`),
    ...plan.actions.map((value) => `action · ${value}`),
    ...plan.relationships.map((value) => `relation · ${value}`),
    ...plan.relationship_tuples.map((value) => `tuple · ${value.subject} ${value.predicate} ${value.object}`),
    ...plan.spoken_terms.map((value) => `spoken · ${value}`),
    ...(plan.visual_concepts || []).map((value) => `visible · ${value}`),
    ...(plan.ocr_terms || []).map((value) => `OCR · ${value}`),
    ...(plan.speaker ? [`speaker · ${plan.speaker}`] : []),
  ];
  (plan.appearance_constraints || []).forEach((value) => {
    const details = [value.upper_clothing_color && `${value.upper_clothing_color} upper clothing`, value.color, value.glasses && "glasses", value.hat && "hat", value.bag && "bag", value.description].filter(Boolean);
    chips.push(`appearance: ${value.entity_class}${details.length ? ` / ${[...new Set(details)].join(" / ")}` : ""}`);
  });
  if ((payload.query_ambiguities || []).length) {
    chips.push(`ambiguous · ${payload.query_ambiguities.join(" ")}`);
  }
  $("#parsed-query").innerHTML = chips.map((value) => `<span class="chip">${escapeHtml(value)}</span>`).join("");
  const notices = [PLANNER_NOTICES[payload.query_planner_status], ANSWER_NOTICES[payload.rag_status]].filter(Boolean);
  const noticeBox = $("#pipeline-notices");
  noticeBox.innerHTML = notices.map((value) => `<span class="notice">${escapeHtml(value)}</span>`).join("");
  noticeBox.classList.toggle("hidden", !notices.length);
  renderGeneratedAnswer(payload.generated_answer, payload.results);
  if (!payload.results.length) {
    results.innerHTML = '<div class="state-panel">No indexed moment matched. Try fewer constraints or process another video.</div>';
    return;
  }
  results.innerHTML = payload.results.map((item, index) => {
    const evidence = [
      ...item.entities.map((value) => `entity: ${value}`),
      ...item.actions.map((value) => `action: ${value}`),
      ...(item.action_evidence || []).slice(0, 4).map((value) => {
        const target = value.target_entity_id ? ` → ${value.target_entity_id}` : "";
        const confidence = value.confidence == null ? "" : ` · ${value.confidence.toFixed(2)}`;
        return `${value.action}${target} · ${value.provenance}${confidence}`;
      }),
      ...(item.ocr_evidence || []).slice(0, 3).map((value) => `OCR: ${value.text} · ${value.source_method} · ${Number(value.confidence).toFixed(2)}`),
      ...item.speakers.map((value) => `speaker: ${value}`),
      ...item.matched_relationships.map((value) => `${value.source_class} ${value.predicate} ${value.target_class} · ${value.source_method}${value.model_version ? ` · ${value.model_version}` : ""}${value.temporal_smoothed ? ` · ${value.observation_count} observations` : ""} · ${Number(value.confidence).toFixed(2)}`),
      ...(!item.matched_relationships.length ? item.relationships.slice(0, 4).map((value) => `${value.predicate} · ${value.source_method}${value.model_version ? ` · ${value.model_version}` : ""}${value.temporal_smoothed ? ` · ${value.observation_count} observations` : ""}`) : []),
    ];
    evidence.push(...(item.appearance_matches || []).map((value) => {
      const attrs = Object.entries(value.attributes || {}).filter(([, evidence]) => evidence.value === true || typeof evidence.value === "string").map(([name, evidence]) => `${name.replaceAll("_", " ")}: ${evidence.value}`);
      const similarity = value.matched_via.includes("appearance_similarity") && value.similarity != null ? `appearance match / cosine ${Number(value.similarity).toFixed(3)}` : "";
      return [value.entity_class, ...attrs, similarity, `${value.observation_count} observations`, "visual_attribute_model"].filter(Boolean).join(" / ");
    }));
    const localization = item.localization;
    const peakLabel = localization ? `<span class="peak">⏱ ${formatPeak(localization.peak_time)}</span>` : "";
    const precision = localization ? `<span class="precision" title="${escapeHtml((localization.evidence || []).join("\n"))}">${escapeHtml(PRECISION_LABEL[localization.precision] || localization.precision)} · ${escapeHtml(localization.source)}</span>` : "";
    const hypothesisTags = (item.supporting_hypotheses || []).map((value) => `<span class="hyp-tag">${escapeHtml(value.replaceAll("_", " "))}</span>`).join("");
    return `<article class="result-card" data-video-id="${escapeHtml(item.video_id)}">
      <div class="thumbnail"><img src="${escapeHtml(item.thumbnail_url)}" alt="Frame near ${formatTime(seekTimeFor(item))}" loading="lazy" onerror="this.style.display='none'" /><span>${formatTime(item.start_time)}–${formatTime(item.end_time)}</span>${peakLabel}</div>
      <div class="result-main"><p class="eyebrow">RESULT ${index + 1} ${hypothesisTags}</p><h3>${escapeHtml(item.video_title)}</h3><div class="result-meta">${confidenceBadge(item)}${precision}</div><p class="reason">${escapeHtml(item.match_reason)}</p><p class="transcript">${escapeHtml(item.transcript || "No speech in this segment.")}</p><div class="evidence">${evidence.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}</div></div>
      <div class="score"><small>score</small><strong>${Math.round(item.score * 100)}</strong><button class="play-button" data-result='${escapeHtml(JSON.stringify(item))}'>Play</button></div>
    </article>`;
  }).join("");
  $$('.play-button').forEach((button) => button.addEventListener("click", () => openPlayer(JSON.parse(button.dataset.result))));
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderGeneratedAnswer(answer, rankedResults) {
  const card = $("#generated-answer");
  if (!answer) {
    card.classList.add("hidden");
    card.innerHTML = "";
    return;
  }
  const citations = answer.citations || [];
  card.className = `answer-card ${answer.supported ? "supported" : "unsupported"}`;
  card.innerHTML = `<p class="eyebrow">${answer.supported ? "GROUNDED ANSWER" : "EVIDENCE CHECK"}</p>
    <p class="answer-text">${escapeHtml(answer.answer)}</p>
    <div class="answer-citations">${citations.map((citation) => `<button class="answer-citation" data-evidence-id="${escapeHtml(citation.evidence_id)}">${escapeHtml(citation.evidence_id)} \u00b7 ${escapeHtml(citation.video_title)} \u00b7 ${citation.peak_time != null ? formatPeak(citation.peak_time) : `${formatTime(citation.start_time)}\u2013${formatTime(citation.end_time)}`}</button>`).join("")}</div>
    ${answer.supported ? "<small>Synthesized only from the cited ranked evidence.</small>" : (citations.length ? "<small>Not confirmed by the system's verdicts; these are the closest moments.</small>" : "")}`;
  $$('.answer-citation').forEach((button) => button.addEventListener("click", () => {
    const citation = citations.find((item) => item.evidence_id === button.dataset.evidenceId);
    if (!citation) return;
    const ranked = rankedResults.find((item) => item.segment_id === citation.segment_id);
    openPlayer(ranked || {
      video_title: citation.video_title,
      start_time: citation.start_time,
      end_time: citation.end_time,
      stream_url: citation.stream_url,
      match_reason: `Grounded answer citation ${citation.evidence_id}`,
      transcript: "",
      localization: citation.peak_time != null ? { peak_time: citation.peak_time } : null,
    });
  }));
}

function openPlayer(item) {
  const dialog = $("#player-dialog");
  const player = $("#player");
  const seekTo = seekTimeFor(item);
  $("#player-title").textContent = item.video_title;
  $("#player-time").textContent = item.localization
    ? `${formatPeak(seekTo)} (segment ${formatTime(item.start_time)}–${formatTime(item.end_time)})`
    : `${formatTime(item.start_time)}–${formatTime(item.end_time)}`;
  $("#player-reason").textContent = item.match_reason;
  $("#player-transcript").textContent = item.transcript || "No transcript for this moment.";
  player.src = item.stream_url;
  // Seek slightly ahead of the peak so the viewer sees the lead-in.
  const target = Math.max(0, seekTo - 0.5);
  player.addEventListener("loadedmetadata", () => { player.currentTime = target; player.play().catch(() => {}); }, { once: true });
  dialog.showModal();
}

$("#close-player").addEventListener("click", () => { $("#player").pause(); $("#player-dialog").close(); });

async function refreshLibrary() {
  const panel = $("#library-state");
  try {
    const response = await fetch("/api/videos");
    if (!response.ok) throw new Error("Could not load video library");
    state.videos = await response.json();
    updateVideoFilter(state.videos);
    panel.classList.add("hidden");
    const list = $("#video-list");
    if (!state.videos.length) {
      list.innerHTML = '<div class="state-panel">No videos yet. Upload an MP4 to build the first index.</div>';
      return;
    }
    list.innerHTML = state.videos.map((video) => {
      const removable = REMOVABLE_STATES.has(video.state);
      return `<article class="video-row" data-id="${escapeHtml(video.video_id)}"><div><h3>${escapeHtml(video.display_name)}</h3><p>${formatTime(video.duration)} · ${video.width}×${video.height} · ${Number(video.fps).toFixed(2)} fps${video.warnings.length ? ` · ${video.warnings.length} warning(s)` : ""}</p><p>${escapeHtml(video.failure_message || video.current_stage || "")}</p></div><span class="badge ${escapeHtml(video.state)}">${escapeHtml(video.state)}</span><div class="video-actions"><button class="retry" data-id="${escapeHtml(video.video_id)}">${video.state === "FAILED" ? "Retry" : "Reprocess"}</button><button class="remove-video" data-id="${escapeHtml(video.video_id)}"${removable ? "" : ' disabled title="Wait until processing finishes"'} aria-label="Remove ${escapeHtml(video.display_name)} from this library">Remove</button></div></article>`;
    }).join("");
    $$('.retry').forEach((button) => button.addEventListener("click", () => reprocess(button.dataset.id)));
    $$('.remove-video').forEach((button) => button.addEventListener("click", () => askRemove(button.dataset.id)));
  } catch (error) {
    panel.className = "state-panel error";
    panel.textContent = error.message;
  }
}

function updateVideoFilter(videos) {
  const filter = $("#video-filter");
  const selected = filter.value;
  filter.innerHTML = '<option value="">All videos</option>' + videos.map((video) => `<option value="${escapeHtml(video.video_id)}">${escapeHtml(video.display_name)}</option>`).join("");
  if (videos.some((video) => video.video_id === selected)) filter.value = selected;
}

async function loadVideoFilter() {
  try {
    const response = await fetch("/api/videos");
    if (!response.ok) return;
    state.videos = await response.json();
    updateVideoFilter(state.videos);
  } catch (_) {}
}

async function reprocess(videoId) {
  const response = await fetch(`/api/videos/${videoId}/process`, { method: "POST" });
  if (!response.ok) {
    const payload = await response.json();
    alert(payload.detail || "Could not start processing");
  }
  refreshLibrary();
}

function showToast(message, kind = "success") {
  const toast = $("#toast");
  toast.className = `toast ${kind}`;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  toast.textContent = message;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.add("hidden"), kind === "error" ? 9000 : 6000);
}

function askRemove(videoId) {
  const video = state.videos.find((item) => item.video_id === videoId);
  if (!video) return;
  state.pendingRemoval = video;
  $("#remove-name").textContent = video.display_name;
  const dialog = $("#remove-dialog");
  dialog.returnValue = "";
  dialog.showModal();
}

// Escape and the Cancel button close the dialog with a returnValue other than
// "confirm", so only the Remove button sends a request.
$("#remove-dialog").addEventListener("close", () => {
  const video = state.pendingRemoval;
  state.pendingRemoval = null;
  if (video && $("#remove-dialog").returnValue === "confirm") removeVideo(video);
});

async function removeVideo(video) {
  const row = $$(".video-row").find((item) => item.dataset.id === video.video_id);
  if (row) {
    row.classList.add("removing");
    row.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  }
  try {
    const response = await fetch(`/api/videos/${encodeURIComponent(video.video_id)}`, { method: "DELETE" });
    let payload = {};
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    state.videos = state.videos.filter((item) => item.video_id !== video.video_id);
    if (row) row.remove();
    updateVideoFilter(state.videos);
    $$(".result-card").filter((card) => card.dataset.videoId === video.video_id).forEach((card) => card.remove());
    if (!$("#results-section").classList.contains("hidden")) {
      const remaining = $$(".result-card").length;
      $("#result-count").textContent = `${remaining} result${remaining === 1 ? "" : "s"}`;
    }
    showToast(`Removed “${video.display_name}” from this library. Its files are in the library trash and can be restored.`);
  } catch (error) {
    showToast(`Could not remove “${video.display_name}”: ${error.message}`, "error");
  }
  refreshLibrary();
}

const fileInput = $("#video-file");
fileInput.addEventListener("change", () => { $("#selected-file").textContent = fileInput.files[0]?.name || "No file selected"; });
const dropZone = $("#drop-zone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    $("#selected-file").textContent = fileInput.files[0].name;
  }
});

$("#upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const progress = $("#upload-progress");
  const bar = progress.querySelector("span");
  const panel = $("#upload-state");
  progress.classList.remove("hidden");
  panel.className = "state-panel";
  panel.textContent = "Uploading…";
  const request = new XMLHttpRequest();
  request.open("POST", "/api/videos");
  request.upload.addEventListener("progress", (value) => { if (value.lengthComputable) bar.style.width = `${(value.loaded / value.total) * 100}%`; });
  request.addEventListener("load", () => {
    let payload = {}; try { payload = JSON.parse(request.responseText); } catch (_) {}
    if (request.status >= 200 && request.status < 300) {
      panel.textContent = payload.duplicate ? "This exact video is already registered." : `Upload accepted. Video ${payload.video_id} is processing.`;
      setTimeout(() => showView("library"), 900);
    } else {
      panel.className = "state-panel error";
      panel.textContent = payload.detail || "Upload failed";
    }
  });
  request.addEventListener("error", () => { panel.className = "state-panel error"; panel.textContent = "Network error during upload"; });
  request.send(form);
});

checkHealth();
loadVideoFilter();
const initial = location.hash.replace("#", "");
if (views.includes(initial)) showView(initial);
state.poller = setInterval(() => { if ($("#library-view").classList.contains("active")) refreshLibrary(); }, 4000);
