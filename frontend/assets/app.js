const views = ["search", "library", "upload"];
const state = { videos: [], poller: null };

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

function showView(name) {
  views.forEach((view) => {
    $(`#${view}-view`).classList.toggle("active", view === name);
    $(`.nav-link[data-view='${view}']`).classList.toggle("active", view === name);
  });
  if (name === "library") refreshLibrary();
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

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  const panel = $("#search-state");
  const section = $("#results-section");
  panel.className = "state-panel";
  panel.textContent = "Ranking transcript, entity, action, speaker, and relationship evidence…";
  section.classList.add("hidden");
  try {
    const selectedVideo = $("#video-filter").value;
    const response = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, video_ids: selectedVideo ? [selectedVideo] : [], limit: 10 }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Search failed");
    panel.classList.add("hidden");
    renderResults(payload);
  } catch (error) {
    panel.className = "state-panel error";
    panel.textContent = error.message;
  }
});

function renderResults(payload) {
  const section = $("#results-section");
  const results = $("#results");
  const plan = payload.parsed_query;
  section.classList.remove("hidden");
  $("#result-count").textContent = `${payload.results.length} result${payload.results.length === 1 ? "" : "s"}`;
  const chips = [
    ...(plan.intent ? [`intent · ${plan.intent.replaceAll("_", " ")}`] : []),
    ...plan.visual_entities.map((value) => `visual · ${value}`),
    ...plan.actions.map((value) => `action · ${value}`),
    ...plan.relationships.map((value) => `relation · ${value}`),
    ...plan.relationship_tuples.map((value) => `tuple · ${value.subject} ${value.predicate} ${value.object}`),
    ...plan.spoken_terms.map((value) => `spoken · ${value}`),
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
    return `<article class="result-card">
      <div class="thumbnail"><img src="${escapeHtml(item.thumbnail_url)}" alt="Frame near ${formatTime(item.start_time)}" loading="lazy" onerror="this.style.display='none'" /><span>${formatTime(item.start_time)}–${formatTime(item.end_time)}</span></div>
      <div class="result-main"><p class="eyebrow">RESULT ${index + 1}</p><h3>${escapeHtml(item.video_title)}</h3><p class="reason">${escapeHtml(item.match_reason)}</p><p class="transcript">${escapeHtml(item.transcript || "No speech in this segment.")}</p><div class="evidence">${evidence.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}</div></div>
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
  const citations = answer.supported ? (answer.citations || []) : [];
  card.className = `answer-card ${answer.supported ? "supported" : "unsupported"}`;
  card.innerHTML = `<p class="eyebrow">GROUNDED ANSWER</p>
    <p class="answer-text">${escapeHtml(answer.answer)}</p>
    <div class="answer-citations">${citations.map((citation) => `<button class="answer-citation" data-evidence-id="${escapeHtml(citation.evidence_id)}">${escapeHtml(citation.evidence_id)} \u00b7 ${escapeHtml(citation.video_title)} \u00b7 ${formatTime(citation.start_time)}\u2013${formatTime(citation.end_time)}</button>`).join("")}</div>
    ${answer.supported ? "<small>Synthesized only from the cited ranked evidence.</small>" : ""}`;
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
    });
  }));
}

function openPlayer(item) {
  const dialog = $("#player-dialog");
  const player = $("#player");
  $("#player-title").textContent = item.video_title;
  $("#player-time").textContent = `${formatTime(item.start_time)}–${formatTime(item.end_time)}`;
  $("#player-reason").textContent = item.match_reason;
  $("#player-transcript").textContent = item.transcript || "No transcript for this moment.";
  player.src = item.stream_url;
  player.addEventListener("loadedmetadata", () => { player.currentTime = item.start_time; player.play().catch(() => {}); }, { once: true });
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
    list.innerHTML = state.videos.map((video) => `<article class="video-row"><div><h3>${escapeHtml(video.display_name)}</h3><p>${formatTime(video.duration)} · ${video.width}×${video.height} · ${Number(video.fps).toFixed(2)} fps${video.warnings.length ? ` · ${video.warnings.length} warning(s)` : ""}</p><p>${escapeHtml(video.failure_message || video.current_stage || "")}</p></div><span class="badge ${escapeHtml(video.state)}">${escapeHtml(video.state)}</span><button class="retry" data-id="${escapeHtml(video.video_id)}">${video.state === "FAILED" ? "Retry" : "Reprocess"}</button></article>`).join("");
    $$('.retry').forEach((button) => button.addEventListener("click", () => reprocess(button.dataset.id)));
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
