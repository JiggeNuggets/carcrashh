(() => {
  const layout = document.querySelector(".preview-layout");
  const statusBox = document.querySelector("#stream-status");
  const stream = document.querySelector("#live-stream");
  if (!layout || !statusBox || !stream) return;

  const feedback = document.querySelector("#action-feedback");
  const snapshotButton = document.querySelector("#btn-snapshot");
  const clipButton = document.querySelector("#btn-trigger-clip");
  const aiToggleButton = document.querySelector("#btn-ai-toggle");
  const previewMode = document.querySelector("#preview-mode");
  const postEventSelect = document.querySelector("#post-event-seconds");
  const csrfToken = layout.dataset.csrf || "";
  let timer;

  const statusNames = {
    connected: "Connected",
    connecting: "Connecting",
    disconnected: "Disconnected",
    reconnecting: "Reconnecting",
    disabled: "Disabled",
    error: "Error",
  };

  function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = value;
  }

  function showFeedback(message, isError = false) {
    if (!feedback) return;
    feedback.className = `action-feedback show ${isError ? "error" : "success"}`;
    feedback.textContent = message;
    window.setTimeout(() => {
      feedback.className = "action-feedback";
      feedback.textContent = "";
    }, 7000);
  }

  async function refreshStatus() {
    try {
      const response = await fetch(layout.dataset.statusUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error("Status unavailable");
      const data = await response.json();
      const label = statusNames[data.status] || "Unknown";
      statusBox.dataset.status = data.status;
      statusBox.querySelector("strong").textContent = label;
      statusBox.querySelector("small").textContent = data.message || "No status details are available.";
      setText("#health-status", label);
      setText("#health-name", data.camera_name || "—");
      setText("#health-source", data.source_type_label || "—");
      setText("#health-fps", Number(data.fps || 0).toFixed(1));
      setText("#health-worker", statusNames[data.current_worker_status] || data.current_worker_status || "Stopped");
      setText("#buf-state", data.buffer_enabled ? "Enabled" : "Disabled");
      setText("#buf-window", `${data.buffer_seconds || 0}s`);
      if (data.buffer) {
        setText("#buf-cached", `${data.buffer.seconds_cached || 0}s`);
        setText("#buf-frames", data.buffer.count || 0);
      }
      const lastFrame = data.last_frame_at
        ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" }).format(new Date(data.last_frame_at))
        : "Waiting for a frame";
      setText("#health-last-frame", lastFrame);
    } catch (_error) {
      statusBox.dataset.status = "error";
      statusBox.querySelector("strong").textContent = "Status check failed";
      statusBox.querySelector("small").textContent = "The server did not return worker health.";
    }
  }

  async function refreshAIStatus() {
    try {
      const response = await fetch(layout.dataset.aiStatusUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error("AI status unavailable");
      const data = await response.json();
      const label = {
        disabled: "Disabled",
        stopped: "Stopped",
        loading: "Loading model",
        running: "Running",
        unavailable: "Model unavailable",
        error: "Processing error",
      }[data.status] || data.status;
      setText("#ai-status", label);
      setText("#ai-message", data.message || "No AI status details.");
      setText("#ai-vehicles", data.detected_vehicles || 0);
      setText("#ai-fps", Number(data.inference_fps || 0).toFixed(2));
      setText("#ai-queue", `${data.queue_depth || 0} / ${data.queue_capacity || 2}`);
      setText("#ai-dropped", data.dropped_frames || 0);
      setText("#ai-score", Number(data.latest_suspicion_score || 0).toFixed(2));
      const temporalText = data.latest_temporal_score == null
        ? ({ disabled: "Off", unavailable: "Unavailable", warming_up: "Warming up" }[data.temporal_status] || "Waiting")
        : Number(data.latest_temporal_score).toFixed(2);
      setText("#ai-temporal", temporalText);
      const eventText = data.latest_event
        ? `${Number(data.latest_event.score).toFixed(2)} · ${data.latest_event.severity || "undetermined"}`
        : "None";
      setText("#ai-latest-event", eventText);
      if (previewMode) previewMode.querySelector('option[value="ai"]').disabled = !data.enabled;
      if (aiToggleButton) aiToggleButton.textContent = data.enabled ? "Disable AI" : "Enable AI";
      if (["unavailable", "error", "disabled", "stopped"].includes(data.status) && previewMode?.value === "ai") {
        previewMode.value = "normal";
        stream.src = stream.dataset.normalSrc;
        showFeedback(`${data.message} Switched to normal live view.`, data.status !== "disabled");
      }
    } catch (_error) {
      setText("#ai-status", "Status unavailable");
      setText("#ai-message", "The server did not return AI worker health.");
    }
  }

  async function postAction(url, fields = {}) {
    const formData = new FormData();
    formData.append("csrfmiddlewaretoken", csrfToken);
    Object.entries(fields).forEach(([key, value]) => formData.append(key, value));
    const response = await fetch(url, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.message || "The action failed.");
    return payload;
  }

  snapshotButton?.addEventListener("click", async () => {
    snapshotButton.disabled = true;
    try {
      const payload = await postAction(layout.dataset.snapshotUrl);
      showFeedback(payload.message);
    } catch (error) {
      showFeedback(error.message || "Could not capture a snapshot.", true);
    } finally {
      snapshotButton.disabled = false;
    }
  });

  clipButton?.addEventListener("click", async () => {
    const postSeconds = Number(postEventSelect?.value || 8);
    if (!window.confirm(`Generate a test clip with the rolling pre-event buffer and ${postSeconds}s of post-event footage?`)) return;
    clipButton.disabled = true;
    const original = clipButton.innerHTML;
    clipButton.innerHTML = '<span class="status-spinner mini"></span> Recording test clip...';
    try {
      const payload = await postAction(layout.dataset.triggerUrl, { post_seconds: postSeconds });
      showFeedback(payload.message);
    } catch (error) {
      showFeedback(error.message || "Could not start the test clip.", true);
    } finally {
      window.setTimeout(() => {
        clipButton.disabled = false;
        clipButton.innerHTML = original;
      }, Math.max(1500, postSeconds * 1000));
    }
  });

  previewMode?.addEventListener("change", () => {
    stream.src = previewMode.value === "ai" ? stream.dataset.aiSrc : stream.dataset.normalSrc;
  });

  aiToggleButton?.addEventListener("click", async () => {
    aiToggleButton.disabled = true;
    try {
      const payload = await postAction(layout.dataset.aiToggleUrl);
      showFeedback(payload.message);
      if (!payload.enabled && previewMode) {
        previewMode.value = "normal";
        stream.src = stream.dataset.normalSrc;
      }
      await refreshAIStatus();
    } catch (error) {
      showFeedback(error.message || "Could not update AI detection.", true);
    } finally {
      aiToggleButton.disabled = false;
    }
  });

  stream.addEventListener("error", () => {
    refreshStatus();
    if (previewMode?.value === "ai") refreshAIStatus();
  });
  timer = window.setInterval(() => {
    refreshStatus();
    refreshAIStatus();
  }, 2000);
  refreshStatus();
  refreshAIStatus();
  window.addEventListener("pagehide", () => {
    window.clearInterval(timer);
    stream.removeAttribute("src");
  });
})();
