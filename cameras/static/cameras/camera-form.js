(() => {
  const form = document.querySelector("#camera-form");
  if (!form) return;

  const sourceInputs = [...form.querySelectorAll('input[name="source_type"]')];
  const groups = [...form.querySelectorAll("[data-source-group]")];
  const tokenInput = form.querySelector('input[name="test_token"]');
  const testButton = document.querySelector("#test-connection");
  const result = document.querySelector("#test-result");

  const selectedSource = () => form.querySelector('input[name="source_type"]:checked')?.value || "";

  function updateSourceFields() {
    const source = selectedSource();
    groups.forEach((group) => {
      const visible = group.dataset.sourceGroup === source;
      group.hidden = !visible;
      group.querySelectorAll("input").forEach((input) => {
        input.disabled = !visible;
      });
    });
  }

  function invalidateTest(event) {
    if ([
      "enabled", "name", "location", "buffer_enabled", "buffer_seconds",
      "ai_enabled", "ai_confidence_threshold", "ai_frame_rate", "ai_inference_size",
      "crash_suspicion_threshold", "incident_cooldown_seconds", "ai_post_event_seconds",
    ].includes(event?.target?.name)) return;
    tokenInput.value = "";
    result.className = "test-result";
    result.textContent = "";
  }

  sourceInputs.forEach((input) => input.addEventListener("change", () => {
    invalidateTest();
    updateSourceFields();
  }));
  form.addEventListener("input", invalidateTest);
  updateSourceFields();

  testButton.addEventListener("click", async () => {
    tokenInput.value = "";
    result.className = "test-result show";
    result.textContent = "Testing the source. This can take up to a few seconds…";
    testButton.disabled = true;

    const data = new FormData(form);
    data.set("source_type", selectedSource());
    if (form.dataset.cameraId) data.set("camera_id", form.dataset.cameraId);

    try {
      const response = await fetch(form.dataset.testUrl, {
        method: "POST",
        body: data,
        headers: { "X-Requested-With": "XMLHttpRequest" },
      });
      const payload = await response.json();
      if (response.ok && payload.ok) {
        tokenInput.value = payload.test_token;
        result.className = "test-result show success";
      } else {
        result.className = "test-result show error";
      }
      result.textContent = payload.message || "The connection test did not complete.";
    } catch (_error) {
      result.className = "test-result show error";
      result.textContent = "The server could not complete the test. Check that it is running and try again.";
    } finally {
      testButton.disabled = false;
    }
  });
})();
