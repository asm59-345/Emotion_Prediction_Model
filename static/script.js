document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const EMOJI = {
    sadness: "😢",
    joy: "😄",
    love: "❤️",
    anger: "😠",
    fear: "😨",
    surprise: "😲",
  };

  const el = {
    statusDot: document.getElementById("statusDot"),
    serverStatusText: document.getElementById("serverStatusText"),
    textInput: document.getElementById("textInput"),
    charCount: document.getElementById("charCount"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    errorMsg: document.getElementById("errorMsg"),
    orb: document.getElementById("orb"),
    orbEmoji: document.getElementById("orbEmoji"),
    resultSection: document.getElementById("resultSection"),
    emotionWord: document.getElementById("emotionWord"),
    emotionEmoji: document.getElementById("emotionEmoji"),
    confidenceText: document.getElementById("confidenceText"),
    latencyTag: document.getElementById("latencyTag"),
    polarityBadge: document.getElementById("polarityBadge"),
    intensityBadge: document.getElementById("intensityBadge"),
    heatmapWords: document.getElementById("heatmapWords"),
    echoedText: document.getElementById("echoedText"),
    barsContainer: document.getElementById("barsContainer"),
    micBtn: document.getElementById("micBtn"),
    micStatusText: document.getElementById("micStatusText"),
    inputEyebrow: document.getElementById("inputEyebrow"),
    // Tabs & sections
    modeTabs: document.querySelectorAll(".tab-btn"),
    consoleSection: document.getElementById("console"),
    uploadSection: document.getElementById("uploadSection"),
    flowSection: document.getElementById("flowSection"),
    timelineContainer: document.getElementById("timelineContainer"),
    dominantEmotionTitle: document.getElementById("dominantEmotionTitle"),
    // Upload elements
    fileInput: document.getElementById("fileInput"),
    browseFileBtn: document.getElementById("browseFileBtn"),
    dropzone: document.getElementById("dropzone"),
    uploadResult: document.getElementById("uploadResult"),
    // Analytics Modal
    analyticsModal: document.getElementById("analyticsModal"),
    analyticsToggleBtn: document.getElementById("analyticsToggleBtn"),
    closeAnalyticsBtn: document.getElementById("closeAnalyticsBtn"),
    totalPredsVal: document.getElementById("totalPredsVal"),
    avgConfVal: document.getElementById("avgConfVal"),
    topEmotionVal: document.getElementById("topEmotionVal"),
    historyTableBody: document.getElementById("historyTableBody"),
  };

  let modelReady = false;
  let currentMode = "single";
  let recognition = null;
  let isRecording = false;

  /* ---------------------------------------------------------------
     Health check
  --------------------------------------------------------------- */
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();

      modelReady = !!data.model_loaded;
      if (modelReady) {
        setStatus("live", "Moodline Pro online • BiGRU Inference Engine ready");
      } else {
        setStatus("warming", "loading model weights…");
        setTimeout(checkHealth, 3000);
      }
    } catch (e) {
      setStatus("down", "can't reach server");
      setTimeout(checkHealth, 5000);
    }
    syncButtonState();
  }

  function setStatus(kind, text) {
    if (el.statusDot) el.statusDot.className = "brand-mark " + kind;
    if (el.serverStatusText) el.serverStatusText.textContent = text;
  }

  /* ---------------------------------------------------------------
     Mode Switching
  --------------------------------------------------------------- */
  el.modeTabs.forEach(tab => {
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      el.modeTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentMode = tab.getAttribute("data-mode");
      switchMode(currentMode);
    });
  });

  function switchMode(mode) {
    hideError();
    if (el.resultSection) el.resultSection.hidden = true;
    if (el.flowSection) el.flowSection.hidden = true;
    if (el.uploadSection) el.uploadSection.hidden = true;
    if (el.consoleSection) el.consoleSection.hidden = false;

    if (mode === "single") {
      el.inputEyebrow.textContent = "INPUT (SENTENCE & SALIENCY)";
      el.textInput.placeholder = "Write a sentence and let it speak for itself\n— e.g. “I was terrified at first, but then I felt overwhelming joy.”";
      el.analyzeBtn.querySelector(".btn-label").textContent = "Read the mood";
    } else if (mode === "flow") {
      el.inputEyebrow.textContent = "INPUT (PARAGRAPH NARRATIVE ARC)";
      el.textInput.placeholder = "Paste a multi-sentence story or conversational paragraph to analyze the emotional flow across each sentence.\n— e.g. “I felt terrified in the dark. Suddenly my friends shouted surprise! I was overwhelmed with joy.”";
      el.analyzeBtn.querySelector(".btn-label").textContent = "Analyze Flow";
    } else if (mode === "upload") {
      el.consoleSection.hidden = true;
      el.uploadSection.hidden = false;
    }
    syncButtonState();
  }

  /* ---------------------------------------------------------------
     Input handling
  --------------------------------------------------------------- */
  el.textInput.addEventListener("input", () => {
    el.charCount.textContent = el.textInput.value.length;
    syncButtonState();
  });

  el.textInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runAnalysis();
    }
  });

  function syncButtonState() {
    const hasText = el.textInput.value.trim().length > 0;
    el.analyzeBtn.disabled = !hasText || !modelReady;
  }

  el.analyzeBtn.addEventListener("click", (e) => {
    e.preventDefault();
    runAnalysis();
  });

  /* ---------------------------------------------------------------
     Voice Dictation (Speech-to-Text)
  --------------------------------------------------------------- */
  if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript + " ";
        }
      }
      if (finalTranscript) {
        el.textInput.value = (el.textInput.value + " " + finalTranscript).trim();
        el.charCount.textContent = el.textInput.value.length;
        syncButtonState();
      }
    };

    recognition.onerror = () => stopRecording();
    recognition.onend = () => stopRecording();

    el.micBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });
  } else {
    if (el.micBtn) el.micBtn.style.display = "none";
  }

  function startRecording() {
    try {
      recognition.start();
      isRecording = true;
      el.micBtn.classList.add("recording");
      el.micStatusText.textContent = "Listening...";
    } catch (e) {}
  }

  function stopRecording() {
    try {
      recognition.stop();
    } catch (e) {}
    isRecording = false;
    if (el.micBtn) {
      el.micBtn.classList.remove("recording");
      el.micStatusText.textContent = "Dictate";
    }
  }

  /* ---------------------------------------------------------------
     Analysis Execution
  --------------------------------------------------------------- */
  async function runAnalysis() {
    const text = el.textInput.value.trim();
    if (!text || !modelReady) return;

    hideError();
    enterThinking();

    try {
      if (currentMode === "flow") {
        const res = await fetch("/api/v1/predict/flow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error("Paragraph flow analysis failed.");
        const data = await res.json();
        renderFlowResult(data);
      } else {
        const res = await fetch("/api/v1/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, explain: true }),
        });
        if (!res.ok) throw new Error("Prediction request failed.");
        const data = await res.json();
        renderResult(data, text);
      }
    } catch (err) {
      exitThinking(false);
      showError(err.message || "Something went wrong. Please try again.");
    }
  }

  function enterThinking() {
    el.analyzeBtn.classList.add("loading");
    el.analyzeBtn.disabled = true;
    el.orb.classList.remove("settled");
    el.orb.classList.add("thinking");
    el.orbEmoji.style.opacity = "0";
  }

  function exitThinking(success) {
    el.analyzeBtn.classList.remove("loading");
    syncButtonState();
    el.orb.classList.remove("thinking");
    if (!success) {
      el.orbEmoji.textContent = "✎";
      el.orbEmoji.style.opacity = "1";
    }
  }

  function renderResult(data, originalText) {
    const emotion = data.predicted_emotion;
    const emoji = data.emoji || EMOJI[emotion] || "🙂";

    document.body.setAttribute("data-emotion", emotion);
    el.orb.classList.add("settled");
    el.orbEmoji.textContent = emoji;
    el.orbEmoji.style.opacity = "1";
    exitThinking(true);

    el.emotionWord.textContent = capitalize(emotion);
    el.emotionEmoji.textContent = emoji;
    el.confidenceText.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
    el.latencyTag.textContent = `⚡ ${data.latency_ms}ms`;
    el.polarityBadge.textContent = capitalize(data.sentiment.polarity);
    el.intensityBadge.textContent = `Intensity ${(data.sentiment.intensity * 100).toFixed(0)}%`;
    el.echoedText.textContent = `“${originalText}”`;

    // Render Word Saliency / Heatmap
    el.heatmapWords.innerHTML = "";
    if (data.attributions && data.attributions.length > 0) {
      data.attributions.forEach(item => {
        const chip = document.createElement("span");
        chip.className = "word-chip";
        if (item.impact_score > 0.15) {
          chip.classList.add("high-impact");
        }
        chip.textContent = item.word;
        chip.title = `Attribution impact: ${(item.impact_score * 100).toFixed(1)}%`;
        el.heatmapWords.appendChild(chip);
      });
    }

    renderBars(data.all_probabilities);

    el.flowSection.hidden = true;
    el.resultSection.hidden = false;
    el.resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderFlowResult(data) {
    exitThinking(true);
    el.resultSection.hidden = true;
    el.flowSection.hidden = false;
    el.dominantEmotionTitle.textContent = `Dominant Emotion: ${capitalize(data.dominant_emotion)} (${data.total_sentences} sentences)`;

    el.timelineContainer.innerHTML = "";
    data.narrative_arc.forEach(step => {
      const card = document.createElement("div");
      card.className = "timeline-step";
      card.innerHTML = `
        <span class="step-badge">${step.emoji}</span>
        <div class="step-body">
          <p class="step-text">“${step.sentence}”</p>
          <span class="step-meta">${capitalize(step.emotion)} • ${(step.confidence * 100).toFixed(1)}% confidence • Polarity: ${capitalize(step.sentiment.polarity)}</span>
        </div>
      `;
      el.timelineContainer.appendChild(card);
    });

    el.flowSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderBars(probs) {
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    el.barsContainer.innerHTML = "";

    entries.forEach(([label, value], i) => {
      const pct = value * 100;
      const row = document.createElement("div");
      row.className = `bar-row bar-${label}`;
      row.innerHTML = `
        <span class="bar-label">${EMOJI[label] || ""} ${label}</span>
        <span class="bar-track"><span class="bar-fill"></span></span>
        <span class="bar-pct">${pct.toFixed(1)}%</span>
      `;
      el.barsContainer.appendChild(row);

      const fill = row.querySelector(".bar-fill");
      setTimeout(() => {
        fill.style.width = pct + "%";
      }, 50 + i * 50);
    });
  }

  /* ---------------------------------------------------------------
     File Upload (Click & Drag-and-Drop)
  --------------------------------------------------------------- */
  if (el.browseFileBtn && el.fileInput) {
    el.browseFileBtn.addEventListener("click", (e) => {
      e.preventDefault();
      el.fileInput.click();
    });
    el.fileInput.addEventListener("change", () => {
      if (el.fileInput.files.length > 0) {
        handleFileUpload(el.fileInput.files[0]);
      }
    });
  }

  if (el.dropzone) {
    el.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.dropzone.style.borderColor = "var(--accent)";
    });
    el.dropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      el.dropzone.style.borderColor = "var(--bg-card-border)";
    });
    el.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropzone.style.borderColor = "var(--bg-card-border)";
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    el.uploadResult.hidden = false;
    el.uploadResult.innerHTML = "<p>Processing file batches with BiGRU engine...</p>";

    try {
      const res = await fetch("/api/v1/predict/upload", {
        method: "POST",
        body: formData
      });
      if (!res.ok) throw new Error("Failed to process file.");
      const data = await res.json();

      let breakdownHtml = Object.entries(data.emotion_distribution)
        .map(([emo, count]) => `<li><b>${EMOJI[emo] || ""} ${capitalize(emo)}:</b> ${count} samples</li>`)
        .join("");

      el.uploadResult.innerHTML = `
        <h3 style="color:var(--accent); margin-bottom:0.5rem;">Processed ${data.total_rows_processed} lines in ${data.processing_time_ms}ms</h3>
        <p><b>Dominant Emotion:</b> ${capitalize(data.dominant_emotion)} | <b>Avg Confidence:</b> ${(data.average_confidence * 100).toFixed(1)}%</p>
        <ul style="margin: 0.8rem 0; padding-left: 1.2rem;">${breakdownHtml}</ul>
      `;
    } catch (e) {
      el.uploadResult.innerHTML = `<p style="color:var(--anger);">${e.message}</p>`;
    }
  }

  /* ---------------------------------------------------------------
     Analytics Modal
  --------------------------------------------------------------- */
  if (el.analyticsToggleBtn) {
    el.analyticsToggleBtn.addEventListener("click", (e) => {
      e.preventDefault();
      openAnalytics();
    });
  }

  if (el.closeAnalyticsBtn) {
    el.closeAnalyticsBtn.addEventListener("click", (e) => {
      e.preventDefault();
      el.analyticsModal.hidden = true;
    });
  }

  if (el.analyticsModal) {
    el.analyticsModal.addEventListener("click", (e) => {
      if (e.target === el.analyticsModal) {
        el.analyticsModal.hidden = true;
      }
    });
  }

  async function openAnalytics() {
    el.analyticsModal.hidden = false;
    try {
      const [sumRes, histRes] = await Promise.all([
        fetch("/api/v1/analytics/summary"),
        fetch("/api/v1/analytics/history?limit=15")
      ]);

      if (sumRes.ok) {
        const sum = await sumRes.json();
        el.totalPredsVal.textContent = sum.total_predictions;
        el.avgConfVal.textContent = `${(sum.average_confidence * 100).toFixed(1)}%`;
        el.topEmotionVal.textContent = sum.most_frequent_emotion ? capitalize(sum.most_frequent_emotion) : "—";
      }

      if (histRes.ok) {
        const history = await histRes.json();
        el.historyTableBody.innerHTML = "";
        if (history.length === 0) {
          el.historyTableBody.innerHTML = `<tr><td colspan="5" class="empty-row">No records logged yet.</td></tr>`;
        } else {
          history.forEach(r => {
            const tr = document.createElement("tr");
            const timeStr = r.created_at ? new Date(r.created_at).toLocaleTimeString() : "—";
            tr.innerHTML = `
              <td>${timeStr}</td>
              <td style="max-width: 250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${r.input_text}">${r.input_text}</td>
              <td>${EMOJI[r.predicted_emotion] || ""} ${capitalize(r.predicted_emotion)}</td>
              <td>${(r.confidence * 100).toFixed(1)}%</td>
              <td><button class="replay-btn" type="button" data-text="${encodeURIComponent(r.input_text)}">Replay</button></td>
            `;
            el.historyTableBody.appendChild(tr);
          });

          el.historyTableBody.querySelectorAll(".replay-btn").forEach(btn => {
            btn.addEventListener("click", () => {
              el.textInput.value = decodeURIComponent(btn.dataset.text);
              el.charCount.textContent = el.textInput.value.length;
              el.analyticsModal.hidden = true;
              syncButtonState();
              runAnalysis();
            });
          });
        }
      }
    } catch (e) {}
  }

  function showError(msg) {
    el.errorMsg.textContent = msg;
    el.errorMsg.hidden = false;
  }
  function hideError() {
    el.errorMsg.hidden = true;
  }

  function capitalize(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
  }

  /* Boot */
  checkHealth();
});