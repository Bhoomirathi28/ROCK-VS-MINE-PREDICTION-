/* Rock or Mine - front-end logic (no framework, no build step). */
(() => {
  "use strict";

  const N = 60;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel) => document.querySelector(sel);

  const state = {
    source: "sample",
    current: null,       // 60 values currently loaded
    batch: null,         // array of readings from a CSV
    trueLabel: null,
    means: null,
  };

  /* ----------------------------------------------------------- tide pool */
  // The reading is drawn as a ring of bubbles in a pool: angle = band, distance = energy.
  const scope = (() => {
    const canvas = $("#scope");
    const ctx = canvas.getContext("2d");
    const fig = canvas.parentElement;
    let size, cx, R, t = 0, blips = [], pulse = null, shown = 0, last = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 400;
      canvas.width = w * dpr; canvas.height = w * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      size = w; cx = w / 2; R = w / 2 - 14;
      if (reduceMotion) draw();
    }

    function setReading(values) {
      blips = values ? values.map((v, i) => ({ a: (i / N) * Math.PI * 2 - Math.PI / 2, r: 0.16 + v * 0.78, v })) : [];
      shown = reduceMotion ? 1 : 0;
      if (reduceMotion) draw();
    }

    function ping(label) {
      fig.dataset.state = label;
      pulse = { k: 0, colour: label === "M" ? "255,255,255" : "255,214,150" };
      if (reduceMotion) { pulse = null; draw(); }
    }

    function draw() {
      ctx.clearRect(0, 0, size, size);

      // soft glow + pool
      ctx.save();
      ctx.shadowColor = "rgba(255,255,255,0.55)"; ctx.shadowBlur = 30;
      const g = ctx.createRadialGradient(cx * 0.85, cx * 0.8, R * 0.1, cx, cx, R);
      g.addColorStop(0, "rgba(140,232,240,0.55)");
      g.addColorStop(0.7, "rgba(23,173,207,0.35)");
      g.addColorStop(1, "rgba(10,93,156,0.35)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cx, cx, R, 0, Math.PI * 2); ctx.fill();
      ctx.restore();

      // rim
      ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(cx, cx, R, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 10;
      ctx.beginPath(); ctx.arc(cx, cx, R - 7, 0, Math.PI * 2); ctx.stroke();

      // slow ripples
      for (let i = 0; i < 3; i++) {
        const k = ((t / 240) + i / 3) % 1;
        ctx.strokeStyle = `rgba(255,255,255,${0.35 * (1 - k)})`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cx, R * (0.12 + k * 0.85), 0, Math.PI * 2); ctx.stroke();
      }

      // the echo
      if (blips.length) {
        const n = Math.ceil(blips.length * shown);
        ctx.fillStyle = "rgba(255,255,255,0.14)"; ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1.5;
        ctx.beginPath();
        blips.slice(0, n).forEach((b, i) => {
          const x = cx + Math.cos(b.a) * R * b.r, y = cx + Math.sin(b.a) * R * b.r;
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        if (n === blips.length) ctx.closePath();
        ctx.fill(); ctx.stroke();
        blips.slice(0, n).forEach((b) => {
          const x = cx + Math.cos(b.a) * R * b.r, y = cx + Math.sin(b.a) * R * b.r, r = 2 + b.v * 4.5;
          ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        });
      }

      if (pulse) {
        ctx.strokeStyle = `rgba(${pulse.colour},${0.9 * (1 - pulse.k)})`; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.arc(cx, cx, R * (0.15 + pulse.k * 0.85), 0, Math.PI * 2); ctx.stroke();
      }
    }

    function tick(now) {
      const dt = last ? Math.min((now - last) / 16.7, 4) : 1;   // frame-rate independent
      last = now;
      t += dt;
      if (shown < 1) shown = Math.min(1, shown + 0.03 * dt);
      if (pulse) { pulse.k += 0.02 * dt; if (pulse.k >= 1) pulse = null; }
      draw();
      requestAnimationFrame(tick);
    }

    window.addEventListener("resize", resize);
    resize();
    if (!reduceMotion) requestAnimationFrame(tick); else draw();
    return { setReading, ping, reset: () => { delete fig.dataset.state; } };
  })();

  /* ------------------------------------------------------- echo profile */
  function drawProfile(values) {
    const svg = $("#profile");
    const W = 600, H = 180, pad = 6;
    const x = (i) => pad + (i / (N - 1)) * (W - pad * 2);
    const y = (v) => H - pad - v * (H - pad * 2);
    const line = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const area = (vals) => `${line(vals)}L${x(N - 1)},${H - pad}L${x(0)},${H - pad}Z`;
    let out = "";
    for (let k = 1; k <= 3; k++) out += `<line x1="0" x2="${W}" y1="${y(k / 4)}" y2="${y(k / 4)}" stroke="#DCEEF3" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
    if (state.means) {
      out += `<path d="${area(state.means.R)}" fill="#C9A86A" fill-opacity=".28"/>`;
      out += `<path d="${area(state.means.M)}" fill="#17ADCF" fill-opacity=".28"/>`;
    }
    if (values) {
      out += `<path d="${line(values)}" fill="none" stroke="#0A5D9C" stroke-width="2.4" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
    }
    svg.innerHTML = out;
  }

  function sparkline(values) {
    const pts = values.map((v, i) => `${(i / (N - 1)) * 100},${28 - v * 26}`).join(" ");
    return `<svg viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg>`;
  }

  /* --------------------------------------------------------------- input */
  function loadReading(values, trueLabel = null) {
    state.current = values; state.batch = null; state.trueLabel = trueLabel;
    scope.setReading(values); scope.reset();
    $("#scopeVerdict").textContent = "Echo received";
    $("#scopeSub").textContent = "Ready to classify";
    drawProfile(values);
    $("#error").textContent = "";
  }

  function parseValues(text) {
    return text.split(/[\s,;]+/).filter(Boolean).map(Number);
  }

  // Tabs (arrow-key navigation per WAI-ARIA)
  const tabs = [...document.querySelectorAll('[role="tab"][aria-controls]')];
  function selectTab(tab) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    state.source = tab.id.replace("tab-", "");
    $("#error").textContent = "";
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next); next.focus();
    });
  });

  // Samples
  fetch("/api/samples").then((r) => r.json()).then((data) => {
    state.means = data.class_means;
    const box = $("#samples");
    data.samples.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "sample"; b.setAttribute("aria-pressed", "false");
      b.innerHTML = `${sparkline(s.features)}<span>Reading ${i + 1}</span>`;
      b.setAttribute("aria-label", `Test reading ${i + 1}`);
      b.addEventListener("click", () => {
        box.querySelectorAll(".sample").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        loadReading(s.features, s.true_class);
      });
      box.appendChild(b);
    });
    drawProfile(null);
  }).catch(() => { $("#error").textContent = "Couldn't load test samples. Check that the server is running."; });

  // Paste
  $("#values").addEventListener("input", (e) => {
    const vals = parseValues(e.target.value);
    const c = $("#valueCount");
    c.textContent = `${vals.length} of ${N} values`;
    c.classList.toggle("ok", vals.length === N);
    if (vals.length === N && vals.every((v) => v >= 0 && v <= 1)) loadReading(vals);
  });

  // CSV upload
  function readFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const rows = String(reader.result).split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
        .map((l) => l.split(",").slice(0, N).map(Number))
        .filter((r) => r.length === N && r.every((v) => Number.isFinite(v)));
      if (!rows.length) {
        $("#fileInfo").textContent = "";
        $("#error").textContent = "No usable rows found. Each row needs 60 numeric values separated by commas.";
        return;
      }
      loadReading(rows[0]);
      state.batch = rows.length > 1 ? rows : null;
      $("#fileInfo").textContent = `${file.name}: ${rows.length} reading${rows.length > 1 ? "s" : ""} loaded`;
      $("#fileInfo").classList.add("ok");
    };
    reader.readAsText(file);
  }
  $("#file").addEventListener("change", (e) => readFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));

  /* ------------------------------------------------------------ classify */
  const pct = (v) => `${(v * 100).toFixed(0)}%`;

  function renderResult(r) {
    $("#resultEmpty").hidden = true; $("#resultBody").hidden = false;
    const v = $("#verdict");
    v.textContent = r.class_name; v.className = `verdict is-${r.label}`;
    const c = r.consensus;
    const agree = r.label === "M" ? c.mine_votes : c.rock_votes;
    $("#verdictMeta").textContent = `${pct(r.confidence)} confident. ${r.model_name}, ${agree} of 4 models agree.`;
    $("#meterFill").style.width = pct(r.probability_mine);
    $("#votes").innerHTML = r.all_models.map((m) => `
      <li class="is-${m.label}${m.model === r.model ? " is-chosen" : ""}">
        <span>${m.model_name.replace("Support Vector Machine", "SVM").replace("K-Nearest Neighbours", "KNN")}</span>
        <span class="bar" title="Probability of mine"><i style="width:${pct(m.probability_mine)}"></i></span>
        <b>${m.label === "M" ? "Mine" : "Rock"}</b>
      </li>`).join("");
    const t = $("#truth");
    if (state.trueLabel) {
      const right = state.trueLabel === r.class_name;
      t.textContent = `This test reading is actually a ${state.trueLabel.toLowerCase()}. ${right ? "The model got it right." : "The model got this one wrong."}`;
    } else t.textContent = "";
    scope.ping(r.label);
    $("#scopeVerdict").textContent = r.class_name;
    $("#scopeSub").textContent = `${pct(r.confidence)} confident`;
  }

  function renderBatch(results) {
    const box = $("#batch"); box.hidden = false;
    box.querySelector("tbody").innerHTML = results.map((r) => r.error
      ? `<tr><td>${r.row}</td><td colspan="3">${r.error}</td></tr>`
      : `<tr><td>${r.row}</td><td>${r.class_name}</td><td>${pct(r.confidence)}</td><td>${Math.max(r.consensus.mine_votes, r.consensus.rock_votes)} of 4</td></tr>`).join("");
  }

  async function post(url, body) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "The server couldn't classify this reading.");
    return data;
  }

  $("#classify").addEventListener("click", async () => {
    const err = $("#error"); err.textContent = "";
    if (state.source === "paste") {
      const vals = parseValues($("#values").value);
      if (vals.length !== N) { err.textContent = `Enter exactly ${N} values. You have ${vals.length}.`; return; }
      state.current = vals; state.trueLabel = null;
    }
    if (!state.current) {
      err.textContent = state.source === "file" ? "Upload a CSV file first." : "Pick a test reading first.";
      return;
    }
    const model = document.querySelector('input[name="model"]:checked').value;
    const btn = $("#classify"); btn.disabled = true; btn.textContent = "Classifying…";
    try {
      renderResult(await post("/api/predict", { features: state.current, model }));
      if (state.source === "file" && state.batch) {
        renderBatch((await post("/api/predict/batch", { readings: state.batch, model })).results);
      } else $("#batch").hidden = true;
    } catch (e) {
      err.textContent = e.message;
    } finally {
      btn.disabled = false; btn.textContent = "Classify reading";
    }
  });

  /* -------------------------------------------------------------- charts */
  document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-selected", "false"));
    chip.setAttribute("aria-selected", "true");
    const img = $("#chartImg"); img.src = chip.dataset.src; img.alt = `${chip.textContent} chart`;
  }));

  /* ----------------------------------------------------------------- PWA */
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js").catch(() => {}));
  }
})();
