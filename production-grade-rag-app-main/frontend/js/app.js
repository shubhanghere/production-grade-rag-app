"use strict";

// ── Element references ──────────────────────────────────────────────────────

const ingestForm    = document.getElementById("ingest-form");
const docSourceEl   = document.getElementById("doc-source");
const docTextEl     = document.getElementById("doc-text");
const ingestBtn     = document.getElementById("ingest-btn");
const ingestStatus  = document.getElementById("ingest-status");
const docListWrap   = document.getElementById("doc-list-wrap");
const docList       = document.getElementById("doc-list");
const fileInput     = document.getElementById("file-input");
const fileNameLabel = document.getElementById("file-name-label");

const chatForm   = document.getElementById("chat-form");
const chatInput  = document.getElementById("chat-input");
const sendBtn    = document.getElementById("send-btn");
const messagesEl = document.getElementById("messages");

// ── Helpers ─────────────────────────────────────────────────────────────────

function setLoading(btn, loading) {
  const label   = btn.querySelector(".btn-label");
  const spinner = btn.querySelector(".spinner");
  btn.disabled  = loading;
  label.classList.toggle("hidden", loading);
  spinner.classList.toggle("hidden", !loading);
}

function showStatus(msg, type /* "success" | "error" */) {
  ingestStatus.textContent = msg;
  ingestStatus.className   = `status-msg ${type}`;
  ingestStatus.classList.remove("hidden");
}

function appendMessage(role /* "user" | "assistant" | "system" */, text, citations = []) {
  // Remove the welcome hint on first real message
  const hint = messagesEl.querySelector(".message-system");
  if (hint && role !== "system") hint.remove();

  const div = document.createElement("div");
  div.className = `message message-${role}`;

  if (role === "assistant") {
    const label = document.createElement("span");
    label.className = "msg-label";
    label.textContent = "Claude";
    div.appendChild(label);
  }

  if (role === "assistant") {
    const body = document.createElement("div");
    body.innerHTML = DOMPurify.sanitize(marked.parse(text));
    div.appendChild(body);
    if (citations.length) div.appendChild(buildSources(citations));
  } else {
    const p = document.createElement("p");
    p.textContent = text;
    div.appendChild(p);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// Only http(s) links become anchors — blocks javascript: and other schemes so
// a malicious ingested URL can't inject.
function isHttpUrl(url) {
  return /^https?:\/\//i.test(url);
}

function buildSources(citations) {
  const wrap = document.createElement("div");
  wrap.className = "message-sources";

  const label = document.createElement("span");
  label.className = "sources-label";
  label.textContent = "Sources";
  wrap.appendChild(label);

  const list = document.createElement("ul");
  list.className = "sources-list";
  for (const c of citations) {
    const li = document.createElement("li");
    if (c.url && isHttpUrl(c.url)) {
      const a = document.createElement("a");
      a.href = c.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = c.title;      // textContent → no HTML injection
      li.appendChild(a);
    } else {
      li.textContent = c.title;
    }
    list.appendChild(li);
  }
  wrap.appendChild(list);
  return wrap;
}

function appendLoadingBubble() {
  const div = document.createElement("div");
  div.className = "message message-loading";

  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");

  const txt = document.createElement("span");
  txt.textContent = "Thinking…";

  div.appendChild(spinner);
  div.appendChild(txt);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// ── File picker ─────────────────────────────────────────────────────────────

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileNameLabel.textContent = file.name;
});

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/v1/documents/upload", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? `Server error ${res.status}`);
  return data;
}

// ── Ingest ───────────────────────────────────────────────────────────────────

ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  ingestStatus.classList.add("hidden");

  const file   = fileInput.files[0];
  const source = docSourceEl.value.trim() || file?.name || "uploaded";
  const text   = docTextEl.value.trim();

  if (!file && !text) {
    showStatus("Choose a file or paste some text before ingesting.", "error");
    return;
  }

  setLoading(ingestBtn, true);

  try {
    // File upload path — server handles parsing (including PDF extraction).
    // Paste path — send raw text as JSON.
    const data = file
      ? await uploadFile(file)
      : await (async () => {
          const res = await fetch("/api/v1/documents", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ text, source }),
          });
          const d = await res.json();
          if (!res.ok) throw new Error(d.detail ?? `Server error ${res.status}`);
          return d;
        })();

    showStatus(`✓ ${data.message}`, "success");

    const li = document.createElement("li");
    li.className = "doc-list-item";
    li.textContent = data.message.replace(" ingested successfully.", "");
    docList.appendChild(li);
    docListWrap.classList.remove("hidden");

    // Clear the form
    docSourceEl.value         = "";
    docTextEl.value           = "";
    fileInput.value           = "";
    fileNameLabel.textContent = "No file chosen";

  } catch (err) {
    showStatus(`Error: ${err.message}`, "error");
  } finally {
    setLoading(ingestBtn, false);
  }
});

// ── Chat ─────────────────────────────────────────────────────────────────────

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = chatInput.value.trim();
  if (!question) return;

  chatInput.value = "";
  chatInput.style.height = "";

  appendMessage("user", question);
  const loadingBubble = appendLoadingBubble();

  setLoading(sendBtn, true);

  try {
    const res = await fetch("/api/v1/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ question }),
    });

    const data = await res.json();
    loadingBubble.remove();

    if (!res.ok) {
      throw new Error(data.detail ?? `Server error ${res.status}`);
    }

    appendMessage("assistant", data.answer, data.citations || []);

  } catch (err) {
    loadingBubble.remove();
    appendMessage("system", `Error: ${err.message}`);
  } finally {
    setLoading(sendBtn, false);
    chatInput.focus();
  }
});

// Submit on Enter (Shift+Enter for newline)
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

// Auto-grow the chat textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
});
