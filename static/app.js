const SAMPLE_RATE = 16000;
const FIELDS = ["name", "system_prompt", "call_greeting", "chat_greeting", "voice_id", "language", "llm_model"];
const OPTIONAL_FIELDS = ["voice_id", "llm_model"];

const $ = (id) => document.getElementById(id);
const wsUrl = (path) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

let assistants = [];
let current = null;
let chat = null;
let call = null;

// ---------- Assistants ----------

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.status === 204 ? null : res.json();
}

async function loadAssistants() {
  assistants = await api("/api/assistants");
  renderList();
  if (!current && assistants.length) select(assistants[0]);
  const editorOpen = !$("editor").classList.contains("hidden");
  $("empty-state").classList.toggle("hidden", assistants.length > 0 || editorOpen);
}

function renderList() {
  const list = $("assistant-list");
  list.innerHTML = "";
  for (const a of assistants) {
    const li = document.createElement("li");
    li.textContent = a.name;
    li.classList.toggle("active", current && current.id === a.id);
    li.onclick = () => select(a);
    list.appendChild(li);
  }
}

function select(assistant) {
  stopCall();
  closeChat();
  current = assistant;
  $("empty-state").classList.add("hidden");
  $("editor").classList.remove("hidden");
  $("editor-title").textContent = assistant ? assistant.name : "New assistant";
  for (const field of FIELDS) $(field).value = assistant ? assistant[field] ?? "" : field === "language" ? "en" : "";
  $("delete-assistant").classList.toggle("hidden", !assistant);
  $("session-area").classList.toggle("hidden", !assistant);
  $("call-transcript").innerHTML = "";
  $("chat-messages").innerHTML = "";
  renderList();
  if (assistant && activeTab() === "chat") openChat();
}

$("new-assistant").onclick = () => select(null);

$("assistant-form").onsubmit = async (event) => {
  event.preventDefault();
  const body = {};
  for (const field of FIELDS) {
    const value = $(field).value.trim();
    body[field] = OPTIONAL_FIELDS.includes(field) && !value ? null : value;
  }
  const saved = current
    ? await api(`/api/assistants/${current.id}`, { method: "PATCH", body: JSON.stringify(body) })
    : await api("/api/assistants", { method: "POST", body: JSON.stringify(body) });
  current = null;
  await loadAssistants();
  select(assistants.find((a) => a.id === saved.id));
};

$("delete-assistant").onclick = async () => {
  if (!current || !confirm(`Delete ${current.name}?`)) return;
  await api(`/api/assistants/${current.id}`, { method: "DELETE" });
  current = null;
  $("editor").classList.add("hidden");
  await loadAssistants();
  if (!assistants.length) $("empty-state").classList.remove("hidden");
};

// ---------- Tabs ----------

const activeTab = () => document.querySelector(".tabs button.active").dataset.tab;

for (const button of document.querySelectorAll(".tabs button")) {
  button.onclick = () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === button));
    $("tab-call").classList.toggle("hidden", button.dataset.tab !== "call");
    $("tab-chat").classList.toggle("hidden", button.dataset.tab !== "chat");
    if (button.dataset.tab === "chat" && !chat) openChat();
  };
}

function addMessage(container, role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// ---------- Chat ----------

function openChat() {
  closeChat();
  $("chat-messages").innerHTML = "";
  const ws = new WebSocket(wsUrl(`/api/assistants/${current.id}/chat`));
  let bubble = null;
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const box = $("chat-messages");
    if (data.type === "delta") {
      bubble = bubble || addMessage(box, "assistant", "");
      bubble.textContent += data.text;
      box.scrollTop = box.scrollHeight;
    } else if (data.type === "done") {
      if (!bubble && data.text) addMessage(box, "assistant", data.text);
      bubble = null;
    } else if (data.type === "error") {
      addMessage(box, "error", data.message);
      bubble = null;
    }
  };
  ws.onclose = (event) => {
    if (event.code === 4404) addMessage($("chat-messages"), "error", "Assistant not found");
  };
  chat = ws;
}

function closeChat() {
  if (chat) chat.close();
  chat = null;
}

$("chat-form").onsubmit = (event) => {
  event.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text || !chat || chat.readyState !== WebSocket.OPEN) return;
  addMessage($("chat-messages"), "user", text);
  chat.send(JSON.stringify({ type: "message", text }));
  $("chat-input").value = "";
};

$("chat-reset").onclick = openChat;

// ---------- Call ----------

class Player {
  constructor(ctx) {
    this.ctx = ctx;
    this.sources = new Set();
    this.nextTime = 0;
    this.leftover = null;
    this.pendingEndId = null;
    this.speaking = false;
    this.onDrained = null;
  }

  enqueue(arrayBuffer) {
    let bytes = new Uint8Array(arrayBuffer);
    if (this.leftover) {
      const merged = new Uint8Array(this.leftover.length + bytes.length);
      merged.set(this.leftover);
      merged.set(bytes, this.leftover.length);
      bytes = merged;
      this.leftover = null;
    }
    // A chunk can split a 16-bit sample; carry the odd byte to the next chunk
    if (bytes.length % 2) {
      this.leftover = bytes.slice(-1);
      bytes = bytes.slice(0, -1);
    }
    if (!bytes.length) return;

    const samples = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    const buffer = this.ctx.createBuffer(1, samples.length, SAMPLE_RATE);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 0x8000;

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);

    const now = this.ctx.currentTime;
    // Small jitter buffer when starting fresh so the first chunks don't stutter
    if (this.nextTime < now + 0.02) this.nextTime = now + 0.06;
    source.start(this.nextTime);
    this.nextTime += buffer.duration;

    this.speaking = true;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      this.checkDrained();
    };
  }

  markEnd(id) {
    this.pendingEndId = id;
    this.checkDrained();
  }

  checkDrained() {
    if (this.sources.size) return;
    // Clips ("mm") have no response id, so going quiet and finishing a response are separate
    this.speaking = false;
    if (this.pendingEndId === null) return;
    const id = this.pendingEndId;
    this.pendingEndId = null;
    this.onDrained?.(id);
  }

  clear() {
    for (const source of this.sources) {
      source.onended = null;
      try { source.stop(); } catch {}
    }
    this.sources.clear();
    this.nextTime = 0;
    this.leftover = null;
    this.pendingEndId = null;
    this.speaking = false;
  }
}

function setCallStatus(text) {
  $("call-status").textContent = text;
}

// ---------- Live state ----------

const STATE_ROWS = [
  ["Speaking", (s) => (s.agent_speaking ? "agent" : s.user_speaking ? "caller" : "nobody")],
  ["Silence", (s) => `${s.silence_ms} ms`],
  ["Usual pause", (s) => `${s.baseline_pause_ms} ms`],
  ["End of turn", (s) => s.end_of_turn.toFixed(2)],
  ["Turn so far", (s) => s.pending_text || "—"],
  ["Words quiet", (s) => `${s.words_quiet_ms} ms`],
  ["Reactions used", (s) => s.reactions.join(", ") || "—"],
  ["Turns", (s) => s.turns],
];

function renderState(state, decision) {
  const table = $("state-table");
  table.innerHTML = "";
  for (const [label, value] of STATE_ROWS) {
    const row = table.insertRow();
    row.insertCell().textContent = label;
    row.insertCell().textContent = value(state);
  }
  $("state-decision").textContent = `${decision.action} · ${decision.rule}`;
  $("state-decision").dataset.action = decision.action;

  if (decision.action !== "wait") {
    const log = $("decision-log");
    const item = document.createElement("li");
    item.textContent = `${(state.call_ms / 1000).toFixed(1)}s  ${decision.action} · ${decision.rule}`;
    log.prepend(item);
    while (log.children.length > 12) log.lastChild.remove();
  }
}

async function startCall() {
  const transcript = $("call-transcript");
  transcript.innerHTML = "";
  $("decision-log").innerHTML = "";
  setCallStatus("Connecting…");
  $("call-button").textContent = "End call";

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    setCallStatus(`Microphone blocked: ${err.message}`);
    $("call-button").textContent = "Start call";
    return;
  }

  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
  const player = new Player(ctx);
  const ws = new WebSocket(wsUrl(`/api/assistants/${current.id}/call`));
  ws.binaryType = "arraybuffer";

  const mic = ctx.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(ctx, "pcm-capture", { processorOptions: { targetRate: SAMPLE_RATE } });
  const sink = ctx.createGain();
  sink.gain.value = 0; // keeps the worklet pulled without playing the mic back
  mic.connect(capture).connect(sink).connect(ctx.destination);

  let ready = false;
  capture.port.onmessage = (event) => {
    if (!ready || ws.readyState !== WebSocket.OPEN) return;
    // On speakers, don't send the mic while the assistant talks, or it hears itself and interrupts
    if (!$("headphones").checked && player.speaking) return;
    ws.send(event.data);
  };

  player.onDrained = (id) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "playback_done", id }));
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      player.enqueue(event.data);
      return;
    }
    const data = JSON.parse(event.data);
    if (data.type === "ready") {
      ready = true;
      setCallStatus("Connected");
    } else if (data.type === "interim") {
      $("call-interim").textContent = data.text;
    } else if (data.type === "transcript") {
      if (data.role === "user") $("call-interim").textContent = "";
      addMessage(transcript, data.role, data.text);
    } else if (data.type === "audio_end") {
      player.markEnd(data.id);
    } else if (data.type === "clear") {
      player.clear();
    } else if (data.type === "state") {
      renderState(data.state, data.decision);
    } else if (data.type === "error") {
      addMessage(transcript, "error", data.message);
    }
  };

  ws.onclose = (event) => {
    setCallStatus(event.code === 4404 ? "Assistant not found" : "Call ended");
    cleanup();
  };

  let cleanedUp = false;
  function cleanup() {
    if (cleanedUp) return;
    cleanedUp = true;
    player.clear();
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    $("call-button").textContent = "Start call";
    $("call-interim").textContent = "";
    call = null;
  }

  call = {
    hangUp() {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "hangup" }));
      ws.close();
      cleanup();
      setCallStatus("Call ended");
    },
  };
}

function stopCall() {
  if (call) call.hangUp();
}

$("call-button").onclick = () => (call ? stopCall() : startCall());

loadAssistants().catch((err) => {
  $("empty-state").textContent = `Could not load assistants: ${err.message}`;
});
