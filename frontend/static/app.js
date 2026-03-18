/**
 * Omega Frontend — Conversation-centered sports analytics UI.
 *
 * Connects to the FastAPI backend via SSE for real-time streaming.
 * Zero external dependencies — vanilla JS.
 */

(() => {
  "use strict";

  // ── Config ──────────────────────────────────────────────
  const API_BASE = window.OMEGA_API_BASE || "http://localhost:8000";
  const STAGES = ["understanding", "planning", "gathering", "analyzing", "composing"];

  // ── State ───────────────────────────────────────────────
  let currentSessionId = null;
  let sessions = {};  // id -> { messages: [], title: "" }
  let isStreaming = false;

  // ── DOM refs ────────────────────────────────────────────
  const $app = document.getElementById("app");
  const $welcomeScreen = document.getElementById("welcome-screen");
  const $chatContainer = document.getElementById("chat-container");
  const $messages = document.getElementById("messages");
  const $chatForm = document.getElementById("chat-form");
  const $userInput = document.getElementById("user-input");
  const $sendBtn = document.getElementById("send-btn");
  const $sessionList = document.getElementById("session-list");
  const $stageBar = document.getElementById("stage-bar");
  const $newChatBtn = document.getElementById("new-chat-btn");
  const $apiStatus = document.getElementById("api-status");

  // ── Init ────────────────────────────────────────────────
  function init() {
    loadSessions();
    renderSessionList();
    checkApiHealth();

    $chatForm.addEventListener("submit", handleSubmit);
    $userInput.addEventListener("input", autoResize);
    $userInput.addEventListener("keydown", handleKeyDown);
    $newChatBtn.addEventListener("click", startNewChat);

    document.querySelectorAll(".example-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $userInput.value = btn.dataset.prompt;
        $userInput.focus();
        autoResize();
      });
    });

    // Restore last active session
    const lastSession = localStorage.getItem("omega_last_session");
    if (lastSession && sessions[lastSession]) {
      switchToSession(lastSession);
    }
  }

  // ── Sessions ────────────────────────────────────────────
  function loadSessions() {
    try {
      const stored = localStorage.getItem("omega_sessions");
      if (stored) sessions = JSON.parse(stored);
    } catch (e) {
      sessions = {};
    }
  }

  function saveSessions() {
    try {
      localStorage.setItem("omega_sessions", JSON.stringify(sessions));
      if (currentSessionId) {
        localStorage.setItem("omega_last_session", currentSessionId);
      }
    } catch (e) {
      // localStorage full or unavailable
    }
  }

  function generateSessionId() {
    return "s_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 6);
  }

  function startNewChat() {
    const id = generateSessionId();
    sessions[id] = { messages: [], title: "New Chat", createdAt: Date.now() };
    currentSessionId = id;
    saveSessions();
    renderSessionList();
    showWelcome();
    $userInput.value = "";
    $userInput.focus();
  }

  function switchToSession(id) {
    if (!sessions[id]) return;
    currentSessionId = id;
    saveSessions();
    renderSessionList();

    if (sessions[id].messages.length === 0) {
      showWelcome();
    } else {
      showChat();
      renderAllMessages(sessions[id].messages);
    }
  }

  function renderSessionList() {
    const sorted = Object.entries(sessions).sort(
      (a, b) => (b[1].createdAt || 0) - (a[1].createdAt || 0)
    );

    $sessionList.innerHTML = sorted
      .map(([id, s]) => {
        const active = id === currentSessionId ? "active" : "";
        const title = escapeHtml(s.title || "New Chat");
        return `<div class="session-item ${active}" data-id="${id}">${title}</div>`;
      })
      .join("");

    $sessionList.querySelectorAll(".session-item").forEach((el) => {
      el.addEventListener("click", () => switchToSession(el.dataset.id));
    });
  }

  // ── View Toggle ─────────────────────────────────────────
  function showWelcome() {
    $welcomeScreen.classList.remove("hidden");
    $chatContainer.classList.add("hidden");
  }

  function showChat() {
    $welcomeScreen.classList.add("hidden");
    $chatContainer.classList.remove("hidden");
  }

  // ── Message Rendering ───────────────────────────────────
  function renderAllMessages(msgs) {
    $messages.innerHTML = "";
    msgs.forEach((m) => appendMessageToDOM(m.role, m.content, m.structured));
    scrollToBottom();
  }

  function appendMessageToDOM(role, content, structured) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    let bodyHtml = formatMessageContent(content);

    // Render bet cards from structured data
    if (structured && structured.bet_cards) {
      bodyHtml += structured.bet_cards.map(renderBetCard).join("");
    }
    if (structured && structured.analysis) {
      bodyHtml += renderAnalysisCard(structured.analysis);
    }

    div.innerHTML = `
      <div class="message-header">
        <span class="message-role ${role}">${role === "user" ? "You" : "Omega"}</span>
        <span class="message-time">${time}</span>
      </div>
      <div class="message-body">${bodyHtml}</div>
    `;

    $messages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function formatMessageContent(text) {
    if (!text) return "";

    // Basic markdown-like formatting
    let html = escapeHtml(text);

    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Code: `code`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Line breaks
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    return `<p>${html}</p>`;
  }

  function renderBetCard(card) {
    const edgeClass =
      card.edge > 0 ? "edge-positive" : card.edge < 0 ? "edge-negative" : "edge-neutral";

    const confClass =
      card.confidence >= 0.8 ? "confidence-high" :
      card.confidence >= 0.6 ? "confidence-medium" : "confidence-low";

    const confLabel =
      card.confidence >= 0.8 ? "High" : card.confidence >= 0.6 ? "Medium" : "Low";

    const hasEdge = card.edge > 0;

    return `
      <div class="bet-card">
        <div class="bet-card-header">
          <span class="bet-card-matchup">${escapeHtml(card.matchup || "")}</span>
          <span class="bet-card-league">${escapeHtml(card.league || "")}</span>
        </div>
        <div class="bet-card-grid">
          ${card.spread != null ? `
            <div class="bet-card-stat">
              <div class="bet-card-stat-label">Spread</div>
              <div class="bet-card-stat-value">${card.spread > 0 ? "+" : ""}${card.spread}</div>
            </div>
          ` : ""}
          ${card.total != null ? `
            <div class="bet-card-stat">
              <div class="bet-card-stat-label">Total</div>
              <div class="bet-card-stat-value">${card.total}</div>
            </div>
          ` : ""}
          ${card.moneyline != null ? `
            <div class="bet-card-stat">
              <div class="bet-card-stat-label">Moneyline</div>
              <div class="bet-card-stat-value">${card.moneyline > 0 ? "+" : ""}${card.moneyline}</div>
            </div>
          ` : ""}
          ${card.edge != null ? `
            <div class="bet-card-stat">
              <div class="bet-card-stat-label">Edge</div>
              <div class="bet-card-stat-value ${edgeClass}">${card.edge > 0 ? "+" : ""}${card.edge.toFixed(1)}%</div>
            </div>
          ` : ""}
        </div>
        ${card.recommendation ? `
          <div class="bet-card-recommendation ${hasEdge ? "" : "no-edge"}">
            <span class="confidence-badge ${confClass}">${confLabel}</span>
            <span>${escapeHtml(card.recommendation)}</span>
          </div>
        ` : ""}
      </div>
    `;
  }

  function renderAnalysisCard(analysis) {
    return `
      <div class="bet-card">
        <div class="bet-card-header">
          <span class="bet-card-matchup">${escapeHtml(analysis.title || "Analysis")}</span>
        </div>
        ${analysis.metrics ? `
          <div class="bet-card-grid">
            ${Object.entries(analysis.metrics).map(([k, v]) => `
              <div class="bet-card-stat">
                <div class="bet-card-stat-label">${escapeHtml(k)}</div>
                <div class="bet-card-stat-value">${escapeHtml(String(v))}</div>
              </div>
            `).join("")}
          </div>
        ` : ""}
      </div>
    `;
  }

  // ── Typing Indicator ────────────────────────────────────
  function showTyping() {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.id = "typing-indicator";
    div.innerHTML = `
      <div class="message-header">
        <span class="message-role assistant">Omega</span>
      </div>
      <div class="message-body">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    $messages.appendChild(div);
    scrollToBottom();
  }

  function hideTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  // ── Stage Progress ──────────────────────────────────────
  function showStageBar() {
    $stageBar.classList.remove("hidden");
    resetStages();
  }

  function hideStageBar() {
    $stageBar.classList.add("hidden");
  }

  function resetStages() {
    document.querySelectorAll(".stage-item").forEach((el) => {
      el.classList.remove("active", "done");
    });
  }

  function setStage(stageName) {
    const stageIndex = STAGES.indexOf(stageName);
    if (stageIndex === -1) return;

    document.querySelectorAll(".stage-item").forEach((el, i) => {
      el.classList.remove("active", "done");
      if (i < stageIndex) el.classList.add("done");
      if (i === stageIndex) el.classList.add("active");
    });
  }

  // ── Form Handling ───────────────────────────────────────
  function handleSubmit(e) {
    e.preventDefault();
    const text = $userInput.value.trim();
    if (!text || isStreaming) return;
    sendMessage(text);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $chatForm.dispatchEvent(new Event("submit"));
    }
  }

  function autoResize() {
    $userInput.style.height = "auto";
    $userInput.style.height = Math.min($userInput.scrollHeight, 120) + "px";
  }

  // ── Send Message ────────────────────────────────────────
  async function sendMessage(text) {
    // Ensure we have a session
    if (!currentSessionId) {
      const id = generateSessionId();
      sessions[id] = { messages: [], title: text.slice(0, 50), createdAt: Date.now() };
      currentSessionId = id;
    }

    // Update session title if it's the first message
    if (sessions[currentSessionId].messages.length === 0) {
      sessions[currentSessionId].title = text.slice(0, 50);
    }

    // Add user message
    sessions[currentSessionId].messages.push({ role: "user", content: text });
    saveSessions();
    renderSessionList();

    // Switch to chat view
    showChat();
    appendMessageToDOM("user", text);

    // Clear input
    $userInput.value = "";
    $userInput.style.height = "auto";

    // Stream response
    isStreaming = true;
    $sendBtn.disabled = true;
    showTyping();
    showStageBar();

    try {
      await streamResponse(text);
    } catch (err) {
      hideTyping();
      hideStageBar();
      appendMessageToDOM("assistant", `Sorry, something went wrong: ${err.message}`);
    }

    isStreaming = false;
    $sendBtn.disabled = false;
    hideStageBar();
  }

  // ── SSE Streaming ───────────────────────────────────────
  async function streamResponse(userMessage) {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        message: userMessage,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let accumulatedText = "";
    let structuredData = null;
    let assistantDiv = null;

    hideTyping();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      let eventType = null;
      let eventData = null;

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          eventData = line.slice(5).trim();
        } else if (line === "" && eventType && eventData) {
          // Complete SSE event
          handleSSEEvent(eventType, eventData);
          eventType = null;
          eventData = null;
        }
      }
    }

    // Finalize: if we accumulated text but no done event fired
    if (accumulatedText && !assistantDiv) {
      assistantDiv = appendMessageToDOM("assistant", accumulatedText, structuredData);
      sessions[currentSessionId].messages.push({
        role: "assistant",
        content: accumulatedText,
        structured: structuredData,
      });
      saveSessions();
    }

    function handleSSEEvent(type, dataStr) {
      let payload;
      try {
        payload = JSON.parse(dataStr);
      } catch (e) {
        return;
      }

      const data = payload.data;

      switch (type) {
        case "stage_update":
          if (data && data.stage) {
            setStage(data.stage);
          }
          break;

        case "partial_text":
          if (data && typeof data.text === "string") {
            accumulatedText += data.text;
            if (!assistantDiv) {
              assistantDiv = appendMessageToDOM("assistant", accumulatedText);
            } else {
              const body = assistantDiv.querySelector(".message-body");
              if (body) body.innerHTML = formatMessageContent(accumulatedText);
            }
            scrollToBottom();
          }
          break;

        case "structured_data":
          structuredData = data;
          // Re-render with structured data
          if (assistantDiv && structuredData) {
            const body = assistantDiv.querySelector(".message-body");
            if (body) {
              let html = formatMessageContent(accumulatedText);
              if (structuredData.bet_cards) {
                html += structuredData.bet_cards.map(renderBetCard).join("");
              }
              if (structuredData.analysis) {
                html += renderAnalysisCard(structuredData.analysis);
              }
              body.innerHTML = html;
            }
          }
          break;

        case "done":
          hideStageBar();
          const finalText = (data && data.final_text) || accumulatedText;
          if (!assistantDiv) {
            assistantDiv = appendMessageToDOM("assistant", finalText, structuredData);
          } else {
            const body = assistantDiv.querySelector(".message-body");
            if (body) {
              let html = formatMessageContent(finalText);
              if (structuredData && structuredData.bet_cards) {
                html += structuredData.bet_cards.map(renderBetCard).join("");
              }
              if (structuredData && structuredData.analysis) {
                html += renderAnalysisCard(structuredData.analysis);
              }
              body.innerHTML = html;
            }
          }
          sessions[currentSessionId].messages.push({
            role: "assistant",
            content: finalText,
            structured: structuredData,
          });
          saveSessions();
          accumulatedText = finalText;
          break;

        case "error":
          hideStageBar();
          const errMsg = (data && data.message) || "Unknown error";
          if (!assistantDiv) {
            assistantDiv = appendMessageToDOM("assistant", `Error: ${errMsg}`);
          }
          sessions[currentSessionId].messages.push({
            role: "assistant",
            content: `Error: ${errMsg}`,
          });
          saveSessions();
          break;
      }
    }
  }

  // ── API Health Check ────────────────────────────────────
  async function checkApiHealth() {
    const dot = $apiStatus.querySelector(".status-dot");
    const text = $apiStatus.querySelector(".status-text");

    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        dot.className = "status-dot connected";
        text.textContent = "Connected";
      } else {
        dot.className = "status-dot error";
        text.textContent = "API Error";
      }
    } catch (e) {
      dot.className = "status-dot";
      text.textContent = "Offline";
    }

    // Re-check every 30s
    setTimeout(checkApiHealth, 30000);
  }

  // ── Helpers ─────────────────────────────────────────────
  function scrollToBottom() {
    requestAnimationFrame(() => {
      $chatContainer.scrollTop = $chatContainer.scrollHeight;
    });
  }

  function escapeHtml(str) {
    const el = document.createElement("span");
    el.textContent = str;
    return el.innerHTML;
  }

  // ── Boot ────────────────────────────────────────────────
  init();
})();
