// ═══════════════════════════════════════════════════
// GOT ORACLE — SCRIPT.JS
// Minimal JS — one fetch call to FastAPI backend
// ═══════════════════════════════════════════════════

// Set a quick question in the textarea
function setQuestion(text) {
  document.getElementById("questionInput").value = text;
  document.getElementById("questionInput").focus();
}

// Clear all chat bubbles and reset UI
function clearAll() {
  document.getElementById("chatBubbles").innerHTML = "";
  document.getElementById("emptyState").style.display = "flex";
  document.getElementById("questionInput").value = "";
  setStatus("Oracle is ready", "ready");
}

// Update the status bar
function setStatus(text, state) {
  document.getElementById("statusText").textContent = text;
  const dot = document.getElementById("statusDot");
  dot.className = "status-dot";
  if (state === "thinking") dot.classList.add("thinking");
  if (state === "error")    dot.classList.add("error");
}

// Add a user question bubble
function addUserBubble(text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble-user";
  bubble.innerHTML = `
    <div class="bubble-label-user">⚔ You ask</div>
    <div class="bubble-text-user">${text}</div>
  `;
  document.getElementById("chatBubbles").appendChild(bubble);
  scrollToBottom();
}

// Add thinking animation bubble — returns the element so we can remove it
function addThinkingBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble-thinking";
  bubble.id = "thinkingBubble";
  bubble.innerHTML = `
    <div class="thinking-dots">
      <span></span><span></span><span></span>
    </div>
  `;
  document.getElementById("chatBubbles").appendChild(bubble);
  scrollToBottom();
  return bubble;
}

// Add oracle answer bubble
function addOracleBubble(text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble-oracle";
  bubble.innerHTML = `
    <div class="bubble-label-oracle">🔮 The Oracle speaks</div>
    <div class="bubble-text-oracle">${text}</div>
  `;
  document.getElementById("chatBubbles").appendChild(bubble);
  scrollToBottom();
}

// Scroll answer panel to bottom
function scrollToBottom() {
  const scroll = document.getElementById("answerScroll");
  scroll.scrollTop = scroll.scrollHeight;
}

// ── MAIN FUNCTION — Ask the Oracle ──────────────────
async function askOracle() {
  const input    = document.getElementById("questionInput");
  const question = input.value.trim();

  // Validate input
  if (!question) return;

  // Hide empty state
  document.getElementById("emptyState").style.display = "none";

  // Disable button while thinking
  const btn = document.getElementById("askBtn");
  btn.disabled = true;
  setStatus("The Oracle is thinking...", "thinking");

  // Add user bubble
  addUserBubble(question);
  input.value = "";

  // Add thinking animation
  const thinkingBubble = addThinkingBubble();

  try {
    // ── ONE FETCH CALL TO FASTAPI ──────────────────
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    });

    const data = await response.json();

    // Remove thinking bubble
    thinkingBubble.remove();

    if (response.ok) {
      addOracleBubble(data.answer);
      setStatus("Oracle is ready", "ready");
    } else {
      addOracleBubble("The ravens have failed to deliver a response. Please try again.");
      setStatus("The ravens returned empty", "error");
    }

  } catch (error) {
    // Remove thinking bubble on error
    thinkingBubble.remove();
    addOracleBubble("The Oracle cannot be reached. Ensure the server is running.");
    setStatus("Oracle is unreachable", "error");
  }

  // Re-enable button
  btn.disabled = false;
}

// ── Allow Enter key to submit (Shift+Enter for newline)
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("questionInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askOracle();
    }
  });
});