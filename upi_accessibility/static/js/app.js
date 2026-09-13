// Accessible UPI Pay demo — client behavior.
// Key accessibility patterns:
//  1. Every state change is spoken (speech synthesis) AND written to an
//     aria-live region, so both screen-reader and voice-only flows work.
//  2. Payment always requires an explicit spoken/typed "confirm" step that
//     reads back payee + amount before money moves.
//  3. Voice input (mic buttons) lets a user fill fields without a keyboard.
//  4. Distinct audio tones for success/failure, independent of screen output.

(function () {
  "use strict";

  const announcer = document.getElementById("sr-announcer");
  const resultSection = document.getElementById("result-section");
  const confirmSection = document.getElementById("confirm-section");
  const confirmText = document.getElementById("confirm-text");
  const payForm = document.getElementById("pay-form");
  const historyList = document.getElementById("history-list");
  const balanceEl = document.getElementById("balance");
  const contrastToggle = document.getElementById("contrast-toggle");

  let pendingPayment = null; // { contact_id, amount, note }

  // --- Speech output -------------------------------------------------

  function speak(text) {
    announcer.textContent = text;
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.95;
    window.speechSynthesis.speak(utter);
  }

  // --- Audio cues (Web Audio API — works even if speech is off) ------

  let audioCtx = null;
  function tone(frequencies, durationMs) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const now = audioCtx.currentTime;
      frequencies.forEach((freq, i) => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.frequency.value = freq;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        const start = now + i * (durationMs / 1000);
        gain.gain.setValueAtTime(0.15, start);
        gain.gain.exponentialRampToValueAtTime(0.001, start + durationMs / 1000);
        osc.start(start);
        osc.stop(start + durationMs / 1000);
      });
    } catch (e) {
      // Audio not available; speech + text announcements still work.
    }
  }
  const playSuccessTone = () => tone([880, 1320], 150);
  const playErrorTone = () => tone([220, 140], 250);

  // --- Voice input (Web Speech API SpeechRecognition) -----------------

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  document.querySelectorAll(".mic-btn").forEach((btn) => {
    if (!SpeechRecognition) {
      btn.disabled = true;
      btn.title = "Voice input isn't supported in this browser";
      return;
    }
    btn.addEventListener("click", () => {
      const targetInput = document.getElementById(btn.dataset.target);
      const recognizer = new SpeechRecognition();
      recognizer.lang = "en-IN";
      recognizer.interimResults = false;
      recognizer.maxAlternatives = 1;

      speak("Listening.");
      btn.setAttribute("aria-busy", "true");

      recognizer.onresult = (event) => {
        const transcript = event.results[0][0].transcript.trim();
        if (targetInput.id === "amount-input") {
          const digits = transcript.replace(/[^\d.]/g, "");
          targetInput.value = digits;
        } else {
          targetInput.value = transcript;
        }
        speak(`You said: ${transcript}`);
      };
      recognizer.onerror = () => {
        speak("Sorry, I didn't catch that. Please try again or type instead.");
      };
      recognizer.onend = () => btn.removeAttribute("aria-busy");

      recognizer.start();
    });
  });

  // --- High-contrast / large-text toggle ------------------------------

  contrastToggle.addEventListener("click", () => {
    const on = document.body.classList.toggle("high-contrast");
    contrastToggle.setAttribute("aria-pressed", String(on));
    contrastToggle.textContent = on
      ? "Turn off high-contrast, large-text mode"
      : "Turn on high-contrast, large-text mode";
    speak(on ? "High contrast mode on." : "High contrast mode off.");
  });

  // --- Payment flow: review -> confirm -----------------------------

  payForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultSection.textContent = "";
    resultSection.className = "";

    const contactId = document.getElementById("contact-select").value;
    const upiId = document.getElementById("upi-id-input").value;
    const amount = document.getElementById("amount-input").value;
    const note = document.getElementById("note-input").value;

    const res = await fetch("/api/pay/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contact_id: contactId, upi_id: upiId, amount, note }),
    });
    const data = await res.json();

    if (!data.ok) {
      resultSection.textContent = data.error;
      resultSection.className = "error";
      speak(data.error);
      playErrorTone();
      return;
    }

    pendingPayment = { contact_id: data.contact.id, amount: data.amount, note: data.note };
    confirmText.textContent = data.confirmation_text;
    confirmSection.hidden = false;
    speak(data.confirmation_text);
    document.getElementById("confirm-btn").focus();
  });

  document.getElementById("cancel-btn").addEventListener("click", () => {
    pendingPayment = null;
    confirmSection.hidden = true;
    speak("Payment cancelled.");
  });

  document.getElementById("confirm-btn").addEventListener("click", async () => {
    if (!pendingPayment) return;
    speak("Sending payment. Please wait.");

    const res = await fetch("/api/pay/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pendingPayment),
    });
    const data = await res.json();

    confirmSection.hidden = true;
    pendingPayment = null;

    if (!data.ok) {
      resultSection.textContent = data.error;
      resultSection.className = "error";
      speak(data.error);
      playErrorTone();
      return;
    }

    resultSection.textContent = data.success_text;
    resultSection.className = "success";
    speak(data.success_text);
    playSuccessTone();
    balanceEl.textContent = `₹${data.balance.toFixed(2)}`;
    payForm.reset();
    loadHistory();
  });

  // --- Transaction history --------------------------------------------

  async function loadHistory() {
    const res = await fetch("/api/history");
    const data = await res.json();
    historyList.innerHTML = "";
    if (data.transactions.length === 0) {
      historyList.innerHTML = "<li>No transactions yet.</li>";
      return;
    }
    data.transactions.forEach((txn) => {
      const li = document.createElement("li");
      li.textContent = `${txn.status === "SUCCESS" ? "Sent" : "Failed"} ₹${txn.amount.toFixed(2)} to ${txn.contact} — transaction ${txn.id}`;
      historyList.appendChild(li);
    });
  }

  loadHistory();
})();
