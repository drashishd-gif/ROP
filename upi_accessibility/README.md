# Accessible UPI Pay — demo prototype

A small Flask + vanilla-JS demo showing interaction patterns that make a
UPI-style "pay someone" flow usable by blind and low-vision users. It
**simulates** a payment (in-memory balance, no real bank/UPI/NPCI
integration) — the point is the accessible interaction design, not a real
payment rail.

## Run it

```bash
pip install flask
python upi_accessibility/app.py
```

Open http://localhost:5001.

## Accessibility patterns demonstrated

- **Semantic structure & landmarks**: real `<header>`/`<main>`/`<section>`,
  `<fieldset>`/`<legend>` grouping, a working skip link, and labels tied to
  every input — so a screen reader can navigate the page by headings/regions
  instead of the user having to swipe through everything linearly.
- **Spoken confirmation before money moves**: after "Review payment", the
  app reads back *"You are about to pay ₹X to <name>, UPI ID <id>"* out loud
  (`speechSynthesis`) and requires a separate explicit "Confirm and send" —
  this catches the single most common failure mode (wrong payee/amount)
  without requiring the user to visually re-check a summary screen.
- **Voice input**: mic buttons next to the UPI ID and amount fields use the
  Web Speech API (`SpeechRecognition`) so a user can speak a payee or amount
  instead of typing/reading it.
- **Live regions for every state change**: balance, confirmation text, and
  the success/failure result are all `aria-live` regions, so a screen reader
  announces them automatically without the user needing to hunt for what
  changed.
- **Non-visual success/failure signal**: distinct audio tones (rising chime
  for success, low buzz for failure) play via the Web Audio API, independent
  of speech/text, so outcomes are perceivable even with a screen reader's
  speech muted or a hearing-impaired-but-sighted setup relying on the visual
  color/text change instead.
- **High-contrast / large-text mode**: a single toggle switches to a
  black/yellow, 22px layout for low-vision users who don't use a screen
  reader.
- **Visible, high-contrast focus outline** on every interactive element, for
  keyboard-only and switch-device navigation.
- **Errors are specific and spoken**, not just a red border: e.g.
  "Insufficient balance. Your available balance is 12500.00 rupees."

## What a real product would still need

This is a UI/interaction-pattern demo, not a payment product. A real app
would need: integration with a licensed PSP's UPI SDK (not a DIY UPI
implementation), biometric/PIN auth flows that are themselves accessible,
fraud/rate-limiting protections, and testing with actual screen-reader users
(TalkBack/VoiceOver) rather than just WAI-ARIA compliance on paper.
