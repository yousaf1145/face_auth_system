
const POLL_INTERVAL_MS = 350;
const MAX_ATTEMPTS = 40; // ~14s of streaming before we give up and show a result

const state = {
  stream: null,
  polling: false,
  attempts: 0,
};

const els = {
  panels: {
    1: document.getElementById('panel-1'),
    2: document.getElementById('panel-2'),
    3: document.getElementById('panel-3'),
  },
  steps: document.querySelectorAll('.step'),
  btnRecognize: document.getElementById('btnRecognize'),
  btnStartLiveness: document.getElementById('btnStartLiveness'),
  btnRestart: document.getElementById('btnRestart'),
  video: document.getElementById('video'),
  captureCanvas: document.getElementById('captureCanvas'),
  scanSweep: document.getElementById('scanSweep'),
  viewportStatus: document.getElementById('viewportStatus'),
  fillReal: document.getElementById('fillReal'),
  fillSpoof: document.getElementById('fillSpoof'),
  fillFlow: document.getElementById('fillFlow'),
  fillCombined: document.getElementById('fillCombined'),
  valReal: document.getElementById('valReal'),
  valSpoof: document.getElementById('valSpoof'),
  valFlow: document.getElementById('valFlow'),
  valCombined: document.getElementById('valCombined'),
  telemetryStatus: document.getElementById('telemetryStatus'),
  resultIcon: document.getElementById('resultIcon'),
  resultTitle: document.getElementById('resultTitle'),
  resultMessage: document.getElementById('resultMessage'),
  resultActions: document.getElementById('resultActions'),
  modalBackdrop: document.getElementById('modalBackdrop'),
  modalIcon: document.getElementById('modalIcon'),
  modalTitle: document.getElementById('modalTitle'),
  modalMessage: document.getElementById('modalMessage'),
  modalClose: document.getElementById('modalClose'),
};

function goToStep(n) {
  Object.values(els.panels).forEach(p => p.classList.remove('active'));
  els.panels[n].classList.add('active');
  els.steps.forEach(step => {
    const stepNum = Number(step.dataset.step);
    step.classList.toggle('active', stepNum === n);
    step.classList.toggle('done', stepNum < n);
  });
}

function showModal({ icon, title, message, tone }) {
  els.modalIcon.textContent = icon;
  els.modalTitle.textContent = title;
  els.modalMessage.textContent = message;
  els.modalIcon.style.color = tone === 'danger' ? 'var(--danger)' : 'var(--amber)';
  els.modalIcon.style.borderColor = tone === 'danger' ? 'var(--danger)' : 'var(--amber)';
  els.modalIcon.style.background = tone === 'danger' ? 'var(--danger-dim)' : 'rgba(242,169,59,0.14)';
  els.modalBackdrop.classList.add('show');
}
els.modalClose.addEventListener('click', () => els.modalBackdrop.classList.remove('show'));

/* ------------------------------------------------------------------ */
/* Step 1 -> 2                                                         */
/* ------------------------------------------------------------------ */
els.btnRecognize.addEventListener('click', async () => {
  goToStep(2);
  await startCamera();
});

async function startCamera() {
  els.viewportStatus.textContent = 'Requesting camera access…';
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    els.video.srcObject = state.stream;
    els.viewportStatus.textContent = 'Camera ready. Press "Start Liveness Check".';
  } catch (err) {
    els.viewportStatus.textContent = 'Camera access denied or unavailable.';
    showModal({
      icon: '⚠',
      title: 'Camera Unavailable',
      message: 'This terminal needs camera access to verify your identity. Please allow camera permissions and try again.',
      tone: 'danger',
    });
  }
}

/* ------------------------------------------------------------------ */
/* Step 2: liveness + recognition streaming                            */
/* ------------------------------------------------------------------ */
els.btnStartLiveness.addEventListener('click', async () => {
  if (state.polling) return;
  state.attempts = 0;
  state.polling = true;
  els.btnStartLiveness.disabled = true;
  els.scanSweep.classList.add('active');
  els.telemetryStatus.textContent = 'Scanning…';

  try {
    await fetch('/api/session/start', { method: 'POST' });
  } catch (e) {
    // non-fatal — pipeline will still self-initialize server-side
  }

  pollLoop();
});

function captureFrame() {
  const canvas = els.captureCanvas;
  canvas.width = els.video.videoWidth || 640;
  canvas.height = els.video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  // Un-mirror before sending so the server sees a natural (non-flipped) frame.
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(els.video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function pollLoop() {
  if (!state.polling) return;
  state.attempts += 1;

  let result;
  try {
    const image = captureFrame();
    const resp = await fetch('/api/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
    result = await resp.json();
  } catch (err) {
    els.telemetryStatus.textContent = 'Connection error — retrying…';
    scheduleNext();
    return;
  }

  updateTelemetry(result);

  const terminal = new Set([
    'AUTHORIZED', 'UNKNOWN_PERSON', 'SPOOF_DETECTED',
    'MULTIPLE_FACES_DETECTED',
  ]);

  if (terminal.has(result.status)) {
    finishScan(result);
    return;
  }

  if (state.attempts >= MAX_ATTEMPTS) {
    finishScan({ status: 'NO_FACE_DETECTED', reason: 'Timed out waiting for a stable, live face.' });
    return;
  }

  scheduleNext();
}

function scheduleNext() {
  setTimeout(pollLoop, POLL_INTERVAL_MS);
}

function updateTelemetry(result) {
  setBar(els.fillReal, els.valReal, result.minifas_real_score);
  setBar(els.fillSpoof, els.valSpoof, result.minifas_spoof_score);
  setBar(els.fillFlow, els.valFlow, result.optical_flow_score);
  setBar(els.fillCombined, els.valCombined, result.combined_liveness_score);

  const statusText = {
    AWAITING_MORE_FRAMES: 'Gathering motion samples…',
    NO_FACE_DETECTED: 'No face in frame — center yourself.',
    MULTIPLE_FACES_DETECTED: 'Multiple faces detected.',
  }[result.status];
  els.telemetryStatus.textContent = statusText || 'Scanning…';
}

function setBar(fillEl, valEl, value) {
  if (value === null || value === undefined) {
    fillEl.style.width = '0%';
    valEl.textContent = '—';
    return;
  }
  const pct = Math.max(0, Math.min(100, value * 100));
  fillEl.style.width = `${pct}%`;
  valEl.textContent = pct.toFixed(0) + '%';
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
 
  els.video.srcObject = null;
}

/* ------------------------------------------------------------------ */
/* Step 3: result                                                      */
/* ------------------------------------------------------------------ */
function finishScan(result) {
  state.polling = false;
  els.scanSweep.classList.remove('active');
  els.btnStartLiveness.disabled = false;
  stopCamera();
  renderResult(result);
  goToStep(3);
}

function renderResult(result) {
  els.resultActions.innerHTML = '';
  els.resultIcon.className = 'result-icon';

  switch (result.status) {
    case 'AUTHORIZED': {
      els.resultIcon.classList.add('ok');
      els.resultIcon.textContent = '✓';
      els.resultTitle.textContent = 'Access Granted';
      els.resultMessage.textContent = `Welcome, ${result.name || 'authorized user'}. Identity confirmed as a live, enrolled person.`;
      const loginBtn = document.createElement('button');
      loginBtn.className = 'btn btn-primary';
      loginBtn.textContent = 'Login';
      loginBtn.addEventListener('click', () => {
        // Hook this up to your real auth/session logic.
        alert(`Logged in as ${result.name}.`);
      });
      els.resultActions.appendChild(loginBtn);
      break;
    }

    case 'UNKNOWN_PERSON': {
      els.resultIcon.classList.add('warn');
      els.resultIcon.textContent = '?';
      els.resultTitle.textContent = 'Access Denied';
      els.resultMessage.textContent = 'A live person was detected, but they do not match any enrolled identity.';
      showModal({
        icon: '?',
        title: 'Unknown Person',
        message: 'This person is not enrolled in the system. Access has been denied.',
        tone: 'warn',
      });
      break;
    }

    case 'SPOOF_DETECTED': {
      els.resultIcon.classList.add('danger');
      els.resultIcon.textContent = '✕';
      els.resultTitle.textContent = 'Spoof Detected';
      els.resultMessage.textContent = 'The presented face did not pass liveness verification — this looked like a printed photo, a phone/screen replay, or a video.';
      showModal({
        icon: '✕',
        title: 'Spoof Detected',
        message: 'A photo, screen, or video replay was presented instead of a live person. Access has been denied and this attempt has been logged.',
        tone: 'danger',
      });
      break;
    }

    case 'MULTIPLE_FACES_DETECTED': {
      els.resultIcon.classList.add('warn');
      els.resultIcon.textContent = '⚠';
      els.resultTitle.textContent = 'Multiple Faces Detected';
      els.resultMessage.textContent = 'Only one person may authenticate at a time. Please ensure only one face is in frame and try again.';
      break;
    }

    default: {
      els.resultIcon.classList.add('warn');
      els.resultIcon.textContent = '⚠';
      els.resultTitle.textContent = 'No Face Detected';
      els.resultMessage.textContent = result.reason || 'No face was detected during the scan. Please try again.';
    }
  }
}

/* ------------------------------------------------------------------ */
/* Restart                                                             */
/* ------------------------------------------------------------------ */
els.btnRestart.addEventListener('click', () => {
  stopCamera();
  state.polling = false;
  state.attempts = 0;
  els.scanSweep.classList.remove('active');
  setBar(els.fillReal, els.valReal, null);
  setBar(els.fillSpoof, els.valSpoof, null);
  setBar(els.fillFlow, els.valFlow, null);
  setBar(els.fillCombined, els.valCombined, null);
  els.telemetryStatus.textContent = 'Awaiting scan…';
  goToStep(1);
});