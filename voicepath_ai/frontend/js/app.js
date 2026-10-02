const API = '';

const DISEASE_INFO = {
  parkinsons:  { icon: '🧠', desc: 'Parkinson\'s affects vocal tremor, speech rate, and voice stability. Voice may become monotone, quieter, or show irregular pauses.', dataset: 'MDVR-KCL (Real Clinical)' },
  respiratory: { icon: '🫁', desc: 'Respiratory conditions affect breathing sounds, speech energy, and may introduce wheezing or coughing patterns in recordings.', dataset: 'Synthetic (Research)' },
  alzheimers:  { icon: '🔮', desc: 'Alzheimer\'s affects speech fluency, introduces long pauses, word-finding hesitations, and changes in pitch variability.', dataset: 'Synthetic (Research)' },
  als:         { icon: '⚡', desc: 'ALS causes dysarthria — irregular amplitude, slurred articulation, nasal voice quality, and reduced speech precision.', dataset: 'Synthetic (Research)' },
  depression:  { icon: '💙', desc: 'Depression lowers vocal energy, reduces pitch variation (monotone), slows speech rate, and affects prosodic richness.', dataset: 'Synthetic (Research)' },
};

let selectedFile = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  initWaveform();
  loadDiseaseData();
  setupDragDrop();
  document.getElementById('file-input').addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
});

// --- Waveform Animation ---
function initWaveform() {
  const container = document.getElementById('wv-anim');
  const bars = 28;
  for (let i = 0; i < bars; i++) {
    const bar = document.createElement('div');
    bar.className = 'wv-bar';
    const h = 15 + Math.random() * 65;
    bar.style.height = h + 'px';
    bar.style.animationDelay = (i * 0.06) + 's';
    container.appendChild(bar);
  }
}

// --- Load disease + performance data ---
async function loadDiseaseData() {
  try {
    const [healthRes, diseasesRes] = await Promise.all([
      fetch(API + '/api/health'),
      fetch(API + '/api/diseases'),
    ]);
    const health   = await healthRes.json();
    const diseases = await diseasesRes.json();

    document.getElementById('stat-models').textContent = health.trained_models?.length || 0;
    renderDiseaseCards(diseases.diseases);
    renderPerfTable(diseases.diseases);
  } catch (e) {
    console.error('Failed to load disease data:', e);
    document.getElementById('perf-tbody').innerHTML =
      '<tr><td colspan="8" class="loading-cell">Could not load metrics — ensure the backend is running.</td></tr>';
  }
}

function renderDiseaseCards(diseases) {
  const grid = document.getElementById('disease-grid');
  grid.innerHTML = '';
  diseases.forEach(d => {
    const info = DISEASE_INFO[d.key] || { icon: '🔬', desc: '', dataset: '' };
    const ready = d.model_ready;
    const card = document.createElement('div');
    card.className = `disease-card ${ready ? 'ready' : 'not-ready'}`;
    card.innerHTML = `
      <div class="disease-icon">${info.icon}</div>
      <div class="disease-name">${d.name}</div>
      <div class="disease-desc">${info.desc}</div>
      <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:10px">Dataset: ${info.dataset}</div>
      <div class="disease-status">
        <div class="status-dot ${ready ? 'dot-ready' : 'dot-pending'}"></div>
        <span>${ready ? 'Model Ready' : 'Training Required'}</span>
      </div>
    `;
    card.addEventListener('click', () => {
      document.getElementById('disease-select').value = d.key;
      document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' });
    });
    grid.appendChild(card);
  });
}

function renderPerfTable(diseases) {
  const tbody = document.getElementById('perf-tbody');
  tbody.innerHTML = '';
  const DISEASE_INFO_ICONS = { parkinsons:'🧠', respiratory:'🫁', alzheimers:'🔮', als:'⚡', depression:'💙' };

  diseases.forEach(d => {
    const m = d.metrics || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${DISEASE_INFO_ICONS[d.key] || ''} ${d.name}</strong></td>
      <td><span class="model-badge">${d.best_model || '—'}</span></td>
      <td>${fmtMetric(m.accuracy)}</td>
      <td>${fmtMetric(m.precision)}</td>
      <td>${fmtMetric(m.recall)}</td>
      <td>${fmtMetric(m.specificity)}</td>
      <td>${fmtMetric(m.f1)}</td>
      <td>${fmtMetric(m.roc_auc)}</td>
    `;
    tbody.appendChild(tr);
  });
  if (!diseases.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="loading-cell">No data</td></tr>';
  }
}

function fmtMetric(v) {
  if (v == null || v === undefined) return '<span class="metric-na">—</span>';
  const pct = Math.round(v * 100);
  const cls = pct >= 75 ? 'metric-good' : 'metric-ok';
  return `<span class="${cls}">${pct}%</span>`;
}

// --- File Handling ---
function setupDragDrop() {
  const zone = document.getElementById('upload-zone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  zone.addEventListener('click', () => document.getElementById('file-input').click());
}

function handleFile(file) {
  const allowed = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError('Unsupported file type. Use WAV, MP3, OGG, or FLAC.');
    return;
  }
  selectedFile = file;
  document.getElementById('upload-zone').style.display = 'none';
  const info = document.getElementById('file-info');
  info.style.display = 'block';
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = formatBytes(file.size);
  const audio = document.getElementById('audio-preview');
  audio.src = URL.createObjectURL(file);
}

function clearFile() {
  selectedFile = null;
  document.getElementById('upload-zone').style.display = 'block';
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('audio-preview').src = '';
  document.getElementById('file-input').value = '';
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// --- Recording ---
async function toggleRecord() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'audio/wav' });
        const file = new File([blob], 'recorded_voice.wav', { type: 'audio/wav' });
        handleFile(file);
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      isRecording = true;
      const btn = document.getElementById('record-btn');
      btn.classList.add('recording');
      btn.innerHTML = '<span class="rec-dot"></span> Stop Recording';
      document.getElementById('record-hint').textContent = 'Recording... click to stop';
    } catch (e) {
      showError('Microphone access denied or not available.');
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    const btn = document.getElementById('record-btn');
    btn.classList.remove('recording');
    btn.innerHTML = '<span class="rec-dot"></span> Record Voice';
    document.getElementById('record-hint').textContent = 'Click to start recording (microphone required)';
  }
}

// --- Analysis ---
async function runAnalysis() {
  if (!selectedFile) { showError('Please upload or record a voice sample first.'); return; }
  const disease = document.getElementById('disease-select').value;
  if (!disease) { showError('Please select a disease to analyze.'); return; }

  const btn = document.getElementById('analyze-btn');
  const btnText = document.getElementById('analyze-btn-text');
  const spinner = document.getElementById('spinner');
  btn.disabled = true;
  btnText.textContent = 'Analyzing...';
  spinner.style.display = 'block';

  try {
    const formData = new FormData();
    formData.append('audio', selectedFile);
    formData.append('disease', disease);

    const res = await fetch(API + '/api/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Analysis failed.');
      return;
    }
    showResult(data);
  } catch (e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Analyze Voice';
    spinner.style.display = 'none';
  }
}

function showResult(data) {
  const card = document.getElementById('result-card');
  card.style.display = 'block';
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const isDisease = data.label === 1;
  const icon = document.getElementById('result-icon');
  icon.textContent = isDisease ? '⚠' : '✓';
  icon.className = `result-icon ${isDisease ? 'positive' : 'negative'}`;

  document.getElementById('result-disease-name').textContent = data.disease_name || data.disease;
  const labelEl = document.getElementById('result-label');
  labelEl.textContent = data.label_name;
  labelEl.className = `result-classification ${isDisease ? 'positive' : 'negative'}`;

  const conf = data.confidence || 0;
  document.getElementById('conf-pct').textContent = conf + '%';
  setTimeout(() => {
    document.getElementById('conf-bar').style.width = conf + '%';
  }, 100);

  const uncertainNote = document.getElementById('uncertain-note');
  uncertainNote.style.display = data.uncertain ? 'block' : 'none';

  // Probabilities
  const probSection = document.getElementById('prob-section');
  const probBars = document.getElementById('prob-bars');
  probBars.innerHTML = '';
  const probs = data.all_probs || {};
  const keys = Object.keys(probs);
  if (keys.length > 0) {
    probSection.style.display = 'block';
    const colors = ['#2196F3', '#66BB6A'];
    keys.forEach((label, i) => {
      const pct = Math.round((probs[label] || 0) * 100);
      probBars.innerHTML += `
        <div class="prob-item">
          <span class="prob-label">${label}</span>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:0%;background:${colors[i % colors.length]}" data-w="${pct}%"></div></div>
          <span class="prob-pct">${pct}%</span>
        </div>`;
    });
    setTimeout(() => {
      document.querySelectorAll('.prob-bar-fill').forEach(el => {
        el.style.width = el.dataset.w;
      });
    }, 150);
  } else {
    probSection.style.display = 'none';
  }

  // Features
  const featGrid = document.getElementById('features-grid');
  featGrid.innerHTML = '';
  const feats = data.top_features || {};
  const featLabels = {
    pitch_mean: 'Pitch Mean (Hz)', pitch_std: 'Pitch Std', rms_mean: 'RMS Energy',
    zcr_mean: 'Zero Crossing Rate', mfcc_1_mean: 'MFCC-1 Mean', spectral_centroid_mean: 'Spectral Centroid'
  };
  Object.entries(feats).forEach(([k, v]) => {
    featGrid.innerHTML += `
      <div class="feature-item">
        <div class="feature-name">${featLabels[k] || k}</div>
        <div class="feature-val">${typeof v === 'number' ? v.toFixed(2) : v}</div>
      </div>`;
  });

  document.getElementById('result-disclaimer').textContent = data.disclaimer || '';
}

function showError(msg) {
  alert('VoicePath AI: ' + msg);
}
