/**
 * Underwater Sonar Object Detection - Client Application
 * Handles reliable image upload, inference API communication, high-precision canvas
 * bounding box rendering, and responsive detection results.
 */

// Application State
const state = {
  selectedFile: null,
  imageElement: null,
  naturalWidth: 0,
  naturalHeight: 0,
  detections: [],
  highlightedIndex: -1,
  isLoading: false,
};

// DOM Elements Cache
const elements = {
  // Banner & Badges
  errorBanner: document.getElementById('errorBanner'),
  errorTitle: document.getElementById('errorTitle'),
  errorMessage: document.getElementById('errorMessage'),
  closeErrorBtn: document.getElementById('closeErrorBtn'),
  modelModeBadge: document.getElementById('modelModeBadge'),

  // Sections
  uploadSection: document.getElementById('uploadSection'),
  previewSection: document.getElementById('previewSection'),
  
  // Upload & Inputs
  dropZone: document.getElementById('dropZone'),
  imageFileInput: document.getElementById('imageFileInput'),
  uploadTriggerBtn: document.getElementById('uploadTriggerBtn'),
  
  // Sample Chips
  sampleSmallObjectBtn: document.getElementById('sampleSmallObjectBtn'),
  sampleAnomalyBtn: document.getElementById('sampleAnomalyBtn'),
  sampleGhostNetBtn: document.getElementById('sampleGhostNetBtn'),

  // Preview & Canvas
  sonarPreviewImg: document.getElementById('sonarPreviewImg'),
  detectionCanvas: document.getElementById('detectionCanvas'),
  canvasContainer: document.getElementById('canvasContainer'),
  fileMetaBadge: document.getElementById('fileMetaBadge'),
  reuploadBtn: document.getElementById('reuploadBtn'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  viewportStatusText: document.getElementById('viewportStatusText'),

  // Action Buttons
  detectObjectsBtn: document.getElementById('detectObjectsBtn'),

  // Results Section
  resultsCountBadge: document.getElementById('resultsCountBadge'),
  resultsIdleState: document.getElementById('resultsIdleState'),
  noObjectsState: document.getElementById('noObjectsState'),
  detectionsList: document.getElementById('detectionsList'),
  
  // JSON Inspector
  jsonInspectorWrapper: document.getElementById('jsonInspectorWrapper'),
  toggleJsonBtn: document.getElementById('toggleJsonBtn'),
  rawJsonResponse: document.getElementById('rawJsonResponse'),
  jsonChevron: document.getElementById('jsonChevron'),
};

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  checkBackendHealth();
  setupCanvasResizeObserver();
});

function setupEventListeners() {
  // 1. File Input Selection (change event)
  elements.imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Reset value on click so selecting the same file triggers change reliably
  elements.imageFileInput.addEventListener('click', () => {
    elements.imageFileInput.value = '';
  });

  // 2. Drag and Drop support on DropZone
  if (elements.dropZone) {
    // Keyboard accessibility (Enter / Space)
    elements.dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.imageFileInput.value = '';
        elements.imageFileInput.click();
      }
    });

    elements.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dropZone.classList.add('drag-over');
    });

    elements.dropZone.addEventListener('dragleave', () => {
      elements.dropZone.classList.remove('drag-over');
    });

    elements.dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dropZone.classList.remove('drag-over');
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // 3. Re-upload / Select New Image button
  if (elements.reuploadBtn) {
    elements.reuploadBtn.addEventListener('click', () => {
      resetToUploadState();
    });
  }

  // Drag and drop onto Preview stage wrapper as well
  const stageWrapper = document.getElementById('imageStageWrapper');
  if (stageWrapper) {
    stageWrapper.addEventListener('dragover', (e) => {
      e.preventDefault();
    });
    stageWrapper.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // Navigation Links: "Sonar AI" or "Launch Sonar Detector"
  document.querySelectorAll('a[href="#sonar-detection"]').forEach((link) => {
    link.addEventListener('click', () => {
      if (!elements.previewSection.classList.contains('hidden')) {
        resetToUploadState();
      }
    });
  });

  // 4. Detect Objects Action
  if (elements.detectObjectsBtn) {
    elements.detectObjectsBtn.addEventListener('click', () => {
      runDetectionInference();
    });
  }

  // 5. Preset Sample Chips
  if (elements.sampleSmallObjectBtn) {
    elements.sampleSmallObjectBtn.addEventListener('click', () => 
      loadSampleImage('/sample_images/sample_small_object.jpg', 'sample_small_object.jpg')
    );
  }
  if (elements.sampleAnomalyBtn) {
    elements.sampleAnomalyBtn.addEventListener('click', () => 
      loadSampleImage('/sample_images/sample_unknown_anomaly.jpg', 'sample_unknown_anomaly.jpg')
    );
  }
  if (elements.sampleGhostNetBtn) {
    elements.sampleGhostNetBtn.addEventListener('click', () => 
      loadSampleImage('/sample_images/sample_ghost_net.jpg', 'sample_ghost_net.jpg')
    );
  }

  // 6. Close Error Banner
  if (elements.closeErrorBtn) {
    elements.closeErrorBtn.addEventListener('click', hideError);
  }

  // 7. Toggle JSON Inspector
  if (elements.toggleJsonBtn) {
    elements.toggleJsonBtn.addEventListener('click', () => {
      const isHidden = elements.rawJsonResponse.classList.contains('hidden');
      if (isHidden) {
        elements.rawJsonResponse.classList.remove('hidden');
        elements.jsonChevron.classList.add('open');
      } else {
        elements.rawJsonResponse.classList.add('hidden');
        elements.jsonChevron.classList.remove('open');
      }
    });
  }

  // 8. Window resize for dynamic canvas bounding box realignment
  window.addEventListener('resize', () => {
    if (state.detections.length > 0) {
      requestAnimationFrame(drawBoundingBoxes);
    }
  });
}

// ==============================================================================
// BACKEND HEALTH CHECK
// ==============================================================================
async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      if (data.model_mode === 'REAL_MODEL') {
        elements.modelModeBadge.textContent = 'AI MODEL ACTIVE • REAL PRETRAINED MODEL';
      } else {
        elements.modelModeBadge.textContent = 'AI MODEL READY • CV SALIENCY & DEMO ENGINE';
      }
    }
  } catch (err) {
    console.warn('Backend offline or health check failed:', err);
    if (elements.modelModeBadge) {
      elements.modelModeBadge.textContent = 'BACKEND OFFLINE';
    }
  }
}

// ==============================================================================
// FILE HANDLING & PREVIEW
// ==============================================================================
function validateFile(file) {
  if (!file) return false;
  
  const maxSizeBytes = 25 * 1024 * 1024; // 25 MB
  const fileExt = '.' + (file.name.split('.').pop() || '').toLowerCase();
  const validExts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.jfif', '.gif'];

  // Accept if valid MIME type or recognized extension
  const isImageMime = file.type && (file.type.startsWith('image/') || file.type === 'application/octet-stream');
  const isValidExt = validExts.includes(fileExt);

  if (!isImageMime && !isValidExt) {
    showError(
      'Unsupported File Format',
      `"${file.name}" is not recognized as a supported image. Please upload a JPG, PNG, WEBP, BMP, or TIFF file.`
    );
    return false;
  }

  if (file.size > maxSizeBytes) {
    showError(
      'File Too Large',
      `The selected file is ${(file.size / (1024 * 1024)).toFixed(1)} MB. Maximum allowed upload size is 25 MB.`
    );
    return false;
  }

  return true;
}

function handleFileSelected(file) {
  hideError();
  if (!validateFile(file)) return;

  state.selectedFile = file;
  state.detections = [];
  state.highlightedIndex = -1;

  const reader = new FileReader();
  reader.onload = (e) => {
    displayImagePreview(e.target.result, file.name);
  };
  reader.onerror = () => {
    showError('File Read Error', 'Failed to read the selected file. Please try again.');
  };
  reader.readAsDataURL(file);
}

async function loadSampleImage(url, filename) {
  hideError();
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const blob = await res.blob();
    const file = new File([blob], filename, { type: 'image/jpeg' });
    handleFileSelected(file);
  } catch (err) {
    showError('Sample Load Error', `Unable to load sample sonar scan: ${err.message}`);
  }
}

function displayImagePreview(imageSrc, filename) {
  elements.sonarPreviewImg.src = imageSrc;
  elements.sonarPreviewImg.onload = () => {
    state.naturalWidth = elements.sonarPreviewImg.naturalWidth;
    state.naturalHeight = elements.sonarPreviewImg.naturalHeight;
    elements.fileMetaBadge.textContent = `${state.naturalWidth} × ${state.naturalHeight} px`;

    // Clear canvas
    clearCanvas();

    // Show preview workspace, hide initial upload zone
    elements.uploadSection.classList.add('hidden');
    elements.previewSection.classList.remove('hidden');

    // Reset results pane to idle state
    resetResultsSection();

    // Smooth scroll to preview
    elements.previewSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Automatically trigger object detection for quick preview
    runDetectionInference();
  };
}

function resetToUploadState() {
  state.selectedFile = null;
  state.detections = [];
  state.highlightedIndex = -1;
  elements.imageFileInput.value = '';
  clearCanvas();
  elements.previewSection.classList.add('hidden');
  elements.uploadSection.classList.remove('hidden');
  hideError();
}

function resetResultsSection() {
  elements.resultsCountBadge.textContent = 'Ready';
  elements.resultsCountBadge.className = 'badge badge-count';
  elements.resultsIdleState.classList.remove('hidden');
  elements.noObjectsState.classList.add('hidden');
  elements.detectionsList.classList.add('hidden');
  elements.detectionsList.innerHTML = '';
  elements.jsonInspectorWrapper.classList.add('hidden');
  elements.rawJsonResponse.classList.add('hidden');
  elements.jsonChevron.classList.remove('open');
  elements.viewportStatusText.textContent = 'Sonar Image Preview';
}

// ==============================================================================
// INFERENCE API EXECUTION (POST /predict)
// ==============================================================================
async function runDetectionInference() {
  if (!state.selectedFile) {
    showError('No Image Selected', 'Please upload or select a sonar image first.');
    return;
  }

  hideError();
  setLoadingState(true);

  const formData = new FormData();
  formData.append('file', state.selectedFile, state.selectedFile.name);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = 'Inference server error occurred.';
      try {
        const errorJson = await response.json();
        if (errorJson.detail) errorDetail = errorJson.detail;
      } catch (e) {
        errorDetail = `Server responded with status ${response.status} (${response.statusText})`;
      }
      throw new Error(errorDetail);
    }

    const data = await response.json();
    handleDetectionResults(data);
  } catch (err) {
    console.error('Detection API error:', err);
    showError(
      'Detection Failed',
      err.message || 'Unable to connect to the sonar inference API. Please ensure the backend is running.'
    );
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  state.isLoading = isLoading;
  if (isLoading) {
    elements.loadingOverlay.classList.remove('hidden');
    elements.detectObjectsBtn.disabled = true;
    elements.detectObjectsBtn.innerHTML = `
      <span class="status-indicator"></span>
      Analyzing acoustic returns...
    `;
    elements.viewportStatusText.textContent = 'Acoustic Signal Processing...';
  } else {
    elements.loadingOverlay.classList.add('hidden');
    elements.detectObjectsBtn.disabled = false;
    elements.detectObjectsBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="btn-icon">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        <line x1="11" y1="8" x2="11" y2="14"></line>
        <line x1="8" y1="11" x2="14" y2="11"></line>
      </svg>
      Re-analyze Image
    `;
  }
}

// ==============================================================================
// RESULTS RENDERING & CLASS COLOR PALETTES
// ==============================================================================
function getClassColorInfo(className) {
  const c = (className || '').toLowerCase();
  if (c.includes('small')) {
    return {
      primary: '#00f5d4', // Electric Cyan
      glow: 'rgba(0, 245, 212, 0.7)',
      fillLow: 'rgba(0, 245, 212, 0.10)',
      fillHigh: 'rgba(0, 245, 212, 0.28)',
      themeClass: 'cyan',
      label: 'Small Object Target'
    };
  }
  if (c.includes('anomaly') || c.includes('unknown') || c.includes('unidentified')) {
    return {
      primary: '#c084fc', // Purple / Violet
      glow: 'rgba(192, 132, 252, 0.7)',
      fillLow: 'rgba(192, 132, 252, 0.10)',
      fillHigh: 'rgba(192, 132, 252, 0.28)',
      themeClass: 'purple',
      label: 'Unknown Anomaly'
    };
  }
  if (c.includes('net') || c.includes('ghost')) {
    return {
      primary: '#f43f5e', // Rose Coral
      glow: 'rgba(244, 63, 94, 0.7)',
      fillLow: 'rgba(244, 63, 94, 0.10)',
      fillHigh: 'rgba(244, 63, 94, 0.28)',
      themeClass: 'rose',
      label: 'Ghost Fishing Net'
    };
  }
  if (c.includes('ship') || c.includes('wreck') || c.includes('vessel')) {
    return {
      primary: '#ffb703', // Gold Amber
      glow: 'rgba(255, 183, 3, 0.7)',
      fillLow: 'rgba(255, 183, 3, 0.10)',
      fillHigh: 'rgba(255, 183, 3, 0.28)',
      themeClass: 'amber',
      label: 'Sunken Shipwreck / Vessel'
    };
  }
  if (c.includes('rod') || c.includes('pipe') || c.includes('steel') || c.includes('structure')) {
    return {
      primary: '#fb923c', // Warm Orange
      glow: 'rgba(251, 146, 60, 0.7)',
      fillLow: 'rgba(251, 146, 60, 0.10)',
      fillHigh: 'rgba(251, 146, 60, 0.28)',
      themeClass: 'orange',
      label: 'Heavy Steel Rod / Structure'
    };
  }
  if (c.includes('pot') || c.includes('trap')) {
    return {
      primary: '#10b981', // Emerald Green
      glow: 'rgba(16, 185, 129, 0.7)',
      fillLow: 'rgba(16, 185, 129, 0.10)',
      fillHigh: 'rgba(16, 185, 129, 0.28)',
      themeClass: 'emerald',
      label: 'Crab Pot Target'
    };
  }
  return {
    primary: '#38bdf8', // Sky Blue
    glow: 'rgba(56, 189, 248, 0.7)',
    fillLow: 'rgba(56, 189, 248, 0.10)',
    fillHigh: 'rgba(56, 189, 248, 0.28)',
    themeClass: 'sky',
    label: className || 'Marine Debris'
  };
}

function handleDetectionResults(data) {
  const detections = (data && Array.isArray(data.detections)) ? data.detections : [];
  state.detections = detections;

  // Update Raw JSON inspector
  elements.rawJsonResponse.textContent = JSON.stringify(data, null, 2);
  elements.jsonInspectorWrapper.classList.remove('hidden');

  // Hide idle state
  elements.resultsIdleState.classList.add('hidden');

  if (detections.length === 0) {
    // Zero objects detected state
    elements.noObjectsState.classList.remove('hidden');
    elements.detectionsList.classList.add('hidden');
    elements.resultsCountBadge.textContent = '0 Objects';
    elements.resultsCountBadge.className = 'badge badge-count';
    elements.viewportStatusText.textContent = 'Analysis Complete: 0 Objects Detected';
    clearCanvas();
  } else {
    // Objects detected state
    elements.noObjectsState.classList.add('hidden');
    elements.detectionsList.classList.remove('hidden');
    elements.resultsCountBadge.textContent = `${detections.length} Detected`;
    elements.resultsCountBadge.className = 'badge badge-count active';
    elements.viewportStatusText.textContent = `Analysis Complete: ${detections.length} Target${detections.length > 1 ? 's' : ''} Found`;

    renderDetectionCards(detections);
    drawBoundingBoxes();
  }
}

function renderDetectionCards(detections) {
  elements.detectionsList.innerHTML = '';

  detections.forEach((det, index) => {
    const card = document.createElement('div');
    const colorInfo = getClassColorInfo(det.class);
    card.className = `detection-item-card ${colorInfo.themeClass}`;
    card.id = `detectionCard_${index}`;

    // Confidence percentage
    const confPercent = Math.round(det.confidence * 100);

    card.innerHTML = `
      <div class="detection-card-top">
        <div class="detection-class-group">
          <span class="field-label">Target #${index + 1} &bull; ${colorInfo.label}</span>
          <span class="class-value">${escapeHtml(det.class)}</span>
        </div>
        <div class="confidence-badge">
          <span class="field-label">Confidence</span>
          <span class="confidence-val" style="color: ${colorInfo.primary};">${confPercent}%</span>
          <div class="confidence-meter-track">
            <div class="confidence-meter-fill" style="width: ${confPercent}%; background: ${colorInfo.primary};"></div>
          </div>
        </div>
      </div>

      <div class="bbox-box">
        <div class="bbox-title">Bounding Box Coordinates</div>
        <div class="bbox-grid">
          <div class="bbox-coord-cell">
            <span class="coord-label">X</span>
            <span class="coord-val">${Math.round(det.x)}</span>
          </div>
          <div class="bbox-coord-cell">
            <span class="coord-label">Y</span>
            <span class="coord-val">${Math.round(det.y)}</span>
          </div>
          <div class="bbox-coord-cell">
            <span class="coord-label">Width</span>
            <span class="coord-val">${Math.round(det.width)}</span>
          </div>
          <div class="bbox-coord-cell">
            <span class="coord-label">Height</span>
            <span class="coord-val">${Math.round(det.height)}</span>
          </div>
        </div>
      </div>
    `;

    // Hover interaction linking list card to canvas bounding box
    card.addEventListener('mouseenter', () => {
      state.highlightedIndex = index;
      drawBoundingBoxes();
      card.classList.add('highlighted');
    });

    card.addEventListener('mouseleave', () => {
      state.highlightedIndex = -1;
      drawBoundingBoxes();
      card.classList.remove('highlighted');
    });

    elements.detectionsList.appendChild(card);
  });
}

// ==============================================================================
// HIGH-PRECISION CANVAS BOUNDING BOX ENGINE
// ==============================================================================
function setupCanvasResizeObserver() {
  const resizeObserver = new ResizeObserver(() => {
    if (state.detections.length > 0) {
      requestAnimationFrame(drawBoundingBoxes);
    }
  });

  if (elements.sonarPreviewImg) {
    resizeObserver.observe(elements.sonarPreviewImg);
  }
}

function clearCanvas() {
  const canvas = elements.detectionCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawBoundingBoxes() {
  const img = elements.sonarPreviewImg;
  const canvas = elements.detectionCanvas;
  if (!img || !canvas || !state.naturalWidth || !state.naturalHeight) return;

  const displayWidth = img.clientWidth;
  const displayHeight = img.clientHeight;

  if (displayWidth === 0 || displayHeight === 0) return;

  // Set canvas resolution to match pixel-ratio and screen size
  const dpr = window.devicePixelRatio || 1;
  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  if (state.detections.length === 0) return;

  // Exact render bounds calculation to prevent letterboxing coordinate drift
  const imgAspect = state.naturalWidth / state.naturalHeight;
  const containerAspect = displayWidth / displayHeight;
  let renderWidth = displayWidth;
  let renderHeight = displayHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (Math.abs(containerAspect - imgAspect) > 0.005) {
    if (containerAspect > imgAspect) {
      renderWidth = displayHeight * imgAspect;
      offsetX = (displayWidth - renderWidth) / 2;
    } else {
      renderHeight = displayWidth / imgAspect;
      offsetY = (displayHeight - renderHeight) / 2;
    }
  }

  const scaleX = renderWidth / state.naturalWidth;
  const scaleY = renderHeight / state.naturalHeight;

  state.detections.forEach((det, idx) => {
    const isHighlighted = state.highlightedIndex === idx;
    const colorInfo = getClassColorInfo(det.class);

    // Exact pixel-aligned screen coordinates
    const x = offsetX + (det.x * scaleX);
    const y = offsetY + (det.y * scaleY);
    const w = det.width * scaleX;
    const h = det.height * scaleY;

    const primaryColor = colorInfo.primary;
    const glowColor = colorInfo.glow;

    ctx.save();

    // 1. Box Glow Effect
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = isHighlighted ? 24 : 10;

    // 2. Outer Bounding Rectangle
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = isHighlighted ? 3.5 : 2;
    ctx.strokeRect(x, y, w, h);

    // 3. Subtle Semi-transparent Fill
    ctx.fillStyle = isHighlighted ? colorInfo.fillHigh : colorInfo.fillLow;
    ctx.fillRect(x, y, w, h);

    // 4. Corner Accents for high-tech sonar HUD appearance
    const cornerLen = Math.min(14, Math.max(5, w / 4), Math.max(5, h / 4));
    ctx.lineWidth = isHighlighted ? 4.5 : 3;
    ctx.beginPath();
    // Top-Left
    ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
    // Top-Right
    ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen);
    // Bottom-Left
    ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h);
    // Bottom-Right
    ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen);
    ctx.stroke();

    // 5. Target HUD Label Tag
    const labelText = `${det.class} ${Math.round(det.confidence * 100)}%`;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(labelText).width;
    const tagHeight = 21;
    const tagWidth = textWidth + 16;

    // Position tag on top of box or inside if too close to top
    let tagY = y - tagHeight - 4;
    if (tagY < 4) tagY = y + 4;

    let tagX = x;
    if (tagX + tagWidth > displayWidth) tagX = Math.max(0, displayWidth - tagWidth - 4);

    // Tag background
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#060f22';
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);

    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(tagX, tagY, tagWidth, tagHeight);

    // Tag text
    ctx.fillStyle = primaryColor;
    ctx.fillText(labelText, tagX + 8, tagY + 14);

    ctx.restore();
  });
}

// ==============================================================================
// ERROR HANDLING UTILITIES
// ==============================================================================
function showError(title, message) {
  if (elements.errorTitle) elements.errorTitle.textContent = title;
  if (elements.errorMessage) elements.errorMessage.textContent = message;
  if (elements.errorBanner) {
    elements.errorBanner.classList.remove('hidden');
    elements.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function hideError() {
  if (elements.errorBanner) {
    elements.errorBanner.classList.add('hidden');
  }
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
