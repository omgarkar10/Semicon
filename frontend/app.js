const state = {
  selectedFile: null,
  parsedPreview: null,
  restoredNpyBase64: null,
  restoredFilename: "restored.npy",
};

const elements = {
  backendStatus: document.getElementById("backend-status"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  browseBtn: document.getElementById("browse-btn"),
  fileMeta: document.getElementById("file-meta"),
  metaFilename: document.getElementById("meta-filename"),
  metaSize: document.getElementById("meta-size"),
  metaShape: document.getElementById("meta-shape"),
  restoreBtn: document.getElementById("restore-btn"),
  restoreStatus: document.getElementById("restore-status"),
  inputCanvas: document.getElementById("input-canvas"),
  inputPlaceholder: document.getElementById("input-placeholder"),
  compareInputCanvas: document.getElementById("compare-input-canvas"),
  compareInputPlaceholder: document.getElementById("compare-input-placeholder"),
  outputImage: document.getElementById("output-image"),
  outputPlaceholder: document.getElementById("output-placeholder"),
  metaInputRes: document.getElementById("meta-input-res"),
  metaOutputRes: document.getElementById("meta-output-res"),
  metaTime: document.getElementById("meta-time"),
  metaDevice: document.getElementById("meta-device"),
  metaTta: document.getElementById("meta-tta"),
  downloadBtn: document.getElementById("download-btn"),
};

function setBackendStatus(mode, text) {
  elements.backendStatus.classList.remove("online", "offline");
  if (mode === "online") {
    elements.backendStatus.classList.add("online");
  } else if (mode === "offline") {
    elements.backendStatus.classList.add("offline");
  }
  elements.statusText.textContent = text;
}

async function checkBackendHealth() {
  setBackendStatus("checking", "Checking");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`);
    }
    const data = await response.json();
    if (data.model_loaded) {
      setBackendStatus("online", "Online");
    } else {
      setBackendStatus("offline", "Offline");
    }
  } catch (error) {
    setBackendStatus("offline", "Offline");
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function parseNpyBuffer(buffer) {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  const expected = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < expected.length; i += 1) {
    if (magic[i] !== expected[i]) {
      throw new Error("Not a valid .npy file.");
    }
  }

  const major = view.getUint8(6);
  const minor = view.getUint8(7);
  let headerLength;
  let headerOffset;

  if (major === 1) {
    headerLength = view.getUint16(8, true);
    headerOffset = 10;
  } else if (major === 2) {
    headerLength = view.getUint32(8, true);
    headerOffset = 12;
  } else if (major === 3) {
    headerLength = view.getUint32(8, true);
    headerOffset = 12;
  } else {
    throw new Error(`Unsupported NumPy format version ${major}.${minor}.`);
  }

  const headerText = new TextDecoder("ascii").decode(
    new Uint8Array(buffer, headerOffset, headerLength),
  );
  const header = parsePythonDictHeader(headerText);
  const descr = header.descr;
  const shape = header.shape;
  const fortranOrder = header.fortran_order;

  if (fortranOrder) {
    throw new Error("Fortran-order arrays are not supported for preview.");
  }

  const bytesOffset = headerOffset + headerLength;
  const values = readNumericArray(buffer, bytesOffset, descr, shape);
  return { shape, values };
}

function parsePythonDictHeader(text) {
  const descrMatch = text.match(/'descr':\s*'([^']+)'/);
  const shapeMatch = text.match(/'shape':\s*\(([^)]*)\)/);
  const fortranMatch = text.match(/'fortran_order':\s*(True|False)/);

  if (!descrMatch || !shapeMatch || !fortranMatch) {
    throw new Error("Unable to parse NumPy header.");
  }

  const shapeParts = shapeMatch[1]
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);

  return {
    descr: descrMatch[1],
    shape: shapeParts,
    fortran_order: fortranMatch[1] === "True",
  };
}

function readNumericArray(buffer, offset, descr, shape) {
  const littleEndian = descr.startsWith("<") || descr.startsWith("|");
  const typeCode = descr.slice(1);
  const view = new DataView(buffer);
  const count = shape.reduce((acc, dim) => acc * dim, 1);
  const values = new Float32Array(count);
  let byteOffset = offset;

  for (let i = 0; i < count; i += 1) {
    values[i] = readTypedValue(view, byteOffset, typeCode, littleEndian);
    byteOffset += dtypeSize(typeCode);
  }

  return values;
}

function dtypeSize(typeCode) {
  const sizes = {
    f2: 2,
    f4: 4,
    f8: 8,
    i1: 1,
    i2: 2,
    i4: 4,
    u1: 1,
    u2: 2,
    u4: 4,
  };
  const size = sizes[typeCode];
  if (!size) {
    throw new Error(`Unsupported dtype for preview: ${typeCode}`);
  }
  return size;
}

function readTypedValue(view, offset, typeCode, littleEndian) {
  switch (typeCode) {
    case "f2":
      return view.getFloat16(offset, littleEndian);
    case "f4":
      return view.getFloat32(offset, littleEndian);
    case "f8":
      return view.getFloat64(offset, littleEndian);
    case "i1":
      return view.getInt8(offset);
    case "i2":
      return view.getInt16(offset, littleEndian);
    case "i4":
      return view.getInt32(offset, littleEndian);
    case "u1":
      return view.getUint8(offset);
    case "u2":
      return view.getUint16(offset, littleEndian);
    case "u4":
      return view.getUint32(offset, littleEndian);
    default:
      throw new Error(`Unsupported dtype for preview: ${typeCode}`);
  }
}

function normalizePreviewShape(shape) {
  if (shape.length === 2) {
    return { height: shape[0], width: shape[1], displayShape: `(${shape[0]}, ${shape[1]})` };
  }
  if (shape.length === 3 && shape[2] === 1) {
    return {
      height: shape[0],
      width: shape[1],
      displayShape: `(${shape[0]}, ${shape[1]}, 1)`,
    };
  }
  throw new Error(`Unsupported shape for preview: (${shape.join(", ")})`);
}

function drawGrayscaleCanvas(canvas, values, width, height, placeholder) {
  const ctx = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    min = Math.min(min, values[i]);
    max = Math.max(max, values[i]);
  }
  const range = max - min || 1;

  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < values.length; i += 1) {
    const gray = Math.round(((values[i] - min) / range) * 255);
    const idx = i * 4;
    imageData.data[idx] = gray;
    imageData.data[idx + 1] = gray;
    imageData.data[idx + 2] = gray;
    imageData.data[idx + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
  placeholder.hidden = true;
}

function clearOutputPreview() {
  elements.outputImage.hidden = true;
  elements.outputImage.removeAttribute("src");
  elements.outputPlaceholder.hidden = false;
  elements.downloadBtn.disabled = true;
  state.restoredNpyBase64 = null;
}

function resetMetadata() {
  elements.metaInputRes.textContent = "—";
  elements.metaOutputRes.textContent = "—";
  elements.metaTime.textContent = "—";
  elements.metaDevice.textContent = "—";
  elements.metaTta.textContent = "—";
}

async function handleSelectedFile(file) {
  if (!file.name.toLowerCase().endsWith(".npy")) {
    elements.restoreStatus.textContent = "Please select a .npy file.";
    return;
  }

  state.selectedFile = file;
  clearOutputPreview();
  resetMetadata();
  elements.restoreStatus.textContent = "";

  try {
    const buffer = await file.arrayBuffer();
    const parsed = parseNpyBuffer(buffer);
    const shapeInfo = normalizePreviewShape(parsed.shape);
    state.parsedPreview = {
      values: parsed.values,
      width: shapeInfo.width,
      height: shapeInfo.height,
    };

    elements.fileMeta.hidden = false;
    elements.metaFilename.textContent = file.name;
    elements.metaSize.textContent = formatBytes(file.size);
    elements.metaShape.textContent = shapeInfo.displayShape;
    elements.restoreBtn.disabled = false;

    drawGrayscaleCanvas(
      elements.inputCanvas,
      parsed.values,
      shapeInfo.width,
      shapeInfo.height,
      elements.inputPlaceholder,
    );
    drawGrayscaleCanvas(
      elements.compareInputCanvas,
      parsed.values,
      shapeInfo.width,
      shapeInfo.height,
      elements.compareInputPlaceholder,
    );
  } catch (error) {
    state.selectedFile = null;
    state.parsedPreview = null;
    elements.restoreBtn.disabled = true;
    elements.fileMeta.hidden = true;
    elements.inputPlaceholder.hidden = false;
    elements.compareInputPlaceholder.hidden = false;
    elements.restoreStatus.textContent = error.message || "Unable to preview this file.";
  }
}

async function restoreImage() {
  if (!state.selectedFile) {
    return;
  }

  elements.restoreBtn.disabled = true;
  elements.restoreStatus.textContent = "Processing...";

  const formData = new FormData();
  formData.append("file", state.selectedFile, state.selectedFile.name);

  try {
    elements.restoreStatus.textContent = "Restoring...";
    const response = await fetch("/api/restore", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = payload.detail || `Restore failed (${response.status})`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    elements.outputImage.src = `data:image/png;base64,${payload.image_png}`;
    elements.outputImage.hidden = false;
    elements.outputPlaceholder.hidden = true;

    state.restoredNpyBase64 = payload.restored_npy;
    state.restoredFilename = payload.restored_npy_filename || "restored.npy";
    elements.downloadBtn.disabled = false;

    elements.metaInputRes.textContent = payload.input_shape.join(" × ");
    elements.metaOutputRes.textContent = payload.output_shape.join(" × ");
    elements.metaTime.textContent = `${payload.inference_time.toFixed(3)} s`;
    elements.metaDevice.textContent = payload.device;
    elements.metaTta.textContent = String(payload.tta_count);

    elements.restoreStatus.textContent = "Completed";
  } catch (error) {
    elements.restoreStatus.textContent = error.message || "Restore failed.";
  } finally {
    elements.restoreBtn.disabled = Boolean(state.selectedFile);
  }
}

function downloadRestoredNpy() {
  if (!state.restoredNpyBase64) {
    return;
  }

  const binary = atob(state.restoredNpyBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = state.restoredFilename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindUploadEvents() {
  elements.browseBtn.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      handleSelectedFile(file);
    }
  });

  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      handleSelectedFile(file);
    }
  });
}

elements.restoreBtn.addEventListener("click", restoreImage);
elements.downloadBtn.addEventListener("click", downloadRestoredNpy);

bindUploadEvents();
checkBackendHealth();
setInterval(checkBackendHealth, 30000);
