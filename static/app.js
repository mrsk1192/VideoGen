const SUPPORTED_LANGS = ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "ar"];
const DEFAULT_LANG = "en";
const LANG_STORAGE_KEY = "videogen_lang";

const I18N = {
  en: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "Text-to-Video / Image-to-Video with server-side model search & download",
    languageLabel: "Language",
    runtimeLoading: "runtime: loading...",
    tabTextToVideo: "Text to Video",
    tabImageToVideo: "Image to Video",
    tabModels: "Models",
    tabSettings: "Settings",
    labelPrompt: "Prompt",
    labelNegativePrompt: "Negative Prompt",
    labelModelOptional: "Model ID (optional)",
    labelModelSelect: "Model Selection",
    labelSteps: "Steps",
    labelFrames: "Frames",
    labelGuidance: "Guidance",
    labelFps: "FPS",
    labelSeed: "Seed",
    labelInputImage: "Input Image",
    labelWidth: "Width",
    labelHeight: "Height",
    btnGenerateTextVideo: "Generate Text Video",
    btnGenerateImageVideo: "Generate Image Video",
    btnRefreshModels: "Refresh",
    labelDownloadSavePathOptional: "Download Save Path (optional)",
    btnBrowsePath: "Browse",
    labelTask: "Task",
    labelQuery: "Query",
    labelLimit: "Limit",
    btnSearchModels: "Search Models",
    headingLocalModels: "Local Models",
    btnRefreshLocalList: "Refresh Local List",
    labelModelsDirectory: "Models Directory",
    labelOutputsDirectory: "Outputs Directory",
    labelTempDirectory: "Temp Directory",
    labelHfToken: "HF Token",
    labelDefaultTextModel: "Default Text Model",
    labelDefaultImageModel: "Default Image Model",
    labelDefaultSteps: "Default Steps",
    labelDefaultFrames: "Default Frames",
    labelDefaultGuidance: "Default Guidance",
    labelDefaultFps: "Default FPS",
    labelDefaultWidth: "Default Width",
    labelDefaultHeight: "Default Height",
    btnSaveSettings: "Save Settings",
    headingPathBrowser: "Folder Browser",
    btnClosePathBrowser: "Close",
    labelCurrentPath: "Current Path",
    btnRoots: "Roots",
    btnUpFolder: "Up",
    btnUseThisPath: "Use This Path",
    statusNoTask: "No task running.",
    placeholderT2VPrompt: "A cinematic drone shot above neon city...",
    placeholderNegativePrompt: "low quality, blurry",
    placeholderI2VPrompt: "Turn this image into a smooth cinematic motion...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderSeed: "random if empty",
    placeholderDownloadSavePath: "empty = use Models Directory from Settings",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderOptional: "optional",
    msgSettingsSaved: "Settings saved.",
    msgNoLocalModels: "No local models in: {path}",
    msgNoModelsFound: "No models found.",
    msgDefaultModelOption: "Use default model ({model})",
    msgDefaultModelNoMeta: "Use default model",
    msgNoModelCatalog: "No models available.",
    msgModelSelectHint: "Select a model to see thumbnail.",
    msgModelNoPreview: "No thumbnail available for this model.",
    msgNoFolders: "No subfolders found.",
    msgOpen: "Open",
    btnDownload: "Download",
    msgModelPreviewAlt: "Model preview",
    msgModelDownloadStarted: "Model download started: {repo} -> {path}",
    msgTextGenerationStarted: "Text generation started: {id}",
    msgImageGenerationStarted: "Image generation started: {id}",
    msgTaskPollFailed: "Task poll failed: {error}",
    msgSaveSettingsFailed: "Save settings failed: {error}",
    msgSearchFailed: "Search failed: {error}",
    msgTextGenerationFailed: "Text generation failed: {error}",
    msgImageGenerationFailed: "Image generation failed: {error}",
    msgLocalModelRefreshFailed: "Local model refresh failed: {error}",
    msgPathBrowserLoadFailed: "Folder browser load failed: {error}",
    msgInitFailed: "Initialization failed: {error}",
    msgInputImageRequired: "Input image is required.",
    msgDefaultModelsDir: "default models dir",
    msgUnknownPath: "(unknown)",
    modelTag: "tag",
    modelDownloads: "downloads",
    modelLikes: "likes",
    taskTypeText2Video: "text2video",
    taskTypeImage2Video: "image2video",
    taskTypeDownload: "download",
    taskTypeUnknown: "unknown",
    statusQueued: "queued",
    statusRunning: "running",
    statusCompleted: "completed",
    statusError: "error",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    runtimeLoadFailed: "runtime load failed: {error}",
    serverQueued: "Queued",
    serverGenerationQueued: "Generation queued",
    serverDownloadQueued: "Download queued",
    serverLoadingModel: "Loading model",
    serverPreparingImage: "Preparing image",
    serverGeneratingFrames: "Generating frames",
    serverEncoding: "Encoding mp4",
    serverDone: "Done",
    serverGenerationFailed: "Generation failed",
    serverDownloadComplete: "Download complete",
    serverDownloadFailed: "Download failed",
  },
  ja: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "テキスト動画生成 / 画像動画生成（サーバー側モデル検索・ダウンロード対応）",
    languageLabel: "言語",
    runtimeLoading: "実行環境: 読み込み中...",
    tabTextToVideo: "テキスト→動画",
    tabImageToVideo: "画像→動画",
    tabModels: "モデル",
    tabSettings: "設定",
    labelPrompt: "プロンプト",
    labelNegativePrompt: "ネガティブプロンプト",
    labelModelOptional: "モデルID（任意）",
    labelModelSelect: "モデル選択",
    labelSteps: "ステップ数",
    labelFrames: "フレーム数",
    labelGuidance: "ガイダンス",
    labelFps: "FPS",
    labelSeed: "シード",
    labelInputImage: "入力画像",
    labelWidth: "幅",
    labelHeight: "高さ",
    btnGenerateTextVideo: "テキスト動画を生成",
    btnGenerateImageVideo: "画像動画を生成",
    btnRefreshModels: "更新",
    labelDownloadSavePathOptional: "モデル保存先（任意）",
    btnBrowsePath: "参照",
    labelTask: "タスク",
    labelQuery: "検索語",
    labelLimit: "件数",
    btnSearchModels: "モデル検索",
    headingLocalModels: "ローカルモデル",
    btnRefreshLocalList: "ローカル一覧を更新",
    labelModelsDirectory: "モデル保存ディレクトリ",
    labelOutputsDirectory: "出力ディレクトリ",
    labelTempDirectory: "一時ディレクトリ",
    labelHfToken: "HFトークン",
    labelDefaultTextModel: "既定のテキストモデル",
    labelDefaultImageModel: "既定の画像モデル",
    labelDefaultSteps: "既定ステップ数",
    labelDefaultFrames: "既定フレーム数",
    labelDefaultGuidance: "既定ガイダンス",
    labelDefaultFps: "既定FPS",
    labelDefaultWidth: "既定幅",
    labelDefaultHeight: "既定高さ",
    btnSaveSettings: "設定を保存",
    headingPathBrowser: "フォルダブラウザ",
    btnClosePathBrowser: "閉じる",
    labelCurrentPath: "現在のパス",
    btnRoots: "ルート",
    btnUpFolder: "上へ",
    btnUseThisPath: "このパスを使用",
    statusNoTask: "実行中のタスクはありません。",
    placeholderT2VPrompt: "ネオン都市上空を飛ぶシネマティックなドローン映像...",
    placeholderNegativePrompt: "低品質, ぼやけ",
    placeholderI2VPrompt: "この画像に滑らかな映画的モーションを付ける...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderSeed: "空欄でランダム",
    placeholderDownloadSavePath: "空欄 = 設定のモデル保存先を使用",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderOptional: "任意",
    msgSettingsSaved: "設定を保存しました。",
    msgNoLocalModels: "ローカルモデルがありません: {path}",
    msgNoModelsFound: "モデルが見つかりません。",
    msgDefaultModelOption: "既定モデルを使用 ({model})",
    msgDefaultModelNoMeta: "既定モデルを使用",
    msgNoModelCatalog: "利用可能なモデルがありません。",
    msgModelSelectHint: "モデルを選択するとサムネイルを表示します。",
    msgModelNoPreview: "このモデルにはサムネイルがありません。",
    msgNoFolders: "サブフォルダがありません。",
    msgOpen: "開く",
    btnDownload: "ダウンロード",
    msgModelPreviewAlt: "モデルプレビュー",
    msgModelDownloadStarted: "モデルダウンロード開始: {repo} -> {path}",
    msgTextGenerationStarted: "テキスト生成を開始しました: {id}",
    msgImageGenerationStarted: "画像生成を開始しました: {id}",
    msgTaskPollFailed: "タスク確認に失敗: {error}",
    msgSaveSettingsFailed: "設定保存に失敗: {error}",
    msgSearchFailed: "検索に失敗: {error}",
    msgTextGenerationFailed: "テキスト生成に失敗: {error}",
    msgImageGenerationFailed: "画像生成に失敗: {error}",
    msgLocalModelRefreshFailed: "ローカル一覧更新に失敗: {error}",
    msgPathBrowserLoadFailed: "フォルダブラウザの読み込み失敗: {error}",
    msgInitFailed: "初期化に失敗: {error}",
    msgInputImageRequired: "入力画像が必要です。",
    msgDefaultModelsDir: "既定のモデル保存先",
    msgUnknownPath: "(不明)",
    modelTag: "タグ",
    modelDownloads: "DL数",
    modelLikes: "いいね",
    taskTypeText2Video: "テキスト動画",
    taskTypeImage2Video: "画像動画",
    taskTypeDownload: "ダウンロード",
    taskTypeUnknown: "不明",
    statusQueued: "待機中",
    statusRunning: "実行中",
    statusCompleted: "完了",
    statusError: "エラー",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    runtimeLoadFailed: "実行環境の取得に失敗: {error}",
    serverQueued: "キュー待ち",
    serverGenerationQueued: "生成キューに追加",
    serverDownloadQueued: "ダウンロードキューに追加",
    serverLoadingModel: "モデルを読み込み中",
    serverPreparingImage: "画像を準備中",
    serverGeneratingFrames: "フレームを生成中",
    serverEncoding: "mp4に変換中",
    serverDone: "完了",
    serverGenerationFailed: "生成に失敗",
    serverDownloadComplete: "ダウンロード完了",
    serverDownloadFailed: "ダウンロード失敗",
  },
  es: {
    languageLabel: "Idioma",
    tabTextToVideo: "Texto a video",
    tabImageToVideo: "Imagen a video",
    tabModels: "Modelos",
    tabSettings: "Configuración",
    btnGenerateTextVideo: "Generar video desde texto",
    btnGenerateImageVideo: "Generar video desde imagen",
    btnSearchModels: "Buscar modelos",
    headingLocalModels: "Modelos locales",
    btnRefreshLocalList: "Actualizar lista local",
    btnSaveSettings: "Guardar configuración",
    statusNoTask: "No hay tareas en ejecución.",
  },
  fr: {
    languageLabel: "Langue",
    tabTextToVideo: "Texte vers vidéo",
    tabImageToVideo: "Image vers vidéo",
    tabModels: "Modèles",
    tabSettings: "Paramètres",
    btnGenerateTextVideo: "Générer une vidéo (texte)",
    btnGenerateImageVideo: "Générer une vidéo (image)",
    btnSearchModels: "Rechercher des modèles",
    headingLocalModels: "Modèles locaux",
    btnRefreshLocalList: "Rafraîchir la liste locale",
    btnSaveSettings: "Enregistrer",
    statusNoTask: "Aucune tâche en cours.",
  },
  de: {
    languageLabel: "Sprache",
    tabTextToVideo: "Text zu Video",
    tabImageToVideo: "Bild zu Video",
    tabModels: "Modelle",
    tabSettings: "Einstellungen",
    btnGenerateTextVideo: "Textvideo erstellen",
    btnGenerateImageVideo: "Bildvideo erstellen",
    btnSearchModels: "Modelle suchen",
    headingLocalModels: "Lokale Modelle",
    btnRefreshLocalList: "Lokale Liste aktualisieren",
    btnSaveSettings: "Einstellungen speichern",
    statusNoTask: "Keine laufende Aufgabe.",
  },
  it: {
    languageLabel: "Lingua",
    tabTextToVideo: "Testo in video",
    tabImageToVideo: "Immagine in video",
    tabModels: "Modelli",
    tabSettings: "Impostazioni",
    btnGenerateTextVideo: "Genera video da testo",
    btnGenerateImageVideo: "Genera video da immagine",
    btnSearchModels: "Cerca modelli",
    headingLocalModels: "Modelli locali",
    btnRefreshLocalList: "Aggiorna elenco locale",
    btnSaveSettings: "Salva impostazioni",
    statusNoTask: "Nessuna attività in esecuzione.",
  },
  pt: {
    languageLabel: "Idioma",
    tabTextToVideo: "Texto para vídeo",
    tabImageToVideo: "Imagem para vídeo",
    tabModels: "Modelos",
    tabSettings: "Configurações",
    btnGenerateTextVideo: "Gerar vídeo de texto",
    btnGenerateImageVideo: "Gerar vídeo de imagem",
    btnSearchModels: "Buscar modelos",
    headingLocalModels: "Modelos locais",
    btnRefreshLocalList: "Atualizar lista local",
    btnSaveSettings: "Salvar configurações",
    statusNoTask: "Nenhuma tarefa em execução.",
  },
  ru: {
    languageLabel: "Язык",
    tabTextToVideo: "Текст в видео",
    tabImageToVideo: "Изображение в видео",
    tabModels: "Модели",
    tabSettings: "Настройки",
    btnGenerateTextVideo: "Сгенерировать видео из текста",
    btnGenerateImageVideo: "Сгенерировать видео из изображения",
    btnSearchModels: "Искать модели",
    headingLocalModels: "Локальные модели",
    btnRefreshLocalList: "Обновить локальный список",
    btnSaveSettings: "Сохранить настройки",
    statusNoTask: "Нет активных задач.",
  },
  ar: {
    languageLabel: "اللغة",
    tabTextToVideo: "نص إلى فيديو",
    tabImageToVideo: "صورة إلى فيديو",
    tabModels: "النماذج",
    tabSettings: "الإعدادات",
    btnGenerateTextVideo: "إنشاء فيديو من النص",
    btnGenerateImageVideo: "إنشاء فيديو من الصورة",
    btnSearchModels: "بحث عن النماذج",
    headingLocalModels: "النماذج المحلية",
    btnRefreshLocalList: "تحديث القائمة المحلية",
    btnSaveSettings: "حفظ الإعدادات",
    statusNoTask: "لا توجد مهام قيد التشغيل.",
  },
};

const state = {
  currentTaskId: null,
  pollTimer: null,
  settings: null,
  language: DEFAULT_LANG,
  modelCatalog: {
    "text-to-video": [],
    "image-to-video": [],
  },
  defaultModels: {
    "text-to-video": "",
    "image-to-video": "",
  },
  pathBrowserTargetInputId: null,
  pathBrowserCurrentPath: "",
  pathBrowserParentPath: null,
  pathBrowserRoots: [],
};

function el(id) {
  return document.getElementById(id);
}

function readNum(id) {
  const raw = el(id).value.trim();
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeLanguage(value) {
  const base = (value || DEFAULT_LANG).toLowerCase().split("-")[0];
  return SUPPORTED_LANGS.includes(base) ? base : DEFAULT_LANG;
}

function t(key, vars = {}) {
  const langPack = I18N[state.language] || I18N[DEFAULT_LANG];
  let template = langPack[key] ?? I18N[DEFAULT_LANG][key] ?? key;
  Object.entries(vars).forEach(([name, value]) => {
    template = template.replaceAll(`{${name}}`, String(value));
  });
  return template;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function detectInitialLanguage() {
  const saved = localStorage.getItem(LANG_STORAGE_KEY);
  if (saved) return normalizeLanguage(saved);
  return normalizeLanguage(navigator.language || DEFAULT_LANG);
}

function applyI18n() {
  document.documentElement.lang = state.language;
  document.documentElement.dir = state.language === "ar" ? "rtl" : "ltr";
  document.title = t("appTitle");
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  if (!state.currentTaskId) {
    showTaskMessage(t("statusNoTask"));
  }
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
}

function setLanguage(languageCode) {
  state.language = normalizeLanguage(languageCode);
  localStorage.setItem(LANG_STORAGE_KEY, state.language);
  el("languageSelect").value = state.language;
  applyI18n();
  loadRuntimeInfo().catch(() => {});
  loadLocalModels().catch(() => {});
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function showTaskMessage(text) {
  el("taskStatus").textContent = text;
}

function setTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      el(`panel-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function translateTaskType(taskType) {
  if (taskType === "text2video") return t("taskTypeText2Video");
  if (taskType === "image2video") return t("taskTypeImage2Video");
  if (taskType === "download") return t("taskTypeDownload");
  return t("taskTypeUnknown");
}

function translateTaskStatus(status) {
  if (status === "queued") return t("statusQueued");
  if (status === "running") return t("statusRunning");
  if (status === "completed") return t("statusCompleted");
  if (status === "error") return t("statusError");
  return status || t("taskTypeUnknown");
}

function translateServerMessage(message) {
  const map = {
    Queued: t("serverQueued"),
    "Generation queued": t("serverGenerationQueued"),
    "Download queued": t("serverDownloadQueued"),
    "Loading model": t("serverLoadingModel"),
    "Preparing image": t("serverPreparingImage"),
    "Generating frames": t("serverGeneratingFrames"),
    "Encoding mp4": t("serverEncoding"),
    Done: t("serverDone"),
    "Generation failed": t("serverGenerationFailed"),
    "Download complete": t("serverDownloadComplete"),
    "Download failed": t("serverDownloadFailed"),
  };
  return map[message] || message || "";
}

async function loadRuntimeInfo() {
  try {
    const info = await api("/api/system/info");
    const flags = [
      `${t("runtimeDevice")}=${info.device}`,
      `${t("runtimeCuda")}=${info.cuda_available}`,
      `${t("runtimeRocm")}=${info.rocm_available}`,
      `${t("runtimeDiffusers")}=${info.diffusers_ready}`,
    ];
    if (info.torch_version) flags.push(`${t("runtimeTorch")}=${info.torch_version}`);
    if (info.import_error) flags.push(`${t("runtimeError")}=${info.import_error}`);
    el("runtimeInfo").textContent = flags.join(" | ");
  } catch (error) {
    el("runtimeInfo").textContent = t("runtimeLoadFailed", { error: error.message });
  }
}

function applySettings(settings) {
  state.settings = settings;
  state.defaultModels["text-to-video"] = settings.defaults.text2video_model || "";
  state.defaultModels["image-to-video"] = settings.defaults.image2video_model || "";
  el("cfgModelsDir").value = settings.paths.models_dir;
  el("cfgOutputsDir").value = settings.paths.outputs_dir;
  el("cfgTmpDir").value = settings.paths.tmp_dir;
  el("cfgToken").value = settings.huggingface.token || "";
  el("cfgTextModel").value = settings.defaults.text2video_model;
  el("cfgImageModel").value = settings.defaults.image2video_model;
  el("cfgSteps").value = settings.defaults.num_inference_steps;
  el("cfgFrames").value = settings.defaults.num_frames;
  el("cfgGuidance").value = settings.defaults.guidance_scale;
  el("cfgFps").value = settings.defaults.fps;
  el("cfgWidth").value = settings.defaults.width;
  el("cfgHeight").value = settings.defaults.height;
  el("t2vSteps").value = settings.defaults.num_inference_steps;
  el("t2vFrames").value = settings.defaults.num_frames;
  el("t2vGuidance").value = settings.defaults.guidance_scale;
  el("t2vFps").value = settings.defaults.fps;
  el("i2vSteps").value = settings.defaults.num_inference_steps;
  el("i2vFrames").value = settings.defaults.num_frames;
  el("i2vGuidance").value = settings.defaults.guidance_scale;
  el("i2vFps").value = settings.defaults.fps;
  el("i2vWidth").value = settings.defaults.width;
  el("i2vHeight").value = settings.defaults.height;
  if (!el("downloadTargetDir").value.trim()) {
    el("downloadTargetDir").value = settings.paths.models_dir;
  }
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
}

async function loadSettings() {
  const settings = await api("/api/settings");
  applySettings(settings);
}

function getModelDom(task) {
  if (task === "text-to-video") {
    return { selectId: "t2vModelSelect", previewId: "t2vModelPreview" };
  }
  return { selectId: "i2vModelSelect", previewId: "i2vModelPreview" };
}

function renderModelPreview(task) {
  const dom = getModelDom(task);
  const select = el(dom.selectId);
  const preview = el(dom.previewId);
  const selectedValue = select.value;
  const catalog = state.modelCatalog[task] || [];
  const defaultId = state.defaultModels[task] || "";

  let chosen = null;
  if (selectedValue) {
    chosen = catalog.find((item) => item.value === selectedValue) || null;
  } else if (defaultId) {
    chosen = catalog.find((item) => item.id === defaultId || item.value === defaultId) || null;
  }

  if (!chosen) {
    preview.innerHTML = `<p>${t("msgModelSelectHint")}</p>`;
    return;
  }

  const name = escapeHtml(chosen.id || chosen.label || selectedValue || defaultId);
  const modelUrl = chosen.model_url ? escapeHtml(chosen.model_url) : "";
  const infoHtml = modelUrl
    ? `<strong><a href="${modelUrl}" target="_blank" rel="noopener noreferrer">${name}</a></strong>`
    : `<strong>${name}</strong>`;

  const imageHtml = chosen.preview_url
    ? `<img class="model-picked-thumb" src="${escapeHtml(chosen.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`
    : `<div class="model-picked-empty">${escapeHtml(t("msgModelNoPreview"))}</div>`;

  preview.innerHTML = `
    <div class="model-picked-card">
      ${imageHtml}
      <div class="model-picked-meta">${infoHtml}<span>${escapeHtml(chosen.source || "model")}</span></div>
    </div>
  `;
}

function renderModelSelect(task, preferredValue = null) {
  const dom = getModelDom(task);
  const select = el(dom.selectId);
  const currentValue = preferredValue === null ? select.value : preferredValue;
  const items = state.modelCatalog[task] || [];
  const defaultId = state.defaultModels[task] || "";
  const defaultLabel = defaultId ? t("msgDefaultModelOption", { model: defaultId }) : t("msgDefaultModelNoMeta");

  let html = `<option value="">${escapeHtml(defaultLabel)}</option>`;
  html += items
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label || item.id || item.value)}</option>`)
    .join("");
  select.innerHTML = html;

  if (currentValue && items.some((item) => item.value === currentValue)) {
    select.value = currentValue;
  } else {
    select.value = "";
  }
  renderModelPreview(task);
}

async function loadModelCatalog(task, keepSelection = true) {
  const dom = getModelDom(task);
  const prev = keepSelection ? el(dom.selectId).value : "";
  const params = new URLSearchParams({
    task,
    limit: "40",
  });
  const data = await api(`/api/models/catalog?${params.toString()}`);
  state.modelCatalog[task] = data.items || [];
  state.defaultModels[task] = data.default_model || state.defaultModels[task] || "";
  renderModelSelect(task, prev);
}

async function saveSettings(event) {
  event.preventDefault();
  const payload = {
    paths: {
      models_dir: el("cfgModelsDir").value.trim(),
      outputs_dir: el("cfgOutputsDir").value.trim(),
      tmp_dir: el("cfgTmpDir").value.trim(),
    },
    huggingface: {
      token: el("cfgToken").value.trim(),
    },
    defaults: {
      text2video_model: el("cfgTextModel").value.trim(),
      image2video_model: el("cfgImageModel").value.trim(),
      num_inference_steps: Number(el("cfgSteps").value),
      num_frames: Number(el("cfgFrames").value),
      guidance_scale: Number(el("cfgGuidance").value),
      fps: Number(el("cfgFps").value),
      width: Number(el("cfgWidth").value),
      height: Number(el("cfgHeight").value),
    },
  };
  const updated = await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  applySettings(updated);
  showTaskMessage(t("msgSettingsSaved"));
}

function renderLocalModels(items, baseDir = "") {
  const container = el("localModels");
  if (!items.length) {
    container.innerHTML = `<p>${t("msgNoLocalModels", { path: baseDir || t("msgUnknownPath") })}</p>`;
    return;
  }
  container.innerHTML = items
    .map((item) => `<div class="row"><strong>${item.repo_hint}</strong><span>${item.path}</span></div>`)
    .join("");
}

async function loadLocalModels() {
  const dir = el("downloadTargetDir")?.value?.trim() || "";
  const params = new URLSearchParams();
  if (dir) params.set("dir", dir);
  const url = params.toString() ? `/api/models/local?${params.toString()}` : "/api/models/local";
  const data = await api(url);
  renderLocalModels(data.items || [], data.base_dir || dir);
}

function closePathBrowser() {
  el("pathBrowser").classList.remove("active");
  state.pathBrowserTargetInputId = null;
}

function renderPathBrowser(data) {
  state.pathBrowserCurrentPath = data.current_path || "";
  state.pathBrowserRoots = data.roots || [];
  el("pathBrowserCurrentPath").textContent = state.pathBrowserCurrentPath || "-";

  const rootsList = el("pathBrowserRootsList");
  rootsList.innerHTML = (state.pathBrowserRoots || [])
    .map(
      (root) =>
        `<button type="button" class="path-btn path-root-btn" data-path="${encodeURIComponent(root)}">${escapeHtml(root)}</button>`,
    )
    .join("");
  rootsList.querySelectorAll(".path-root-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      await loadPathBrowser(decodeURIComponent(button.dataset.path));
    });
  });

  const list = el("pathBrowserList");
  const dirs = data.directories || [];
  if (!dirs.length) {
    list.innerHTML = `<p>${t("msgNoFolders")}</p>`;
    return;
  }
  list.innerHTML = dirs
    .map(
      (item) => `
      <div class="row path-browser-list-row">
        <div>
          <strong>${escapeHtml(item.name)}</strong>
          <span>${escapeHtml(item.path)}</span>
        </div>
        <button type="button" class="path-btn path-open-btn" data-path="${encodeURIComponent(item.path)}">${t("msgOpen")}</button>
      </div>`,
    )
    .join("");

  list.querySelectorAll(".path-open-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      await loadPathBrowser(decodeURIComponent(button.dataset.path));
    });
  });
}

async function loadPathBrowser(path = "") {
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  const url = params.toString() ? `/api/fs/directories?${params.toString()}` : "/api/fs/directories";
  const data = await api(url);
  state.pathBrowserParentPath = data.parent_path || null;
  renderPathBrowser(data);
}

async function openPathBrowser(targetInputId) {
  state.pathBrowserTargetInputId = targetInputId;
  el("pathBrowser").classList.add("active");
  const initialPath = (el(targetInputId)?.value || "").trim();
  try {
    await loadPathBrowser(initialPath);
  } catch (error) {
    showTaskMessage(t("msgPathBrowserLoadFailed", { error: error.message }));
  }
}

function bindPathBrowser() {
  el("browseDownloadTargetDir").addEventListener("click", async () => {
    await openPathBrowser("downloadTargetDir");
  });
  el("browseCfgModelsDir").addEventListener("click", async () => {
    await openPathBrowser("cfgModelsDir");
  });
  el("pathBrowserClose").addEventListener("click", () => {
    closePathBrowser();
  });
  el("pathBrowserUse").addEventListener("click", async () => {
    const targetId = state.pathBrowserTargetInputId;
    if (!targetId || !state.pathBrowserCurrentPath) return;
    el(targetId).value = state.pathBrowserCurrentPath;
    closePathBrowser();
    if (targetId === "downloadTargetDir") {
      try {
        await loadLocalModels();
      } catch (error) {
        showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
      }
    }
  });
  el("pathBrowserUp").addEventListener("click", async () => {
    if (!state.pathBrowserParentPath) return;
    try {
      await loadPathBrowser(state.pathBrowserParentPath);
    } catch (error) {
      showTaskMessage(t("msgPathBrowserLoadFailed", { error: error.message }));
    }
  });
  el("pathBrowserRoots").addEventListener("click", async () => {
    const firstRoot = (state.pathBrowserRoots || [])[0];
    if (!firstRoot) return;
    try {
      await loadPathBrowser(firstRoot);
    } catch (error) {
      showTaskMessage(t("msgPathBrowserLoadFailed", { error: error.message }));
    }
  });
}

function renderSearchResults(items) {
  const container = el("searchResults");
  if (!items.length) {
    container.innerHTML = `<p>${t("msgNoModelsFound")}</p>`;
    return;
  }
  container.innerHTML = items
    .map(
      (item) => `
      <div class="row model-row">
        <div class="model-main">
          ${
            item.preview_url
              ? `<img class="model-preview" src="${item.preview_url}" alt="${t("msgModelPreviewAlt")}" loading="lazy" onerror="this.style.display='none'" />`
              : ""
          }
          <div>
            <strong><a href="${item.model_url || "#"}" target="_blank" rel="noopener noreferrer">${item.id}</a></strong>
            <span>${t("modelTag")}=${item.pipeline_tag || "n/a"} | ${t("modelDownloads")}=${item.downloads ?? "n/a"} | ${t("modelLikes")}=${item.likes ?? "n/a"}</span>
          </div>
        </div>
        <button type="button" class="download-btn" data-repo="${item.id}">${t("btnDownload")}</button>
      </div>`,
    )
    .join("");
  container.querySelectorAll(".download-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      await startModelDownload(button.dataset.repo);
    });
  });
}

async function searchModels(event) {
  if (event) event.preventDefault();
  const params = new URLSearchParams({
    task: el("searchTask").value,
    query: el("searchQuery").value.trim(),
    limit: String(Number(el("searchLimit").value || "12")),
  });
  const data = await api(`/api/models/search?${params.toString()}`);
  renderSearchResults(data.items || []);
}

async function startModelDownload(repoId) {
  const targetDir = el("downloadTargetDir").value.trim();
  const data = await api("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: repoId,
      target_dir: targetDir || null,
    }),
  });
  showTaskMessage(
    t("msgModelDownloadStarted", {
      repo: repoId,
      path: targetDir || t("msgDefaultModelsDir"),
    }),
  );
  trackTask(data.task_id);
}

async function generateText2Video(event) {
  event.preventDefault();
  const selectedModel = el("t2vModelSelect").value.trim();
  const payload = {
    prompt: el("t2vPrompt").value.trim(),
    negative_prompt: el("t2vNegative").value.trim(),
    model_id: selectedModel || null,
    num_inference_steps: Number(el("t2vSteps").value),
    num_frames: Number(el("t2vFrames").value),
    guidance_scale: Number(el("t2vGuidance").value),
    fps: Number(el("t2vFps").value),
    seed: readNum("t2vSeed"),
  };
  const data = await api("/api/generate/text2video", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showTaskMessage(t("msgTextGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

async function generateImage2Video(event) {
  event.preventDefault();
  const imageFile = el("i2vImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("prompt", el("i2vPrompt").value.trim());
  formData.append("negative_prompt", el("i2vNegative").value.trim());
  formData.append("model_id", el("i2vModelSelect").value.trim());
  formData.append("num_inference_steps", String(Number(el("i2vSteps").value)));
  formData.append("num_frames", String(Number(el("i2vFrames").value)));
  formData.append("guidance_scale", String(Number(el("i2vGuidance").value)));
  formData.append("fps", String(Number(el("i2vFps").value)));
  formData.append("width", String(Number(el("i2vWidth").value)));
  formData.append("height", String(Number(el("i2vHeight").value)));
  if (el("i2vSeed").value.trim()) formData.append("seed", el("i2vSeed").value.trim());

  const data = await api("/api/generate/image2video", {
    method: "POST",
    body: formData,
  });
  showTaskMessage(t("msgImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

function renderTask(task) {
  const base = t("taskLine", {
    id: task.id,
    type: translateTaskType(task.task_type),
    status: translateTaskStatus(task.status),
    progress: Math.round((task.progress || 0) * 100),
    message: translateServerMessage(task.message || ""),
  });
  if (task.error) {
    showTaskMessage(`${base} | ${t("taskError", { error: task.error })}`);
  } else {
    showTaskMessage(base);
  }
  if (task.status === "completed" && task.result?.video_file) {
    const video = el("preview");
    video.src = `/api/videos/${encodeURIComponent(task.result.video_file)}?t=${Date.now()}`;
    video.style.display = "block";
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollTask() {
  if (!state.currentTaskId) return;
  try {
    const task = await api(`/api/tasks/${state.currentTaskId}`);
    renderTask(task);
    if (task.status === "completed" || task.status === "error") {
      stopPolling();
      if (task.task_type === "download" && task.status === "completed") {
        await loadLocalModels();
      }
    }
  } catch (error) {
    stopPolling();
    showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
  }
}

function trackTask(taskId) {
  state.currentTaskId = taskId;
  stopPolling();
  pollTask();
  state.pollTimer = setInterval(pollTask, 3000);
}

function bindLanguageSelector() {
  el("languageSelect").addEventListener("change", (event) => {
    setLanguage(event.target.value);
  });
}

function bindModelSelectors() {
  el("t2vModelSelect").addEventListener("change", () => {
    renderModelPreview("text-to-video");
  });
  el("i2vModelSelect").addEventListener("change", () => {
    renderModelPreview("image-to-video");
  });
  el("refreshT2VModels").addEventListener("click", async () => {
    try {
      await loadModelCatalog("text-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("refreshI2VModels").addEventListener("click", async () => {
    try {
      await loadModelCatalog("image-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
}

async function bootstrap() {
  state.language = detectInitialLanguage();
  bindLanguageSelector();
  setLanguage(state.language);
  setTabs();
  bindPathBrowser();
  bindModelSelectors();

  el("settingsForm").addEventListener("submit", async (event) => {
    try {
      await saveSettings(event);
    } catch (error) {
      showTaskMessage(t("msgSaveSettingsFailed", { error: error.message }));
    }
  });
  el("searchForm").addEventListener("submit", async (event) => {
    try {
      await searchModels(event);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("text2videoForm").addEventListener("submit", async (event) => {
    try {
      await generateText2Video(event);
    } catch (error) {
      showTaskMessage(t("msgTextGenerationFailed", { error: error.message }));
    }
  });
  el("image2videoForm").addEventListener("submit", async (event) => {
    try {
      await generateImage2Video(event);
    } catch (error) {
      showTaskMessage(t("msgImageGenerationFailed", { error: error.message }));
    }
  });
  el("searchTask").addEventListener("change", async () => {
    try {
      await searchModels();
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("refreshLocalModels").addEventListener("click", async () => {
    try {
      await loadLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  try {
    await Promise.all([loadRuntimeInfo(), loadSettings(), loadLocalModels()]);
    await Promise.all([loadModelCatalog("text-to-video", false), loadModelCatalog("image-to-video", false)]);
    await searchModels();
  } catch (error) {
    showTaskMessage(t("msgInitFailed", { error: error.message }));
  }
}

bootstrap();
