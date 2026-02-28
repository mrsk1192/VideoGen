const SUPPORTED_LANGS = ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "ar"];
const DEFAULT_LANG = "en";
const LANG_STORAGE_KEY = "videogen_lang";
const TASK_STORAGE_KEY = "videogen_last_task_id";
const TASK_POLL_INTERVAL_MS = 1000;
const DOWNLOAD_TASK_POLL_INTERVAL_MS = 1500;
const EXTERNAL_I18N_URL = "/static/i18n/messages.json";

const I18N = {
  en: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "Text-to-Image / Image-to-Image / Text-to-Video / Image-to-Video",
    languageLabel: "Language",
    runtimeLoading: "runtime: loading...",
    tabTextToImage: "Text to Image",
    tabImageToImage: "Image to Image",
    tabTextToVideo: "Text to Video",
    tabImageToVideo: "Image to Video",
    tabModels: "Model Search",
    tabLocalModels: "Local Models",
    tabOutputs: "Outputs",
    tabSettings: "Settings",
    labelPrompt: "Prompt",
    labelNegativePrompt: "Negative Prompt",
    labelModelOptional: "Model ID (optional)",
    labelModelSelect: "Model Selection",
    labelSteps: "Steps",
    labelFrames: "Frames",
    labelDurationSeconds: "Duration (sec)",
    labelGuidance: "Guidance",
    labelFps: "FPS",
    labelT2VBackend: "T2V Backend",
    labelSeed: "Seed",
    labelInputImage: "Input Image",
    labelWidth: "Width",
    labelHeight: "Height",
    labelLoraSelect: "LoRA Selection",
    labelVaeSelect: "VAE Selection",
    labelLoraScale: "LoRA Scale",
    labelStrength: "Strength",
    btnGenerateTextVideo: "Generate Text Video",
    btnGenerateImageVideo: "Generate Image Video",
    btnGenerateImageImage: "Generate Image Image",
    btnRefreshModels: "Refresh",
    btnRefreshLoras: "Refresh LoRAs",
    btnRefreshVaes: "Refresh VAEs",
    labelDownloadSavePathOptional: "Download Save Path (optional)",
    btnBrowsePath: "Browse",
    labelTask: "Task",
    labelSearchSource: "Source",
    labelSearchBaseModel: "Base Model",
    labelSearchSort: "Sort",
    labelSearchNsfw: "NSFW",
    labelSearchSizeMinMb: "Min Size (GB)",
    labelSearchSizeMaxMb: "Max Size (GB)",
    labelSearchModelKind: "Model Kind",
    labelSearchViewMode: "View",
    labelQuery: "Query",
    labelLimit: "Limit",
    btnSearchModels: "Search Models",
    headingLocalModels: "Local Models",
    headingOutputs: "Outputs",
    btnRefreshLocalList: "Refresh Local List",
    btnRescanLocalModels: "Rescan",
    btnExpandAllLocalTree: "Expand All",
    btnRefreshOutputs: "Refresh Outputs",
    labelLocalModelsPath: "Local Models Path",
    labelLocalViewMode: "View Mode",
    labelLocalTreeSearch: "Tree Search",
    localViewTree: "Tree",
    localViewFlat: "Flat",
    labelLocalLineage: "Lineage",
    labelModelsDirectory: "Models Directory",
    labelListenPort: "Listen Port",
    labelRocmAotriton: "ROCm AOTriton Experimental",
    labelOutputsDirectory: "Outputs Directory",
    labelTempDirectory: "Temp Directory",
    labelLogLevel: "Log Level",
    labelHfToken: "HF Token",
    labelDefaultT2VBackend: "Default T2V Backend",
    labelT2VNpuRunner: "T2V NPU Runner",
    labelT2VNpuModelDir: "T2V NPU Model Directory",
    labelVramDirectThresholdGb: "VRAM Direct Load Threshold (GB)",
    labelEnableDeviceMapAuto: "Enable device_map='auto'",
    labelEnableModelCpuOffload: "Enable CPU Offload Fallback",
    labelDefaultTextModel: "Default: Text to Video",
    labelDefaultImageModel: "Default: Image to Video",
    labelDefaultTextImageModel: "Default: Text to Image",
    labelDefaultImageImageModel: "Default: Image to Image",
    labelDefaultSteps: "Default Steps",
    labelDefaultFrames: "Default Duration (sec)",
    labelDefaultGuidance: "Default Guidance",
    labelDefaultFps: "Default FPS",
    labelDefaultWidth: "Default Width",
    labelDefaultHeight: "Default Height",
    labelClearHfCache: "Clear Hugging Face Cache",
    helpLanguage:
      "Impact: Changes UI language for labels/messages; generation behavior itself is unchanged.\nExample: English",
    helpPromptText2Video:
      "Impact: Main instruction that drives scene, motion, and style in text-to-video.\nExample: A cinematic drone shot over a neon city at night",
    helpPromptText2Image:
      "Impact: Main instruction that drives composition, subject, and style in text-to-image.\nExample: A photorealistic portrait of a fox in a tailored suit",
    helpNegativePromptText2Image:
      "Impact: Suppresses unwanted artifacts or styles in generated images.\nExample: blurry, low quality, watermark",
    helpLoraSelectText2Image:
      "Impact: Applies LoRA adapters to alter style/subject behavior for this model. Multiple selection is supported.\nExample: your-org/anime-style-lora",
    helpVaeSelectText2Image:
      "Impact: Replaces VAE for text-to-image quality and color/contrast characteristics.\nExample: stabilityai/sd-vae-ft-mse",
    helpModelSelectText2Image:
      "Impact: Changes the text-to-image model and affects style, quality, speed, and VRAM usage.\nExample: runwayml/stable-diffusion-v1-5",
    helpStepsText2Image:
      "Impact: Higher values may improve detail but increase generation time.\nExample: 30",
    helpGuidanceText2Image:
      "Impact: Higher guidance follows prompt more strongly but can reduce natural variation.\nExample: 7.5",
    helpWidthText2Image:
      "Impact: Higher width increases resolution/detail but uses more VRAM.\nExample: 512",
    helpHeightText2Image:
      "Impact: Higher height increases resolution/detail but uses more VRAM.\nExample: 512",
    helpSeedText2Image:
      "Impact: Fixes randomness for reproducible images; blank uses random seed.\nExample: 12345",
    helpPromptImage2Image:
      "Impact: Controls how the input image should be transformed while preserving structure.\nExample: Convert this photo into a cinematic film poster",
    helpNegativePromptImage2Image:
      "Impact: Reduces unwanted artifacts/styles during image-to-image generation.\nExample: blurry, distorted, watermark",
    helpModelSelectImage2Image:
      "Impact: Changes image-to-image model; style quality, speed, and VRAM usage can vary.\nExample: runwayml/stable-diffusion-v1-5",
    helpLoraSelectImage2Image:
      "Impact: Applies LoRA style/subject adaptation to image-to-image output. Multiple selection is supported.\nExample: your-org/lineart-lora",
    helpVaeSelectImage2Image:
      "Impact: Replaces VAE for image-to-image decode quality and color profile.\nExample: stabilityai/sd-vae-ft-mse",
    helpStepsImage2Image:
      "Impact: Higher values may improve detail but increase generation time.\nExample: 30",
    helpGuidanceImage2Image:
      "Impact: Higher guidance follows prompt more strongly but may reduce natural variation.\nExample: 7.5",
    helpStrengthImage2Image:
      "Impact: Controls how much the result changes from the input image.\nExample: 0.8",
    helpWidthImage2Image:
      "Impact: Higher width increases output detail but also VRAM use.\nExample: 512",
    helpHeightImage2Image:
      "Impact: Higher height increases output detail but also VRAM use.\nExample: 512",
    helpSeedImage2Image:
      "Impact: Fixes randomness for reproducible outputs; blank uses random seed.\nExample: 12345",
    helpNegativePromptText2Video:
      "Impact: Suppresses unwanted artifacts or styles during generation.\nExample: blurry, low quality, watermark",
    helpModelSelectText2Video:
      "Impact: Changes the text-to-video model; quality, speed, VRAM usage, and style can vary.\nExample: damo-vilab/text-to-video-ms-1.7b",
    helpLoraSelectText2Video:
      "Impact: Applies LoRA adapters for text-to-video generation when compatible. Multiple selection is supported.\nExample: your-org/cinematic-look-lora",
    helpStepsText2Video:
      "Impact: Higher values usually improve quality but increase time and GPU load.\nExample: 30",
    helpFramesText2Video:
      "Impact: More frames create longer/smoother clips but raise VRAM and render time.\nExample: 16",
    helpDurationSeconds:
      "Impact: Video length in seconds. Longer duration increases generation time and VRAM usage.\nExample: 2.0",
    helpGuidanceText2Video:
      "Impact: Higher guidance follows prompt more strongly but may look less natural.\nExample: 9.0",
    helpFpsText2Video:
      "Impact: Controls output playback smoothness/speed.\nExample: 8",
    helpSeedText2Video:
      "Impact: Fixes randomness for reproducible outputs; blank uses random seed.\nExample: 12345",
    helpInputImage:
      "Impact: Source image defines starting composition and appearance for image-to-video.\nExample: C:\\Images\\input.png",
    helpPromptImage2Video:
      "Impact: Adds motion/style instructions on top of the input image.\nExample: Smooth cinematic camera pan with subtle parallax",
    helpNegativePromptImage2Video:
      "Impact: Reduces unwanted artifacts/flicker in generated video.\nExample: flicker, artifact, noisy",
    helpModelSelectImage2Video:
      "Impact: Changes the image-to-video model and affects motion quality, speed, and VRAM use.\nExample: ali-vilab/i2vgen-xl",
    helpLoraSelectImage2Video:
      "Impact: Applies LoRA adapters for image-to-video generation when compatible. Multiple selection is supported.\nExample: your-org/motion-style-lora",
    helpStepsImage2Video:
      "Impact: Higher values can improve quality but increase generation time.\nExample: 30",
    helpFramesImage2Video:
      "Impact: More frames increase clip duration and memory cost.\nExample: 16",
    helpGuidanceImage2Video:
      "Impact: Higher values increase prompt adherence; too high may reduce naturalness.\nExample: 9.0",
    helpFpsImage2Video:
      "Impact: Sets playback smoothness/speed for encoded video.\nExample: 8",
    helpWidthImage2Video:
      "Impact: Increases output resolution/detail and also VRAM usage.\nExample: 512",
    helpHeightImage2Video:
      "Impact: Increases output resolution/detail and also VRAM usage.\nExample: 512",
    helpSeedImage2Video:
      "Impact: Fixes randomness for reproducibility; blank gives random result each run.\nExample: 12345",
    helpDownloadSavePath:
      "Impact: Sets where downloaded models are stored for this operation.\nExample: D:\\ModelStore\\VideoGen",
    helpTask:
      "Impact: Filters model search target by generation type.\nExample: text-to-video",
    helpSearchSource:
      "Impact: Selects which service to query for models; CivitAI is available for image tasks.\nExample: all",
    helpSearchBaseModel:
      "Impact: Filters search results by base model family; helps narrow compatible checkpoints quickly.\nExample: StableDiffusion XL",
    helpQuery:
      "Impact: Narrows search results to models matching keywords; empty shows popular models.\nExample: i2vgen",
    helpLimit:
      "Impact: Controls number of search results shown; larger values increase list size and fetch time.\nExample: 30",
    helpSearchSort:
      "Impact: Changes ranking of search results by metric/time.\nExample: downloads",
    helpSearchNsfw:
      "Impact: Includes or excludes NSFW models from provider search (provider support dependent).\nExample: exclude",
    helpSearchSizeMinMb:
      "Impact: Filters out models smaller than this file size (GB). Unknown-size models are excluded when size filter is active.\nExample: 1.0",
    helpSearchSizeMaxMb:
      "Impact: Filters out models larger than this file size (GB). Unknown-size models are excluded when size filter is active.\nExample: 12.0",
    helpSearchModelKind:
      "Impact: Filters model type such as checkpoint/LoRA/VAE when provider supports it.\nExample: checkpoint",
    helpSearchViewMode:
      "Impact: Changes only the card layout (grid/list), not search results.\nExample: grid",
    helpLocalModelsPath:
      "Impact: Changes which folder is scanned and shown in the Local Models screen.\nExample: D:\\ModelStore\\VideoGen",
    helpLocalViewMode:
      "Impact: Switches local model presentation between hierarchical tree and flat list.\nExample: Tree",
    helpLocalTreeSearch:
      "Impact: Filters model items in the tree by partial name match.\nExample: sdxl",
    helpLocalLineage:
      "Impact: Filters local model list by model family lineage.\nExample: StableDiffusion XL",
    helpModelsDirectory:
      "Impact: Changes where downloaded models are stored and where local model scanning occurs.\nExample: C:\\AI\\VideoGen\\models",
    helpListenPort:
      "Impact: Changes server listening port used at next startup; restart is required after save.\nExample: 8000",
    helpRocmAotriton:
      "Impact: Enables/disables ROCm experimental AOTriton SDPA/Flash attention path at next startup. ON can improve step speed, OFF may improve stability on some environments.\nExample: 1=enabled, 0=disabled",
    helpOutputsDirectory:
      "Impact: Changes where generated video files are written.\nExample: D:\\VideoOutputs",
    helpTempDirectory:
      "Impact: Changes where temporary upload/intermediate files are stored during processing.\nExample: C:\\AI\\VideoGen\\tmp",
    helpLogLevel:
      "Impact: Controls log verbosity for troubleshooting. DEBUG outputs detailed internal steps and may increase log size.\nExample: DEBUG",
    helpHfToken:
      "Impact: Enables access to private/gated Hugging Face models and can improve API limits.\nExample: hf_xxxxxxxxxxxxxxxxxxxxx",
    helpT2VBackend:
      "Impact: Selects execution backend for this Text-to-Video request. auto uses settings/default routing; NPU requires a configured runner.\nExample: npu",
    helpDefaultT2VBackend:
      "Impact: Sets default Text-to-Video backend when form backend is auto.\nExample: auto",
    helpT2VNpuRunner:
      "Impact: Sets executable/script path for NPU Text-to-Video runner. The runner must accept '--input-json <path>' and create output video.\nExample: C:\\AI\\NPU\\t2v_runner.bat",
    helpT2VNpuModelDir:
      "Impact: Optional model directory passed to NPU runner for ONNX/NPU models.\nExample: C:\\AI\\VideoGen\\models_npu\\text2video",
    logLevelDebug: "DEBUG",
    logLevelInfo: "INFO",
    logLevelWarning: "WARNING",
    logLevelError: "ERROR",
    helpDefaultTextModel:
      "Impact: Sets the model preselected/used for text-to-video when no explicit model is selected.\nExample: damo-vilab/text-to-video-ms-1.7b",
    helpDefaultImageModel:
      "Impact: Sets the model preselected/used for image-to-video when no explicit model is selected.\nExample: ali-vilab/i2vgen-xl",
    helpDefaultTextImageModel:
      "Impact: Sets the model preselected/used for text-to-image when no explicit model is selected.\nExample: runwayml/stable-diffusion-v1-5",
    helpDefaultImageImageModel:
      "Impact: Sets the model preselected/used for image-to-image when no explicit model is selected.\nExample: runwayml/stable-diffusion-v1-5",
    helpLoraScale:
      "Impact: Controls LoRA effect strength; larger values emphasize LoRA style/features more. This value is applied to all selected LoRAs.\nExample: 1.0",
    helpDefaultSteps:
      "Impact: Higher values improve quality but increase generation time and GPU load.\nExample: 30",
    helpDefaultFrames:
      "Impact: Sets default output duration in seconds for video generation; longer duration increases processing time and memory usage.\nExample: 2.0",
    helpDefaultGuidance:
      "Impact: Higher guidance follows prompt more strongly but may reduce natural motion.\nExample: 9.0",
    helpDefaultFps:
      "Impact: Controls playback speed and smoothness of output video.\nExample: 8",
    helpDefaultWidth:
      "Impact: Higher resolution improves detail but significantly increases VRAM usage.\nExample: 512",
    helpDefaultHeight:
      "Impact: Higher resolution improves detail but significantly increases VRAM usage.\nExample: 512",
    helpClearHfCache:
      "Impact: Deletes downloaded Hugging Face cache files to reclaim disk space; next model load/download may take longer.\nExample: Run this after low disk warning",
    btnSaveSettings: "Save Settings",
    btnRunCleanup: "Run Cleanup",
    labelCleanupNow: "Cleanup Runtime Storage",
    labelCleanupIncludeCache: "Include HF Cache",
    labelCleanupMaxAgeDays: "Cleanup Max Age (days)",
    labelCleanupMaxOutputsCount: "Cleanup Max Outputs",
    labelCleanupMaxTmpCount: "Cleanup Max Tmp",
    labelCleanupMaxCacheGb: "Cleanup Max Cache (GB)",
    btnClearHfCache: "Clear Cache",
    btnGenerateTextImage: "Generate Text Image",
    headingPathBrowser: "Folder Browser",
    btnClosePathBrowser: "Close",
    labelCurrentPath: "Current Path",
    btnRoots: "Roots",
    btnUpFolder: "Up",
    btnUseThisPath: "Use This Path",
    statusNoTask: "No task running.",
    placeholderT2IPrompt: "A portrait photo of a fox wearing a suit, studio light",
    placeholderI2IPrompt: "Refine this image into a cinematic poster style",
    placeholderT2VPrompt: "A cinematic drone shot above neon city...",
    placeholderNegativePrompt: "low quality, blurry",
    placeholderI2VPrompt: "Turn this image into a smooth cinematic motion...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderI2INegativePrompt: "blurry, low quality",
    placeholderSeed: "random if empty",
    placeholderDownloadSavePath: "empty = use Models Directory from Settings",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderLocalModelsPath: "empty = use Models Directory from Settings",
    placeholderLocalTreeSearch: "filter model name...",
    placeholderOptional: "optional",
    searchSourceAll: "All",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "All base models",
    localLineageAll: "All lineages",
    msgSettingsSaved: "Settings saved.",
    msgNoLocalModels: "No local models in: {path}",
    msgNoLocalTreeModels: "No tree items in: {path}",
    msgSelectLocalTreeItem: "Select a tree model item.",
    msgTreeFilterHint: "Filter: {query}",
    msgTreeFilterClearHint: "Filter: (none)",
    msgApplyUnsupportedCategory: "This model/category cannot be applied to the selected task.",
    msgReveal: "Reveal",
    msgLocalModelRevealed: "Opened in Explorer: {path}",
    msgLocalModelRevealFailed: "Reveal failed: {error}",
    msgLocalRescanned: "Local model tree rescanned.",
    msgLocalRescanFailed: "Local model tree rescan failed: {error}",
    msgNoOutputs: "No outputs in: {path}",
    msgPortChangeSaved: "Listen port saved. Restart `start.bat` to apply new port.",
    msgServerSettingRestartRequired: "Server setting saved. Restart `start.bat` to apply changes.",
    msgNoModelsFound: "No models found.",
    msgModelSearchLoading: "Searching models...",
    msgModelDetailEmpty: "Select a model to view detail.",
    msgModelDetailLoading: "Loading model detail...",
    msgModelDetailLoadFailed: "Model detail failed: {error}",
    downloadsLabel: "Downloads",
    msgNoDownloads: "No download tasks.",
    msgDownloadsRefreshFailed: "Failed to refresh downloads: {error}",
    msgDownloadPathSynced: "Local Models Path was switched to download destination: {path}",
    msgDownloadListSummary: "{running}/{total}",
    msgModelInstalled: "Downloaded",
    msgModelNotInstalled: "Not downloaded",
    msgSearchPage: "Page {page}",
    msgApply: "Apply",
    msgDetail: "Detail",
    msgDetailDescription: "Description",
    msgDetailTags: "Tags",
    msgDetailVersions: "Version",
    msgDetailFiles: "File",
    msgDetailRevision: "Revision",
    msgSearchModelApplied: "Model set for {task}: {model}",
    msgDefaultModelOption: "Use default model ({model})",
    msgDefaultModelNoMeta: "Use default model",
    msgNoModelCatalog: "No models available.",
    msgNoLoraCatalog: "No LoRAs available for this model.",
    msgNoLoraOption: "No LoRA",
    msgNoVaeOption: "No VAE",
    msgSearchLineage: "Search lineage",
    msgSetTaskModel: "Set {task}",
    msgLocalModelApplied: "Local model set for {task}: {model}",
    msgLineageSearchStarted: "Searching lineage from base model: {base}",
    msgModelSelectHint: "Select a model to see thumbnail.",
    msgModelNoPreview: "No thumbnail available for this model.",
    msgNoFolders: "No subfolders found.",
    msgOpen: "Open",
    btnDownload: "Download",
    btnPrev: "Prev",
    btnNext: "Next",
    btnDeleteModel: "Delete",
    msgAlreadyDownloaded: "Downloaded",
    msgModelPreviewAlt: "Model preview",
    msgModelDownloadStarted: "Model download started: {repo} -> {path}",
    msgTextImageGenerationStarted: "Text-to-image generation started: {id}",
    msgImageImageGenerationStarted: "Image-to-image generation started: {id}",
    msgTextGenerationStarted: "Text generation started: {id}",
    msgImageGenerationStarted: "Image generation started: {id}",
    msgTaskPollFailed: "Task poll failed: {error}",
    msgTaskErrorPopup: "Task failed ({type}). Reason: {error}",
    msgConfirmDeleteModel: "Delete local model '{model}'?",
    msgModelDeleted: "Model deleted: {model}",
    msgModelDeleteFailed: "Model delete failed: {error}",
    msgConfirmDeleteOutput: "Delete output '{name}'?",
    msgOutputDeleted: "Output deleted: {name}",
    msgOutputDeleteFailed: "Output delete failed: {error}",
    msgOutputsRefreshFailed: "Outputs refresh failed: {error}",
    msgConfirmClearHfCache: "Delete Hugging Face cache now? Cached files will be re-downloaded later.",
    msgHfCacheCleared: "Hugging Face cache cleared. removed={removed}, skipped={skipped}, failed={failed}",
    msgHfCacheClearFailed: "Hugging Face cache clear failed: {error}",
    msgSaveSettingsFailed: "Save settings failed: {error}",
    msgSearchFailed: "Search failed: {error}",
    msgSearchSizeRangeInvalid: "Invalid size filter range (GB): max must be greater than or equal to min.",
    msgTextGenerationFailed: "Text generation failed: {error}",
    msgImageGenerationFailed: "Image generation failed: {error}",
    msgLocalModelRefreshFailed: "Local model refresh failed: {error}",
    msgPathBrowserLoadFailed: "Folder browser load failed: {error}",
    msgInitFailed: "Initialization failed: {error}",
    msgInputImageRequired: "Input image is required.",
    msgDefaultModelsDir: "default models dir",
    msgSelectLocalModel: "Select local model",
    msgDefaultModelNotLocal: "(not local) {model}",
    msgUnknownPath: "(unknown)",
    modelTag: "tag",
    outputUpdated: "updated",
    outputTypeImage: "image",
    outputTypeVideo: "video",
    outputTypeOther: "other",
    modelKind: "kind",
    modelKindBase: "Base",
    modelKindLora: "LoRA",
    modelKindVae: "VAE",
    modelBase: "base",
    modelDownloads: "downloads",
    modelLikes: "likes",
    modelSize: "size",
    modelSource: "source",
    taskTypeText2Image: "text2image",
    taskTypeImage2Image: "image2image",
    taskTypeText2Video: "text2video",
    taskTypeImage2Video: "image2video",
    taskTypeDownload: "download",
    taskTypeUnknown: "unknown",
    statusQueued: "queued",
    statusRunning: "running",
    statusCompleted: "completed",
    statusError: "error",
    labelTaskLog: "Task Log",
    labelTaskStep: "Step",
    taskStepUnknown: "Step: idle",
    stepQueued: "queued",
    stepRuntimeDiagnostics: "environment diagnostics",
    stepModelLoad: "loading model",
    stepModelLoadGpu: "loading to VRAM",
    stepModelLoadAutoMap: "loading with auto device_map",
    stepModelLoadCpuOffload: "enabling CPU offload",
    stepModelLoadCpu: "loading with low CPU memory mode",
    stepLoraApply: "applying LoRA",
    stepPrepare: "preparing",
    stepInference: "inferencing",
    stepGenerating: "generating",
    stepExtractingOutput: "extracting output",
    stepDecodingFrames: "decoding frames",
    stepPostprocessing: "post-processing",
    stepEncodingVideo: "encoding video",
    stepSaving: "saving",
    stepDecode: "decoding",
    stepEncode: "encoding",
    stepSave: "saving",
    stepMemoryCleanup: "releasing memory",
    stepDone: "done",
    stepFailed: "failed",
    stepCancelled: "cancelled",
    modelSupportLabel: "support",
    modelSupportReady: "ready",
    modelSupportLimited: "limited",
    modelSupportRequiresPatch: "requires patch",
    modelSupportNotSupported: "not supported",
    msgVideoModelTaskUnsupported: "Model '{model}' does not support task '{task}'.",
    msgVideoModelRuntimeUnsupported: "Model '{model}' cannot run on current runtime: {reason}",
    statusCancelled: "cancelled",
    btnCancelTask: "Cancel Task",
    msgTaskCancelRequested: "Cancellation requested for task={id}",
    msgCleanupDone: "Cleanup completed: outputs={outputs}, tmp={tmp}, cache_paths={cache}",
    msgCleanupFailed: "Cleanup failed: {error}",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeNpu: "npu",
    runtimeNpuReason: "npu_reason",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    backendAuto: "Auto",
    backendCuda: "GPU (CUDA/ROCm)",
    backendNpu: "NPU",
    runtimeLoadFailed: "runtime load failed: {error}",
    serverQueued: "Queued",
    serverGenerationQueued: "Generation queued",
    serverDownloadQueued: "Download queued",
    serverLoadingModel: "Loading model",
    serverLoadingModelVram: "Loading model to VRAM",
    serverLoadingModelAutoMap: "Loading with auto device_map",
    serverLoadingModelCpuOffload: "Enabling CPU offload",
    serverLoadingModelCpuLowMem: "Loading with low CPU memory mode",
    serverLoadingLora: "Applying LoRA",
    serverPreparingImage: "Preparing image",
    serverGeneratingImage: "Generating image",
    serverGeneratingFrames: "Generating frames",
    serverDecodingLatents: "Decoding latents",
    serverDecodingLatentsCpuFallback: "Decoding latents",
    serverPostprocessingImage: "Postprocessing image",
    serverEncoding: "Encoding mp4",
    serverMemoryCleanup: "Releasing memory",
    serverSavingPng: "Saving png",
    serverDone: "Done",
    serverGenerationFailed: "Generation failed",
    serverDownloadComplete: "Download complete",
    serverDownloadFailed: "Download failed",
  },
  ja: {
    appTitle: "ROCm VideoGen Studio",
    appSubtitle: "テキスト画像生成 / 画像画像生成 / テキスト動画生成 / 画像動画生成",
    languageLabel: "言語",
    runtimeLoading: "実行環境: 読み込み中...",
    tabTextToImage: "テキスト→画像",
    tabImageToImage: "画像→画像",
    tabTextToVideo: "テキスト→動画",
    tabImageToVideo: "画像→動画",
    tabModels: "モデル検索",
    tabLocalModels: "ローカルモデル",
    tabOutputs: "成果物",
    tabSettings: "設定",
    labelPrompt: "プロンプト",
    labelNegativePrompt: "ネガティブプロンプト",
    labelModelOptional: "モデルID（任意）",
    labelModelSelect: "モデル選択",
    labelSteps: "ステップ数",
    labelFrames: "フレーム数",
    labelDurationSeconds: "長さ（秒）",
    labelGuidance: "ガイダンス",
    labelFps: "FPS",
    labelT2VBackend: "T2Vバックエンド",
    labelSeed: "シード",
    labelInputImage: "入力画像",
    labelWidth: "幅",
    labelHeight: "高さ",
    labelLoraSelect: "LoRA選択",
    labelVaeSelect: "VAE選択",
    labelLoraScale: "LoRA強度",
    labelStrength: "変換強度",
    btnGenerateTextVideo: "テキスト動画を生成",
    btnGenerateImageVideo: "画像動画を生成",
    btnGenerateImageImage: "画像画像を生成",
    btnRefreshModels: "更新",
    btnRefreshLoras: "LoRA更新",
    btnRefreshVaes: "VAE更新",
    labelDownloadSavePathOptional: "モデル保存先（任意）",
    btnBrowsePath: "参照",
    labelTask: "タスク",
    labelSearchSource: "検索元",
    labelSearchBaseModel: "ベースモデル",
    labelSearchSort: "並び替え",
    labelSearchNsfw: "NSFW",
    labelSearchSizeMinMb: "最小サイズ(GB)",
    labelSearchSizeMaxMb: "最大サイズ(GB)",
    labelSearchModelKind: "モデル種別",
    labelSearchViewMode: "表示",
    labelQuery: "検索語",
    labelLimit: "件数",
    btnSearchModels: "モデル検索",
    headingLocalModels: "ローカルモデル",
    headingOutputs: "成果物",
    btnRefreshLocalList: "ローカル一覧を更新",
    btnRescanLocalModels: "再走査",
    btnExpandAllLocalTree: "Treeをすべて展開",
    btnRefreshOutputs: "成果物一覧を更新",
    labelLocalModelsPath: "ローカルモデル表示パス",
    labelLocalViewMode: "表示モード",
    labelLocalTreeSearch: "ツリー検索",
    localViewTree: "ツリー",
    localViewFlat: "フラット",
    labelLocalLineage: "系譜",
    labelModelsDirectory: "モデル保存ディレクトリ",
    labelListenPort: "リッスンポート",
    labelRocmAotriton: "ROCm AOTriton 実験機能",
    labelOutputsDirectory: "出力ディレクトリ",
    labelTempDirectory: "一時ディレクトリ",
    labelLogLevel: "ログレベル",
    labelHfToken: "HFトークン",
    labelDefaultT2VBackend: "既定T2Vバックエンド",
    labelT2VNpuRunner: "T2V NPUランナー",
    labelT2VNpuModelDir: "T2V NPUモデルディレクトリ",
    labelVramDirectThresholdGb: "VRAM直ロードしきい値(GB)",
    labelEnableDeviceMapAuto: "device_map='auto' を有効化",
    labelEnableModelCpuOffload: "CPUオフロードフォールバックを有効化",
    labelDefaultTextModel: "既定: テキスト→動画",
    labelDefaultImageModel: "既定: 画像→動画",
    labelDefaultTextImageModel: "既定: テキスト→画像",
    labelDefaultImageImageModel: "既定: 画像→画像",
    labelDefaultSteps: "既定ステップ数",
    labelDefaultFrames: "既定動画時間(秒)",
    labelDefaultGuidance: "既定ガイダンス",
    labelDefaultFps: "既定FPS",
    labelDefaultWidth: "既定幅",
    labelDefaultHeight: "既定高さ",
    labelClearHfCache: "Hugging Face キャッシュ削除",
    helpLanguage:
      "影響: 画面の表示言語が変わります。生成ロジック自体には影響しません。\n例: 日本語",
    helpPromptText2Video:
      "影響: テキスト→動画の内容・動き・雰囲気を決める主指示です。\n例: 夜のネオン都市を俯瞰するシネマティックなドローン映像",
    helpPromptText2Image:
      "影響: テキスト→画像の構図・被写体・画風を決める主指示です。\n例: スーツを着たキツネの写実的なポートレート",
    helpNegativePromptText2Image:
      "影響: 画像で避けたい要素や品質低下要因を抑制します。\n例: blurry, low quality, watermark",
    helpLoraSelectText2Image:
      "影響: このモデルにLoRAを適用し、画風や被写体特性を調整できます。複数選択に対応します。\n例: your-org/anime-style-lora",
    helpVaeSelectText2Image:
      "影響: テキスト→画像のVAEを差し替え、発色や復元品質に影響します。\n例: stabilityai/sd-vae-ft-mse",
    helpModelSelectText2Image:
      "影響: 使用モデルにより画風・品質・速度・VRAM使用量が変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpStepsText2Image:
      "影響: 値を上げると細部品質向上が見込めますが、時間が増えます。\n例: 30",
    helpGuidanceText2Image:
      "影響: 値を上げると指示忠実度が上がりますが、自然な揺らぎが減る場合があります。\n例: 7.5",
    helpWidthText2Image:
      "影響: 値を上げると解像度/細部が向上しますがVRAM使用量が増えます。\n例: 512",
    helpHeightText2Image:
      "影響: 値を上げると解像度/細部が向上しますがVRAM使用量が増えます。\n例: 512",
    helpSeedText2Image:
      "影響: 乱数を固定して再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpPromptImage2Image:
      "影響: 入力画像をどの方向に変換するかを指定します。\n例: この写真を映画ポスター風に変換",
    helpNegativePromptImage2Image:
      "影響: 画像変換で不要な要素や劣化を抑制します。\n例: blurry, distorted, watermark",
    helpModelSelectImage2Image:
      "影響: 画像→画像モデルにより画風・品質・速度・VRAM使用量が変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpLoraSelectImage2Image:
      "影響: 画像→画像にLoRAを適用してスタイルや特徴を強調します。複数選択に対応します。\n例: your-org/lineart-lora",
    helpVaeSelectImage2Image:
      "影響: 画像→画像のVAEを差し替え、発色や復元品質に影響します。\n例: stabilityai/sd-vae-ft-mse",
    helpStepsImage2Image:
      "影響: 値を上げると細部品質が向上しやすい一方、処理時間が増えます。\n例: 30",
    helpGuidanceImage2Image:
      "影響: 値を上げるとプロンプト忠実度が上がります。\n例: 7.5",
    helpStrengthImage2Image:
      "影響: 入力画像からどれだけ変化させるかを調整します。\n例: 0.8",
    helpWidthImage2Image:
      "影響: 値を上げると精細になりますがVRAM使用量が増えます。\n例: 512",
    helpHeightImage2Image:
      "影響: 値を上げると精細になりますがVRAM使用量が増えます。\n例: 512",
    helpSeedImage2Image:
      "影響: 乱数を固定し再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpNegativePromptText2Video:
      "影響: 出したくない要素や品質低下要因を抑制します。\n例: blurry, low quality, watermark",
    helpModelSelectText2Video:
      "影響: 使用モデルにより品質・速度・VRAM使用量・画風が変わります。\n例: damo-vilab/text-to-video-ms-1.7b",
    helpLoraSelectText2Video:
      "影響: 互換モデルの場合、LoRAで画風や特徴を追加できます。複数選択に対応します。\n例: your-org/cinematic-look-lora",
    helpStepsText2Video:
      "影響: 値を上げると品質向上が見込めますが、時間とGPU負荷が増えます。\n例: 30",
    helpFramesText2Video:
      "影響: 値を上げると動画が長く滑らかになりますが、VRAMと処理時間が増えます。\n例: 16",
    helpDurationSeconds:
      "生成する動画の長さ（秒）。長くすると生成時間とVRAM使用量が増えます。\n例: 2.0",
    helpGuidanceText2Video:
      "影響: 値を上げると指示への忠実度が上がりますが、不自然になる場合があります。\n例: 9.0",
    helpFpsText2Video:
      "影響: 出力動画の再生速度・滑らかさに影響します。\n例: 8",
    helpSeedText2Video:
      "影響: 乱数を固定し、同条件で再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpInputImage:
      "影響: 画像→動画の構図や見た目の基準画像になります。\n例: C:\\Images\\input.png",
    helpPromptImage2Video:
      "影響: 入力画像に対する動きや演出の指示を与えます。\n例: 滑らかなカメラパンと弱いパララックス",
    helpNegativePromptImage2Video:
      "影響: ちらつきやノイズなど不要な要素を抑制します。\n例: flicker, artifact, noisy",
    helpModelSelectImage2Video:
      "影響: 使用モデルで動き品質・速度・VRAM使用量が変化します。\n例: ali-vilab/i2vgen-xl",
    helpLoraSelectImage2Video:
      "影響: 互換モデルの場合、LoRAで画風や特徴を追加できます。複数選択に対応します。\n例: your-org/motion-style-lora",
    helpStepsImage2Video:
      "影響: 値を上げると品質向上が見込めますが処理時間が増えます。\n例: 30",
    helpFramesImage2Video:
      "影響: 値を上げると動画長が伸びますがメモリ消費も増えます。\n例: 16",
    helpGuidanceImage2Video:
      "影響: 値を上げると指示忠実度が上がりますが自然さが下がる場合があります。\n例: 9.0",
    helpFpsImage2Video:
      "影響: 出力動画の滑らかさ・再生速度に影響します。\n例: 8",
    helpWidthImage2Video:
      "影響: 解像度が上がり精細になりますがVRAM使用量が増えます。\n例: 512",
    helpHeightImage2Video:
      "影響: 解像度が上がり精細になりますがVRAM使用量が増えます。\n例: 512",
    helpSeedImage2Video:
      "影響: 乱数固定により再現しやすくなります。空欄は毎回ランダムです。\n例: 12345",
    helpDownloadSavePath:
      "影響: この操作でダウンロードするモデルの保存先が変わります。\n例: D:\\ModelStore\\VideoGen",
    helpTask:
      "影響: モデル検索対象を生成タイプで絞り込みます。\n例: text-to-video",
    helpSearchSource:
      "影響: モデル検索の参照先を選びます。CivitAIは画像系タスクで利用できます。\n例: all",
    helpSearchBaseModel:
      "影響: ベースモデル系譜で検索結果を絞り込み、互換候補を見つけやすくします。\n例: StableDiffusion XL",
    helpQuery:
      "影響: キーワード一致で検索結果を絞ります。空欄では人気モデルを表示します。\n例: i2vgen",
    helpLimit:
      "影響: 表示件数を調整します。値が大きいほど表示数が増え、取得時間も増える場合があります。\n例: 30",
    helpSearchSort:
      "影響: 検索結果の並び順（DL数・いいね・更新時刻など）を変更します。\n例: downloads",
    helpSearchNsfw:
      "影響: NSFWモデルを含める/除外する指定です（提供元対応に依存）。\n例: exclude",
    helpSearchSizeMinMb:
      "影響: 指定したサイズ(GB)未満のモデルを除外します。サイズ不明のモデルはサイズフィルタ有効時に除外されます。\n例: 1.0",
    helpSearchSizeMaxMb:
      "影響: 指定したサイズ(GB)より大きいモデルを除外します。サイズ不明のモデルはサイズフィルタ有効時に除外されます。\n例: 12.0",
    helpSearchModelKind:
      "影響: checkpoint/LoRA/VAEなどモデル種別で絞り込みます（提供元対応に依存）。\n例: checkpoint",
    helpSearchViewMode:
      "影響: 結果の表示レイアウトのみを切り替えます（検索結果自体は変わりません）。\n例: grid",
    helpLocalModelsPath:
      "影響: ローカルモデル画面で一覧表示するフォルダを切り替えます。\n例: D:\\ModelStore\\VideoGen",
    helpLocalViewMode:
      "影響: ローカルモデル表示を階層ツリー/フラット一覧で切り替えます。\n例: ツリー",
    helpLocalTreeSearch:
      "影響: ツリー内のモデル名を部分一致で絞り込みます。\n例: sdxl",
    helpLocalLineage:
      "影響: ローカルモデル一覧をモデル系譜で絞り込みます。\n例: StableDiffusion XL",
    helpModelsDirectory:
      "影響: ダウンロードモデルの保存先とローカルモデル検索先が変わります。\n例: C:\\AI\\VideoGen\\models",
    helpListenPort:
      "影響: 次回起動時のサーバー待受ポートが変わります。保存後に再起動が必要です。\n例: 8000",
    helpRocmAotriton:
      "影響: 次回起動時の ROCm 実験的 AOTriton（SDPA/Flash attention）を有効/無効化します。ONでSTEP速度が上がる場合があり、OFFで安定する場合があります。\n例: 1=有効, 0=無効",
    helpOutputsDirectory:
      "影響: 生成した動画ファイルの出力先が変わります。\n例: D:\\VideoOutputs",
    helpTempDirectory:
      "影響: アップロード画像や中間ファイルの一時保存先が変わります。\n例: C:\\AI\\VideoGen\\tmp",
    helpLogLevel:
      "影響: トラブル調査向けのログ詳細度を変更します。DEBUGは内部状態を詳細出力し、ログ量が増えます。\n例: DEBUG",
    helpHfToken:
      "影響: 非公開/制限付き Hugging Face モデルへのアクセスやAPI利用制限に影響します。\n例: hf_xxxxxxxxxxxxxxxxxxxxx",
    helpT2VBackend:
      "影響: このテキスト→動画リクエストの実行バックエンドを選択します。autoは設定値/自動判定に従います。NPUはランナー設定が必要です。\n例: npu",
    helpDefaultT2VBackend:
      "影響: テキスト→動画フォームでバックエンドがauto時に使う既定値を設定します。\n例: auto",
    helpT2VNpuRunner:
      "影響: NPU用テキスト→動画ランナーの実行ファイル/スクリプトパスを設定します。ランナーは '--input-json <path>' を受け取り出力動画を作成する必要があります。\n例: C:\\AI\\NPU\\t2v_runner.bat",
    helpT2VNpuModelDir:
      "影響: NPUランナー向けのモデルディレクトリを任意指定します（ONNX等）。未指定時は選択モデルの解決先を使用します。\n例: C:\\AI\\VideoGen\\models_npu\\text2video",
    logLevelDebug: "DEBUG",
    logLevelInfo: "INFO",
    logLevelWarning: "WARNING",
    logLevelError: "ERROR",
    helpDefaultTextModel:
      "影響: テキスト→動画でモデル未指定時に使われる既定モデルが変わります。\n例: damo-vilab/text-to-video-ms-1.7b",
    helpDefaultImageModel:
      "影響: 画像→動画でモデル未指定時に使われる既定モデルが変わります。\n例: ali-vilab/i2vgen-xl",
    helpDefaultTextImageModel:
      "影響: テキスト→画像でモデル未指定時に使われる既定モデルが変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpDefaultImageImageModel:
      "影響: 画像→画像でモデル未指定時に使われる既定モデルが変わります。\n例: runwayml/stable-diffusion-v1-5",
    helpLoraScale:
      "影響: LoRAの効き具合を調整します。大きいほどLoRAの特徴が強く反映されます。複数選択時は全LoRAに同じ値を適用します。\n例: 1.0",
    helpDefaultSteps:
      "影響: 値を上げると品質向上が見込めますが、生成時間とGPU負荷が増えます。\n例: 30",
    helpDefaultFrames:
      "影響: 動画生成時の既定の長さ(秒)を設定します。長くするほど処理時間とメモリ使用量が増えます。\n例: 2.0",
    helpDefaultGuidance:
      "影響: 値を上げるとプロンプト忠実度が上がりますが、不自然な動きになる場合があります。\n例: 9.0",
    helpDefaultFps:
      "影響: 出力動画の再生速度・滑らかさに影響します。\n例: 8",
    helpDefaultWidth:
      "影響: 解像度が上がり精細になりますが、VRAM使用量が増えます。\n例: 512",
    helpDefaultHeight:
      "影響: 解像度が上がり精細になりますが、VRAM使用量が増えます。\n例: 512",
    helpClearHfCache:
      "影響: Hugging Face のキャッシュファイルを削除してディスク容量を回復します。次回は再ダウンロードが発生し時間がかかります。\n例: 容量不足警告が出た後に実行",
    btnSaveSettings: "設定を保存",
    btnRunCleanup: "クリーンアップ実行",
    labelCleanupNow: "ランタイムストレージのクリーンアップ",
    labelCleanupIncludeCache: "HFキャッシュも対象にする",
    labelCleanupMaxAgeDays: "クリーンアップ保持日数",
    labelCleanupMaxOutputsCount: "クリーンアップ出力上限",
    labelCleanupMaxTmpCount: "クリーンアップ一時ファイル上限",
    labelCleanupMaxCacheGb: "クリーンアップキャッシュ上限(GB)",
    btnClearHfCache: "キャッシュ削除",
    btnGenerateTextImage: "テキスト画像を生成",
    headingPathBrowser: "フォルダブラウザ",
    btnClosePathBrowser: "閉じる",
    labelCurrentPath: "現在のパス",
    btnRoots: "ルート",
    btnUpFolder: "上へ",
    btnUseThisPath: "このパスを使用",
    statusNoTask: "実行中のタスクはありません。",
    placeholderT2IPrompt: "スーツを着たキツネのスタジオポートレート...",
    placeholderI2IPrompt: "この画像を映画ポスター風に変換する...",
    placeholderT2VPrompt: "ネオン都市上空を飛ぶシネマティックなドローン映像...",
    placeholderNegativePrompt: "低品質, ぼやけ",
    placeholderI2VPrompt: "この画像に滑らかな映画的モーションを付ける...",
    placeholderI2VNegativePrompt: "artifact, flicker",
    placeholderI2INegativePrompt: "blur, low quality",
    placeholderSeed: "空欄でランダム",
    placeholderDownloadSavePath: "空欄 = 設定のモデル保存先を使用",
    placeholderSearchQuery: "i2vgen, text-to-video...",
    placeholderLocalModelsPath: "空欄 = 設定のモデル保存先を使用",
    placeholderLocalTreeSearch: "モデル名でフィルタ...",
    placeholderOptional: "任意",
    searchSourceAll: "すべて",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "すべてのベースモデル",
    localLineageAll: "すべての系譜",
    msgSettingsSaved: "設定を保存しました。",
    msgNoLocalModels: "ローカルモデルがありません: {path}",
    msgNoLocalTreeModels: "ツリー表示できるモデルがありません: {path}",
    msgSelectLocalTreeItem: "ツリーのモデル項目を選択してください。",
    msgTreeFilterHint: "フィルタ: {query}",
    msgTreeFilterClearHint: "フィルタ: (なし)",
    msgApplyUnsupportedCategory: "選択中タスクではこのモデル種別を適用できません。",
    msgReveal: "場所を開く",
    msgLocalModelRevealed: "Explorerで開きました: {path}",
    msgLocalModelRevealFailed: "場所を開く処理に失敗: {error}",
    msgLocalRescanned: "ローカルモデルツリーを再走査しました。",
    msgLocalRescanFailed: "ローカルモデルツリーの再走査に失敗: {error}",
    msgNoOutputs: "成果物がありません: {path}",
    msgPortChangeSaved: "リッスンポートを保存しました。反映には `start.bat` の再起動が必要です。",
    msgServerSettingRestartRequired: "サーバー設定を保存しました。反映には `start.bat` の再起動が必要です。",
    msgNoModelsFound: "モデルが見つかりません。",
    msgModelSearchLoading: "モデルを検索中...",
    msgModelDetailEmpty: "モデルを選択すると詳細を表示します。",
    msgModelDetailLoading: "モデル詳細を読み込み中...",
    msgModelDetailLoadFailed: "モデル詳細の取得に失敗: {error}",
    downloadsLabel: "ダウンロード",
    msgNoDownloads: "ダウンロードタスクはありません。",
    msgDownloadsRefreshFailed: "ダウンロード一覧の更新に失敗: {error}",
    msgDownloadPathSynced: "ローカルモデル表示先をダウンロード保存先に合わせました: {path}",
    msgDownloadListSummary: "{running}/{total}",
    msgModelInstalled: "ダウンロード済み",
    msgModelNotInstalled: "未ダウンロード",
    msgSearchPage: "ページ {page}",
    msgApply: "適用",
    msgDetail: "詳細",
    msgDetailDescription: "説明",
    msgDetailTags: "タグ",
    msgDetailVersions: "バージョン",
    msgDetailFiles: "ファイル",
    msgDetailRevision: "リビジョン",
    msgSearchModelApplied: "{task} にモデルを設定: {model}",
    msgDefaultModelOption: "既定モデルを使用 ({model})",
    msgDefaultModelNoMeta: "既定モデルを使用",
    msgNoModelCatalog: "利用可能なモデルがありません。",
    msgNoLoraCatalog: "このモデルで利用可能なLoRAがありません。",
    msgNoLoraOption: "LoRAなし",
    msgNoVaeOption: "VAEなし",
    msgSearchLineage: "系譜検索",
    msgSetTaskModel: "{task}に設定",
    msgLocalModelApplied: "{task} にローカルモデルを設定: {model}",
    msgLineageSearchStarted: "ベースモデルから系譜を検索: {base}",
    msgModelSelectHint: "モデルを選択するとサムネイルを表示します。",
    msgModelNoPreview: "このモデルにはサムネイルがありません。",
    msgNoFolders: "サブフォルダがありません。",
    msgOpen: "開く",
    btnDownload: "ダウンロード",
    btnPrev: "前へ",
    btnNext: "次へ",
    btnDeleteModel: "削除",
    msgAlreadyDownloaded: "ダウンロード済み",
    msgModelPreviewAlt: "モデルプレビュー",
    msgModelDownloadStarted: "モデルダウンロード開始: {repo} -> {path}",
    msgTextImageGenerationStarted: "テキスト→画像生成を開始しました: {id}",
    msgImageImageGenerationStarted: "画像→画像生成を開始しました: {id}",
    msgTextGenerationStarted: "テキスト生成を開始しました: {id}",
    msgImageGenerationStarted: "画像生成を開始しました: {id}",
    msgTaskPollFailed: "タスク確認に失敗: {error}",
    msgTaskErrorPopup: "タスクが失敗しました（{type}）。理由: {error}",
    msgConfirmDeleteModel: "ローカルモデル '{model}' を削除しますか？",
    msgModelDeleted: "モデルを削除しました: {model}",
    msgModelDeleteFailed: "モデル削除に失敗: {error}",
    msgConfirmDeleteOutput: "成果物 '{name}' を削除しますか？",
    msgOutputDeleted: "成果物を削除しました: {name}",
    msgOutputDeleteFailed: "成果物の削除に失敗: {error}",
    msgOutputsRefreshFailed: "成果物一覧の更新に失敗: {error}",
    msgConfirmClearHfCache: "Hugging Face キャッシュを削除しますか？削除後は必要なファイルが再ダウンロードされます。",
    msgHfCacheCleared: "Hugging Face キャッシュを削除しました。削除={removed}, スキップ={skipped}, 失敗={failed}",
    msgHfCacheClearFailed: "Hugging Face キャッシュ削除に失敗: {error}",
    msgSaveSettingsFailed: "設定保存に失敗: {error}",
    msgSearchFailed: "検索に失敗: {error}",
    msgSearchSizeRangeInvalid: "サイズ範囲(GB)が不正です。最大値は最小値以上にしてください。",
    msgTextGenerationFailed: "テキスト生成に失敗: {error}",
    msgImageGenerationFailed: "画像生成に失敗: {error}",
    msgLocalModelRefreshFailed: "ローカル一覧更新に失敗: {error}",
    msgPathBrowserLoadFailed: "フォルダブラウザの読み込み失敗: {error}",
    msgInitFailed: "初期化に失敗: {error}",
    msgInputImageRequired: "入力画像が必要です。",
    msgDefaultModelsDir: "既定のモデル保存先",
    msgSelectLocalModel: "ローカルモデルを選択",
    msgDefaultModelNotLocal: "（ローカル未配置）{model}",
    msgUnknownPath: "(不明)",
    modelTag: "タグ",
    outputUpdated: "更新日時",
    outputTypeImage: "画像",
    outputTypeVideo: "動画",
    outputTypeOther: "その他",
    modelKind: "種別",
    modelKindBase: "ベース",
    modelKindLora: "LoRA",
    modelKindVae: "VAE",
    modelBase: "ベースモデル",
    modelDownloads: "DL数",
    modelLikes: "いいね",
    modelSize: "サイズ",
    modelSource: "ソース",
    taskTypeText2Image: "テキスト画像",
    taskTypeImage2Image: "画像画像",
    taskTypeText2Video: "テキスト動画",
    taskTypeImage2Video: "画像動画",
    taskTypeDownload: "ダウンロード",
    taskTypeUnknown: "不明",
    statusQueued: "待機中",
    statusRunning: "実行中",
    statusCompleted: "完了",
    statusError: "エラー",
    labelTaskLog: "タスクログ",
    labelTaskStep: "ステップ",
    taskStepUnknown: "ステップ: 待機",
    stepQueued: "待機中",
    stepRuntimeDiagnostics: "環境診断中",
    stepModelLoad: "モデル読込中",
    stepModelLoadGpu: "VRAMへロード中",
    stepModelLoadAutoMap: "auto device_mapでロード中",
    stepModelLoadCpuOffload: "CPUオフロード有効化中",
    stepModelLoadCpu: "CPU低メモリモードでロード中",
    stepLoraApply: "LoRA適用中",
    stepPrepare: "準備中",
    stepInference: "推論中",
    stepGenerating: "生成中",
    stepExtractingOutput: "出力抽出中",
    stepDecodingFrames: "フレームデコード中",
    stepPostprocessing: "後処理中",
    stepEncodingVideo: "動画エンコード中",
    stepSaving: "保存中",
    stepDecode: "デコード中",
    stepEncode: "エンコード中",
    stepSave: "保存中",
    stepMemoryCleanup: "メモリ解放中",
    stepDone: "完了",
    stepFailed: "失敗",
    stepCancelled: "キャンセル",
    modelSupportLabel: "対応",
    modelSupportReady: "対応",
    modelSupportLimited: "制限あり",
    modelSupportRequiresPatch: "パッチ必要",
    modelSupportNotSupported: "未対応",
    msgVideoModelTaskUnsupported: "モデル '{model}' はタスク '{task}' に対応していません。",
    msgVideoModelRuntimeUnsupported: "モデル '{model}' は現在の実行環境で利用できません: {reason}",
    statusCancelled: "キャンセル",
    btnCancelTask: "タスクをキャンセル",
    msgTaskCancelRequested: "タスクのキャンセルを要求しました: {id}",
    msgCleanupDone: "クリーンアップ完了: outputs={outputs}, tmp={tmp}, cache_paths={cache}",
    msgCleanupFailed: "クリーンアップ失敗: {error}",
    taskLine: "task={id} | type={type} | status={status} | {progress}% | {message}",
    taskError: "error={error}",
    runtimeDevice: "device",
    runtimeCuda: "cuda",
    runtimeRocm: "rocm",
    runtimeNpu: "npu",
    runtimeNpuReason: "npu_reason",
    runtimeDiffusers: "diffusers",
    runtimeTorch: "torch",
    runtimeError: "error",
    backendAuto: "自動",
    backendCuda: "GPU (CUDA/ROCm)",
    backendNpu: "NPU",
    runtimeLoadFailed: "実行環境の取得に失敗: {error}",
    serverQueued: "キュー待ち",
    serverGenerationQueued: "生成キューに追加",
    serverDownloadQueued: "ダウンロードキューに追加",
    serverLoadingModel: "モデルを読み込み中",
    serverLoadingModelVram: "VRAMにロード中",
    serverLoadingModelAutoMap: "自動device_mapでロード中",
    serverLoadingModelCpuOffload: "CPUオフロード有効化中",
    serverLoadingModelCpuLowMem: "CPU低メモリモードでロード中",
    serverLoadingLora: "LoRAを適用中",
    serverPreparingImage: "画像を準備中",
    serverGeneratingImage: "画像を生成中",
    serverGeneratingFrames: "フレームを生成中",
    serverDecodingLatents: "潜在表現をデコード中",
    serverDecodingLatentsCpuFallback: "潜在表現をデコード中",
    serverPostprocessingImage: "画像を後処理中",
    serverEncoding: "mp4に変換中",
    serverMemoryCleanup: "メモリ解放中",
    serverSavingPng: "pngを保存中",
    serverDone: "完了",
    serverGenerationFailed: "生成に失敗",
    serverDownloadComplete: "ダウンロード完了",
    serverDownloadFailed: "ダウンロード失敗",
  },
  es: {
    languageLabel: "Idioma",
    tabTextToVideo: "Texto a video",
    tabImageToVideo: "Imagen a video",
    tabModels: "Buscar modelos",
    tabLocalModels: "Modelos locales",
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
    tabModels: "Recherche modèles",
    tabLocalModels: "Modèles locaux",
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
    tabModels: "Modellsuche",
    tabLocalModels: "Lokale Modelle",
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
    tabModels: "Ricerca modelli",
    tabLocalModels: "Modelli locali",
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
    tabModels: "Buscar modelos",
    tabLocalModels: "Modelos locais",
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
    tabModels: "Поиск моделей",
    tabLocalModels: "Локальные модели",
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
    tabModels: "بحث النماذج",
    tabLocalModels: "النماذج المحلية",
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
  taskPollDelayMs: TASK_POLL_INTERVAL_MS,
  settings: null,
  localModels: [],
  localModelsBaseDir: "",
  localTreeData: null,
  localTreeFilter: "",
  localTreeSelected: null,
  localTreeOpenState: {},
  localViewMode: "tree",
  settingsLocalModels: [],
  lastSearchResults: [],
  language: DEFAULT_LANG,
  modelCatalog: {
    "text-to-image": [],
    "image-to-image": [],
    "text-to-video": [],
    "image-to-video": [],
  },
  defaultModels: {
    "text-to-image": "",
    "image-to-image": "",
    "text-to-video": "",
    "image-to-video": "",
  },
  loraCatalog: {
    "text-to-image": [],
    "image-to-image": [],
    "text-to-video": [],
    "image-to-video": [],
  },
  vaeCatalog: {
    "text-to-image": [],
    "image-to-image": [],
  },
  localLineageFilter: "all",
  outputs: [],
  outputsBaseDir: "",
  runtimeInfo: null,
  videoModelSpecs: {},
  searchPage: 1,
  searchNextCursor: null,
  searchPrevCursor: null,
  searchViewMode: "grid",
  searchDetail: null,
  downloadTasks: [],
  downloadPollTimer: null,
  downloadPollDelayMs: DOWNLOAD_TASK_POLL_INTERVAL_MS,
  downloadDigest: {},
  processedDownloadCompletions: {},
  downloadsPopoverOpen: false,
  activeTab: "text-image",
  currentTaskSnapshot: null,
  lastErrorPopupTaskId: null,
  generationBusy: false,
  taskLogLines: [],
};

const SEARCH_BASE_MODEL_OPTIONS_BY_TASK = {
  "text-to-image": [
    "all",
    "StableDiffusion 1.x",
    "StableDiffusion 1.5",
    "StableDiffusion 2.x",
    "StableDiffusion 2.1",
    "StableDiffusion XL",
    "FLUX",
    "PixArt",
    "AuraFlow",
    "Wan",
    "Other",
  ],
  "image-to-image": [
    "all",
    "StableDiffusion 1.x",
    "StableDiffusion 1.5",
    "StableDiffusion 2.x",
    "StableDiffusion 2.1",
    "StableDiffusion XL",
    "FLUX",
    "PixArt",
    "AuraFlow",
    "Wan",
    "Other",
  ],
  "text-to-video": ["all", "TextToVideoSD", "Wan", "Other"],
  "image-to-video": ["all", "I2VGenXL", "Wan", "Other"],
};

function el(id) {
  return document.getElementById(id);
}

function bindClick(id, handler) {
  const node = el(id);
  if (!node) return;
  node.addEventListener("click", handler);
}

function readNum(id) {
  const raw = el(id).value.trim();
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function getSelectedValues(selectId) {
  const node = el(selectId);
  if (!node) return [];
  return Array.from(node.selectedOptions || [])
    .map((opt) => String(opt.value || "").trim())
    .filter((v) => v);
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

function formatModelSize(sizeBytes) {
  const size = Number(sizeBytes);
  if (!Number.isFinite(size) || size <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatDateTime(isoString) {
  if (!isoString) return "n/a";
  const dt = new Date(isoString);
  if (Number.isNaN(dt.getTime())) return String(isoString);
  return dt.toLocaleString();
}

function closeModelDetailModal() {
  const modal = el("modelDetailModal");
  if (!modal) return;
  modal.classList.remove("open");
  document.body.style.overflow = "";
}

function openModelDetailModal() {
  const modal = el("modelDetailModal");
  if (!modal) return;
  modal.classList.add("open");
  document.body.style.overflow = "hidden";
}

function updateDownloadsBadge() {
  const badge = el("downloadsBadge");
  if (!badge) return;
  const tasks = state.downloadTasks || [];
  const running = tasks.filter((task) => task.status === "queued" || task.status === "running").length;
  const total = tasks.length;
  badge.textContent = total > 0 ? t("msgDownloadListSummary", { running, total }) : "0";
  badge.classList.toggle("active", running > 0);
}

function renderDownloadsList() {
  const container = el("downloadsList");
  if (!container) return;
  const tasks = state.downloadTasks || [];
  if (!tasks.length) {
    container.innerHTML = `<div class="downloads-empty">${escapeHtml(t("msgNoDownloads"))}</div>`;
    updateDownloadsBadge();
    return;
  }
  container.innerHTML = tasks
    .map((task) => {
      const progress = Math.round(taskProgressValue(task) * 100);
      const downloaded = Number(task.downloaded_bytes);
      const total = Number(task.total_bytes);
      const bytesText =
        Number.isFinite(downloaded) && downloaded >= 0 && Number.isFinite(total) && total > 0
          ? `${formatModelSize(downloaded)} / ${formatModelSize(total)}`
          : Number.isFinite(downloaded) && downloaded >= 0
            ? `${formatModelSize(downloaded)}`
            : "n/a";
      const errorText = task.error ? `<div class="downloads-item-meta">error=${escapeHtml(task.error)}</div>` : "";
      const messageText = translateServerMessage(task.message || "");
      const repoHint = task.result?.repo_id || task.result?.model || task.id;
      const cancellable = task.status === "queued" || task.status === "running";
      return `
        <article class="downloads-item" data-task-id="${escapeHtml(task.id)}">
          <div class="downloads-item-head">
            <div class="downloads-item-title">${escapeHtml(repoHint || task.id)}</div>
            <span class="downloads-item-status ${escapeHtml(task.status || "")}">${escapeHtml(translateTaskStatus(task.status || ""))}</span>
          </div>
          <div class="downloads-item-meta">${escapeHtml(messageText || "-")}</div>
          <div class="downloads-progress"><div class="downloads-progress-fill" style="width:${progress}%"></div></div>
          <div class="downloads-item-meta">${escapeHtml(progress)}% | ${escapeHtml(bytesText)} | ${escapeHtml(formatDateTime(task.updated_at))}</div>
          ${cancellable ? `<button type="button" class="ghost-btn download-cancel-btn" data-task-id="${escapeHtml(task.id)}">${escapeHtml(t("btnCancelTask"))}</button>` : ""}
          ${errorText}
        </article>
      `;
    })
    .join("");
  container.querySelectorAll(".downloads-item").forEach((node) => {
    node.addEventListener("click", () => {
      const taskId = node.getAttribute("data-task-id") || "";
      if (!taskId) return;
      const task = (state.downloadTasks || []).find((entry) => entry.id === taskId);
      if (!task) return;
      const localPath = task.result?.local_path || "-";
      const errorText = task.error || "-";
      showTaskMessage(`download=${task.id} | status=${task.status} | path=${localPath} | error=${errorText}`);
    });
  });
  container.querySelectorAll(".download-cancel-btn").forEach((node) => {
    node.addEventListener("click", async (event) => {
      event.stopPropagation();
      const taskId = node.getAttribute("data-task-id") || "";
      if (!taskId) return;
      try {
        await api("/api/tasks/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_id: taskId }),
        });
        showTaskMessage(t("msgTaskCancelRequested", { id: taskId }));
        await refreshDownloadTasks();
      } catch (error) {
        showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
      }
    });
  });
  updateDownloadsBadge();
}

function setDownloadsPopoverOpen(isOpen) {
  state.downloadsPopoverOpen = Boolean(isOpen);
  const popover = el("downloadsPopover");
  if (!popover) return;
  popover.classList.toggle("open", state.downloadsPopoverOpen);
}

async function refreshDownloadTasks() {
  const params = new URLSearchParams({
    task_type: "download",
    status: "all",
    limit: "30",
  });
  const data = await api(`/api/tasks?${params.toString()}`);
  state.downloadTasks = data.tasks || [];
  for (const task of state.downloadTasks) {
    const prev = state.downloadDigest[task.id] || "";
    const current = `${task.status}|${task.updated_at || ""}`;
    state.downloadDigest[task.id] = current;
    if (task.status === "completed" && !state.processedDownloadCompletions[task.id]) {
      state.processedDownloadCompletions[task.id] = true;
      const baseDir = String(task.result?.base_dir || "").trim();
      if (baseDir && el("localModelsDir") && el("localModelsDir").value.trim() !== baseDir) {
        el("localModelsDir").value = baseDir;
        showTaskMessage(t("msgDownloadPathSynced", { path: baseDir }));
      }
      try {
        await loadLocalModels({ forceRescan: true });
      } catch (error) {
        showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
      }
      continue;
    }
    if ((task.status === "completed" || task.status === "error") && prev && prev !== current && task.status === "completed") {
      try {
        await loadLocalModels({ forceRescan: true });
      } catch (error) {
        showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
      }
    }
  }
  renderDownloadsList();
}

function startDownloadPolling() {
  if (state.downloadPollTimer) {
    clearTimeout(state.downloadPollTimer);
    state.downloadPollTimer = null;
  }
  const poll = async () => {
    try {
      await refreshDownloadTasks();
      state.downloadPollDelayMs = DOWNLOAD_TASK_POLL_INTERVAL_MS;
    } catch (error) {
      state.downloadPollDelayMs = Math.min(15000, Math.round((state.downloadPollDelayMs || DOWNLOAD_TASK_POLL_INTERVAL_MS) * 1.8));
      if (state.downloadsPopoverOpen) {
        showTaskMessage(t("msgDownloadsRefreshFailed", { error: error.message }));
      }
    } finally {
      if (state.downloadPollTimer) {
        clearTimeout(state.downloadPollTimer);
      }
      state.downloadPollTimer = setTimeout(poll, state.downloadPollDelayMs || DOWNLOAD_TASK_POLL_INTERVAL_MS);
    }
  };
  state.downloadPollDelayMs = DOWNLOAD_TASK_POLL_INTERVAL_MS;
  poll();
}

function getModelOptionLabel(item) {
  const baseLabel = item.label || item.id || item.value;
  const sizeText = formatModelSize(item.size_bytes);
  if (sizeText === "n/a") return baseLabel;
  return `${baseLabel} (${t("modelSize")}: ${sizeText})`;
}

function normalizeModelId(value) {
  return String(value || "").trim().toLowerCase();
}

function getInstalledModelIdSet() {
  const installed = new Set();
  (state.localModels || []).forEach((item) => {
    installed.add(normalizeModelId(item.repo_hint));
    installed.add(normalizeModelId(item.name));
  });
  return installed;
}

function parseCivitaiId(value) {
  const raw = String(value || "").trim();
  const match = raw.match(/^civitai\/(\d+)/i) || raw.match(/^(\d+)$/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function openExternalUrl(url) {
  const target = String(url || "").trim();
  if (!target || target === "#") return;
  window.open(target, "_blank", "noopener,noreferrer");
}

function taskShortName(task) {
  if (task === "text-to-image") return "T2I";
  if (task === "image-to-image") return "I2I";
  if (task === "text-to-video") return "T2V";
  if (task === "image-to-video") return "I2V";
  return task;
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
  document.querySelectorAll("[data-help-key]").forEach((node) => {
    const text = t(node.dataset.helpKey);
    node.setAttribute("data-help", text);
    node.setAttribute("title", text);
  });
  refreshSearchSourceOptions();
  renderSearchBaseModelOptions();
  if (!state.currentTaskId) {
    showTaskMessage(t("statusNoTask"));
    renderTaskProgress(null);
  }
  renderModelSelect("text-to-image");
  renderModelSelect("image-to-image");
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
  renderLoraSelect("text-to-image");
  renderLoraSelect("image-to-image");
  renderLoraSelect("text-to-video");
  renderLoraSelect("image-to-video");
  renderVaeSelect("text-to-image");
  renderVaeSelect("image-to-image");
  renderSettingsDefaultModelSelects();
  if (el("localViewMode")) {
    el("localViewMode").value = state.localViewMode === "flat" ? "flat" : "tree";
  }
  if (el("localTreeSearch")) {
    el("localTreeSearch").value = state.localTreeFilter || "";
  }
  renderLocalLineageOptions(state.localModels || []);
  renderLocalViewMode();
  renderLocalModels(state.localModels || [], state.localModelsBaseDir || "");
  renderLocalModelTree(state.localTreeData);
  renderLocalTreeSelection();
  renderOutputs(state.outputs || [], state.outputsBaseDir || "");
  if (!state.searchDetail) {
    const modalTitle = el("modelDetailModalTitle");
    const modalContent = el("modelDetailModalContent");
    if (modalTitle) modalTitle.textContent = t("msgModelDetailEmpty");
    if (modalContent) modalContent.innerHTML = "";
  }
  renderDownloadsList();
  if (el("cancelCurrentTaskBtn")) {
    el("cancelCurrentTaskBtn").textContent = t("btnCancelTask");
  }
  if (el("searchPrevBtn")) el("searchPrevBtn").textContent = t("btnPrev");
  if (el("searchNextBtn")) el("searchNextBtn").textContent = t("btnNext");
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
    renderSearchPagination({ page: state.searchPage, has_prev: Boolean(state.searchPrevCursor), has_next: Boolean(state.searchNextCursor) });
  }
}

function setLanguage(languageCode) {
  state.language = normalizeLanguage(languageCode);
  localStorage.setItem(LANG_STORAGE_KEY, state.language);
  el("languageSelect").value = state.language;
  applyI18n();
  loadRuntimeInfo().catch(() => {});
  loadLocalModels().catch(() => {});
}

async function loadExternalI18n() {
  try {
    const response = await fetch(EXTERNAL_I18N_URL, { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload || typeof payload !== "object") return;
    Object.entries(payload).forEach(([lang, dict]) => {
      if (!I18N[lang] || !dict || typeof dict !== "object") return;
      I18N[lang] = { ...I18N[lang], ...dict };
    });
  } catch (_error) {
    // 外部辞書がなくても既定辞書で動作可能なため、初期化を止めない。
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

async function withApiState(action, hooks = {}) {
  const { onLoading, onSuccess, onError, onFinally } = hooks;
  if (onLoading) onLoading();
  try {
    const result = await action();
    if (onSuccess) onSuccess(result);
    return result;
  } catch (error) {
    if (onError) onError(error);
    throw error;
  } finally {
    if (onFinally) onFinally();
  }
}

function showTaskMessage(text) {
  el("taskStatus").textContent = text;
  appendTaskLog(text);
}

function appendTaskLog(text) {
  const line = `[${new Date().toLocaleTimeString()}] ${String(text || "")}`;
  state.taskLogLines.push(line);
  if (state.taskLogLines.length > 200) {
    state.taskLogLines = state.taskLogLines.slice(state.taskLogLines.length - 200);
  }
  const logNode = el("taskLogContent");
  if (logNode) {
    logNode.textContent = state.taskLogLines.join("\n");
    logNode.scrollTop = logNode.scrollHeight;
  }
}

function setGenerationBusy(isBusy) {
  state.generationBusy = Boolean(isBusy);
  const submitSelectors = [
    "#text2videoForm button[type='submit']",
    "#image2videoForm button[type='submit']",
    "#text2imageForm button[type='submit']",
    "#image2imageForm button[type='submit']",
  ];
  submitSelectors.forEach((selector) => {
    const node = document.querySelector(selector);
    if (!node) return;
    node.disabled = state.generationBusy;
    node.classList.toggle("is-busy", state.generationBusy);
  });
  const cancelBtn = el("cancelCurrentTaskBtn");
  if (cancelBtn) {
    cancelBtn.disabled = !state.generationBusy || !state.currentTaskId;
  }
}

function showTaskErrorPopup(task) {
  if (!task || task.status !== "error") return;
  const taskId = String(task.id || "");
  if (taskId && state.lastErrorPopupTaskId === taskId) return;
  state.lastErrorPopupTaskId = taskId || `error-${Date.now()}`;
  const typeLabel = translateTaskType(task.task_type || "");
  const reason = String(task.error || task.message || translateTaskStatus(task.status) || "unknown");
  window.alert(t("msgTaskErrorPopup", { type: typeLabel, error: reason }));
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}

function downloadProgressFromBytes(task) {
  if (!task || task.task_type !== "download") return null;
  const downloaded = Number(task.downloaded_bytes);
  const total = Number(task.total_bytes);
  if (!Number.isFinite(downloaded) || !Number.isFinite(total) || total <= 0) return null;
  return clamp01(downloaded / total);
}

function taskProgressValue(task) {
  const bytesRatio = downloadProgressFromBytes(task);
  if (bytesRatio != null) return bytesRatio;
  return clamp01(task?.progress);
}

function formatDuration(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function taskTimeBase(task) {
  return task.started_at || task.created_at || null;
}

function estimateElapsedAndEta(task, progressValue) {
  const base = taskTimeBase(task);
  if (!base) return { elapsedSec: 0, etaSec: null };
  const started = new Date(base).getTime();
  if (!Number.isFinite(started)) return { elapsedSec: 0, etaSec: null };
  const elapsedSec = Math.max(0, (Date.now() - started) / 1000);
  const p = clamp01(progressValue);
  if (p <= 0 || p >= 1 || task.status !== "running") return { elapsedSec, etaSec: null };
  const totalSec = elapsedSec / p;
  const etaSec = Math.max(0, totalSec - elapsedSec);
  return { elapsedSec, etaSec };
}

function renderTaskProgress(task) {
  const wrap = el("taskProgressWrap");
  const bar = el("taskProgressBar");
  const label = el("taskProgressLabel");
  if (!wrap || !bar || !label) return;
  const hideOnModelTabs = state.activeTab === "models" || state.activeTab === "local-models";
  wrap.style.display = hideOnModelTabs ? "none" : "";
  if (!task) {
    bar.style.width = "0%";
    label.textContent = "0% | ETA --:-- | ELAPSED 00:00";
    return;
  }
  const progressValue = taskProgressValue(task);
  const pct = Math.round(progressValue * 100);
  const { elapsedSec, etaSec } = estimateElapsedAndEta(task, progressValue);
  const etaText = etaSec == null ? "--:--" : formatDuration(etaSec);
  const elapsedText = formatDuration(elapsedSec);
  const downloaded = Number(task.downloaded_bytes);
  const total = Number(task.total_bytes);
  const downloadedText = Number.isFinite(downloaded) && downloaded <= 0 ? "0 B" : formatModelSize(downloaded);
  const bytesText =
    Number.isFinite(downloaded) && downloaded >= 0 && Number.isFinite(total) && total > 0
      ? ` | ${downloadedText} / ${formatModelSize(total)}`
      : "";
  bar.style.width = `${pct}%`;
  label.textContent = `${pct}%${bytesText} | ETA ${etaText} | ELAPSED ${elapsedText}`;
}

function saveLastTaskId(taskId) {
  if (taskId) {
    localStorage.setItem(TASK_STORAGE_KEY, taskId);
  } else {
    localStorage.removeItem(TASK_STORAGE_KEY);
  }
}

function setTabs() {
  const active = document.querySelector(".tab.active");
  state.activeTab = active?.dataset?.tab || "text-image";
  renderTaskProgress(state.currentTaskSnapshot);
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      el(`panel-${button.dataset.tab}`).classList.add("active");
      state.activeTab = button.dataset.tab || "text-image";
      renderTaskProgress(state.currentTaskSnapshot);
    });
  });
}

function refreshSearchSourceOptions() {
  const taskNode = el("searchTask");
  const sourceNode = el("searchSource");
  if (!taskNode || !sourceNode) return;
  const supportsCivitai = ["text-to-image", "image-to-image"].includes(taskNode.value);
  const civitaiOption = sourceNode.querySelector('option[value="civitai"]');
  if (civitaiOption) {
    civitaiOption.disabled = !supportsCivitai;
  }
  if (!supportsCivitai && sourceNode.value === "civitai") {
    sourceNode.value = "all";
  }
}

function renderSearchBaseModelOptions() {
  const taskNode = el("searchTask");
  const select = el("searchBaseModel");
  if (!taskNode || !select) return;
  const task = taskNode.value || "text-to-image";
  const options = SEARCH_BASE_MODEL_OPTIONS_BY_TASK[task] || ["all", "Other"];
  const current = select.value || "all";
  select.innerHTML = options
    .map((value) => {
      if (value === "all") {
        return `<option value="all">${escapeHtml(t("searchBaseModelAll"))}</option>`;
      }
      return `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`;
    })
    .join("");
  select.value = options.includes(current) ? current : "all";
}

function inferLocalLineage(item) {
  const text = `${item?.base_model || ""} ${item?.base_name || ""} ${item?.repo_hint || ""} ${item?.class_name || ""}`.toLowerCase();
  if (text.includes("stable-diffusion-xl") || text.includes("sdxl") || /\bxl\b/.test(text)) return "StableDiffusion XL";
  if (text.includes("stable-diffusion-2-1") || text.includes("v2-1") || text.includes("2.1")) return "StableDiffusion 2.1";
  if (text.includes("stable-diffusion-2") || /\bsd2\b/.test(text) || /\b2\.0\b/.test(text)) return "StableDiffusion 2.x";
  if (text.includes("stable-diffusion-1-5") || text.includes("v1-5") || text.includes("1.5")) return "StableDiffusion 1.5";
  if (text.includes("stable-diffusion-1") || /\bsd1\b/.test(text) || text.includes("1.4") || text.includes("1.0")) return "StableDiffusion 1.x";
  if (text.includes("flux")) return "FLUX";
  if (text.includes("pixart")) return "PixArt";
  if (text.includes("auraflow")) return "AuraFlow";
  if (text.includes("wan")) return "Wan";
  if (text.includes("i2vgen")) return "I2VGenXL";
  if (text.includes("texttovideosdpipeline") || text.includes("text-to-video")) return "TextToVideoSD";
  return "Other";
}

function localModelKind(item) {
  if (item?.is_lora) return t("modelKindLora");
  if (item?.is_vae) return t("modelKindVae");
  return t("modelKindBase");
}

function outputTypeLabel(kind) {
  if (kind === "image") return t("outputTypeImage");
  if (kind === "video") return t("outputTypeVideo");
  return t("outputTypeOther");
}

function renderLocalLineageOptions(items) {
  const select = el("localLineageFilter");
  if (!select) return;
  const values = Array.from(new Set((items || []).map((item) => inferLocalLineage(item)))).sort((a, b) => a.localeCompare(b));
  const options = [`<option value="all">${escapeHtml(t("localLineageAll"))}</option>`];
  values.forEach((lineage) => {
    options.push(`<option value="${escapeHtml(lineage)}">${escapeHtml(lineage)}</option>`);
  });
  select.innerHTML = options.join("");
  const desired = state.localLineageFilter || "all";
  select.value = Array.from(select.options).some((opt) => opt.value === desired) ? desired : "all";
  state.localLineageFilter = select.value;
}

function translateTaskType(taskType) {
  if (taskType === "text2image") return t("taskTypeText2Image");
  if (taskType === "image2image") return t("taskTypeImage2Image");
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
  if (status === "cancelled") return t("statusCancelled");
  return status || t("taskTypeUnknown");
}

function translateTaskStep(step) {
  const stepMap = {
    queued: t("stepQueued"),
    runtime_diagnostics: t("stepRuntimeDiagnostics"),
    model_load: t("stepModelLoad"),
    model_load_gpu: t("stepModelLoadGpu"),
    model_load_auto_map: t("stepModelLoadAutoMap"),
    model_load_cpu_offload: t("stepModelLoadCpuOffload"),
    model_load_cpu: t("stepModelLoadCpu"),
    lora_apply: t("stepLoraApply"),
    prepare: t("stepPrepare"),
    inference: t("stepInference"),
    generating: t("stepGenerating"),
    extracting_output: t("stepExtractingOutput"),
    decoding_frames: t("stepDecodingFrames"),
    postprocessing: t("stepPostprocessing"),
    encoding_video: t("stepEncodingVideo"),
    saving: t("stepSaving"),
    decode: t("stepDecode"),
    encode: t("stepEncode"),
    save: t("stepSave"),
    memory_cleanup: t("stepMemoryCleanup"),
    done: t("stepDone"),
    failed: t("stepFailed"),
    cancelled: t("stepCancelled"),
  };
  const normalized = String(step || "").trim();
  return stepMap[normalized] || normalized || t("taskStepUnknown");
}

function renderTaskStep(task) {
  const row = el("taskStepRow");
  const label = el("taskStepLabel");
  if (!row || !label) return;
  if (!task) {
    row.className = "task-step-row idle";
    label.textContent = t("taskStepUnknown");
    return;
  }
  const status = String(task.status || "");
  const rowStatusClass =
    status === "completed" ? "done" : status === "error" ? "failed" : status === "cancelled" ? "cancelled" : "running";
  row.className = `task-step-row ${rowStatusClass}`;
  label.textContent = `${t("labelTaskStep")}: ${translateTaskStep(task.step)}`;
}

function translateServerMessage(message) {
  const raw = (message || "").trim();
  if (!raw) return "";
  const progressSuffix = raw.match(/\(\d+\/\d+\)$/)?.[0] || "";
  if (raw.startsWith("Generating image")) {
    return `${t("serverGeneratingImage")}${progressSuffix ? ` ${progressSuffix}` : ""}`;
  }
  if (raw.startsWith("Generating frames")) {
    return `${t("serverGeneratingFrames")}${progressSuffix ? ` ${progressSuffix}` : ""}`;
  }
  const map = {
    Queued: t("serverQueued"),
    "Generation queued": t("serverGenerationQueued"),
    "Download queued": t("serverDownloadQueued"),
    "Loading model": t("serverLoadingModel"),
    "VRAMにロード中 (device_map={'': 'cuda'})": t("serverLoadingModelVram"),
    "VRAMにロード中 (device_map='cuda')": t("serverLoadingModelVram"),
    "自動device_mapでロード中": t("serverLoadingModelAutoMap"),
    "CPUオフロード有効化中": t("serverLoadingModelCpuOffload"),
    "CPU低メモリモードでロード中": t("serverLoadingModelCpuLowMem"),
    "Applying LoRA": t("serverLoadingLora"),
    "Preparing image": t("serverPreparingImage"),
    "Generating image": t("serverGeneratingImage"),
    "Generating frames": t("serverGeneratingFrames"),
    "Decoding latents": t("serverDecodingLatents"),
    "Decoding latents (CPU fallback)": t("serverDecodingLatents"),
    "Postprocessing image": t("serverPostprocessingImage"),
    "Encoding mp4": t("serverEncoding"),
    "メモリ解放中": t("serverMemoryCleanup"),
    "Saving png": t("serverSavingPng"),
    Done: t("serverDone"),
    "Generation failed": t("serverGenerationFailed"),
    "Download complete": t("serverDownloadComplete"),
    "Download failed": t("serverDownloadFailed"),
    Cancelled: t("statusCancelled"),
    "Cancellation requested": t("statusCancelled"),
  };
  return map[raw] || raw;
}

async function loadRuntimeInfo() {
  try {
    const info = await api("/api/runtime");
    state.runtimeInfo = info;
    const flags = [
      `${t("runtimeDevice")}=${info.device}`,
      `${t("runtimeCuda")}=${info.cuda_available}`,
      `${t("runtimeRocm")}=${info.rocm_available}`,
      `${t("runtimeNpu")}=${info.npu_available}`,
      `${t("runtimeDiffusers")}=${info.diffusers_ready}`,
    ];
    if (info.torch_version) flags.push(`${t("runtimeTorch")}=${info.torch_version}`);
    if (info.dtype) flags.push(`dtype=${info.dtype}`);
    if (info.bf16_supported !== undefined) flags.push(`bf16_supported=${info.bf16_supported}`);
    if (info.hardware_profile) {
      const hw = info.hardware_profile;
      if (hw.gpu_total_bytes) {
        flags.push(`VRAM=${formatModelSize(hw.gpu_free_bytes)}/${formatModelSize(hw.gpu_total_bytes)}`);
      }
      if (hw.host_total_bytes) {
        flags.push(`RAM=${formatModelSize(hw.host_available_bytes)}/${formatModelSize(hw.host_total_bytes)}`);
      }
      if (hw.gpu_name) {
        flags.push(`GPU=${hw.gpu_name}`);
      }
    }
    if (info.load_policy_preview?.selected_policy_name) {
      flags.push(`load_policy=${info.load_policy_preview.selected_policy_name}`);
      flags.push(`vram_threshold_gb=${info.load_policy_preview.vram_gpu_direct_load_threshold_gb}`);
    }
    if (info.last_load_policy?.policy?.name) {
      flags.push(`last_load_policy=${info.last_load_policy.policy.name}`);
    }
    if (info.npu_reason) flags.push(`${t("runtimeNpuReason")}=${info.npu_reason}`);
    if (info.aotriton_mismatch?.warning) flags.push(`AOTriton=${info.aotriton_mismatch.warning}`);
    if (info.import_error) flags.push(`${t("runtimeError")}=${info.import_error}`);
    el("runtimeInfo").textContent = flags.join(" | ");
    applyNpuAvailability(info);
  } catch (error) {
    state.runtimeInfo = null;
    el("runtimeInfo").textContent = t("runtimeLoadFailed", { error: error.message });
    applyNpuAvailability(null);
  }
}

async function loadVideoModelSpecs() {
  try {
    const payload = await api("/api/video/models");
    const specs = {};
    for (const item of payload?.items || []) {
      if (!item?.key) continue;
      specs[String(item.key)] = item;
    }
    state.videoModelSpecs = specs;
    renderModelPreview("text-to-video");
    renderModelPreview("image-to-video");
  } catch (_error) {
    state.videoModelSpecs = {};
  }
}

function inferVideoModelKeyFromRef(modelRef) {
  const text = String(modelRef || "").toLowerCase();
  if (!text) return "";
  if (text.includes("wan")) return "wan";
  if (text.includes("stable-video-diffusion") || text.includes("stablevideodiffusion") || text.includes("img2vid")) {
    return "stablevideodiffusion";
  }
  if (text.includes("cogvideox") || text.includes("cogvideo")) return "cogvideox";
  if (text.includes("ltx")) return "ltxvideo";
  if (text.includes("hunyuan")) return "hunyuanvideo";
  if (text.includes("sana")) return "sanavideo";
  if (text.includes("animatediff")) return "animatediff";
  if (text.includes("text-to-video-ms") || text.includes("texttovideosd")) return "text2videosd";
  return "";
}

function supportLevelText(level) {
  const normalized = String(level || "").trim().toLowerCase();
  if (normalized === "ready") return t("modelSupportReady");
  if (normalized === "limited") return t("modelSupportLimited");
  if (normalized === "requires_patch") return t("modelSupportRequiresPatch");
  if (normalized === "not_supported") return t("modelSupportNotSupported");
  return normalized || t("modelSupportLimited");
}

function videoSpecForTaskModel(task, modelRef) {
  if (task !== "text-to-video" && task !== "image-to-video") return null;
  const key = inferVideoModelKeyFromRef(modelRef);
  if (!key) return null;
  const spec = state.videoModelSpecs[key];
  if (!spec) return null;
  const tasks = Array.isArray(spec.tasks) ? spec.tasks : [];
  if (tasks.length > 0 && !tasks.includes(task)) return spec;
  return spec;
}

function assertVideoModelUsable(task, modelRef) {
  const spec = videoSpecForTaskModel(task, modelRef);
  if (!spec) return;
  const key = inferVideoModelKeyFromRef(modelRef);
  const tasks = Array.isArray(spec.tasks) ? spec.tasks : [];
  if (tasks.length > 0 && !tasks.includes(task)) {
    throw new Error(t("msgVideoModelTaskUnsupported", { model: modelRef || spec.display_name || key, task }));
  }
  const taskSupport = spec.task_support || {};
  if (Object.prototype.hasOwnProperty.call(taskSupport, task) && taskSupport[task] === false) {
    const reason = String(spec.status_reason || spec.rocm_notes || "required pipeline class is missing");
    throw new Error(t("msgVideoModelRuntimeUnsupported", { model: spec.display_name || key, reason }));
  }
  const level = String(spec.effective_support_level || spec.support_level || "ready");
  if (level === "not_supported" || level === "requires_patch") {
    const reason = String(spec.status_reason || spec.rocm_notes || "unsupported model on current runtime");
    throw new Error(t("msgVideoModelRuntimeUnsupported", { model: spec.display_name || key, reason }));
  }
}

function applyNpuAvailability(info) {
  const npuAvailable = Boolean(info?.npu_available);
  const npuRunnable = Boolean(info?.t2v_npu_runner_configured);
  const t2vBackend = el("t2vBackendSelect");
  if (t2vBackend) {
    const npuOption = Array.from(t2vBackend.options).find((opt) => opt.value === "npu");
    if (npuOption) npuOption.disabled = !npuRunnable;
    if (!npuRunnable && t2vBackend.value === "npu") {
      t2vBackend.value = "auto";
    }
  }
  const cfgBackend = el("cfgT2VBackend");
  if (cfgBackend) {
    const npuOption = Array.from(cfgBackend.options).find((opt) => opt.value === "npu");
    if (npuOption) npuOption.disabled = !npuRunnable;
    if (!npuRunnable && cfgBackend.value === "npu") {
      cfgBackend.value = "auto";
    }
  }
}

function applySettings(settings) {
  state.settings = settings;
  state.defaultModels["text-to-image"] = settings.defaults.text2image_model || "";
  state.defaultModels["image-to-image"] = settings.defaults.image2image_model || "";
  state.defaultModels["text-to-video"] = settings.defaults.text2video_model || "";
  state.defaultModels["image-to-video"] = settings.defaults.image2video_model || "";
  const serverSettings = settings.server || {};
  const supportedBackends = ["auto", "cuda", "npu"];
  const defaultT2vBackend = supportedBackends.includes(String(serverSettings.t2v_backend || "").toLowerCase())
    ? String(serverSettings.t2v_backend).toLowerCase()
    : "auto";
  el("cfgModelsDir").value = settings.paths.models_dir;
  el("cfgListenPort").value = serverSettings?.listen_port ?? 8000;
  if (el("cfgRocmAotriton")) {
    el("cfgRocmAotriton").checked = serverSettings?.rocm_aotriton_experimental !== false;
  }
  if (el("cfgPreferredDtype")) {
    const preferred = String(serverSettings?.preferred_dtype || "bf16").toLowerCase();
    el("cfgPreferredDtype").value = preferred === "float16" ? "float16" : "bf16";
  }
  if (el("cfgVramDirectThresholdGb")) {
    el("cfgVramDirectThresholdGb").value = Number(serverSettings?.vram_gpu_direct_load_threshold_gb || 48);
  }
  if (el("cfgEnableDeviceMapAuto")) {
    el("cfgEnableDeviceMapAuto").checked = serverSettings?.enable_device_map_auto !== false;
  }
  if (el("cfgEnableModelCpuOffload")) {
    el("cfgEnableModelCpuOffload").checked = serverSettings?.enable_model_cpu_offload !== false;
  }
  if (el("cfgGpuMaxConcurrency")) {
    el("cfgGpuMaxConcurrency").value = Number(serverSettings?.gpu_max_concurrency || 1);
  }
  if (el("cfgSoftwareVideoFallback")) {
    el("cfgSoftwareVideoFallback").checked = Boolean(serverSettings?.allow_software_video_fallback);
  }
  const storage = settings.storage || {};
  if (el("cfgCleanupMaxAgeDays")) {
    el("cfgCleanupMaxAgeDays").value = Number(storage.cleanup_max_age_days || 7);
  }
  if (el("cfgCleanupMaxOutputsCount")) {
    el("cfgCleanupMaxOutputsCount").value = Number(storage.cleanup_max_outputs_count || 200);
  }
  if (el("cfgCleanupMaxTmpCount")) {
    el("cfgCleanupMaxTmpCount").value = Number(storage.cleanup_max_tmp_count || 300);
  }
  if (el("cfgCleanupMaxCacheGb")) {
    el("cfgCleanupMaxCacheGb").value = Number(storage.cleanup_max_cache_size_gb || 30);
  }
  if (el("cfgT2VBackend")) {
    el("cfgT2VBackend").value = defaultT2vBackend;
  }
  if (el("cfgT2VNpuRunner")) {
    el("cfgT2VNpuRunner").value = serverSettings?.t2v_npu_runner || "";
  }
  if (el("cfgT2VNpuModelDir")) {
    el("cfgT2VNpuModelDir").value = serverSettings?.t2v_npu_model_dir || "";
  }
  if (el("t2vBackendSelect")) {
    el("t2vBackendSelect").value = defaultT2vBackend;
  }
  el("cfgOutputsDir").value = settings.paths.outputs_dir;
  el("cfgTmpDir").value = settings.paths.tmp_dir;
  el("cfgLogLevel").value = (settings.logging?.level || "INFO").toUpperCase();
  el("cfgToken").value = settings.huggingface.token || "";
  renderSettingsDefaultModelSelects(
    settings.defaults.text2video_model || "",
    settings.defaults.image2video_model || "",
    settings.defaults.text2image_model || "",
    settings.defaults.image2image_model || "",
  );
  const defaultFps = Number(settings.defaults.fps || 8) || 8;
  const defaultDurationSeconds =
    settings.defaults.duration_seconds !== undefined && settings.defaults.duration_seconds !== null
      ? Number(settings.defaults.duration_seconds)
      : Number(settings.defaults.num_frames || 16) / Math.max(1, defaultFps);
  el("cfgSteps").value = settings.defaults.num_inference_steps;
  el("cfgFrames").value = Number.isFinite(defaultDurationSeconds) ? defaultDurationSeconds : 2.0;
  el("cfgGuidance").value = settings.defaults.guidance_scale;
  el("cfgFps").value = settings.defaults.fps;
  el("cfgWidth").value = settings.defaults.width;
  el("cfgHeight").value = settings.defaults.height;
  el("t2iSteps").value = settings.defaults.num_inference_steps;
  el("t2iGuidance").value = settings.defaults.guidance_scale;
  el("t2iWidth").value = settings.defaults.width;
  el("t2iHeight").value = settings.defaults.height;
  el("i2iSteps").value = settings.defaults.num_inference_steps;
  el("i2iGuidance").value = settings.defaults.guidance_scale;
  el("i2iWidth").value = settings.defaults.width;
  el("i2iHeight").value = settings.defaults.height;
  el("t2vSteps").value = settings.defaults.num_inference_steps;
  el("t2vFrames").value = Number.isFinite(defaultDurationSeconds) ? defaultDurationSeconds : 2.0;
  el("t2vGuidance").value = settings.defaults.guidance_scale;
  el("t2vFps").value = settings.defaults.fps;
  el("i2vSteps").value = settings.defaults.num_inference_steps;
  el("i2vFrames").value = Number.isFinite(defaultDurationSeconds) ? defaultDurationSeconds : 2.0;
  el("i2vGuidance").value = settings.defaults.guidance_scale;
  el("i2vFps").value = settings.defaults.fps;
  el("i2vWidth").value = settings.defaults.width;
  el("i2vHeight").value = settings.defaults.height;
  if (!el("downloadTargetDir").value.trim()) {
    el("downloadTargetDir").value = settings.paths.models_dir;
  }
  if (!el("localModelsDir").value.trim()) {
    el("localModelsDir").value = settings.paths.models_dir;
  }
  renderModelSelect("text-to-image");
  renderModelSelect("image-to-image");
  renderModelSelect("text-to-video");
  renderModelSelect("image-to-video");
  applyNpuAvailability(state.runtimeInfo);
}

function renderDefaultModelSettingSelect(selectId, selectedValue) {
  const select = el(selectId);
  if (!select) return;
  const taskBySelect = {
    cfgTextModel: "text-to-video",
    cfgImageModel: "image-to-video",
    cfgTextImageModel: "text-to-image",
    cfgImageImageModel: "image-to-image",
  };
  const task = taskBySelect[selectId];
  const localIds = Array.from(new Set((state.modelCatalog[task] || []).map((item) => item.id || "")))
    .filter((id) => id)
    .sort((a, b) => String(a).localeCompare(String(b)));
  const normalizedSelected = String(selectedValue || "").trim();
  const options = [`<option value="">${escapeHtml(t("msgSelectLocalModel"))}</option>`];
  localIds.forEach((modelId) => {
    options.push(`<option value="${escapeHtml(modelId)}">${escapeHtml(modelId)}</option>`);
  });
  if (normalizedSelected && !localIds.includes(normalizedSelected)) {
    options.push(
      `<option value="${escapeHtml(normalizedSelected)}">${escapeHtml(t("msgDefaultModelNotLocal", { model: normalizedSelected }))}</option>`,
    );
  }
  select.innerHTML = options.join("");
  if (normalizedSelected && Array.from(select.options).some((opt) => opt.value === normalizedSelected)) {
    select.value = normalizedSelected;
  } else {
    select.value = "";
  }
}

function renderSettingsDefaultModelSelects(textModelValue = null, imageModelValue = null, textImageModelValue = null, imageImageModelValue = null) {
  const textValue = textModelValue !== null ? textModelValue : el("cfgTextModel")?.value || state.settings?.defaults?.text2video_model || "";
  const imageValue =
    imageModelValue !== null ? imageModelValue : el("cfgImageModel")?.value || state.settings?.defaults?.image2video_model || "";
  const textImageValue =
    textImageModelValue !== null
      ? textImageModelValue
      : el("cfgTextImageModel")?.value || state.settings?.defaults?.text2image_model || "";
  const imageImageValue =
    imageImageModelValue !== null
      ? imageImageModelValue
      : el("cfgImageImageModel")?.value || state.settings?.defaults?.image2image_model || "";
  renderDefaultModelSettingSelect("cfgTextModel", textValue);
  renderDefaultModelSettingSelect("cfgImageModel", imageValue);
  renderDefaultModelSettingSelect("cfgTextImageModel", textImageValue);
  renderDefaultModelSettingSelect("cfgImageImageModel", imageImageValue);
}

async function loadSettingsLocalModels() {
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  renderSettingsDefaultModelSelects();
}

async function loadSettings() {
  const settings = await api("/api/settings");
  applySettings(settings);
  await loadSettingsLocalModels();
}

function getModelDom(task) {
  if (task === "text-to-image") {
    return { selectId: "t2iModelSelect", previewId: "t2iModelPreview" };
  }
  if (task === "image-to-image") {
    return { selectId: "i2iModelSelect", previewId: "i2iModelPreview" };
  }
  if (task === "text-to-video") {
    return { selectId: "t2vModelSelect", previewId: "t2vModelPreview" };
  }
  return { selectId: "i2vModelSelect", previewId: "i2vModelPreview" };
}

function getLoraDom(task) {
  if (task === "text-to-image") return { selectId: "t2iLoraSelect" };
  if (task === "image-to-image") return { selectId: "i2iLoraSelect" };
  if (task === "text-to-video") return { selectId: "t2vLoraSelect" };
  return { selectId: "i2vLoraSelect" };
}

function getVaeDom(task) {
  if (task === "text-to-image") return { selectId: "t2iVaeSelect" };
  return { selectId: "i2iVaeSelect" };
}

function getSelectedOrDefaultModelValue(task) {
  const modelDom = getModelDom(task);
  const selected = (el(modelDom.selectId)?.value || "").trim();
  if (selected) return selected;
  return (state.defaultModels[task] || "").trim();
}

function renderLoraSelect(task, preferredValue = null) {
  const dom = getLoraDom(task);
  const select = el(dom.selectId);
  if (!select) return;
  const currentValues = preferredValue === null ? getSelectedValues(dom.selectId) : preferredValue;
  const items = state.loraCatalog[task] || [];
  let html = `<option value="">${escapeHtml(t("msgNoLoraOption"))}</option>`;
  html += items.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`).join("");
  select.innerHTML = html;
  const currentSet = new Set(currentValues || []);
  Array.from(select.options).forEach((option, idx) => {
    option.selected = currentSet.has(option.value);
    if (!currentSet.size && idx === 0) {
      option.selected = true;
    }
  });
}

async function loadLoraCatalog(task, keepSelection = true) {
  const dom = getLoraDom(task);
  const select = el(dom.selectId);
  const prev = keepSelection && select ? getSelectedValues(dom.selectId) : [];
  const params = new URLSearchParams({
    task,
    limit: "200",
  });
  const modelRef = getSelectedOrDefaultModelValue(task);
  if (modelRef) params.set("model_ref", modelRef);
  const data = await api(`/api/models/loras/catalog?${params.toString()}`);
  state.loraCatalog[task] = data.items || [];
  renderLoraSelect(task, prev);
}

function renderVaeSelect(task, preferredValue = null) {
  const dom = getVaeDom(task);
  const select = el(dom.selectId);
  if (!select) return;
  const currentValue = preferredValue === null ? select.value : preferredValue;
  const items = state.vaeCatalog[task] || [];
  let html = `<option value="">${escapeHtml(t("msgNoVaeOption"))}</option>`;
  html += items.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`).join("");
  select.innerHTML = html;
  if (currentValue && items.some((item) => item.value === currentValue)) {
    select.value = currentValue;
  } else {
    select.value = "";
  }
}

async function loadVaeCatalog(task, keepSelection = true) {
  const dom = getVaeDom(task);
  const select = el(dom.selectId);
  const prev = keepSelection && select ? select.value : "";
  const data = await api("/api/models/vaes/catalog?limit=200");
  state.vaeCatalog[task] = data.items || [];
  renderVaeSelect(task, prev);
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
  const sizeText = formatModelSize(chosen.size_bytes);
  const metaText =
    sizeText === "n/a"
      ? escapeHtml(chosen.source || "model")
      : `${escapeHtml(chosen.source || "model")} | ${escapeHtml(t("modelSize"))}: ${escapeHtml(sizeText)}`;
  const resolvedRef = selectedValue || chosen.value || chosen.id || defaultId || "";
  const videoSpec = videoSpecForTaskModel(task, resolvedRef);
  let supportHtml = "";
  if (videoSpec) {
    const level = String(videoSpec.effective_support_level || videoSpec.support_level || "ready").trim().toLowerCase();
    const supportClass = `support-${escapeHtml(level.replaceAll("_", "-"))}`;
    const reason = String(videoSpec.status_reason || videoSpec.rocm_notes || "").trim();
    const reasonHtml = reason ? `<small>${escapeHtml(reason)}</small>` : "";
    supportHtml = `<div class="model-support-note ${supportClass}">${escapeHtml(t("modelSupportLabel"))}: ${escapeHtml(
      supportLevelText(level),
    )}${reasonHtml}</div>`;
  }

  preview.innerHTML = `
    <div class="model-picked-card">
      ${imageHtml}
      <div class="model-picked-meta">${infoHtml}<span>${metaText}</span>${supportHtml}</div>
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
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(getModelOptionLabel(item))}</option>`)
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
  await loadLoraCatalog(task, keepSelection);
  if (task === "text-to-image" || task === "image-to-image") {
    await loadVaeCatalog(task, keepSelection);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const prevPort = Number(state.settings?.server?.listen_port || 0);
  const prevRocmAotriton = state.settings?.server?.rocm_aotriton_experimental !== false;
  const payload = {
    server: {
      listen_port: Number(el("cfgListenPort").value),
      rocm_aotriton_experimental: Boolean(el("cfgRocmAotriton")?.checked),
      preferred_dtype: (el("cfgPreferredDtype")?.value || "bf16").trim(),
      vram_gpu_direct_load_threshold_gb: Number(el("cfgVramDirectThresholdGb")?.value || 48),
      enable_device_map_auto: Boolean(el("cfgEnableDeviceMapAuto")?.checked),
      enable_model_cpu_offload: Boolean(el("cfgEnableModelCpuOffload")?.checked),
      gpu_max_concurrency: Number(el("cfgGpuMaxConcurrency")?.value || 1),
      allow_software_video_fallback: Boolean(el("cfgSoftwareVideoFallback")?.checked),
      t2v_backend: (el("cfgT2VBackend")?.value || "auto").trim(),
      t2v_npu_runner: (el("cfgT2VNpuRunner")?.value || "").trim(),
      t2v_npu_model_dir: (el("cfgT2VNpuModelDir")?.value || "").trim(),
    },
    paths: {
      models_dir: el("cfgModelsDir").value.trim(),
      outputs_dir: el("cfgOutputsDir").value.trim(),
      tmp_dir: el("cfgTmpDir").value.trim(),
    },
    logging: {
      level: el("cfgLogLevel").value.trim() || "INFO",
    },
    huggingface: {
      token: el("cfgToken").value.trim(),
    },
    defaults: {
      text2image_model: el("cfgTextImageModel").value.trim(),
      image2image_model: el("cfgImageImageModel").value.trim(),
      text2video_model: el("cfgTextModel").value.trim(),
      image2video_model: el("cfgImageModel").value.trim(),
      num_inference_steps: Number(el("cfgSteps").value),
      duration_seconds: Number(el("cfgFrames").value),
      guidance_scale: Number(el("cfgGuidance").value),
      fps: Number(el("cfgFps").value),
      width: Number(el("cfgWidth").value),
      height: Number(el("cfgHeight").value),
    },
    storage: {
      cleanup_enabled: true,
      cleanup_max_age_days: Number(el("cfgCleanupMaxAgeDays")?.value || 7),
      cleanup_max_outputs_count: Number(el("cfgCleanupMaxOutputsCount")?.value || 200),
      cleanup_max_tmp_count: Number(el("cfgCleanupMaxTmpCount")?.value || 300),
      cleanup_max_cache_size_gb: Number(el("cfgCleanupMaxCacheGb")?.value || 30),
    },
  };
  const updated = await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  applySettings(updated);
  await Promise.all([loadSettingsLocalModels(), loadLocalModels()]);
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  const newPort = Number(updated.server?.listen_port || 0);
  const newRocmAotriton = updated.server?.rocm_aotriton_experimental !== false;
  if ((prevPort && newPort && prevPort !== newPort) || prevRocmAotriton !== newRocmAotriton) {
    if (prevPort && newPort && prevPort !== newPort) {
      showTaskMessage(t("msgPortChangeSaved"));
    } else {
      showTaskMessage(t("msgServerSettingRestartRequired"));
    }
  } else {
    showTaskMessage(t("msgSettingsSaved"));
  }
}

async function clearHfCache() {
  if (!window.confirm(t("msgConfirmClearHfCache"))) return;
  const result = await api("/api/cache/hf/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dry_run: false }),
  });
  showTaskMessage(
    t("msgHfCacheCleared", {
      removed: (result.removed_paths || []).length,
      skipped: (result.skipped || []).length,
      failed: (result.failed || []).length,
    }),
  );
}

async function runCleanupNow() {
  const includeCache = Boolean(el("cleanupIncludeCache")?.checked);
  const result = await api("/api/cleanup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_cache: includeCache }),
  });
  showTaskMessage(
    t("msgCleanupDone", {
      outputs: (result.removed_outputs || []).length,
      tmp: (result.removed_tmp || []).length,
      cache: (result.removed_cache_paths || []).length,
    }),
  );
}

async function deleteLocalModel(item, baseDir = "") {
  const name = item?.repo_hint || item?.name || "";
  if (!window.confirm(t("msgConfirmDeleteModel", { model: name }))) return;
  const modelName = String(item?.name || "").trim();
  const modelPath = String(item?.path || "").trim();
  if (!modelName && !modelPath) {
    throw new Error("local model identifier is empty");
  }
  await api("/api/models/local/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: modelName || null,
      path: modelPath || null,
      base_dir: baseDir || null,
    }),
  });
  await Promise.all([loadLocalModels(), loadSettingsLocalModels()]);
  await Promise.all([
    loadModelCatalog("text-to-image", true),
    loadModelCatalog("image-to-image", true),
    loadModelCatalog("text-to-video", true),
    loadModelCatalog("image-to-video", true),
  ]);
  showTaskMessage(t("msgModelDeleted", { model: name }));
}

function normalizeLocalCategory(item) {
  const raw = String(item?.category || "").trim().toLowerCase();
  if (raw === "lora") return "Lora";
  if (raw === "vae" || raw === "vea") return "VAE";
  if (item?.is_lora) return "Lora";
  if (item?.is_vae) return "VAE";
  return "BaseModel";
}

function canApplyLocalItem(task, item) {
  if (!task) return false;
  if (Object.prototype.hasOwnProperty.call(item || {}, "apply_supported") && item?.apply_supported === false) {
    return false;
  }
  const category = normalizeLocalCategory(item);
  if (category === "BaseModel") return true;
  if (category === "Lora") return true;
  if (category === "VAE") return task === "text-to-image" || task === "image-to-image";
  return false;
}

async function applyLocalModelToTask(task, item) {
  const repoHint = String(item?.repo_hint || item?.model_id || item?.display_name || item?.name || "").trim();
  const category = normalizeLocalCategory(item);
  if (!canApplyLocalItem(task, item)) {
    showTaskMessage(t("msgApplyUnsupportedCategory"));
    return;
  }
  if (category === "BaseModel") {
    const modelDom = getModelDom(task);
    const select = el(modelDom.selectId);
    if (!select) return;
    const catalog = state.modelCatalog[task] || [];
    if (!catalog.some((entry) => entry.value === item.path)) {
      catalog.push({
        source: "local",
        label: `[local] ${repoHint || item.path}`,
        value: item.path,
        id: repoHint || item.path,
        size_bytes: item.size_bytes || null,
        preview_url: item.preview_url || null,
        model_url: item.model_url || null,
      });
    }
    state.modelCatalog[task] = catalog;
    renderModelSelect(task, item.path);
    try {
      await loadLoraCatalog(task, false);
      if (task === "text-to-image" || task === "image-to-image") {
        await loadVaeCatalog(task, false);
      }
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
    showTaskMessage(t("msgLocalModelApplied", { task: taskShortName(task), model: repoHint || item.path }));
    return;
  }
  if (category === "Lora") {
    const dom = getLoraDom(task);
    const select = el(dom.selectId);
    if (!select) {
      showTaskMessage(t("msgApplyUnsupportedCategory"));
      return;
    }
    const catalog = state.loraCatalog[task] || [];
    if (!catalog.some((entry) => entry.value === item.path)) {
      catalog.push({
        source: "local",
        label: `[lora] ${repoHint || item.path}`,
        value: item.path,
        id: repoHint || item.path,
        base_model: item.base_name || item.base_model || null,
        size_bytes: item.size_bytes || null,
        preview_url: item.preview_url || null,
        model_url: item.model_url || null,
      });
      state.loraCatalog[task] = catalog;
    }
    const previous = getSelectedValues(dom.selectId);
    const values = Array.from(new Set([...(previous || []).filter((value) => value), item.path]));
    renderLoraSelect(task, values);
    showTaskMessage(t("msgLocalModelApplied", { task: `${taskShortName(task)}/LoRA`, model: repoHint || item.path }));
    return;
  }
  if (category === "VAE") {
    const dom = getVaeDom(task);
    const select = el(dom.selectId);
    if (!select) {
      showTaskMessage(t("msgApplyUnsupportedCategory"));
      return;
    }
    const catalog = state.vaeCatalog[task] || [];
    if (!catalog.some((entry) => entry.value === item.path)) {
      catalog.push({
        source: "local",
        label: `[vae] ${repoHint || item.path}`,
        value: item.path,
        id: repoHint || item.path,
        size_bytes: item.size_bytes || null,
        preview_url: item.preview_url || null,
        model_url: item.model_url || null,
      });
      state.vaeCatalog[task] = catalog;
    }
    renderVaeSelect(task, item.path);
    showTaskMessage(t("msgLocalModelApplied", { task: `${taskShortName(task)}/VAE`, model: repoHint || item.path }));
  }
}

function renderLocalViewMode() {
  const panel = el("panel-local-models");
  if (!panel) return;
  panel.dataset.view = state.localViewMode === "flat" ? "flat" : "tree";
}

function findLocalTreeItemByPath(treeData, targetPath) {
  const normalized = String(targetPath || "").trim().toLowerCase();
  if (!normalized) return null;
  const tasks = Array.isArray(treeData?.tasks) ? treeData.tasks : [];
  for (const task of tasks) {
    const bases = Array.isArray(task?.bases) ? task.bases : [];
    for (const base of bases) {
      const categories = Array.isArray(base?.categories) ? base.categories : [];
      for (const category of categories) {
        const items = Array.isArray(category?.items) ? category.items : [];
        for (const item of items) {
          if (String(item?.path || "").trim().toLowerCase() === normalized) {
            return item;
          }
        }
      }
    }
  }
  return null;
}

function buildFilteredLocalTree(treeData, rawQuery) {
  const query = String(rawQuery || "").trim().toLowerCase();
  const tasks = Array.isArray(treeData?.tasks) ? treeData.tasks : [];
  if (!query) return tasks;
  const filteredTasks = [];
  for (const task of tasks) {
    const bases = Array.isArray(task?.bases) ? task.bases : [];
    const filteredBases = [];
    for (const base of bases) {
      const categories = Array.isArray(base?.categories) ? base.categories : [];
      const filteredCategories = [];
      for (const category of categories) {
        const items = Array.isArray(category?.items) ? category.items : [];
        const matchedItems = items.filter((item) => {
          const target = `${item?.display_name || ""} ${item?.name || ""} ${item?.path || ""}`.toLowerCase();
          return target.includes(query);
        });
        if (matchedItems.length > 0) {
          filteredCategories.push({
            ...category,
            item_count: matchedItems.length,
            items: matchedItems,
          });
        }
      }
      if (filteredCategories.length > 0) {
        const baseCount = filteredCategories.reduce((sum, cat) => sum + Number(cat.item_count || 0), 0);
        filteredBases.push({
          ...base,
          item_count: baseCount,
          categories: filteredCategories,
        });
      }
    }
    if (filteredBases.length > 0) {
      const taskCount = filteredBases.reduce((sum, base) => sum + Number(base.item_count || 0), 0);
      filteredTasks.push({
        ...task,
        item_count: taskCount,
        bases: filteredBases,
      });
    }
  }
  return filteredTasks;
}

function snapshotLocalTreeOpenState(container) {
  if (!container) return;
  container.querySelectorAll("details[data-node-key]").forEach((node) => {
    const key = String(node.dataset.nodeKey || "").trim();
    if (!key) return;
    state.localTreeOpenState[key] = !!node.open;
  });
}

function getLocalTreeNodeOpen(key, fallbackOpen) {
  if (Object.prototype.hasOwnProperty.call(state.localTreeOpenState, key)) {
    return !!state.localTreeOpenState[key];
  }
  return !!fallbackOpen;
}


function setAllLocalTreeNodesOpen(opened) {
  const tasks = buildFilteredLocalTree(state.localTreeData, String(state.localTreeFilter || "").trim());
  for (const task of tasks) {
    const taskKey = `task:${task.task || ""}`;
    state.localTreeOpenState[taskKey] = !!opened;
    const bases = Array.isArray(task?.bases) ? task.bases : [];
    for (const base of bases) {
      const baseKey = `base:${task.task || ""}/${base.base_name || ""}`;
      state.localTreeOpenState[baseKey] = !!opened;
      const categories = Array.isArray(base?.categories) ? base.categories : [];
      for (const category of categories) {
        const categoryKey = `category:${task.task || ""}/${base.base_name || ""}/${category.category || ""}`;
        state.localTreeOpenState[categoryKey] = !!opened;
      }
    }
  }
  const container = el("localModelsTree");
  if (!container) return;
  const nodes = container.querySelectorAll("details[data-node-key]");
  if (!nodes.length) {
    renderLocalModelTree(state.localTreeData);
    return;
  }
  nodes.forEach((node) => {
    const key = String(node.dataset.nodeKey || "").trim();
    if (key) {
      state.localTreeOpenState[key] = !!opened;
    }
    node.open = !!opened;
  });
}

function renderLocalTreeSelection() {
  const container = el("localModelSelection");
  if (!container) return;
  const selected = state.localTreeSelected;
  if (!selected) {
    container.innerHTML = `<div class="local-selection-empty">${escapeHtml(t("msgSelectLocalTreeItem"))}</div>`;
    return;
  }
  const sizeText = formatModelSize(selected.size_bytes);
  const applyEnabled = canApplyLocalItem(selected.task_api, selected);
  const canDelete = Boolean(selected?.can_delete);
  container.innerHTML = `
    <div class="local-selection-name">${escapeHtml(selected.display_name || selected.name || selected.path)}</div>
    <div class="local-selection-meta">${escapeHtml(t("modelKind"))}=${escapeHtml(selected.category || "n/a")} | ${escapeHtml(t("modelBase"))}=${escapeHtml(
      selected.base_name || "n/a",
    )} | ${escapeHtml(t("modelSource"))}=${escapeHtml(selected.provider || "local")} | ${escapeHtml(t("modelSize"))}=${escapeHtml(
      sizeText,
    )} | ${escapeHtml(t("labelTask"))}=${escapeHtml(selected.task_dir || "n/a")}</div>
    <div class="local-selection-actions">
      <button type="button" id="localSelectionApplyBtn" ${applyEnabled ? "" : "disabled"}>${escapeHtml(t("msgApply"))}</button>
      <button type="button" id="localSelectionRevealBtn">${escapeHtml(t("msgReveal"))}</button>
      ${canDelete ? `<button type="button" id="localSelectionDeleteBtn">${escapeHtml(t("btnDeleteModel"))}</button>` : ""}
    </div>
  `;
  bindClick("localSelectionApplyBtn", async () => {
    if (!applyEnabled) {
      showTaskMessage(t("msgApplyUnsupportedCategory"));
      return;
    }
    await applyLocalModelToTask(selected.task_api, selected);
  });
  bindClick("localSelectionRevealBtn", async () => {
    try {
      await api("/api/models/local/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path: selected.path,
          base_dir: state.localModelsBaseDir || null,
        }),
      });
      showTaskMessage(t("msgLocalModelRevealed", { path: selected.path }));
    } catch (error) {
      showTaskMessage(t("msgLocalModelRevealFailed", { error: error.message }));
    }
  });
  bindClick("localSelectionDeleteBtn", async () => {
    try {
      await deleteLocalModel(selected, state.localModelsBaseDir || "");
    } catch (error) {
      showTaskMessage(t("msgModelDeleteFailed", { error: error.message }));
    }
  });
}

function renderLocalModelTree(treeData) {
  const container = el("localModelsTree");
  if (!container) return;
  snapshotLocalTreeOpenState(container);
  const query = String(state.localTreeFilter || "").trim();
  const tasks = buildFilteredLocalTree(treeData, query);
  if (!tasks.length) {
    container.innerHTML = `<div class="local-tree-empty">${escapeHtml(t("msgNoLocalTreeModels", { path: state.localModelsBaseDir || t("msgUnknownPath") }))}</div>`;
    renderLocalTreeSelection();
    return;
  }
  const refs = [];
  const taskHtml = tasks
    .map((task, taskIdx) => {
      const taskKey = `task:${task.task || ""}`;
      const bases = Array.isArray(task.bases) ? task.bases : [];
      const baseHtml = bases
        .map((base) => {
          const baseKey = `base:${task.task || ""}/${base.base_name || ""}`;
          const categoryHtml = (base.categories || [])
            .map((category) => {
              const categoryKey = `category:${task.task || ""}/${base.base_name || ""}/${category.category || ""}`;
              const itemHtml = (category.items || [])
                .map((item) => {
                  refs.push(item);
                  const idx = refs.length - 1;
                  const selected = state.localTreeSelected && state.localTreeSelected.path === item.path;
                  const applyEnabled = canApplyLocalItem(item.task_api, item);
                  const thumbHtml = item.preview_url
                    ? `<img class="local-tree-thumb" src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />`
                    : "";
                  return `
                    <div class="local-tree-item ${selected ? "selected" : ""}">
                      <div class="local-tree-thumb-wrap">
                        ${thumbHtml}
                        <div class="local-tree-thumb-empty" style="${item.preview_url ? "display:none;" : ""}">${escapeHtml(t("msgModelNoPreview"))}</div>
                      </div>
                      <div class="local-tree-item-main">
                        <div class="local-tree-item-name">${escapeHtml(item.display_name || item.name || item.path)}</div>
                        <div class="local-tree-item-meta">${escapeHtml(t("modelKind"))}=${escapeHtml(item.category || "n/a")} | ${escapeHtml(
                          t("modelBase"),
                        )}=${escapeHtml(item.base_name || "n/a")} | ${escapeHtml(t("modelSource"))}=${escapeHtml(
                          item.provider || "local",
                        )} | ${escapeHtml(t("modelSize"))}=${escapeHtml(formatModelSize(item.size_bytes))}</div>
                      </div>
                       <div class="local-tree-item-actions">
                         <button type="button" class="local-tree-select-btn" data-index="${idx}">${escapeHtml(t("msgDetail"))}</button>
                         <button type="button" class="local-tree-apply-btn" data-index="${idx}" ${applyEnabled ? "" : "disabled"}>${escapeHtml(
                           t("msgApply"),
                         )}</button>
                         <button type="button" class="local-tree-reveal-btn" data-index="${idx}">${escapeHtml(t("msgReveal"))}</button>
                         ${item.can_delete ? `<button type="button" class="local-tree-delete-btn" data-index="${idx}">${escapeHtml(t("btnDeleteModel"))}</button>` : ""}
                       </div>
                     </div>
                   `;
                })
                .join("");
              return `
                <details data-node-key="${escapeHtml(categoryKey)}" ${query || getLocalTreeNodeOpen(categoryKey, false) ? "open" : ""}>
                  <summary class="local-tree-summary">
                    <div class="local-tree-label">
                      <span class="local-tree-title">${escapeHtml(category.category)}</span>
                    </div>
                    <span class="local-tree-badge">${escapeHtml(category.item_count)}</span>
                  </summary>
                  <div class="local-tree-children">${itemHtml}</div>
                </details>
              `;
            })
            .join("");
          return `
            <details data-node-key="${escapeHtml(baseKey)}" ${query || getLocalTreeNodeOpen(baseKey, false) ? "open" : ""}>
              <summary class="local-tree-summary">
                <div class="local-tree-label">
                  <span class="local-tree-title">${escapeHtml(base.base_name)}</span>
                </div>
                <span class="local-tree-badge">${escapeHtml(base.item_count)}</span>
              </summary>
              <div class="local-tree-children">${categoryHtml}</div>
            </details>
          `;
        })
        .join("");
      return `
        <details data-node-key="${escapeHtml(taskKey)}" ${query || getLocalTreeNodeOpen(taskKey, taskIdx === 0) ? "open" : ""}>
          <summary class="local-tree-summary">
            <div class="local-tree-label">
              <span class="local-tree-title">${escapeHtml(task.task)}</span>
            </div>
            <span class="local-tree-badge">${escapeHtml(task.item_count)}</span>
          </summary>
          <div class="local-tree-children">${baseHtml}</div>
        </details>
      `;
    })
    .join("");
  container.innerHTML = taskHtml;
  container.querySelectorAll("details[data-node-key]").forEach((node) => {
    node.addEventListener("toggle", () => {
      const key = String(node.dataset.nodeKey || "").trim();
      if (!key) return;
      state.localTreeOpenState[key] = !!node.open;
    });
  });
  container.querySelectorAll(".local-tree-select-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= refs.length) return;
      state.localTreeSelected = refs[idx];
      renderLocalModelTree(treeData);
      renderLocalTreeSelection();
      openLocalModelDetail(refs[idx]);
    });
  });
  container.querySelectorAll(".local-tree-apply-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= refs.length) return;
      const item = refs[idx];
      if (!canApplyLocalItem(item.task_api, item)) {
        showTaskMessage(t("msgApplyUnsupportedCategory"));
        return;
      }
      await applyLocalModelToTask(item.task_api, item);
    });
  });
  container.querySelectorAll(".local-tree-reveal-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= refs.length) return;
      const item = refs[idx];
      try {
        await api("/api/models/local/reveal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path: item.path,
            base_dir: state.localModelsBaseDir || null,
          }),
        });
        showTaskMessage(t("msgLocalModelRevealed", { path: item.path }));
      } catch (error) {
        showTaskMessage(t("msgLocalModelRevealFailed", { error: error.message }));
      }
    });
  });
  container.querySelectorAll(".local-tree-delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= refs.length) return;
      const item = refs[idx];
      try {
        await deleteLocalModel(item, state.localModelsBaseDir || "");
      } catch (error) {
        showTaskMessage(t("msgModelDeleteFailed", { error: error.message }));
      }
    });
  });
  renderLocalTreeSelection();
}

function applyLocalTreePayload(data, dir = "") {
  const selectedPath = state.localTreeSelected?.path || "";
  state.localTreeData = data || null;
  state.localModels = data?.flat_items || [];
  state.localModelsBaseDir = data?.model_root || dir || "";
  state.localTreeSelected = findLocalTreeItemByPath(state.localTreeData, selectedPath);
  renderLocalLineageOptions(state.localModels);
  renderLocalViewMode();
  renderLocalModels(state.localModels, state.localModelsBaseDir);
  renderLocalModelTree(state.localTreeData);
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
  }
}

function renderLocalModels(items, baseDir = "") {
  const container = el("localModels");
  const activeLineage = state.localLineageFilter || "all";
  const filteredItems =
    activeLineage === "all" ? [...(items || [])] : (items || []).filter((item) => inferLocalLineage(item) === activeLineage);
  if (!filteredItems.length) {
    container.innerHTML = `<p>${t("msgNoLocalModels", { path: baseDir || t("msgUnknownPath") })}</p>`;
    return;
  }
  const taskShortLabel = {
    "text-to-image": "T2I",
    "image-to-image": "I2I",
    "text-to-video": "T2V",
    "image-to-video": "I2V",
  };
  container.innerHTML = filteredItems
    .map(
      (item, index) => `
      <div class="row model-row">
        <div class="model-main">
          ${
            item.preview_url
              ? `<img class="model-preview" src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`
              : ""
          }
          <div>
            <strong>${escapeHtml(item.repo_hint)}</strong>
            <span>${escapeHtml(t("modelKind"))}=${escapeHtml(localModelKind(item))} | ${escapeHtml(t("modelTag"))}=${escapeHtml(item.class_name || "n/a")} | ${escapeHtml(t("modelSource"))}=local</span>
            <span>lineage=${escapeHtml(inferLocalLineage(item))} | ${escapeHtml(t("modelBase"))}=${escapeHtml(item.base_model || "n/a")}</span>
          </div>
        </div>
        <div class="local-actions">
          ${(item.compatible_tasks || [])
            .map(
              (task) =>
                `<button type="button" class="local-apply-btn" data-index="${index}" data-task="${escapeHtml(task)}">${escapeHtml(
                  t("msgSetTaskModel", { task: taskShortLabel[task] || task }),
                )}</button>`,
            )
            .join("")}
          ${item.can_delete ? `<button type="button" class="local-delete-btn" data-index="${index}">${t("btnDeleteModel")}</button>` : ""}
        </div>
      </div>`,
    )
    .join("");
  container.querySelectorAll(".local-apply-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      const task = button.dataset.task || "";
      if (!Number.isInteger(index) || index < 0 || index >= filteredItems.length) return;
      const item = filteredItems[index];
      await applyLocalModelToTask(task, item);
    });
  });
  container.querySelectorAll(".local-delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      if (!Number.isInteger(index) || index < 0 || index >= filteredItems.length) return;
      try {
        await deleteLocalModel(filteredItems[index], baseDir);
      } catch (error) {
        showTaskMessage(t("msgModelDeleteFailed", { error: error.message }));
      }
    });
  });
}

async function loadLocalModels(options = {}) {
  const forceRescan = Boolean(options?.forceRescan);
  const dir = el("localModelsDir")?.value?.trim() || "";
  const params = new URLSearchParams();
  if (dir) params.set("dir", dir);
  if (el("localModelsTree")) {
    el("localModelsTree").innerHTML = `<div class="loading-inline">${escapeHtml(t("runtimeLoading"))}</div>`;
  }
  if (el("localModels")) {
    el("localModels").innerHTML = `<div class="loading-inline">${escapeHtml(t("runtimeLoading"))}</div>`;
  }
  try {
    if (forceRescan) {
      const tree = await api("/api/models/local/rescan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dir: dir || null }),
      });
      applyLocalTreePayload(tree, dir);
      return;
    }
    const treeUrl = params.toString() ? `/api/models/local/tree?${params.toString()}` : "/api/models/local/tree";
    const tree = await api(treeUrl);
    let mergedTree = tree;
    if (!Array.isArray(tree?.flat_items) || tree.flat_items.length === 0) {
      const flatUrl = params.toString() ? `/api/models/local?${params.toString()}` : "/api/models/local";
      const legacy = await api(flatUrl);
      mergedTree = {
        ...(tree || {}),
        model_root: tree?.model_root || legacy.base_dir || dir || "",
        flat_items: legacy.items || [],
      };
    }
    applyLocalTreePayload(mergedTree, dir);
  } catch (treeError) {
    const flatUrl = params.toString() ? `/api/models/local?${params.toString()}` : "/api/models/local";
    const data = await api(flatUrl);
    state.localModels = data.items || [];
    state.localModelsBaseDir = data.base_dir || dir || "";
    state.localTreeData = null;
    state.localTreeSelected = null;
    renderLocalLineageOptions(state.localModels);
    renderLocalViewMode();
    renderLocalModels(state.localModels, state.localModelsBaseDir);
    renderLocalModelTree(state.localTreeData);
    renderLocalTreeSelection();
  }
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
  }
}

async function rescanLocalModels() {
  await loadLocalModels({ forceRescan: true });
  showTaskMessage(t("msgLocalRescanned"));
}

function renderOutputs(items, baseDir = "") {
  const pathNode = el("outputsPath");
  if (pathNode) {
    pathNode.textContent = baseDir || "";
  }
  const container = el("outputsList");
  if (!container) return;
  if (!items.length) {
    container.innerHTML = `<p>${t("msgNoOutputs", { path: baseDir || t("msgUnknownPath") })}</p>`;
    return;
  }
  container.innerHTML = items
    .map((item, index) => {
      const viewUrl = item.view_url || "";
      let previewHtml = "";
      if (item.kind === "image" && viewUrl) {
        previewHtml = `<img class="model-preview" src="${escapeHtml(viewUrl)}" alt="${escapeHtml(item.name || "output")}" loading="lazy" onerror="this.style.display='none'" />`;
      } else if (item.kind === "video" && viewUrl) {
        previewHtml = `<video class="model-preview" src="${escapeHtml(viewUrl)}" preload="metadata" muted playsinline></video>`;
      }
      const openLink = viewUrl ? `<a href="${escapeHtml(viewUrl)}" target="_blank" rel="noopener noreferrer">${t("msgOpen")}</a>` : "";
      return `
      <div class="row model-row">
        <div class="model-main">
          ${previewHtml}
          <div>
            <strong>${escapeHtml(item.name || "")}</strong>
            <span>${escapeHtml(t("modelSize"))}=${escapeHtml(formatModelSize(item.size_bytes))} | ${escapeHtml(t("outputUpdated"))}=${escapeHtml(formatDateTime(item.updated_at))} | ${escapeHtml(t("modelTag"))}=${escapeHtml(outputTypeLabel(item.kind))}</span>
          </div>
        </div>
        <div class="local-actions">
          ${openLink}
          <button type="button" class="output-delete-btn" data-index="${index}">${t("btnDeleteModel")}</button>
        </div>
      </div>`;
    })
    .join("");
  container.querySelectorAll(".output-delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.index || "-1");
      if (!Number.isInteger(index) || index < 0 || index >= items.length) return;
      try {
        await deleteOutput(items[index]);
      } catch (error) {
        showTaskMessage(t("msgOutputDeleteFailed", { error: error.message }));
      }
    });
  });
}

async function loadOutputs() {
  const data = await api("/api/outputs?limit=500");
  state.outputs = data.items || [];
  state.outputsBaseDir = data.base_dir || "";
  renderOutputs(state.outputs, state.outputsBaseDir);
}

async function deleteOutput(item) {
  const name = String(item?.name || "").trim();
  if (!name) return;
  if (!window.confirm(t("msgConfirmDeleteOutput", { name }))) return;
  await api("/api/outputs/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_name: name }),
  });
  await loadOutputs();
  showTaskMessage(t("msgOutputDeleted", { name }));
}

function searchItemInstalled(item) {
  if (item?.installed === true) return true;
  const installed = getInstalledModelIdSet();
  return installed.has(normalizeModelId(item?.id));
}

function renderSearchPagination(pageInfo = null) {
  const prevBtn = el("searchPrevBtn");
  const nextBtn = el("searchNextBtn");
  const label = el("searchPageInfo");
  const page = Number(pageInfo?.page || state.searchPage || 1);
  if (label) {
    label.textContent = t("msgSearchPage", { page });
  }
  if (prevBtn) prevBtn.disabled = !(pageInfo?.has_prev || state.searchPrevCursor);
  if (nextBtn) nextBtn.disabled = !(pageInfo?.has_next || state.searchNextCursor);
}

function selectedDetailDownloadOptions(item) {
  const detail = state.searchDetail;
  if (!detail || normalizeModelId(detail.item?.id) !== normalizeModelId(item?.id)) {
    return {};
  }
  if (detail.item.source === "huggingface") {
    const revisionInput = el("detailHfRevisionInput");
    const revisionSelect = el("detailHfRevision");
    const revision = (revisionInput?.value || revisionSelect?.value || "main").trim() || "main";
    return {
      source: "huggingface",
      hf_revision: revision,
    };
  }
  if (detail.item.source === "civitai") {
    const modelId = parseCivitaiId(detail.item.id);
    const versionId = Number(el("detailVersionSelect")?.value || "");
    const fileId = Number(el("detailFileSelect")?.value || "");
    return {
      source: "civitai",
      civitai_model_id: modelId || null,
      civitai_version_id: Number.isInteger(versionId) && versionId > 0 ? versionId : null,
      civitai_file_id: Number.isInteger(fileId) && fileId > 0 ? fileId : null,
    };
  }
  return {};
}

function buildDetailFiles(version) {
  const files = Array.isArray(version?.files) ? version.files : [];
  if (!files.length) return `<option value="">-</option>`;
  return files
    .map((file) => `<option value="${escapeHtml(file.id)}">${escapeHtml(file.name)} (${escapeHtml(formatModelSize(file.size))})</option>`)
    .join("");
}

function renderModelDetail(item, detail) {
  const titleNode = el("modelDetailModalTitle");
  const content = el("modelDetailModalContent");
  if (!content) return;
  const previews = Array.isArray(detail.previews) ? detail.previews.filter(Boolean) : [];
  const tags = Array.isArray(detail.tags) ? detail.tags : [];
  const versions = Array.isArray(detail.versions) ? detail.versions : [];
  const defaultVersionId = detail.default_version_id != null ? String(detail.default_version_id) : "";
  let selectedVersion = versions.find((v) => String(v.id) === defaultVersionId) || versions[0] || null;
  const hfRevision = selectedVersion?.name || "main";
  const description = String(detail.description || "").trim();
  const sourceUrl = item.model_url || detail.model_url || "#";
  const modelId = String(item.id || detail.id || "");
  if (titleNode) {
    titleNode.textContent = detail.title || item.title || item.id || t("msgModelDetailEmpty");
  }
  content.innerHTML = `
    <div class="model-detail-head">
      <h4 class="model-detail-title">${escapeHtml(detail.title || item.title || item.id || "")}</h4>
      <div class="model-detail-id">${escapeHtml(modelId)}</div>
      <div class="model-detail-meta">${escapeHtml(t("modelSource"))}: ${escapeHtml(item.source || detail.source || "")}</div>
    </div>
    <div class="model-detail-actions">
      <button id="detailOpenBtn" type="button">${escapeHtml(t("msgOpen"))}</button>
      <button id="detailDownloadBtn" type="button">${escapeHtml(t("btnDownload"))}</button>
    </div>
    <div>
      <strong>${escapeHtml(t("msgDetailDescription"))}</strong>
      <div class="model-detail-text">${escapeHtml(description || "-")}</div>
    </div>
    <div>
      <strong>${escapeHtml(t("msgDetailTags"))}</strong>
      <div class="model-detail-tags">${tags.slice(0, 30).map((tag) => `<span class="model-detail-tag">${escapeHtml(tag)}</span>`).join("") || "-"}</div>
    </div>
    <div>
      <strong>${escapeHtml(t("msgModelPreviewAlt"))}</strong>
      <div class="model-detail-gallery">${
        previews.length
          ? previews
              .slice(0, 12)
              .map((url) => `<img src="${escapeHtml(url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'" />`)
              .join("")
          : `<div class="model-detail-gallery-empty">${escapeHtml(t("msgModelNoPreview"))}</div>`
      }</div>
    </div>
    <div class="model-detail-grid">
      ${
        item.source === "huggingface"
          ? `
      <label>
        <span>${escapeHtml(t("msgDetailRevision"))}</span>
        <select id="detailHfRevision">${versions
          .map((version) => `<option value="${escapeHtml(version.name)}">${escapeHtml(version.name)}</option>`)
          .join("")}</select>
      </label>
      <label>
        <span>${escapeHtml(t("msgDetailRevision"))} (manual)</span>
        <input id="detailHfRevisionInput" value="${escapeHtml(hfRevision)}" />
      </label>
      `
          : `
      <label>
        <span>${escapeHtml(t("msgDetailVersions"))}</span>
        <select id="detailVersionSelect">${versions
          .map((version) => `<option value="${escapeHtml(version.id)}">${escapeHtml(version.name || version.id)}</option>`)
          .join("")}</select>
      </label>
      <label>
        <span>${escapeHtml(t("msgDetailFiles"))}</span>
        <select id="detailFileSelect">${buildDetailFiles(selectedVersion)}</select>
      </label>
      `
      }
    </div>
  `;

  bindClick("detailOpenBtn", async () => {
    openExternalUrl(sourceUrl);
  });
  if (item.source === "huggingface") {
    const revisionSelect = el("detailHfRevision");
    const revisionInput = el("detailHfRevisionInput");
    if (revisionSelect) {
      revisionSelect.value = hfRevision;
      revisionSelect.addEventListener("change", () => {
        if (revisionInput) revisionInput.value = revisionSelect.value;
      });
    }
  } else {
    const versionSelect = el("detailVersionSelect");
    const fileSelect = el("detailFileSelect");
    if (versionSelect) {
      versionSelect.value = selectedVersion ? String(selectedVersion.id) : "";
      versionSelect.addEventListener("change", () => {
        const nextVersion = versions.find((version) => String(version.id) === versionSelect.value) || null;
        selectedVersion = nextVersion;
        if (fileSelect) {
          fileSelect.innerHTML = buildDetailFiles(nextVersion);
        }
      });
    }
  }
  bindClick("detailDownloadBtn", async () => {
    await startModelDownload(item, selectedDetailDownloadOptions(item));
  });
  openModelDetailModal();
}

async function openModelDetail(item) {
  const titleNode = el("modelDetailModalTitle");
  const content = el("modelDetailModalContent");
  if (titleNode) titleNode.textContent = t("msgModelDetailLoading");
  if (content) content.innerHTML = `<div class="loading-inline">${escapeHtml(t("msgModelDetailLoading"))}</div>`;
  openModelDetailModal();
  const source = item?.source === "civitai" ? "civitai" : "huggingface";
  try {
    const params = new URLSearchParams({
      source,
      id: String(item.id || ""),
    });
    const detail = await withApiState(async () => api(`/api/models/detail?${params.toString()}`));
    state.searchDetail = { item, detail };
    renderModelDetail(item, detail);
  } catch (error) {
    state.searchDetail = null;
    if (titleNode) titleNode.textContent = t("msgModelDetailEmpty");
    if (content) content.innerHTML = `<div class="downloads-empty">${escapeHtml(t("msgModelDetailLoadFailed", { error: error.message }))}</div>`;
  }
}

function openLocalModelDetail(item) {
  if (!item) return;
  state.localTreeSelected = item;
  renderLocalTreeSelection();
  renderLocalModelTree(state.localTreeData);
  const titleNode = el("modelDetailModalTitle");
  const content = el("modelDetailModalContent");
  if (titleNode) {
    titleNode.textContent = item.display_name || item.name || item.path || t("msgModelDetailEmpty");
  }
  if (!content) {
    openModelDetailModal();
    return;
  }
  const category = normalizeLocalCategory(item);
  const sizeText = formatModelSize(item.size_bytes);
  const previewHtml = item.preview_url
    ? `<img src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />`
    : "";
  const applyEnabled = canApplyLocalItem(item.task_api, item);
  const canDelete = Boolean(item?.can_delete);
  content.innerHTML = `
    <div class="model-detail-head">
      <h4 class="model-detail-title">${escapeHtml(item.display_name || item.name || item.path || "-")}</h4>
      <div class="model-detail-id">${escapeHtml(item.path || "-")}</div>
      <div class="model-detail-meta">${escapeHtml(t("modelKind"))}: ${escapeHtml(category)} | ${escapeHtml(t("labelTask"))}: ${escapeHtml(
        item.task_dir || "n/a",
      )}</div>
    </div>
    <div class="model-detail-actions">
      <button id="localDetailApplyBtn" type="button" ${applyEnabled ? "" : "disabled"}>${escapeHtml(t("msgApply"))}</button>
      <button id="localDetailRevealBtn" type="button">${escapeHtml(t("msgReveal"))}</button>
      ${canDelete ? `<button id="localDetailDeleteBtn" type="button">${escapeHtml(t("btnDeleteModel"))}</button>` : ""}
    </div>
    <div>
      <strong>${escapeHtml(t("msgModelPreviewAlt"))}</strong>
      <div class="model-detail-gallery local-detail-gallery">
        ${previewHtml}
        <div class="model-detail-gallery-empty" style="${item.preview_url ? "display:none;" : ""}">${escapeHtml(t("msgModelNoPreview"))}</div>
      </div>
    </div>
    <div class="model-detail-text">${escapeHtml(t("modelBase"))}: ${escapeHtml(item.base_name || "n/a")} | ${escapeHtml(t("modelSource"))}: ${escapeHtml(
      item.provider || "local",
    )} | ${escapeHtml(t("modelSize"))}: ${escapeHtml(sizeText)}</div>
  `;
  bindClick("localDetailApplyBtn", async () => {
    if (!applyEnabled) {
      showTaskMessage(t("msgApplyUnsupportedCategory"));
      return;
    }
    await applyLocalModelToTask(item.task_api, item);
  });
  bindClick("localDetailRevealBtn", async () => {
    try {
      await api("/api/models/local/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path: item.path,
          base_dir: state.localModelsBaseDir || null,
        }),
      });
      showTaskMessage(t("msgLocalModelRevealed", { path: item.path }));
    } catch (error) {
      showTaskMessage(t("msgLocalModelRevealFailed", { error: error.message }));
    }
  });
  bindClick("localDetailDeleteBtn", async () => {
    try {
      await deleteLocalModel(item, state.localModelsBaseDir || "");
      closeModelDetailModal();
    } catch (error) {
      showTaskMessage(t("msgModelDeleteFailed", { error: error.message }));
    }
  });
  openModelDetailModal();
}

async function applySearchResultModel(item) {
  const task = el("searchTask").value;
  const modelDom = getModelDom(task);
  const select = el(modelDom.selectId);
  if (!select) return;
  const catalog = state.modelCatalog[task] || [];
  const value = String(item.id || "").trim();
  if (!value) return;
  if (!catalog.some((entry) => entry.value === value)) {
    catalog.push({
      source: item.source || "remote",
      label: `[${item.source || "remote"}] ${item.id}`,
      value,
      id: item.id,
      size_bytes: item.size_bytes || null,
      preview_url: item.preview_url || null,
      model_url: item.model_url || null,
    });
  }
  state.modelCatalog[task] = catalog;
  renderModelSelect(task, value);
  try {
    await loadLoraCatalog(task, false);
    if (task === "text-to-image" || task === "image-to-image") {
      await loadVaeCatalog(task, false);
    }
  } catch (error) {
    showTaskMessage(t("msgSearchFailed", { error: error.message }));
  }
  showTaskMessage(t("msgSearchModelApplied", { task: taskShortName(task), model: item.id }));
}

function renderSearchResults(items) {
  const cards = el("searchCards");
  const viewMode = (el("searchViewMode")?.value || "grid").trim();
  state.searchViewMode = viewMode === "list" ? "list" : "grid";
  if (!cards) return;
  cards.classList.toggle("list-mode", state.searchViewMode === "list");
  if (!items.length) {
    cards.innerHTML = `<p>${escapeHtml(t("msgNoModelsFound"))}</p>`;
    renderSearchPagination({ page: state.searchPage, has_prev: false, has_next: false });
    return;
  }

  cards.innerHTML = items
    .map((item, index) => {
      const installed = searchItemInstalled(item);
      const supportsDownload = item.download_supported !== false;
      const preview = `
        <div class="model-card-cover-wrap">
          ${
            item.preview_url
              ? `<img class="model-card-cover" src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(t("msgModelPreviewAlt"))}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />`
              : ""
          }
          <div class="model-card-cover-empty" style="${item.preview_url ? "display:none;" : ""}">${escapeHtml(t("msgModelNoPreview"))}</div>
        </div>
      `;
      const statusBadge = installed
        ? `<span class="model-status-badge downloaded">${escapeHtml(t("msgModelInstalled"))}</span>`
        : `<span class="model-status-badge">${escapeHtml(t("msgModelNotInstalled"))}</span>`;
      return `
        <article class="model-card ${state.searchViewMode === "list" ? "list-mode" : ""}" data-index="${index}">
          ${preview}
          <div class="model-card-body">
            <h4 class="model-card-title">${escapeHtml(item.title || item.name || item.id || "-")}</h4>
            <div class="model-card-id">${escapeHtml(item.id || "-")}</div>
            <div class="model-meta-line">${escapeHtml(t("modelSource"))}: ${escapeHtml(item.source || "unknown")} | ${escapeHtml(t("modelBase"))}: ${escapeHtml(
              item.base_model || "n/a",
            )}</div>
            <div class="model-meta-line">${escapeHtml(t("modelDownloads"))}: ${escapeHtml(item.downloads ?? "n/a")} | ${escapeHtml(
              t("modelLikes"),
            )}: ${escapeHtml(item.likes ?? "n/a")} | ${escapeHtml(t("modelSize"))}: ${escapeHtml(formatModelSize(item.size_bytes))}</div>
            <div>${statusBadge}</div>
            <div class="model-card-actions">
              <button type="button" class="search-open-btn" data-index="${index}">${escapeHtml(t("msgOpen"))}</button>
              <button type="button" class="search-detail-btn" data-index="${index}">${escapeHtml(t("msgDetail"))}</button>
              <button type="button" class="search-download-btn" data-index="${index}" ${!supportsDownload || installed ? "disabled" : ""}>${escapeHtml(
                t("btnDownload"),
              )}</button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  cards.querySelectorAll(".search-open-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      openExternalUrl(items[idx]?.model_url || "");
    });
  });
  cards.querySelectorAll(".search-detail-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      await openModelDetail(items[idx]);
    });
  });
  cards.querySelectorAll(".search-download-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const idx = Number(button.dataset.index || "-1");
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return;
      const item = items[idx];
      await startModelDownload(item, selectedDetailDownloadOptions(item));
    });
  });
}

async function searchModels(event, options = {}) {
  if (event) event.preventDefault();
  const resetPage = options.resetPage !== false;
  if (resetPage) {
    state.searchPage = 1;
  } else if (Number.isInteger(options.page) && options.page > 0) {
    state.searchPage = options.page;
  }
  const rawLimit = Number(el("searchLimit").value || "30");
  const limit = Math.min(100, Math.max(1, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 30));
  el("searchLimit").value = String(limit);
  const baseModel = (el("searchBaseModel")?.value || "all").trim();
  const sizeMinGb = readNum("searchSizeMinMb");
  const sizeMaxGb = readNum("searchSizeMaxMb");
  if (sizeMinGb !== null && sizeMaxGb !== null && sizeMaxGb < sizeMinGb) {
    throw new Error(t("msgSearchSizeRangeInvalid"));
  }
  if (el("searchCards")) {
    el("searchCards").innerHTML = `<div class="loading-inline">${escapeHtml(t("msgModelSearchLoading"))}</div>`;
  }
  const sizeMinMb = sizeMinGb !== null && sizeMinGb > 0 ? sizeMinGb * 1024 : null;
  const sizeMaxMb = sizeMaxGb !== null && sizeMaxGb > 0 ? sizeMaxGb * 1024 : null;
  const params = new URLSearchParams({
    task: el("searchTask").value,
    source: el("searchSource").value || "all",
    query: el("searchQuery").value.trim(),
    limit: String(limit),
    page: String(state.searchPage),
    sort: el("searchSort")?.value || "downloads",
    nsfw: el("searchNsfw")?.value || "exclude",
    model_kind: (el("searchModelKind")?.value || "").trim(),
  });
  if (baseModel && baseModel !== "all") {
    params.set("base_model", baseModel);
  }
  if (sizeMinMb !== null && sizeMinMb > 0) {
    params.set("size_min_mb", String(sizeMinMb));
  }
  if (sizeMaxMb !== null && sizeMaxMb > 0) {
    params.set("size_max_mb", String(sizeMaxMb));
  }
  const data = await withApiState(
    async () => api(`/api/models/search2?${params.toString()}`),
    {
      onError: (error) => {
        if (el("searchCards")) {
          el("searchCards").innerHTML = `<div class="downloads-empty">${escapeHtml(error.message)}</div>`;
        }
      },
    },
  );
  state.lastSearchResults = data.items || [];
  state.searchNextCursor = data.next_cursor || null;
  state.searchPrevCursor = data.prev_cursor || null;
  state.searchPage = Number(data.page_info?.page || state.searchPage || 1);
  state.searchDetail = null;
  closeModelDetailModal();
  if (el("modelDetailModalTitle")) el("modelDetailModalTitle").textContent = t("msgModelDetailEmpty");
  if (el("modelDetailModalContent")) el("modelDetailModalContent").innerHTML = "";
  renderSearchResults(state.lastSearchResults);
  renderSearchPagination(data.page_info || null);
  const note = el("searchFilterNote");
  if (note) {
    const minText = sizeMinGb !== null && sizeMinGb > 0 ? String(sizeMinGb) : "0";
    const maxText = sizeMaxGb !== null && sizeMaxGb > 0 ? String(sizeMaxGb) : "inf";
    note.textContent = `${t("modelSource")}: ${params.get("source")} | ${t("labelSearchSort")}: ${params.get("sort")} | ${t("labelLimit")}: ${limit} | ${t("modelSize")}: ${minText}..${maxText} GB`;
  }
}

async function startModelDownload(repoOrItem, extra = {}) {
  const item = repoOrItem && typeof repoOrItem === "object" ? repoOrItem : null;
  const repoId = String(item?.id || repoOrItem || "").trim();
  if (!repoId) return;
  const targetDir = el("downloadTargetDir").value.trim();
  if (targetDir && el("localModelsDir") && el("localModelsDir").value.trim() !== targetDir) {
    el("localModelsDir").value = targetDir;
  }
  const searchTask = (el("searchTask")?.value || "").trim();
  const requestedTask = String(extra.task || item?.task || searchTask || "").trim();
  const requestedBaseModel = String(extra.base_model || item?.base_model || "").trim();
  const requestedModelKind = String(extra.model_kind || item?.type || "").trim();
  const data = await api("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: repoId,
      source: extra.source || null,
      hf_revision: extra.hf_revision || null,
      civitai_model_id: extra.civitai_model_id || null,
      civitai_version_id: extra.civitai_version_id || null,
      civitai_file_id: extra.civitai_file_id || null,
      target_dir: targetDir || null,
      task: requestedTask || null,
      base_model: requestedBaseModel || null,
      model_kind: requestedModelKind || null,
    }),
  });
  showTaskMessage(
    t("msgModelDownloadStarted", {
      repo: repoId,
      path: targetDir || t("msgDefaultModelsDir"),
    }),
  );
  try {
    await refreshDownloadTasks();
  } catch (error) {
    showTaskMessage(t("msgDownloadsRefreshFailed", { error: error.message }));
  }
}

async function generateText2Video(event) {
  event.preventDefault();
  setGenerationBusy(true);
  const selectedModel = el("t2vModelSelect").value.trim();
  const validationModel = selectedModel || state.defaultModels["text-to-video"] || "";
  assertVideoModelUsable("text-to-video", validationModel);
  const loraIds = getSelectedValues("t2vLoraSelect");
  const payload = {
    prompt: el("t2vPrompt").value.trim(),
    negative_prompt: el("t2vNegative").value.trim(),
    model_id: selectedModel || null,
    lora_id: loraIds[0] || null,
    lora_ids: loraIds,
    lora_scale: Number(el("t2vLoraScale").value),
    backend: (el("t2vBackendSelect")?.value || "auto").trim(),
    num_inference_steps: Number(el("t2vSteps").value),
    duration_seconds: Number(el("t2vFrames").value),
    guidance_scale: Number(el("t2vGuidance").value),
    fps: Number(el("t2vFps").value),
    seed: readNum("t2vSeed"),
  };
  try {
    const data = await api("/api/generate/text2video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showTaskMessage(t("msgTextGenerationStarted", { id: data.task_id }));
    trackTask(data.task_id);
  } catch (error) {
    setGenerationBusy(false);
    throw error;
  }
}

async function generateImage2Video(event) {
  event.preventDefault();
  const imageFile = el("i2vImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
  setGenerationBusy(true);
  const selectedModel = el("i2vModelSelect").value.trim();
  const validationModel = selectedModel || state.defaultModels["image-to-video"] || "";
  assertVideoModelUsable("image-to-video", validationModel);
  const formData = new FormData();
  const loraIds = getSelectedValues("i2vLoraSelect");
  formData.append("image", imageFile);
  formData.append("prompt", el("i2vPrompt").value.trim());
  formData.append("negative_prompt", el("i2vNegative").value.trim());
  formData.append("model_id", selectedModel);
  formData.append("lora_id", loraIds[0] || "");
  loraIds.forEach((value) => formData.append("lora_ids", value));
  formData.append("lora_scale", String(Number(el("i2vLoraScale").value)));
  formData.append("num_inference_steps", String(Number(el("i2vSteps").value)));
  formData.append("duration_seconds", String(Number(el("i2vFrames").value)));
  formData.append("guidance_scale", String(Number(el("i2vGuidance").value)));
  formData.append("fps", String(Number(el("i2vFps").value)));
  formData.append("width", String(Number(el("i2vWidth").value)));
  formData.append("height", String(Number(el("i2vHeight").value)));
  if (el("i2vSeed").value.trim()) formData.append("seed", el("i2vSeed").value.trim());

  try {
    const data = await api("/api/generate/image2video", {
      method: "POST",
      body: formData,
    });
    showTaskMessage(t("msgImageGenerationStarted", { id: data.task_id }));
    trackTask(data.task_id);
  } catch (error) {
    setGenerationBusy(false);
    throw error;
  }
}

async function generateText2Image(event) {
  event.preventDefault();
  setGenerationBusy(true);
  const selectedModel = el("t2iModelSelect").value.trim();
  const loraIds = getSelectedValues("t2iLoraSelect");
  const payload = {
    prompt: el("t2iPrompt").value.trim(),
    negative_prompt: el("t2iNegative").value.trim(),
    model_id: selectedModel || null,
    lora_id: loraIds[0] || null,
    lora_ids: loraIds,
    lora_scale: Number(el("t2iLoraScale").value),
    vae_id: el("t2iVaeSelect").value.trim() || null,
    num_inference_steps: Number(el("t2iSteps").value),
    guidance_scale: Number(el("t2iGuidance").value),
    width: Number(el("t2iWidth").value),
    height: Number(el("t2iHeight").value),
    seed: readNum("t2iSeed"),
  };
  try {
    const data = await api("/api/generate/text2image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showTaskMessage(t("msgTextImageGenerationStarted", { id: data.task_id }));
    trackTask(data.task_id);
  } catch (error) {
    setGenerationBusy(false);
    throw error;
  }
}

async function generateImage2Image(event) {
  event.preventDefault();
  const imageFile = el("i2iImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
  setGenerationBusy(true);
  const formData = new FormData();
  const loraIds = getSelectedValues("i2iLoraSelect");
  formData.append("image", imageFile);
  formData.append("prompt", el("i2iPrompt").value.trim());
  formData.append("negative_prompt", el("i2iNegative").value.trim());
  formData.append("model_id", el("i2iModelSelect").value.trim());
  formData.append("lora_id", loraIds[0] || "");
  loraIds.forEach((value) => formData.append("lora_ids", value));
  formData.append("lora_scale", String(Number(el("i2iLoraScale").value)));
  formData.append("vae_id", el("i2iVaeSelect").value.trim());
  formData.append("num_inference_steps", String(Number(el("i2iSteps").value)));
  formData.append("guidance_scale", String(Number(el("i2iGuidance").value)));
  formData.append("strength", String(Number(el("i2iStrength").value)));
  formData.append("width", String(Number(el("i2iWidth").value)));
  formData.append("height", String(Number(el("i2iHeight").value)));
  if (el("i2iSeed").value.trim()) formData.append("seed", el("i2iSeed").value.trim());
  try {
    const data = await api("/api/generate/image2image", {
      method: "POST",
      body: formData,
    });
    showTaskMessage(t("msgImageImageGenerationStarted", { id: data.task_id }));
    trackTask(data.task_id);
  } catch (error) {
    setGenerationBusy(false);
    throw error;
  }
}

function renderTask(task) {
  if (!task) {
    state.currentTaskSnapshot = null;
    renderTaskStep(null);
    setGenerationBusy(false);
    return;
  }
  state.currentTaskSnapshot = task || null;
  renderTaskStep(task);
  renderTaskProgress(task);
  showTaskErrorPopup(task);
  const suppressDownloadProgressText =
    task.task_type === "download" && (state.activeTab === "models" || state.activeTab === "local-models");
  if (suppressDownloadProgressText) {
    const messageText = translateServerMessage(task.message || "");
    if (task.error) {
      showTaskMessage(`${messageText} | ${t("taskError", { error: task.error })}`);
    } else {
      showTaskMessage(messageText || translateTaskStatus(task.status));
    }
    return;
  }
  const base = t("taskLine", {
    id: task.id,
    type: translateTaskType(task.task_type),
    status: translateTaskStatus(task.status),
    progress: Math.round(taskProgressValue(task) * 100),
    message: translateServerMessage(task.message || ""),
  });
  if (task.error) {
    showTaskMessage(`${base} | ${t("taskError", { error: task.error })}`);
  } else {
    showTaskMessage(base);
  }
  const cancelBtn = el("cancelCurrentTaskBtn");
  if (cancelBtn) {
    cancelBtn.disabled = !(state.currentTaskId && (task.status === "queued" || task.status === "running"));
  }
  const video = el("preview");
  const image = el("imagePreview");
  if (task.status === "completed" && task.result?.video_file) {
    image.style.display = "none";
    image.removeAttribute("src");
    video.src = `/api/videos/${encodeURIComponent(task.result.video_file)}?t=${Date.now()}`;
    video.style.display = "block";
  } else if (task.status === "completed" && task.result?.image_file) {
    video.style.display = "none";
    video.removeAttribute("src");
    image.src = `/api/images/${encodeURIComponent(task.result.image_file)}?t=${Date.now()}`;
    image.style.display = "block";
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollTask() {
  if (!state.currentTaskId) return;
  try {
    const task = await api(`/api/tasks/${state.currentTaskId}`);
    state.taskPollDelayMs = TASK_POLL_INTERVAL_MS;
    renderTask(task);
    if (task.status === "completed" || task.status === "error" || task.status === "cancelled") {
      stopPolling();
      setGenerationBusy(false);
      if (task.task_type === "download" && task.status === "completed") {
        await loadLocalModels();
      }
      if (task.status === "completed" && task.task_type !== "download") {
        try {
          await loadOutputs();
        } catch (error) {
          showTaskMessage(t("msgOutputsRefreshFailed", { error: error.message }));
        }
      }
      return;
    }
    stopPolling();
    state.pollTimer = setTimeout(pollTask, state.taskPollDelayMs);
  } catch (error) {
    stopPolling();
    state.taskPollDelayMs = Math.min(15000, Math.round((state.taskPollDelayMs || TASK_POLL_INTERVAL_MS) * 1.8));
    if (String(error.message).includes("404")) {
      saveLastTaskId(null);
      state.currentTaskId = null;
      state.currentTaskSnapshot = null;
      renderTaskProgress(null);
      setGenerationBusy(false);
      return;
    }
    showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
    state.pollTimer = setTimeout(pollTask, state.taskPollDelayMs);
  }
}

function trackTask(taskId) {
  state.currentTaskId = taskId;
  saveLastTaskId(taskId);
  setGenerationBusy(true);
  stopPolling();
  state.taskPollDelayMs = TASK_POLL_INTERVAL_MS;
  pollTask();
}

async function cancelCurrentTask() {
  if (!state.currentTaskId) return;
  const payload = { task_id: state.currentTaskId };
  await api("/api/tasks/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showTaskMessage(t("msgTaskCancelRequested", { id: state.currentTaskId }));
}

async function restoreLastTask() {
  const lastTaskId = localStorage.getItem(TASK_STORAGE_KEY);
  if (!lastTaskId) return;
  try {
    const task = await api(`/api/tasks/${lastTaskId}`);
    state.currentTaskId = lastTaskId;
    renderTask(task);
    if (task.status === "queued" || task.status === "running") {
      setGenerationBusy(true);
      stopPolling();
      state.taskPollDelayMs = TASK_POLL_INTERVAL_MS;
      pollTask();
    }
  } catch (error) {
    saveLastTaskId(null);
  }
}

function bindModelSelectors() {
  if (el("t2iModelSelect")) {
    el("t2iModelSelect").addEventListener("change", async () => {
      renderModelPreview("text-to-image");
      try {
        await loadLoraCatalog("text-to-image", false);
        await loadVaeCatalog("text-to-image", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("i2iModelSelect")) {
    el("i2iModelSelect").addEventListener("change", async () => {
      renderModelPreview("image-to-image");
      try {
        await loadLoraCatalog("image-to-image", false);
        await loadVaeCatalog("image-to-image", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("t2vModelSelect")) {
    el("t2vModelSelect").addEventListener("change", async () => {
      renderModelPreview("text-to-video");
      try {
        await loadLoraCatalog("text-to-video", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  if (el("i2vModelSelect")) {
    el("i2vModelSelect").addEventListener("change", async () => {
      renderModelPreview("image-to-video");
      try {
        await loadLoraCatalog("image-to-video", false);
      } catch (error) {
        showTaskMessage(t("msgSearchFailed", { error: error.message }));
      }
    });
  }
  bindClick("refreshT2IModels", async () => {
    try {
      await loadModelCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2IModels", async () => {
    try {
      await loadModelCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2VModels", async () => {
    try {
      await loadModelCatalog("text-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2VModels", async () => {
    try {
      await loadModelCatalog("image-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2ILoras", async () => {
    try {
      await loadLoraCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2ILoras", async () => {
    try {
      await loadLoraCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2VLoras", async () => {
    try {
      await loadLoraCatalog("text-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2VLoras", async () => {
    try {
      await loadLoraCatalog("image-to-video", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshT2IVaes", async () => {
    try {
      await loadVaeCatalog("text-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  bindClick("refreshI2IVaes", async () => {
    try {
      await loadVaeCatalog("image-to-image", true);
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
}

function bindLanguageSelector() {
  if (!el("languageSelect")) return;
  el("languageSelect").addEventListener("change", (event) => {
    setLanguage(event.target.value);
  });
}

async function bootstrap() {
  await loadExternalI18n();
  state.language = detectInitialLanguage();
  bindLanguageSelector();
  setLanguage(state.language);
  renderTaskStep(null);
  setTabs();
  bindModelSelectors();
  state.localViewMode = el("localViewMode")?.value === "flat" ? "flat" : "tree";
  state.localTreeFilter = el("localTreeSearch")?.value || "";
  renderLocalViewMode();

  el("settingsForm").addEventListener("submit", async (event) => {
    try {
      await saveSettings(event);
    } catch (error) {
      showTaskMessage(t("msgSaveSettingsFailed", { error: error.message }));
    }
  });
  bindClick("clearHfCacheBtn", async () => {
    try {
      await clearHfCache();
    } catch (error) {
      showTaskMessage(t("msgHfCacheClearFailed", { error: error.message }));
    }
  });
  bindClick("runCleanupBtn", async () => {
    try {
      await runCleanupNow();
    } catch (error) {
      showTaskMessage(t("msgCleanupFailed", { error: error.message }));
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
  el("image2imageForm").addEventListener("submit", async (event) => {
    try {
      await generateImage2Image(event);
    } catch (error) {
      showTaskMessage(t("msgImageGenerationFailed", { error: error.message }));
    }
  });
  el("text2imageForm").addEventListener("submit", async (event) => {
    try {
      await generateText2Image(event);
    } catch (error) {
      showTaskMessage(t("msgTextGenerationFailed", { error: error.message }));
    }
  });
  el("searchTask").addEventListener("change", async () => {
    refreshSearchSourceOptions();
    renderSearchBaseModelOptions();
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSource").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchBaseModel").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSort").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchNsfw").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSizeMinMb").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSizeMaxMb").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchModelKind").addEventListener("change", async () => {
    try {
      await searchModels(null, { resetPage: true });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchViewMode").addEventListener("change", () => {
    renderSearchResults(state.lastSearchResults || []);
  });
  el("searchPrevBtn").addEventListener("click", async () => {
    const page = Math.max(1, Number(state.searchPage || 1) - 1);
    try {
      await searchModels(null, { resetPage: false, page });
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchNextBtn").addEventListener("click", async () => {
    const page = Math.max(1, Number(state.searchPage || 1) + 1);
    try {
      await searchModels(null, { resetPage: false, page });
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
  el("rescanLocalModels").addEventListener("click", async () => {
    try {
      await rescanLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalRescanFailed", { error: error.message }));
    }
  });
  el("expandAllLocalTree").addEventListener("click", () => {
    if (state.localViewMode !== "tree") {
      state.localViewMode = "tree";
      if (el("localViewMode")) {
        el("localViewMode").value = "tree";
      }
      renderLocalViewMode();
    }
    setAllLocalTreeNodesOpen(true);
  });
  el("localViewMode").addEventListener("change", () => {
    state.localViewMode = el("localViewMode").value === "flat" ? "flat" : "tree";
    renderLocalViewMode();
  });
  el("localTreeSearch").addEventListener("input", () => {
    state.localTreeFilter = el("localTreeSearch").value || "";
    renderLocalModelTree(state.localTreeData);
  });
  el("refreshOutputs").addEventListener("click", async () => {
    try {
      await loadOutputs();
    } catch (error) {
      showTaskMessage(t("msgOutputsRefreshFailed", { error: error.message }));
    }
  });
  el("localModelsDir").addEventListener("change", async () => {
    try {
      await loadLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  el("localLineageFilter").addEventListener("change", () => {
    state.localLineageFilter = el("localLineageFilter").value || "all";
    renderLocalModels(state.localModels || [], state.localModelsBaseDir || "");
  });
  el("cfgModelsDir").addEventListener("change", async () => {
    try {
      await loadSettingsLocalModels();
    } catch (error) {
      showTaskMessage(t("msgLocalModelRefreshFailed", { error: error.message }));
    }
  });
  bindClick("downloadsToggle", () => {
    setDownloadsPopoverOpen(!state.downloadsPopoverOpen);
  });
  bindClick("downloadsRefreshBtn", async () => {
    try {
      await refreshDownloadTasks();
    } catch (error) {
      showTaskMessage(t("msgDownloadsRefreshFailed", { error: error.message }));
    }
  });
  bindClick("cancelCurrentTaskBtn", async () => {
    try {
      await cancelCurrentTask();
    } catch (error) {
      showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
    }
  });
  bindClick("modelDetailModalClose", () => {
    closeModelDetailModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModelDetailModal();
      setDownloadsPopoverOpen(false);
    }
  });
  document.addEventListener("click", (event) => {
    const target = event.target;
    const modal = el("modelDetailModal");
    if (modal && modal.classList.contains("open") && target === modal) {
      closeModelDetailModal();
    }
    const popover = el("downloadsPopover");
    const toggle = el("downloadsToggle");
    if (!popover || !toggle) return;
    if (!state.downloadsPopoverOpen) return;
    if (popover.contains(target) || toggle.contains(target)) return;
    setDownloadsPopoverOpen(false);
  });
  try {
    await Promise.all([loadRuntimeInfo(), loadSettings(), loadVideoModelSpecs()]);
    await Promise.all([loadLocalModels(), loadOutputs()]);
    await Promise.all([
      loadModelCatalog("text-to-image", false),
      loadModelCatalog("image-to-image", false),
      loadModelCatalog("text-to-video", false),
      loadModelCatalog("image-to-video", false),
    ]);
    await searchModels(null, { resetPage: true });
    await restoreLastTask();
    await refreshDownloadTasks();
    startDownloadPolling();
  } catch (error) {
    showTaskMessage(t("msgInitFailed", { error: error.message }));
  }
}

bootstrap();
