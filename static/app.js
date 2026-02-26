const SUPPORTED_LANGS = ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "ar"];
const DEFAULT_LANG = "en";
const LANG_STORAGE_KEY = "videogen_lang";
const TASK_STORAGE_KEY = "videogen_last_task_id";
const TASK_POLL_INTERVAL_MS = 1000;

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
    labelQuery: "Query",
    labelLimit: "Limit",
    btnSearchModels: "Search Models",
    headingLocalModels: "Local Models",
    headingOutputs: "Outputs",
    btnRefreshLocalList: "Refresh Local List",
    btnRefreshOutputs: "Refresh Outputs",
    labelLocalModelsPath: "Local Models Path",
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
    labelDefaultTextModel: "Default: Text to Video",
    labelDefaultImageModel: "Default: Image to Video",
    labelDefaultTextImageModel: "Default: Text to Image",
    labelDefaultImageImageModel: "Default: Image to Image",
    labelDefaultSteps: "Default Steps",
    labelDefaultFrames: "Default Frames",
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
    helpLocalModelsPath:
      "Impact: Changes which folder is scanned and shown in the Local Models screen.\nExample: D:\\ModelStore\\VideoGen",
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
      "Impact: Higher frame count makes longer/smoother clips but increases VRAM and processing time.\nExample: 16",
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
    placeholderOptional: "optional",
    searchSourceAll: "All",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "All base models",
    localLineageAll: "All lineages",
    msgSettingsSaved: "Settings saved.",
    msgNoLocalModels: "No local models in: {path}",
    msgNoOutputs: "No outputs in: {path}",
    msgPortChangeSaved: "Listen port saved. Restart `start.bat` to apply new port.",
    msgServerSettingRestartRequired: "Server setting saved. Restart `start.bat` to apply changes.",
    msgNoModelsFound: "No models found.",
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
    btnDeleteModel: "Delete",
    msgAlreadyDownloaded: "Downloaded",
    msgModelPreviewAlt: "Model preview",
    msgModelDownloadStarted: "Model download started: {repo} -> {path}",
    msgTextImageGenerationStarted: "Text-to-image generation started: {id}",
    msgImageImageGenerationStarted: "Image-to-image generation started: {id}",
    msgTextGenerationStarted: "Text generation started: {id}",
    msgImageGenerationStarted: "Image generation started: {id}",
    msgTaskPollFailed: "Task poll failed: {error}",
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
    serverLoadingLora: "Applying LoRA",
    serverPreparingImage: "Preparing image",
    serverGeneratingImage: "Generating image",
    serverGeneratingFrames: "Generating frames",
    serverDecodingLatents: "Decoding latents",
    serverDecodingLatentsCpuFallback: "Decoding latents",
    serverPostprocessingImage: "Postprocessing image",
    serverEncoding: "Encoding mp4",
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
    labelQuery: "検索語",
    labelLimit: "件数",
    btnSearchModels: "モデル検索",
    headingLocalModels: "ローカルモデル",
    headingOutputs: "成果物",
    btnRefreshLocalList: "ローカル一覧を更新",
    btnRefreshOutputs: "成果物一覧を更新",
    labelLocalModelsPath: "ローカルモデル表示パス",
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
    labelDefaultTextModel: "既定: テキスト→動画",
    labelDefaultImageModel: "既定: 画像→動画",
    labelDefaultTextImageModel: "既定: テキスト→画像",
    labelDefaultImageImageModel: "既定: 画像→画像",
    labelDefaultSteps: "既定ステップ数",
    labelDefaultFrames: "既定フレーム数",
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
    helpLocalModelsPath:
      "影響: ローカルモデル画面で一覧表示するフォルダを切り替えます。\n例: D:\\ModelStore\\VideoGen",
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
      "影響: 値を上げると動画が長く滑らかになりますが、VRAM消費と処理時間が増えます。\n例: 16",
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
    placeholderOptional: "任意",
    searchSourceAll: "すべて",
    searchSourceHf: "Hugging Face",
    searchSourceCivitai: "CivitAI",
    searchBaseModelAll: "すべてのベースモデル",
    localLineageAll: "すべての系譜",
    msgSettingsSaved: "設定を保存しました。",
    msgNoLocalModels: "ローカルモデルがありません: {path}",
    msgNoOutputs: "成果物がありません: {path}",
    msgPortChangeSaved: "リッスンポートを保存しました。反映には `start.bat` の再起動が必要です。",
    msgServerSettingRestartRequired: "サーバー設定を保存しました。反映には `start.bat` の再起動が必要です。",
    msgNoModelsFound: "モデルが見つかりません。",
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
    btnDeleteModel: "削除",
    msgAlreadyDownloaded: "ダウンロード済み",
    msgModelPreviewAlt: "モデルプレビュー",
    msgModelDownloadStarted: "モデルダウンロード開始: {repo} -> {path}",
    msgTextImageGenerationStarted: "テキスト→画像生成を開始しました: {id}",
    msgImageImageGenerationStarted: "画像→画像生成を開始しました: {id}",
    msgTextGenerationStarted: "テキスト生成を開始しました: {id}",
    msgImageGenerationStarted: "画像生成を開始しました: {id}",
    msgTaskPollFailed: "タスク確認に失敗: {error}",
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
    serverLoadingLora: "LoRAを適用中",
    serverPreparingImage: "画像を準備中",
    serverGeneratingImage: "画像を生成中",
    serverGeneratingFrames: "フレームを生成中",
    serverDecodingLatents: "潜在表現をデコード中",
    serverDecodingLatentsCpuFallback: "潜在表現をデコード中",
    serverPostprocessingImage: "画像を後処理中",
    serverEncoding: "mp4に変換中",
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
  settings: null,
  localModels: [],
  localModelsBaseDir: "",
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
  return new Set((state.localModels || []).map((item) => normalizeModelId(item.repo_hint)));
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
  renderLocalLineageOptions(state.localModels || []);
  renderLocalModels(state.localModels || [], state.localModelsBaseDir || "");
  renderOutputs(state.outputs || [], state.outputsBaseDir || "");
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
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
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      el(`panel-${button.dataset.tab}`).classList.add("active");
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
  const text = `${item?.base_model || ""} ${item?.repo_hint || ""} ${item?.class_name || ""}`.toLowerCase();
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
  return status || t("taskTypeUnknown");
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
    "Applying LoRA": t("serverLoadingLora"),
    "Preparing image": t("serverPreparingImage"),
    "Generating image": t("serverGeneratingImage"),
    "Generating frames": t("serverGeneratingFrames"),
    "Decoding latents": t("serverDecodingLatents"),
    "Decoding latents (CPU fallback)": t("serverDecodingLatents"),
    "Postprocessing image": t("serverPostprocessingImage"),
    "Encoding mp4": t("serverEncoding"),
    "Saving png": t("serverSavingPng"),
    Done: t("serverDone"),
    "Generation failed": t("serverGenerationFailed"),
    "Download complete": t("serverDownloadComplete"),
    "Download failed": t("serverDownloadFailed"),
  };
  return map[raw] || raw;
}

async function loadRuntimeInfo() {
  try {
    const info = await api("/api/system/info");
    state.runtimeInfo = info;
    const flags = [
      `${t("runtimeDevice")}=${info.device}`,
      `${t("runtimeCuda")}=${info.cuda_available}`,
      `${t("runtimeRocm")}=${info.rocm_available}`,
      `${t("runtimeNpu")}=${info.npu_available}`,
      `${t("runtimeDiffusers")}=${info.diffusers_ready}`,
    ];
    if (info.torch_version) flags.push(`${t("runtimeTorch")}=${info.torch_version}`);
    if (info.npu_reason) flags.push(`${t("runtimeNpuReason")}=${info.npu_reason}`);
    if (info.import_error) flags.push(`${t("runtimeError")}=${info.import_error}`);
    el("runtimeInfo").textContent = flags.join(" | ");
    applyNpuAvailability(info);
  } catch (error) {
    state.runtimeInfo = null;
    el("runtimeInfo").textContent = t("runtimeLoadFailed", { error: error.message });
    applyNpuAvailability(null);
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
  el("cfgSteps").value = settings.defaults.num_inference_steps;
  el("cfgFrames").value = settings.defaults.num_frames;
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

  preview.innerHTML = `
    <div class="model-picked-card">
      ${imageHtml}
      <div class="model-picked-meta">${infoHtml}<span>${metaText}</span></div>
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

async function deleteLocalModel(item, baseDir = "") {
  const name = item?.repo_hint || item?.name || "";
  if (!window.confirm(t("msgConfirmDeleteModel", { model: name }))) return;
  await api("/api/models/local/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: item.name,
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
      const modelDom = getModelDom(task);
      const select = el(modelDom.selectId);
      if (!select) return;
      const catalog = state.modelCatalog[task] || [];
      if (!catalog.some((entry) => entry.value === item.path)) {
        catalog.push({
          source: "local",
          label: `[local] ${item.repo_hint}`,
          value: item.path,
          id: item.repo_hint,
          size_bytes: null,
          preview_url: item.preview_url || null,
          model_url: item.repo_hint ? `https://huggingface.co/${encodeURIComponent(item.repo_hint).replaceAll("%2F", "/")}` : null,
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
      showTaskMessage(t("msgLocalModelApplied", { task, model: item.repo_hint }));
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

async function loadLocalModels() {
  const dir = el("localModelsDir")?.value?.trim() || "";
  const params = new URLSearchParams();
  if (dir) params.set("dir", dir);
  const url = params.toString() ? `/api/models/local?${params.toString()}` : "/api/models/local";
  const data = await api(url);
  state.localModels = data.items || [];
  state.localModelsBaseDir = data.base_dir || dir || "";
  renderLocalLineageOptions(state.localModels);
  renderLocalModels(state.localModels, state.localModelsBaseDir);
  if ((state.lastSearchResults || []).length) {
    renderSearchResults(state.lastSearchResults);
  }
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

function renderSearchResults(items) {
  const container = el("searchResults");
  if (!items.length) {
    container.innerHTML = `<p>${t("msgNoModelsFound")}</p>`;
    return;
  }
  const installed = getInstalledModelIdSet();
  container.innerHTML = items
    .map(
      (item) => {
        const supportsDownload = item.download_supported !== false;
        const isInstalled = supportsDownload && installed.has(normalizeModelId(item.id));
        const actionHtml = !supportsDownload
          ? `<a href="${item.model_url || "#"}" target="_blank" rel="noopener noreferrer">${t("msgOpen")}</a>`
          : isInstalled
            ? `<span class="downloaded-flag">${t("msgAlreadyDownloaded")}</span>`
            : `<button type="button" class="download-btn" data-repo="${item.id}">${t("btnDownload")}</button>`;
        return `
      <div class="row model-row">
        <div class="model-main">
          ${
            item.preview_url
              ? `<img class="model-preview" src="${item.preview_url}" alt="${t("msgModelPreviewAlt")}" loading="lazy" onerror="this.style.display='none'" />`
              : ""
          }
          <div>
            <strong><a href="${item.model_url || "#"}" target="_blank" rel="noopener noreferrer">${item.id}</a></strong>
            <span>${t("modelTag")}=${item.pipeline_tag || "n/a"} | ${t("modelBase")}=${item.base_model || "n/a"} | ${t("modelSource")}=${item.source || "unknown"} | ${t("modelDownloads")}=${item.downloads ?? "n/a"} | ${t("modelLikes")}=${item.likes ?? "n/a"} | ${t("modelSize")}=${formatModelSize(item.size_bytes)}</span>
          </div>
        </div>
        ${actionHtml}
      </div>`;
      },
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
  const rawLimit = Number(el("searchLimit").value || "30");
  const limit = Math.min(50, Math.max(1, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 30));
  el("searchLimit").value = String(limit);
  const baseModel = (el("searchBaseModel")?.value || "all").trim();
  const params = new URLSearchParams({
    task: el("searchTask").value,
    source: el("searchSource").value || "all",
    query: el("searchQuery").value.trim(),
    limit: String(limit),
  });
  if (baseModel && baseModel !== "all") {
    params.set("base_model", baseModel);
  }
  const data = await api(`/api/models/search?${params.toString()}`);
  state.lastSearchResults = data.items || [];
  renderSearchResults(state.lastSearchResults);
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
  const loraIds = getSelectedValues("i2vLoraSelect");
  formData.append("image", imageFile);
  formData.append("prompt", el("i2vPrompt").value.trim());
  formData.append("negative_prompt", el("i2vNegative").value.trim());
  formData.append("model_id", el("i2vModelSelect").value.trim());
  formData.append("lora_id", loraIds[0] || "");
  loraIds.forEach((value) => formData.append("lora_ids", value));
  formData.append("lora_scale", String(Number(el("i2vLoraScale").value)));
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

async function generateText2Image(event) {
  event.preventDefault();
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
  const data = await api("/api/generate/text2image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showTaskMessage(t("msgTextImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

async function generateImage2Image(event) {
  event.preventDefault();
  const imageFile = el("i2iImage").files[0];
  if (!imageFile) throw new Error(t("msgInputImageRequired"));
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
  const data = await api("/api/generate/image2image", {
    method: "POST",
    body: formData,
  });
  showTaskMessage(t("msgImageImageGenerationStarted", { id: data.task_id }));
  trackTask(data.task_id);
}

function renderTask(task) {
  renderTaskProgress(task);
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
      if (task.status === "completed" && task.task_type !== "download") {
        try {
          await loadOutputs();
        } catch (error) {
          showTaskMessage(t("msgOutputsRefreshFailed", { error: error.message }));
        }
      }
    }
  } catch (error) {
    stopPolling();
    if (String(error.message).includes("404")) {
      saveLastTaskId(null);
      state.currentTaskId = null;
      renderTaskProgress(null);
    }
    showTaskMessage(t("msgTaskPollFailed", { error: error.message }));
  }
}

function trackTask(taskId) {
  state.currentTaskId = taskId;
  saveLastTaskId(taskId);
  stopPolling();
  pollTask();
  state.pollTimer = setInterval(pollTask, TASK_POLL_INTERVAL_MS);
}

async function restoreLastTask() {
  const lastTaskId = localStorage.getItem(TASK_STORAGE_KEY);
  if (!lastTaskId) return;
  try {
    const task = await api(`/api/tasks/${lastTaskId}`);
    state.currentTaskId = lastTaskId;
    renderTask(task);
    if (task.status === "queued" || task.status === "running") {
      stopPolling();
      state.pollTimer = setInterval(pollTask, TASK_POLL_INTERVAL_MS);
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
  state.language = detectInitialLanguage();
  bindLanguageSelector();
  setLanguage(state.language);
  setTabs();
  bindModelSelectors();

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
      await searchModels();
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchSource").addEventListener("change", async () => {
    try {
      await searchModels();
    } catch (error) {
      showTaskMessage(t("msgSearchFailed", { error: error.message }));
    }
  });
  el("searchBaseModel").addEventListener("change", async () => {
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
  try {
    await Promise.all([loadRuntimeInfo(), loadSettings()]);
    await Promise.all([loadLocalModels(), loadOutputs()]);
    await Promise.all([
      loadModelCatalog("text-to-image", false),
      loadModelCatalog("image-to-image", false),
      loadModelCatalog("text-to-video", false),
      loadModelCatalog("image-to-video", false),
    ]);
    await searchModels();
    await restoreLastTask();
  } catch (error) {
    showTaskMessage(t("msgInitFailed", { error: error.message }));
  }
}

bootstrap();
