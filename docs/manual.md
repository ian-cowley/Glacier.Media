# Glacier.Media User Manual

`Glacier.Media` splits its multimodal APIs into three subsystem guides:

- [Audio Guide](manual/audio.md) — `KittenTtsEngine` neural TTS, `ProceduralFormantTtsEngine` / `KokoroTtsEngine` vocal-tract TTS, `OpenAiWhisperClient` / `WhisperEngine` STT, `VoicePipeline`, `MelSpectrogram`, and `WavWriter`
- [Image & Video Guide](manual/image-and-video.md) — FLUX.1 (`DiffusionGgufModel`, `DiffusionGgufPipeline`), Wan 2.1 (`VideoGenerationPipeline`, `CameraMotion`), `ImageDecoder`, `PngWriter`, `BmpWriter`, `ApngWriter`, `GifWriter`, and `AviWriter`
- [Vision & Frame Sampling Guide](manual/vision.md) — `VisionPipeline`, `VisionPreprocessor`, `VisionTransformer`, `MultimodalProjector`, `VideoPipeline`, and `VideoFrameSampler`

---

## Shared Math & Experimental Diagnostics

- **`Glacier.Media.Media.MediaKernels`:** Shared SIMD math helpers (matrix multiplication, GELU/SiLU/RMSNorm/LayerNorm, softmax, RoPE) reused across the vision, audio, image, and video pipelines.
- **Experimental Flag (`GLACIER_EXP001`):** Procedural test pipelines (`ImageGenerationPipeline`, `VisionPipeline`, `SpatioTemporalDiT`, `TemporalLatentVaeDecoder`) and HTTP-backed `VoicePipeline` / `WhisperEngine` carry `[Experimental("GLACIER_EXP001")]`. Neural production pipelines (`KittenTtsEngine`, `DiffusionGgufPipeline`, `VideoGenerationPipeline`) require explicit SafeTensors/GGUF weight files and fail fast with `FileNotFoundException` rather than silently producing noise.
