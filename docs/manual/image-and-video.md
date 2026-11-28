# Glacier.Media — Image & Video Generation Manual

This guide documents FLUX.1 image synthesis, Wan 2.1 video synthesis, and image/video file encoders in `Glacier.Media.Image` and `Glacier.Media.Video`.

---

## 1. FLUX.1 Text-to-Image (`DiffusionGgufModel` & `DiffusionGgufPipeline`)

`DiffusionGgufPipeline` runs quantized FLUX.1 weights (`UniversalArchitecture.Flux`) alongside pure C# CLIP-L (`FluxClipEncoder` + `ClipTokenizer`), T5-XXL (`FluxT5Encoder`), and neural VAE (`FluxVaeDecoder` enforcing `F32` decode precision):

```csharp
using Glacier.Media.Image;
using Glacier.Media.Image.Gguf;

using var model = DiffusionGgufModel.Open("flux1-schnell-Q4_K_M.gguf");
using var pipeline = new DiffusionGgufPipeline(
    model,
    vaeSafetensorsPath: "models/ae.safetensors",
    clipSafetensorsPath: "models/clip_l.safetensors",
    t5GgufPath: "models/t5xxl.gguf");

ImageGenerationResult result = pipeline.Generate(
    prompt: "Sunlit alpine lake reflection, high detail",
    width: 512,
    height: 512,
    numSteps: 4,
    cfgScale: 3.5f,
    seed: 42);

PngWriter.SavePng24("lake.png", result.RgbPixels, result.Width, result.Height);
// Or uncompressed bitmap:
// BmpWriter.SaveBmp24("lake.bmp", result.RgbPixels, result.Width, result.Height);
```

Inspect `pipeline.ActiveBackend` (`"Cuda"` or `"Cpu"`) to confirm hardware execution.

---

## 2. Wan 2.1 Text-to-Video (`VideoGenerationPipeline`)

`VideoGenerationPipeline` combines `WanDiT`, `FluxT5Encoder`, and causal 3D VAE decoding (`Wan3DVaeDecoder`) with `FlowMatchingScheduler`:

```csharp
using Glacier.Media.Video;

using var video = new VideoGenerationPipeline(
    modelPath: "models/Wan2.1-T2V-1.3B-Q4_K_M.gguf",
    t5Path: "models/t5xxl.gguf",
    vaePath: "models/wan_2.1_vae.safetensors");

VideoGenerationResult clip = video.Generate(
    prompt: "Drone glide over ocean cliffs at sunset",
    width: 832,
    height: 480,
    numFrames: 17,
    fps: 16,
    numSteps: 25,
    motion: CameraMotion.PanRight,
    seed: 42,
    guidanceScale: 5.0f);
```

Export generated frames (`byte[][]` / frame buffers) directly in pure C# using:

- `ApngWriter` — Animated PNG export
- `GifWriter` — Quantized animated GIF export
- `AviWriter` — Uncompressed/MJPEG AVI container export

---

## 3. Experimental Procedural Image Pipeline (`ImageGenerationPipeline`)

`ImageGenerationPipeline` (`[Experimental("GLACIER_EXP001")]`) provides a lightweight weightless Flow-Matching DiT (`DiffusionTransformer`) + `LatentVaeDecoder` harness for deterministic testing without large GGUF downloads. Both `width` and `height` must be multiples of `16`.
