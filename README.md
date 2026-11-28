# Glacier.Media

[![NuGet CI](https://github.com/ian-cowley/Glacier.Media/actions/workflows/publish-nuget.yml/badge.svg)](https://github.com/ian-cowley/Glacier.Media/actions/workflows/publish-nuget.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)
![NativeAOT](https://img.shields.io/badge/NativeAOT-Compatible-brightgreen.svg)

`Glacier.Media` is a pure C# generative audio, image, video, and vision library for .NET 10 built on `Glacier.Inference`. It runs local neural speech synthesis (`KittenTtsEngine`), procedural vocal-tract synthesis, FLUX.1 image generation, Wan 2.1 video generation, OpenAI-compatible speech transcription, and zero-dependency media container encoding (WAV, PNG, BMP, APNG, GIF, AVI).

---

## Install

```bash
dotnet add package Glacier.Media
```

Or build and test from source:

```bash
git clone https://github.com/ian-cowley/Glacier.Media.git
cd Glacier.Media
dotnet test Glacier.Media.slnx -c Release
```

---

## Quick Start

Synthesize speech to a 24 kHz PCM WAV file with zero external weights using `ProceduralFormantTtsEngine` (or `KokoroTtsEngine`):

```csharp
using Glacier.Media.Audio;

using var tts = new ProceduralFormantTtsEngine();
tts.SynthesizeToFile(
    text: "Glacier Media synthesizes speech directly in C sharp.",
    outputPath: "hello.wav",
    voice: KokoroVoice.AfHeart,
    speed: 1.0f);
```

Run FLUX.1 text-to-image generation using local GGUF / SafeTensors weights (`DiffusionGgufModel` + `DiffusionGgufPipeline`) and export to PNG:

```csharp
using Glacier.Media.Image;
using Glacier.Media.Image.Gguf;

using var model = DiffusionGgufModel.Open("flux1-schnell-Q4_K_M.gguf");
using var pipeline = new DiffusionGgufPipeline(
    model,
    vaeSafetensorsPath: "ae.safetensors",
    clipSafetensorsPath: "clip_l.safetensors",
    t5GgufPath: "t5xxl.gguf");

ImageGenerationResult img = pipeline.Generate("Misty Nordic fjord at dawn", width: 512, height: 512, numSteps: 4);
PngWriter.SavePng24("fjord.png", img.RgbPixels, img.Width, img.Height);
```

---

## What You Can Do

- **Neural & Procedural Speech (`Glacier.Media.Audio`):** Run distilled StyleTTS 2 / ALBERT / HiFi-GAN synthesis in pure C# (`KittenTtsEngine`), generate dependency-free 24 kHz formant speech (`ProceduralFormantTtsEngine`, `KokoroTtsEngine`), and transcribe audio via OpenAI-compatible HTTP `/v1/audio/transcriptions` endpoints or local NPU sidecars (`WhisperEngine`, `OpenAiWhisperClient`).
- **FLUX.1 Image Generation (`Glacier.Media.Image`):** Tokenize and encode prompts with built-in CLIP-L BPE and T5-XXL SentencePiece encoders, execute Flow-Matching Euler DiT trajectories (`FluxDiT`), and decode latents via `FluxVaeDecoder` into `PngWriter` / `BmpWriter` outputs.
- **Wan 2.1 Video Generation (`Glacier.Media.Video`):** Generate multi-frame video sequences using `VideoGenerationPipeline`, 3D spatio-temporal `WanDiT`, causal 3D VAE decoding (`Wan3DVaeDecoder`), and pure C# `ApngWriter`, `GifWriter`, and `AviWriter`.
- **Vision & Spatial Projection (`Glacier.Media.Vision`):** Decode images (`ImageDecoder`), extract normalized ViT patches (`VisionPreprocessor`), and test 2D spatial merging (`MultimodalProjector`).

---

## Platform & Compatibility

- **Runtime:** .NET 10 (`net10.0`), Native AOT compatible (`IsAotCompatible=true`).
- **Compute Backends:** AVX-512 / AVX2 CPU SIMD via `MediaKernels` and `KittenTensorOps`, plus optional CUDA acceleration via `Glacier.Inference` for `FluxDiT`, `WanDiT`, and `Wan3DVaeDecoder`.
- **Experimental Surface:** `ImageGenerationPipeline`, `VoicePipeline`, `WhisperEngine`, `VisionPipeline`, `SpatioTemporalDiT`, and `TemporalLatentVaeDecoder` carry `[Experimental("GLACIER_EXP001")]`.

---

## Documentation

- [User Manual Index](docs/manual.md) — Overview and shared containers/kernels
  - [Audio (TTS, STT, Voice Pipeline)](docs/manual/audio.md)
  - [Image & Video Generation](docs/manual/image-and-video.md)
  - [Vision & Frame Sampling](docs/manual/vision.md)
- [Architecture](docs/architecture.md) — FLUX.1 / Wan 2.1 DiT flow-matching pipelines, KittenTTS, and audio/video containers
- [Changelog](CHANGELOG.md) — Release history (`v1.1.3`)

---

## License & Credits

Licensed under the MIT License.

Developed by Ian Cowley, Antigravity (Google DeepMind) and Claude.
