# Changelog

All notable changes to `Glacier.Media` are documented here.

## v1.1.3

- Hardened image dimension and buffer bounds validation in `ImageDecoder`.
- Enforced `F32` decode dtype precision in `FluxVaeDecoder`.
- Aligned `Glacier.Inference` package dependency to `1.10.1`.

## v1.1.0

- Marked procedural test pipelines (`ImageGenerationPipeline`, `VisionPipeline`, `SpatioTemporalDiT`, `TemporalLatentVaeDecoder`) and HTTP client proxies (`WhisperEngine`, `VoicePipeline`) with `[Experimental("GLACIER_EXP001")]`.
- Separated `KokoroTtsEngine` procedural presets from neural `KittenTtsEngine` SafeTensors synthesis (`IsNeural = false`).
- Added fail-fast weight checks in `DiffusionGgufPipeline` and `VideoGenerationPipeline` to prevent silent fallback to uninitialized noise.
