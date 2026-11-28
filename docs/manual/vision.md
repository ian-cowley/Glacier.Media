# Glacier.Media — Vision & Video Decoding Manual

This guide covers image decoding, ViT patch extraction, spatial projection, and video frame sampling in `Glacier.Media.Vision`, `Glacier.Media.Image`, and `Glacier.Media.Video`.

---

## 1. Image Decoding (`ImageDecoder`)

`Glacier.Media.Image.ImageDecoder` decodes raw image bytes with strict dimension and buffer bounds enforcement into RGB/RGBA pixel arrays for vision preprocessing or inspection.

---

## 2. Experimental Vision Pipeline (`VisionPipeline`)

`Glacier.Media.Vision.VisionPipeline` (`[Experimental("GLACIER_EXP001")]`) pairs `VisionPreprocessor`, `VisionTransformer`, and `MultimodalProjector`:

```csharp
#pragma warning disable GLACIER_EXP001
using Glacier.Media.Vision;

using var vision = new VisionPipeline(numLayers: 4, visionDim: 768, llmDim: 3584, patchSize: 14);
var (tokenCount, embeddings, meta) = vision.ProcessImage(rgbBytes, width: 640, height: 480, channels: 3);
Console.WriteLine($"Extracted {tokenCount} merged tokens ({meta.Width}x{meta.Height})");
#pragma warning restore GLACIER_EXP001
```

Key behavior:

- `VisionPreprocessor.ExtractPatches` resizes to `448x448` (`DefaultTargetWidth` x `DefaultTargetHeight`), extracts `14x14` patches (`1024` patches), and applies SigLIP normalization when enabled.
- `MultimodalProjector` applies 2D spatial merging (`factor = 2`, reducing `32x32` patches to `256` tokens) projected into `llmDim`.
- `VisionPipeline.Query(...)` intentionally throws `NotSupportedException` because multimodal language model weights are not loaded by this procedural pipeline.

---

## 3. Video Container & Frame Sampling (`VideoPipeline`, `VideoFrameSampler`)

`Glacier.Media.Video.VideoPipeline` and `VideoFrameSampler` extract keyframe metadata (`VideoMetadata`) and uniformly sampled frame buffers from supported inputs without requiring FFmpeg binaries.
