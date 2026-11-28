namespace Glacier.Media.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Inference.Gpu;
using Glacier.Media.Audio;
using Glacier.Media.Audio.Kitten;
using Glacier.Media.Video;
using Glacier.Media.Video.Wan;
using Xunit;

#pragma warning disable GLACIER_EXP001
public class WanAndCapabilitiesHardeningTests
{
    [Fact]
    public void ProceduralFormantAndKokoroAndWhisperAndVideoPipeline_ExposeHonestCapabilities()
    {
        using var procedural = new ProceduralFormantTtsEngine();
        using var kokoro = new KokoroTtsEngine();

        Assert.False(procedural.IsNeural);
        Assert.False(procedural.SupportsModelWeights);
        Assert.Equal("ProceduralFormant", procedural.ActiveBackend);

        Assert.False(kokoro.IsNeural);
        Assert.False(kokoro.SupportsModelWeights);
        Assert.Equal("ProceduralFormant", kokoro.ActiveBackend);

        float[] procAudio = procedural.Synthesize("Honest capability verification.", KokoroVoice.BfEmma, 1.0f);
        float[] kokoroAudio = kokoro.Synthesize("Honest capability verification.", KokoroVoice.BfEmma, 1.0f);
        Assert.NotEmpty(procAudio);
        Assert.Equal(procAudio, kokoroAudio);

        using var whisperOffline = new WhisperEngine(new WhisperEngineOptions { AutoDiscoverNpu = false });
        Assert.False(whisperOffline.SupportsLocalDecoding);
        Assert.True(whisperOffline.RequiresRemoteEndpoint);
        Assert.Equal("Unconfigured", whisperOffline.ActiveBackend);

        using var pipeline = new VoicePipeline();
        Assert.False(pipeline.IsNeuralSynthesis);

        using var video = new VideoPipeline(numLayers: 2, visionDim: 128, llmDim: 256, patchSize: 14);
        Assert.False(video.SupportsVlmQuery);
        Assert.False(video.IsNeuralVlm);
        Assert.Equal("ProceduralMetadataAndEmbeddings", video.ActiveBackend);

        var frames = new List<byte[]> { new byte[28 * 28 * 3], new byte[28 * 28 * 3] };
        Array.Fill(frames[1], (byte)200);
        VideoMetadata meta = video.AnalyzeFrames(frames, 28, 28, targetKeyframes: 2);
        string summary = VideoPipeline.SummarizeMetadata(meta);
        Assert.Contains("2 frames", summary);
        Assert.Contains("Pattern:", summary);

        using var videoWithDelegate = new VideoPipeline(
            numLayers: 2,
            visionDim: 128,
            llmDim: 256,
            patchSize: 14,
            vlmQueryHandler: (m, _, q) => $"{q} -> {VideoPipeline.SummarizeMetadata(m)}");
        Assert.True(videoWithDelegate.SupportsVlmQuery);
        Assert.Equal("ExternalVlmDelegate", videoWithDelegate.ActiveBackend);
        string delegated = videoWithDelegate.Query(meta, "Status");
        Assert.StartsWith("Status ->", delegated);
    }

    [Fact]
    public void Wan3DVaeDecoder_CpuPath_DecodesSyntheticSafetensorsDeterministically()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_wan_vae_cpu_{Guid.NewGuid():N}.safetensors");
        try
        {
            WriteMinimalWanVaeSafetensors(path, "F16");
            using var decoder = Wan3DVaeDecoder.Open(path, enableGpu: false);
            Assert.False(decoder.IsGpuAccelerated);
            Assert.Equal("Cpu", decoder.ActiveBackend);

            const int temporalLatentFrames = 2;
            const int latentH = 1;
            const int latentW = 1;
            const int targetFrames = 5;
            float[] latents = new float[temporalLatentFrames * 16 * latentH * latentW];
            for (int i = 0; i < latents.Length; i++) latents[i] = (i % 7 - 3) * 0.25f;

            var frames1 = decoder.DecodeVideo(latents, temporalLatentFrames, targetFrames, latentH, latentW);
            var frames2 = decoder.DecodeVideo(latents, temporalLatentFrames, targetFrames, latentH, latentW);

            Assert.Equal(targetFrames, frames1.Count);
            Assert.Equal(targetFrames, frames2.Count);
            for (int f = 0; f < targetFrames; f++)
            {
                Assert.Equal(8 * 8 * 3, frames1[f].Length);
                Assert.Equal(frames1[f], frames2[f]);
            }

            decoder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => decoder.DecodeVideo(latents, temporalLatentFrames, targetFrames, latentH, latentW));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [CudaFact(false)]
    [Trait("Hardware", "Gpu")]
    public void Wan3DVaeDecoder_GpuPath_MatchesCpuPathOnSyntheticSafetensors()
    {
        if (!GpuContext.IsSupported) return;

        string path = Path.Combine(Path.GetTempPath(), $"glacier_wan_vae_gpu_{Guid.NewGuid():N}.safetensors");
        try
        {
            WriteMinimalWanVaeSafetensors(path, "F16");
            using var cpuDecoder = Wan3DVaeDecoder.Open(path, enableGpu: false);
            using var gpuDecoder = Wan3DVaeDecoder.Open(path, enableGpu: true);

            Assert.False(cpuDecoder.IsGpuAccelerated);
            Assert.True(gpuDecoder.IsGpuAccelerated);
            Assert.Equal("Cuda", gpuDecoder.ActiveBackend);

            const int temporalLatentFrames = 2;
            const int latentH = 2;
            const int latentW = 2;
            const int targetFrames = 5;
            float[] latents = new float[temporalLatentFrames * 16 * latentH * latentW];
            for (int i = 0; i < latents.Length; i++) latents[i] = MathF.Sin(i * 0.37f);

            var cpuFrames = cpuDecoder.DecodeVideo(latents, temporalLatentFrames, targetFrames, latentH, latentW);
            var gpuFrames = gpuDecoder.DecodeVideo(latents, temporalLatentFrames, targetFrames, latentH, latentW);

            Assert.Equal(targetFrames, cpuFrames.Count);
            Assert.Equal(targetFrames, gpuFrames.Count);

            int maxByteDiff = 0;
            for (int f = 0; f < targetFrames; f++)
            {
                Assert.Equal(cpuFrames[f].Length, gpuFrames[f].Length);
                for (int b = 0; b < cpuFrames[f].Length; b++)
                {
                    int diff = Math.Abs(cpuFrames[f][b] - gpuFrames[f][b]);
                    if (diff > maxByteDiff) maxByteDiff = diff;
                }
            }

            Assert.True(maxByteDiff <= 2, $"CPU vs GPU Wan3DVaeDecoder max RGB byte diff too large: {maxByteDiff}");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static void WriteMinimalWanVaeSafetensors(string path, string dtype)
    {
        int elemBytes = dtype == "F16" ? 2 : 4;
        var specs = new List<(string Name, int[] Shape)>
        {
            ("conv2.weight", [16, 16, 1, 1]),
            ("conv2.bias", [16]),
            ("decoder.conv1.weight", [384, 16, 3, 3, 3]),
            ("decoder.conv1.bias", [384]),
            ("decoder.middle.1.norm.gamma", [384]),
            ("decoder.middle.1.to_qkv.weight", [384 * 3, 384]),
            ("decoder.middle.1.to_qkv.bias", [384 * 3]),
            ("decoder.middle.1.proj.weight", [384, 384]),
            ("decoder.middle.1.proj.bias", [384]),
            ("decoder.head.0.gamma", [96]),
            ("decoder.head.2.weight", [3, 96, 3, 3, 3]),
            ("decoder.head.2.bias", [3])
        };

        void AddResNet(string prefix, int inC, int outC, bool withShortcut = false)
        {
            if (withShortcut)
            {
                specs.Add(($"{prefix}.shortcut.weight", [outC, inC]));
                specs.Add(($"{prefix}.shortcut.bias", [outC]));
            }
            specs.Add(($"{prefix}.residual.0.gamma", [inC]));
            specs.Add(($"{prefix}.residual.2.weight", [outC, inC, 3, 3, 3]));
            specs.Add(($"{prefix}.residual.2.bias", [outC]));
            specs.Add(($"{prefix}.residual.3.gamma", [outC]));
            specs.Add(($"{prefix}.residual.6.weight", [outC, outC, 3, 3, 3]));
            specs.Add(($"{prefix}.residual.6.bias", [outC]));
        }

        AddResNet("decoder.middle.0", 384, 384);
        AddResNet("decoder.middle.2", 384, 384);

        AddResNet("decoder.upsamples.0", 384, 384);
        AddResNet("decoder.upsamples.1", 384, 384);
        AddResNet("decoder.upsamples.2", 384, 384);
        specs.Add(("decoder.upsamples.3.time_conv.weight", [768, 384, 3]));
        specs.Add(("decoder.upsamples.3.time_conv.bias", [768]));
        specs.Add(("decoder.upsamples.3.resample.1.weight", [192, 384, 3, 3]));
        specs.Add(("decoder.upsamples.3.resample.1.bias", [192]));

        AddResNet("decoder.upsamples.4", 192, 384, withShortcut: true);
        AddResNet("decoder.upsamples.5", 384, 384);
        AddResNet("decoder.upsamples.6", 384, 384);
        specs.Add(("decoder.upsamples.7.time_conv.weight", [768, 384, 3]));
        specs.Add(("decoder.upsamples.7.time_conv.bias", [768]));
        specs.Add(("decoder.upsamples.7.resample.1.weight", [192, 384, 3, 3]));
        specs.Add(("decoder.upsamples.7.resample.1.bias", [192]));

        AddResNet("decoder.upsamples.8", 192, 192);
        AddResNet("decoder.upsamples.9", 192, 192);
        AddResNet("decoder.upsamples.10", 192, 192);
        specs.Add(("decoder.upsamples.11.resample.1.weight", [96, 192, 3, 3]));
        specs.Add(("decoder.upsamples.11.resample.1.bias", [96]));

        AddResNet("decoder.upsamples.12", 96, 96);
        AddResNet("decoder.upsamples.13", 96, 96);
        AddResNet("decoder.upsamples.14", 96, 96);

        ulong maxBytes = 0;
        var sb = new StringBuilder("{");
        for (int i = 0; i < specs.Count; i++)
        {
            if (i > 0) sb.Append(',');
            ulong elems = 1;
            foreach (int d in specs[i].Shape) elems *= (ulong)d;
            ulong bytes = elems * (ulong)elemBytes;
            if (bytes > maxBytes) maxBytes = bytes;

            sb.Append(System.Text.Json.JsonSerializer.Serialize(specs[i].Name));
            sb.Append(':');
            sb.Append($$"""{"dtype":"{{dtype}}","shape":[{{string.Join(',', specs[i].Shape)}}],"data_offsets":[0,{{bytes}}]}""");
        }
        sb.Append('}');

        byte[] headerBytes = Encoding.UTF8.GetBytes(sb.ToString());
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);
        bw.Write((ulong)headerBytes.Length);
        bw.Write(headerBytes);

        int count = (int)(maxBytes / (ulong)elemBytes);
        var rng = new Random(12345);
        if (dtype == "F16")
        {
            for (int i = 0; i < count; i++)
            {
                float val = (float)(rng.NextDouble() * 0.02 - 0.01);
                bw.Write(BitConverter.HalfToUInt16Bits((Half)val));
            }
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                bw.Write((float)(rng.NextDouble() * 0.02 - 0.01));
            }
        }
    }
}
