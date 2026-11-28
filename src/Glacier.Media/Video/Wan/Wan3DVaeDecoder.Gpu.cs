namespace Glacier.Media.Video.Wan;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gpu;

public unsafe sealed partial class Wan3DVaeDecoder
{
    private readonly GpuContext? _gpu;
    private readonly IntPtr _gpuModule;
    private readonly IntPtr _fnConv2d3x3;
    private readonly IntPtr _fnConv2d1x1;
    private readonly IntPtr _fnUpsample2x;
    private readonly IntPtr _fnTensorAdd;
    private readonly IntPtr _fnVaeClampRgb;
    private readonly IntPtr _fnWanRmsNormSilu;
    private readonly IntPtr _fnWanSpatialAttn;

    private readonly Dictionary<string, IntPtr> _gpuWeights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (IntPtr W0, IntPtr W1, IntPtr W2)> _gpuDeinterleavedWeights = new(StringComparer.Ordinal);

    public bool IsGpuAccelerated => _gpu != null;
    public string ActiveBackend => _gpu != null ? "Cuda" : "Cpu";

    private sealed class GpuFrame
    {
        public IntPtr Ptr { get; private set; }
        public int RefCount { get; private set; } = 1;

        public GpuFrame(IntPtr ptr)
        {
            Ptr = ptr;
        }

        public GpuFrame Retain()
        {
            RefCount++;
            return this;
        }

        public void Release(GpuContext gpu)
        {
            if (RefCount <= 0) return;
            RefCount--;
            if (RefCount == 0 && Ptr != IntPtr.Zero)
            {
                gpu.FreeDevice(Ptr);
                Ptr = IntPtr.Zero;
            }
        }

        public void ForceFree(GpuContext gpu)
        {
            if (Ptr != IntPtr.Zero)
            {
                gpu.FreeDevice(Ptr);
                Ptr = IntPtr.Zero;
            }
            RefCount = 0;
        }
    }

    private sealed class GpuCausalFeatureCache
    {
        private readonly Dictionary<string, (GpuFrame? T0, GpuFrame? T1)> _slots = new(StringComparer.Ordinal);

        public (GpuFrame? T0, GpuFrame? T1) GetPastFrames(string prefix)
            => _slots.TryGetValue(prefix, out var pair) ? pair : (null, null);

        public void MarkTimeConvRep(string prefix, GpuContext gpu)
        {
            if (_slots.TryGetValue(prefix, out var old))
            {
                old.T0?.Release(gpu);
                old.T1?.Release(gpu);
            }
            _slots[prefix] = (null, null);
        }

        public void Update(string prefix, GpuFrame? t0, GpuFrame? t1, GpuContext gpu)
        {
            t0?.Retain();
            t1?.Retain();
            if (_slots.TryGetValue(prefix, out var old))
            {
                old.T0?.Release(gpu);
                old.T1?.Release(gpu);
            }
            _slots[prefix] = (t0, t1);
        }

        public void Clear(GpuContext gpu)
        {
            foreach (var kvp in _slots.Values)
            {
                kvp.T0?.Release(gpu);
                kvp.T1?.Release(gpu);
            }
            _slots.Clear();
        }
    }

    private static void ReleaseFrames(List<GpuFrame> frames, GpuContext gpu)
    {
        for (int i = 0; i < frames.Count; i++)
        {
            frames[i].Release(gpu);
        }
    }

    private IntPtr GetGpuWeight(string name)
    {
        if (_gpuWeights.TryGetValue(name, out var d)) return d;

        float[] data = LoadTensorFloats(_weights, name);
        nuint bytes = (nuint)((long)data.Length * sizeof(float));
        IntPtr dMem = _gpu!.AllocateDevice(bytes);
        fixed (float* p = data)
        {
            _gpu.CopyToDevice(dMem, (IntPtr)p, bytes);
        }
        _gpuWeights[name] = dMem;
        return dMem;
    }

    private IntPtr? TryGetGpuWeight(string name)
    {
        if (_gpuWeights.TryGetValue(name, out var d)) return d;
        if (!_weights.ContainsTensor(name)) return null;
        return GetGpuWeight(name);
    }

    private (IntPtr W0, IntPtr W1, IntPtr W2) GetGpuDeinterleavedWeight(string name, int outC, int inC, int spatialKernelSize)
    {
        if (_gpuDeinterleavedWeights.TryGetValue(name, out var cached)) return cached;

        float[] raw = LoadTensorFloats(_weights, name);
        int sliceLen = checked(outC * inC * spatialKernelSize);
        int expectedLen = checked(sliceLen * 3);
        if (raw.Length != expectedLen)
            throw new InvalidDataException($"Tensor '{name}' length {raw.Length} does not match expected {expectedLen} ({outC}x{inC}x3x{spatialKernelSize}).");

        float[] w0 = new float[sliceLen];
        float[] w1 = new float[sliceLen];
        float[] w2 = new float[sliceLen];

        int pairs = outC * inC;
        for (int p = 0; p < pairs; p++)
        {
            int srcBase = p * 3 * spatialKernelSize;
            int dstBase = p * spatialKernelSize;
            Array.Copy(raw, srcBase, w0, dstBase, spatialKernelSize);
            Array.Copy(raw, srcBase + spatialKernelSize, w1, dstBase, spatialKernelSize);
            Array.Copy(raw, srcBase + 2 * spatialKernelSize, w2, dstBase, spatialKernelSize);
        }

        nuint sliceBytes = (nuint)((long)sliceLen * sizeof(float));
        IntPtr d0 = _gpu!.AllocateDevice(sliceBytes);
        IntPtr d1 = IntPtr.Zero;
        IntPtr d2 = IntPtr.Zero;
        try
        {
            d1 = _gpu.AllocateDevice(sliceBytes);
            d2 = _gpu.AllocateDevice(sliceBytes);
            fixed (float* p0 = w0, p1 = w1, p2 = w2)
            {
                _gpu.CopyToDevice(d0, (IntPtr)p0, sliceBytes);
                _gpu.CopyToDevice(d1, (IntPtr)p1, sliceBytes);
                _gpu.CopyToDevice(d2, (IntPtr)p2, sliceBytes);
            }
        }
        catch
        {
            if (d0 != IntPtr.Zero) _gpu.FreeDevice(d0);
            if (d1 != IntPtr.Zero) _gpu.FreeDevice(d1);
            if (d2 != IntPtr.Zero) _gpu.FreeDevice(d2);
            throw;
        }

        var tuple = (d0, d1, d2);
        _gpuDeinterleavedWeights[name] = tuple;
        return tuple;
    }

    private void LaunchConv3x3(IntPtr dSrc, IntPtr dDst, IntPtr dW, IntPtr dB, int inC, int outC, int h, int w)
    {
        uint gridX = (uint)((w + 15) / 16);
        uint gridY = (uint)((h + 15) / 16);
        uint gridZ = (uint)outC;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dW;
        pArgs[3] = &dB;
        pArgs[4] = &inC;
        pArgs[5] = &outC;
        pArgs[6] = &h;
        pArgs[7] = &w;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnConv2d3x3, gridX, gridY, gridZ, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(conv2d_3x3)");
    }

    private void LaunchConv1x1(IntPtr dSrc, IntPtr dDst, IntPtr dW, IntPtr dB, int inC, int outC, int spatial)
    {
        uint gridX = (uint)((spatial + 15) / 16);
        uint gridY = (uint)((outC + 15) / 16);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dW;
        pArgs[3] = &dB;
        pArgs[4] = &inC;
        pArgs[5] = &outC;
        pArgs[6] = &spatial;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnConv2d1x1, gridX, gridY, 1, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(conv2d_1x1)");
    }

    private void LaunchWanRmsNormSilu(IntPtr dSrc, IntPtr dDst, IntPtr dGamma, int channels, int spatial, bool applySilu)
    {
        uint gridX = (uint)((spatial + 255) / 256);
        int siluFlag = applySilu ? 1 : 0;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dGamma;
        pArgs[3] = &channels;
        pArgs[4] = &spatial;
        pArgs[5] = &siluFlag;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnWanRmsNormSilu, gridX, 1, 1, 256, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(wan_rmsnorm_silu)");
    }

    private void LaunchUpsample2x(IntPtr dSrc, IntPtr dDst, int channels, int inH, int inW)
    {
        int outW = inW * 2;
        int outH = inH * 2;
        uint gridX = (uint)((outW + 15) / 16);
        uint gridY = (uint)((outH + 15) / 16);
        uint gridZ = (uint)channels;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &channels;
        pArgs[3] = &inH;
        pArgs[4] = &inW;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnUpsample2x, gridX, gridY, gridZ, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(upsample2x_nearest)");
    }

    private void LaunchTensorAdd(IntPtr dTarget, IntPtr dSource, int count)
    {
        uint gridX = (uint)((count + 255) / 256);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dTarget;
        pArgs[1] = &dSource;
        pArgs[2] = &count;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnTensorAdd, gridX, 1, 1, 256, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(tensor_add)");
    }

    private void LaunchWanSpatialAttention(IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dOut, int spatial, int channels)
    {
        if (channels > 384)
            throw new NotSupportedException($"CUDA wan_spatial_attention requires channels <= 384, got {channels}.");

        uint gridX = (uint)spatial;
        float scale = 1.0f / MathF.Sqrt(channels);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dOut;
        pArgs[4] = &spatial;
        pArgs[5] = &channels;
        pArgs[6] = &scale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnWanSpatialAttn, gridX, 1, 1, 128, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(wan_spatial_attention)");
    }

    private void LaunchVaeClampRgb(IntPtr dPlanarFloat, IntPtr dPackedRgb, int h, int w)
    {
        uint gridX = (uint)((w + 15) / 16);
        uint gridY = (uint)((h + 15) / 16);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dPlanarFloat;
        pArgs[1] = &dPackedRgb;
        pArgs[2] = &h;
        pArgs[3] = &w;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVaeClampRgb, gridX, gridY, 1, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(vae_clamp_rgb)");
    }

    private List<byte[]> DecodeVideoGpu(
        ReadOnlySpan<float> spatioTemporalLatents,
        int temporalLatentFrames,
        int targetFrames,
        int latentH,
        int latentW)
    {
        var gpu = _gpu!;
        gpu.MakeCurrent();

        int spatialLatent = latentH * latentW;
        int frameLatentSize = 16 * spatialLatent;
        int finalH = latentH * 8;
        int finalW = latentW * 8;
        int rgbByteCount = finalH * finalW * 3;

        // 1. Denormalize latents on host (16 * H_lat * W_lat per chunk is small)
        float[] z = new float[temporalLatentFrames * frameLatentSize];
        for (int f = 0; f < temporalLatentFrames; f++)
        {
            int fOff = f * frameLatentSize;
            for (int c = 0; c < 16; c++)
            {
                float mean = LatentsMean[c];
                float std = LatentsStd[c];
                int cOff = fOff + c * spatialLatent;
                for (int s = 0; s < spatialLatent; s++)
                {
                    z[cOff + s] = spatioTemporalLatents[cOff + s] * std + mean;
                }
            }
        }

        IntPtr dConv2W = GetGpuWeight("conv2.weight");
        IntPtr dConv2B = GetGpuWeight("conv2.bias");
        IntPtr dLatentTmp = gpu.AllocateDevice((nuint)(frameLatentSize * sizeof(float)));
        IntPtr dRgbDevice = gpu.AllocateDevice((nuint)rgbByteCount);

        var cache = new GpuCausalFeatureCache();
        var tracked = new List<GpuFrame>();
        var outputFrames = new List<byte[]>(targetFrames);

        GpuFrame AllocFrame(int floatCount)
        {
            IntPtr ptr = gpu.AllocateDevice((nuint)((long)floatCount * sizeof(float)));
            var f = new GpuFrame(ptr);
            tracked.Add(f);
            return f;
        }

        try
        {
            for (int chunkIdx = 0; chunkIdx < temporalLatentFrames; chunkIdx++)
            {
                bool firstChunk = (chunkIdx == 0);

                fixed (float* pZ = &z[chunkIdx * frameLatentSize])
                {
                    gpu.CopyToDevice(dLatentTmp, (IntPtr)pZ, (nuint)(frameLatentSize * sizeof(float)));
                }

                GpuFrame postQuant = AllocFrame(frameLatentSize);
                LaunchConv1x1(dLatentTmp, postQuant.Ptr, dConv2W, dConv2B, 16, 16, spatialLatent);

                List<GpuFrame> decodedRgbPlanar = DecodeChunkGpu(postQuant, latentH, latentW, cache, firstChunk, AllocFrame);
                try
                {
                    for (int fIdx = 0; fIdx < decodedRgbPlanar.Count; fIdx++)
                    {
                        LaunchVaeClampRgb(decodedRgbPlanar[fIdx].Ptr, dRgbDevice, finalH, finalW);
                        gpu.Synchronize();

                        byte[] rgb = new byte[rgbByteCount];
                        fixed (byte* pRgb = rgb)
                        {
                            gpu.CopyToHost((IntPtr)pRgb, dRgbDevice, (nuint)rgbByteCount);
                        }
                        outputFrames.Add(rgb);
                        if (outputFrames.Count >= targetFrames) break;
                    }
                }
                finally
                {
                    ReleaseFrames(decodedRgbPlanar, gpu);
                }

                if (outputFrames.Count >= targetFrames) break;
            }
        }
        finally
        {
            cache.Clear(gpu);
            for (int i = 0; i < tracked.Count; i++)
            {
                tracked[i].ForceFree(gpu);
            }
            gpu.FreeDevice(dLatentTmp);
            gpu.FreeDevice(dRgbDevice);
        }

        if (outputFrames.Count > targetFrames)
        {
            return outputFrames.GetRange(0, targetFrames);
        }
        return outputFrames;
    }

    private List<GpuFrame> DecodeChunkGpu(
        GpuFrame chunkInput,
        int latentH,
        int latentW,
        GpuCausalFeatureCache cache,
        bool firstChunk,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        int curH = latentH;
        int curW = latentW;
        int curC = 384;

        List<GpuFrame> frames;
        var initialList = new List<GpuFrame>(1) { chunkInput };
        try
        {
            frames = RunCausalConv3DGpu(initialList, 16, curC, curH, curW, "decoder.conv1", cache, allocFrame);
        }
        finally
        {
            chunkInput.Release(gpu);
        }

        frames = RunResNetBlockGpu(frames, curC, curC, curH, curW, "decoder.middle.0", cache, allocFrame);
        frames = RunSpatialAttentionGpu(frames, curC, curH, curW, "decoder.middle.1", allocFrame);
        frames = RunResNetBlockGpu(frames, curC, curC, curH, curW, "decoder.middle.2", cache, allocFrame);

        frames = RunResNetBlockGpu(frames, 384, 384, curH, curW, "decoder.upsamples.0", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 384, 384, curH, curW, "decoder.upsamples.1", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 384, 384, curH, curW, "decoder.upsamples.2", cache, allocFrame);

        frames = RunResample3DGpu(frames, 384, 192, ref curH, ref curW, "decoder.upsamples.3", cache, firstChunk, allocFrame);
        curC = 192;

        frames = RunResNetBlockWithShortcutGpu(frames, 192, 384, curH, curW, "decoder.upsamples.4", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 384, 384, curH, curW, "decoder.upsamples.5", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 384, 384, curH, curW, "decoder.upsamples.6", cache, allocFrame);

        frames = RunResample3DGpu(frames, 384, 192, ref curH, ref curW, "decoder.upsamples.7", cache, firstChunk, allocFrame);
        curC = 192;

        frames = RunResNetBlockGpu(frames, 192, 192, curH, curW, "decoder.upsamples.8", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 192, 192, curH, curW, "decoder.upsamples.9", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 192, 192, curH, curW, "decoder.upsamples.10", cache, allocFrame);

        frames = RunResample2DGpu(frames, 192, 96, ref curH, ref curW, "decoder.upsamples.11", allocFrame);
        curC = 96;

        frames = RunResNetBlockGpu(frames, 96, 96, curH, curW, "decoder.upsamples.12", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 96, 96, curH, curW, "decoder.upsamples.13", cache, allocFrame);
        frames = RunResNetBlockGpu(frames, 96, 96, curH, curW, "decoder.upsamples.14", cache, allocFrame);

        // Head norm + silu
        IntPtr dNormGamma = GetGpuWeight("decoder.head.0.gamma");
        int spatial = curH * curW;
        var normedFrames = new List<GpuFrame>(frames.Count);
        try
        {
            for (int i = 0; i < frames.Count; i++)
            {
                var normed = allocFrame(curC * spatial);
                LaunchWanRmsNormSilu(frames[i].Ptr, normed.Ptr, dNormGamma, curC, spatial, applySilu: true);
                normedFrames.Add(normed);
            }
        }
        finally
        {
            ReleaseFrames(frames, gpu);
        }

        try
        {
            return RunCausalConv3DGpu(normedFrames, 96, 3, curH, curW, "decoder.head.2", cache, allocFrame);
        }
        finally
        {
            ReleaseFrames(normedFrames, gpu);
        }
    }

    private List<GpuFrame> RunCausalConv3DGpu(
        List<GpuFrame> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        GpuCausalFeatureCache cache,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        var (cachedT0, cachedT1) = cache.GetPastFrames(prefix);
        var (dW0, dW1, dW2) = GetGpuDeinterleavedWeight($"{prefix}.weight", outC, inC, 9);
        IntPtr dB = TryGetGpuWeight($"{prefix}.bias") ?? IntPtr.Zero;

        int count = inputFrames.Count;
        int outFloats = outC * H * W;
        var output = new List<GpuFrame>(count);
        GpuFrame? accumTmp = (cachedT0 != null || cachedT1 != null || count > 1) ? allocFrame(outFloats) : null;

        try
        {
            for (int k = 0; k < count; k++)
            {
                GpuFrame? t0 = k == 0 ? cachedT0 : (k == 1 ? cachedT1 : inputFrames[k - 2]);
                GpuFrame? t1 = k == 0 ? cachedT1 : inputFrames[k - 1];
                GpuFrame t2 = inputFrames[k];

                GpuFrame dst = allocFrame(outFloats);
                LaunchConv3x3(t2.Ptr, dst.Ptr, dW2, dB, inC, outC, H, W);

                if (t1 != null)
                {
                    LaunchConv3x3(t1.Ptr, accumTmp!.Ptr, dW1, IntPtr.Zero, inC, outC, H, W);
                    LaunchTensorAdd(dst.Ptr, accumTmp.Ptr, outFloats);
                }

                if (t0 != null)
                {
                    LaunchConv3x3(t0.Ptr, accumTmp!.Ptr, dW0, IntPtr.Zero, inC, outC, H, W);
                    LaunchTensorAdd(dst.Ptr, accumTmp.Ptr, outFloats);
                }

                output.Add(dst);
            }
        }
        finally
        {
            accumTmp?.Release(gpu);
        }

        if (count == 1)
        {
            cache.Update(prefix, cachedT1, inputFrames[0], gpu);
        }
        else if (count >= 2)
        {
            cache.Update(prefix, inputFrames[^2], inputFrames[^1], gpu);
        }

        return output;
    }

    private List<GpuFrame> RunResNetBlockGpu(
        List<GpuFrame> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        GpuCausalFeatureCache cache,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        int spatial = H * W;

        IntPtr dNorm1Gamma = GetGpuWeight($"{prefix}.residual.0.gamma");
        var norm1Frames = new List<GpuFrame>(inputFrames.Count);
        for (int i = 0; i < inputFrames.Count; i++)
        {
            var n1 = allocFrame(inC * spatial);
            LaunchWanRmsNormSilu(inputFrames[i].Ptr, n1.Ptr, dNorm1Gamma, inC, spatial, applySilu: true);
            norm1Frames.Add(n1);
        }

        List<GpuFrame> conv1Frames;
        try
        {
            conv1Frames = RunCausalConv3DGpu(norm1Frames, inC, outC, H, W, $"{prefix}.residual.2", cache, allocFrame);
        }
        finally
        {
            ReleaseFrames(norm1Frames, gpu);
        }

        IntPtr dNorm2Gamma = GetGpuWeight($"{prefix}.residual.3.gamma");
        var norm2Frames = new List<GpuFrame>(conv1Frames.Count);
        try
        {
            for (int i = 0; i < conv1Frames.Count; i++)
            {
                var n2 = allocFrame(outC * spatial);
                LaunchWanRmsNormSilu(conv1Frames[i].Ptr, n2.Ptr, dNorm2Gamma, outC, spatial, applySilu: true);
                norm2Frames.Add(n2);
            }
        }
        finally
        {
            ReleaseFrames(conv1Frames, gpu);
        }

        List<GpuFrame> conv2Frames;
        try
        {
            conv2Frames = RunCausalConv3DGpu(norm2Frames, outC, outC, H, W, $"{prefix}.residual.6", cache, allocFrame);
        }
        finally
        {
            ReleaseFrames(norm2Frames, gpu);
        }

        int totalFloats = outC * spatial;
        for (int k = 0; k < conv2Frames.Count; k++)
        {
            LaunchTensorAdd(conv2Frames[k].Ptr, inputFrames[k].Ptr, totalFloats);
        }
        ReleaseFrames(inputFrames, gpu);

        return conv2Frames;
    }

    private List<GpuFrame> RunResNetBlockWithShortcutGpu(
        List<GpuFrame> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        GpuCausalFeatureCache cache,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        int spatial = H * W;
        IntPtr dShortW = GetGpuWeight($"{prefix}.shortcut.weight");
        IntPtr dShortB = TryGetGpuWeight($"{prefix}.shortcut.bias") ?? IntPtr.Zero;

        var shortcutFrames = new List<GpuFrame>(inputFrames.Count);
        for (int i = 0; i < inputFrames.Count; i++)
        {
            var sc = allocFrame(outC * spatial);
            LaunchConv1x1(inputFrames[i].Ptr, sc.Ptr, dShortW, dShortB, inC, outC, spatial);
            shortcutFrames.Add(sc);
        }

        IntPtr dNorm1Gamma = GetGpuWeight($"{prefix}.residual.0.gamma");
        var norm1Frames = new List<GpuFrame>(inputFrames.Count);
        try
        {
            for (int i = 0; i < inputFrames.Count; i++)
            {
                var n1 = allocFrame(inC * spatial);
                LaunchWanRmsNormSilu(inputFrames[i].Ptr, n1.Ptr, dNorm1Gamma, inC, spatial, applySilu: true);
                norm1Frames.Add(n1);
            }
        }
        finally
        {
            ReleaseFrames(inputFrames, gpu);
        }

        List<GpuFrame> conv1Frames;
        try
        {
            conv1Frames = RunCausalConv3DGpu(norm1Frames, inC, outC, H, W, $"{prefix}.residual.2", cache, allocFrame);
        }
        finally
        {
            ReleaseFrames(norm1Frames, gpu);
        }

        IntPtr dNorm2Gamma = GetGpuWeight($"{prefix}.residual.3.gamma");
        var norm2Frames = new List<GpuFrame>(conv1Frames.Count);
        try
        {
            for (int i = 0; i < conv1Frames.Count; i++)
            {
                var n2 = allocFrame(outC * spatial);
                LaunchWanRmsNormSilu(conv1Frames[i].Ptr, n2.Ptr, dNorm2Gamma, outC, spatial, applySilu: true);
                norm2Frames.Add(n2);
            }
        }
        finally
        {
            ReleaseFrames(conv1Frames, gpu);
        }

        List<GpuFrame> conv2Frames;
        try
        {
            conv2Frames = RunCausalConv3DGpu(norm2Frames, outC, outC, H, W, $"{prefix}.residual.6", cache, allocFrame);
        }
        finally
        {
            ReleaseFrames(norm2Frames, gpu);
        }

        int totalFloats = outC * spatial;
        for (int k = 0; k < conv2Frames.Count; k++)
        {
            LaunchTensorAdd(conv2Frames[k].Ptr, shortcutFrames[k].Ptr, totalFloats);
        }
        ReleaseFrames(shortcutFrames, gpu);

        return conv2Frames;
    }

    private List<GpuFrame> RunSpatialAttentionGpu(
        List<GpuFrame> inputFrames,
        int channels,
        int H,
        int W,
        string prefix,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        int spatial = H * W;
        IntPtr dNormGamma = GetGpuWeight($"{prefix}.norm.gamma");
        IntPtr dQkvW = GetGpuWeight($"{prefix}.to_qkv.weight");
        IntPtr dQkvB = TryGetGpuWeight($"{prefix}.to_qkv.bias") ?? IntPtr.Zero;
        IntPtr dProjW = GetGpuWeight($"{prefix}.proj.weight");
        IntPtr dProjB = TryGetGpuWeight($"{prefix}.proj.bias") ?? IntPtr.Zero;

        var outputFrames = new List<GpuFrame>(inputFrames.Count);
        try
        {
            for (int i = 0; i < inputFrames.Count; i++)
            {
                var frame = inputFrames[i];
                var normed = allocFrame(channels * spatial);
                var qkv = allocFrame(channels * 3 * spatial);
                var attnOut = allocFrame(channels * spatial);
                var projOut = allocFrame(channels * spatial);

                try
                {
                    LaunchWanRmsNormSilu(frame.Ptr, normed.Ptr, dNormGamma, channels, spatial, applySilu: false);
                    LaunchConv1x1(normed.Ptr, qkv.Ptr, dQkvW, dQkvB, channels, channels * 3, spatial);

                    nint planeBytes = (nint)((long)channels * spatial * sizeof(float));
                    IntPtr dQ = qkv.Ptr;
                    IntPtr dK = qkv.Ptr + planeBytes;
                    IntPtr dV = qkv.Ptr + planeBytes * 2;

                    LaunchWanSpatialAttention(dQ, dK, dV, attnOut.Ptr, spatial, channels);
                    LaunchConv1x1(attnOut.Ptr, projOut.Ptr, dProjW, dProjB, channels, channels, spatial);
                    LaunchTensorAdd(projOut.Ptr, frame.Ptr, channels * spatial);

                    outputFrames.Add(projOut);
                }
                catch
                {
                    projOut.Release(gpu);
                    throw;
                }
                finally
                {
                    normed.Release(gpu);
                    qkv.Release(gpu);
                    attnOut.Release(gpu);
                }
            }
        }
        finally
        {
            ReleaseFrames(inputFrames, gpu);
        }

        return outputFrames;
    }

    private List<GpuFrame> RunResample3DGpu(
        List<GpuFrame> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix,
        GpuCausalFeatureCache cache,
        bool firstChunk,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        var timeSlices = new List<GpuFrame>();
        int spatial = H * W;

        if (firstChunk)
        {
            cache.MarkTimeConvRep(prefix, gpu);
            for (int i = 0; i < inputFrames.Count; i++)
            {
                timeSlices.Add(inputFrames[i]);
            }
        }
        else
        {
            var (cachedT0, cachedT1) = cache.GetPastFrames(prefix);
            var (dW0, dW1, dW2) = GetGpuDeinterleavedWeight($"{prefix}.time_conv.weight", inC * 2, inC, 1);
            IntPtr dB = TryGetGpuWeight($"{prefix}.time_conv.bias") ?? IntPtr.Zero;

            int outFloats = inC * 2 * spatial;
            int frameFloats = inC * spatial;
            nuint frameBytes = (nuint)((long)frameFloats * sizeof(float));
            GpuFrame? accumTmp = (cachedT0 != null || cachedT1 != null || inputFrames.Count > 1) ? allocFrame(outFloats) : null;
            GpuFrame timeConvDst = allocFrame(outFloats);

            try
            {
                for (int k = 0; k < inputFrames.Count; k++)
                {
                    GpuFrame? t0 = (k == 0) ? cachedT0 : ((k == 1) ? cachedT1 : inputFrames[k - 2]);
                    GpuFrame? t1 = (k == 0) ? cachedT1 : inputFrames[k - 1];
                    GpuFrame t2 = inputFrames[k];

                    LaunchConv1x1(t2.Ptr, timeConvDst.Ptr, dW2, dB, inC, inC * 2, spatial);
                    if (t1 != null)
                    {
                        LaunchConv1x1(t1.Ptr, accumTmp!.Ptr, dW1, IntPtr.Zero, inC, inC * 2, spatial);
                        LaunchTensorAdd(timeConvDst.Ptr, accumTmp.Ptr, outFloats);
                    }
                    if (t0 != null)
                    {
                        LaunchConv1x1(t0.Ptr, accumTmp!.Ptr, dW0, IntPtr.Zero, inC, inC * 2, spatial);
                        LaunchTensorAdd(timeConvDst.Ptr, accumTmp.Ptr, outFloats);
                    }

                    GpuFrame frame0 = allocFrame(frameFloats);
                    GpuFrame frame1 = allocFrame(frameFloats);
                    gpu.MakeCurrent();
                    CuDriver.Check(CuDriver.MemcpyDtoD(frame0.Ptr, timeConvDst.Ptr, frameBytes), "cuMemcpyDtoD(frame0)");
                    CuDriver.Check(CuDriver.MemcpyDtoD(frame1.Ptr, timeConvDst.Ptr + (nint)frameBytes, frameBytes), "cuMemcpyDtoD(frame1)");
                    timeSlices.Add(frame0);
                    timeSlices.Add(frame1);
                }

                if (inputFrames.Count == 1)
                {
                    cache.Update(prefix, cachedT1, inputFrames[0], gpu);
                }
                else if (inputFrames.Count >= 2)
                {
                    cache.Update(prefix, inputFrames[^2], inputFrames[^1], gpu);
                }
            }
            finally
            {
                accumTmp?.Release(gpu);
                timeConvDst.Release(gpu);
                ReleaseFrames(inputFrames, gpu);
            }
        }

        int outH = H * 2;
        int outW = W * 2;
        IntPtr dConvW = GetGpuWeight($"{prefix}.resample.1.weight");
        IntPtr dConvB = TryGetGpuWeight($"{prefix}.resample.1.bias") ?? IntPtr.Zero;

        var result = new List<GpuFrame>(timeSlices.Count);
        try
        {
            for (int i = 0; i < timeSlices.Count; i++)
            {
                var upsampled = allocFrame(inC * outH * outW);
                var convOut = allocFrame(outC * outH * outW);
                try
                {
                    LaunchUpsample2x(timeSlices[i].Ptr, upsampled.Ptr, inC, H, W);
                    LaunchConv3x3(upsampled.Ptr, convOut.Ptr, dConvW, dConvB, inC, outC, outH, outW);
                    result.Add(convOut);
                }
                catch
                {
                    convOut.Release(gpu);
                    throw;
                }
                finally
                {
                    upsampled.Release(gpu);
                }
            }
        }
        finally
        {
            ReleaseFrames(timeSlices, gpu);
        }

        H = outH;
        W = outW;
        return result;
    }

    private List<GpuFrame> RunResample2DGpu(
        List<GpuFrame> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix,
        Func<int, GpuFrame> allocFrame)
    {
        var gpu = _gpu!;
        int outH = H * 2;
        int outW = W * 2;
        IntPtr dConvW = GetGpuWeight($"{prefix}.resample.1.weight");
        IntPtr dConvB = TryGetGpuWeight($"{prefix}.resample.1.bias") ?? IntPtr.Zero;

        var result = new List<GpuFrame>(inputFrames.Count);
        try
        {
            for (int i = 0; i < inputFrames.Count; i++)
            {
                var upsampled = allocFrame(inC * outH * outW);
                var convOut = allocFrame(outC * outH * outW);
                try
                {
                    LaunchUpsample2x(inputFrames[i].Ptr, upsampled.Ptr, inC, H, W);
                    LaunchConv3x3(upsampled.Ptr, convOut.Ptr, dConvW, dConvB, inC, outC, outH, outW);
                    result.Add(convOut);
                }
                catch
                {
                    convOut.Release(gpu);
                    throw;
                }
                finally
                {
                    upsampled.Release(gpu);
                }
            }
        }
        finally
        {
            ReleaseFrames(inputFrames, gpu);
        }

        H = outH;
        W = outW;
        return result;
    }
}
