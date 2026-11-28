namespace Glacier.Media.Video.Wan;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

public unsafe sealed partial class Wan3DVaeDecoder
{
    private List<float[]> RunResample3D(
        List<float[]> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix,
        CausalFeatureCache cache,
        bool firstChunk)
    {
        List<float[]> timeSlices = [];

        if (firstChunk)
        {
            // On chunk 0, time_conv is bypassed, keeping t=1 frame
            cache.MarkTimeConvRep(prefix);
            timeSlices.AddRange(inputFrames);
        }
        else
        {
            // On subsequent chunks, time_conv doubles time dimension t -> 2t
            var (cachedT0, cachedT1) = cache.GetPastFrames(prefix);
            float[] timeConvW = GetW($"{prefix}.time_conv.weight");
            float[]? timeConvB = TryGetW($"{prefix}.time_conv.bias");

            for (int k = 0; k < inputFrames.Count; k++)
            {
                float[]? t0 = (k == 0) ? cachedT0 : ((k == 1) ? cachedT1 : inputFrames[k - 2]);
                float[]? t1 = (k == 0) ? cachedT1 : inputFrames[k - 1];
                float[] t2 = inputFrames[k];

                float[] timeConvDst = new float[inC * 2 * H * W];
                fixed (float* pT0 = t0, pT1 = t1, pT2 = t2, pW = timeConvW, pB = timeConvB, pDst = timeConvDst)
                {
                    Conv1x1_3D_Causal(pT0, pT1, pT2, pDst, pW, pB, inC, inC * 2, H, W);
                }

                // Channel interleave: lower inC channels -> Frame 0, upper inC channels -> Frame 1
                int frameSize = inC * H * W;
                float[] frame0 = new float[frameSize];
                float[] frame1 = new float[frameSize];
                Array.Copy(timeConvDst, 0, frame0, 0, frameSize);
                Array.Copy(timeConvDst, frameSize, frame1, 0, frameSize);

                timeSlices.Add(frame0);
                timeSlices.Add(frame1);
            }

            if (inputFrames.Count == 1)
            {
                cache.Update(prefix, cachedT1, inputFrames[0]);
            }
            else
            {
                cache.Update(prefix, inputFrames[^2], inputFrames[^1]);
            }
        }

        // Spatial 2x upsampling on each temporal slice
        int outH = H * 2;
        int outW = W * 2;
        float[] convW = GetW($"{prefix}.resample.1.weight");
        float[]? convB = TryGetW($"{prefix}.resample.1.bias");

        List<float[]> result = new(timeSlices.Count);
        foreach (var slice in timeSlices)
        {
            float[] upsampled = new float[inC * outH * outW];
            fixed (float* pSrc = slice, pDst = upsampled)
            {
                Upsample2x_Nearest(pSrc, pDst, inC, H, W);
            }

            float[] convOut = new float[outC * outH * outW];
            fixed (float* pSrc = upsampled, pDst = convOut, pW = convW, pB = convB)
            {
                Conv2D_3x3(pSrc, pDst, pW, pB, inC, outC, outH, outW);
            }
            result.Add(convOut);
        }

        H = outH;
        W = outW;
        return result;
    }

    private List<float[]> RunResample2D(
        List<float[]> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix)
    {
        int outH = H * 2;
        int outW = W * 2;
        float[] convW = GetW($"{prefix}.resample.1.weight");
        float[]? convB = TryGetW($"{prefix}.resample.1.bias");

        List<float[]> result = new(inputFrames.Count);
        foreach (var frame in inputFrames)
        {
            float[] upsampled = new float[inC * outH * outW];
            fixed (float* pSrc = frame, pDst = upsampled)
            {
                Upsample2x_Nearest(pSrc, pDst, inC, H, W);
            }

            float[] convOut = new float[outC * outH * outW];
            fixed (float* pSrc = upsampled, pDst = convOut, pW = convW, pB = convB)
            {
                Conv2D_3x3(pSrc, pDst, pW, pB, inC, outC, outH, outW);
            }
            result.Add(convOut);
        }

        H = outH;
        W = outW;
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WanRMSNorm(float* src, float* dst, float* gamma, int channels, int spatial, bool applySilu)
    {
        float scale = MathF.Sqrt(channels);
        const int tileSize = 1024;
        int numTiles = (spatial + tileSize - 1) / tileSize;

        Parallel.For(0, numTiles, tileIdx =>
        {
            int sStart = tileIdx * tileSize;
            int sEnd = Math.Min(sStart + tileSize, spatial);
            int curTile = sEnd - sStart;

            float* sumSq = stackalloc float[curTile];
            new Span<float>(sumSq, curTile).Clear();

            for (int c = 0; c < channels; c++)
            {
                float* pSrcC = src + (long)c * spatial + sStart;
                if (Vector256.IsHardwareAccelerated)
                {
                    int i = 0;
                    for (; i <= curTile - 8; i += 8)
                    {
                        var vSrc = Vector256.Load(pSrcC + i);
                        var vSum = Vector256.Load(sumSq + i);
                        vSum = Vector256.FusedMultiplyAdd(vSrc, vSrc, vSum);
                        vSum.Store(sumSq + i);
                    }
                    for (; i < curTile; i++)
                    {
                        float v = pSrcC[i];
                        sumSq[i] += v * v;
                    }
                }
                else
                {
                    for (int i = 0; i < curTile; i++)
                    {
                        float v = pSrcC[i];
                        sumSq[i] += v * v;
                    }
                }
            }

            float* invRms = stackalloc float[curTile];
            for (int i = 0; i < curTile; i++)
            {
                invRms[i] = (1.0f / MathF.Sqrt(sumSq[i] + 1e-12f)) * scale;
            }

            for (int c = 0; c < channels; c++)
            {
                float g = gamma[c];
                float* pSrcC = src + (long)c * spatial + sStart;
                float* pDstC = dst + (long)c * spatial + sStart;

                if (Vector256.IsHardwareAccelerated && !applySilu)
                {
                    var vG = Vector256.Create(g);
                    int i = 0;
                    for (; i <= curTile - 8; i += 8)
                    {
                        var vSrc = Vector256.Load(pSrcC + i);
                        var vInv = Vector256.Load(invRms + i);
                        var vNorm = Vector256.Multiply(Vector256.Multiply(vSrc, vInv), vG);
                        vNorm.Store(pDstC + i);
                    }
                    for (; i < curTile; i++)
                    {
                        pDstC[i] = pSrcC[i] * invRms[i] * g;
                    }
                }
                else
                {
                    for (int i = 0; i < curTile; i++)
                    {
                        float norm = pSrcC[i] * invRms[i] * g;
                        if (applySilu)
                        {
                            norm /= (1.0f + MathF.Exp(-norm));
                        }
                        pDstC[i] = norm;
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateConv3x3Row(float* pSrc, float* pDstRow, float* w, int y, int H, int W)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int yin = y + dy;
            if (yin < 0 || yin >= H) continue;

            float* pSrcRow = pSrc + (long)yin * W;
            float* wRow = w + (dy + 1) * 3;

            float w0 = wRow[0];
            float w1 = wRow[1];
            float w2 = wRow[2];

            if (w0 == 0f && w1 == 0f && w2 == 0f) continue;

            if (Vector256.IsHardwareAccelerated && W >= 16)
            {
                var vW0 = Vector256.Create(w0);
                var vW1 = Vector256.Create(w1);
                var vW2 = Vector256.Create(w2);

                pDstRow[0] += pSrcRow[0] * w1 + pSrcRow[1] * w2;

                int x = 1;
                int xEnd = W - 9;
                for (; x <= xEnd; x += 8)
                {
                    var vDst = Vector256.Load(pDstRow + x);
                    var vLeft = Vector256.Load(pSrcRow + x - 1);
                    var vMid  = Vector256.Load(pSrcRow + x);
                    var vRight = Vector256.Load(pSrcRow + x + 1);

                    vDst = Vector256.FusedMultiplyAdd(vLeft, vW0, vDst);
                    vDst = Vector256.FusedMultiplyAdd(vMid, vW1, vDst);
                    vDst = Vector256.FusedMultiplyAdd(vRight, vW2, vDst);

                    vDst.Store(pDstRow + x);
                }

                for (; x < W - 1; x++)
                {
                    pDstRow[x] += pSrcRow[x - 1] * w0 + pSrcRow[x] * w1 + pSrcRow[x + 1] * w2;
                }

                pDstRow[W - 1] += pSrcRow[W - 2] * w0 + pSrcRow[W - 1] * w1;
            }
            else
            {
                pDstRow[0] += pSrcRow[0] * w1 + (W > 1 ? pSrcRow[1] * w2 : 0f);
                for (int x = 1; x < W - 1; x++)
                {
                    pDstRow[x] += pSrcRow[x - 1] * w0 + pSrcRow[x] * w1 + pSrcRow[x + 1] * w2;
                }
                if (W > 1) pDstRow[W - 1] += pSrcRow[W - 2] * w0 + pSrcRow[W - 1] * w1;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv3D_Causal(
        float* pT0, float* pT1, float* pT2, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 27;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int y = 0; y < H; y++)
            {
                float* pDstRow = pDstOc + (long)y * W;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vB = Vector256.Create(b);
                    int x = 0;
                    for (; x <= W - 8; x += 8) vB.Store(pDstRow + x);
                    for (; x < W; x++) pDstRow[x] = b;
                }
                else
                {
                    for (int x = 0; x < W; x++) pDstRow[x] = b;
                }

                for (int ic = 0; ic < inC; ic++)
                {
                    float* pWIc = pWOc + ic * 27;
                    long icSpatial = (long)ic * spatial;

                    float* pSrcT0 = pT0 != null ? pT0 + icSpatial : null;
                    float* pSrcT1 = pT1 != null ? pT1 + icSpatial : null;
                    float* pSrcT2 = pT2 + icSpatial;

                    if (pSrcT0 != null)
                    {
                        AccumulateConv3x3Row(pSrcT0, pDstRow, pWIc + 0, y, H, W);
                    }
                    if (pSrcT1 != null)
                    {
                        AccumulateConv3x3Row(pSrcT1, pDstRow, pWIc + 9, y, H, W);
                    }
                    {
                        AccumulateConv3x3Row(pSrcT2, pDstRow, pWIc + 18, y, H, W);
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv1x1_3D_Causal(
        float* pT0, float* pT1, float* pT2, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 3;
            float* pDstOc = pDst + (long)oc * spatial;

            if (Vector256.IsHardwareAccelerated)
            {
                var vB = Vector256.Create(b);
                int s = 0;
                for (; s <= spatial - 8; s += 8) vB.Store(pDstOc + s);
                for (; s < spatial; s++) pDstOc[s] = b;
            }
            else
            {
                for (int s = 0; s < spatial; s++) pDstOc[s] = b;
            }

            for (int ic = 0; ic < inC; ic++)
            {
                float* wIc = pWOc + ic * 3;
                long icSpatial = (long)ic * spatial;

                float w0 = wIc[0];
                float w1 = wIc[1];
                float w2 = wIc[2];

                float* pSrcT0 = pT0 != null ? pT0 + icSpatial : null;
                float* pSrcT1 = pT1 != null ? pT1 + icSpatial : null;
                float* pSrcT2 = pT2 + icSpatial;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vW0 = pSrcT0 != null ? Vector256.Create(w0) : Vector256<float>.Zero;
                    var vW1 = pSrcT1 != null ? Vector256.Create(w1) : Vector256<float>.Zero;
                    var vW2 = Vector256.Create(w2);

                    int s = 0;
                    for (; s <= spatial - 8; s += 8)
                    {
                        var vDst = Vector256.Load(pDstOc + s);
                        if (pSrcT0 != null)
                        {
                            var v0 = Vector256.Load(pSrcT0 + s);
                            vDst = Vector256.FusedMultiplyAdd(v0, vW0, vDst);
                        }
                        if (pSrcT1 != null)
                        {
                            var v1 = Vector256.Load(pSrcT1 + s);
                            vDst = Vector256.FusedMultiplyAdd(v1, vW1, vDst);
                        }
                        var v2 = Vector256.Load(pSrcT2 + s);
                        vDst = Vector256.FusedMultiplyAdd(v2, vW2, vDst);
                        vDst.Store(pDstOc + s);
                    }
                    for (; s < spatial; s++)
                    {
                        float sum = 0f;
                        if (pSrcT0 != null) sum += pSrcT0[s] * w0;
                        if (pSrcT1 != null) sum += pSrcT1[s] * w1;
                        sum += pSrcT2[s] * w2;
                        pDstOc[s] += sum;
                    }
                }
                else
                {
                    for (int s = 0; s < spatial; s++)
                    {
                        float sum = 0f;
                        if (pSrcT0 != null) sum += pSrcT0[s] * w0;
                        if (pSrcT1 != null) sum += pSrcT1[s] * w1;
                        sum += pSrcT2[s] * w2;
                        pDstOc[s] += sum;
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2D_1x1(
        float* pSrc, float* pDst, float* pW, float* pB,
        int inC, int outC, int spatial)
    {
        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC;
            float* pDstOc = pDst + (long)oc * spatial;

            if (Vector256.IsHardwareAccelerated)
            {
                var vB = Vector256.Create(b);
                int s = 0;
                for (; s <= spatial - 8; s += 8) vB.Store(pDstOc + s);
                for (; s < spatial; s++) pDstOc[s] = b;
            }
            else
            {
                for (int s = 0; s < spatial; s++) pDstOc[s] = b;
            }

            for (int ic = 0; ic < inC; ic++)
            {
                float w = pWOc[ic];
                if (w == 0f) continue;
                float* pSrcIc = pSrc + (long)ic * spatial;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vW = Vector256.Create(w);
                    int s = 0;
                    for (; s <= spatial - 8; s += 8)
                    {
                        var vDst = Vector256.Load(pDstOc + s);
                        var vSrc = Vector256.Load(pSrcIc + s);
                        vDst = Vector256.FusedMultiplyAdd(vSrc, vW, vDst);
                        vDst.Store(pDstOc + s);
                    }
                    for (; s < spatial; s++)
                    {
                        pDstOc[s] += pSrcIc[s] * w;
                    }
                }
                else
                {
                    for (int s = 0; s < spatial; s++)
                    {
                        pDstOc[s] += pSrcIc[s] * w;
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2D_3x3(
        float* pSrc, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 9;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int y = 0; y < H; y++)
            {
                float* pDstRow = pDstOc + (long)y * W;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vB = Vector256.Create(b);
                    int x = 0;
                    for (; x <= W - 8; x += 8) vB.Store(pDstRow + x);
                    for (; x < W; x++) pDstRow[x] = b;
                }
                else
                {
                    for (int x = 0; x < W; x++) pDstRow[x] = b;
                }

                for (int ic = 0; ic < inC; ic++)
                {
                    float* wIc = pWOc + ic * 9;
                    float* pSrcIc = pSrc + (long)ic * spatial;
                    AccumulateConv3x3Row(pSrcIc, pDstRow, wIc, y, H, W);
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddInPlaceSimd(float* dst, float* src, int count)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && count >= 16)
        {
            for (; i <= count - 16; i += 16)
            {
                (Vector512.Load(dst + i) + Vector512.Load(src + i)).Store(dst + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= count - 8)
        {
            for (; i <= count - 8; i += 8)
            {
                (Vector256.Load(dst + i) + Vector256.Load(src + i)).Store(dst + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= count - 4)
        {
            for (; i <= count - 4; i += 4)
            {
                (Vector128.Load(dst + i) + Vector128.Load(src + i)).Store(dst + i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Upsample2x_Nearest(float* src, float* dst, int channels, int inH, int inW)
    {
        int outH = inH * 2;
        int outW = inW * 2;
        int inSpatial = inH * inW;
        int outSpatial = outH * outW;
        nuint rowBytes = (nuint)((long)outW * sizeof(float));

        Parallel.For(0, channels, c =>
        {
            float* pSrcC = src + (long)c * inSpatial;
            float* pDstC = dst + (long)c * outSpatial;

            for (int inY = 0; inY < inH; inY++)
            {
                float* pSrcRow = pSrcC + (long)inY * inW;
                float* pDstRow0 = pDstC + (long)(inY * 2) * outW;
                float* pDstRow1 = pDstRow0 + outW;

                int x = 0;
                if (Vector256.IsHardwareAccelerated && inW >= 8)
                {
                    var idxLo = Vector128.Create(0, 0, 1, 1);
                    var idxHi = Vector128.Create(2, 2, 3, 3);
                    for (; x <= inW - 8; x += 8)
                    {
                        var v = Vector256.Load(pSrcRow + x);
                        Vector256.Create(Vector128.Shuffle(v.GetLower(), idxLo), Vector128.Shuffle(v.GetLower(), idxHi)).Store(pDstRow0 + x * 2);
                        Vector256.Create(Vector128.Shuffle(v.GetUpper(), idxLo), Vector128.Shuffle(v.GetUpper(), idxHi)).Store(pDstRow0 + x * 2 + 8);
                    }
                }
                if (Vector128.IsHardwareAccelerated && x <= inW - 4)
                {
                    var idxLo = Vector128.Create(0, 0, 1, 1);
                    var idxHi = Vector128.Create(2, 2, 3, 3);
                    for (; x <= inW - 4; x += 4)
                    {
                        var v = Vector128.Load(pSrcRow + x);
                        Vector128.Shuffle(v, idxLo).Store(pDstRow0 + x * 2);
                        Vector128.Shuffle(v, idxHi).Store(pDstRow0 + x * 2 + 4);
                    }
                }
                for (; x < inW; x++)
                {
                    float val = pSrcRow[x];
                    pDstRow0[x * 2] = val;
                    pDstRow0[x * 2 + 1] = val;
                }

                Buffer.MemoryCopy(pDstRow0, pDstRow1, rowBytes, rowBytes);
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ClampToRgb(float* srcPlanar, byte* dstRgb, int H, int W)
    {
        int spatial = H * W;
        const int tileSize = 1024;
        int numTiles = (spatial + tileSize - 1) / tileSize;

        Parallel.For(0, numTiles, tileIdx =>
        {
            int sStart = tileIdx * tileSize;
            int sEnd = Math.Min(sStart + tileSize, spatial);
            int s = sStart;

            if (Vector256.IsHardwareAccelerated && sEnd - sStart >= 8)
            {
                var vHalf = Vector256.Create(0.5f);
                var vScale = Vector256.Create(255.0f);
                var vZero = Vector256<float>.Zero;
                int* tmpR = stackalloc int[8];
                int* tmpG = stackalloc int[8];
                int* tmpB = stackalloc int[8];

                for (; s <= sEnd - 8; s += 8)
                {
                    var vr = Vector256.FusedMultiplyAdd(Vector256.Load(srcPlanar + s), vHalf, vHalf) * vScale;
                    var vg = Vector256.FusedMultiplyAdd(Vector256.Load(srcPlanar + spatial + s), vHalf, vHalf) * vScale;
                    var vb = Vector256.FusedMultiplyAdd(Vector256.Load(srcPlanar + (long)spatial * 2 + s), vHalf, vHalf) * vScale;

                    vr = Vector256.Min(Vector256.Max(vr, vZero), vScale);
                    vg = Vector256.Min(Vector256.Max(vg, vZero), vScale);
                    vb = Vector256.Min(Vector256.Max(vb, vZero), vScale);

                    Vector256.ConvertToInt32(vr).Store(tmpR);
                    Vector256.ConvertToInt32(vg).Store(tmpG);
                    Vector256.ConvertToInt32(vb).Store(tmpB);

                    for (int k = 0; k < 8; k++)
                    {
                        int dstIdx = (s + k) * 3;
                        dstRgb[dstIdx] = (byte)tmpR[k];
                        dstRgb[dstIdx + 1] = (byte)tmpG[k];
                        dstRgb[dstIdx + 2] = (byte)tmpB[k];
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated && sEnd - sStart >= 4)
            {
                var vHalf = Vector128.Create(0.5f);
                var vScale = Vector128.Create(255.0f);
                var vZero = Vector128<float>.Zero;
                int* tmpR = stackalloc int[4];
                int* tmpG = stackalloc int[4];
                int* tmpB = stackalloc int[4];

                for (; s <= sEnd - 4; s += 4)
                {
                    var vr = Vector128.FusedMultiplyAdd(Vector128.Load(srcPlanar + s), vHalf, vHalf) * vScale;
                    var vg = Vector128.FusedMultiplyAdd(Vector128.Load(srcPlanar + spatial + s), vHalf, vHalf) * vScale;
                    var vb = Vector128.FusedMultiplyAdd(Vector128.Load(srcPlanar + (long)spatial * 2 + s), vHalf, vHalf) * vScale;

                    vr = Vector128.Min(Vector128.Max(vr, vZero), vScale);
                    vg = Vector128.Min(Vector128.Max(vg, vZero), vScale);
                    vb = Vector128.Min(Vector128.Max(vb, vZero), vScale);

                    Vector128.ConvertToInt32(vr).Store(tmpR);
                    Vector128.ConvertToInt32(vg).Store(tmpG);
                    Vector128.ConvertToInt32(vb).Store(tmpB);

                    for (int k = 0; k < 4; k++)
                    {
                        int dstIdx = (s + k) * 3;
                        dstRgb[dstIdx] = (byte)tmpR[k];
                        dstRgb[dstIdx + 1] = (byte)tmpG[k];
                        dstRgb[dstIdx + 2] = (byte)tmpB[k];
                    }
                }
            }

            for (; s < sEnd; s++)
            {
                float r = srcPlanar[s];
                float g = srcPlanar[spatial + s];
                float b = srcPlanar[(long)spatial * 2 + s];

                int ir = (int)((r * 0.5f + 0.5f) * 255.0f);
                int ig = (int)((g * 0.5f + 0.5f) * 255.0f);
                int ib = (int)((b * 0.5f + 0.5f) * 255.0f);

                int dstIdx = s * 3;
                dstRgb[dstIdx] = (byte)(ir < 0 ? 0 : (ir > 255 ? 255 : ir));
                dstRgb[dstIdx + 1] = (byte)(ig < 0 ? 0 : (ig > 255 ? 255 : ig));
                dstRgb[dstIdx + 2] = (byte)(ib < 0 ? 0 : (ib > 255 ? 255 : ib));
            }
        });
    }
}
