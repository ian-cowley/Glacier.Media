namespace Glacier.Media.Audio.Kitten;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

/// <summary>
/// High-performance pure C# SIMD tensor mathematical operations for StyleTTS 2 / KittenTTS.
/// Fully hardware-accelerated via .NET 10 hardware intrinsics (AVX-512 / AVX2 / NEON).
/// </summary>
public static unsafe class KittenTensorOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = a.Length;
        int i = 0;
        float sum = 0f;
        fixed (float* pA = a, pB = b)
        {
            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vSum = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    vSum = Vector512.FusedMultiplyAdd(Vector512.Load(pA + i), Vector512.Load(pB + i), vSum);
                }
                sum += Vector512.Sum(vSum);
            }
            if (Vector256.IsHardwareAccelerated && i <= length - 8)
            {
                var vSum = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    vSum = Vector256.FusedMultiplyAdd(Vector256.Load(pA + i), Vector256.Load(pB + i), vSum);
                }
                sum += Vector256.Sum(vSum);
            }
            if (Vector128.IsHardwareAccelerated && i <= length - 4)
            {
                var vSum = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    vSum = Vector128.FusedMultiplyAdd(Vector128.Load(pA + i), Vector128.Load(pB + i), vSum);
                }
                sum += Vector128.Sum(vSum);
            }
            for (; i < length; i++)
            {
                sum += pA[i] * pB[i];
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> a)
    {
        int length = a.Length;
        int i = 0;
        float sum = 0f;
        fixed (float* pA = a)
        {
            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vSum = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    vSum += Vector512.Load(pA + i);
                }
                sum += Vector512.Sum(vSum);
            }
            if (Vector256.IsHardwareAccelerated && i <= length - 8)
            {
                var vSum = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    vSum += Vector256.Load(pA + i);
                }
                sum += Vector256.Sum(vSum);
            }
            if (Vector128.IsHardwareAccelerated && i <= length - 4)
            {
                var vSum = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    vSum += Vector128.Load(pA + i);
                }
                sum += Vector128.Sum(vSum);
            }
            for (; i < length; i++)
            {
                sum += pA[i];
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ReadOnlySpan<float> a)
    {
        if (a.IsEmpty) return 0f;
        int length = a.Length;
        fixed (float* pA = a)
        {
            float max = pA[0];
            int i = 0;

            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vMax = Vector512.Load(pA);
                i = 16;
                for (; i <= length - 16; i += 16)
                {
                    vMax = Vector512.Max(vMax, Vector512.Load(pA + i));
                }
                for (int lane = 0; lane < 16; lane++)
                {
                    float val = vMax[lane];
                    if (val > max) max = val;
                }
            }

            if (Vector256.IsHardwareAccelerated && i <= length - 8)
            {
                var vMax = Vector256.Load(pA + i);
                i += 8;
                for (; i <= length - 8; i += 8)
                {
                    vMax = Vector256.Max(vMax, Vector256.Load(pA + i));
                }
                for (int lane = 0; lane < 8; lane++)
                {
                    float val = vMax[lane];
                    if (val > max) max = val;
                }
            }

            if (Vector128.IsHardwareAccelerated && i <= length - 4)
            {
                var vMax = Vector128.Load(pA + i);
                i += 4;
                for (; i <= length - 4; i += 4)
                {
                    vMax = Vector128.Max(vMax, Vector128.Load(pA + i));
                }
                for (int lane = 0; lane < 4; lane++)
                {
                    float val = vMax[lane];
                    if (val > max) max = val;
                }
            }

            for (; i < length; i++)
            {
                float val = pA[i];
                if (val > max) max = val;
            }
            return max;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Linear(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inFeatures,
        int outFeatures)
    {
        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pOut = output)
        fixed (float* pB = bias)
        {
            for (int o = 0; o < outFeatures; o++)
            {
                float sum = pB != null ? pB[o] : 0.0f;
                var rowW = new ReadOnlySpan<float>(pW + o * inFeatures, inFeatures);
                sum += Dot(input, rowW);
                pOut[o] = sum;
            }
        }
    }

    /// <summary>
    /// Batched linear projection over sequence tokens: input [seqLen, inFeatures] * W^T [inFeatures, outFeatures] + b.
    /// </summary>
    public static void BatchLinear(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int seqLen,
        int inFeatures,
        int outFeatures)
    {
        for (int s = 0; s < seqLen; s++)
        {
            var inSlice = input.Slice(s * inFeatures, inFeatures);
            var outSlice = output.Slice(s * outFeatures, outFeatures);
            Linear(inSlice, weight, bias, outSlice, inFeatures, outFeatures);
        }
    }

    /// <summary>
    /// Standard 1D Convolution over channels: input [channels, length] -> output [outChannels, outLength].
    /// Weight layout: [outChannels, inChannels, kernelSize].
    /// </summary>
    public static void Conv1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inChannels,
        int outChannels,
        int length,
        int kernelSize,
        int padding = 0,
        int stride = 1,
        int dilation = 1)
    {
        int effKernel = (kernelSize - 1) * dilation + 1;
        int outLength = (length + 2 * padding - effKernel) / stride + 1;
        if (outLength <= 0) return;

        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pB = bias)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* wPtr = pW;
            float* bPtr = pB;
            float* outPtr = pOut;

            Parallel.For(0, outChannels, oc =>
            {
                float baseBias = bPtr != null ? bPtr[oc] : 0.0f;
                int ocOffset = oc * outLength;
                float* pOutOc = outPtr + ocOffset;

                if (stride == 1)
                {
                    FillConstant(pOutOc, outLength, baseBias);

                    int tStart = Math.Clamp(padding, 0, outLength);
                    int tEnd = Math.Clamp(length + padding - effKernel + 1, tStart, outLength);

                    for (int t = 0; t < tStart; t++)
                    {
                        AccumulateConv1DScalarAtT(inPtr, wPtr, pOutOc, oc, inChannels, length, kernelSize, padding, stride, dilation, t);
                    }

                    int interiorLen = tEnd - tStart;
                    if (interiorLen > 0)
                    {
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            int icInOffset = ic * length;
                            int icWOffset = (oc * inChannels + ic) * kernelSize;
                            float* pDstInterior = pOutOc + tStart;

                            for (int k = 0; k < kernelSize; k++)
                            {
                                float w = wPtr[icWOffset + k];
                                if (w == 0.0f) continue;

                                float* pSrcInterior = inPtr + icInOffset + (tStart - padding + k * dilation);
                                AccumulateScaledSimd(pSrcInterior, pDstInterior, w, interiorLen);
                            }
                        }
                    }

                    for (int t = tEnd; t < outLength; t++)
                    {
                        AccumulateConv1DScalarAtT(inPtr, wPtr, pOutOc, oc, inChannels, length, kernelSize, padding, stride, dilation, t);
                    }
                }
                else if (dilation == 1 && kernelSize >= 4)
                {
                    int tStart = padding <= 0 ? 0 : Math.Min(outLength, (padding + stride - 1) / stride);
                    int maxIn = length + padding - kernelSize;
                    int tEnd = maxIn < 0 ? tStart : Math.Clamp(maxIn / stride + 1, tStart, outLength);

                    for (int t = 0; t < tStart; t++)
                    {
                        pOutOc[t] = baseBias;
                        AccumulateConv1DScalarAtT(inPtr, wPtr, pOutOc, oc, inChannels, length, kernelSize, padding, stride, dilation, t);
                    }

                    for (int t = tStart; t < tEnd; t++)
                    {
                        float sum = baseBias;
                        int inStart = t * stride - padding;
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            var inSpan = new ReadOnlySpan<float>(inPtr + ic * length + inStart, kernelSize);
                            var wSpan = new ReadOnlySpan<float>(wPtr + (oc * inChannels + ic) * kernelSize, kernelSize);
                            sum += Dot(inSpan, wSpan);
                        }
                        pOutOc[t] = sum;
                    }

                    for (int t = tEnd; t < outLength; t++)
                    {
                        pOutOc[t] = baseBias;
                        AccumulateConv1DScalarAtT(inPtr, wPtr, pOutOc, oc, inChannels, length, kernelSize, padding, stride, dilation, t);
                    }
                }
                else
                {
                    for (int t = 0; t < outLength; t++)
                    {
                        pOutOc[t] = baseBias;
                        AccumulateConv1DScalarAtT(inPtr, wPtr, pOutOc, oc, inChannels, length, kernelSize, padding, stride, dilation, t);
                    }
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FillConstant(float* dst, int length, float value)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && length >= 16)
        {
            var v = Vector512.Create(value);
            for (; i <= length - 16; i += 16)
                v.Store(dst + i);
        }
        if (Vector256.IsHardwareAccelerated && i <= length - 8)
        {
            var v = Vector256.Create(value);
            for (; i <= length - 8; i += 8)
                v.Store(dst + i);
        }
        if (Vector128.IsHardwareAccelerated && i <= length - 4)
        {
            var v = Vector128.Create(value);
            for (; i <= length - 4; i += 4)
                v.Store(dst + i);
        }
        for (; i < length; i++)
        {
            dst[i] = value;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateScaledSimd(float* src, float* dst, float w, int length)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && length >= 16)
        {
            var vW = Vector512.Create(w);
            for (; i <= length - 16; i += 16)
            {
                var vDst = Vector512.Load(dst + i);
                vDst = Vector512.FusedMultiplyAdd(Vector512.Load(src + i), vW, vDst);
                vDst.Store(dst + i);
            }
        }
        if (Vector256.IsHardwareAccelerated && i <= length - 8)
        {
            var vW = Vector256.Create(w);
            for (; i <= length - 8; i += 8)
            {
                var vDst = Vector256.Load(dst + i);
                vDst = Vector256.FusedMultiplyAdd(Vector256.Load(src + i), vW, vDst);
                vDst.Store(dst + i);
            }
        }
        if (Vector128.IsHardwareAccelerated && i <= length - 4)
        {
            var vW = Vector128.Create(w);
            for (; i <= length - 4; i += 4)
            {
                var vDst = Vector128.Load(dst + i);
                vDst = Vector128.FusedMultiplyAdd(Vector128.Load(src + i), vW, vDst);
                vDst.Store(dst + i);
            }
        }
        for (; i < length; i++)
        {
            dst[i] += src[i] * w;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateConv1DScalarAtT(
        float* inPtr,
        float* wPtr,
        float* pOutOc,
        int oc,
        int inChannels,
        int length,
        int kernelSize,
        int padding,
        int stride,
        int dilation,
        int t)
    {
        float sum = pOutOc[t];
        int inStart = t * stride - padding;

        for (int ic = 0; ic < inChannels; ic++)
        {
            int icInOffset = ic * length;
            int icWOffset = (oc * inChannels + ic) * kernelSize;

            for (int k = 0; k < kernelSize; k++)
            {
                int inIdx = inStart + k * dilation;
                if ((uint)inIdx < (uint)length)
                {
                    sum += inPtr[icInOffset + inIdx] * wPtr[icWOffset + k];
                }
            }
        }

        pOutOc[t] = sum;
    }

    /// <summary>
    /// 1D Transposed Convolution (upsampling): input [inChannels, length] -> output [outChannels, outLength].
    /// Weight layout: [inChannels, outChannels, kernelSize].
    /// </summary>
    public static void ConvTranspose1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inChannels,
        int outChannels,
        int length,
        int kernelSize,
        int stride,
        int padding = 0,
        int outputPadding = 0)
    {
        int outLength = (length - 1) * stride + kernelSize + outputPadding - 2 * padding;
        if (outLength <= 0) return;

        output.Clear();

        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pB = bias)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* wPtr = pW;
            float* bPtr = pB;
            float* outPtr = pOut;

            Parallel.For(0, outChannels, oc =>
            {
                float bVal = bPtr != null ? bPtr[oc] : 0.0f;
                int ocOutOffset = oc * outLength;
                if (bVal != 0.0f)
                {
                    for (int t = 0; t < outLength; t++)
                    {
                        outPtr[ocOutOffset + t] = bVal;
                    }
                }

                for (int ic = 0; ic < inChannels; ic++)
                {
                    int icInOffset = ic * length;
                    int wOffset = (ic * outChannels + oc) * kernelSize;

                    for (int i = 0; i < length; i++)
                    {
                        float inVal = inPtr[icInOffset + i];
                        if (inVal == 0.0f) continue;
                        int outStart = i * stride;

                        for (int k = 0; k < kernelSize; k++)
                        {
                            int outIdx = outStart + k - padding;
                            if (outIdx >= 0 && outIdx < outLength)
                            {
                                outPtr[ocOutOffset + outIdx] += inVal * wPtr[wOffset + k];
                            }
                        }
                    }
                }
            });
        }
    }

    /// <summary>
    /// Layer Normalization over the last dimension (hiddenSize).
    /// </summary>
    public static void LayerNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        Span<float> output,
        int count,
        int hiddenSize,
        float eps = 1e-5f)
    {
        for (int c = 0; c < count; c++)
        {
            var inSlice = input.Slice(c * hiddenSize, hiddenSize);
            var outSlice = output.Slice(c * hiddenSize, hiddenSize);

            float mean = Sum(inSlice) / hiddenSize;
            float varSum = 0.0f;
            for (int i = 0; i < hiddenSize; i++)
            {
                float diff = inSlice[i] - mean;
                varSum += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt((varSum / hiddenSize) + eps);

            for (int i = 0; i < hiddenSize; i++)
            {
                float g = gamma.Length > 0 ? gamma[i] : 1.0f;
                float b = beta.Length > 0 ? beta[i] : 0.0f;
                outSlice[i] = ((inSlice[i] - mean) * invStd) * g + b;
            }
        }
    }

    /// <summary>
    /// Instance Normalization over time dimension for channels [channels, time].
    /// </summary>
    public static void InstanceNorm(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int time,
        float eps = 1e-5f)
    {
        if (time == 0) return;

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* outPtr = pOut;

            Parallel.For(0, channels, c =>
            {
                float* inSlice = inPtr + c * time;
                float* outSlice = outPtr + c * time;

                float sum = 0.0f;
                for (int t = 0; t < time; t++) sum += inSlice[t];
                float mean = sum / time;

                float varSum = 0.0f;
                for (int t = 0; t < time; t++)
                {
                    float diff = inSlice[t] - mean;
                    varSum += diff * diff;
                }
                float invStd = 1.0f / MathF.Sqrt((varSum / time) + eps);

                for (int t = 0; t < time; t++)
                {
                    outSlice[t] = (inSlice[t] - mean) * invStd;
                }
            });
        }
    }

    /// <summary>
    /// Adaptive Instance Normalization (AdaIN):
    /// normed = InstanceNorm(x)
    /// out = normed * (gamma + 1.0) + beta
    /// where gamma, beta come from style projection [2 * channels].
    /// </summary>
    public static void AdaIN(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> styleProjection,
        Span<float> output,
        int channels,
        int time)
    {
        InstanceNorm(input, output, channels, time);

        fixed (float* pProj = styleProjection)
        fixed (float* pOut = output)
        {
            for (int c = 0; c < channels; c++)
            {
                float gamma = pProj[c] + 1.0f;
                float beta = pProj[channels + c];
                int offset = c * time;

                for (int t = 0; t < time; t++)
                {
                    pOut[offset + t] = pOut[offset + t] * gamma + beta;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LeakyRelu(Span<float> span, float alpha = 0.2f)
    {
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            if (v < 0.0f) span[i] = v * alpha;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    public static void GeluInPlace(Span<float> span)
    {
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = Gelu(span[i]);
        }
    }

    public static void Softmax(Span<float> span)
    {
        float max = Max(span);
        float sum = 0.0f;
        for (int i = 0; i < span.Length; i++)
        {
            float exp = MathF.Exp(span[i] - max);
            span[i] = exp;
            sum += exp;
        }
        float invSum = 1.0f / MathF.Max(sum, 1e-9f);
        for (int i = 0; i < span.Length; i++)
        {
            span[i] *= invSum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }
}
