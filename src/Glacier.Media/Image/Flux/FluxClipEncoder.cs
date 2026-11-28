namespace Glacier.Media.Image.Flux;

using System;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Format;

/// <summary>
/// Pure C# .NET 10 CLIP-L text encoder for Flux conditioning.
/// Encodes textual prompts into pooled embedding y [768] directly from clip_l.safetensors
/// using real byte-level BPE tokenization (<see cref="ClipTokenizer"/>) and SIMD F16 GEMM.
/// </summary>
public unsafe sealed class FluxClipEncoder : IDisposable
{
    private readonly SafetensorsFile _safetensors;
    private readonly ClipTokenizer _tokenizer;
    private readonly Half* _tokenEmbeddings; // [49408, 768]
    private readonly Half* _posEmbeddings;   // [77, 768]

    // 12 Transformer layers
    private readonly LayerWeights[] _layers = new LayerWeights[12];
    private readonly Half* _finalNormWeight;
    private readonly Half* _finalNormBias;
    private bool _disposed;

    public ClipTokenizer Tokenizer => _tokenizer;

    private sealed class LayerWeights
    {
        public Half* Ln1Weight, Ln1Bias;
        public Half* QWeight, QBias;
        public Half* KWeight, KBias;
        public Half* VWeight, VBias;
        public Half* OutWeight, OutBias;
        public Half* Ln2Weight, Ln2Bias;
        public Half* Fc1Weight, Fc1Bias;
        public Half* Fc2Weight, Fc2Bias;
    }

    private FluxClipEncoder(SafetensorsFile safetensors, ClipTokenizer tokenizer)
    {
        _safetensors = safetensors;
        _tokenizer = tokenizer;
        _tokenEmbeddings = GetF16Tensor(safetensors, "text_model.embeddings.token_embedding.weight");
        _posEmbeddings = GetF16Tensor(safetensors, "text_model.embeddings.position_embedding.weight");

        for (int i = 0; i < 12; i++)
        {
            string pfx = $"text_model.encoder.layers.{i}.";
            _layers[i] = new LayerWeights
            {
                Ln1Weight = GetF16Tensor(safetensors, pfx + "layer_norm1.weight"),
                Ln1Bias = GetF16Tensor(safetensors, pfx + "layer_norm1.bias"),
                QWeight = GetF16Tensor(safetensors, pfx + "self_attn.q_proj.weight"),
                QBias = GetF16Tensor(safetensors, pfx + "self_attn.q_proj.bias"),
                KWeight = GetF16Tensor(safetensors, pfx + "self_attn.k_proj.weight"),
                KBias = GetF16Tensor(safetensors, pfx + "self_attn.k_proj.bias"),
                VWeight = GetF16Tensor(safetensors, pfx + "self_attn.v_proj.weight"),
                VBias = GetF16Tensor(safetensors, pfx + "self_attn.v_proj.bias"),
                OutWeight = GetF16Tensor(safetensors, pfx + "self_attn.out_proj.weight"),
                OutBias = GetF16Tensor(safetensors, pfx + "self_attn.out_proj.bias"),
                Ln2Weight = GetF16Tensor(safetensors, pfx + "layer_norm2.weight"),
                Ln2Bias = GetF16Tensor(safetensors, pfx + "layer_norm2.bias"),
                Fc1Weight = GetF16Tensor(safetensors, pfx + "mlp.fc1.weight"),
                Fc1Bias = GetF16Tensor(safetensors, pfx + "mlp.fc1.bias"),
                Fc2Weight = GetF16Tensor(safetensors, pfx + "mlp.fc2.weight"),
                Fc2Bias = GetF16Tensor(safetensors, pfx + "mlp.fc2.bias"),
            };
        }

        _finalNormWeight = GetF16Tensor(safetensors, "text_model.final_layer_norm.weight");
        _finalNormBias = GetF16Tensor(safetensors, "text_model.final_layer_norm.bias");
    }

    private static Half* GetF16Tensor(SafetensorsFile safetensors, string name)
    {
        if (!safetensors.Tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Tensor '{name}' not found in CLIP Safetensors file.");

        if (!string.Equals(info.Dtype, "F16", StringComparison.Ordinal))
            throw new NotSupportedException($"FluxClipEncoder requires F16 Safetensors weights, but tensor '{name}' has dtype '{info.Dtype}'.");

        if (info.ByteSize != (ulong)info.ElementCount * sizeof(ushort))
            throw new InvalidDataException($"Tensor '{name}' byte size {info.ByteSize} does not match F16 element count {info.ElementCount}.");

        return (Half*)safetensors.GetTensorPointer(name);
    }

    /// <summary>
    /// Opens a <see cref="FluxClipEncoder"/> from a Safetensors file and resolves its CLIP BPE tokenizer.
    /// Probes explicit paths first, then <c>vocab.json</c>+<c>merges.txt</c> or <c>tokenizer.json</c>
    /// in the model directory or its <c>tokenizer/</c> subdirectory. Throws <see cref="FileNotFoundException"/>
    /// if no valid CLIP BPE tokenizer files exist.
    /// </summary>
    public static FluxClipEncoder Open(string safetensorsPath, string? vocabJsonPath = null, string? mergesTxtPath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsPath);
        var sf = SafetensorsFile.Open(safetensorsPath);
        try
        {
            ClipTokenizer tokenizer = ResolveTokenizer(safetensorsPath, vocabJsonPath, mergesTxtPath);
            return new FluxClipEncoder(sf, tokenizer);
        }
        catch
        {
            sf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Opens a <see cref="FluxClipEncoder"/> from a Safetensors file with a pre-loaded <see cref="ClipTokenizer"/>.
    /// </summary>
    public static FluxClipEncoder Open(string safetensorsPath, ClipTokenizer tokenizer)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsPath);
        ArgumentNullException.ThrowIfNull(tokenizer);

        var sf = SafetensorsFile.Open(safetensorsPath);
        try
        {
            return new FluxClipEncoder(sf, tokenizer);
        }
        catch
        {
            sf.Dispose();
            throw;
        }
    }

    private static ClipTokenizer ResolveTokenizer(string safetensorsPath, string? vocabJsonPath, string? mergesTxtPath)
    {
        if (vocabJsonPath != null || mergesTxtPath != null)
        {
            string? resolvedVocab = vocabJsonPath;
            string? resolvedMerges = mergesTxtPath;

            if (resolvedVocab != null && resolvedMerges == null)
            {
                string vDir = Path.GetDirectoryName(Path.GetFullPath(resolvedVocab)) ?? Directory.GetCurrentDirectory();
                resolvedMerges = Path.Combine(vDir, "merges.txt");
            }
            else if (resolvedMerges != null && resolvedVocab == null)
            {
                string mDir = Path.GetDirectoryName(Path.GetFullPath(resolvedMerges)) ?? Directory.GetCurrentDirectory();
                resolvedVocab = Path.Combine(mDir, "vocab.json");
            }

            return ClipTokenizer.Load(resolvedVocab!, resolvedMerges!);
        }

        string modelDir = Path.GetDirectoryName(Path.GetFullPath(safetensorsPath)) ?? Directory.GetCurrentDirectory();
        string[] candidateDirs = [modelDir, Path.Combine(modelDir, "tokenizer")];

        foreach (string dir in candidateDirs)
        {
            string vocab = Path.Combine(dir, "vocab.json");
            string merges = Path.Combine(dir, "merges.txt");
            if (File.Exists(vocab) && File.Exists(merges))
            {
                return ClipTokenizer.Load(vocab, merges);
            }

            string tokenizerJson = Path.Combine(dir, "tokenizer.json");
            if (File.Exists(tokenizerJson))
            {
                return ClipTokenizer.LoadFromTokenizerJson(tokenizerJson);
            }
        }

        string expectedVocab = Path.Combine(modelDir, "vocab.json");
        throw new FileNotFoundException(
            $"CLIP BPE tokenizer files (vocab.json + merges.txt or tokenizer.json) were not found alongside '{safetensorsPath}'. " +
            "Provide explicit tokenizer paths or place vocab.json/merges.txt or tokenizer.json in the model directory (or tokenizer/ subfolder).",
            expectedVocab);
    }

    /// <summary>
    /// Encodes a textual prompt into 768-dim pooled conditioning vector y.
    /// </summary>
    public float[] EncodePrompt(string prompt)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);

        const int seqLen = 77;
        const int hiddenDim = 768;
        const int numHeads = 12;
        const int headDim = 64;

        // 1. Real CLIP BPE tokenization
        int[] tokens = Tokenize(prompt, seqLen, out int eosIndex);
        eosIndex = Math.Clamp(eosIndex, 0, seqLen - 1);

        // 2. Initial embeddings: token + positional
        float* hidden = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* normBuf = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* q = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* k = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* v = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* attnOut = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* mlpInt = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * 3072 * sizeof(float)), 64);

        try
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokens[s];
                Half* tokPtr = _tokenEmbeddings + (long)tid * hiddenDim;
                Half* posPtr = _posEmbeddings + (long)s * hiddenDim;
                float* row = hidden + s * hiddenDim;
                for (int d = 0; d < hiddenDim; d++)
                {
                    row[d] = (float)tokPtr[d] + (float)posPtr[d];
                }
            }

            // 3. 12 Transformer layers
            float* scoreRow = stackalloc float[seqLen];
            for (int l = 0; l < 12; l++)
            {
                var layer = _layers[l];

                // LN1
                ApplyLayerNorm(hidden, normBuf, layer.Ln1Weight, layer.Ln1Bias, seqLen, hiddenDim);

                // QKV projections: [seqLen, 768] * [768, 768]
                MatMulF16(normBuf, layer.QWeight, layer.QBias, q, seqLen, hiddenDim, hiddenDim);
                MatMulF16(normBuf, layer.KWeight, layer.KBias, k, seqLen, hiddenDim, hiddenDim);
                MatMulF16(normBuf, layer.VWeight, layer.VBias, v, seqLen, hiddenDim, hiddenDim);

                // Multi-head Attention
                float scale = 1.0f / MathF.Sqrt(headDim);
                for (int h = 0; h < numHeads; h++)
                {
                    int headOffset = h * headDim;
                    for (int i = 0; i < seqLen; i++)
                    {
                        float* qRow = q + i * hiddenDim + headOffset;

                        float maxScore = float.NegativeInfinity;
                        for (int j = 0; j <= i; j++) // Causal mask
                        {
                            float dot = 0f;
                            float* kRow = k + j * hiddenDim + headOffset;
                            for (int d = 0; d < headDim; d++) dot += qRow[d] * kRow[d];
                            float sc = dot * scale;
                            scoreRow[j] = sc;
                            if (sc > maxScore) maxScore = sc;
                        }
                        for (int j = i + 1; j < seqLen; j++) scoreRow[j] = float.NegativeInfinity;

                        // Softmax
                        float sumExp = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float exp = MathF.Exp(scoreRow[j] - maxScore);
                            scoreRow[j] = exp;
                            sumExp += exp;
                        }
                        float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);
                        for (int j = 0; j <= i; j++) scoreRow[j] *= invSum;

                        // Weighted V sum
                        float* outRow = attnOut + i * hiddenDim + headOffset;
                        for (int d = 0; d < headDim; d++) outRow[d] = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float weight = scoreRow[j];
                            float* vRow = v + j * hiddenDim + headOffset;
                            for (int d = 0; d < headDim; d++) outRow[d] += weight * vRow[d];
                        }
                    }
                }


                // Out proj & residual: [seqLen, 768] * [768, 768]
                MatMulF16(attnOut, layer.OutWeight, layer.OutBias, normBuf, seqLen, hiddenDim, hiddenDim);
                for (int i = 0; i < seqLen * hiddenDim; i++) hidden[i] += normBuf[i];

                // LN2
                ApplyLayerNorm(hidden, normBuf, layer.Ln2Weight, layer.Ln2Bias, seqLen, hiddenDim);

                // MLP: fc1 [seqLen, 768] * [3072, 768] -> [seqLen, 3072]
                MatMulF16(normBuf, layer.Fc1Weight, layer.Fc1Bias, mlpInt, seqLen, hiddenDim, 3072);

                // QuickGELU: x * sigmoid(1.702 * x)
                for (int i = 0; i < seqLen * 3072; i++)
                {
                    float x = mlpInt[i];
                    mlpInt[i] = x / (1.0f + MathF.Exp(-1.702f * x));
                }

                // fc2: [seqLen, 3072] * [768, 3072] -> [seqLen, 768] + residual
                MatMulF16(mlpInt, layer.Fc2Weight, layer.Fc2Bias, normBuf, seqLen, 3072, hiddenDim);
                for (int i = 0; i < seqLen * hiddenDim; i++) hidden[i] += normBuf[i];
            }

            // 4. Extract EOS pooled vector & apply final layer norm
            float* eosHidden = hidden + eosIndex * hiddenDim;
            var pooled = new float[hiddenDim];
            fixed (float* pOut = pooled)
            {
                ApplyLayerNorm(eosHidden, pOut, _finalNormWeight, _finalNormBias, 1, hiddenDim);
            }
            return pooled;
        }
        finally
        {
            NativeMemory.AlignedFree(hidden);
            NativeMemory.AlignedFree(normBuf);
            NativeMemory.AlignedFree(q);
            NativeMemory.AlignedFree(k);
            NativeMemory.AlignedFree(v);
            NativeMemory.AlignedFree(attnOut);
            NativeMemory.AlignedFree(mlpInt);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyLayerNorm(float* src, float* dst, Half* weight, Half* bias, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float* s = src + r * dim;
            float* d = dst + r * dim;

            float mean = 0f;
            for (int i = 0; i < dim; i++) mean += s[i];
            mean /= dim;

            float var = 0f;
            for (int i = 0; i < dim; i++)
            {
                float diff = s[i] - mean;
                var += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(var / dim + 1e-5f);

            for (int i = 0; i < dim; i++)
            {
                d[i] = (s[i] - mean) * invStd * (float)weight[i] + (float)bias[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void MatMulF16(float* x, Half* w, Half* b, float* y, int nRows, int inDim, int outDim)
    {
        if (nRows <= 0 || outDim <= 0)
            return;

        if (inDim <= 0)
        {
            for (int r = 0; r < nRows; r++)
            {
                float* outVec = y + (long)r * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    outVec[o] = b != null ? (float)b[o] : 0f;
                }
            }
            return;
        }

        const int chunkSize = 16;
        int numChunks = (outDim + chunkSize - 1) / chunkSize;
        if ((long)nRows * outDim >= 4096 && numChunks > 1)
        {
            nint xPtr = (nint)x;
            nint wPtr = (nint)w;
            nint bPtr = (nint)b;
            nint yPtr = (nint)y;
            Parallel.For(0, numChunks, chunkIdx =>
            {
                int oStart = chunkIdx * chunkSize;
                int oEnd = Math.Min(oStart + chunkSize, outDim);
                MatMulF16Range((float*)xPtr, (Half*)wPtr, (Half*)bPtr, (float*)yPtr, nRows, inDim, outDim, oStart, oEnd);
            });
        }
        else
        {
            MatMulF16Range(x, w, b, y, nRows, inDim, outDim, 0, outDim);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MatMulF16Range(
        float* x,
        Half* w,
        Half* b,
        float* y,
        int nRows,
        int inDim,
        int outDim,
        int oStart,
        int oEnd)
    {
        for (int o = oStart; o < oEnd; o++)
        {
            ushort* wRow = (ushort*)(w + (long)o * inDim);
            float bias = b != null ? (float)b[o] : 0f;

            int r = 0;
            for (; r + 3 < nRows; r += 4)
            {
                float* x0 = x + (long)r * inDim;
                float* x1 = x0 + inDim;
                float* x2 = x1 + inDim;
                float* x3 = x2 + inDim;

                float sum0 = bias;
                float sum1 = bias;
                float sum2 = bias;
                float sum3 = bias;
                int i = 0;

                if (Vector512.IsHardwareAccelerated && inDim >= 16)
                {
                    var acc0 = Vector512<float>.Zero;
                    var acc1 = Vector512<float>.Zero;
                    var acc2 = Vector512<float>.Zero;
                    var acc3 = Vector512<float>.Zero;

                    for (; i <= inDim - 16; i += 16)
                    {
                        var vw = ConvertHalfToFloat16(wRow + i);
                        acc0 = Vector512.FusedMultiplyAdd(Vector512.Load(x0 + i), vw, acc0);
                        acc1 = Vector512.FusedMultiplyAdd(Vector512.Load(x1 + i), vw, acc1);
                        acc2 = Vector512.FusedMultiplyAdd(Vector512.Load(x2 + i), vw, acc2);
                        acc3 = Vector512.FusedMultiplyAdd(Vector512.Load(x3 + i), vw, acc3);
                    }

                    sum0 += Vector512.Sum(acc0);
                    sum1 += Vector512.Sum(acc1);
                    sum2 += Vector512.Sum(acc2);
                    sum3 += Vector512.Sum(acc3);
                }

                if (Vector256.IsHardwareAccelerated && i <= inDim - 8)
                {
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;
                    var acc2 = Vector256<float>.Zero;
                    var acc3 = Vector256<float>.Zero;

                    for (; i <= inDim - 8; i += 8)
                    {
                        var vw = ConvertHalfToFloat8(wRow + i);
                        acc0 = Vector256.FusedMultiplyAdd(Vector256.Load(x0 + i), vw, acc0);
                        acc1 = Vector256.FusedMultiplyAdd(Vector256.Load(x1 + i), vw, acc1);
                        acc2 = Vector256.FusedMultiplyAdd(Vector256.Load(x2 + i), vw, acc2);
                        acc3 = Vector256.FusedMultiplyAdd(Vector256.Load(x3 + i), vw, acc3);
                    }

                    sum0 += Vector256.Sum(acc0);
                    sum1 += Vector256.Sum(acc1);
                    sum2 += Vector256.Sum(acc2);
                    sum3 += Vector256.Sum(acc3);
                }

                if (Vector128.IsHardwareAccelerated && i <= inDim - 4)
                {
                    var acc0 = Vector128<float>.Zero;
                    var acc1 = Vector128<float>.Zero;
                    var acc2 = Vector128<float>.Zero;
                    var acc3 = Vector128<float>.Zero;

                    for (; i <= inDim - 4; i += 4)
                    {
                        var vw = ConvertHalfToFloat4(wRow + i);
                        acc0 = Vector128.FusedMultiplyAdd(Vector128.Load(x0 + i), vw, acc0);
                        acc1 = Vector128.FusedMultiplyAdd(Vector128.Load(x1 + i), vw, acc1);
                        acc2 = Vector128.FusedMultiplyAdd(Vector128.Load(x2 + i), vw, acc2);
                        acc3 = Vector128.FusedMultiplyAdd(Vector128.Load(x3 + i), vw, acc3);
                    }

                    sum0 += Vector128.Sum(acc0);
                    sum1 += Vector128.Sum(acc1);
                    sum2 += Vector128.Sum(acc2);
                    sum3 += Vector128.Sum(acc3);
                }

                for (; i < inDim; i++)
                {
                    float wf = (float)BitConverter.UInt16BitsToHalf(wRow[i]);
                    sum0 += x0[i] * wf;
                    sum1 += x1[i] * wf;
                    sum2 += x2[i] * wf;
                    sum3 += x3[i] * wf;
                }

                y[(long)r * outDim + o] = sum0;
                y[(long)(r + 1) * outDim + o] = sum1;
                y[(long)(r + 2) * outDim + o] = sum2;
                y[(long)(r + 3) * outDim + o] = sum3;
            }

            for (; r < nRows; r++)
            {
                float* x0 = x + (long)r * inDim;
                float sum = bias;
                int i = 0;

                if (Vector512.IsHardwareAccelerated && inDim >= 16)
                {
                    var acc = Vector512<float>.Zero;
                    for (; i <= inDim - 16; i += 16)
                    {
                        acc = Vector512.FusedMultiplyAdd(Vector512.Load(x0 + i), ConvertHalfToFloat16(wRow + i), acc);
                    }
                    sum += Vector512.Sum(acc);
                }

                if (Vector256.IsHardwareAccelerated && i <= inDim - 8)
                {
                    var acc = Vector256<float>.Zero;
                    for (; i <= inDim - 8; i += 8)
                    {
                        acc = Vector256.FusedMultiplyAdd(Vector256.Load(x0 + i), ConvertHalfToFloat8(wRow + i), acc);
                    }
                    sum += Vector256.Sum(acc);
                }

                if (Vector128.IsHardwareAccelerated && i <= inDim - 4)
                {
                    var acc = Vector128<float>.Zero;
                    for (; i <= inDim - 4; i += 4)
                    {
                        acc = Vector128.FusedMultiplyAdd(Vector128.Load(x0 + i), ConvertHalfToFloat4(wRow + i), acc);
                    }
                    sum += Vector128.Sum(acc);
                }

                for (; i < inDim; i++)
                {
                    sum += x0[i] * (float)BitConverter.UInt16BitsToHalf(wRow[i]);
                }

                y[(long)r * outDim + o] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> ConvertHalfToFloat16(ushort* ptr)
    {
        var raw16 = Vector256.Load(ptr);
        var (lo, hi) = Vector256.Widen(raw16);
        var h = Vector512.Create(lo, hi);

        var sign = (h & Vector512.Create(0x8000u)) << 16;
        var abs = h & Vector512.Create(0x7FFFu);
        var vMinNorm = Vector512.Create(0x0400u);
        var vInfExp = Vector512.Create(0x7C00u);
        var normalBits = ((abs + Vector512.Create(0x1C000u)) << 13) | sign;

        if (Vector512.GreaterThanOrEqualAll(abs, vMinNorm) && Vector512.LessThanAll(abs, vInfExp))
        {
            return normalBits.AsSingle();
        }

        var isZero = Vector512.Equals(abs, Vector512<uint>.Zero);
        var isSubnormal = Vector512.LessThan(abs, vMinNorm);
        var isInfOrNan = Vector512.GreaterThanOrEqual(abs, vInfExp);
        var isNan = Vector512.GreaterThan(abs, vInfExp);

        var subnormalBits = (Vector512.ConvertToSingle(abs.AsInt32()) * Vector512.Create(1.0f / 16777216.0f)).AsUInt32() | sign;
        var infNanBits = sign | Vector512.Create(0x7F800000u) | ((abs & Vector512.Create(0x03FFu)) << 13) | (isNan & Vector512.Create(0x00400000u));

        var res = Vector512.ConditionalSelect(isSubnormal, subnormalBits, normalBits);
        res = Vector512.ConditionalSelect(isZero, sign, res);
        res = Vector512.ConditionalSelect(isInfOrNan, infNanBits, res);
        return res.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> ConvertHalfToFloat8(ushort* ptr)
    {
        var raw8 = Vector128.Load(ptr);
        var (lo, hi) = Vector128.Widen(raw8);
        var h = Vector256.Create(lo, hi);

        var sign = (h & Vector256.Create(0x8000u)) << 16;
        var abs = h & Vector256.Create(0x7FFFu);
        var vMinNorm = Vector256.Create(0x0400u);
        var vInfExp = Vector256.Create(0x7C00u);
        var normalBits = ((abs + Vector256.Create(0x1C000u)) << 13) | sign;

        if (Vector256.GreaterThanOrEqualAll(abs, vMinNorm) && Vector256.LessThanAll(abs, vInfExp))
        {
            return normalBits.AsSingle();
        }

        var isZero = Vector256.Equals(abs, Vector256<uint>.Zero);
        var isSubnormal = Vector256.LessThan(abs, vMinNorm);
        var isInfOrNan = Vector256.GreaterThanOrEqual(abs, vInfExp);
        var isNan = Vector256.GreaterThan(abs, vInfExp);

        var subnormalBits = (Vector256.ConvertToSingle(abs.AsInt32()) * Vector256.Create(1.0f / 16777216.0f)).AsUInt32() | sign;
        var infNanBits = sign | Vector256.Create(0x7F800000u) | ((abs & Vector256.Create(0x03FFu)) << 13) | (isNan & Vector256.Create(0x00400000u));

        var res = Vector256.ConditionalSelect(isSubnormal, subnormalBits, normalBits);
        res = Vector256.ConditionalSelect(isZero, sign, res);
        res = Vector256.ConditionalSelect(isInfOrNan, infNanBits, res);
        return res.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> ConvertHalfToFloat4(ushort* ptr)
    {
        var raw4 = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ptr)).AsUInt16();
        var h = Vector128.WidenLower(raw4);

        var sign = (h & Vector128.Create(0x8000u)) << 16;
        var abs = h & Vector128.Create(0x7FFFu);
        var vMinNorm = Vector128.Create(0x0400u);
        var vInfExp = Vector128.Create(0x7C00u);
        var normalBits = ((abs + Vector128.Create(0x1C000u)) << 13) | sign;

        if (Vector128.GreaterThanOrEqualAll(abs, vMinNorm) && Vector128.LessThanAll(abs, vInfExp))
        {
            return normalBits.AsSingle();
        }

        var isZero = Vector128.Equals(abs, Vector128<uint>.Zero);
        var isSubnormal = Vector128.LessThan(abs, vMinNorm);
        var isInfOrNan = Vector128.GreaterThanOrEqual(abs, vInfExp);
        var isNan = Vector128.GreaterThan(abs, vInfExp);

        var subnormalBits = (Vector128.ConvertToSingle(abs.AsInt32()) * Vector128.Create(1.0f / 16777216.0f)).AsUInt32() | sign;
        var infNanBits = sign | Vector128.Create(0x7F800000u) | ((abs & Vector128.Create(0x03FFu)) << 13) | (isNan & Vector128.Create(0x00400000u));

        var res = Vector128.ConditionalSelect(isSubnormal, subnormalBits, normalBits);
        res = Vector128.ConditionalSelect(isZero, sign, res);
        res = Vector128.ConditionalSelect(isInfOrNan, infNanBits, res);
        return res.AsSingle();
    }

    private int[] Tokenize(string prompt, int maxLen, out int eosIndex)
        => _tokenizer.Encode(prompt, maxLen, out eosIndex);

    public void Dispose()
    {
        if (!_disposed)
        {
            _safetensors.Dispose();
            _disposed = true;
        }
    }
}
