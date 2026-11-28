namespace Glacier.Media.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Inference.Format;
using Glacier.Media.Audio.Kitten;
using Xunit;

public class KittenHardeningTests
{
    [Theory]
    [InlineData("F16", 2)]
    [InlineData("BF16", 2)]
    [InlineData("I32", 4)]
    public void AlbertEncoder_RejectsNonF32Dtype_WithNotSupportedException(string dtype, int elemBytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_albert_dtype_{dtype}_{Guid.NewGuid():N}.safetensors");
        try
        {
            WriteMinimalAlbertSafetensors(path, dtype, elemBytes);
            using var sf = SafetensorsFile.Open(path);
            Assert.Throws<NotSupportedException>(() => new AlbertEncoder(sf));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void KittenLoaders_RejectNonF32Dtype_AcrossVoiceStoreAndBiLstm()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_kitten_voice_f16_{Guid.NewGuid():N}.safetensors");
        try
        {
            ulong voiceElems = 400 * 256;
            ulong voiceBytes = voiceElems * 2;
            byte[] header = Encoding.UTF8.GetBytes(
                "{\"bella\":{\"dtype\":\"F16\",\"shape\":[400,256],\"data_offsets\":[0," + voiceBytes + "]}}");

            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write((ulong)header.Length);
                bw.Write(header);
                bw.Flush();
                fs.SetLength(8 + header.Length + (long)voiceBytes);
            }

            Assert.Throws<NotSupportedException>(() => KittenVoiceStore.Load(path));

            // Verify KittenTtsEngine partial initialization failure also disposes model and voice handles cleanly
            string modelPath = Path.Combine(Path.GetTempPath(), $"glacier_kitten_model_f32_{Guid.NewGuid():N}.safetensors");
            try
            {
                WriteMinimalAlbertSafetensors(modelPath, "F32", sizeof(float));
                Assert.Throws<NotSupportedException>(() => new KittenTtsEngine(modelPath, path));
                using (var exclusive = new FileStream(modelPath, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                {
                    Assert.True(exclusive.Length > 0);
                }
            }
            finally
            {
                if (File.Exists(modelPath)) File.Delete(modelPath);
            }

            using var sf = SafetensorsFile.Open(path);
            Assert.Throws<NotSupportedException>(() => new KittenBiLstm(sf, "bella", "bella", "bella", 128, 64));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void KittenVoiceStore_Load_ReleasesFileHandleImmediatelyAfterEagerCopy()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_kitten_voice_f32_{Guid.NewGuid():N}.safetensors");
        try
        {
            ulong voiceElems = 400 * 256;
            ulong voiceBytes = voiceElems * sizeof(float);
            byte[] header = Encoding.UTF8.GetBytes(
                "{\"bella\":{\"dtype\":\"F32\",\"shape\":[400,256],\"data_offsets\":[0," + voiceBytes + "]}}");

            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write((ulong)header.Length);
                bw.Write(header);
                bw.Flush();
                fs.SetLength(8 + header.Length + (long)voiceBytes);
            }

            using var store = KittenVoiceStore.Load(path);
            Assert.Contains("bella", store.AvailableVoices);

            // Must succeed without FileShare.Delete even while store is still alive
            using (var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
                Assert.True(exclusive.Length > 0);
            }
            File.Delete(path);
            Assert.False(File.Exists(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void AlbertEncoder_Forward_PoolsScratchBuffersSafelyAcrossVaryingSequenceLengths()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_albert_f32_{Guid.NewGuid():N}.safetensors");
        try
        {
            WriteMinimalAlbertSafetensors(path, "F32", sizeof(float));
            using var sf = SafetensorsFile.Open(path);
            var albert = new AlbertEncoder(sf);

            // Run a longer sequence first so ArrayPool holds larger buffers, then run shorter sequences
            // to prove exact Span slicing prevents stale tail contamination.
            int[] longIds = [0, 5, 12, 33, 44, 55, 66, 77, 88, 99, 101, 111, 122, 133, 144, 155, 166, 170, 177];
            float[] longOut = albert.Forward(longIds);
            Assert.Equal(longIds.Length * 128, longOut.Length);
            Assert.All(longOut, v =>
            {
                Assert.False(float.IsNaN(v));
                Assert.False(float.IsInfinity(v));
            });

            int[] shortIds = [0, 12, 44];
            float[] shortOut1 = albert.Forward(shortIds);
            float[] shortOut2 = albert.Forward(shortIds);
            Assert.Equal(shortIds.Length * 128, shortOut1.Length);
            Assert.Equal(shortOut1, shortOut2);

            // Verify empty sequence returns empty cleanly
            Assert.Empty(albert.Forward(ReadOnlySpan<int>.Empty));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(33)]
    [InlineData(129)]
    public void KittenTensorOps_Max_MatchesScalarOracle(int length)
    {
        var rng = new Random(length * 97 + 13);
        float[] data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = (float)(rng.NextDouble() * 200.0 - 100.0);
        }

        float expected = 0f;
        if (length > 0)
        {
            expected = data[0];
            for (int i = 1; i < length; i++)
            {
                if (data[i] > expected) expected = data[i];
            }
        }

        float actual = KittenTensorOps.Max(data);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(3, 4, 1, 1, 0, 1, 1, true)]
    [InlineData(4, 8, 17, 3, 1, 1, 1, true)]
    [InlineData(8, 16, 65, 5, 2, 1, 1, false)]
    [InlineData(6, 12, 49, 7, 9, 1, 3, true)]
    [InlineData(4, 6, 33, 3, 5, 1, 5, true)]
    [InlineData(1, 2, 32, 3, 1, 2, 1, true)]
    [InlineData(1, 11, 120, 20, 0, 5, 1, false)]
    [InlineData(22, 16, 96, 12, 3, 6, 1, true)]
    public void KittenTensorOps_Conv1D_MatchesScalarOracle(
        int inChannels,
        int outChannels,
        int length,
        int kernelSize,
        int padding,
        int stride,
        int dilation,
        bool withBias)
    {
        var rng = new Random(inChannels * 17 + outChannels * 31 + length + kernelSize + stride + dilation);
        float[] input = new float[inChannels * length];
        float[] weight = new float[outChannels * inChannels * kernelSize];
        float[] bias = withBias ? new float[outChannels] : Array.Empty<float>();

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.3f;
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        int effKernel = (kernelSize - 1) * dilation + 1;
        int outLength = (length + 2 * padding - effKernel) / stride + 1;
        Assert.True(outLength > 0);

        float[] expected = new float[outChannels * outLength];
        for (int oc = 0; oc < outChannels; oc++)
        {
            float baseBias = withBias ? bias[oc] : 0f;
            for (int t = 0; t < outLength; t++)
            {
                float sum = baseBias;
                int inStart = t * stride - padding;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int inIdx = inStart + k * dilation;
                        if (inIdx >= 0 && inIdx < length)
                        {
                            sum += input[ic * length + inIdx] * weight[(oc * inChannels + ic) * kernelSize + k];
                        }
                    }
                }
                expected[oc * outLength + t] = sum;
            }
        }

        float[] actual = new float[outChannels * outLength];
        KittenTensorOps.Conv1D(input, weight, bias, actual, inChannels, outChannels, length, kernelSize, padding, stride, dilation);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-4f,
                $"Conv1D mismatch at {i}: expected {expected[i]}, actual {actual[i]}");
        }
    }

    private static void WriteMinimalAlbertSafetensors(string path, string dtype, int elemBytes)
    {
        string prefix = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
        var specs = new List<(string Name, int[] Shape)>
        {
            ("bert.embeddings.word_embeddings.weight", [178, 128]),
            ("bert.embeddings.token_type_embeddings.weight", [2, 128]),
            ("bert.embeddings.position_embeddings.weight", [512, 128]),
            ("bert.embeddings.LayerNorm.weight", [128]),
            ("bert.embeddings.LayerNorm.bias", [128]),
            ("bert.encoder.embedding_hidden_mapping_in.weight", [768, 128]),
            ("bert.encoder.embedding_hidden_mapping_in.bias", [768]),
            (prefix + "attention.query.weight", [768, 768]),
            (prefix + "attention.query.bias", [768]),
            (prefix + "attention.key.weight", [768, 768]),
            (prefix + "attention.key.bias", [768]),
            (prefix + "attention.value.weight", [768, 768]),
            (prefix + "attention.value.bias", [768]),
            (prefix + "attention.dense.weight", [768, 768]),
            (prefix + "attention.dense.bias", [768]),
            (prefix + "attention.LayerNorm.weight", [768]),
            (prefix + "attention.LayerNorm.bias", [768]),
            (prefix + "ffn.weight", [2048, 768]),
            (prefix + "ffn.bias", [2048]),
            (prefix + "ffn_output.weight", [768, 2048]),
            (prefix + "ffn_output.bias", [768]),
            (prefix + "full_layer_layer_norm.weight", [768]),
            (prefix + "full_layer_layer_norm.bias", [768]),
            ("bert_encoder.weight", [128, 768]),
            ("bert_encoder.bias", [128])
        };

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

        if (dtype == "F32")
        {
            int floatCount = (int)(maxBytes / sizeof(float));
            var rng = new Random(2026);
            for (int i = 0; i < floatCount; i++)
            {
                bw.Write((float)(rng.NextDouble() * 0.04 - 0.02));
            }
        }
        else
        {
            bw.Flush();
            fs.SetLength(8 + headerBytes.Length + (long)maxBytes);
        }
    }
}
