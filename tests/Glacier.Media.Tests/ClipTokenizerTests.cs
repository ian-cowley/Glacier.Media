namespace Glacier.Media.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Intrinsics;
using System.Text;
using Glacier.Media.Image.Flux;
using Xunit;

public class ClipTokenizerTests
{
    private static (Dictionary<string, int> Vocab, string[] Merges) BuildSampleClipVocabAndMerges()
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        int id = 0;

        // Add single byte tokens + </w> variants for printable ASCII and UTF-8 bytes
        for (int b = 0; b < 256; b++)
        {
            char c = ClipTokenizer.EncodeByteToChar((byte)b);
            string s = c.ToString();
            vocab[s] = id++;
            vocab[s + "</w>"] = id++;
        }

        // Merges with explicit ranks (earlier rank wins)
        string[] merges =
        [
            "#version: 0.2",
            "t h",
            "th e</w>",
            "c a",
            "ca t</w>",
            "a t</w>",
            "i n",
            "in g</w>",
            "p l",
            "a y",
            "pl ay",
            "play ing</w>",
            "d o",
            "do g</w>",
            "g l",
            "gl o",
            "glo w",
            "glow ing</w>",
            "' s</w>"
        ];

        // Add merged tokens to vocab
        foreach (string line in merges)
        {
            if (line.StartsWith("#", StringComparison.Ordinal)) continue;
            string merged = line.Replace(" ", string.Empty);
            if (!vocab.ContainsKey(merged))
            {
                vocab[merged] = id++;
            }
        }

        // Standard CLIP special token IDs
        vocab["<|startoftext|>"] = 49406;
        vocab["<|endoftext|>"] = 49407;

        return (vocab, merges);
    }

    [Fact]
    public void ByteToChar_RoundtripsAll256Bytes()
    {
        var seen = new HashSet<char>();
        for (int b = 0; b < 256; b++)
        {
            char c = ClipTokenizer.EncodeByteToChar((byte)b);
            Assert.True(seen.Add(c), $"Duplicate mapped char for byte {b}");
            Assert.True(ClipTokenizer.TryDecodeCharToByte(c, out byte decoded));
            Assert.Equal((byte)b, decoded);
        }
    }

    [Fact]
    public void ClipTokenizer_BpeMergePriority_Lowercase_Contractions_Digits_AndDecodeRoundtrip()
    {
        var (vocab, merges) = BuildSampleClipVocabAndMerges();
        var tok = ClipTokenizer.FromVocabularyAndMerges(vocab, merges);

        // Lowercasing + whitespace normalization + contraction + digit splitting + </w> boundary
        int[] tokens = tok.Encode("  The   Cat's   playing  42  ", maxLength: 77, out int eosIndex);
        Assert.Equal(77, tokens.Length);
        Assert.Equal(49406, tokens[0]);
        Assert.Equal(49407, tokens[eosIndex]);
        for (int i = eosIndex; i < 77; i++)
        {
            Assert.Equal(49407, tokens[i]);
        }

        // Merged tokens: "the</w>", "cat</w>", "'s</w>", "playing</w>", "4</w>", "2</w>"
        Assert.Equal(vocab["the</w>"], tokens[1]);
        Assert.Equal(vocab["cat</w>"], tokens[2]);
        Assert.Equal(vocab["'s</w>"], tokens[3]);
        Assert.Equal(vocab["playing</w>"], tokens[4]);
        // Digits split individually by \p{N}
        Assert.Equal(vocab["4</w>"], tokens[5]);
        Assert.Equal(vocab["2</w>"], tokens[6]);
        Assert.Equal(7, eosIndex);

        string decoded = tok.Decode(tokens);
        Assert.Equal("the cat 's playing 4 2", decoded);

        // Empty / whitespace prompt -> [BOS, EOS, EOS...]
        int[] emptyTokens = tok.Encode("   \t\n  ", maxLength: 77, out int emptyEos);
        Assert.Equal(49406, emptyTokens[0]);
        Assert.Equal(1, emptyEos);
        Assert.Equal(49407, emptyTokens[1]);
        Assert.Equal(string.Empty, tok.Decode(emptyTokens));

        // Truncation preserves EOS at final index maxLength - 1
        int[] shortTokens = tok.Encode("the cat playing glowing dog", maxLength: 4, out int shortEos);
        Assert.Equal(4, shortTokens.Length);
        Assert.Equal(49406, shortTokens[0]);
        Assert.Equal(3, shortEos);
        Assert.Equal(49407, shortTokens[3]);
    }

    [Fact]
    public void ClipTokenizer_LoadFromDiskAndTokenizerJson_MatchesInMemoryExact()
    {
        var (vocab, merges) = BuildSampleClipVocabAndMerges();
        var memTok = ClipTokenizer.FromVocabularyAndMerges(vocab, merges);

        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_clip_tok_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            string vocabPath = Path.Combine(tempDir, "vocab.json");
            string mergesPath = Path.Combine(tempDir, "merges.txt");
            string tokenizerJsonPath = Path.Combine(tempDir, "tokenizer.json");
            string tokenizerJsonArrayMergesPath = Path.Combine(tempDir, "tokenizer_arr.json");

            var vocabSb = new StringBuilder("{");
            bool first = true;
            foreach (var kvp in vocab)
            {
                if (!first) vocabSb.Append(',');
                first = false;
                vocabSb.Append(System.Text.Json.JsonSerializer.Serialize(kvp.Key));
                vocabSb.Append(':');
                vocabSb.Append(kvp.Value);
            }
            vocabSb.Append('}');
            File.WriteAllText(vocabPath, vocabSb.ToString(), Encoding.UTF8);
            File.WriteAllLines(mergesPath, merges, Encoding.UTF8);

            var mergeLinesNoHeader = new List<string>();
            var mergeArraysNoHeader = new List<string[]>();
            foreach (string m in merges)
            {
                if (m.StartsWith("#", StringComparison.Ordinal)) continue;
                mergeLinesNoHeader.Add(m);
                mergeArraysNoHeader.Add(m.Split(' '));
            }

            string tokJson = $$"""
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 49406, "content": "<|startoftext|>" },
                { "id": 49407, "content": "<|endoftext|>" }
              ],
              "model": {
                "type": "BPE",
                "vocab": {{vocabSb}},
                "merges": {{System.Text.Json.JsonSerializer.Serialize(mergeLinesNoHeader)}}
              }
            }
            """;
            File.WriteAllText(tokenizerJsonPath, tokJson, Encoding.UTF8);

            string tokArrJson = $$"""
            {
              "version": "1.0",
              "model": {
                "type": "BPE",
                "vocab": {{vocabSb}},
                "merges": {{System.Text.Json.JsonSerializer.Serialize(mergeArraysNoHeader)}}
              }
            }
            """;
            File.WriteAllText(tokenizerJsonArrayMergesPath, tokArrJson, Encoding.UTF8);

            var fileTok = ClipTokenizer.Load(vocabPath, mergesPath);
            var jsonTok = ClipTokenizer.LoadFromTokenizerJson(tokenizerJsonPath);
            var jsonArrTok = ClipTokenizer.LoadFromTokenizerJson(tokenizerJsonArrayMergesPath);

            const string prompt = "The dog and the glowing cat's playing!";
            int[] expected = memTok.Encode(prompt, 77, out int expEos);
            int[] actualFile = fileTok.Encode(prompt, 77, out int fileEos);
            int[] actualJson = jsonTok.Encode(prompt, 77, out int jsonEos);
            int[] actualJsonArr = jsonArrTok.Encode(prompt, 77, out int jsonArrEos);

            Assert.Equal(expEos, fileEos);
            Assert.Equal(expEos, jsonEos);
            Assert.Equal(expEos, jsonArrEos);
            Assert.Equal(expected, actualFile);
            Assert.Equal(expected, actualJson);
            Assert.Equal(expected, actualJsonArr);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void FluxClipEncoder_Open_FailsFastWhenTokenizerMissing_AndRejectsNonF16Dtype()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_clip_open_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            string f16Path = WriteMinimalClipSafetensors(Path.Combine(tempDir, "clip_f16.safetensors"), dtype: "F16");
            // No vocab.json / merges.txt / tokenizer.json adjacent -> MUST fail fast with FileNotFoundException (and dispose SafetensorsFile)
            Assert.Throws<FileNotFoundException>(() => FluxClipEncoder.Open(f16Path));
            // Verify SafetensorsFile was cleanly disposed (no locked handle prevents deletion)
            File.Delete(f16Path);

            // Non-F16 dtype -> MUST throw NotSupportedException
            string f32Path = WriteMinimalClipSafetensors(Path.Combine(tempDir, "clip_f32.safetensors"), dtype: "F32");
            var (vocab, merges) = BuildSampleClipVocabAndMerges();
            var tok = ClipTokenizer.FromVocabularyAndMerges(vocab, merges);
            Assert.Throws<NotSupportedException>(() => FluxClipEncoder.Open(f32Path, tok));

            // Valid F16 + tokenizer/ subfolder auto-discovery succeeds and encodes distinct prompts distinctly
            string validPath = WriteMinimalClipSafetensors(Path.Combine(tempDir, "clip_l.safetensors"), dtype: "F16");
            string tokSubDir = Path.Combine(tempDir, "tokenizer");
            Directory.CreateDirectory(tokSubDir);
            var vocabJson = new StringBuilder("{");
            bool first = true;
            foreach (var kvp in vocab)
            {
                if (!first) vocabJson.Append(',');
                first = false;
                vocabJson.Append(System.Text.Json.JsonSerializer.Serialize(kvp.Key));
                vocabJson.Append(':');
                vocabJson.Append(kvp.Value);
            }
            vocabJson.Append('}');
            File.WriteAllText(Path.Combine(tokSubDir, "vocab.json"), vocabJson.ToString());
            File.WriteAllLines(Path.Combine(tokSubDir, "merges.txt"), merges);

            using var encoder = FluxClipEncoder.Open(validPath);
            float[] embA = encoder.EncodePrompt("the cat");
            float[] embB = encoder.EncodePrompt("the glowing cat playing dog");
            Assert.Equal(768, embA.Length);
            Assert.Equal(768, embB.Length);
            Assert.NotEqual(embA, embB);

            encoder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => encoder.EncodePrompt("the cat"));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public unsafe void FluxClipEncoder_SimdHalfToFloat_BitExactAcrossAll65536Values()
    {
        ushort[] allHalfBits = new ushort[65536];
        for (int i = 0; i < 65536; i++)
            allHalfBits[i] = (ushort)i;

        float[] simd16 = new float[65536];
        float[] simd8 = new float[65536];
        float[] simd4 = new float[65536];

        fixed (ushort* pIn = allHalfBits)
        fixed (float* pOut16 = simd16, pOut8 = simd8, pOut4 = simd4)
        {
            for (int i = 0; i < 65536; i += 16)
            {
                if (Vector512.IsHardwareAccelerated)
                    FluxClipEncoder.ConvertHalfToFloat16(pIn + i).Store(pOut16 + i);
                if (Vector256.IsHardwareAccelerated)
                {
                    FluxClipEncoder.ConvertHalfToFloat8(pIn + i).Store(pOut8 + i);
                    FluxClipEncoder.ConvertHalfToFloat8(pIn + i + 8).Store(pOut8 + i + 8);
                }
                if (Vector128.IsHardwareAccelerated)
                {
                    FluxClipEncoder.ConvertHalfToFloat4(pIn + i).Store(pOut4 + i);
                    FluxClipEncoder.ConvertHalfToFloat4(pIn + i + 4).Store(pOut4 + i + 4);
                    FluxClipEncoder.ConvertHalfToFloat4(pIn + i + 8).Store(pOut4 + i + 8);
                    FluxClipEncoder.ConvertHalfToFloat4(pIn + i + 12).Store(pOut4 + i + 12);
                }
            }
        }

        for (int i = 0; i < 65536; i++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)i);
            uint expBits = BitConverter.SingleToUInt32Bits(expected);

            if (Vector512.IsHardwareAccelerated)
                Assert.Equal(expBits, BitConverter.SingleToUInt32Bits(simd16[i]));
            if (Vector256.IsHardwareAccelerated)
                Assert.Equal(expBits, BitConverter.SingleToUInt32Bits(simd8[i]));
            if (Vector128.IsHardwareAccelerated)
                Assert.Equal(expBits, BitConverter.SingleToUInt32Bits(simd4[i]));
        }
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(1, 7, 15)]
    [InlineData(4, 16, 15)]
    [InlineData(7, 64, 15)]
    [InlineData(77, 768, 768)]
    public unsafe void FluxClipEncoder_MatMulF16_MatchesScalarOracle(int nRows, int inDim, int outDim)
    {
        var rng = new Random(nRows * 1009 + inDim * 37 + outDim);
        float[] x = new float[nRows * inDim];
        Half[] w = new Half[outDim * inDim];
        Half[] b = new Half[outDim];

        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((float)(rng.NextDouble() * 2.0 - 1.0) * 0.25f);
        for (int i = 0; i < b.Length; i++) b[i] = (Half)((float)(rng.NextDouble() * 0.5 - 0.25));

        float[] expected = new float[nRows * outDim];
        for (int r = 0; r < nRows; r++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float sum = (float)b[o];
                int xOff = r * inDim;
                int wOff = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    sum += x[xOff + i] * (float)w[wOff + i];
                }
                expected[r * outDim + o] = sum;
            }
        }

        float[] actual = new float[nRows * outDim];
        fixed (float* pX = x, pY = actual)
        fixed (Half* pW = w, pB = b)
        {
            FluxClipEncoder.MatMulF16(pX, pW, pB, pY, nRows, inDim, outDim);
        }

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-4f,
                $"MatMulF16 mismatch at index {i}: expected {expected[i]}, actual {actual[i]} (nRows={nRows}, inDim={inDim}, outDim={outDim})");
        }
    }

    private static string WriteMinimalClipSafetensors(string path, string dtype)
    {
        int elemBytes = dtype == "F16" ? 2 : 4;
        var specs = new List<(string Name, int[] Shape)>
        {
            ("text_model.embeddings.token_embedding.weight", [49408, 768]),
            ("text_model.embeddings.position_embedding.weight", [77, 768]),
            ("text_model.final_layer_norm.weight", [768]),
            ("text_model.final_layer_norm.bias", [768])
        };

        for (int i = 0; i < 12; i++)
        {
            string pfx = $"text_model.encoder.layers.{i}.";
            specs.Add((pfx + "layer_norm1.weight", [768]));
            specs.Add((pfx + "layer_norm1.bias", [768]));
            specs.Add((pfx + "self_attn.q_proj.weight", [768, 768]));
            specs.Add((pfx + "self_attn.q_proj.bias", [768]));
            specs.Add((pfx + "self_attn.k_proj.weight", [768, 768]));
            specs.Add((pfx + "self_attn.k_proj.bias", [768]));
            specs.Add((pfx + "self_attn.v_proj.weight", [768, 768]));
            specs.Add((pfx + "self_attn.v_proj.bias", [768]));
            specs.Add((pfx + "self_attn.out_proj.weight", [768, 768]));
            specs.Add((pfx + "self_attn.out_proj.bias", [768]));
            specs.Add((pfx + "layer_norm2.weight", [768]));
            specs.Add((pfx + "layer_norm2.bias", [768]));
            specs.Add((pfx + "mlp.fc1.weight", [3072, 768]));
            specs.Add((pfx + "mlp.fc1.bias", [3072]));
            specs.Add((pfx + "mlp.fc2.weight", [768, 3072]));
            specs.Add((pfx + "mlp.fc2.bias", [768]));
        }

        // Write compact header where small test tensors sit in a sparse zero-filled data section?
        // SafetensorsFile validates data offsets sequentially within file size, so allocate exact total size only for F32 error test or minimal valid F16 file.
        // Note: 49408*768*2 = ~75 MB is fast to write sparsely via FileStream.SetLength, and we populate distinct rows for token embeddings!
        ulong offset = 0;
        var headerSb = new StringBuilder("{");
        var tokEmbOffset = 0UL;
        for (int i = 0; i < specs.Count; i++)
        {
            if (i > 0) headerSb.Append(',');
            ulong count = 1;
            foreach (int d in specs[i].Shape) count *= (ulong)d;
            ulong bytes = count * (ulong)elemBytes;
            if (specs[i].Name == "text_model.embeddings.token_embedding.weight")
                tokEmbOffset = offset;

            headerSb.Append(System.Text.Json.JsonSerializer.Serialize(specs[i].Name));
            headerSb.Append(':');
            headerSb.Append($$"""{"dtype":"{{dtype}}","shape":[{{string.Join(',', specs[i].Shape)}}],"data_offsets":[{{offset}},{{offset + bytes}}]}""");
            offset += bytes;
        }
        headerSb.Append('}');

        byte[] headerBytes = Encoding.UTF8.GetBytes(headerSb.ToString());
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None))
        {
            using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);
            bw.Write((ulong)headerBytes.Length);
            bw.Write(headerBytes);
            bw.Flush();

            long dataStart = 8 + headerBytes.Length;
            fs.SetLength(dataStart + (long)offset);

            if (dtype == "F16")
            {
                // Populate non-zero final_layer_norm weight=1.0 and distinct token embedding rows for sample IDs
                // so EncodePrompt produces distinct outputs for different prompts.
                for (int tid = 0; tid < 49408; tid++)
                {
                    if (tid == 49406 || tid == 49407 || tid >= 500)
                    {
                        fs.Position = dataStart + (long)tokEmbOffset + (long)tid * 768 * 2;
                        ushort bits = BitConverter.HalfToUInt16Bits((Half)(((tid % 31) + 1) * 0.01f));
                        byte lo = (byte)(bits & 0xFF);
                        byte hi = (byte)(bits >> 8);
                        bw.Write(lo);
                        bw.Write(hi);
                    }
                }

                // Populate non-zero position embeddings per position and final_layer_norm weight=1.0
                // so EncodePrompt produces distinct outputs for prompts with different EOS positions.
                long posEmbOffset = 49408L * 768 * 2;
                for (int s = 0; s < 77; s++)
                {
                    fs.Position = dataStart + posEmbOffset + (long)s * 768 * 2;
                    for (int d = 0; d < 8; d++)
                    {
                        ushort posBits = BitConverter.HalfToUInt16Bits((Half)(((s + 1) * (d + 1)) * 0.01f));
                        bw.Write(posBits);
                    }
                }

                // final_layer_norm.weight at offset of spec #2
                long lnWeightOff = dataStart + ((49408L * 768 + 77L * 768) * 2);
                fs.Position = lnWeightOff;
                ushort oneBits = BitConverter.HalfToUInt16Bits((Half)1.0f);
                for (int d = 0; d < 768; d++)
                    bw.Write(oneBits);
            }
        }

        return path;
    }
}
