namespace Glacier.Media.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Media.Image.Flux;
using Xunit;

public class FluxT5EncoderTests
{
    private static string WriteMinimalT5Gguf(string path)
    {
        string[] tokens =
        [
            "<pad>",       // 0
            "</s>",        // 1
            "<unk>",       // 2
            "\u2581a",     // 3
            "\u2581glow",  // 4
            "ing",         // 5
            "\u2581glowing", // 6 (higher score than glow + ing)
            "\u2581crystal", // 7
            "\u2581orb",   // 8
            "\u2581in",    // 9
            "z"            // 10
        ];
        float[] scores = [0f, 0f, -100f, -1.0f, -2.0f, -2.0f, -1.5f, -1.2f, -1.3f, -1.1f, -3.0f];

        var tensorNames = new List<string>
        {
            "token_embd.weight",
            "enc.output_norm.weight",
            "enc.blk.0.attn_rel_b.weight"
        };
        for (int l = 0; l < FluxT5Encoder.NumLayers; l++)
        {
            string pfx = $"enc.blk.{l}.";
            tensorNames.Add(pfx + "attn_norm.weight");
            tensorNames.Add(pfx + "attn_q.weight");
            tensorNames.Add(pfx + "attn_k.weight");
            tensorNames.Add(pfx + "attn_v.weight");
            tensorNames.Add(pfx + "attn_o.weight");
            tensorNames.Add(pfx + "ffn_norm.weight");
            tensorNames.Add(pfx + "ffn_gate.weight");
            tensorNames.Add(pfx + "ffn_up.weight");
            tensorNames.Add(pfx + "ffn_down.weight");
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

        void WriteGgufString(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)bytes.Length);
            bw.Write(bytes);
        }

        bw.Write(0x46554747u); // 'GGUF'
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensorNames.Count);
        bw.Write(2UL);         // 2 metadata KVs

        // 1. tokenizer.ggml.tokens (ARRAY = 9, STRING = 8)
        WriteGgufString("tokenizer.ggml.tokens");
        bw.Write(9u);
        bw.Write(8u);
        bw.Write((ulong)tokens.Length);
        foreach (string t in tokens) WriteGgufString(t);

        // 2. tokenizer.ggml.scores (ARRAY = 9, FLOAT32 = 6)
        WriteGgufString("tokenizer.ggml.scores");
        bw.Write(9u);
        bw.Write(6u);
        bw.Write((ulong)scores.Length);
        foreach (float sc in scores) bw.Write(sc);

        // Tensor directory: all point to offset 0, 1-element F32 (ggmlType = 0)
        foreach (string tName in tensorNames)
        {
            WriteGgufString(tName);
            bw.Write(1u); // nDims
            bw.Write(1UL); // dim[0]
            bw.Write(0u); // F32
            bw.Write(0UL); // offset
        }

        // Pad to 32-byte alignment + 4 bytes data
        long pos = fs.Position;
        long aligned = (pos + 31) & ~31L;
        while (fs.Position < aligned) bw.Write((byte)0);
        bw.Write(0.0f);

        return path;
    }

    [Fact]
    public void FluxT5Encoder_ViterbiTokenize_SelectsHighestScoringPath_AndHandlesEdgeCases()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_t5_mock_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteMinimalT5Gguf(path);
            using var encoder = FluxT5Encoder.Open(path);

            // "glowing" has score -1.5 > (-2.0 + -2.0 = -4.0), so Viterbi must pick token 6
            var (tokens, validCount) = encoder.Tokenize("  a glowing crystal orb  ", maxTokens: 16);
            Assert.Equal(16, tokens.Length);
            Assert.Equal(5, validCount); // 4 tokens + EOS
            Assert.Equal(3, tokens[0]);  // "\u2581a"
            Assert.Equal(6, tokens[1]);  // "\u2581glowing"
            Assert.Equal(7, tokens[2]);  // "\u2581crystal"
            Assert.Equal(8, tokens[3]);  // "\u2581orb"
            Assert.Equal(1, tokens[4]);  // EOS
            for (int i = 5; i < 16; i++) Assert.Equal(0, tokens[i]); // PAD

            // Empty prompt -> [EOS, PAD...]
            var (emptyTokens, emptyValid) = encoder.Tokenize("   ", maxTokens: 8);
            Assert.Equal(1, emptyValid);
            Assert.Equal(1, emptyTokens[0]);

            // Unknown character falls back to <unk> (2)
            var (unkTokens, unkValid) = encoder.Tokenize("q", maxTokens: 8);
            Assert.Equal(3, unkValid); // "\u2581" unknown -> 2, "q" -> 2, EOS -> 1
            Assert.Equal(2, unkTokens[0]);
            Assert.Equal(2, unkTokens[1]);
            Assert.Equal(1, unkTokens[2]);

            // Disposal & null guards
            Assert.Throws<ArgumentNullException>(() => encoder.Tokenize(null!));
            encoder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => encoder.Tokenize("a glowing crystal orb"));
            Assert.Throws<ObjectDisposedException>(() => encoder.Encode("a glowing crystal orb"));
            Assert.Throws<ObjectDisposedException>(() => encoder.EncodeWithCount("a glowing crystal orb"));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void FluxT5Encoder_Tokenize_UsesSpanLookupWithoutSubstringChurn()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_t5_alloc_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteMinimalT5Gguf(path);
            using var encoder = FluxT5Encoder.Open(path);

            string longPrompt = string.Join(' ', System.Linq.Enumerable.Repeat("a glowing crystal orb in", 10));

            // Warmup JIT
            for (int i = 0; i < 10; i++)
                _ = encoder.Tokenize(longPrompt, maxTokens: 128);

            // Measure allocated bytes for Tokenize on long prompt (~250 chars -> old Substring path would allocate ~8,000 strings / >250 KB)
            long before = GC.GetAllocatedBytesForCurrentThread();
            var (tokens, validCount) = encoder.Tokenize(longPrompt, maxTokens: 128);
            long after = GC.GetAllocatedBytesForCurrentThread();

            long allocBytes = after - before;
            Assert.True(validCount > 20);
            Assert.Equal(128, tokens.Length);
            // Only the 2 strings from space-prefix + Replace and the returned int[128] (~544B) should allocate; well under 4 KB
            Assert.True(allocBytes < 4096, $"Expected < 4096 bytes allocated by pooled span-based Tokenize, got {allocBytes} bytes.");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
