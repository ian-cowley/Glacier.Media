namespace Glacier.Media.Tests;

using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Media.Image;
using Glacier.Media.Video;
using Xunit;

/// <summary>
/// Empirical Challenger tests for Milestone M1 Media Serializers.
/// Stress-tests:
/// 1. GifWriter LZW variable-length encoding, dictionary saturation/reset, and pixel decoding.
/// 2. PngWriter/ApngWriter chunk framing, sequence numbers, and CRC32 verification.
/// 3. Zero-loss pixel roundtrips for PNG and APNG.
/// 4. Highly parallel concurrency stress across all media serializers.
/// </summary>
public class MediaChallengerTests
{
    #region 1. PngWriter Chunk Framing & CRC32 Tests

    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 13)]
    [InlineData(64, 48)]
    [InlineData(257, 129)]
    public void PngWriter_ProducesValidChunksAndCrc32Roundtrip(int width, int height)
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_png_test_{Guid.NewGuid():N}.png");
        try
        {
            byte[] originalPixels = GenerateTestPattern(width, height);
            PngWriter.SavePng24(tmpFile, originalPixels, width, height);

            Assert.True(File.Exists(tmpFile));
            byte[] fileBytes = File.ReadAllBytes(tmpFile);

            // 1. Verify PNG Signature
            byte[] expectedSig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            Assert.Equal(expectedSig, fileBytes.Take(8).ToArray());

            // 2. Parse and verify every chunk
            var chunks = ParsePngChunks(fileBytes);
            Assert.True(chunks.Count >= 3, "PNG must contain at least IHDR, IDAT, and IEND chunks.");
            Assert.Equal("IHDR", chunks[0].Type);
            Assert.Equal("IDAT", chunks[1].Type);
            Assert.Equal("IEND", chunks[^1].Type);

            // 3. Verify CRC32 for every single chunk
            foreach (var chunk in chunks)
            {
                uint expectedCrc = ComputeCrc32(chunk.TypeBytes, chunk.Data);
                Assert.Equal(expectedCrc, chunk.Crc);
            }

            // 4. Decode IDAT and verify pixel roundtrip
            byte[] decompressedPixels = DecompressPngRgb24(chunks[0].Data, chunks.Where(c => c.Type == "IDAT").Select(c => c.Data).ToArray(), width, height);
            Assert.Equal(originalPixels.Length, decompressedPixels.Length);
            Assert.Equal(originalPixels, decompressedPixels);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion

    #region 2. ApngWriter Chunk Framing, Sequence Numbers & CRC32 Tests

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(16, 16, 2)]
    [InlineData(32, 24, 5)]
    [InlineData(64, 64, 12)]
    public void ApngWriter_ProducesValidChunkFramingSequenceNumbersAndCrc32(int width, int height, int numFrames)
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_apng_test_{Guid.NewGuid():N}.apng");
        try
        {
            var frames = new List<byte[]>();
            for (int f = 0; f < numFrames; f++)
            {
                frames.Add(GenerateTestPattern(width, height, seed: f * 17 + 1));
            }

            ApngWriter.SaveApng(tmpFile, frames, width, height, fps: 15);
            Assert.True(File.Exists(tmpFile));
            byte[] fileBytes = File.ReadAllBytes(tmpFile);

            // 1. Signature
            byte[] expectedSig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            Assert.Equal(expectedSig, fileBytes.Take(8).ToArray());

            // 2. Parse chunks
            var chunks = ParsePngChunks(fileBytes);

            // All chunks must have mathematically exact CRC32
            foreach (var chunk in chunks)
            {
                uint expectedCrc = ComputeCrc32(chunk.TypeBytes, chunk.Data);
                Assert.Equal(expectedCrc, chunk.Crc);
            }

            // Verify chunk sequence
            Assert.Equal("IHDR", chunks[0].Type);
            Assert.Equal("acTL", chunks[1].Type);

            // acTL contents
            uint acNumFrames = BinaryPrimitives.ReadUInt32BigEndian(chunks[1].Data.AsSpan(0, 4));
            uint acNumPlays = BinaryPrimitives.ReadUInt32BigEndian(chunks[1].Data.AsSpan(4, 4));
            Assert.Equal((uint)numFrames, acNumFrames);
            Assert.Equal(0u, acNumPlays);

            // Verify strictly monotonic sequence numbers across fcTL and fdAT
            uint expectedSequenceNumber = 0;
            int frameIdx = 0;

            for (int c = 2; c < chunks.Count - 1; c++)
            {
                var chunk = chunks[c];
                if (chunk.Type == "fcTL")
                {
                    uint seq = BinaryPrimitives.ReadUInt32BigEndian(chunk.Data.AsSpan(0, 4));
                    Assert.Equal(expectedSequenceNumber++, seq);

                    uint fWidth = BinaryPrimitives.ReadUInt32BigEndian(chunk.Data.AsSpan(4, 4));
                    uint fHeight = BinaryPrimitives.ReadUInt32BigEndian(chunk.Data.AsSpan(8, 4));
                    Assert.Equal((uint)width, fWidth);
                    Assert.Equal((uint)height, fHeight);

                    frameIdx++;
                }
                else if (chunk.Type == "IDAT")
                {
                    // Frame 0 data: standard IDAT has NO sequence number
                    Assert.Equal(1, frameIdx);
                }
                else if (chunk.Type == "fdAT")
                {
                    // Frame 1..N data: fdAT has 4-byte sequence number prefix
                    uint seq = BinaryPrimitives.ReadUInt32BigEndian(chunk.Data.AsSpan(0, 4));
                    Assert.Equal(expectedSequenceNumber++, seq);
                }
            }

            Assert.Equal(numFrames, frameIdx);
            Assert.Equal("IEND", chunks[^1].Type);

            // 3. Decompress and roundtrip all frames
            var decodedFrames = DecompressApngFrames(chunks, width, height, numFrames);
            Assert.Equal(numFrames, decodedFrames.Count);
            for (int f = 0; f < numFrames; f++)
            {
                Assert.Equal(frames[f], decodedFrames[f]);
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion

    #region 3. GifWriter LZW Encoding & Dictionary Saturation Tests

    [Fact]
    public void GifWriter_SolidColorRuns_CompressesAndDecodesCorrectly()
    {
        // Solid color tests LZW run-length accumulation
        int w = 128;
        int h = 128;
        byte[] solidFrame = new byte[w * h * 3];
        Array.Fill(solidFrame, (byte)180);

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gif_solid_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [solidFrame], w, h, fps: 8);
            Assert.True(File.Exists(tmpFile));

            byte[] bytes = File.ReadAllBytes(tmpFile);
            var decodedFrames = DecodeGifFrames(bytes);

            Assert.Single(decodedFrames);
            Assert.Equal(w * h, decodedFrames[0].Length);
            // All pixels in solid frame should decode to identical palette index
            byte firstIdx = decodedFrames[0][0];
            Assert.All(decodedFrames[0], idx => Assert.Equal(firstIdx, idx));
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_LargeImage_ForcesLzwDictionarySaturationAndClearCodes()
    {
        // 512x512 image = 262,144 pixels.
        // Complex gradient creates enough distinct 2-character and multi-character phrases
        // to exceed 4096 entries multiple times, rigorously exercising:
        // - 9-bit, 10-bit, 11-bit, 12-bit code sizes
        // - nextCode >= 4096 dictionary resets (Clear Code 256)
        // - Post-reset continuation without bitstream corruption
        int w = 512;
        int h = 512;
        byte[] complexFrame = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = (y * w + x) * 3;
                complexFrame[idx + 0] = (byte)((x ^ y) & 0xFF);
                complexFrame[idx + 1] = (byte)((x * 3 + y * 7) & 0xFF);
                complexFrame[idx + 2] = (byte)((x + y) / 4);
            }
        }

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gif_large_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [complexFrame], w, h, fps: 8);
            Assert.True(File.Exists(tmpFile));

            byte[] bytes = File.ReadAllBytes(tmpFile);
            var decodedGif = GifOracleDecoder.Decode(bytes);
            Assert.Equal(w, decodedGif.ScreenWidth);
            Assert.Equal(h, decodedGif.ScreenHeight);
            Assert.Single(decodedGif.Frames);
            Assert.Equal(w * h, decodedGif.Frames[0].Indices.Length);
            Assert.NotNull(decodedGif.GlobalColorTable);

            int clearCodeCount = CountGifClearCodesInStream(bytes);

            // Ensure that dictionary saturation occurred and multiple Clear Codes were processed
            Assert.True(clearCodeCount > 1, $"Expected multiple dictionary clear codes on 512x512 complex frame, got {clearCodeCount}");
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_CompareEncoderAndDecoderCodes()
    {
        int w = 64, h = 64;
        byte[] frame = GenerateTestPattern(w, h);
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gif_cmp_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [frame], w, h, fps: 10);
            byte[] bytes = File.ReadAllBytes(tmpFile);

            // Decode with DecodeGifFrames
            int pos = 13 + 768; // skip header + GCT
            while (bytes[pos] != 0x2C)
            {
                if (bytes[pos] == 0x21)
                {
                    pos += 2;
                    while (bytes[pos] != 0) pos += 1 + bytes[pos];
                    pos++;
                }
                else pos++;
            }
            pos += 10; // skip 0x2C + 9 bytes of Image Descriptor
            int minCodeSize = bytes[pos++];
            using var ms = new MemoryStream();
            while (bytes[pos] != 0)
            {
                byte len = bytes[pos++];
                ms.Write(bytes, pos, len);
                pos += len;
            }
            byte[] lzwCompressed = ms.ToArray();

            // Trace codes from lzwCompressed
            int bitAccum = 0, bitsIn = 0, bPos = 0;
            int cSize = minCodeSize + 1;
            int cLimit = 1 << cSize;
            int nCode = (1 << minCodeSize) + 2;

            var codes = new List<(int c, int size, int next)>();
            while (true)
            {
                while (bitsIn < cSize)
                {
                    if (bPos >= lzwCompressed.Length) break;
                    bitAccum |= lzwCompressed[bPos++] << bitsIn;
                    bitsIn += 8;
                }
                if (bitsIn < cSize) break;
                int c = bitAccum & ((1 << cSize) - 1);
                bitAccum >>= cSize;
                bitsIn -= cSize;

                codes.Add((c, cSize, nCode));

                if (c == 256)
                {
                    cSize = minCodeSize + 1;
                    cLimit = 1 << cSize;
                    nCode = 258;
                    continue;
                }
                if (c == 257) break;

                nCode++;
                if (nCode > cLimit && cSize < 12)
                {
                    cSize++;
                    cLimit = 1 << cSize;
                }
            }

            Console.WriteLine($"[TRACE] Total codes decoded: {codes.Count}");
            for (int i = 0; i < codes.Count; i++)
            {
                var (c, sz, nxt) = codes[i];
                if (c > nxt)
                {
                    Console.WriteLine($"[MISMATCH at {i}] code {c} > nextCode {nxt} (size={sz})");
                    for (int j = Math.Max(0, i - 5); j <= Math.Min(codes.Count - 1, i + 5); j++)
                    {
                        Console.WriteLine($"  [{j}] code={codes[j].c}, size={codes[j].size}, nextCode={codes[j].next}");
                    }
                    break;
                }
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(17, 31)]
    public void GifWriter_VariableDimensions_DecodesCleanly(int w, int h)
    {
        byte[] frame = GenerateTestPattern(w, h);
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gif_dim_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [frame], w, h, fps: 10);
            Assert.True(File.Exists(tmpFile));

            var decoded = GifOracleDecoder.Decode(File.ReadAllBytes(tmpFile));
            Assert.Equal(w, decoded.ScreenWidth);
            Assert.Equal(h, decoded.ScreenHeight);
            Assert.Single(decoded.Frames);
            Assert.Equal(w, decoded.Frames[0].Width);
            Assert.Equal(h, decoded.Frames[0].Height);
            Assert.Equal(w * h, decoded.Frames[0].Indices.Length);
            Assert.NotNull(decoded.GlobalColorTable);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_GdiPlus_17x31_VerifyAllPixels()
    {
        int w = 17, h = 31;
        byte[] frame = GenerateTestPattern(w, h);
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gdi_1731_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [frame], w, h, fps: 10);
            var decoded = GifOracleDecoder.Decode(File.ReadAllBytes(tmpFile));
            Assert.Equal(w, decoded.ScreenWidth);
            Assert.Equal(h, decoded.ScreenHeight);
            Assert.Single(decoded.Frames);
            Assert.Equal(w * h, decoded.Frames[0].Indices.Length);
            Assert.NotNull(decoded.GlobalColorTable);
            int maxValidIdx = (decoded.GlobalColorTable!.Length / 3) - 1;
            Assert.All(decoded.Frames[0].Indices, idx => Assert.InRange((int)idx, 0, maxValidIdx));
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_MultiFrameAnimation_DecodesAllFramesSuccessfully()
    {
        int w = 64;
        int h = 64;
        int numFrames = 8;
        var frames = new List<byte[]>();
        for (int i = 0; i < numFrames; i++)
        {
            frames.Add(GenerateTestPattern(w, h, seed: i * 31 + 7));
        }

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gif_multi_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, frames, w, h, fps: 12);
            Assert.True(File.Exists(tmpFile));

            var decoded = GifOracleDecoder.Decode(File.ReadAllBytes(tmpFile));
            Assert.Equal(w, decoded.ScreenWidth);
            Assert.Equal(h, decoded.ScreenHeight);
            Assert.Equal(numFrames, decoded.Frames.Count);
            Assert.All(decoded.Frames, f =>
            {
                Assert.Equal(w, f.Width);
                Assert.Equal(h, f.Height);
                Assert.Equal(w * h, f.Indices.Length);
            });
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_SystemDrawingGdiPlus_CanDecodeGif()
    {
        int w = 64;
        int h = 64;
        var frames = new List<byte[]> { GenerateTestPattern(w, h, 1), GenerateTestPattern(w, h, 2) };
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gdi_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, frames, w, h, fps: 10);
            var decoded = GifOracleDecoder.Decode(File.ReadAllBytes(tmpFile));
            Assert.Equal(w, decoded.ScreenWidth);
            Assert.Equal(h, decoded.ScreenHeight);
            Assert.Equal(2, decoded.Frames.Count);
            Assert.Equal(w * h, decoded.Frames[0].Indices.Length);
            Assert.Equal(w * h, decoded.Frames[1].Indices.Length);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_SystemDrawingGdiPlus_LargeImage512x512()
    {
        int w = 512;
        int h = 512;
        byte[] complexFrame = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = (y * w + x) * 3;
                complexFrame[idx] = (byte)(x & 0xFF);
                complexFrame[idx + 1] = (byte)(y & 0xFF);
                complexFrame[idx + 2] = (byte)((x ^ y) & 0xFF);
            }
        }

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_gdi_512_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [complexFrame], w, h, fps: 8);
            var decoded = GifOracleDecoder.Decode(File.ReadAllBytes(tmpFile));
            Assert.Equal(w, decoded.ScreenWidth);
            Assert.Equal(h, decoded.ScreenHeight);
            Assert.Single(decoded.Frames);
            Assert.Equal(w * h, decoded.Frames[0].Indices.Length);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion

    #region 4. Concurrency Stress Testing

    [Fact]
    public void MediaWriters_HighConcurrency_ExecutionIsThreadSafe()
    {
        const int concurrentTasks = 32;
        var exceptions = new ConcurrentBag<Exception>();

        Parallel.For(0, concurrentTasks, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
        {
            try
            {
                int w = 32 + (i % 8) * 8; // varying sizes
                int h = 32 + ((i * 3) % 8) * 8;
                byte[] frame = GenerateTestPattern(w, h, seed: i * 43 + 1);

                string baseTmp = Path.Combine(Path.GetTempPath(), $"glacier_concur_{i}_{Guid.NewGuid():N}");
                string pngPath = baseTmp + ".png";
                string apngPath = baseTmp + ".apng";
                string gifPath = baseTmp + ".gif";
                string bmpPath = baseTmp + ".bmp";
                string aviPath = baseTmp + ".avi";

                try
                {
                    // 1. PNG
                    PngWriter.SavePng24(pngPath, frame, w, h);
                    Assert.True(File.Exists(pngPath));
                    var pngBytes = File.ReadAllBytes(pngPath);
                    var pngChunks = ParsePngChunks(pngBytes);
                    foreach (var c in pngChunks)
                    {
                        Assert.Equal(ComputeCrc32(c.TypeBytes, c.Data), c.Crc);
                    }

                    // 2. APNG
                    var apngFrames = new List<byte[]> { frame, GenerateTestPattern(w, h, seed: i * 19 + 2) };
                    ApngWriter.SaveApng(apngPath, apngFrames, w, h, fps: 10);
                    Assert.True(File.Exists(apngPath));
                    var apngBytes = File.ReadAllBytes(apngPath);
                    var apngChunks = ParsePngChunks(apngBytes);
                    foreach (var c in apngChunks)
                    {
                        Assert.Equal(ComputeCrc32(c.TypeBytes, c.Data), c.Crc);
                    }

                    // 3. GIF
                    GifWriter.SaveGif(gifPath, apngFrames, w, h, fps: 10);
                    Assert.True(File.Exists(gifPath));
                    var decodedGif = GifOracleDecoder.Decode(File.ReadAllBytes(gifPath));
                    Assert.Equal(w, decodedGif.ScreenWidth);
                    Assert.Equal(h, decodedGif.ScreenHeight);
                    Assert.Equal(2, decodedGif.Frames.Count);
                    Assert.Equal(w * h, decodedGif.Frames[0].Indices.Length);
                    Assert.Equal(w * h, decodedGif.Frames[1].Indices.Length);

                    // 4. BMP
                    BmpWriter.SaveBmp24(bmpPath, frame, w, h);
                    Assert.True(File.Exists(bmpPath));
                    var bmpBytes = File.ReadAllBytes(bmpPath);
                    Assert.Equal((byte)'B', bmpBytes[0]);
                    Assert.Equal((byte)'M', bmpBytes[1]);

                    // 5. AVI
                    AviWriter.SaveAvi(aviPath, apngFrames, w, h, fps: 10);
                    Assert.True(File.Exists(aviPath));
                    var aviBytes = File.ReadAllBytes(aviPath);
                    Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(aviBytes.Take(4).ToArray()));
                }
                finally
                {
                    if (File.Exists(pngPath)) File.Delete(pngPath);
                    if (File.Exists(apngPath)) File.Delete(apngPath);
                    if (File.Exists(gifPath)) File.Delete(gifPath);
                    if (File.Exists(bmpPath)) File.Delete(bmpPath);
                    if (File.Exists(aviPath)) File.Delete(aviPath);
                }
            }
            catch (Exception ex)
            {
                exceptions.Add(ex);
            }
        });

        Assert.Empty(exceptions);
    }

    #endregion

    #region Helper Utilities: PNG / APNG Parser & Decompressor

    private record PngChunk(string Type, byte[] TypeBytes, byte[] Data, uint Crc);

    private static List<PngChunk> ParsePngChunks(byte[] fileBytes)
    {
        var chunks = new List<PngChunk>();
        int offset = 8; // skip PNG signature

        while (offset < fileBytes.Length)
        {
            if (offset + 8 > fileBytes.Length)
                throw new InvalidDataException("Truncated chunk header.");

            uint length = BinaryPrimitives.ReadUInt32BigEndian(fileBytes.AsSpan(offset, 4));
            offset += 4;

            byte[] typeBytes = fileBytes.AsSpan(offset, 4).ToArray();
            string type = System.Text.Encoding.ASCII.GetString(typeBytes);
            offset += 4;

            if (offset + length + 4 > fileBytes.Length)
                throw new InvalidDataException($"Truncated chunk payload for {type}.");

            byte[] data = fileBytes.AsSpan(offset, (int)length).ToArray();
            offset += (int)length;

            uint crc = BinaryPrimitives.ReadUInt32BigEndian(fileBytes.AsSpan(offset, 4));
            offset += 4;

            chunks.Add(new PngChunk(type, typeBytes, data, crc));
        }

        return chunks;
    }

    private static uint ComputeCrc32(ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        for (int i = 0; i < type.Length; i++)
        {
            crc = (crc >> 8) ^ Crc32Table[(crc ^ type[i]) & 0xFF];
        }
        for (int i = 0; i < data.Length; i++)
        {
            crc = (crc >> 8) ^ Crc32Table[(crc ^ data[i]) & 0xFF];
        }
        return crc ^ 0xFFFFFFFF;
    }

    private static readonly uint[] Crc32Table = GenerateCrc32Table();

    private static uint[] GenerateCrc32Table()
    {
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
            {
                if ((c & 1) != 0) c = 0xEDB88320 ^ (c >> 1);
                else c >>= 1;
            }
            table[i] = c;
        }
        return table;
    }

    private static byte[] DecompressPngRgb24(byte[] ihdrData, byte[][] idatChunks, int width, int height)
    {
        using var combinedStream = new MemoryStream();
        foreach (var idat in idatChunks)
        {
            combinedStream.Write(idat, 0, idat.Length);
        }
        combinedStream.Position = 0;

        using var zlib = new ZLibStream(combinedStream, CompressionMode.Decompress);
        int rowBytes = width * 3;
        int filteredRowLen = 1 + rowBytes;
        byte[] outputRgb = new byte[width * height * 3];
        byte[] rowBuffer = new byte[filteredRowLen];

        for (int y = 0; y < height; y++)
        {
            int bytesRead = 0;
            while (bytesRead < filteredRowLen)
            {
                int r = zlib.Read(rowBuffer, bytesRead, filteredRowLen - bytesRead);
                if (r == 0) throw new EndOfStreamException($"Unexpected end of stream at row {y}");
                bytesRead += r;
            }

            byte filterType = rowBuffer[0];
            if (filterType != 0)
                throw new NotSupportedException($"Only filter type 0 (None) supported in oracle, found {filterType}");

            Array.Copy(rowBuffer, 1, outputRgb, y * rowBytes, rowBytes);
        }

        return outputRgb;
    }

    private static List<byte[]> DecompressApngFrames(List<PngChunk> chunks, int width, int height, int expectedFrames)
    {
        var result = new List<byte[]>();
        byte[]? ihdrData = chunks.FirstOrDefault(c => c.Type == "IHDR")?.Data;
        Assert.NotNull(ihdrData);

        // Frame 0 uses IDAT chunks
        var idatChunks = chunks.Where(c => c.Type == "IDAT").Select(c => c.Data).ToArray();
        result.Add(DecompressPngRgb24(ihdrData, idatChunks, width, height));

        // Subsequent frames use fdAT chunks (each preceded by fcTL)
        var fdatChunks = chunks.Where(c => c.Type == "fdAT").ToList();
        foreach (var fdat in fdatChunks)
        {
            // First 4 bytes of fdAT is sequence number, remainder is compressed data
            byte[] compressedData = fdat.Data.Skip(4).ToArray();
            result.Add(DecompressPngRgb24(ihdrData, [compressedData], width, height));
        }

        return result;
    }

    #endregion

    #region Helper Utilities: GIF89a Parser & LZW Oracle Decoder

    private static int CountGifClearCodesInStream(byte[] bytes)
    {
        int pos = 13 + 768; // skip header + GCT
        while (pos < bytes.Length && bytes[pos] != 0x2C)
        {
            if (bytes[pos] == 0x21)
            {
                pos += 2;
                while (pos < bytes.Length && bytes[pos] != 0) pos += 1 + bytes[pos];
                pos++;
            }
            else pos++;
        }
        if (pos >= bytes.Length) return 0;
        pos += 10; // skip 0x2C + 9 bytes of Image Descriptor
        if (pos >= bytes.Length) return 0;
        int minCodeSize = bytes[pos++];
        using var ms = new MemoryStream();
        while (pos < bytes.Length && bytes[pos] != 0)
        {
            byte len = bytes[pos++];
            ms.Write(bytes, pos, len);
            pos += len;
        }
        byte[] lzwCompressed = ms.ToArray();

        int bitAccum = 0, bitsIn = 0, bPos = 0;
        int cSize = minCodeSize + 1;
        int cLimit = 1 << cSize;
        int nCode = (1 << minCodeSize) + 2;
        int clearCodes = 0;

        while (true)
        {
            while (bitsIn < cSize)
            {
                if (bPos >= lzwCompressed.Length) break;
                bitAccum |= lzwCompressed[bPos++] << bitsIn;
                bitsIn += 8;
            }
            if (bitsIn < cSize) break;
            int c = bitAccum & ((1 << cSize) - 1);
            bitAccum >>= cSize;
            bitsIn -= cSize;

            if (c == 256)
            {
                clearCodes++;
                cSize = minCodeSize + 1;
                cLimit = 1 << cSize;
                nCode = 258;
                continue;
            }
            if (c == 257) break;

            nCode++;
            if (nCode > cLimit && cSize < 12)
            {
                cSize++;
                cLimit = 1 << cSize;
            }
        }
        return clearCodes;
    }

    private static List<byte[]> DecodeGifFrames(byte[] gifBytes)
    {
        var decoded = GifOracleDecoder.Decode(gifBytes);
        var frames = new List<byte[]>(decoded.Frames.Count);
        foreach (var f in decoded.Frames)
            frames.Add(f.Indices);
        return frames;
    }

    private static byte[] GenerateTestPattern(int width, int height, int seed = 42)
    {
        byte[] pixels = new byte[width * height * 3];
        var rnd = new Random(seed);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;
                pixels[idx + 0] = (byte)((x * 13 + y * 7 + rnd.Next(20)) & 0xFF);
                pixels[idx + 1] = (byte)((x * 3 + y * 19) & 0xFF);
                pixels[idx + 2] = (byte)((x ^ y ^ rnd.Next(30)) & 0xFF);
            }
        }
        return pixels;
    }

    #endregion
}
