namespace Glacier.Media.Tests;

using System;
using System.IO;
using Glacier.Media.Image;
using Glacier.Media.Image.Flux;
using Xunit;

#pragma warning disable GLACIER_EXP001
public class ImageGenerationTests
{
    [Fact]
    public void BmpWriter_WritesValid24BitBitmapHeader()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.bmp");
        try
        {
            int w = 64;
            int h = 48;
            byte[] pixels = new byte[w * h * 3];
            Array.Fill(pixels, (byte)200);

            BmpWriter.SaveBmp24(tmpFile, pixels, w, h);

            Assert.True(File.Exists(tmpFile));
            byte[] bytes = File.ReadAllBytes(tmpFile);
            Assert.True(bytes.Length > 54);

            // Verify 'BM' signature
            Assert.Equal((byte)'B', bytes[0]);
            Assert.Equal((byte)'M', bytes[1]);

            // Verify width and height in DIB header
            int width = BitConverter.ToInt32(bytes, 18);
            int height = Math.Abs(BitConverter.ToInt32(bytes, 22));
            short bpp = BitConverter.ToInt16(bytes, 28);

            Assert.Equal(w, width);
            Assert.Equal(h, height);
            Assert.Equal(24, bpp);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void PngWriter_WritesValid24BitPngFile()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.png");
        try
        {
            int w = 64;
            int h = 48;
            byte[] pixels = new byte[w * h * 3];
            Array.Fill(pixels, (byte)180);

            PngWriter.SavePng24(tmpFile, pixels, w, h);

            Assert.True(File.Exists(tmpFile));
            byte[] bytes = File.ReadAllBytes(tmpFile);
            Assert.True(bytes.Length > 8);

            // Verify PNG magic signature: 0x89 'P' 'N' 'G' '\r' '\n' 0x1A '\n'
            byte[] expectedSig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(expectedSig[i], bytes[i]);
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }


    [Fact]
    public void FlowMatchingScheduler_GeneratesDecreasingTrajectoryAndSteps()
    {
        int steps = 4;
        var scheduler = new FlowMatchingScheduler(steps);
        var timesteps = scheduler.Timesteps;

        Assert.Equal(steps + 1, timesteps.Length);
        Assert.Equal(1.0f, timesteps[0]);
        Assert.Equal(0.0f, timesteps[^1]);

        for (int i = 0; i < steps; i++)
        {
            Assert.True(timesteps[i] > timesteps[i + 1]);
        }

        // Test Euler step: x_{t + dt} = x_t + dt * v
        float[] latents = [1.0f, 2.0f];
        float[] velocity = [0.5f, -1.0f];
        FlowMatchingScheduler.Step(latents, velocity, 1.0f, 0.75f); // dt = -0.25

        Assert.Equal(1.0f - 0.25f * 0.5f, latents[0], precision: 5);
        Assert.Equal(2.0f - 0.25f * (-1.0f), latents[1], precision: 5);
    }

    [Fact]
    public void DiffusionTransformer_PredictsFiniteVelocity()
    {
        int latentH = 16;
        int latentW = 16;
        int channels = 16;
        int totalLatents = channels * latentH * latentW;

        using var dit = new DiffusionTransformer(numLayers: 2, hiddenDim: 128, numHeads: 4, latentChannels: 16);

        float[] latents = new float[totalLatents];
        float[] velocity = new float[totalLatents];
        float[] promptEmb = new float[128];

        for (int i = 0; i < latents.Length; i++) latents[i] = MathF.Sin(i * 0.05f);

        dit.PredictVelocity(latents, latentH, latentW, 0.5f, promptEmb, velocity);

        bool hasNonZero = false;
        for (int i = 0; i < velocity.Length; i++)
        {
            Assert.False(float.IsNaN(velocity[i]));
            Assert.False(float.IsInfinity(velocity[i]));
            if (MathF.Abs(velocity[i]) > 1e-4f) hasNonZero = true;
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void LatentVaeDecoder_Expands8xSpatiallyToValidRgb()
    {
        int latentH = 8;
        int latentW = 8;
        int channels = 16;
        int targetH = latentH * 8; // 64
        int targetW = latentW * 8; // 64

        using var vae = new LatentVaeDecoder(channels);

        float[] latents = new float[channels * latentH * latentW];
        for (int i = 0; i < latents.Length; i++) latents[i] = 0.5f;

        byte[] rgb = new byte[targetH * targetW * 3];
        vae.Decode(latents, latentH, latentW, rgb);

        Assert.NotEmpty(rgb);
        Assert.Equal(targetH * targetW * 3, rgb.Length);
    }

    [Fact]
    public void ImageGenerationPipeline_GeneratesCompleteImage()
    {
        using var pipeline = new ImageGenerationPipeline(numLayers: 2, hiddenDim: 128, numHeads: 4);

        int w = 64;
        int h = 64;
        var result = pipeline.Generate("A futuristic cybernetic glacier city at sunset", width: w, height: h, numSteps: 2, seed: 42);

        Assert.Equal(w, result.Width);
        Assert.Equal(h, result.Height);
        Assert.Equal(2, result.Steps);
        Assert.True(result.ElapsedMilliseconds >= 0);
        Assert.Equal(w * h * 3, result.RgbPixels.Length);
    }

    [Fact]
    public void PipelineGeneration_ProducesValidContinuousImage()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_pipeline_{Guid.NewGuid():N}.bmp");
        try
        {
            const int dim = 128;
            using var pipeline = new ImageGenerationPipeline(numLayers: 2, hiddenDim: 128, numHeads: 4);
            var result = pipeline.Generate("A majestic crystal glacier fortress illuminated by cyan auroras", width: dim, height: dim, numSteps: 2, seed: 42);

            BmpWriter.SaveBmp24(tmpFile, result.RgbPixels, dim, dim);
            Assert.True(File.Exists(tmpFile));

            // Verify smooth spatial continuity: no scanline alternation between adjacent rows
            for (int y = 8; y < dim - 8; y += 8)
            {
                int idxCurrent = (y * dim + (dim / 2)) * 3;
                int idxPrev = ((y - 1) * dim + (dim / 2)) * 3;

                // Differences between consecutive scanlines must be smooth (< 40 delta, not 255 black/white flip)
                int deltaR = Math.Abs(result.RgbPixels[idxCurrent] - result.RgbPixels[idxPrev]);
                int deltaG = Math.Abs(result.RgbPixels[idxCurrent + 1] - result.RgbPixels[idxPrev + 1]);
                int deltaB = Math.Abs(result.RgbPixels[idxCurrent + 2] - result.RgbPixels[idxPrev + 2]);

                Assert.True(deltaR < 40, $"Scanline jump in R at y={y}: delta={deltaR}");
                Assert.True(deltaG < 40, $"Scanline jump in G at y={y}: delta={deltaG}");
                Assert.True(deltaB < 40, $"Scanline jump in B at y={y}: delta={deltaB}");
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void ImageDecoder_RejectsMaliciousPngBmpBounds_AndDecodesTopDownBmpCorrectly()
    {
        // Truncated PNG chunk length should throw InvalidDataException
        byte[] badPng =
        [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, (byte)'I', (byte)'H', (byte)'D', (byte)'R',
            0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00,
            0xFD, 0xD4, 0x9A, 0x73,
            0x7F, 0xFF, 0xFF, 0xFF, (byte)'I', (byte)'D', (byte)'A', (byte)'T', 0x00, 0x00, 0x00, 0x00
        ];
        Assert.Throws<InvalidDataException>(() => ImageDecoder.DecodePng(badPng));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.DecodePngRgba(badPng));

        // Top-down BMP (height = -2, width = 1, 24-bit BGR) + int.MinValue guard
        byte[] topDownBmp = new byte[54 + 8];
        topDownBmp[0] = (byte)'B';
        topDownBmp[1] = (byte)'M';
        BitConverter.TryWriteBytes(topDownBmp.AsSpan(10, 4), 54); // pixelOffset
        BitConverter.TryWriteBytes(topDownBmp.AsSpan(18, 4), 1);  // width = 1
        BitConverter.TryWriteBytes(topDownBmp.AsSpan(22, 4), -2); // top-down height = 2
        BitConverter.TryWriteBytes(topDownBmp.AsSpan(28, 2), (short)24);
        // Row 0 (top): R=10, G=20, B=30 (stored BGR)
        topDownBmp[54] = 30; topDownBmp[55] = 20; topDownBmp[56] = 10;
        // Row 1 (bottom): R=200, G=100, B=50 (stored BGR)
        topDownBmp[58] = 50; topDownBmp[59] = 100; topDownBmp[60] = 200;

        var (pixels, w, h) = ImageDecoder.DecodeBmp(topDownBmp);
        Assert.Equal(1, w);
        Assert.Equal(2, h);
        Assert.Equal((byte)10, pixels[0]);
        Assert.Equal((byte)200, pixels[3]);

        // int.MinValue height guard
        BitConverter.TryWriteBytes(topDownBmp.AsSpan(22, 4), int.MinValue);
        Assert.Throws<InvalidDataException>(() => ImageDecoder.DecodeBmp(topDownBmp));
    }

    [Fact]
    public void FluxVaeDecoder_RejectsNonF32SafetensorsDtype()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_vae_f16_{Guid.NewGuid():N}.safetensors");
        try
        {
            byte[] headerJson = System.Text.Encoding.UTF8.GetBytes(
                """{"decoder.conv_in.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}""");
            using (var fs = File.Create(path))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write((ulong)headerJson.Length);
                bw.Write(headerJson);
                bw.Write((ushort)0);
            }

            using var vae = FluxVaeDecoder.Open(path, enableGpu: false);
            byte[] outRgb = new byte[8 * 8 * 3];
            Assert.Throws<NotSupportedException>(() => vae.Decode(new float[16], 1, 1, outRgb));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}






