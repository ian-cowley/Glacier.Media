namespace Glacier.Media.Tests;

#pragma warning disable GLACIER_EXP001

using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.Media.Audio;
using Xunit;

[Collection("NpuEnv")]
public sealed class Tier5M7Phase2HardeningTests
{
    private sealed class StubWhisperHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _responder;

        public StubWhisperHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => _responder(request, cancellationToken);
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    [Fact]
    public async Task OpenAiWhisperClient_FileNameSanitization_LargeWav_ErrorStatus_MalformedJson_Cancellation_AndDoubleDispose()
    {
        string? capturedDisposition = null;
        int capturedByteLength = 0;

        using var handler = new StubWhisperHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (path.EndsWith("/audio/transcriptions", StringComparison.OrdinalIgnoreCase))
            {
                if (req.Content != null)
                {
                    var raw = await req.Content.ReadAsStringAsync(ct);
                    capturedDisposition = raw;
                    capturedByteLength = raw.Length;
                }
                return Json(HttpStatusCode.OK, """{"text":"sanitized-ok"}""");
            }
            return Json(HttpStatusCode.NotFound, "{}");
        });

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625/v1", "whisper-v3-turbo", httpClient: http);

        // 1. Adversarial fileName injection (CR/LF, quotes, path traversal) + >1 MB PCM16 payload (600_000 float samples = 1.2 MB WAV)
        float[] largeSamples = new float[600_000];
        for (int i = 0; i < largeSamples.Length; i += 256)
            largeSamples[i] = 0.25f;

        byte[] wavPayload = WavWriter.ToPcm16Bytes(largeSamples, sampleRate: 16000, channels: 1);
        string transcript = await client.TranscribeWavAsync(
            wavPayload,
            fileName: "../evil\"\r\nX-Injected: bad;.wav");

        Assert.Equal("sanitized-ok", transcript);
        Assert.NotNull(capturedDisposition);
        Assert.True(capturedByteLength > 1_000_000);
        Assert.Contains("filename=\"..evilX-Injected: bad.wav\"", capturedDisposition!, StringComparison.Ordinal);
        Assert.DoesNotContain("\"\r", capturedDisposition!, StringComparison.Ordinal);
        Assert.DoesNotContain("\"\n", capturedDisposition!, StringComparison.Ordinal);

        // 2. Cancellation mid-request
        using var slowHandler = new StubWhisperHandler(async (_, ct) =>
        {
            await Task.Delay(TimeSpan.FromSeconds(10), ct);
            return Json(HttpStatusCode.OK, """{"text":"late"}""");
        });
        using var slowHttp = new HttpClient(slowHandler);
        using var slowClient = new OpenAiWhisperClient("http://127.0.0.1:52625/v1", "whisper", httpClient: slowHttp);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(40));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            slowClient.TranscribeAsync(new float[1600], 16000, cancellationToken: cts.Token));

        // 3. HTTP 401/429/503 + malformed JSON + missing "text"
        foreach (var status in new[] { HttpStatusCode.Unauthorized, HttpStatusCode.TooManyRequests, HttpStatusCode.ServiceUnavailable })
        {
            using var errH = new StubWhisperHandler((_, _) => Task.FromResult(Json(status, """{"error":{"message":"fail"}}""")));
            using var errHttp = new HttpClient(errH);
            using var errCli = new OpenAiWhisperClient("http://127.0.0.1:52625/v1", "whisper", httpClient: errHttp);
            await Assert.ThrowsAsync<HttpRequestException>(() => errCli.TranscribeAsync(new float[1600], 16000));
        }

        foreach (string badBody in new[] { "{malformed", "{}", """{"language":"en"}""" })
        {
            using var badH = new StubWhisperHandler((_, _) => Task.FromResult(Json(HttpStatusCode.OK, badBody)));
            using var badHttp = new HttpClient(badH);
            using var badCli = new OpenAiWhisperClient("http://127.0.0.1:52625/v1", "whisper", httpClient: badHttp);
            await Assert.ThrowsAnyAsync<Exception>(() => badCli.TranscribeAsync(new float[1600], 16000));
        }

        // 4. Double Dispose safe + post-Dispose throws ObjectDisposedException
        client.Dispose();
        client.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.TranscribeAsync(new float[160], 16000));
    }

    [Fact]
    public void WavWriter_ExtremeSamplesAndBoundaries_ClipsSafelyWithoutThrow()
    {
        string tempWav = Path.Combine(Path.GetTempPath(), $"tier5_wav_{Guid.NewGuid():N}.wav");
        try
        {
            float[] weird = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, -10.0f, +10.0f, 0.5f];
            WavWriter.WritePcm16(tempWav, weird, sampleRate: 16000, channels: 1);
            Assert.True(File.Exists(tempWav));
            Assert.True(new FileInfo(tempWav).Length >= 44 + weird.Length * 2);

            float[] readBack = WavWriter.ReadMonoFloat(File.ReadAllBytes(tempWav), out int rate);
            Assert.Equal(16000, rate);
            Assert.Equal(weird.Length, readBack.Length);

            // Invalid sampleRate / channels must throw ArgumentOutOfRangeException
            Assert.Throws<ArgumentOutOfRangeException>(() => WavWriter.ToPcm16Bytes(new float[] { 0.1f }, sampleRate: 0, channels: 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => WavWriter.ToPcm16Bytes(new float[] { 0.1f }, sampleRate: 16000, channels: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => WavWriter.WritePcm16(tempWav, new float[] { 0.1f }, sampleRate: -1, channels: 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => WavWriter.WriteFloat32(tempWav, new float[] { 0.1f }, sampleRate: 16000, channels: -2));

            // Stereo (channels=2) must interleave cleanly so all data bytes match dataChunkSize and ReadMonoFloat restores values
            float[] stereoSrc = [0.5f, -0.5f];
            byte[] stereoWav = WavWriter.ToPcm16Bytes(stereoSrc, sampleRate: 16000, channels: 2);
            Assert.Equal(44 + stereoSrc.Length * 2 * 2, stereoWav.Length);
            float[] stereoRoundTrip = WavWriter.ReadMonoFloat(stereoWav, out int stereoRate);
            Assert.Equal(16000, stereoRate);
            Assert.Equal(2, stereoRoundTrip.Length);
            Assert.InRange(stereoRoundTrip[0], 0.49f, 0.51f);
            Assert.InRange(stereoRoundTrip[1], -0.51f, -0.49f);
        }
        finally
        {
            if (File.Exists(tempWav)) File.Delete(tempWav);
        }
    }

    [Fact]
    public async Task WhisperEngine_AutoNpuEndpointOptionOverridesAutoDiscoverFalse_AndDisposeRaceIsSafe()
    {
        string? prevEnv = Environment.GetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar);
        NpuSidecarDiscovery.InvalidateCache();
        try
        {
            Environment.SetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar, "http://127.0.0.1:52625");
            using var handler = new StubWhisperHandler((req, _) =>
            {
                string path = req.RequestUri?.AbsolutePath ?? "";
                if (path.EndsWith("/models", StringComparison.OrdinalIgnoreCase))
                    return Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"id":"whisper-v3-turbo"}]}"""));
                if (path.EndsWith("/audio/transcriptions", StringComparison.OrdinalIgnoreCase))
                    return Task.FromResult(Json(HttpStatusCode.OK, """{"text":"auto-npu-honored"}"""));
                return Task.FromResult(Json(HttpStatusCode.NotFound, "{}"));
            });
            using var http = new HttpClient(handler);

            using var engine = new WhisperEngine(new WhisperEngineOptions
            {
                Endpoint = "auto-npu",
                AutoDiscoverNpu = false,
                HttpClient = http
            });

            string result = await engine.TranscribeAsync(new float[800], 16000);
            Assert.Equal("auto-npu-honored", result);

            // Concurrent TranscribeAsync + Dispose race stress
            using var slowHandler = new StubWhisperHandler(async (req, ct) =>
            {
                string path = req.RequestUri?.AbsolutePath ?? "";
                if (path.EndsWith("/models", StringComparison.OrdinalIgnoreCase))
                {
                    await Task.Delay(30, ct);
                    return Json(HttpStatusCode.OK, """{"data":[{"id":"whisper-v3-turbo"}]}""");
                }
                return Json(HttpStatusCode.OK, """{"text":"ok"}""");
            });
            using var slowHttp = new HttpClient(slowHandler);
            var raceEngine = new WhisperEngine(new WhisperEngineOptions
            {
                Endpoint = "auto-npu",
                HttpClient = slowHttp
            });

            var tasks = new Task[8];
            for (int i = 0; i < tasks.Length; i++)
            {
                tasks[i] = Task.Run(async () =>
                {
                    try
                    {
                        await raceEngine.TranscribeAsync(new float[160], 16000);
                    }
                    catch (Exception ex) when (ex is ObjectDisposedException or OperationCanceledException or NotSupportedException)
                    {
                        // Expected during concurrent disposal
                    }
                });
            }
            await Task.Delay(10);
            raceEngine.Dispose();
            await Task.WhenAll(tasks);

            // VoicePipeline.CreateAutoAsync returns cleanly even when default KittenTts model is missing
            using var pipeline = await VoicePipeline.CreateAutoAsync(
                httpClient: http,
                candidates: [("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM)],
                tryLoadKittenTts: true);
            Assert.Equal("auto-npu-honored", await pipeline.ListenAsync(new float[1600]));
        }
        finally
        {
            Environment.SetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar, prevEnv);
            NpuSidecarDiscovery.InvalidateCache();
        }
    }
}
