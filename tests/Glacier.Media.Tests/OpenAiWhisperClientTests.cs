namespace Glacier.Media.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.Media.Audio;
using Xunit;

[Collection("NpuEnv")]
public sealed class OpenAiWhisperClientTests
{
    [Fact]
    public async Task TranscribeAsync_Encodes16kHzMonoPcm16WavAndMultipartFields()
    {
        byte[]? capturedBody = null;
        string? capturedContentType = null;
        string? capturedAuth = null;
        Uri? capturedUri = null;

        using var handler = new MockWhisperHandler(async (req, ct) =>
        {
            capturedUri = req.RequestUri;
            capturedAuth = req.Headers.Authorization?.ToString();
            capturedContentType = req.Content?.Headers.ContentType?.ToString();
            capturedBody = req.Content != null ? await req.Content.ReadAsByteArrayAsync(ct) : null;
            return MockWhisperHandler.JsonResponse("""{"text":"Hello NPU whisper","language":"en","duration":0.1}""");
        });

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient(
            baseUrl: "http://127.0.0.1:52625/v1/",
            modelId: "whisper-v3-turbo",
            apiKey: null,
            httpClient: http);

        float[] input = CreateRampSamples(1600);
        string transcript = await client.TranscribeAsync(input, sampleRate: OpenAiWhisperClient.DefaultSampleRate);

        Assert.Equal("Hello NPU whisper", transcript);
        Assert.Equal("http://127.0.0.1:52625/v1/audio/transcriptions", capturedUri?.ToString());
        Assert.Null(capturedAuth);
        Assert.NotNull(capturedContentType);
        Assert.NotNull(capturedBody);

        var parts = ParseMultipartParts(capturedBody!, capturedContentType!);
        Assert.True(parts.TryGetValue("model", out var modelPart));
        Assert.Equal("whisper-v3-turbo", Encoding.UTF8.GetString(modelPart.Body));
        Assert.False(parts.ContainsKey("language"));
        Assert.False(parts.ContainsKey("response_format"));

        Assert.True(parts.TryGetValue("file", out var filePart));
        Assert.Equal("audio.wav", filePart.FileName);
        Assert.Equal("audio/wav", filePart.ContentType);

        // Verify RIFF / WAVE PCM-16 16kHz mono header
        byte[] wav = filePart.Body;
        Assert.Equal(44 + input.Length * 2, wav.Length);
        Assert.Equal("RIFF", Encoding.ASCII.GetString(wav, 0, 4));
        Assert.Equal("WAVE", Encoding.ASCII.GetString(wav, 8, 4));
        Assert.Equal("fmt ", Encoding.ASCII.GetString(wav, 12, 4));
        Assert.Equal(1, BitConverter.ToInt16(wav, 20)); // PCM
        Assert.Equal(1, BitConverter.ToInt16(wav, 22)); // mono
        Assert.Equal(16000, BitConverter.ToInt32(wav, 24)); // sample rate (NOT 24000)
        Assert.Equal(16, BitConverter.ToInt16(wav, 34)); // bits per sample
        Assert.Equal("data", Encoding.ASCII.GetString(wav, 36, 4));
        Assert.Equal(input.Length * 2, BitConverter.ToInt32(wav, 40));

        float[] roundTrip = WavWriter.ReadMonoFloat(wav, out int parsedSampleRate);
        Assert.Equal(16000, parsedSampleRate);
        Assert.Equal(input.Length, roundTrip.Length);
        for (int i = 0; i < input.Length; i++)
            Assert.InRange(Math.Abs(input[i] - roundTrip[i]), 0f, 1e-4f);
    }

    [Fact]
    public async Task TranscribeAsync_PassesExplicitSampleRateOptionalFieldsAndBearerToken()
    {
        Dictionary<string, MultipartPart>? capturedParts = null;
        string? authScheme = null;
        string? authParam = null;

        using var handler = new MockWhisperHandler(async (req, ct) =>
        {
            authScheme = req.Headers.Authorization?.Scheme;
            authParam = req.Headers.Authorization?.Parameter;
            byte[] raw = await req.Content!.ReadAsByteArrayAsync(ct);
            capturedParts = ParseMultipartParts(raw, req.Content.Headers.ContentType!.ToString());
            return MockWhisperHandler.JsonResponse("""{"text":"Verbose ok","language":"fr","duration":0.25,"segments":[{"id":0,"start":0.0,"end":0.25,"text":"Verbose ok"}]}""");
        });

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient(
            baseUrl: "http://127.0.0.1:8000",
            modelId: null,
            apiKey: "dummy-api-key",
            httpClient: http);

        Assert.Equal(OpenAiWhisperClient.DefaultModelId, client.ModelId);

        float[] samples = CreateRampSamples(800);
        string result = await client.TranscribeAsync(
            samples,
            sampleRate: 24000,
            language: "fr",
            responseFormat: "verbose_json");

        Assert.Equal("Verbose ok", result);
        Assert.Equal("Bearer", authScheme);
        Assert.Equal("dummy-api-key", authParam);
        Assert.NotNull(capturedParts);
        Assert.Equal("fr", Encoding.UTF8.GetString(capturedParts!["language"].Body));
        Assert.Equal("verbose_json", Encoding.UTF8.GetString(capturedParts["response_format"].Body));
        Assert.Equal(24000, BitConverter.ToInt32(capturedParts["file"].Body, 24));
    }

    [Theory]
    [InlineData("text", "  plain transcript output \n", "plain transcript output")]
    [InlineData("srt", "1\n00:00:00,000 --> 00:00:01,000\nhello\n", "1\n00:00:00,000 --> 00:00:01,000\nhello")]
    [InlineData("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello")]
    public async Task TranscribeAsync_PlainTextFormatsReturnTrimmedBodyWithoutJsonParsing(
        string format,
        string serverBody,
        string expected)
    {
        using var handler = new MockWhisperHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(serverBody, Encoding.UTF8, "text/plain")
            }));

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: http);

        string result = await client.TranscribeAsync(new float[160], responseFormat: format);
        Assert.Equal(expected, result);
    }

    [Fact]
    public async Task TranscribeAsync_EmptyVsSilentAudioBehavior()
    {
        int requestCount = 0;
        using var handler = new MockWhisperHandler((_, _) =>
        {
            requestCount++;
            return Task.FromResult(MockWhisperHandler.JsonResponse("""{"text":""}"""));
        });

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: http);

        string emptyResult = await client.TranscribeAsync(ReadOnlyMemory<float>.Empty);
        Assert.Equal(string.Empty, emptyResult);
        Assert.Equal(0, requestCount);

        string silentResult = await client.TranscribeAsync(new float[1600]);
        Assert.Equal(string.Empty, silentResult);
        Assert.Equal(1, requestCount);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.TranscribeAsync(new float[16], sampleRate: 0));
        await Assert.ThrowsAsync<ArgumentException>(() => client.TranscribeWavAsync(ReadOnlyMemory<byte>.Empty));
    }

    [Theory]
    [InlineData("http://127.0.0.1:52625", "http://127.0.0.1:52625")]
    [InlineData("http://127.0.0.1:52625/", "http://127.0.0.1:52625")]
    [InlineData("http://127.0.0.1:52625/v1", "http://127.0.0.1:52625")]
    [InlineData("http://127.0.0.1:52625/v1/", "http://127.0.0.1:52625")]
    [InlineData("  https://localhost:8443/V1/  ", "https://localhost:8443")]
    public void NormalizeBaseUrl_StripsTrailingSlashAndV1(string input, string expectedBase)
    {
        using var client = new OpenAiWhisperClient(input);
        Assert.Equal(expectedBase, client.BaseUrl);
        Assert.Equal($"{expectedBase}/v1/audio/transcriptions", client.TranscriptionsUrl);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not-a-valid-url")]
    [InlineData("ftp://127.0.0.1:52625")]
    public void Constructor_InvalidBaseUrlOrWhitespaceModelIdThrowsArgumentException(string badUrl)
    {
        Assert.Throws<ArgumentException>(() => new OpenAiWhisperClient(badUrl));
        Assert.Throws<ArgumentException>(() => new OpenAiWhisperClient("http://127.0.0.1:52625", modelId: "   "));
    }

    [Theory]
    [InlineData("auto-npu", true)]
    [InlineData("AUTO-NPU", true)]
    [InlineData(" npu ", true)]
    [InlineData("http://127.0.0.1:52625", false)]
    [InlineData(null, false)]
    [InlineData("", false)]
    public void IsAutoNpuEndpoint_MatchesExpectedKeywords(string? endpoint, bool expected)
    {
        Assert.Equal(expected, OpenAiWhisperClient.IsAutoNpuEndpoint(endpoint));
    }

    [Fact]
    public async Task TranscribeAsync_HttpErrorsAndInvalidJsonThrowDiagnostics()
    {
        using var errorHandler = new MockWhisperHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.InternalServerError)
            {
                Content = new StringContent("""{"error":{"message":"model not loaded"}}""", Encoding.UTF8, "application/json")
            }));
        using var errorHttp = new HttpClient(errorHandler);
        using var errorClient = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: errorHttp);

        var httpEx = await Assert.ThrowsAsync<HttpRequestException>(() => errorClient.TranscribeAsync(new float[160]));
        Assert.Contains("500", httpEx.Message);
        Assert.Contains("model not loaded", httpEx.Message);
        Assert.Contains("http://127.0.0.1:52625/v1/audio/transcriptions", httpEx.Message);

        foreach (string badBody in new[] { "", "not-json", "null", """{"language":"en"}""" })
        {
            using var badHandler = new MockWhisperHandler((_, _) =>
                Task.FromResult(MockWhisperHandler.JsonResponse(badBody)));
            using var badHttp = new HttpClient(badHandler);
            using var badClient = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: badHttp);

            await Assert.ThrowsAsync<InvalidOperationException>(() => badClient.TranscribeAsync(new float[160]));
        }
    }

    [Fact]
    public async Task Dispose_RespectsHttpClientOwnershipAndRejectsSubsequentCalls()
    {
        using var injectedHandler = new TrackingHandler();
        using var injectedHttp = new HttpClient(injectedHandler);

        var client = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: injectedHttp);
        client.Dispose();

        Assert.False(injectedHandler.Disposed);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.TranscribeAsync(new float[16]));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.TranscribeWavAsync(new byte[] { 1, 2, 3 }));

        // Verify injected HttpClient still works after OpenAiWhisperClient disposal
        using var another = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: injectedHttp);
        string text = await another.TranscribeAsync(new float[16]);
        Assert.Equal("ok", text);
    }

    [Fact]
    public async Task DiscoverAsync_SelectsWhisperModelWhenPresentAndHandlesAutoNpu()
    {
        var requestedUrls = new List<string>();
        using var handler = new MockWhisperHandler((req, _) =>
        {
            requestedUrls.Add(req.RequestUri!.ToString());
            if (req.RequestUri!.AbsolutePath == "/v1/models")
            {
                return Task.FromResult(MockWhisperHandler.JsonResponse(
                    """{"data":[{"id":"gemma3:4b"},{"id":"whisper-large-v3-turbo"}]}"""));
            }

            if (req.RequestUri.AbsolutePath == "/v1/audio/transcriptions")
            {
                return Task.FromResult(MockWhisperHandler.JsonResponse("""{"text":"discovered"}"""));
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        });

        using var http = new HttpClient(handler);
        var candidates = new[] { ("http://127.0.0.1:19999", NpuSidecarRuntime.FastFlowLM) };

        using OpenAiWhisperClient? discovered = await OpenAiWhisperClient.DiscoverAsync(
            httpClient: http,
            candidates: candidates,
            explicitEndpoint: "auto-npu");

        Assert.NotNull(discovered);
        Assert.Equal("http://127.0.0.1:19999", discovered!.BaseUrl);
        Assert.Equal("whisper-large-v3-turbo", discovered.ModelId);
        Assert.DoesNotContain(requestedUrls, u => u.Contains("auto-npu", StringComparison.OrdinalIgnoreCase));

        string transcript = await discovered.TranscribeAsync(new float[320]);
        Assert.Equal("discovered", transcript);
    }

    [Fact]
    public async Task DiscoverAsync_ReturnsNullWhenAllCandidatesFail()
    {
        using var handler = new MockWhisperHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound)));
        using var http = new HttpClient(handler);

        OpenAiWhisperClient? discovered = await OpenAiWhisperClient.DiscoverAsync(
            httpClient: http,
            candidates: new[] { ("http://127.0.0.1:19998", NpuSidecarRuntime.FastFlowLM) });

        Assert.Null(discovered);
    }

    [Fact]
    public async Task LoopbackTcpServer_EndToEndDiscoveryAndTranscriptionWireProtocol()
    {
        await using var server = new LoopbackWhisperServer("whisper-npu-v3", "wire hello");
        var candidates = new[] { (server.BaseUrl, NpuSidecarRuntime.FastFlowLM) };

        using OpenAiWhisperClient? client = await OpenAiWhisperClient.DiscoverAsync(
            httpClient: null,
            candidates: candidates,
            explicitEndpoint: null,
            modelId: null,
            apiKey: "fake-bearer-token");

        Assert.NotNull(client);
        Assert.Equal("whisper-npu-v3", client!.ModelId);

        float[] samples = CreateRampSamples(640);
        string transcript = await client.TranscribeAsync(samples, sampleRate: 16000, language: "en");

        Assert.Equal("wire hello", transcript);
        Assert.NotNull(server.LastAuthorization);
        Assert.Equal("Bearer fake-bearer-token", server.LastAuthorization);
        Assert.NotNull(server.LastMultipartBody);
        string wireText = Encoding.ASCII.GetString(server.LastMultipartBody!);
        Assert.True(
            wireText.Contains("name=\"model\"", StringComparison.Ordinal) || wireText.Contains("name=model", StringComparison.Ordinal),
            $"Expected model part in wire payload: {wireText}");
        Assert.True(
            wireText.Contains("filename=\"audio.wav\"", StringComparison.Ordinal) || wireText.Contains("filename=audio.wav", StringComparison.Ordinal),
            $"Expected audio.wav filename in wire payload: {wireText}");
    }

    private static float[] CreateRampSamples(int count)
    {
        var samples = new float[count];
        for (int i = 0; i < count; i++)
            samples[i] = MathF.Sin(2f * MathF.PI * 440f * i / 16000f) * 0.5f;
        return samples;
    }

    private sealed record MultipartPart(string Name, string? FileName, string? ContentType, byte[] Body);

    private static Dictionary<string, MultipartPart> ParseMultipartParts(byte[] body, string contentTypeHeader)
    {
        const string marker = "boundary=";
        int idx = contentTypeHeader.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        Assert.True(idx >= 0, $"Missing boundary in Content-Type: {contentTypeHeader}");
        string boundary = contentTypeHeader[(idx + marker.Length)..].Trim().Trim('"');

        byte[] delimiter = Encoding.ASCII.GetBytes($"--{boundary}");
        var result = new Dictionary<string, MultipartPart>(StringComparer.Ordinal);

        int pos = IndexOf(body, delimiter, 0);
        while (pos >= 0)
        {
            int afterDelim = pos + delimiter.Length;
            if (afterDelim + 1 < body.Length && body[afterDelim] == (byte)'-' && body[afterDelim + 1] == (byte)'-')
                break;

            if (afterDelim + 1 < body.Length && body[afterDelim] == (byte)'\r' && body[afterDelim + 1] == (byte)'\n')
                afterDelim += 2;

            byte[] headerSep = "\r\n\r\n"u8.ToArray();
            int headerEnd = IndexOf(body, headerSep, afterDelim);
            if (headerEnd < 0)
                break;

            string headersText = Encoding.ASCII.GetString(body, afterDelim, headerEnd - afterDelim);
            int partBodyStart = headerEnd + headerSep.Length;
            int nextDelim = IndexOf(body, delimiter, partBodyStart);
            if (nextDelim < 0)
                break;

            int partBodyEnd = nextDelim;
            if (partBodyEnd >= 2 && body[partBodyEnd - 2] == (byte)'\r' && body[partBodyEnd - 1] == (byte)'\n')
                partBodyEnd -= 2;

            byte[] partBytes = body[partBodyStart..partBodyEnd];

            string? name = null;
            string? fileName = null;
            string? partContentType = null;
            foreach (string line in headersText.Split("\r\n", StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.StartsWith("Content-Disposition:", StringComparison.OrdinalIgnoreCase))
                {
                    name = ExtractDispositionAttribute(line, "name");
                    fileName = ExtractDispositionAttribute(line, "filename");
                }
                else if (line.StartsWith("Content-Type:", StringComparison.OrdinalIgnoreCase))
                {
                    partContentType = line["Content-Type:".Length..].Trim();
                }
            }

            if (name != null)
                result[name] = new MultipartPart(name, fileName, partContentType, partBytes);

            pos = nextDelim;
        }

        return result;
    }

    private static string? ExtractDispositionAttribute(string header, string key)
    {
        string token = $"{key}=";
        int searchStart = 0;
        while (searchStart < header.Length)
        {
            int idx = header.IndexOf(token, searchStart, StringComparison.OrdinalIgnoreCase);
            if (idx < 0)
                return null;

            bool validBoundary = idx == 0 || header[idx - 1] is ' ' or ';' or '\t';
            if (!validBoundary)
            {
                searchStart = idx + 1;
                continue;
            }

            int start = idx + token.Length;
            if (start >= header.Length)
                return null;

            if (header[start] == '"')
            {
                start++;
                int end = header.IndexOf('"', start);
                return end >= start ? header[start..end] : null;
            }

            int sep = header.IndexOf(';', start);
            string raw = sep >= 0 ? header[start..sep] : header[start..];
            return raw.Trim();
        }

        return null;
    }

    private static int IndexOf(byte[] haystack, byte[] needle, int start)
    {
        for (int i = start; i <= haystack.Length - needle.Length; i++)
        {
            if (haystack.AsSpan(i, needle.Length).SequenceEqual(needle))
                return i;
        }
        return -1;
    }

    private sealed class MockWhisperHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _fn;

        public MockWhisperHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> fn)
        {
            _fn = fn;
        }

        public static HttpResponseMessage JsonResponse(string json) =>
            new(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => _fn(request, cancellationToken);
    }

    private sealed class TrackingHandler : HttpMessageHandler
    {
        public bool Disposed { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"text":"ok"}""", Encoding.UTF8, "application/json")
            });
        }

        protected override void Dispose(bool disposing)
        {
            Disposed = true;
            base.Dispose(disposing);
        }
    }
}

internal sealed class LoopbackWhisperServer : IAsyncDisposable
{
    private readonly TcpListener _listener;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _acceptLoop;
    private readonly string _modelId;
    private readonly string _transcript;

    public string BaseUrl { get; }
    public string? LastAuthorization { get; private set; }
    public byte[]? LastMultipartBody { get; private set; }

    public LoopbackWhisperServer(string modelId, string transcript)
    {
        _modelId = modelId;
        _transcript = transcript;
        _listener = new TcpListener(IPAddress.Loopback, 0);
        _listener.Start();
        int port = ((IPEndPoint)_listener.LocalEndpoint).Port;
        BaseUrl = $"http://127.0.0.1:{port}";
        _acceptLoop = Task.Run(AcceptLoopAsync);
    }

    private async Task AcceptLoopAsync()
    {
        while (!_cts.IsCancellationRequested)
        {
            TcpClient? client = null;
            try
            {
                client = await _listener.AcceptTcpClientAsync(_cts.Token);
                _ = Task.Run(() => HandleClientAsync(client));
            }
            catch (OperationCanceledException)
            {
                client?.Dispose();
                break;
            }
            catch (ObjectDisposedException)
            {
                client?.Dispose();
                break;
            }
            catch (SocketException)
            {
                client?.Dispose();
                break;
            }
        }
    }

    private async Task HandleClientAsync(TcpClient client)
    {
        using (client)
        await using (NetworkStream stream = client.GetStream())
        {
            var headerBytes = new MemoryStream();
            byte[] single = new byte[1];
            int match = 0;
            while (headerBytes.Length < 65536)
            {
                int read = await stream.ReadAsync(single, 0, 1, _cts.Token);
                if (read == 0)
                    return;
                headerBytes.WriteByte(single[0]);
                match = (match, single[0]) switch
                {
                    (0, (byte)'\r') => 1,
                    (1, (byte)'\n') => 2,
                    (2, (byte)'\r') => 3,
                    (3, (byte)'\n') => 4,
                    (_, (byte)'\r') => 1,
                    _ => 0
                };
                if (match == 4)
                    break;
            }

            string headerText = Encoding.ASCII.GetString(headerBytes.ToArray());
            string[] lines = headerText.Split("\r\n", StringSplitOptions.RemoveEmptyEntries);
            if (lines.Length == 0)
                return;

            string[] requestLine = lines[0].Split(' ');
            string method = requestLine[0];
            string path = requestLine.Length > 1 ? requestLine[1] : "/";

            int contentLength = 0;
            string? auth = null;
            foreach (string line in lines.Skip(1))
            {
                int colon = line.IndexOf(':');
                if (colon <= 0)
                    continue;
                string name = line[..colon].Trim();
                string value = line[(colon + 1)..].Trim();
                if (name.Equals("Content-Length", StringComparison.OrdinalIgnoreCase))
                    int.TryParse(value, out contentLength);
                else if (name.Equals("Authorization", StringComparison.OrdinalIgnoreCase))
                    auth = value;
            }

            byte[] body = new byte[contentLength];
            int totalRead = 0;
            while (totalRead < contentLength)
            {
                int n = await stream.ReadAsync(body, totalRead, contentLength - totalRead, _cts.Token);
                if (n == 0)
                    break;
                totalRead += n;
            }

            string responseJson;
            int statusCode;
            if (method == "GET" && path == "/v1/models")
            {
                statusCode = 200;
                responseJson = $$"""{"data":[{"id":"{{_modelId}}"}]}""";
            }
            else if (method == "POST" && path == "/v1/audio/transcriptions")
            {
                LastAuthorization = auth;
                LastMultipartBody = body;
                statusCode = 200;
                responseJson = $$"""{"text":"{{_transcript}}"}""";
            }
            else
            {
                statusCode = 404;
                responseJson = """{"error":{"message":"not found"}}""";
            }

            byte[] payload = Encoding.UTF8.GetBytes(responseJson);
            string responseHeader =
                $"HTTP/1.1 {statusCode} {(statusCode == 200 ? "OK" : "Not Found")}\r\n" +
                $"Content-Type: application/json\r\n" +
                $"Content-Length: {payload.Length}\r\n" +
                $"Connection: close\r\n\r\n";

            byte[] headerOut = Encoding.ASCII.GetBytes(responseHeader);
            await stream.WriteAsync(headerOut, _cts.Token);
            await stream.WriteAsync(payload, _cts.Token);
            await stream.FlushAsync(_cts.Token);
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
        _listener.Stop();
        try
        {
            await _acceptLoop;
        }
        catch
        {
        }
        _cts.Dispose();
    }
}
