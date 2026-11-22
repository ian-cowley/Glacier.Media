namespace Glacier.Media.Audio;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;

/// <summary>
/// NativeAOT-compatible OpenAI Whisper HTTP client targeting <c>POST /v1/audio/transcriptions</c>
/// on local NPU sidecars (FastFlowLM <c>flm serve</c>, AMD Lemonade Server) or custom OpenAI-compatible STT servers.
/// </summary>
public sealed class OpenAiWhisperClient : IDisposable
{
    public const int DefaultSampleRate = 16000;
    public const string DefaultModelId = "whisper-1";

    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly string? _apiKey;
    private bool _disposed;

    public string BaseUrl { get; }
    public string TranscriptionsUrl { get; }
    public string ModelId { get; }

    public OpenAiWhisperClient(
        string baseUrl,
        string? modelId = null,
        string? apiKey = null,
        HttpClient? httpClient = null,
        TimeSpan? timeout = null)
    {
        BaseUrl = NormalizeBaseUrl(baseUrl);
        int suffixIdx = BaseUrl.IndexOfAny(['?', '#']);
        TranscriptionsUrl = suffixIdx >= 0
            ? $"{BaseUrl[..suffixIdx]}/v1/audio/transcriptions{BaseUrl[suffixIdx..]}"
            : $"{BaseUrl}/v1/audio/transcriptions";

        if (modelId is not null && string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Whisper model ID cannot be empty or whitespace.", nameof(modelId));

        ModelId = modelId is null ? DefaultModelId : modelId.Trim();
        _apiKey = string.IsNullOrWhiteSpace(apiKey) ? null : apiKey.Trim();

        if (httpClient != null)
        {
            _http = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _http = new HttpClient
            {
                Timeout = timeout ?? TimeSpan.FromSeconds(60)
            };
            _ownsHttpClient = true;
        }
    }

    public static bool IsAutoNpuEndpoint(string? endpoint)
    {
        if (string.IsNullOrWhiteSpace(endpoint))
            return false;

        string trimmed = endpoint.Trim();
        return trimmed.Equals("auto-npu", StringComparison.OrdinalIgnoreCase)
            || trimmed.Equals("npu", StringComparison.OrdinalIgnoreCase);
    }

    public static string NormalizeBaseUrl(string baseUrl)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseUrl);

        string trimmed = baseUrl.Trim();
        int suffixIdx = trimmed.IndexOfAny(['?', '#']);
        string basePart = (suffixIdx >= 0 ? trimmed[..suffixIdx] : trimmed).TrimEnd('/');
        string suffix = suffixIdx >= 0 ? trimmed[suffixIdx..] : string.Empty;

        if (basePart.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
            basePart = basePart[..^3].TrimEnd('/');

        string normalized = basePart + suffix;
        string transcriptionsProbe = $"{basePart}/v1/audio/transcriptions{suffix}";
        if (string.IsNullOrWhiteSpace(basePart)
            || !Uri.TryCreate(normalized, UriKind.Absolute, out var uri)
            || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
            || !Uri.TryCreate(transcriptionsProbe, UriKind.Absolute, out _))
        {
            throw new ArgumentException($"Invalid Whisper endpoint base URL: '{baseUrl}'.", nameof(baseUrl));
        }

        return normalized;
    }

    public static Task<OpenAiWhisperClient?> DiscoverAsync(
        HttpClient? httpClient = null,
        string? explicitEndpoint = null,
        string? modelId = null,
        string? apiKey = null,
        TimeSpan? probeTimeout = null,
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {
        return DiscoverAsync(
            httpClient: httpClient,
            candidates: null,
            explicitEndpoint: explicitEndpoint,
            modelId: modelId,
            apiKey: apiKey,
            probeTimeout: probeTimeout,
            requestTimeout: requestTimeout,
            cancellationToken: cancellationToken);
    }

    internal static async Task<OpenAiWhisperClient?> DiscoverAsync(
        HttpClient? httpClient,
        IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? candidates,
        string? explicitEndpoint = null,
        string? modelId = null,
        string? apiKey = null,
        TimeSpan? probeTimeout = null,
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {
        if (modelId is not null && string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Whisper model ID cannot be empty or whitespace.", nameof(modelId));

        string? resolvedExplicit = IsAutoNpuEndpoint(explicitEndpoint) ? null : explicitEndpoint;
        var targetCandidates = candidates ?? NpuSidecarDiscovery.GetDefaultCandidateEndpoints(resolvedExplicit);

        NpuSidecarEndpoint? endpoint = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: httpClient,
            candidates: targetCandidates,
            perEndpointTimeout: probeTimeout,
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (endpoint == null)
            return null;

        string resolvedModel = ResolveModelId(endpoint, modelId);
        return new OpenAiWhisperClient(
            baseUrl: endpoint.BaseUrl,
            modelId: resolvedModel,
            apiKey: apiKey,
            httpClient: httpClient,
            timeout: requestTimeout);
    }

    public Task<string> TranscribeAsync(
        ReadOnlyMemory<float> samples,
        int sampleRate = DefaultSampleRate,
        string? language = null,
        string? responseFormat = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);

        if (samples.IsEmpty)
            return Task.FromResult(string.Empty);

        byte[] wavBytes = WavWriter.ToPcm16Bytes(samples.Span, sampleRate, channels: 1);
        return TranscribeWavAsync(wavBytes, language, responseFormat, "audio.wav", cancellationToken);
    }

    public async Task<string> TranscribeWavAsync(
        ReadOnlyMemory<byte> wavBytes,
        string? language = null,
        string? responseFormat = null,
        string fileName = "audio.wav",
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (wavBytes.IsEmpty)
            throw new ArgumentException("WAV audio payload cannot be empty.", nameof(wavBytes));

        string safeFileName = SanitizeMultipartFileName(fileName);

        ByteArrayContent fileContent;
        if (MemoryMarshal.TryGetArray(wavBytes, out ArraySegment<byte> segment) && segment.Array != null)
        {
            fileContent = new ByteArrayContent(segment.Array, segment.Offset, segment.Count);
        }
        else
        {
            fileContent = new ByteArrayContent(wavBytes.ToArray());
        }

        fileContent.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");

        using var form = new MultipartFormDataContent();
        form.Add(fileContent, "file", safeFileName);
        form.Add(new StringContent(ModelId), "model");

        if (!string.IsNullOrWhiteSpace(language))
            form.Add(new StringContent(language.Trim()), "language");

        string? normalizedFormat = string.IsNullOrWhiteSpace(responseFormat) ? null : responseFormat.Trim();
        if (normalizedFormat != null)
            form.Add(new StringContent(normalizedFormat), "response_format");

        using var request = new HttpRequestMessage(HttpMethod.Post, TranscriptionsUrl)
        {
            Content = form
        };

        if (_apiKey != null)
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            string? detail = TryExtractErrorMessage(body);
            string message = detail != null
                ? $"Whisper transcription endpoint '{TranscriptionsUrl}' returned HTTP {(int)response.StatusCode} ({detail}): {body}"
                : $"Whisper transcription endpoint '{TranscriptionsUrl}' returned HTTP {(int)response.StatusCode}: {body}";
            throw new HttpRequestException(message, inner: null, statusCode: response.StatusCode);
        }

        if (normalizedFormat != null &&
            (normalizedFormat.Equals("text", StringComparison.OrdinalIgnoreCase)
             || normalizedFormat.Equals("srt", StringComparison.OrdinalIgnoreCase)
             || normalizedFormat.Equals("vtt", StringComparison.OrdinalIgnoreCase)))
        {
            return body.Trim();
        }

        if (string.IsNullOrWhiteSpace(body))
        {
            throw new InvalidOperationException(
                $"Whisper transcription endpoint '{TranscriptionsUrl}' returned an empty JSON response.");
        }

        WhisperTranscriptionResponse? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize(body, MediaJsonContext.Default.WhisperTranscriptionResponse);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                $"Whisper transcription endpoint '{TranscriptionsUrl}' returned malformed JSON.", ex);
        }

        if (parsed?.Text is null)
        {
            throw new InvalidOperationException(
                $"Whisper transcription endpoint '{TranscriptionsUrl}' returned JSON without a 'text' property.");
        }

        return parsed.Text;
    }

    private static string ResolveModelId(NpuSidecarEndpoint endpoint, string? explicitModelId)
    {
        if (!string.IsNullOrWhiteSpace(explicitModelId))
            return explicitModelId.Trim();

        for (int i = 0; i < endpoint.Models.Count; i++)
        {
            string candidate = endpoint.Models[i];
            if (candidate.Contains("whisper", StringComparison.OrdinalIgnoreCase))
                return candidate;
        }

        return !string.IsNullOrWhiteSpace(endpoint.PrimaryModel)
            ? endpoint.PrimaryModel
            : DefaultModelId;
    }

    private static string? TryExtractErrorMessage(string body)
    {
        if (string.IsNullOrWhiteSpace(body) || !body.TrimStart().StartsWith('{'))
            return null;

        try
        {
            var err = JsonSerializer.Deserialize(body, MediaJsonContext.Default.WhisperErrorResponse);
            return string.IsNullOrWhiteSpace(err?.Error?.Message) ? null : err.Error.Message;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string SanitizeMultipartFileName(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return "audio.wav";

        Span<char> buffer = stackalloc char[Math.Min(fileName.Length, 256)];
        int written = 0;
        foreach (char c in fileName.Trim())
        {
            if (char.IsControl(c) || c == '"' || c == ';' || c == '/' || c == '\\')
                continue;
            if (written == buffer.Length)
                break;
            buffer[written++] = c;
        }

        string sanitized = written == 0 ? string.Empty : new string(buffer[..written]).Trim();
        return string.IsNullOrWhiteSpace(sanitized) ? "audio.wav" : sanitized;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_ownsHttpClient)
                _http.Dispose();
        }
    }
}
