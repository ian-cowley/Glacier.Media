namespace Glacier.Media.Audio;

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;

/// <summary>
/// Configuration options for <see cref="WhisperEngine"/> HTTP <c>/v1/audio/transcriptions</c> STT client routing.
/// </summary>
public sealed class WhisperEngineOptions
{
    public int EncoderLayers { get; init; } = 4;
    public int DecoderLayers { get; init; } = 4;
    public int SampleRate { get; init; } = 16000;
    public string? Endpoint { get; init; }
    public string? ModelId { get; init; }
    public string? ApiKey { get; init; }
    public string? DefaultLanguage { get; init; }
    public bool AutoDiscoverNpu { get; init; } = true;
    public HttpClient? HttpClient { get; init; }
    public TimeSpan? Timeout { get; init; }
    public TimeSpan? ProbeTimeout { get; init; }
}

/// <summary>
/// Experimental Speech-to-Text (STT) HTTP client/proxy targeting OpenAI-compatible transcription endpoints
/// (<c>POST /v1/audio/transcriptions</c>) or auto-discovered local NPU sidecars.
/// Local in-process GGUF Whisper decoding is not implemented (<see cref="SupportsLocalDecoding"/> is <see langword="false"/>);
/// throws <see cref="NotSupportedException"/> when no remote/sidecar endpoint is reachable.
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed class WhisperEngine : IDisposable
{
    public const int DefaultSampleRate = 16000;
    public const int MelBins = 80;
    public const int EncoderDim = 512;
    public const int DecoderDim = 512;
    public const int EncoderHeads = 8;
    public const int DecoderHeads = 8;
    public const int HeadDim = EncoderDim / EncoderHeads; // 64

    // Special Whisper tokens
    public const int TokenStartOfTranscript = 50258;
    public const int TokenEnglish = 50259;
    public const int TokenTranscribe = 50359;
    public const int TokenNoTimestamps = 50363;
    public const int TokenEndOfTranscript = 50257;

    private const string UnsupportedFallbackMessage =
        "WhisperEngine is an HTTP /v1/audio/transcriptions client proxy with no local GGUF/neural decoding; configure an explicit endpoint or start a reachable local OpenAI-compatible STT/NPU sidecar.";

    private readonly int _encoderLayers;
    private readonly int _decoderLayers;
    private readonly int _defaultSampleRate;
    private readonly string? _defaultLanguage;
    private readonly bool _autoDiscoverNpu;
    private readonly string? _explicitEndpoint;
    private readonly string? _modelId;
    private readonly string? _apiKey;
    private readonly HttpClient? _probeHttpClient;
    private readonly TimeSpan? _timeout;
    private readonly TimeSpan? _probeTimeout;
    private readonly IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? _discoveryCandidates;
    private readonly SemaphoreSlim _discoveryLock = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();

    private OpenAiWhisperClient? _client;
    private bool _ownsClient;
    private bool _discoveryAttempted;
    private int _disposed;

    public int EncoderLayers => _encoderLayers;
    public int DecoderLayers => _decoderLayers;
    public bool HasRemoteClient => Volatile.Read(ref _client) != null;
    public bool SupportsLocalDecoding => false;
    public bool RequiresRemoteEndpoint => true;
    public string ActiveBackend => HasRemoteClient ? "RemoteOpenAiHttp" : "Unconfigured";

    public WhisperEngine(int encoderLayers = 4, int decoderLayers = 4)
        : this(new WhisperEngineOptions
        {
            EncoderLayers = encoderLayers,
            DecoderLayers = decoderLayers
        })
    {
    }

    public WhisperEngine(
        OpenAiWhisperClient client,
        bool ownsClient = false,
        int encoderLayers = 4,
        int decoderLayers = 4)
    {
        ArgumentNullException.ThrowIfNull(client);

        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _defaultSampleRate = DefaultSampleRate;
        _client = client;
        _ownsClient = ownsClient;
        _autoDiscoverNpu = false;
    }

    public WhisperEngine(
        string endpointUrl,
        string? modelId = null,
        string? apiKey = null,
        HttpClient? httpClient = null,
        int encoderLayers = 4,
        int decoderLayers = 4)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(endpointUrl);

        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _defaultSampleRate = DefaultSampleRate;

        if (OpenAiWhisperClient.IsAutoNpuEndpoint(endpointUrl))
        {
            _autoDiscoverNpu = true;
            _modelId = modelId;
            _apiKey = apiKey;
            _probeHttpClient = httpClient;
        }
        else
        {
            _client = new OpenAiWhisperClient(endpointUrl, modelId, apiKey, httpClient);
            _ownsClient = true;
            _autoDiscoverNpu = false;
        }
    }

    public WhisperEngine(WhisperEngineOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.SampleRate);

        _encoderLayers = options.EncoderLayers;
        _decoderLayers = options.DecoderLayers;
        _defaultSampleRate = options.SampleRate;
        _defaultLanguage = options.DefaultLanguage;

        bool isAutoNpu = OpenAiWhisperClient.IsAutoNpuEndpoint(options.Endpoint);
        if (!string.IsNullOrWhiteSpace(options.Endpoint) && !isAutoNpu)
        {
            _client = new OpenAiWhisperClient(
                baseUrl: options.Endpoint,
                modelId: options.ModelId,
                apiKey: options.ApiKey,
                httpClient: options.HttpClient,
                timeout: options.Timeout);
            _ownsClient = true;
            _autoDiscoverNpu = false;
        }
        else
        {
            _autoDiscoverNpu = isAutoNpu || options.AutoDiscoverNpu;
            _modelId = options.ModelId;
            _apiKey = options.ApiKey;
            _probeHttpClient = options.HttpClient;
            _timeout = options.Timeout;
            _probeTimeout = options.ProbeTimeout;
        }
    }

    internal WhisperEngine(
        HttpClient? probeHttpClient,
        IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? discoveryCandidates,
        string? endpoint = null,
        string? modelId = null,
        string? apiKey = null,
        bool autoDiscoverNpu = true,
        TimeSpan? probeTimeout = null,
        TimeSpan? timeout = null,
        int encoderLayers = 4,
        int decoderLayers = 4)
    {
        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _defaultSampleRate = DefaultSampleRate;
        _probeHttpClient = probeHttpClient;
        _discoveryCandidates = discoveryCandidates;
        _explicitEndpoint = endpoint;
        _modelId = modelId;
        _apiKey = apiKey;
        _autoDiscoverNpu = autoDiscoverNpu;
        _probeTimeout = probeTimeout;
        _timeout = timeout;
    }

    /// <summary>
    /// Transcribes 16kHz mono audio samples into text via a configured or auto-discovered HTTP STT endpoint.
    /// Throws <see cref="NotSupportedException"/> if no remote endpoint or sidecar is reachable.
    /// </summary>
    public string Transcribe(ReadOnlySpan<float> audio)
    {
        return Transcribe(audio, _defaultSampleRate, language: null);
    }

    /// <summary>
    /// Transcribes mono audio samples at the specified sample rate into text via a configured or auto-discovered HTTP STT endpoint.
    /// Throws <see cref="NotSupportedException"/> if no remote endpoint or sidecar is reachable.
    /// </summary>
    public string Transcribe(ReadOnlySpan<float> audio, int sampleRate, string? language = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);

        float[] audioCopy = audio.IsEmpty ? Array.Empty<float>() : audio.ToArray();
        if (SynchronizationContext.Current is not null)
        {
            return Task.Run(() => TranscribeAsync(audioCopy, sampleRate, language, CancellationToken.None))
                .GetAwaiter()
                .GetResult();
        }

        return TranscribeAsync(audioCopy, sampleRate, language, CancellationToken.None)
            .GetAwaiter()
            .GetResult();
    }

    /// <summary>
    /// Asynchronously transcribes 16kHz mono audio samples into text via a configured or auto-discovered local STT sidecar.
    /// </summary>
    public Task<string> TranscribeAsync(
        ReadOnlyMemory<float> audio,
        CancellationToken cancellationToken = default)
    {
        return TranscribeAsync(audio, _defaultSampleRate, language: null, cancellationToken);
    }

    /// <summary>
    /// Asynchronously transcribes mono audio samples at the specified sample rate into text via a configured or auto-discovered local STT sidecar.
    /// </summary>
    public async Task<string> TranscribeAsync(
        ReadOnlyMemory<float> audio,
        int sampleRate,
        string? language = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);

        OpenAiWhisperClient? client = await ResolveClientAsync(cancellationToken).ConfigureAwait(false);
        if (client == null)
            throw new NotSupportedException(UnsupportedFallbackMessage);

        return await client.TranscribeAsync(
            samples: audio,
            sampleRate: sampleRate,
            language: language ?? _defaultLanguage,
            responseFormat: null,
            cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    private async Task<OpenAiWhisperClient?> ResolveClientAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var existing = Volatile.Read(ref _client);
        if (existing != null)
            return existing;

        if (!_autoDiscoverNpu || _discoveryAttempted)
            return null;

        try
        {
            await _discoveryLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(WhisperEngine));
        }

        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            if (_client != null)
                return _client;
            if (_discoveryAttempted)
                return null;

            CancellationTokenSource linkedCts;
            try
            {
                linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _disposeCts.Token);
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(WhisperEngine));
            }

            OpenAiWhisperClient? discovered;
            using (linkedCts)
            {
                try
                {
                    discovered = await OpenAiWhisperClient.DiscoverAsync(
                        httpClient: _probeHttpClient,
                        candidates: _discoveryCandidates,
                        explicitEndpoint: _explicitEndpoint,
                        modelId: _modelId,
                        apiKey: _apiKey,
                        probeTimeout: _probeTimeout,
                        requestTimeout: _timeout,
                        cancellationToken: linkedCts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (Volatile.Read(ref _disposed) != 0 && !cancellationToken.IsCancellationRequested)
                {
                    throw new ObjectDisposedException(nameof(WhisperEngine));
                }
            }

            if (Volatile.Read(ref _disposed) != 0)
            {
                discovered?.Dispose();
                throw new ObjectDisposedException(nameof(WhisperEngine));
            }

            _discoveryAttempted = true;
            if (discovered != null)
            {
                _ownsClient = true;
                Volatile.Write(ref _client, discovered);
            }

            return discovered;
        }
        finally
        {
            try
            {
                _discoveryLock.Release();
            }
            catch (ObjectDisposedException)
            {
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        try { _disposeCts.Cancel(); } catch (ObjectDisposedException) { }

        if (_ownsClient)
            Interlocked.Exchange(ref _client, null)?.Dispose();

        if (_discoveryLock.Wait(0))
        {
            _discoveryLock.Release();
            _discoveryLock.Dispose();
        }

        _disposeCts.Dispose();
    }
}
