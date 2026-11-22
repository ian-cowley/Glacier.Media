namespace Glacier.Media.Audio;

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.Media.Audio.Kitten;

/// <summary>
/// Experimental conversational voice pipeline linking local NPU/Whisper STT (<see cref="WhisperEngine"/>)
/// with procedural formant TTS (<see cref="KokoroTtsEngine"/>) and optional neural TTS (<see cref="KittenTtsEngine"/>).
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed class VoicePipeline : IDisposable
{
#pragma warning disable GLACIER_EXP001
    private readonly WhisperEngine _stt;
    private readonly KokoroTtsEngine _tts;
    private readonly KittenTtsEngine? _kittenTts;
    private readonly bool _ownsStt;
    private readonly bool _ownsTts;
    private readonly bool _ownsKittenTts;
    private bool _disposed;

    public WhisperEngine Stt => _stt;
    public KokoroTtsEngine Tts => _tts;
    public KittenTtsEngine? NeuralTts => _kittenTts;

    public VoicePipeline()
    {
        _stt = new WhisperEngine();
        _tts = new KokoroTtsEngine();
        _ownsStt = true;
        _ownsTts = true;
    }

    public VoicePipeline(
        OpenAiWhisperClient whisperClient,
        KokoroTtsEngine? tts = null,
        KittenTtsEngine? kittenTts = null,
        bool ownsWhisperClient = false)
    {
        ArgumentNullException.ThrowIfNull(whisperClient);

        _stt = new WhisperEngine(whisperClient, ownsClient: ownsWhisperClient);
        _tts = tts ?? new KokoroTtsEngine();
        _kittenTts = kittenTts;
        _ownsStt = true;
        _ownsTts = tts == null;
        _ownsKittenTts = false;
    }

    public VoicePipeline(
        WhisperEngine stt,
        KokoroTtsEngine? tts = null,
        KittenTtsEngine? kittenTts = null,
        bool ownsEngines = false)
    {
        ArgumentNullException.ThrowIfNull(stt);

        _stt = stt;
        _tts = tts ?? new KokoroTtsEngine();
        _kittenTts = kittenTts;
        _ownsStt = ownsEngines;
        _ownsTts = tts == null || ownsEngines;
        _ownsKittenTts = kittenTts != null && ownsEngines;
    }

    public VoicePipeline(
        string sttEndpointUrl,
        string? sttModelId = null,
        string? apiKey = null,
        HttpClient? httpClient = null,
        KokoroTtsEngine? tts = null,
        KittenTtsEngine? kittenTts = null)
    {
        _stt = new WhisperEngine(sttEndpointUrl, sttModelId, apiKey, httpClient);
        _tts = tts ?? new KokoroTtsEngine();
        _kittenTts = kittenTts;
        _ownsStt = true;
        _ownsTts = tts == null;
        _ownsKittenTts = false;
    }

    public static Task<VoicePipeline> CreateAutoAsync(
        string? sttModelId = null,
        bool tryLoadKittenTts = false,
        CancellationToken cancellationToken = default)
    {
        return CreateAutoAsync(
            httpClient: null,
            candidates: null,
            sttModelId: sttModelId,
            tryLoadKittenTts: tryLoadKittenTts,
            cancellationToken: cancellationToken);
    }

    internal static async Task<VoicePipeline> CreateAutoAsync(
        HttpClient? httpClient,
        IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? candidates,
        string? sttModelId = null,
        bool tryLoadKittenTts = false,
        CancellationToken cancellationToken = default)
    {
        OpenAiWhisperClient? discovered = await OpenAiWhisperClient.DiscoverAsync(
            httpClient: httpClient,
            candidates: candidates,
            explicitEndpoint: null,
            modelId: sttModelId,
            cancellationToken: cancellationToken).ConfigureAwait(false);

        WhisperEngine stt = discovered != null
            ? new WhisperEngine(discovered, ownsClient: true)
            : new WhisperEngine(new WhisperEngineOptions { AutoDiscoverNpu = false });

        KittenTtsEngine? kitten = null;
        try
        {
            if (tryLoadKittenTts)
            {
                try
                {
                    kitten = KittenTtsEngine.CreateDefault();
                }
                catch (FileNotFoundException)
                {
                    kitten = null;
                }
            }

            return new VoicePipeline(stt, tts: null, kittenTts: kitten, ownsEngines: true);
        }
        catch
        {
            kitten?.Dispose();
            stt.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Transcribes 16kHz audio samples into text.
    /// </summary>
    public string Listen(ReadOnlySpan<float> audio16kHz)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stt.Transcribe(audio16kHz);
    }

    /// <summary>
    /// Transcribes audio samples at the given sample rate into text.
    /// </summary>
    public string Listen(ReadOnlySpan<float> audio, int sampleRate, string? language = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stt.Transcribe(audio, sampleRate, language);
    }

    /// <summary>
    /// Asynchronously transcribes 16kHz audio samples into text.
    /// </summary>
    public Task<string> ListenAsync(
        ReadOnlyMemory<float> audio16kHz,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stt.TranscribeAsync(audio16kHz, cancellationToken);
    }

    /// <summary>
    /// Asynchronously transcribes audio samples at the given sample rate into text.
    /// </summary>
    public Task<string> ListenAsync(
        ReadOnlyMemory<float> audio,
        int sampleRate,
        string? language = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _stt.TranscribeAsync(audio, sampleRate, language, cancellationToken);
    }
#pragma warning restore GLACIER_EXP001

    /// <summary>
    /// Synthesizes text into 24kHz audio samples via procedural formant synthesis.
    /// </summary>
    public float[] Speak(string text, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _tts.Synthesize(text, voice, speed);
    }

    /// <summary>
    /// Synthesizes text using <see cref="KittenTtsEngine"/> if loaded, falling back cleanly to <see cref="KokoroTtsEngine"/>.
    /// </summary>
    public float[] SpeakNeural(string text, string voice = "bella", float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_kittenTts != null)
            return _kittenTts.Synthesize(text, voice, speed);

        return _tts.Synthesize(text, MapKittenVoiceToKokoro(voice), speed);
    }

    /// <summary>
    /// Synthesizes text directly to a WAV file.
    /// </summary>
    public void SpeakToFile(string text, string outputPath, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _tts.SynthesizeToFile(text, outputPath, voice, speed);
    }

    /// <summary>
    /// Executes a conversational turn: ingests audio, transcribes to text, applies a response delegate,
    /// and synthesizes spoken response audio.
    /// </summary>
    public (string TranscribedText, string ResponseText, float[] ResponseAudio) ConversationalTurn(
        ReadOnlySpan<float> userAudio16kHz,
        Func<string, string> responseGenerator,
        KokoroVoice voice = KokoroVoice.AfHeart)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(responseGenerator);

        string inputPrompt = Listen(userAudio16kHz);
        string response = responseGenerator(inputPrompt);
        float[] audioOut = Speak(response, voice);

        return (inputPrompt, response, audioOut);
    }

    /// <summary>
    /// Asynchronously executes a conversational turn using a synchronous response generator.
    /// </summary>
    public async Task<(string TranscribedText, string ResponseText, float[] ResponseAudio)> ConversationalTurnAsync(
        ReadOnlyMemory<float> userAudio16kHz,
        Func<string, string> responseGenerator,
        KokoroVoice voice = KokoroVoice.AfHeart,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(responseGenerator);

        string inputPrompt = await ListenAsync(userAudio16kHz, cancellationToken).ConfigureAwait(false);
        string response = responseGenerator(inputPrompt);
        float[] audioOut = Speak(response, voice);

        return (inputPrompt, response, audioOut);
    }

    /// <summary>
    /// Asynchronously executes a conversational turn using an asynchronous response generator.
    /// </summary>
    public async Task<(string TranscribedText, string ResponseText, float[] ResponseAudio)> ConversationalTurnAsync(
        ReadOnlyMemory<float> userAudio16kHz,
        Func<string, CancellationToken, Task<string>> responseGenerator,
        KokoroVoice voice = KokoroVoice.AfHeart,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(responseGenerator);

        string inputPrompt = await ListenAsync(userAudio16kHz, cancellationToken).ConfigureAwait(false);
        string response = await responseGenerator(inputPrompt, cancellationToken).ConfigureAwait(false);
        float[] audioOut = Speak(response, voice);

        return (inputPrompt, response, audioOut);
    }

    private static KokoroVoice MapKittenVoiceToKokoro(string? voice)
    {
        if (string.IsNullOrWhiteSpace(voice))
            return KokoroVoice.AfHeart;

        return voice.Trim().ToLowerInvariant() switch
        {
            "bella" => KokoroVoice.AfBella,
            "luna" => KokoroVoice.AfSky,
            "kiki" => KokoroVoice.BfEmma,
            "rosie" => KokoroVoice.BfAlice,
            "bruno" => KokoroVoice.AmAdam,
            "jasper" => KokoroVoice.AmMichael,
            "leo" => KokoroVoice.AmEric,
            "hugo" => KokoroVoice.BmGeorge,
            _ => KokoroVoice.AfHeart
        };
    }

#pragma warning disable GLACIER_EXP001
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_ownsStt)
                _stt.Dispose();
            if (_ownsTts)
                _tts.Dispose();
            if (_ownsKittenTts)
                _kittenTts?.Dispose();
        }
    }
#pragma warning restore GLACIER_EXP001
}
