namespace Glacier.Media.Tests;

using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.Media.Audio;
using Xunit;

#pragma warning disable GLACIER_EXP001

[CollectionDefinition("NpuEnv", DisableParallelization = true)]
public sealed class NpuEnvCollection
{
}

[Collection("NpuEnv")]
public sealed class WhisperVoicePipelineEnvTests
{
    [Fact]
    public async Task WhisperEngine_DefaultCtor_DiscoversSidecarViaEnvVarAndTranscribes()
    {
        await using var server = new LoopbackWhisperServer("whisper-v3-env", "env discovered speech");
        string? previousEnv = Environment.GetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar, server.BaseUrl);

            using var engine = new WhisperEngine();
            Assert.False(engine.HasRemoteClient);

            string transcript = await engine.TranscribeAsync(new float[320]);
            Assert.Equal("env discovered speech", transcript);
            Assert.True(engine.HasRemoteClient);
        }
        finally
        {
            Environment.SetEnvironmentVariable(NpuSidecarDiscovery.EndpointEnvVar, previousEnv);
        }
    }
}

[Collection("NpuEnv")]
public sealed class WhisperVoicePipelineTests
{
    [Fact]
    public async Task WhisperEngine_WithExplicitClient_SupportsSyncAndAsyncTranscription()
    {
        int calls = 0;
        using var handler = new StubHttpHandler((_, _) =>
        {
            calls++;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"text":"hello whisper engine"}""", Encoding.UTF8, "application/json")
            });
        });

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625", "whisper-1", httpClient: http);
        using var engine = new WhisperEngine(client, ownsClient: false);

        Assert.True(engine.HasRemoteClient);

        float[] audio = new float[1600];
        Assert.Equal("hello whisper engine", engine.Transcribe(audio));
        Assert.Equal("hello whisper engine", engine.Transcribe(audio, sampleRate: 24000, language: "en"));
        Assert.Equal("hello whisper engine", await engine.TranscribeAsync(audio));
        Assert.Equal(3, calls);
    }

    [Fact]
    public async Task WhisperEngine_SyncTranscribeUnderSingleThreadedSyncContextDoesNotDeadlock()
    {
        using var handler = new StubHttpHandler(async (_, ct) =>
        {
            await Task.Delay(10, ct);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"text":"no deadlock"}""", Encoding.UTF8, "application/json")
            };
        });

        using var http = new HttpClient(handler);
        using var engine = new WhisperEngine("http://127.0.0.1:52625", httpClient: http);

        var previous = SynchronizationContext.Current;
        var blockingContext = new RejectingSyncContext();
        try
        {
            SynchronizationContext.SetSynchronizationContext(blockingContext);
            string result = await Task.FromResult(engine.Transcribe(new float[320]));
            Assert.Equal("no deadlock", result);
        }
        finally
        {
            SynchronizationContext.SetSynchronizationContext(previous);
        }
    }

    [Fact]
    public async Task WhisperEngine_OfflineFallback_ThrowsNotSupportedEvenOnEmptyAudio()
    {
        int closedPort = GetClosedLoopbackPort();
        var unreachableCandidates = new[] { ($"http://127.0.0.1:{closedPort}", NpuSidecarRuntime.FastFlowLM) };

        using var engine = new WhisperEngine(
            probeHttpClient: null,
            discoveryCandidates: unreachableCandidates,
            autoDiscoverNpu: true,
            probeTimeout: TimeSpan.FromMilliseconds(150));

        var ex1 = Assert.Throws<NotSupportedException>(() => engine.Transcribe(new float[160]));
        Assert.Contains("WhisperEngine requires GGUF Whisper model weights", ex1.Message);

        var ex2 = Assert.Throws<NotSupportedException>(() => engine.Transcribe(ReadOnlySpan<float>.Empty));
        Assert.Contains("WhisperEngine requires GGUF Whisper model weights", ex2.Message);

        await Assert.ThrowsAsync<NotSupportedException>(() => engine.TranscribeAsync(new float[160]));
        Assert.False(engine.HasRemoteClient);

        using var disabledEngine = new WhisperEngine(new WhisperEngineOptions { AutoDiscoverNpu = false });
        Assert.Throws<NotSupportedException>(() => disabledEngine.Transcribe(new float[160]));
    }

    [Fact]
    public void WhisperEngine_Dispose_ThrowsObjectDisposedAndRespectsClientOwnership()
    {
        using var handler = new StubHttpHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"text":"ok"}""", Encoding.UTF8, "application/json")
            }));
        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: http);

        var engine = new WhisperEngine(client, ownsClient: false);
        engine.Dispose();

        Assert.Throws<ObjectDisposedException>(() => engine.Transcribe(new float[16]));
        Assert.Equal("ok", client.TranscribeAsync(new float[16]).GetAwaiter().GetResult());
    }

    [Fact]
    public async Task VoicePipeline_WithMockedWhisperClient_ExecutesListenSpeakAndConversationalTurns()
    {
        using var handler = new StubHttpHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"text":"what is the status"}""", Encoding.UTF8, "application/json")
            }));

        using var http = new HttpClient(handler);
        using var client = new OpenAiWhisperClient("http://127.0.0.1:52625", httpClient: http);
        using var pipeline = new VoicePipeline(client);

        Assert.Null(pipeline.NeuralTts);
        Assert.True(pipeline.Stt.HasRemoteClient);

        float[] input = new float[800];
        Assert.Equal("what is the status", pipeline.Listen(input));
        Assert.Equal("what is the status", await pipeline.ListenAsync(input));

        var syncTurn = pipeline.ConversationalTurn(input, text => $"Heard: {text}", KokoroVoice.AfHeart);
        Assert.Equal("what is the status", syncTurn.TranscribedText);
        Assert.Equal("Heard: what is the status", syncTurn.ResponseText);
        Assert.NotEmpty(syncTurn.ResponseAudio);

        var asyncTurnSyncDelegate = await pipeline.ConversationalTurnAsync(
            input,
            text => $"SyncDelegate: {text}");
        Assert.Equal("what is the status", asyncTurnSyncDelegate.TranscribedText);
        Assert.Equal("SyncDelegate: what is the status", asyncTurnSyncDelegate.ResponseText);
        Assert.NotEmpty(asyncTurnSyncDelegate.ResponseAudio);

        var asyncTurnAsyncDelegate = await pipeline.ConversationalTurnAsync(
            input,
            async (text, ct) =>
            {
                await Task.Yield();
                return $"AsyncDelegate: {text}";
            });
        Assert.Equal("what is the status", asyncTurnAsyncDelegate.TranscribedText);
        Assert.Equal("AsyncDelegate: what is the status", asyncTurnAsyncDelegate.ResponseText);
        Assert.NotEmpty(asyncTurnAsyncDelegate.ResponseAudio);

        // SpeakNeural cleanly falls back to KokoroTtsEngine when NeuralTts is null
        float[] neuralFallbackAudio = pipeline.SpeakNeural("Fallback speech", "bella");
        Assert.NotEmpty(neuralFallbackAudio);
    }

    [Fact]
    public async Task VoicePipeline_OfflineDefaultCtor_AllowsSpeakAndFailsListenCleanly()
    {
        // Default VoicePipeline() MUST NOT throw in constructor when offline
        using var pipeline = new VoicePipeline();

        float[] spoken = pipeline.Speak("Glacier local speech test", KokoroVoice.AfHeart);
        Assert.NotEmpty(spoken);

        string tempWav = Path.Combine(Path.GetTempPath(), $"glacier_tts_{Guid.NewGuid():N}.wav");
        try
        {
            pipeline.SpeakToFile("Testing output file", tempWav, KokoroVoice.BmGeorge);
            byte[] wavBytes = await File.ReadAllBytesAsync(tempWav);
            float[] parsed = WavWriter.ReadMonoFloat(wavBytes, out int sr);
            Assert.Equal(KokoroTtsEngine.DefaultSampleRate, sr);
            Assert.NotEmpty(parsed);
        }
        finally
        {
            if (File.Exists(tempWav))
                File.Delete(tempWav);
        }

        // Deterministic offline STT check via unreachable candidate
        int closedPort = GetClosedLoopbackPort();
        var offlineStt = new WhisperEngine(
            probeHttpClient: null,
            discoveryCandidates: new[] { ($"http://127.0.0.1:{closedPort}", NpuSidecarRuntime.FastFlowLM) },
            autoDiscoverNpu: true,
            probeTimeout: TimeSpan.FromMilliseconds(150));

        using var offlinePipeline = new VoicePipeline(offlineStt, ownsEngines: true);
        Assert.Throws<NotSupportedException>(() => offlinePipeline.Listen(new float[160]));
        await Assert.ThrowsAsync<NotSupportedException>(() => offlinePipeline.ListenAsync(new float[160]));
        Assert.Throws<NotSupportedException>(() => offlinePipeline.ConversationalTurn(new float[160], s => s));
        await Assert.ThrowsAsync<NotSupportedException>(() => offlinePipeline.ConversationalTurnAsync(new float[160], s => s));
    }

    [Fact]
    public async Task VoicePipeline_CreateAutoAsync_DiscoversOrFallsBackCleanly()
    {
        await using var server = new LoopbackWhisperServer("whisper-auto", "auto pipeline ok");
        using var onlinePipeline = await VoicePipeline.CreateAutoAsync(
            httpClient: null,
            candidates: new[] { (server.BaseUrl, NpuSidecarRuntime.FastFlowLM) },
            tryLoadKittenTts: false);

        Assert.True(onlinePipeline.Stt.HasRemoteClient);
        Assert.Equal("auto pipeline ok", await onlinePipeline.ListenAsync(new float[320]));

        int closedPort = GetClosedLoopbackPort();
        using var offlinePipeline = await VoicePipeline.CreateAutoAsync(
            httpClient: null,
            candidates: new[] { ($"http://127.0.0.1:{closedPort}", NpuSidecarRuntime.FastFlowLM) },
            tryLoadKittenTts: false);

        Assert.False(offlinePipeline.Stt.HasRemoteClient);
        await Assert.ThrowsAsync<NotSupportedException>(() => offlinePipeline.ListenAsync(new float[320]));
    }

    private static int GetClosedLoopbackPort()
    {
        using var probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        int port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private sealed class StubHttpHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _fn;

        public StubHttpHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> fn)
        {
            _fn = fn;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => _fn(request, cancellationToken);
    }

    private sealed class RejectingSyncContext : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state)
        {
            throw new InvalidOperationException("Sync-over-async attempted to post back to captured SynchronizationContext.");
        }
    }
}
