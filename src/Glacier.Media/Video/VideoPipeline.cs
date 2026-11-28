namespace Glacier.Media.Video;

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using Glacier.Media.Vision;

/// <summary>
/// Temporal metadata and motion statistics computed across a video sequence.
/// </summary>
public sealed record VideoMetadata(
    int TotalFrames,
    int KeyframeSampleCount,
    float AverageMotionEnergy,
    float MaxMotionEnergy,
    int PeakMotionFrame,
    string MotionPattern,
    string TemporalSummary);

/// <summary>
/// Experimental procedural Video ingestion pipeline supporting uniform keyframe sampling, motion profiling,
/// 3D-RoPE spatial-temporal embeddings, and synthetic visual feature extraction.
/// VLM natural-language query (<see cref="Query(IReadOnlyList{byte[]}, int, int, string, int, int)"/>) requires
/// an explicit external VLM query delegate; otherwise <see cref="SupportsVlmQuery"/> is <see langword="false"/>.
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed class VideoPipeline : IDisposable
{
#pragma warning disable GLACIER_EXP001
    private readonly VisionPipeline _visionPipeline;
    private readonly VideoRoPE _videoRope;
    private readonly Func<VideoMetadata, float[], string, string>? _vlmQueryHandler;
    private bool _disposed;

    public VisionPipeline Vision => _visionPipeline;
    public VideoRoPE VideoRoPE => _videoRope;
    public bool SupportsVlmQuery => _vlmQueryHandler != null;
    public bool IsNeuralVlm => false;
    public string ActiveBackend => _vlmQueryHandler != null ? "ExternalVlmDelegate" : "ProceduralMetadataAndEmbeddings";

    public VideoPipeline(
        int numLayers = 4,
        int visionDim = 768,
        int llmDim = 3584,
        int patchSize = 14,
        Func<VideoMetadata, float[], string, string>? vlmQueryHandler = null)
    {
        _visionPipeline = new VisionPipeline(numLayers, visionDim, llmDim, patchSize);
        _videoRope = new VideoRoPE(_visionPipeline.Vit.HeadDim, 10000.0f);
        _vlmQueryHandler = vlmQueryHandler;
    }

    /// <summary>
    /// Computes deterministic keyframe and motion dynamics metadata across a sequence of video frames.
    /// </summary>
    public VideoMetadata AnalyzeFrames(
        IReadOnlyList<byte[]> frames,
        int width,
        int height,
        int targetKeyframes = 4,
        int channels = 3)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _ = width;
        _ = height;
        _ = channels;

        if (frames == null || frames.Count == 0)
        {
            throw new ArgumentException("Video must contain at least 1 frame.", nameof(frames));
        }

        int[] keyframeIndices = VideoFrameSampler.SampleUniformIndices(frames.Count, targetKeyframes);
        int sampleCount = keyframeIndices.Length;

        float totalMotion = 0.0f;
        float maxMotion = 0.0f;
        int peakFrame = 0;

        for (int i = 1; i < frames.Count; i++)
        {
            float motion = VideoFrameSampler.ComputeMotionEnergy(frames[i - 1], frames[i]);
            totalMotion += motion;
            if (motion > maxMotion)
            {
                maxMotion = motion;
                peakFrame = i;
            }
        }

        float avgMotion = frames.Count > 1 ? totalMotion / (frames.Count - 1) : 0.0f;
        string motionPattern = avgMotion switch
        {
            > 0.25f => "High dynamic activity / rapid camera motion / fast subject movement",
            > 0.10f => "Moderate continuous motion / steady panning or walking pace",
            > 0.02f => "Low subtle motion / stationary camera with minor subject action",
            _ => "Static / stationary shot with minimal temporal variation"
        };

        string summary = $"Sequence of {frames.Count} frames sampled down to {sampleCount} keyframes. Mean motion energy: {avgMotion * 100:F1}%. Peak action at frame #{peakFrame}.";

        return new VideoMetadata(
            TotalFrames: frames.Count,
            KeyframeSampleCount: sampleCount,
            AverageMotionEnergy: avgMotion,
            MaxMotionEnergy: maxMotion,
            PeakMotionFrame: peakFrame,
            MotionPattern: motionPattern,
            TemporalSummary: summary);
    }

    /// <summary>
    /// Formats a deterministic, factual inspection summary from computed <see cref="VideoMetadata"/>.
    /// </summary>
    public static string SummarizeMetadata(VideoMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return $"{metadata.TemporalSummary} Pattern: {metadata.MotionPattern}. Max motion energy: {metadata.MaxMotionEnergy * 100:F1}% (frame #{metadata.PeakMotionFrame}).";
    }

    /// <summary>
    /// Processes a sequence of video frames and extracts synthetic temporal tokens and motion statistics.
    /// </summary>
    public (int TotalTokens, float[] TemporalEmbeddings, VideoMetadata Metadata) ProcessVideo(
        IReadOnlyList<byte[]> frames,
        int width,
        int height,
        int targetKeyframes = 4,
        int channels = 3)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var metadata = AnalyzeFrames(frames, width, height, targetKeyframes, channels);
        int[] keyframeIndices = VideoFrameSampler.SampleUniformIndices(frames.Count, targetKeyframes);
        int sampleCount = keyframeIndices.Length;

        int tokensPerFrame = 256; // Standard 2x2 merged tokens for 448x448
        int totalVisualTokens = sampleCount * tokensPerFrame;
        var allEmbeddings = new float[totalVisualTokens * _visionPipeline.Projector.LlmDim];

        for (int k = 0; k < sampleCount; k++)
        {
            int frameIdx = keyframeIndices[k];
            var (tokens, emb, _) = _visionPipeline.ProcessImage(frames[frameIdx], width, height, channels);
            int copyCount = Math.Min(tokens, tokensPerFrame) * _visionPipeline.Projector.LlmDim;
            Array.Copy(emb, 0, allEmbeddings, k * tokensPerFrame * _visionPipeline.Projector.LlmDim, copyCount);
        }

        return (totalVisualTokens, allEmbeddings, metadata);
    }
#pragma warning restore GLACIER_EXP001

    /// <summary>
    /// Queries a video sequence via the configured external VLM query delegate.
    /// Throws <see cref="NotSupportedException"/> when <see cref="SupportsVlmQuery"/> is <see langword="false"/>.
    /// </summary>
    public string Query(
        IReadOnlyList<byte[]> frames,
        int width,
        int height,
        string prompt,
        int targetKeyframes = 4,
        int channels = 3)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_vlmQueryHandler == null)
        {
            throw new NotSupportedException(
                "VideoPipeline.Query VLM text generation requires multimodal video-LLM weights or an external VLM query delegate. Use AnalyzeFrames or SummarizeMetadata for deterministic motion inspection.");
        }

        ArgumentNullException.ThrowIfNull(prompt);
        var (_, emb, meta) = ProcessVideo(frames, width, height, targetKeyframes, channels);
        return _vlmQueryHandler(meta, emb, prompt);
    }

    /// <summary>
    /// Queries precomputed <see cref="VideoMetadata"/> via the configured external VLM query delegate.
    /// Throws <see cref="NotSupportedException"/> when <see cref="SupportsVlmQuery"/> is <see langword="false"/>.
    /// </summary>
    public string Query(VideoMetadata meta, string prompt, int totalTokens = 1024)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_vlmQueryHandler == null)
        {
            throw new NotSupportedException(
                "VideoPipeline.Query VLM text generation requires multimodal video-LLM weights or an external VLM query delegate. Use SummarizeMetadata for deterministic motion inspection.");
        }

        ArgumentNullException.ThrowIfNull(meta);
        ArgumentNullException.ThrowIfNull(prompt);
        _ = totalTokens;
        return _vlmQueryHandler(meta, Array.Empty<float>(), prompt);
    }

#pragma warning disable GLACIER_EXP001
    public void Dispose()
    {
        if (!_disposed)
        {
            _visionPipeline.Dispose();
            _disposed = true;
        }
    }
#pragma warning restore GLACIER_EXP001
}
