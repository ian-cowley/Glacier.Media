namespace Glacier.Media.Audio;

using System;
using System.Collections.Generic;

/// <summary>
/// Predefined voice style presets named after Kokoro-82M voices that map to procedural
/// Liljencrants-Fant vocal-tract formant profiles (no neural weights loaded).
/// </summary>
public enum KokoroVoice
{
    // American Female Voices
    AfHeart,    // American Female: Warm, natural, expressive narrator
    AfBella,    // American Female: Bright, conversational, clear
    AfSarah,    // American Female: Calm, smooth, executive
    AfSky,      // American Female: Light, youthful, melodic

    // American Male Voices
    AmAdam,     // American Male: Deep broadcast baritone, authoritative
    AmMichael,  // American Male: Clear, engaging, professional
    AmEcho,     // American Male: Resonant, youthful conversational
    AmEric,     // American Male: Grounded, warm storyteller

    // British Female Voices
    BfEmma,     // British Female: Elegant, articulate Received Pronunciation (RP)
    BfIsabella, // British Female: Refined, classic BBC documentary style
    BfAlice,    // British Female: Gentle, conversational Southern English
    BfLily,     // British Female: Crisp, melodic modern English

    // British Male Voices
    BmGeorge,   // British Male: Authoritative, rich classical RP orator
    BmLewis,    // British Male: Warm, engaging British storyteller
    BmDaniel,   // British Male: Deep resonant theatrical narrator
    BmFable     // British Male: Conversational, friendly London gentleman
}

/// <summary>
/// Procedural Liljencrants-Fant glottal formant Text-to-Speech (TTS) synthesizer preset facade over
/// <see cref="ProceduralFormantTtsEngine"/> using <see cref="KokoroVoice"/> formant profiles.
/// Does not load neural Kokoro-82M weights (<see cref="IsNeural"/> is <see langword="false"/>);
/// for neural SafeTensors TTS synthesis, use <see cref="Kitten.KittenTtsEngine"/>.
/// </summary>
public sealed class KokoroTtsEngine : IDisposable
{
    public const int DefaultSampleRate = ProceduralFormantTtsEngine.DefaultSampleRate;
    public int SampleRate => _engine.SampleRate;
    public bool IsNeural => _engine.IsNeural;
    public bool SupportsModelWeights => _engine.SupportsModelWeights;
    public string ActiveBackend => _engine.ActiveBackend;

    private readonly ProceduralFormantTtsEngine _engine = new();
    private bool _disposed;

    public IReadOnlyDictionary<KokoroVoice, VoiceProfile> VoiceProfiles => _engine.VoiceProfiles;

    /// <summary>
    /// Synthesizes plain English text into 24kHz single-channel float audio samples using procedural vocal-tract synthesis.
    /// </summary>
    public float[] Synthesize(string text, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _engine.Synthesize(text, voice, speed);
    }

    /// <summary>
    /// Synthesizes speech and writes it directly to a 24kHz 16-bit PCM WAV file on disk.
    /// </summary>
    public void SynthesizeToFile(string text, string outputPath, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _engine.SynthesizeToFile(text, outputPath, voice, speed);
    }

    /// <summary>
    /// Retrieves the procedural formant voice profile definition for a given preset enum.
    /// </summary>
    public VoiceProfile GetVoiceProfile(KokoroVoice voice) => _engine.GetVoiceProfile(voice);

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _engine.Dispose();
        }
    }
}
