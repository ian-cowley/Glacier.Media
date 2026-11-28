# Glacier.Media — Audio Manual (TTS, STT & Voice)

This guide covers speech synthesis, transcription, spectrograms, and WAV output in `Glacier.Media.Audio`.

---

## 1. Neural Text-to-Speech (`KittenTtsEngine`)

`Glacier.Media.Audio.Kitten.KittenTtsEngine` executes a distilled StyleTTS 2 architecture (`AlbertEncoder`, `KittenTextEncoder`, `KittenPredictor`, HiFi-GAN `KittenDecoder`) directly in C# from SafeTensors weights:

```csharp
using Glacier.Media.Audio;
using Glacier.Media.Audio.Kitten;

using var tts = new KittenTtsEngine(
    modelPath: "models/kitten/kitten-nano.safetensors",
    voicesPath: "models/kitten/kitten-voices.safetensors",
    configPath: "models/kitten/config.json");

// Or auto-discover files under ./models/kitten:
// using var tts = KittenTtsEngine.CreateDefault();

foreach (string voice in tts.AvailableVoices)
{
    Console.WriteLine(voice);
}
```

Use `tts.RegisterVoice(name, styleVector)` to add custom `[256]` or `[400, 256]` speaker embeddings.

---

## 2. Procedural Formant TTS (`ProceduralFormantTtsEngine` & `KokoroTtsEngine`)

When neural SafeTensors weights are absent or zero-dependency speech output is needed, `ProceduralFormantTtsEngine` (and its `KokoroTtsEngine` preset wrapper) synthesize 24 kHz mono audio using a Liljencrants-Fant glottal source and rule-based US/UK G2P (`PhonemeEngine`):

```csharp
using Glacier.Media.Audio;

using var engine = new KokoroTtsEngine(); // IsNeural == false, SupportsModelWeights == false
float[] samples = engine.Synthesize("All subsystems nominal.", KokoroVoice.BmGeorge, speed: 1.0f);
WavWriter.WritePcm16("status.wav", samples, engine.SampleRate, channels: 1);
```

16 presets (`KokoroVoice.AfHeart`, `AfBella`, `AfSarah`, `AfSky`, `AmAdam`, `AmMichael`, `AmEcho`, `AmEric`, `BfEmma`, `BfIsabella`, `BfAlice`, `BfLily`, `BmGeorge`, `BmLewis`, `BmDaniel`, `BmFable`) map to American or British RP formant profiles in `VocalTractSynthesizer`.

---

## 3. Speech-to-Text (`OpenAiWhisperClient` & `WhisperEngine`)

`WhisperEngine` (`[Experimental("GLACIER_EXP001")]`) routes audio transcriptions to an OpenAI-compatible `POST /v1/audio/transcriptions` REST endpoint or auto-discovered local NPU sidecar via `OpenAiWhisperClient`:

```csharp
#pragma warning disable GLACIER_EXP001
using var stt = new WhisperEngine(new WhisperEngineOptions
{
    Endpoint = "http://127.0.0.1:8000",
    ModelId = "whisper-large-v3",
    SampleRate = 16000
});
string text = await stt.TranscribeAsync(audioSamples16kHz);
#pragma warning restore GLACIER_EXP001
```

Because local in-process GGUF Whisper decoding is not implemented (`SupportsLocalDecoding == false`), `WhisperEngine` throws `NotSupportedException` if no remote or NPU sidecar endpoint is reachable.

---

## 4. `VoicePipeline`, `WavWriter` & `MelSpectrogram`

- **`VoicePipeline` (`[Experimental("GLACIER_EXP001")]`):** Combines `WhisperEngine` (`Listen` / `ListenAsync`) with `KokoroTtsEngine` and optional `KittenTtsEngine` (`Speak`, or `await VoicePipeline.CreateAutoAsync(tryLoadKittenTts: true)`).
- **`WavWriter`:** Writes 16-bit PCM (`WritePcm16`) and 32-bit IEEE float RIFF WAV files cleanly without native codecs.
- **`MelSpectrogram` & `AudioPlayer`:** Computes log-mel spectrograms for audio analysis and plays WAV/PCM streams on desktop platforms.
