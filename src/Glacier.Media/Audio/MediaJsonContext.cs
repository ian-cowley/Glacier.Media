namespace Glacier.Media.Audio;

using System.Collections.Generic;
using System.Text.Json.Serialization;

internal sealed class WhisperTranscriptionResponse
{
    [JsonPropertyName("text")]
    public string? Text { get; set; }

    [JsonPropertyName("language")]
    public string? Language { get; set; }

    [JsonPropertyName("duration")]
    public double? Duration { get; set; }

    [JsonPropertyName("segments")]
    public List<WhisperSegment>? Segments { get; set; }
}

internal sealed class WhisperSegment
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("start")]
    public double Start { get; set; }

    [JsonPropertyName("end")]
    public double End { get; set; }

    [JsonPropertyName("text")]
    public string? Text { get; set; }
}

internal sealed class WhisperErrorResponse
{
    [JsonPropertyName("error")]
    public WhisperErrorDetail? Error { get; set; }
}

internal sealed class WhisperErrorDetail
{
    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("type")]
    public string? Type { get; set; }

    [JsonPropertyName("code")]
    public string? Code { get; set; }
}

[JsonSourceGenerationOptions(
    PropertyNameCaseInsensitive = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(WhisperTranscriptionResponse))]
[JsonSerializable(typeof(WhisperSegment))]
[JsonSerializable(typeof(WhisperErrorResponse))]
[JsonSerializable(typeof(WhisperErrorDetail))]
internal partial class MediaJsonContext : JsonSerializerContext
{
}
