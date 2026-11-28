namespace Glacier.Media.Image.Flux;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

/// <summary>
/// Pure C# .NET 10 NativeAOT-compatible byte-level BPE tokenizer matching OpenAI CLIP-L (<c>CLIPTokenizer</c>).
/// Supports loading standard <c>(vocab.json, merges.txt)</c>, HuggingFace <c>tokenizer.json</c>,
/// or in-memory vocabularies and merge rules with zero external dependencies.
/// </summary>
public sealed partial class ClipTokenizer
{
    public const int DefaultBosTokenId = 49406;
    public const int DefaultEosTokenId = 49407;
    public const int DefaultMaxLength = 77;
    private const int MaxCacheEntries = 16_384;

    private static readonly char[] ByteToChar = new char[256];
    private static readonly Dictionary<char, byte> CharToByte = new(256);
    private static readonly string[] CharToStringLut = new string[256];
    private static readonly string[] CharToEndOfWordStringLut = new string[256];

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string Left, string Right), int> _bpePairRanks;
    private readonly ConcurrentDictionary<string, int[]> _wordCache = new(StringComparer.Ordinal);

    public int VocabSize => _vocab.Count;
    public int BosTokenId { get; }
    public int EosTokenId { get; }

    static ClipTokenizer()
    {
        var bs = new List<int>(256);
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < 256; i++)
        {
            byte b = (byte)bs[i];
            char c = (char)cs[i];
            ByteToChar[b] = c;
            CharToByte[c] = b;
            string s = c.ToString();
            CharToStringLut[b] = s;
            CharToEndOfWordStringLut[b] = string.Concat(s, "</w>");
        }
    }

    internal static char EncodeByteToChar(byte b) => ByteToChar[b];

    internal static bool TryDecodeCharToByte(char c, out byte b) => CharToByte.TryGetValue(c, out b);

    private ClipTokenizer(
        Dictionary<string, int> vocab,
        Dictionary<(string Left, string Right), int> bpePairRanks,
        int? bosTokenId = null,
        int? eosTokenId = null)
    {
        _vocab = vocab;
        _bpePairRanks = bpePairRanks;
        _idToToken = new Dictionary<int, string>(vocab.Count);
        foreach (var kvp in vocab)
        {
            _idToToken.TryAdd(kvp.Value, kvp.Key);
        }

        if (bosTokenId.HasValue)
        {
            BosTokenId = bosTokenId.Value;
        }
        else if (vocab.TryGetValue("<|startoftext|>", out int bos))
        {
            BosTokenId = bos;
        }
        else
        {
            BosTokenId = DefaultBosTokenId;
        }

        if (eosTokenId.HasValue)
        {
            EosTokenId = eosTokenId.Value;
        }
        else if (vocab.TryGetValue("<|endoftext|>", out int eos))
        {
            EosTokenId = eos;
        }
        else
        {
            EosTokenId = DefaultEosTokenId;
        }
    }

    /// <summary>
    /// Loads a CLIP BPE tokenizer from standard <c>vocab.json</c> and <c>merges.txt</c> files.
    /// </summary>
    public static ClipTokenizer Load(string vocabJsonPath, string mergesTxtPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(vocabJsonPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(mergesTxtPath);

        if (!File.Exists(vocabJsonPath))
            throw new FileNotFoundException($"CLIP vocabulary file not found: '{vocabJsonPath}'.", vocabJsonPath);
        if (!File.Exists(mergesTxtPath))
            throw new FileNotFoundException($"CLIP BPE merges file not found: '{mergesTxtPath}'.", mergesTxtPath);

        byte[] vocabBytes = File.ReadAllBytes(vocabJsonPath);
        var vocab = ParseVocabJson(vocabBytes);
        var merges = File.ReadLines(mergesTxtPath);
        var ranks = ParseMerges(merges);
        return new ClipTokenizer(vocab, ranks);
    }

    /// <summary>
    /// Loads a CLIP BPE tokenizer from a HuggingFace <c>tokenizer.json</c> file using NativeAOT-safe <see cref="Utf8JsonReader"/>.
    /// </summary>
    public static ClipTokenizer LoadFromTokenizerJson(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"CLIP tokenizer.json file not found: '{tokenizerJsonPath}'.", tokenizerJsonPath);

        byte[] jsonBytes = File.ReadAllBytes(tokenizerJsonPath);
        ReadOnlySpan<byte> utf8Span = StripUtf8Bom(jsonBytes);
        var reader = new Utf8JsonReader(utf8Span, new JsonReaderOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        });

        Dictionary<string, int>? vocab = null;
        Dictionary<(string, string), int>? ranks = null;
        int? bosId = null;
        int? eosId = null;

        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            throw new InvalidDataException("Invalid tokenizer.json: expected root JSON object.");

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
                break;

            if (reader.TokenType != JsonTokenType.PropertyName)
                continue;

            string? propName = reader.GetString();
            if (!reader.Read()) break;

            if (string.Equals(propName, "added_tokens", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.StartArray)
            {
                ParseAddedTokens(ref reader, ref bosId, ref eosId, out var addedMap);
                if (addedMap.Count > 0)
                {
                    vocab ??= new Dictionary<string, int>(StringComparer.Ordinal);
                    foreach (var kvp in addedMap)
                        vocab.TryAdd(kvp.Key, kvp.Value);
                }
            }
            else if (string.Equals(propName, "model", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.StartObject)
            {
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;
                    if (reader.TokenType != JsonTokenType.PropertyName)
                        continue;

                    string? modelProp = reader.GetString();
                    if (!reader.Read()) break;

                    if (string.Equals(modelProp, "vocab", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.StartObject)
                    {
                        var modelVocab = ReadVocabObject(ref reader);
                        if (vocab == null)
                        {
                            vocab = modelVocab;
                        }
                        else
                        {
                            foreach (var kvp in modelVocab)
                                vocab.TryAdd(kvp.Key, kvp.Value);
                        }
                    }
                    else if (string.Equals(modelProp, "merges", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.StartArray)
                    {
                        ranks = ReadMergesArray(ref reader);
                    }
                    else
                    {
                        reader.Skip();
                    }
                }
            }
            else
            {
                reader.Skip();
            }
        }

        if (vocab == null || vocab.Count == 0)
            throw new InvalidDataException($"Invalid tokenizer.json '{tokenizerJsonPath}': missing or empty 'model.vocab'.");

        ranks ??= new Dictionary<(string, string), int>();
        return new ClipTokenizer(vocab, ranks, bosId, eosId);
    }

    /// <summary>
    /// Constructs a <see cref="ClipTokenizer"/> directly from in-memory vocabulary and merge lines.
    /// </summary>
    public static ClipTokenizer FromVocabularyAndMerges(
        IReadOnlyDictionary<string, int> vocab,
        IEnumerable<string> merges)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(merges);

        var vocabCopy = new Dictionary<string, int>(vocab.Count, StringComparer.Ordinal);
        foreach (var kvp in vocab)
        {
            vocabCopy[kvp.Key] = kvp.Value;
        }

        var ranks = ParseMerges(merges);
        return new ClipTokenizer(vocabCopy, ranks);
    }

    /// <summary>
    /// Encodes a prompt into CLIP token IDs framed with BOS/EOS and padded/truncated to <paramref name="maxLength"/>.
    /// </summary>
    public int[] Encode(string prompt, int maxLength = DefaultMaxLength)
        => Encode(prompt, maxLength, out _);

    /// <summary>
    /// Encodes a prompt into CLIP token IDs framed with BOS/EOS, returning the index of the terminating EOS token.
    /// </summary>
    public int[] Encode(string prompt, int maxLength, out int eosIndex)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (maxLength < 2)
            throw new ArgumentOutOfRangeException(nameof(maxLength), "maxLength must be at least 2 to hold BOS and EOS tokens.");

        var tokens = new int[maxLength];
        tokens[0] = BosTokenId;
        int idx = 1;

        string normalized = NormalizeWhitespaceAndLower(prompt);
        if (normalized.Length > 0)
        {
            foreach (ValueMatch m in ClipTokenPattern().EnumerateMatches(normalized))
            {
                if (idx >= maxLength - 1)
                    break;

                ReadOnlySpan<char> pieceSpan = normalized.AsSpan(m.Index, m.Length);
                if (pieceSpan.Equals("<|startoftext|>", StringComparison.Ordinal))
                {
                    tokens[idx++] = BosTokenId;
                    continue;
                }
                if (pieceSpan.Equals("<|endoftext|>", StringComparison.Ordinal))
                {
                    tokens[idx++] = EosTokenId;
                    continue;
                }

                string piece = normalized.Substring(m.Index, m.Length);
                int[] pieceIds = EncodeWord(piece);
                for (int k = 0; k < pieceIds.Length && idx < maxLength - 1; k++)
                {
                    tokens[idx++] = pieceIds[k];
                }
            }
        }

        eosIndex = idx;
        tokens[idx++] = EosTokenId;
        while (idx < maxLength)
        {
            tokens[idx++] = EosTokenId;
        }

        return tokens;
    }

    /// <summary>
    /// Decodes a sequence of CLIP token IDs back into UTF-8 text.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        if (tokens.IsEmpty)
            return string.Empty;

        var byteList = new List<byte>(tokens.Length * 4);
        Span<byte> utf8 = stackalloc byte[4];
        for (int i = 0; i < tokens.Length; i++)
        {
            int id = tokens[i];
            if (id == BosTokenId || id == EosTokenId)
                continue;

            if (!_idToToken.TryGetValue(id, out string? tok))
                continue;

            bool isEndOfWord = tok.EndsWith("</w>", StringComparison.Ordinal);
            ReadOnlySpan<char> core = isEndOfWord ? tok.AsSpan(0, tok.Length - 4) : tok.AsSpan();

            for (int c = 0; c < core.Length; c++)
            {
                char ch = core[c];
                if (CharToByte.TryGetValue(ch, out byte b))
                {
                    byteList.Add(b);
                }
                else
                {
                    int written = Encoding.UTF8.GetBytes(core.Slice(c, 1), utf8);
                    for (int w = 0; w < written; w++)
                        byteList.Add(utf8[w]);
                }
            }

            if (isEndOfWord)
            {
                byteList.Add((byte)' ');
            }
        }

        if (byteList.Count == 0)
            return string.Empty;

        string decoded = Encoding.UTF8.GetString(byteList.ToArray());
        return decoded.TrimEnd();
    }

    private int[] EncodeWord(string word)
    {
        if (_wordCache.TryGetValue(word, out int[]? cached))
            return cached;

        int maxBytes = Encoding.UTF8.GetMaxByteCount(word.Length);
        byte[]? rented = maxBytes > 256 ? ArrayPool<byte>.Shared.Rent(maxBytes) : null;
        try
        {
            Span<byte> byteBuf = rented != null ? rented.AsSpan(0, maxBytes) : stackalloc byte[256];
            int byteLen = Encoding.UTF8.GetBytes(word.AsSpan(), byteBuf);
            if (byteLen == 0)
                return Array.Empty<int>();

            var symbols = new List<string>(byteLen);
            for (int i = 0; i < byteLen - 1; i++)
            {
                symbols.Add(CharToStringLut[byteBuf[i]]);
            }
            symbols.Add(CharToEndOfWordStringLut[byteBuf[byteLen - 1]]);

            if (symbols.Count > 1 && _bpePairRanks.Count > 0)
            {
                ApplyBpeMerges(symbols);
            }

            var ids = new List<int>(symbols.Count);
            for (int i = 0; i < symbols.Count; i++)
            {
                string s = symbols[i];
                if (_vocab.TryGetValue(s, out int id))
                {
                    ids.Add(id);
                }
            }

            int[] result = ids.ToArray();
            if (_wordCache.Count < MaxCacheEntries)
            {
                _wordCache.TryAdd(word, result);
            }
            return result;
        }
        finally
        {
            if (rented != null)
                ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private void ApplyBpeMerges(List<string> symbols)
    {
        int n = symbols.Count;
        if (n <= 1 || _bpePairRanks.Count == 0)
            return;

        var sym = new string[n];
        var prev = new int[n];
        var next = new int[n];
        var alive = new bool[n];
        var gen = new int[n];

        for (int i = 0; i < n; i++)
        {
            sym[i] = symbols[i];
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
            alive[i] = true;
        }

        var pq = new PriorityQueue<(int Left, int Right, int LeftGen, int RightGen), (int Rank, int Pos)>(n);

        void TryPush(int l)
        {
            if (l < 0 || !alive[l]) return;
            int r = next[l];
            if (r < 0 || !alive[r]) return;
            if (_bpePairRanks.TryGetValue((sym[l], sym[r]), out int rank))
            {
                pq.Enqueue((l, r, gen[l], gen[r]), (rank, l));
            }
        }

        for (int i = 0; i < n - 1; i++)
        {
            TryPush(i);
        }

        while (pq.TryDequeue(out var item, out _))
        {
            int l = item.Left;
            int r = item.Right;
            if (!alive[l] || !alive[r] || next[l] != r || gen[l] != item.LeftGen || gen[r] != item.RightGen)
                continue;

            sym[l] = string.Concat(sym[l], sym[r]);
            gen[l]++;
            alive[r] = false;
            int rr = next[r];
            next[l] = rr;
            if (rr >= 0)
                prev[rr] = l;

            TryPush(prev[l]);
            TryPush(l);
        }

        symbols.Clear();
        for (int cur = 0; cur >= 0; cur = next[cur])
        {
            if (alive[cur])
                symbols.Add(sym[cur]);
        }
    }

    private static string NormalizeWhitespaceAndLower(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        var sb = new StringBuilder(text.Length);
        bool inSpace = true; // Trim leading whitespace
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (char.IsWhiteSpace(c))
            {
                if (!inSpace)
                {
                    sb.Append(' ');
                    inSpace = true;
                }
            }
            else
            {
                sb.Append(c);
                inSpace = false;
            }
        }

        if (sb.Length > 0 && sb[^1] == ' ')
            sb.Length--;

        return sb.ToString().ToLowerInvariant();
    }

    private static ReadOnlySpan<byte> StripUtf8Bom(ReadOnlySpan<byte> utf8)
        => utf8.Length >= 3 && utf8[0] == 0xEF && utf8[1] == 0xBB && utf8[2] == 0xBF
            ? utf8.Slice(3)
            : utf8;

    private static Dictionary<string, int> ParseVocabJson(ReadOnlySpan<byte> utf8Json)
    {
        var reader = new Utf8JsonReader(StripUtf8Bom(utf8Json), new JsonReaderOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        });

        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            throw new InvalidDataException("Invalid CLIP vocab.json: expected root JSON object.");

        return ReadVocabObject(ref reader);
    }

    private static Dictionary<string, int> ReadVocabObject(ref Utf8JsonReader reader)
    {
        var vocab = new Dictionary<string, int>(49408, StringComparer.Ordinal);
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
                break;

            if (reader.TokenType != JsonTokenType.PropertyName)
                continue;

            string? token = reader.GetString();
            if (!reader.Read()) break;

            if (token != null && reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out int id))
            {
                vocab[token] = id;
            }
            else
            {
                reader.Skip();
            }
        }
        return vocab;
    }

    private static Dictionary<(string Left, string Right), int> ParseMerges(IEnumerable<string> merges)
    {
        var ranks = new Dictionary<(string Left, string Right), int>(50000);
        int rank = 0;
        foreach (string raw in merges)
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;
            if (raw.StartsWith("#version:", StringComparison.Ordinal))
                continue;

            int sp = raw.IndexOf(' ');
            if (sp <= 0 || sp >= raw.Length - 1)
                continue;

            string left = raw.Substring(0, sp);
            string right = raw.Substring(sp + 1);
            int extraSp = right.IndexOf(' ');
            if (extraSp >= 0)
                right = right.Substring(0, extraSp);

            if (left.Length == 0 || right.Length == 0)
                continue;

            if (ranks.TryAdd((left, right), rank))
            {
                rank++;
            }
        }
        return ranks;
    }

    private static Dictionary<(string Left, string Right), int> ReadMergesArray(ref Utf8JsonReader reader)
    {
        var ranks = new Dictionary<(string Left, string Right), int>(50000);
        int rank = 0;
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndArray)
                break;

            if (reader.TokenType == JsonTokenType.String)
            {
                string? line = reader.GetString();
                if (!string.IsNullOrWhiteSpace(line) && !line.StartsWith("#version:", StringComparison.Ordinal))
                {
                    int sp = line.IndexOf(' ');
                    if (sp > 0 && sp < line.Length - 1)
                    {
                        string left = line.Substring(0, sp);
                        string right = line.Substring(sp + 1);
                        if (ranks.TryAdd((left, right), rank))
                            rank++;
                    }
                }
            }
            else if (reader.TokenType == JsonTokenType.StartArray)
            {
                string? left = null;
                string? right = null;
                int idx = 0;
                while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                {
                    if (reader.TokenType == JsonTokenType.String)
                    {
                        if (idx == 0) left = reader.GetString();
                        else if (idx == 1) right = reader.GetString();
                    }
                    else
                    {
                        reader.Skip();
                    }
                    idx++;
                }
                if (!string.IsNullOrEmpty(left) && !string.IsNullOrEmpty(right) && ranks.TryAdd((left, right), rank))
                {
                    rank++;
                }
            }
            else
            {
                reader.Skip();
            }
        }
        return ranks;
    }

    private static void ParseAddedTokens(
        ref Utf8JsonReader reader,
        ref int? bosId,
        ref int? eosId,
        out Dictionary<string, int> added)
    {
        added = new Dictionary<string, int>(StringComparer.Ordinal);
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndArray)
                break;

            if (reader.TokenType != JsonTokenType.StartObject)
            {
                reader.Skip();
                continue;
            }

            int? id = null;
            string? content = null;
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;
                if (reader.TokenType != JsonTokenType.PropertyName)
                    continue;

                string? field = reader.GetString();
                if (!reader.Read()) break;

                if (string.Equals(field, "id", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out int parsedId))
                    id = parsedId;
                else if (string.Equals(field, "content", StringComparison.Ordinal) && reader.TokenType == JsonTokenType.String)
                    content = reader.GetString();
                else
                    reader.Skip();
            }

            if (id.HasValue && !string.IsNullOrEmpty(content))
            {
                added[content] = id.Value;
                if (string.Equals(content, "<|startoftext|>", StringComparison.Ordinal))
                    bosId = id.Value;
                else if (string.Equals(content, "<|endoftext|>", StringComparison.Ordinal))
                    eosId = id.Value;
            }
        }
    }

    [GeneratedRegex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ClipTokenPattern();
}
