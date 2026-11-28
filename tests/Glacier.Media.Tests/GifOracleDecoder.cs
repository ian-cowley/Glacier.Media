namespace Glacier.Media.Tests;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;

/// <summary>
/// Pure C# GIF89a container parser and LZW decompression oracle for cross-platform test verification.
/// </summary>
internal static class GifOracleDecoder
{
    public sealed record DecodedGifFrame(int Width, int Height, byte[] Indices);

    public sealed record DecodedGif(
        int ScreenWidth,
        int ScreenHeight,
        List<DecodedGifFrame> Frames,
        byte[]? GlobalColorTable,
        int TotalClearCodes);

    public static DecodedGif Decode(byte[] gifBytes)
    {
        if (gifBytes.Length < 13)
            throw new InvalidDataException("Truncated GIF header.");

        string sig = Encoding.ASCII.GetString(gifBytes, 0, 6);
        if (sig != "GIF89a" && sig != "GIF87a")
            throw new InvalidDataException($"Invalid GIF signature: {sig}");

        int pos = 6;
        int screenWidth = BinaryPrimitives.ReadUInt16LittleEndian(gifBytes.AsSpan(pos, 2));
        int screenHeight = BinaryPrimitives.ReadUInt16LittleEndian(gifBytes.AsSpan(pos + 2, 2));
        byte packed = gifBytes[pos + 4];
        bool hasGct = (packed & 0x80) != 0;
        int gctSize = 1 << ((packed & 0x07) + 1);
        pos += 7;

        byte[]? gct = null;
        if (hasGct)
        {
            int gctBytes = gctSize * 3;
            gct = gifBytes.AsSpan(pos, gctBytes).ToArray();
            pos += gctBytes;
        }

        int totalClearCodes = 0;
        var frames = new List<DecodedGifFrame>();

        while (pos < gifBytes.Length && gifBytes[pos] != 0x3B)
        {
            byte blockType = gifBytes[pos++];

            if (blockType == 0x21)
            {
                pos++; // extension label
                while (pos < gifBytes.Length && gifBytes[pos] != 0)
                {
                    byte subLen = gifBytes[pos++];
                    pos += subLen;
                }
                pos++;
            }
            else if (blockType == 0x2C)
            {
                int imgWidth = BinaryPrimitives.ReadUInt16LittleEndian(gifBytes.AsSpan(pos + 4, 2));
                int imgHeight = BinaryPrimitives.ReadUInt16LittleEndian(gifBytes.AsSpan(pos + 6, 2));
                byte imgPacked = gifBytes[pos + 8];
                pos += 9;

                bool hasLct = (imgPacked & 0x80) != 0;
                if (hasLct)
                {
                    int lctSize = 1 << ((imgPacked & 0x07) + 1);
                    pos += lctSize * 3;
                }

                int minCodeSize = gifBytes[pos++];
                using var lzwStream = new MemoryStream();
                while (pos < gifBytes.Length && gifBytes[pos] != 0)
                {
                    byte subLen = gifBytes[pos++];
                    lzwStream.Write(gifBytes, pos, subLen);
                    pos += subLen;
                }
                pos++;

                byte[] decodedPixels = DecompressGifLzw(
                    lzwStream.ToArray(),
                    minCodeSize,
                    imgWidth * imgHeight,
                    out int clearCodes);

                totalClearCodes += clearCodes;
                frames.Add(new DecodedGifFrame(imgWidth, imgHeight, decodedPixels));
            }
            else
            {
                throw new InvalidDataException($"Unknown GIF block type: 0x{blockType:X2} at pos {pos - 1}");
            }
        }

        return new DecodedGif(screenWidth, screenHeight, frames, gct, totalClearCodes);
    }

    public static byte[] DecompressGifLzw(byte[] compressedData, int minCodeSize, int expectedPixels, out int clearCodeCount)
    {
        clearCodeCount = 0;
        int clearCode = 1 << minCodeSize;
        int eoiCode = clearCode + 1;

        int codeSize = minCodeSize + 1;
        int codeLimit = 1 << codeSize;
        int nextCode = eoiCode + 1;

        int[] prefix = new int[4096];
        Array.Fill(prefix, -1);
        byte[] suffix = new byte[4096];
        byte[] pixelStack = new byte[4096];

        for (int i = 0; i < clearCode; i++)
        {
            suffix[i] = (byte)i;
        }

        byte[] output = new byte[expectedPixels];
        int outIdx = 0;

        int bitAccumulator = 0;
        int bitsInAccumulator = 0;
        int bytePos = 0;

        int ReadCode()
        {
            while (bitsInAccumulator < codeSize)
            {
                if (bytePos >= compressedData.Length) return eoiCode;
                bitAccumulator |= compressedData[bytePos++] << bitsInAccumulator;
                bitsInAccumulator += 8;
            }

            int code = bitAccumulator & ((1 << codeSize) - 1);
            bitAccumulator >>= codeSize;
            bitsInAccumulator -= codeSize;
            return code;
        }

        int firstChar = 0;
        int oldCode = -1;

        while (outIdx < expectedPixels)
        {
            int code = ReadCode();

            if (code == clearCode)
            {
                clearCodeCount++;
                codeSize = minCodeSize + 1;
                codeLimit = 1 << codeSize;
                nextCode = eoiCode + 1;
                oldCode = -1;
                continue;
            }

            if (code == eoiCode) break;

            if (oldCode == -1)
            {
                output[outIdx++] = (byte)code;
                oldCode = code;
                firstChar = code;
                continue;
            }

            int inCode = code;
            int stackIdx = 0;

            if (code >= nextCode)
            {
                pixelStack[stackIdx++] = (byte)firstChar;
                code = oldCode;
            }

            while (code >= clearCode)
            {
                if (code >= 4096 || code < 0)
                    throw new InvalidDataException($"Out of bounds LZW code={code} for inCode={inCode}.");

                pixelStack[stackIdx++] = suffix[code];
                code = prefix[code];
            }

            if (code >= 0 && code < 4096)
            {
                firstChar = suffix[code];
                pixelStack[stackIdx++] = (byte)firstChar;
            }
            else
            {
                throw new InvalidDataException($"Out of bounds code={code} for inCode={inCode}");
            }

            while (stackIdx > 0 && outIdx < expectedPixels)
            {
                output[outIdx++] = pixelStack[--stackIdx];
            }

            if (nextCode < 4096)
            {
                prefix[nextCode] = oldCode;
                suffix[nextCode] = (byte)firstChar;
                nextCode++;

                if (nextCode >= codeLimit && codeSize < 12)
                {
                    codeSize++;
                    codeLimit = 1 << codeSize;
                }
            }

            oldCode = inCode;
        }

        return output;
    }
}
