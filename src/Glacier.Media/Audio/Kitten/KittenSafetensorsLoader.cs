namespace Glacier.Media.Audio.Kitten;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Format;

/// <summary>
/// Shared Safetensors F32 validation and loading helpers for KittenTTS modules.
/// </summary>
internal static unsafe class KittenSafetensorsLoader
{
    public static float* ValidateAndGetFloat32Pointer(
        SafetensorsFile file,
        string tensorName,
        out ulong elementCount,
        out ulong byteSize,
        int? expectedElementCount = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentException.ThrowIfNullOrEmpty(tensorName);

        if (!file.Tensors.TryGetValue(tensorName, out var info))
            throw new KeyNotFoundException($"Tensor '{tensorName}' not found in KittenTTS Safetensors file.");

        if (!string.Equals(info.Dtype, "F32", StringComparison.Ordinal))
            throw new NotSupportedException($"KittenTTS requires F32 Safetensors weights, but tensor '{tensorName}' has dtype '{info.Dtype}'.");

        if (info.ElementCount < 0)
            throw new InvalidDataException($"Tensor '{tensorName}' has negative element count {info.ElementCount}.");

        ulong elemCount = (ulong)info.ElementCount;
        ulong expectedBytes = checked(elemCount * (ulong)sizeof(float));
        if (info.ByteSize != expectedBytes)
            throw new InvalidDataException($"Tensor '{tensorName}' byte size {info.ByteSize} does not match F32 element count {elemCount} (expected {expectedBytes} bytes).");

        if (expectedElementCount is int expectedCount && (expectedCount < 0 || elemCount != (ulong)expectedCount))
            throw new InvalidDataException($"Tensor '{tensorName}' element count {elemCount} does not match expected {expectedCount}.");

        elementCount = elemCount;
        byteSize = info.ByteSize;
        return (float*)file.GetTensorPointer(tensorName);
    }

    public static float[] LoadFloat32Array(
        SafetensorsFile file,
        string tensorName,
        int? expectedElementCount = null)
    {
        float* pSrc = ValidateAndGetFloat32Pointer(file, tensorName, out ulong elementCount, out ulong byteSize, expectedElementCount);
        float[] arr = new float[checked((int)elementCount)];
        if (elementCount > 0)
        {
            fixed (float* pDst = arr)
            {
                Buffer.MemoryCopy(pSrc, pDst, (long)byteSize, (long)byteSize);
            }
        }
        return arr;
    }
}
