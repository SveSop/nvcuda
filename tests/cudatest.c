#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

typedef CUresult (*pcuInit)(unsigned int flags);
typedef CUresult (*pcuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (*pcuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (*pcuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
typedef CUresult (*pcuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);
typedef CUresult (*pcuDeviceGetLuid)(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
typedef CUresult (*pcuGetExportTable)(const void** table, const CUuuid *id);

pcuInit cuInit;
pcuDeviceGet cuDeviceGet;
pcuDeviceGetName cuDeviceGetName;
pcuDeviceComputeCapability cuDeviceComputeCapability;
pcuDeviceGetUuid cuDeviceGetUuid;
pcuDeviceGetLuid cuDeviceGetLuid;
pcuGetExportTable cuGetExportTable;

static const struct
{
    int size;
    cuuint64_t addrList[100]; // Should not overflow
} *addrTable = NULL;

// CUDA Relay UUID's
static const CUuuid UUID_Relay1                     = {{0x6B, 0xD5, 0xFB, 0x6C, 0x5B, 0xF4, 0xE7, 0x4A,
                                                        0x89, 0x87, 0xD9, 0x39, 0x12, 0xFD, 0x9D, 0xF9}};
                                                    // {6bd5fb6c-5bf4-e74a-8987-d93912fd9df9}
static const CUuuid UUID_Relay2                     = {{0xA0, 0x94, 0x79, 0x8C, 0x2E, 0x74, 0x2E, 0x74,
                                                        0x93, 0xF2, 0x08, 0x00, 0x20, 0x0C, 0x0A, 0x66}};
                                                    // {a094798c-2e74-2e74-93f2-0800200c0a66}
static const CUuuid UUID_Relay3                     = {{0x42, 0xD8, 0x5A, 0x81, 0x23, 0xF6, 0xCB, 0x47,
                                                        0x82, 0x98, 0xF6, 0xE7, 0x8A, 0x3A, 0xEC, 0xDC}};
                                                    // {42d85a81-23f6-cb47-8298-f6e78a3aecdc}
static const CUuuid UUID_ContextStorage             = {{0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11,
                                                        0xA8, 0xC3, 0x68, 0xF3, 0x55, 0xD8, 0x95, 0x93}};
                                                    // {c693336e-1121-df11-a8c3-68f355d89593}
static const CUuuid UUID_Relay5                     = {{0x0C, 0xA5, 0x0B, 0x8C, 0x10, 0x04, 0x92, 0x9A,
                                                        0x89, 0xA7, 0xD0, 0xDF, 0x10, 0xE7, 0x72, 0x86}};
                                                    // {0ca50b8c-1004-929a-89a7-f0ff10e77286}
static const CUuuid UUID_TlsNotifyInterface         = {{0x19, 0x5B, 0xCB, 0xF4, 0xD6, 0x7D, 0x02, 0x4A,
                                                        0xAC, 0xC5, 0x1D, 0x29, 0xCE, 0xA6, 0x31, 0xAE}};
                                                    // {195bcbf4-d67d-024a-acc5-1d29cea631ae}
static const CUuuid UUID_Encryption                 = {{0xD4, 0x08, 0x20, 0x55, 0xBD, 0xE6, 0x70, 0x4B,
                                                        0x8D, 0x34, 0xBA, 0x12, 0x3C, 0x66, 0xE1, 0xF2}};
                                                    // {d4082055-bde6-704b-8d34-ba123c66e1f2}
static const CUuuid UUID_Relay8                     = {{0x21, 0x31, 0x8C, 0x60, 0x97, 0x14, 0x32, 0x48,
                                                        0x8C, 0xA6, 0x41, 0xFF, 0x73, 0x24, 0xC8, 0xF2}};
                                                    // {21318c60-9714-3248-8ca6-41ff7324c8f2}
static const CUuuid UUID_Relay9                     = {{0x6E, 0x16, 0x3F, 0xBE, 0xB9, 0x58, 0x44, 0x4D,
                                                        0x83, 0x5C, 0xE1, 0x82, 0xAF, 0xF1, 0x99, 0x1E}};
                                                    // {6e163fbe-b958-444d-835c-e182aff1991e}
static const CUuuid UUID_Relay10                    = {{0x26, 0x3E, 0x88, 0x60, 0x7C, 0xD2, 0x61, 0x43,
                                                        0x92, 0xF6, 0xBB, 0xD5, 0x00, 0x6D, 0xFA, 0x7E}};
                                                    // {263e8860-7cd2-6143-92f6-bbd5006dfa7e}
static const CUuuid UUID_Relay11                    = {{0xB1, 0x05, 0x41, 0xE1, 0xF7, 0xC7, 0xC7, 0x4A,
                                                        0x9F, 0x64, 0xF2, 0x23, 0xBE, 0x99, 0xF1, 0xE2}};
                                                    // {b10541e1-f7c7-c74a-9f64-f223be99f1e2}
static const CUuuid UUID_OpticalFlow                = {{0x9A, 0xF0, 0x70, 0x7B, 0x8E, 0x2D, 0xD8, 0x4C,
                                                        0x8E, 0x4E, 0xB9, 0x94, 0xC8, 0x2D, 0xDC, 0x35}};
                                                    // {9af0707b-8e2d-d84c-8e4e-b994c82ddc35}
static const CUuuid UUID_Relay12                    = {{0xF8, 0xCF, 0xF9, 0x51, 0x21, 0x46, 0x8B, 0x4E,
                                                        0xB9, 0xE2, 0xFB, 0x46, 0x9E, 0x7C, 0x0D, 0xD9}};
                                                    // {f8cff951-2146-8b4e-b9e2-fb469e7c0dd9}
static const CUuuid UUID_Relay13                    = {{0xDD, 0x9E, 0x98, 0xA4, 0xF4, 0x10, 0x48, 0x44,
                                                        0x99, 0xB0, 0xE2, 0xAE, 0xB7, 0xB4, 0x32, 0xB5}};
                                                    // {dd9e98a4-f410-4844-99b0-e2aeb7b432b5}

static char* cuda_print_uuid(const CUuuid *id, char *buffer, int size)
{
    snprintf(buffer, size, "{%02x%02x%02x%02x-%02x%02x-%02x%02x-"\
                           "%02x%02x-%02x%02x%02x%02x%02x%02x}",
        id->bytes[0] & 0xFF, id->bytes[1] & 0xFF, id->bytes[2] & 0xFF, id->bytes[3] & 0xFF,
        id->bytes[4] & 0xFF, id->bytes[5] & 0xFF, id->bytes[6] & 0xFF, id->bytes[7] & 0xFF,
        id->bytes[8] & 0xFF, id->bytes[9] & 0xFF, id->bytes[10] & 0xFF, id->bytes[11] & 0xFF,
        id->bytes[12] & 0xFF, id->bytes[13] & 0xFF, id->bytes[14] & 0xFF, id->bytes[15] & 0xFF);
    return buffer;
}

int main() {
    CUresult ret;
    CUdevice dev;
    CUuuid uuid;
    unsigned int deviceNodeMask;
    char luid[64];
    char name[100];
    char buffer[128];
    int major = 0;
    int minor = 0;

    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda)
    {
        printf("Failed to load nvcuda.dll\n");
        return 1;
    }

    #define LOAD_FUNCPTR(name) if ((*(void **)(&(name)) = (void *)GetProcAddress(hCuda, #name)) == NULL) { printf("Failed to get symbol: %s\n", #name); exit(1); }

    LOAD_FUNCPTR(cuInit);
    LOAD_FUNCPTR(cuDeviceGet);
    LOAD_FUNCPTR(cuDeviceGetName);
    LOAD_FUNCPTR(cuDeviceComputeCapability);
    LOAD_FUNCPTR(cuDeviceGetUuid);
    LOAD_FUNCPTR(cuDeviceGetLuid);
    LOAD_FUNCPTR(cuGetExportTable);

    #undef LOAD_FUNCPTR

    (ret = cuInit(0)) == CUDA_SUCCESS
    ? printf("cuInit success!\n")
    : (printf("cuInit returned error: %d\n", ret), exit(1));

    (ret = cuDeviceGet(&dev, 0)) == CUDA_SUCCESS // Only run on first device for now
    ? printf("cuDeviceGet found device: %d\n", dev)
    : (printf("cuDeviceGet returned error: %d\n", ret), exit(1));

    (ret = cuDeviceGetName(&name[0], sizeof(name), dev)) == CUDA_SUCCESS
    ? printf("Running tests on :\n==============================\n%s\n", name)
    : (printf("cuDeviceGetName returned error: %d\n", ret), exit(1));

    (ret = cuDeviceComputeCapability(&major, &minor, dev)) == CUDA_SUCCESS
    ? printf("SM version: %d.%d\n==============================\n", major, minor)
    : (printf("cuDeviceComputeCapability returned error: %d\n", ret), exit(1));

    (ret = cuDeviceGetUuid(&uuid, dev)) == CUDA_SUCCESS
    ? printf("Device %d UUID: %s\n", dev, cuda_print_uuid(&uuid, buffer, sizeof(buffer)))
    : (printf("cuDeviceGetLuid returned error: %d\n", ret), exit(1));

    (ret = cuDeviceGetLuid(luid, &deviceNodeMask, dev)) == CUDA_SUCCESS
    ? printf("Device %d LUID: %02x%02x%02x%02x-%02x%02x%02x%02x\n", dev,
                (unsigned char)luid[0], (unsigned char)luid[1], (unsigned char)luid[2], (unsigned char)luid[3],
                (unsigned char)luid[4], (unsigned char)luid[5], (unsigned char)luid[6], (unsigned char)luid[7])
    : (printf("cuDeviceGetLuid returned error: %d\n", ret), exit(1));

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay1)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay1: %d\n", ret)
    : printf("UUID_Relay1 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay2)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay2: %d\n", ret)
    : printf("UUID_Relay2 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay3)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay3: %d\n", ret)
    : printf("UUID_Relay3 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_ContextStorage)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_ContextStorage: %d\n", ret)
    : printf("UUID_ContextStorage success\n");
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay5)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay5: %d\n", ret)
    : printf("UUID_Relay5 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_TlsNotifyInterface)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_TlsNotifyInterface: %d\n", ret)
    : printf("UUID_TlsNotifyInterface size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Encryption)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Encryption: %d\n", ret)
    : printf("UUID_Encryption size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay8)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay8: %d\n", ret)
    : printf("UUID_Relay8 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay9)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay9: %d\n", ret)
    : printf("UUID_Relay9 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay10)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay10: %d\n", ret)
    : printf("UUID_Relay10 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void**)&addrTable, &UUID_Relay11)) != CUDA_SUCCESS || !addrTable
        ? printf("Failed to retrieve UUID_Relay11: %d\n", ret)
        : printf("UUID_Relay11 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_OpticalFlow)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_OpticalFlow: %d\n", ret)
    : printf("UUID_OpticalFlow size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay12)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay12: %d\n", ret)
    : printf("UUID_Relay12 size: %d\n", addrTable->size);
    addrTable = NULL;

    (ret = cuGetExportTable((const void **)&addrTable, &UUID_Relay13)) != CUDA_SUCCESS || !addrTable
    ? printf("Failed to retrieve UUID_Relay13: %d\n", ret)
    : printf("UUID_Relay13 size: %d\n", addrTable->size);
    addrTable = NULL;

    return 0;
}
