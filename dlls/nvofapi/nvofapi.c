/*
 * Copyright (C) 2024 Sveinar Søpler
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "config.h"
#include <dlfcn.h>
#include <stdarg.h>
#include <assert.h>
#include <stdint.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "windef.h"
#include "winbase.h"
#include "winternl.h"
#include "wine/debug.h"
#include "wine/list.h"

#include "cuda.h"
#include "opticalflow.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvofapi);

static NV_OF_STATUS (*pNvOFAPICreateInstanceCuda)(uint32_t apiVer, void *functionList);
static NV_OF_STATUS (*pNvOFAPICreateInstanceVk)(uint32_t apiVer, void *functionList);
static NV_OF_STATUS (*pNvOFGetMaxSupportedApiVersion)(uint32_t *version);

static void *nvofapi_handle = NULL;

static BOOL load_functions(void)
{
    if (!(nvofapi_handle = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW)))
    {
        FIXME("Wine cannot find the Optical Flow library, Optical Flow is disabled.\n");
        return FALSE;
    }

    #define LOAD_FUNCPTR(f) if((*(void **)(&p##f) = dlsym(nvofapi_handle, #f)) == NULL){ERR("Can't find symbol %s\n", #f); return FALSE;}

    LOAD_FUNCPTR(NvOFAPICreateInstanceCuda);
    LOAD_FUNCPTR(NvOFAPICreateInstanceVk);
    LOAD_FUNCPTR(NvOFGetMaxSupportedApiVersion);

    #undef LOAD_FUNCPTR

    return TRUE;
}

/* Optical Flow CUDA function table */
struct ofCuda_table
{
    NV_OF_STATUS (WINAPI *nvCreateOpticalFlowCuda)(CUcontext device, NvOFHandle *hOf);
    NV_OF_STATUS (WINAPI *nvOFInitCuda)(NvOFHandle hOf, const void *initParams);
    NV_OF_STATUS (WINAPI *nvOFCreateGPUBufferCuda)(NvOFHandle hOf, const void *bufferDesc, void *bufferType, NvOFGPUBufferHandle *hOfGpuBuffer);
    CUarray (WINAPI *nvOFGPUBufferGetCUarray)(NvOFGPUBufferHandle ofGpuBuffer);
    CUdeviceptr (WINAPI *nvOFGPUBufferGetCUdeviceptr)(NvOFGPUBufferHandle ofGpuBuffer);
    NV_OF_STATUS (WINAPI *nvOFGPUBufferGetStrideInfo)(NvOFGPUBufferHandle ofGpuBuffer, void *strideInfo);
    NV_OF_STATUS (WINAPI *nvOFSetIOCudaStreams)(NvOFHandle hOf, CUstream inputStream, CUstream outputStream);
    NV_OF_STATUS (WINAPI *nvOFExecute)(NvOFHandle hOf, const void *executeInParams, void *executeOutParams);
    NV_OF_STATUS (WINAPI *nvOFDestroyGPUBufferCuda)(NvOFGPUBufferHandle buffer);
    NV_OF_STATUS (WINAPI *nvOFDestroyCuda)(NvOFHandle hOf);
    NV_OF_STATUS (WINAPI *nvOFGetLastErrorCuda)(NvOFHandle hOf, char lastError[], uint32_t *size);
    NV_OF_STATUS (WINAPI *nvOFGetCapsCuda)(NvOFHandle hOf, void *capsParam, uint32_t *capsVal, uint32_t *size);
};
static struct
{
    NV_OF_STATUS (*nvCreateOpticalFlowCuda)(CUcontext device, NvOFHandle *hOf);
    NV_OF_STATUS (*nvOFInitCuda)(NvOFHandle hOf, const void *initParams);
    NV_OF_STATUS (*nvOFCreateGPUBufferCuda)(NvOFHandle hOf, const void *bufferDesc, void *bufferType, NvOFGPUBufferHandle *hOfGpuBuffer);
    CUarray (*nvOFGPUBufferGetCUarray)(NvOFGPUBufferHandle ofGpuBuffer);
    CUdeviceptr (*nvOFGPUBufferGetCUdeviceptr)(NvOFGPUBufferHandle ofGpuBuffer);
    NV_OF_STATUS (*nvOFGPUBufferGetStrideInfo)(NvOFGPUBufferHandle ofGpuBuffer, void *strideInfo);
    NV_OF_STATUS (*nvOFSetIOCudaStreams)(NvOFHandle hOf, CUstream inputStream, CUstream outputStream);
    NV_OF_STATUS (*nvOFExecute)(NvOFHandle hOf, const void *executeInParams, void *executeOutParams);
    NV_OF_STATUS (*nvOFDestroyGPUBufferCuda)(NvOFGPUBufferHandle buffer);
    NV_OF_STATUS (*nvOFDestroyCuda)(NvOFHandle hOf);
    NV_OF_STATUS (*nvOFGetLastErrorCuda)(NvOFHandle hOf, char lastError[], uint32_t *size);
    NV_OF_STATUS (*nvOFGetCapsCuda)(NvOFHandle hOf, void *capsParam, uint32_t *capsVal, uint32_t *size);
} *ofCuda_orig = NULL;

struct ofVK_table
{
    NV_OF_STATUS (WINAPI *nvCreateOpticalFlowVk)(void *instance, void *physicalDevice, void *device, NvOFHandle *hOFInstance);
    NV_OF_STATUS (WINAPI *nvOFInitVk)(NvOFHandle hOf, const void *initParams);
    NV_OF_STATUS (WINAPI *nvOFGetSurfaceFormatCountVk)(NvOFHandle hOf, const void *bufUsage, const void *ofMode, uint32_t *const pCount);
    NV_OF_STATUS (WINAPI *nvOFGetSurfaceFormatVk)(NvOFHandle hOf, const void *bufUsage, const void *ofMode, VkFormat *const pFormat);
    NV_OF_STATUS (WINAPI *nvOFRegisterResourceVk)(NvOFHandle hOf, void *registerParams);
    NV_OF_STATUS (WINAPI *nvOFUnregisterResourceVk)(void *unregisterParams);
    NV_OF_STATUS (WINAPI *nvOFExecuteVk)(NvOFHandle hOf, const void *executeInParams, void *executeOutParams);
    NV_OF_STATUS (WINAPI *nvOFDestroyVk)(NvOFHandle hOf);
    NV_OF_STATUS (WINAPI *nvOFGetLastErrorVk)(NvOFHandle hOf, char lastError[], uint32_t *size);
    NV_OF_STATUS (WINAPI *nvOFGetCapsVk)(NvOFHandle hOf, void *capsParam, uint32_t *capsVal, uint32_t *size);
};
static struct
{
    NV_OF_STATUS (*nvCreateOpticalFlowVk)(void *instance, void *physicalDevice, void *device, void *hOFInstance);
    NV_OF_STATUS (*nvOFInitVk)(NvOFHandle hOf, const void *initParams);
    NV_OF_STATUS (*nvOFGetSurfaceFormatCountVk)(NvOFHandle hOf, const void *bufUsage, const void *ofMode, uint32_t *const pCount);
    NV_OF_STATUS (*nvOFGetSurfaceFormatVk)(NvOFHandle hOf, const void *bufUsage, const void *ofMode, VkFormat *const pFormat);
    NV_OF_STATUS (*nvOFRegisterResourceVk)(NvOFHandle hOf, void *registerParams);
    NV_OF_STATUS (*nvOFUnregisterResourceVk)(void *unregisterParams);
    NV_OF_STATUS (*nvOFExecuteVk)(NvOFHandle hOf, const void *executeInParams, void *executeOutParams);
    NV_OF_STATUS (*nvOFDestroyVk)(NvOFHandle hOf);
    NV_OF_STATUS (*nvOFGetLastErrorVk)(NvOFHandle hOf, char lastError[], uint32_t *size);
    NV_OF_STATUS (*nvOFGetCapsVk)(NvOFHandle hOf, void* capsParam, uint32_t *capsVal, uint32_t *size);
} *ofVK_orig = NULL;

static NV_OF_STATUS WINAPI nvCreateOpticalFlowCuda(CUcontext device, NvOFHandle *hOf)
{
    TRACE("(%p, %p)\n", device, hOf);
    return ofCuda_orig->nvCreateOpticalFlowCuda(device, hOf);
}

static NV_OF_STATUS WINAPI nvOFInitCuda(NvOFHandle hOf, const void *initParams)
{
    TRACE("(%p, %p)\n", hOf, initParams);
    return ofCuda_orig->nvOFInitCuda(hOf, initParams);
}

static NV_OF_STATUS WINAPI nvOFCreateGPUBufferCuda(NvOFHandle hOf, const void *bufferDesc, void *bufferType, NvOFGPUBufferHandle *hOfGpuBuffer)
{
    TRACE("(%p, %p, %p, %p)\n", hOf, bufferDesc, bufferType, hOfGpuBuffer);
    return ofCuda_orig->nvOFCreateGPUBufferCuda(hOf, bufferDesc, bufferType, hOfGpuBuffer);
}

static CUarray WINAPI nvOFGPUBufferGetCUarray(NvOFGPUBufferHandle ofGpuBuffer)
{
    TRACE("(%p)\n", ofGpuBuffer);
    return ofCuda_orig->nvOFGPUBufferGetCUarray(ofGpuBuffer);
}

static CUdeviceptr WINAPI nvOFGPUBufferGetCUdeviceptr(NvOFGPUBufferHandle ofGpuBuffer)
{
    TRACE("(%p)\n", ofGpuBuffer);
    return ofCuda_orig->nvOFGPUBufferGetCUdeviceptr(ofGpuBuffer);
}

static NV_OF_STATUS WINAPI nvOFGPUBufferGetStrideInfo(NvOFGPUBufferHandle ofGpuBuffer, void *strideInfo)
{
    TRACE("(%p, %p)\n", ofGpuBuffer, strideInfo);
    return ofCuda_orig->nvOFGPUBufferGetStrideInfo(ofGpuBuffer, strideInfo);
}

static NV_OF_STATUS WINAPI nvOFSetIOCudaStreams(NvOFHandle hOf, CUstream inputStream, CUstream outputStream)
{
    TRACE("(%p, %p, %p)\n", hOf, inputStream, outputStream);
    return ofCuda_orig->nvOFSetIOCudaStreams(hOf, inputStream, outputStream);
}

static NV_OF_STATUS WINAPI nvOFExecute(NvOFHandle hOf, const void *executeInParams, void *executeOutParams)
{
    TRACE("(%p, %p, %p)\n", hOf, executeInParams, executeOutParams);
    return ofCuda_orig->nvOFExecute(hOf, executeInParams, executeOutParams);
}

static NV_OF_STATUS WINAPI nvOFDestroyGPUBufferCuda(NvOFGPUBufferHandle buffer)
{
    TRACE("(%p)\n", buffer);
    return ofCuda_orig->nvOFDestroyGPUBufferCuda(buffer);
}

static NV_OF_STATUS WINAPI nvOFDestroyCuda(NvOFHandle hOf)
{
    TRACE("(%p)\n", hOf);
    return ofCuda_orig->nvOFDestroyCuda(hOf);
}

static NV_OF_STATUS WINAPI nvOFGetLastErrorCuda(NvOFHandle hOf, char lastError[], uint32_t *size)
{
    NV_OF_STATUS ret = ofCuda_orig->nvOFGetLastErrorCuda(hOf, lastError, size);
    TRACE("(%p, %s, %u)\n", hOf, lastError, *size);
    return ret;
}

static NV_OF_STATUS WINAPI nvOFGetCapsCuda(NvOFHandle hOf, void *capsParam, uint32_t *capsVal, uint32_t *size)
{
    TRACE("(%p, %p, %p, %p)\n", hOf, capsParam, capsVal, size);
    return ofCuda_orig->nvOFGetCapsCuda(hOf, capsParam, capsVal, size);
}

static struct ofCuda_table ofCuda_Impl =
{
    nvCreateOpticalFlowCuda,
    nvOFInitCuda,
    nvOFCreateGPUBufferCuda,
    nvOFGPUBufferGetCUarray,
    nvOFGPUBufferGetCUdeviceptr,
    nvOFGPUBufferGetStrideInfo,
    nvOFSetIOCudaStreams,
    nvOFExecute,
    nvOFDestroyGPUBufferCuda,
    nvOFDestroyCuda,
    nvOFGetLastErrorCuda,
    nvOFGetCapsCuda,
};

static NV_OF_STATUS WINAPI nvCreateOpticalFlowVk(void *instance, void *physicalDevice, void *device, NvOFHandle *hOFInstance)
{
    TRACE("(%p, %p, %p, %p)\n", instance, physicalDevice, device, hOFInstance);
    return ofVK_orig->nvCreateOpticalFlowVk(instance, physicalDevice, device, hOFInstance);
}

static NV_OF_STATUS WINAPI nvOFInitVk(NvOFHandle hOf, const void *initParams)
{
    TRACE("(%p, %p)\n", hOf, initParams);
    return ofVK_orig->nvOFInitVk(hOf, initParams);
}

static NV_OF_STATUS WINAPI nvOFGetSurfaceFormatCountVk(NvOFHandle hOf, const void *bufUsage, const void *ofMode, uint32_t *const pCount)
{
    TRACE("(%p, %p, %p, %p)\n", hOf, bufUsage, ofMode, pCount);
    return ofVK_orig->nvOFGetSurfaceFormatCountVk(hOf, bufUsage, ofMode, pCount);
}

static NV_OF_STATUS WINAPI nvOFGetSurfaceFormatVk(NvOFHandle hOf, const void *bufUsage, const void *ofMode, VkFormat *const pFormat)
{
    TRACE("(%p, %p, %p, %p)\n", hOf, bufUsage, ofMode, pFormat);
    return ofVK_orig->nvOFGetSurfaceFormatVk(hOf, bufUsage, ofMode, pFormat);
}

static NV_OF_STATUS WINAPI nvOFRegisterResourceVk(NvOFHandle hOf, void *registerParams)
{
    TRACE("(%p, %p)\n", hOf, registerParams);
    return ofVK_orig->nvOFRegisterResourceVk(hOf, registerParams);
}

static NV_OF_STATUS WINAPI nvOFUnregisterResourceVk(void *unregisterParams)
{
    TRACE("(%p)\n", unregisterParams);
    return ofVK_orig->nvOFUnregisterResourceVk(unregisterParams);
}

static NV_OF_STATUS WINAPI nvOFExecuteVk(NvOFHandle hOf, const void *executeInParams, void *executeOutParams)
{
    TRACE("(%p, %p, %p)\n", hOf, executeInParams, executeOutParams);
    return ofVK_orig->nvOFExecuteVk(hOf, executeInParams, executeOutParams);
}

static NV_OF_STATUS WINAPI nvOFDestroyVk(NvOFHandle hOf)
{
    TRACE("(%p)\n", hOf);
    return ofVK_orig->nvOFDestroyVk(hOf);
}

static NV_OF_STATUS WINAPI nvOFGetLastErrorVk(NvOFHandle hOf, char lastError[], uint32_t *size)
{
    NV_OF_STATUS ret = ofVK_orig->nvOFGetLastErrorVk(hOf, lastError, size);
    TRACE("(%p, %s, %u)\n", hOf, lastError, *size);
    return ret;
}

static NV_OF_STATUS WINAPI nvOFGetCapsVk(NvOFHandle hOf, void *capsParam, uint32_t *capsVal, uint32_t *size)
{
    TRACE("(%p, %p, %p, %p)\n", hOf, capsParam, capsVal, size);
    return ofVK_orig->nvOFGetCapsVk(hOf, capsParam, capsVal, size);
}

static struct ofVK_table ofVK_Impl =
{
    nvCreateOpticalFlowVk,
    nvOFInitVk,
    nvOFGetSurfaceFormatCountVk,
    nvOFGetSurfaceFormatVk,
    nvOFRegisterResourceVk,
    nvOFUnregisterResourceVk,
    nvOFExecuteVk,
    nvOFDestroyVk,
    nvOFGetLastErrorVk,
    nvOFGetCapsVk,
};

/* Main functions */
NV_OF_STATUS WINAPI wine_NvOFAPICreateInstanceCuda(uint32_t apiVer, void *functionList)
{
    ofCuda_orig = HeapAlloc(GetProcessHeap(), 0, sizeof(*ofCuda_orig));

    TRACE("(%u, %p)\n", apiVer, functionList);
    NV_OF_STATUS ret = pNvOFAPICreateInstanceCuda(apiVer, ofCuda_orig);
    if(!ret) memcpy(functionList, &ofCuda_Impl, sizeof(ofCuda_Impl));
    else
    {
        TRACE("Failed to create instanceCuda: %d\n", ret);
        HeapFree(GetProcessHeap(), 0, ofCuda_orig);
    }
    return ret;
}

NV_OF_STATUS WINAPI wine_NvOFAPICreateInstanceVk(uint32_t apiVer, void *functionList)
{
    ofVK_orig = HeapAlloc(GetProcessHeap(), 0, sizeof(*ofVK_orig));

    TRACE("(%u, %p)\n", apiVer, functionList);
    NV_OF_STATUS ret = pNvOFAPICreateInstanceVk(apiVer, functionList);
    if(!ret) memcpy(functionList, &ofVK_Impl, sizeof(ofVK_Impl));
    else
    {
        TRACE("Failed to create instanceVK: %d\n", ret);
        HeapFree(GetProcessHeap(), 0, ofVK_orig);
    }
    return ret;
}

NV_OF_STATUS WINAPI wine_NvOFGetMaxSupportedApiVersion(uint32_t *version)
{
    TRACE("(%p)\n", version);
    return pNvOFGetMaxSupportedApiVersion(version);
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    TRACE("(%p, %u, %p)\n", instance, reason, reserved);

    switch (reason)
    {
        case DLL_PROCESS_ATTACH:
            if (!load_functions()) return FALSE;
            break;
        case DLL_PROCESS_DETACH:
            if (reserved) break;
            if (nvofapi_handle) dlclose(nvofapi_handle);
            break;
        case DLL_THREAD_ATTACH:
            break;
        case DLL_THREAD_DETACH:
            break;
    }

    return TRUE;
}
