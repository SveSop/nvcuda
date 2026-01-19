/*
 * Copyright (C) 2014-2015 Michael Müller
 * Copyright (C) 2014-2015 Sebastian Lackner
 * Copyright (C) 2022-2025 Sveinar Søpler
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

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "windef.h"
#include "winbase.h"
#include "winternl.h"
#include "wine/debug.h"
#include "wine/list.h"
#include "wine/wgl.h"
#include "cuda.h"
#include "nvcuda.h"
#include "d3d9.h"
#include "d3d11.h"

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define DEV_PTR "%llu"
#else
#define DEV_PTR "%u"
#endif

WINE_DEFAULT_DEBUG_CHANNEL(nvcuda);

static struct          list stream_callbacks         = LIST_INIT( stream_callbacks );
static pthread_mutex_t stream_callback_mutex         = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  stream_callback_request       = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  stream_callback_reply         = PTHREAD_COND_INITIALIZER;
static const char      devpropkey_gpu_vulkan_uuidA[] = "Properties\\{233A9EF3-AFC4-4ABD-B564-C32F21F1535C}\\0002";
static const char      devpropkey_gpu_luidA[]        = "Properties\\{60B193CB-5276-4D0F-96FC-F173ABAD3EC6}\\0002";
static LONG            num_stream_callbacks;

struct stream_callback_entry
{
    struct list entry;
    enum
    {
        STREAM_CALLBACK_ABANDONED,
        STREAM_CALLBACK_PENDING,
        STREAM_CALLBACK_EXECUTED
    } status;
    void (WINAPI *callback)(CUstream hStream, CUresult status, void *userData);
    struct
    {
        CUstream stream;
        CUresult status;
        void *userdata;
    } args;
};

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

static inline UINT asciiz_to_unicode( WCHAR *dst, const char *src )
{
    WCHAR *p = dst;
    while ((*p++ = *src++));
    return (p - dst) * sizeof(WCHAR);
}

static HKEY reg_open_key( HKEY root, const WCHAR *name, ULONG name_len )
{
    UNICODE_STRING nameW = { name_len, name_len, (WCHAR *)name };
    OBJECT_ATTRIBUTES attr;
    HANDLE ret;

    attr.Length = sizeof(attr);
    attr.RootDirectory = root;
    attr.ObjectName = &nameW;
    attr.Attributes = 0;
    attr.SecurityDescriptor = NULL;
    attr.SecurityQualityOfService = NULL;

    if (NtOpenKeyEx( &ret, MAXIMUM_ALLOWED, &attr, 0 )) return 0;
    return ret;
}

static HKEY reg_open_ascii_key( HKEY root, const char *name )
{
    WCHAR nameW[MAX_PATH];
    return reg_open_key( root, nameW, asciiz_to_unicode( nameW, name ) - sizeof(WCHAR) );
}

static ULONG query_reg_value( HKEY hkey, const WCHAR *name,
                       KEY_VALUE_PARTIAL_INFORMATION *info, ULONG size )
{
    unsigned int name_size = name ? lstrlenW( name ) * sizeof(WCHAR) : 0;
    UNICODE_STRING nameW = { name_size, name_size, (WCHAR *)name };

    if (NtQueryValueKey( hkey, &nameW, KeyValuePartialInformation,
                         info, size, &size ))
        return 0;

    return size - FIELD_OFFSET(KEY_VALUE_PARTIAL_INFORMATION, Data);
}

/* CUDA Funcion declarations */
static CUresult (*pcuArray3DCreate)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
static CUresult (*pcuArray3DCreate_v2)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
static CUresult (*pcuArray3DGetDescriptor)(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
static CUresult (*pcuArray3DGetDescriptor_v2)(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
static CUresult (*pcuArrayCreate)(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
static CUresult (*pcuArrayCreate_v2)(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
static CUresult (*pcuArrayDestroy)(CUarray hArray);
static CUresult (*pcuArrayGetDescriptor)(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
static CUresult (*pcuArrayGetDescriptor_v2)(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
static CUresult (*pcuCtxAttach)(CUcontext *pctx, unsigned int flags);
static CUresult (*pcuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
static CUresult (*pcuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);
static CUresult (*pcuCtxDestroy)(CUcontext ctx);
static CUresult (*pcuCtxDestroy_v2)(CUcontext ctx);
static CUresult (*pcuCtxDetach)(CUcontext ctx);
static CUresult (*pcuCtxDisablePeerAccess)(CUcontext peerContext);
static CUresult (*pcuCtxEnablePeerAccess)(CUcontext peerContext, unsigned int Flags);
static CUresult (*pcuCtxGetApiVersion)(CUcontext ctx, unsigned int *version);
static CUresult (*pcuCtxGetCacheConfig)(CUfunc_cache *pconfig);
static CUresult (*pcuCtxGetCurrent)(CUcontext *pctx);
static CUresult (*pcuCtxGetDevice)(CUdevice *device);
static CUresult (*pcuCtxGetLimit)(size_t *pvalue, CUlimit limit);
static CUresult (*pcuCtxGetSharedMemConfig)(CUsharedconfig *pConfig);
static CUresult (*pcuCtxGetStreamPriorityRange)(int *leastPriority, int *greatestPriority);
static CUresult (*pcuCtxPopCurrent)(CUcontext *pctx);
static CUresult (*pcuCtxPopCurrent_v2)(CUcontext *pctx);
static CUresult (*pcuCtxPushCurrent)(CUcontext ctx);
static CUresult (*pcuCtxPushCurrent_v2)(CUcontext ctx);
static CUresult (*pcuCtxSetCacheConfig)(CUfunc_cache config);
static CUresult (*pcuCtxSetCurrent)(CUcontext ctx);
static CUresult (*pcuCtxSetLimit)(CUlimit limit, size_t value);
static CUresult (*pcuCtxSetSharedMemConfig)(CUsharedconfig config);
static CUresult (*pcuCtxSynchronize)(void);
static CUresult (*pcuDeviceCanAccessPeer)(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
static CUresult (*pcuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
static CUresult (*pcuDeviceGet)(CUdevice *device, int ordinal);
static CUresult (*pcuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
static CUresult (*pcuDeviceGetByPCIBusId)(CUdevice *dev, const char *pciBusId);
static CUresult (*pcuDeviceGetCount)(int *count);
static CUresult (*pcuDeviceGetName)(char *name, int len, CUdevice dev);
static CUresult (*pcuDeviceGetPCIBusId)(char *pciBusId, int len, CUdevice dev);
static CUresult (*pcuDeviceGetProperties)(CUdevprop *prop, CUdevice dev);
static CUresult (*pcuDeviceTotalMem)(unsigned int *bytes, CUdevice dev);
static CUresult (*pcuDeviceTotalMem_v2)(size_t *bytes, CUdevice dev);
static CUresult (*pcuDriverGetVersion)(int *version);
static CUresult (*pcuEventCreate)(CUevent *phEvent, unsigned int Flags);
static CUresult (*pcuEventDestroy)(CUevent hEvent);
static CUresult (*pcuEventDestroy_v2)(CUevent hEvent);
static CUresult (*pcuEventElapsedTime)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
static CUresult (*pcuEventQuery)(CUevent hEvent);
static CUresult (*pcuEventRecord)(CUevent hEvent, CUstream hStream);
static CUresult (*pcuEventSynchronize)(CUevent hEvent);
static CUresult (*pcuFuncGetAttribute)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
static CUresult (*pcuFuncSetAttribute)(CUfunction hfunc, CUfunction_attribute attrib, int value);
static CUresult (*pcuFuncSetBlockShape)(CUfunction hfunc, int x, int y, int z);
static CUresult (*pcuFuncSetCacheConfig)(CUfunction hfunc, CUfunc_cache config);
static CUresult (*pcuFuncSetSharedMemConfig)(CUfunction hfunc, CUsharedconfig config);
static CUresult (*pcuFuncSetSharedSize)(CUfunction hfunc, unsigned int bytes);
static CUresult (*pcuGLCtxCreate)(CUcontext *pCtx, unsigned int Flags, CUdevice device);
static CUresult (*pcuGLCtxCreate_v2)(CUcontext *pCtx, unsigned int Flags, CUdevice device);
static CUresult (*pcuGLGetDevices)(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                                   unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
static CUresult (*pcuGLInit)(void);
static CUresult (*pcuGLMapBufferObject)(CUdeviceptr *dptr, size_t *size, GLuint buffer);
static CUresult (*pcuGLMapBufferObjectAsync)(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream);
static CUresult (*pcuGLMapBufferObjectAsync_v2)(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream);
static CUresult (*pcuGLMapBufferObject_v2)(CUdeviceptr *dptr, size_t *size, GLuint buffer);
static CUresult (*pcuGLRegisterBufferObject)(GLuint buffer);
static CUresult (*pcuGLSetBufferObjectMapFlags)(GLuint buffer, unsigned int Flags);
static CUresult (*pcuGLUnmapBufferObject)(GLuint buffer);
static CUresult (*pcuGLUnmapBufferObjectAsync)(GLuint buffer, CUstream hStream);
static CUresult (*pcuGLUnregisterBufferObject)(GLuint buffer);
static CUresult (*pcuGetErrorName)(CUresult error, const char **pStr);
static CUresult (*pcuGetErrorString)(CUresult error, const char **pStr);
static CUresult (*pcuGetExportTable)(const void** table, const CUuuid *id);
static CUresult (*pcuGraphicsGLRegisterBuffer)(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
static CUresult (*pcuGraphicsGLRegisterImage)(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);
static CUresult (*pcuGraphicsMapResources)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
static CUresult (*pcuGraphicsResourceGetMappedMipmappedArray)(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource);
static CUresult (*pcuGraphicsResourceGetMappedPointer)(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, CUgraphicsResource resource);
static CUresult (*pcuGraphicsResourceGetMappedPointer_v2)(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource);
static CUresult (*pcuGraphicsResourceSetMapFlags)(CUgraphicsResource resource, unsigned int flags);
static CUresult (*pcuGraphicsSubResourceGetMappedArray)(CUarray *pArray, CUgraphicsResource resource,
                                                        unsigned int arrayIndex, unsigned int mipLevel);
static CUresult (*pcuGraphicsUnmapResources)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
static CUresult (*pcuGraphicsUnregisterResource)(CUgraphicsResource resource);
static CUresult (*pcuInit)(unsigned int flags);
static CUresult (*pcuIpcCloseMemHandle)(CUdeviceptr_v2 dptr);
static CUresult (*pcuIpcGetEventHandle)(CUipcEventHandle *pHandle, CUevent event);
static CUresult (*pcuIpcGetMemHandle)(CUipcMemHandle *pHandle, CUdeviceptr_v2 dptr);
static CUresult (*pcuIpcOpenEventHandle)(CUevent *phEvent, CUipcEventHandle handle);
static CUresult (*pcuIpcOpenMemHandle)(CUdeviceptr_v2 *pdptr, CUipcMemHandle handle, unsigned int Flags);
static CUresult (*pcuLaunch)(CUfunction f);
static CUresult (*pcuLaunchGrid)(CUfunction f, int grid_width, int grid_height);
static CUresult (*pcuLaunchGridAsync)(CUfunction f, int grid_width, int grid_height, CUstream hStream);
static CUresult (*pcuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
static CUresult (*pcuLinkAddData)(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                                  unsigned int numOptions, CUjit_option *options, void **optionValues);
static CUresult (*pcuLinkComplete)(CUlinkState state, void **cubinOut, size_t *sizeOut);
static CUresult (*pcuLinkCreate)(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
static CUresult (*pcuLinkDestroy)(CUlinkState state);
static CUresult (*pcuMemAlloc)(CUdeviceptr *dptr, unsigned int bytesize);
static CUresult (*pcuMemAllocHost)(void **pp, unsigned int bytesize);
static CUresult (*pcuMemAllocHost_v2)(void **pp, size_t bytesize);
static CUresult (*pcuMemAllocManaged)(CUdeviceptr_v2 *dptr, size_t bytesize, unsigned int flags);
static CUresult (*pcuMemAllocPitch)(CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
static CUresult (*pcuMemAllocPitch_v2)(CUdeviceptr_v2 *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
static CUresult (*pcuMemAlloc_v2)(CUdeviceptr_v2 *dptr, size_t bytesize);
static CUresult (*pcuMemFree)(CUdeviceptr_v1 dptr);
static CUresult (*pcuMemFreeHost)(void *p);
static CUresult (*pcuMemFree_v2)(CUdeviceptr_v2 dptr);
static CUresult (*pcuMemGetAddressRange)(CUdeviceptr_v1 *pbase, unsigned int *psize, CUdeviceptr dptr);
static CUresult (*pcuMemGetAddressRange_v2)(CUdeviceptr_v2 *pbase, size_t *psize, CUdeviceptr_v2 dptr);
static CUresult (*pcuMemGetInfo)(unsigned int *free, unsigned int *total);
static CUresult (*pcuMemGetInfo_v2)(size_t *free, size_t *total);
static CUresult (*pcuMemHostAlloc)(void **pp, size_t bytesize, unsigned int Flags);
static CUresult (*pcuMemHostGetDevicePointer)(CUdeviceptr *pdptr, void *p, unsigned int Flags);
static CUresult (*pcuMemHostGetDevicePointer_v2)(CUdeviceptr_v2 *pdptr, void *p, unsigned int Flags);
static CUresult (*pcuMemHostGetFlags)(unsigned int *pFlags, void *p);
static CUresult (*pcuMemHostRegister)(void *p, size_t bytesize, unsigned int Flags);
static CUresult (*pcuMemHostUnregister)(void *p);
static CUresult (*pcuMemcpy)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount);
static CUresult (*pcuMemcpy2D)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy2DAsync)(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy2DAsync_v2)(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy2DUnaligned)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy2DUnaligned_v2)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy2D_v2)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy3D)(const CUDA_MEMCPY3D *pCopy);
static CUresult (*pcuMemcpy3DAsync)(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy3DAsync_v2)(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy3DPeer)(const CUDA_MEMCPY3D_PEER *pCopy);
static CUresult (*pcuMemcpy3DPeerAsync)(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy3D_v2)(const CUDA_MEMCPY3D *pCopy);
static CUresult (*pcuMemcpyAsync)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyAtoA)(CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
static CUresult (*pcuMemcpyAtoA_v2)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyAtoD)(CUdeviceptr_v1 dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
static CUresult (*pcuMemcpyAtoD_v2)(CUdeviceptr_v2 dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyAtoH)(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
static CUresult (*pcuMemcpyAtoHAsync)(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyAtoHAsync_v2)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyAtoH_v2)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyDtoA)(CUarray dstArray, unsigned int dstOffset, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
static CUresult (*pcuMemcpyDtoA_v2)(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyDtoD)(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
static CUresult (*pcuMemcpyDtoDAsync)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoDAsync_v2)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoD_v2)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyDtoH)(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
static CUresult (*pcuMemcpyDtoHAsync)(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoHAsync_v2)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyHtoA)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
static CUresult (*pcuMemcpyHtoAAsync)(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoAAsync_v2)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoA_v2)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
static CUresult (*pcuMemcpyHtoD)(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount);
static CUresult (*pcuMemcpyHtoDAsync)(CUdeviceptr_v1 dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoDAsync_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoD_v2)(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount);
static CUresult (*pcuMemcpyPeer)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
static CUresult (*pcuMemcpyPeerAsync)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                      CUcontext srcContext, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemsetD16)(CUdeviceptr_v1 dstDevice, unsigned short us, unsigned int N);
static CUresult (*pcuMemsetD16Async)(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD16_v2)(CUdeviceptr dstDevice, unsigned short us, size_t N);
static CUresult (*pcuMemsetD2D16)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
static CUresult (*pcuMemsetD2D16Async)(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D16_v2)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
static CUresult (*pcuMemsetD2D32)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
static CUresult (*pcuMemsetD2D32Async)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D32_v2)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
static CUresult (*pcuMemsetD2D8)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
static CUresult (*pcuMemsetD2D8Async)(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D8_v2)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
static CUresult (*pcuMemsetD32)(CUdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N);
static CUresult (*pcuMemsetD32Async)(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD32_v2)(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N);
static CUresult (*pcuMemsetD8)(CUdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N);
static CUresult (*pcuMemsetD8Async)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD8_v2)(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N);
static CUresult (*pcuMipmappedArrayCreate)(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                           unsigned int numMipmapLevels);
static CUresult (*pcuMipmappedArrayDestroy)(CUmipmappedArray hMipmappedArray);
static CUresult (*pcuMipmappedArrayGetLevel)(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
static CUresult (*pcuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
static CUresult (*pcuModuleGetGlobal)(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
static CUresult (*pcuModuleGetGlobal_v2)(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name);
static CUresult (*pcuModuleGetSurfRef)(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
static CUresult (*pcuModuleGetTexRef)(CUtexref *pTexRef, CUmodule hmod, const char *name);
static CUresult (*pcuModuleLoad)(CUmodule *module, const char *fname);
static CUresult (*pcuModuleLoadData)(CUmodule *module, const void *image);
static CUresult (*pcuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
static CUresult (*pcuModuleLoadFatBinary)(CUmodule *module, const void *fatCubin);
static CUresult (*pcuModuleUnload)(CUmodule hmod);
static CUresult (*pcuParamSetSize)(CUfunction hfunc, unsigned int numbytes);
static CUresult (*pcuParamSetTexRef)(CUfunction hfunc, int texunit, CUtexref hTexRef);
static CUresult (*pcuParamSetf)(CUfunction hfunc, int offset, float value);
static CUresult (*pcuParamSeti)(CUfunction hfunc, int offset, unsigned int value);
static CUresult (*pcuParamSetv)(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
static CUresult (*pcuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
static CUresult (*pcuPointerSetAttribute)(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr);
static CUresult (*pcuProfilerStart)(void);
static CUresult (*pcuProfilerStop)(void);
static CUresult (*pcuStreamAddCallback)(CUstream hStream, void *callback, void *userData, unsigned int flags);
static CUresult (*pcuStreamAttachMemAsync)(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
static CUresult (*pcuStreamCreate)(CUstream *phStream, unsigned int Flags);
static CUresult (*pcuStreamCreateWithPriority)(CUstream *phStream, unsigned int flags, int priority);
static CUresult (*pcuStreamDestroy)(CUstream hStream);
static CUresult (*pcuStreamDestroy_v2)(CUstream hStream);
static CUresult (*pcuStreamGetFlags)(CUstream hStream, unsigned int *flags);
static CUresult (*pcuStreamGetPriority)(CUstream hStream, int *priority);
static CUresult (*pcuStreamQuery)(CUstream hStream);
static CUresult (*pcuStreamSynchronize)(CUstream hStream);
static CUresult (*pcuStreamWaitEvent)(CUstream hStream, CUevent hEvent, unsigned int Flags);
static CUresult (*pcuSurfObjectCreate)(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc);
static CUresult (*pcuSurfObjectDestroy)(CUsurfObject surfObject);
static CUresult (*pcuSurfObjectGetResourceDesc)(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject);
static CUresult (*pcuSurfRefGetArray)(CUarray *phArray, CUsurfref hSurfRef);
static CUresult (*pcuSurfRefSetArray)(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
static CUresult (*pcuTexObjectCreate)(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                                      const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
static CUresult (*pcuTexObjectDestroy)(CUtexObject texObject);
static CUresult (*pcuTexObjectGetResourceDesc)(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject);
static CUresult (*pcuTexObjectGetResourceViewDesc)(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject);
static CUresult (*pcuTexObjectGetTextureDesc)(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject);
static CUresult (*pcuTexRefCreate)(CUtexref *pTexRef);
static CUresult (*pcuTexRefDestroy)(CUtexref hTexRef);
static CUresult (*pcuTexRefGetAddress)(CUdeviceptr *pdptr, CUtexref hTexRef);
static CUresult (*pcuTexRefGetAddressMode)(CUaddress_mode *pam, CUtexref hTexRef, int dim);
static CUresult (*pcuTexRefGetAddress_v2)(CUdeviceptr *pdptr, CUtexref hTexRef);
static CUresult (*pcuTexRefGetArray)(CUarray *phArray, CUtexref hTexRef);
static CUresult (*pcuTexRefGetFilterMode)(CUfilter_mode *pfm, CUtexref hTexRef);
static CUresult (*pcuTexRefGetFlags)(unsigned int *pFlags, CUtexref hTexRef);
static CUresult (*pcuTexRefGetFormat)(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef);
static CUresult (*pcuTexRefGetMaxAnisotropy)(int *pmaxAniso, CUtexref hTexRef);
static CUresult (*pcuTexRefGetMipmapFilterMode)(CUfilter_mode *pfm, CUtexref hTexRef);
static CUresult (*pcuTexRefGetMipmapLevelBias)(float *pbias, CUtexref hTexRef);
static CUresult (*pcuTexRefGetMipmapLevelClamp)(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef);
static CUresult (*pcuTexRefGetMipmappedArray)(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef);
static CUresult (*pcuTexRefSetAddress)(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr_v1 dptr, size_t bytes);
static CUresult (*pcuTexRefSetAddress2D)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
static CUresult (*pcuTexRefSetAddress2D_v2)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
static CUresult (*pcuTexRefSetAddress2D_v3)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
static CUresult (*pcuTexRefSetAddressMode)(CUtexref hTexRef, int dim, CUaddress_mode am);
static CUresult (*pcuTexRefSetAddress_v2)(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
static CUresult (*pcuTexRefSetArray)(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
static CUresult (*pcuTexRefSetFilterMode)(CUtexref hTexRef, CUfilter_mode fm);
static CUresult (*pcuTexRefSetFlags)(CUtexref hTexRef, unsigned int Flags);
static CUresult (*pcuTexRefSetFormat)(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
static CUresult (*pcuTexRefSetMaxAnisotropy)(CUtexref hTexRef, unsigned int maxAniso);
static CUresult (*pcuTexRefSetMipmapFilterMode)(CUtexref hTexRef, CUfilter_mode fm);
static CUresult (*pcuTexRefSetMipmapLevelBias)(CUtexref hTexRef, float bias);
static CUresult (*pcuTexRefSetMipmapLevelClamp)(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
static CUresult (*pcuTexRefSetMipmappedArray)(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
static CUresult (*pcuProfilerInitialize)(const char *configFile, const char *outputFile, void *outputMode);
static CUresult (*pcuLinkAddFile)(void *state, void *type, const char *path, unsigned int numOptions, void *options, void **optionValues);

/* CUDA 6.5 */
static CUresult (*pcuGLGetDevices_v2)(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                                      unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
static CUresult (*pcuGraphicsResourceSetMapFlags_v2)(CUgraphicsResource resource, unsigned int flags);
static CUresult (*pcuLinkAddData_v2)(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                                     unsigned int numOptions, CUjit_option *options, void **optionValues);
static CUresult (*pcuLinkCreate_v2)(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
static CUresult (*pcuMemHostRegister_v2)(void *p, size_t bytesize, unsigned int Flags);
static CUresult (*pcuOccupancyMaxActiveBlocksPerMultiprocessor)(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
static CUresult (*pcuOccupancyMaxPotentialBlockSize)(int *minGridSize, int *blockSize, CUfunction func,
                                                     void *blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
static CUresult (*pcuLinkAddFile_v2)(void *state, void *type, const char *path, unsigned int numOptions, void *options, void **optionValues);

/* CUDA 7.0 */
static CUresult (*pcuCtxGetFlags)(unsigned int *flags);
static CUresult (*pcuDevicePrimaryCtxGetState)(CUdevice dev, unsigned int *flags, int *active);
static CUresult (*pcuDevicePrimaryCtxRelease)(CUdevice dev);
static CUresult (*pcuDevicePrimaryCtxReset)(CUdevice dev);
static CUresult (*pcuDevicePrimaryCtxRetain)(CUcontext *pctx, CUdevice dev);
static CUresult (*pcuDevicePrimaryCtxSetFlags)(CUdevice dev, unsigned int flags);
static CUresult (*pcuEventRecord_ptsz)(CUevent hEvent, CUstream hStream);
static CUresult (*pcuGLMapBufferObjectAsync_v2_ptsz)(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream);
static CUresult (*pcuGLMapBufferObject_v2_ptds)(CUdeviceptr *dptr, size_t *size, GLuint buffer);
static CUresult (*pcuGraphicsMapResources_ptsz)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
static CUresult (*pcuGraphicsUnmapResources_ptsz)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
static CUresult (*pcuLaunchKernel_ptsz)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
static CUresult (*pcuMemcpy2DAsync_v2_ptsz)(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy2DUnaligned_v2_ptds)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy2D_v2_ptds)(const CUDA_MEMCPY2D *pCopy);
static CUresult (*pcuMemcpy3DAsync_v2_ptsz)(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy3DPeerAsync_ptsz)(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
static CUresult (*pcuMemcpy3DPeer_ptds)(const CUDA_MEMCPY3D_PEER *pCopy);
static CUresult (*pcuMemcpy3D_v2_ptds)(const CUDA_MEMCPY3D *pCopy);
static CUresult (*pcuMemcpyAsync_ptsz)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyAtoA_v2_ptds)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyAtoD_v2_ptds)(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyAtoHAsync_v2_ptsz)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyAtoH_v2_ptds)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
static CUresult (*pcuMemcpyDtoA_v2_ptds)(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyDtoDAsync_v2_ptsz)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoD_v2_ptds)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyDtoHAsync_v2_ptsz)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyDtoH_v2_ptds)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount);
static CUresult (*pcuMemcpyHtoAAsync_v2_ptsz)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoA_v2_ptds)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
static CUresult (*pcuMemcpyHtoDAsync_v2_ptsz)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyHtoD_v2_ptds)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
static CUresult (*pcuMemcpyPeerAsync_ptsz)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                           CUcontext srcContext, size_t ByteCount, CUstream hStream);
static CUresult (*pcuMemcpyPeer_ptds)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
static CUresult (*pcuMemcpy_ptds)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount);
static CUresult (*pcuMemsetD16Async_ptsz)(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD16_v2_ptds)(CUdeviceptr dstDevice, unsigned short us, size_t N);
static CUresult (*pcuMemsetD2D16Async_ptsz)(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D16_v2_ptds)(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
static CUresult (*pcuMemsetD2D32Async_ptsz)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D32_v2_ptds)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
static CUresult (*pcuMemsetD2D8Async_ptsz)(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
static CUresult (*pcuMemsetD2D8_v2_ptds)(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
static CUresult (*pcuMemsetD32Async_ptsz)(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD32_v2_ptds)(CUdeviceptr dstDevice, unsigned int ui, size_t N);
static CUresult (*pcuMemsetD8Async_ptsz)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
static CUresult (*pcuMemsetD8_v2_ptds)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
static CUresult (*pcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(int *numBlocks, CUfunction func, int blockSize,
                                                                         size_t dynamicSMemSize, unsigned int flags);
static CUresult (*pcuOccupancyMaxPotentialBlockSizeWithFlags)(int *minGridSize, int *blockSize, CUfunction func, void *blockSizeToDynamicSMemSize,
                                                              size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
static CUresult (*pcuPointerGetAttributes)(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
static CUresult (*pcuStreamAddCallback_ptsz)(CUstream hStream, void *callback, void *userData, unsigned int flags);
static CUresult (*pcuStreamAttachMemAsync_ptsz)(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
static CUresult (*pcuStreamGetFlags_ptsz)(CUstream hStream, unsigned int *flags);
static CUresult (*pcuStreamGetPriority_ptsz)(CUstream hStream, int *priority);
static CUresult (*pcuStreamQuery_ptsz)(CUstream hStream);
static CUresult (*pcuStreamSynchronize_ptsz)(CUstream hStream);
static CUresult (*pcuStreamWaitEvent_ptsz)(CUstream hStream, CUevent hEvent, unsigned int Flags);

/* Cuda 8.0 */
static CUresult (*pcuDeviceGetP2PAttribute)(int *value, void *attrib, CUdevice_v1 srcDevice, CUdevice_v1 dstDevice);
static CUresult (*pcuTexRefSetBorderColor)(CUtexref hTexRef, float *pBorderColor);
static CUresult (*pcuTexRefGetBorderColor)(float *pBorderColor, CUtexref hTexRef);
static CUresult (*pcuStreamWaitValue32)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
static CUresult (*pcuStreamWriteValue32)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
static CUresult (*pcuStreamWaitValue64)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
static CUresult (*pcuStreamWriteValue64)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
static CUresult (*pcuStreamBatchMemOp)(CUstream stream, unsigned int count, void *paramArray, unsigned int flags);
static CUresult (*pcuMemAdvise)(CUdeviceptr_v2 devPtr, size_t count, void *advice, CUdevice_v1 device);
static CUresult (*pcuMemPrefetchAsync)(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream);
static CUresult (*pcuMemPrefetchAsync_ptsz)(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream);
static CUresult (*pcuMemRangeGetAttribute)(void *data, size_t dataSize, void *attribute, CUdeviceptr_v2 devPtr, size_t count);
static CUresult (*pcuMemRangeGetAttributes)(void **data, size_t *dataSizes, void *attributes, size_t numAttributes, CUdeviceptr_v2 devPtr, size_t count);

/* Cuda 9.0 */
static CUresult (*pcuLaunchCooperativeKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
static CUresult (*pcuLaunchCooperativeKernel_ptsz)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
static CUresult (*pcuLaunchCooperativeKernelMultiDevice)(void *launchParamsList, unsigned int numDevices, unsigned int flags);
static CUresult (*pcuStreamGetCtx)(CUstream hStream, CUcontext *pctx);
static CUresult (*pcuStreamGetCtx_ptsz)(CUstream hStream, CUcontext *pctx);
static CUresult (*pcuStreamWaitValue32_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
static CUresult (*pcuStreamWriteValue32_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
static CUresult (*pcuStreamWaitValue64_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
static CUresult (*pcuStreamWriteValue64_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);

/* Cuda 10.0 */
static CUresult (*pcuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);
static CUresult (*pcuDeviceGetLuid)(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
static CUresult (*pcuStreamIsCapturing)(CUstream hStream, CUstreamCaptureStatus *captureStatus);
static CUresult (*pcuStreamIsCapturing_ptsz)(CUstream hStream, CUstreamCaptureStatus *captureStatus);
static CUresult (*pcuGraphCreate)(CUgraph *phGraph, unsigned int flags);
static CUresult (*pcuGraphAddMemcpyNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
static CUresult (*pcuGraphAddMemsetNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *memsetParams, CUcontext ctx);
static CUresult (*pcuGraphAddKernelNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *nodeParams);
static CUresult (*pcuGraphAddHostNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *nodeParams);
static CUresult (*pcuGraphGetNodes)(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
static CUresult (*pcuGraphInstantiate)(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
static CUresult (*pcuGraphClone)(CUgraph *phGraphClone, CUgraph originalGraph);
static CUresult (*pcuGraphLaunch)(CUgraphExec hGraphExec, CUstream hStream);
static CUresult (*pcuGraphLaunch_ptsz)(CUgraphExec hGraphExec, CUstream hStream);
static CUresult (*pcuGraphExecKernelNodeSetParams)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_NODE_PARAMS *nodeParams);
static CUresult (*pcuStreamBeginCapture_v2)(CUstream hStream, CUstreamCaptureMode mode);
static CUresult (*pcuStreamBeginCapture_v2_ptsz)(CUstream hStream, CUstreamCaptureMode mode);
static CUresult (*pcuStreamEndCapture)(CUstream hStream, CUgraph *phGraph);
static CUresult (*pcuStreamEndCapture_ptsz)(CUstream hStream, CUgraph *phGraph);
static CUresult (*pcuGraphDestroyNode)(CUgraphNode hNode);
static CUresult (*pcuGraphDestroy)(CUgraph hGraph);
static CUresult (*pcuGraphExecDestroy)(CUgraphExec hGraphExec);
static CUresult (*pcuMemGetAllocationGranularity)(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
static CUresult (*pcuLaunchHostFunc)(CUstream hStream, CUhostFn fn, void *userData);
static CUresult (*pcuLaunchHostFunc_ptsz)(CUstream hStream, CUhostFn fn, void *userData);
static CUresult (*pcuImportExternalMemory)(void *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
static CUresult (*pcuExternalMemoryGetMappedBuffer)(CUdeviceptr_v2 *devPtr, void *extMem, const void *bufferDesc);
static CUresult (*pcuExternalMemoryGetMappedMipmappedArray)(CUmipmappedArray *mipmap, void *extMem, const void *mipmapDesc);
static CUresult (*pcuDestroyExternalMemory)(void *extMem);
static CUresult (*pcuImportExternalSemaphore)(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
static CUresult (*pcuSignalExternalSemaphoresAsync)(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream);
static CUresult (*pcuSignalExternalSemaphoresAsync_ptsz)(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream);
static CUresult (*pcuWaitExternalSemaphoresAsync)(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream);
static CUresult (*pcuWaitExternalSemaphoresAsync_ptsz)(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream);
static CUresult (*pcuDestroyExternalSemaphore)(CUexternalSemaphore extSem);
static CUresult (*pcuOccupancyAvailableDynamicSMemPerBlock)(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
static CUresult (*pcuGraphKernelNodeGetParams)(CUgraphNode hNode, void *nodeParams);
static CUresult (*pcuGraphKernelNodeSetParams)(CUgraphNode hNode, const void *nodeParams);
static CUresult (*pcuGraphMemcpyNodeGetParams)(CUgraphNode hNode, void *nodeParams);
static CUresult (*pcuGraphMemcpyNodeSetParams)(CUgraphNode hNode, const void *nodeParams);
static CUresult (*pcuGraphMemsetNodeGetParams)(CUgraphNode hNode, void *nodeParams);
static CUresult (*pcuGraphMemsetNodeSetParams)(CUgraphNode hNode, const void *nodeParams);
static CUresult (*pcuGraphHostNodeGetParams)(CUgraphNode hNode, void *nodeParams);
static CUresult (*pcuGraphHostNodeSetParams)(CUgraphNode hNode, const void *nodeParams);
static CUresult (*pcuGraphAddChildGraphNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
static CUresult (*pcuGraphChildGraphNodeGetGraph)(CUgraphNode hNode, CUgraph *phGraph);
static CUresult (*pcuGraphAddEmptyNode)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
static CUresult (*pcuGraphNodeFindInClone)(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
static CUresult (*pcuGraphNodeGetType)(CUgraphNode hNode, void *type);
static CUresult (*pcuGraphGetRootNodes)(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
static CUresult (*pcuGraphGetEdges)(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
static CUresult (*pcuGraphNodeGetDependencies)(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
static CUresult (*pcuGraphNodeGetDependentNodes)(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
static CUresult (*pcuGraphAddDependencies)(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
static CUresult (*pcuGraphRemoveDependencies)(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
static CUresult (*pcuGraphExecMemcpyNodeSetParams)(CUgraphExec hGraphExec, CUgraphNode hNode, const void *copyParams, CUcontext ctx);
static CUresult (*pcuGraphExecMemsetNodeSetParams)(CUgraphExec hGraphExec, CUgraphNode hNode, const void *memsetParams, CUcontext ctx);
static CUresult (*pcuGraphExecHostNodeSetParams)(CUgraphExec hGraphExec, CUgraphNode hNode, const void *nodeParams);
static CUresult (*pcuThreadExchangeStreamCaptureMode)(CUstreamCaptureMode *mode);
static CUresult (*pcuGraphExecUpdate)(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, void *updateResult_out);
static CUresult (*pcuMemSetAccess)(CUdeviceptr_v2 ptr, size_t size, const void *desc, size_t count);
static CUresult (*pcuStreamBeginCapture)(CUstream hStream);
static CUresult (*pcuStreamBeginCapture_ptsz)(CUstream hStream);
static CUresult (*pcuMemCreate)(CUmemGenericAllocationHandle_v1 *handle, size_t size, const void *prop, unsigned long long flags);
static CUresult (*pcuMemRelease)(CUmemGenericAllocationHandle_v1 handle);
static CUresult (*pcuMemAddressFree)(CUdeviceptr_v2 ptr, size_t size);
static CUresult (*pcuMemGetAccess)(unsigned long long *flags, const void *location, CUdeviceptr_v2 ptr);
static CUresult (*pcuMemMap)(CUdeviceptr_v2 ptr, size_t size, size_t offset, CUmemGenericAllocationHandle_v1 handle, unsigned long long flags);
static CUresult (*pcuMemUnmap)(CUdeviceptr_v2 ptr, size_t size);
static CUresult (*pcuStreamGetCaptureInfo)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
static CUresult (*pcuStreamGetCaptureInfo_ptsz)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
static CUresult (*pcuMemAddressReserve)(CUdeviceptr_v2 *ptr, size_t size, size_t alignment, CUdeviceptr_v2 addr, unsigned long long flags);
static CUresult (*pcuMemExportToShareableHandle)(void* shareableHandle, void* handle, void* handleType, unsigned long long flags);
static CUresult (*pcuMemGetAllocationPropertiesFromHandle)(void* prop, CUmemGenericAllocationHandle_v1 handle);
static CUresult (*pcuMemImportFromShareableHandle)(void* handle, void* osHandle, void* shHandleType);

static void *cuda_handle = NULL;

static BOOL load_functions(void)
{
    if (!(cuda_handle = dlopen("libcuda.so.1", RTLD_NOW)))
    {
        FIXME("Wine cannot find the cuda library, CUDA support is disabled.\n");
        return FALSE;
    }

    #define LOAD_FUNCPTR(f) if((*(void **)(&p##f) = dlsym(cuda_handle, #f)) == NULL){ERR("Can't find symbol %s\n", #f); return FALSE;}
    #define TRY_LOAD_FUNCPTR(f) *(void **)(&p##f) = dlsym(cuda_handle, #f)

    LOAD_FUNCPTR(cuArray3DCreate);
    LOAD_FUNCPTR(cuArray3DCreate_v2);
    LOAD_FUNCPTR(cuArray3DGetDescriptor);
    LOAD_FUNCPTR(cuArray3DGetDescriptor_v2);
    LOAD_FUNCPTR(cuArrayCreate);
    LOAD_FUNCPTR(cuArrayCreate_v2);
    LOAD_FUNCPTR(cuArrayDestroy);
    LOAD_FUNCPTR(cuArrayGetDescriptor);
    LOAD_FUNCPTR(cuArrayGetDescriptor_v2);
    LOAD_FUNCPTR(cuCtxAttach);
    LOAD_FUNCPTR(cuCtxCreate);
    LOAD_FUNCPTR(cuCtxCreate_v2);
    LOAD_FUNCPTR(cuCtxDestroy);
    LOAD_FUNCPTR(cuCtxDestroy_v2);
    LOAD_FUNCPTR(cuCtxDetach);
    LOAD_FUNCPTR(cuCtxDisablePeerAccess);
    LOAD_FUNCPTR(cuCtxEnablePeerAccess);
    LOAD_FUNCPTR(cuCtxGetApiVersion);
    LOAD_FUNCPTR(cuCtxGetCacheConfig);
    LOAD_FUNCPTR(cuCtxGetCurrent);
    LOAD_FUNCPTR(cuCtxGetDevice);
    LOAD_FUNCPTR(cuCtxGetLimit);
    LOAD_FUNCPTR(cuCtxGetSharedMemConfig);
    LOAD_FUNCPTR(cuCtxGetStreamPriorityRange);
    LOAD_FUNCPTR(cuCtxPopCurrent);
    LOAD_FUNCPTR(cuCtxPopCurrent_v2);
    LOAD_FUNCPTR(cuCtxPushCurrent);
    LOAD_FUNCPTR(cuCtxPushCurrent_v2);
    LOAD_FUNCPTR(cuCtxSetCacheConfig);
    LOAD_FUNCPTR(cuCtxSetCurrent);
    LOAD_FUNCPTR(cuCtxSetLimit);
    LOAD_FUNCPTR(cuCtxSetSharedMemConfig);
    LOAD_FUNCPTR(cuCtxSynchronize);
    LOAD_FUNCPTR(cuDeviceCanAccessPeer);
    LOAD_FUNCPTR(cuDeviceComputeCapability);
    LOAD_FUNCPTR(cuDeviceGet);
    LOAD_FUNCPTR(cuDeviceGetAttribute);
    LOAD_FUNCPTR(cuDeviceGetByPCIBusId);
    LOAD_FUNCPTR(cuDeviceGetCount);
    LOAD_FUNCPTR(cuDeviceGetName);
    LOAD_FUNCPTR(cuDeviceGetPCIBusId);
    LOAD_FUNCPTR(cuDeviceGetProperties);
    LOAD_FUNCPTR(cuDeviceTotalMem);
    LOAD_FUNCPTR(cuDeviceTotalMem_v2);
    LOAD_FUNCPTR(cuDriverGetVersion);
    LOAD_FUNCPTR(cuEventCreate);
    LOAD_FUNCPTR(cuEventDestroy);
    LOAD_FUNCPTR(cuEventDestroy_v2);
    LOAD_FUNCPTR(cuEventElapsedTime);
    LOAD_FUNCPTR(cuEventQuery);
    LOAD_FUNCPTR(cuEventRecord);
    LOAD_FUNCPTR(cuEventSynchronize);
    LOAD_FUNCPTR(cuFuncGetAttribute);
    LOAD_FUNCPTR(cuFuncSetAttribute);
    LOAD_FUNCPTR(cuFuncSetBlockShape);
    LOAD_FUNCPTR(cuFuncSetCacheConfig);
    LOAD_FUNCPTR(cuFuncSetSharedMemConfig);
    LOAD_FUNCPTR(cuFuncSetSharedSize);
    LOAD_FUNCPTR(cuGLCtxCreate);
    LOAD_FUNCPTR(cuGLCtxCreate_v2);
    LOAD_FUNCPTR(cuGLGetDevices);
    LOAD_FUNCPTR(cuGLInit);
    LOAD_FUNCPTR(cuGLMapBufferObject);
    LOAD_FUNCPTR(cuGLMapBufferObjectAsync);
    LOAD_FUNCPTR(cuGLMapBufferObjectAsync_v2);
    LOAD_FUNCPTR(cuGLMapBufferObject_v2);
    LOAD_FUNCPTR(cuGLRegisterBufferObject);
    LOAD_FUNCPTR(cuGLSetBufferObjectMapFlags);
    LOAD_FUNCPTR(cuGLUnmapBufferObject);
    LOAD_FUNCPTR(cuGLUnmapBufferObjectAsync);
    LOAD_FUNCPTR(cuGLUnregisterBufferObject);
    LOAD_FUNCPTR(cuGetErrorName);
    LOAD_FUNCPTR(cuGetErrorString);
    LOAD_FUNCPTR(cuGetExportTable);
    LOAD_FUNCPTR(cuGraphicsGLRegisterBuffer);
    LOAD_FUNCPTR(cuGraphicsGLRegisterImage);
    LOAD_FUNCPTR(cuGraphicsMapResources);
    LOAD_FUNCPTR(cuGraphicsResourceGetMappedMipmappedArray);
    LOAD_FUNCPTR(cuGraphicsResourceGetMappedPointer);
    LOAD_FUNCPTR(cuGraphicsResourceGetMappedPointer_v2);
    LOAD_FUNCPTR(cuGraphicsResourceSetMapFlags);
    LOAD_FUNCPTR(cuGraphicsSubResourceGetMappedArray);
    LOAD_FUNCPTR(cuGraphicsUnmapResources);
    LOAD_FUNCPTR(cuGraphicsUnregisterResource);
    LOAD_FUNCPTR(cuInit);
    LOAD_FUNCPTR(cuIpcCloseMemHandle);
    LOAD_FUNCPTR(cuIpcGetEventHandle);
    LOAD_FUNCPTR(cuIpcGetMemHandle);
    LOAD_FUNCPTR(cuIpcOpenEventHandle);
    LOAD_FUNCPTR(cuIpcOpenMemHandle);
    LOAD_FUNCPTR(cuLaunch);
    LOAD_FUNCPTR(cuLaunchGrid);
    LOAD_FUNCPTR(cuLaunchGridAsync);
    LOAD_FUNCPTR(cuLaunchKernel);
    LOAD_FUNCPTR(cuLinkAddData);
    LOAD_FUNCPTR(cuLinkComplete);
    LOAD_FUNCPTR(cuLinkCreate);
    LOAD_FUNCPTR(cuLinkDestroy);
    LOAD_FUNCPTR(cuMemAlloc);
    LOAD_FUNCPTR(cuMemAllocHost);
    LOAD_FUNCPTR(cuMemAllocHost_v2);
    LOAD_FUNCPTR(cuMemAllocManaged);
    LOAD_FUNCPTR(cuMemAllocPitch);
    LOAD_FUNCPTR(cuMemAllocPitch_v2);
    LOAD_FUNCPTR(cuMemAlloc_v2);
    LOAD_FUNCPTR(cuMemFree);
    LOAD_FUNCPTR(cuMemFreeHost);
    LOAD_FUNCPTR(cuMemFree_v2);
    LOAD_FUNCPTR(cuMemGetAddressRange);
    LOAD_FUNCPTR(cuMemGetAddressRange_v2);
    LOAD_FUNCPTR(cuMemGetInfo);
    LOAD_FUNCPTR(cuMemGetInfo_v2);
    LOAD_FUNCPTR(cuMemHostAlloc);
    LOAD_FUNCPTR(cuMemHostGetDevicePointer);
    LOAD_FUNCPTR(cuMemHostGetDevicePointer_v2);
    LOAD_FUNCPTR(cuMemHostGetFlags);
    LOAD_FUNCPTR(cuMemHostRegister);
    LOAD_FUNCPTR(cuMemHostUnregister);
    LOAD_FUNCPTR(cuMemcpy);
    LOAD_FUNCPTR(cuMemcpy2D);
    LOAD_FUNCPTR(cuMemcpy2DAsync);
    LOAD_FUNCPTR(cuMemcpy2DAsync_v2);
    LOAD_FUNCPTR(cuMemcpy2DUnaligned);
    LOAD_FUNCPTR(cuMemcpy2DUnaligned_v2);
    LOAD_FUNCPTR(cuMemcpy2D_v2);
    LOAD_FUNCPTR(cuMemcpy3D);
    LOAD_FUNCPTR(cuMemcpy3DAsync);
    LOAD_FUNCPTR(cuMemcpy3DAsync_v2);
    LOAD_FUNCPTR(cuMemcpy3DPeer);
    LOAD_FUNCPTR(cuMemcpy3DPeerAsync);
    LOAD_FUNCPTR(cuMemcpy3D_v2);
    LOAD_FUNCPTR(cuMemcpyAsync);
    LOAD_FUNCPTR(cuMemcpyAtoA);
    LOAD_FUNCPTR(cuMemcpyAtoA_v2);
    LOAD_FUNCPTR(cuMemcpyAtoD);
    LOAD_FUNCPTR(cuMemcpyAtoD_v2);
    LOAD_FUNCPTR(cuMemcpyAtoH);
    LOAD_FUNCPTR(cuMemcpyAtoHAsync);
    LOAD_FUNCPTR(cuMemcpyAtoHAsync_v2);
    LOAD_FUNCPTR(cuMemcpyAtoH_v2);
    LOAD_FUNCPTR(cuMemcpyDtoA);
    LOAD_FUNCPTR(cuMemcpyDtoA_v2);
    LOAD_FUNCPTR(cuMemcpyDtoD);
    LOAD_FUNCPTR(cuMemcpyDtoDAsync);
    LOAD_FUNCPTR(cuMemcpyDtoDAsync_v2);
    LOAD_FUNCPTR(cuMemcpyDtoD_v2);
    LOAD_FUNCPTR(cuMemcpyDtoH);
    LOAD_FUNCPTR(cuMemcpyDtoHAsync);
    LOAD_FUNCPTR(cuMemcpyDtoHAsync_v2);
    LOAD_FUNCPTR(cuMemcpyDtoH_v2);
    LOAD_FUNCPTR(cuMemcpyHtoA);
    LOAD_FUNCPTR(cuMemcpyHtoAAsync);
    LOAD_FUNCPTR(cuMemcpyHtoAAsync_v2);
    LOAD_FUNCPTR(cuMemcpyHtoA_v2);
    LOAD_FUNCPTR(cuMemcpyHtoD);
    LOAD_FUNCPTR(cuMemcpyHtoDAsync);
    LOAD_FUNCPTR(cuMemcpyHtoDAsync_v2);
    LOAD_FUNCPTR(cuMemcpyHtoD_v2);
    LOAD_FUNCPTR(cuMemcpyPeer);
    LOAD_FUNCPTR(cuMemcpyPeerAsync);
    LOAD_FUNCPTR(cuMemsetD16);
    LOAD_FUNCPTR(cuMemsetD16Async);
    LOAD_FUNCPTR(cuMemsetD16_v2);
    LOAD_FUNCPTR(cuMemsetD2D16);
    LOAD_FUNCPTR(cuMemsetD2D16Async);
    LOAD_FUNCPTR(cuMemsetD2D16_v2);
    LOAD_FUNCPTR(cuMemsetD2D32);
    LOAD_FUNCPTR(cuMemsetD2D32Async);
    LOAD_FUNCPTR(cuMemsetD2D32_v2);
    LOAD_FUNCPTR(cuMemsetD2D8);
    LOAD_FUNCPTR(cuMemsetD2D8Async);
    LOAD_FUNCPTR(cuMemsetD2D8_v2);
    LOAD_FUNCPTR(cuMemsetD32);
    LOAD_FUNCPTR(cuMemsetD32Async);
    LOAD_FUNCPTR(cuMemsetD32_v2);
    LOAD_FUNCPTR(cuMemsetD8);
    LOAD_FUNCPTR(cuMemsetD8Async);
    LOAD_FUNCPTR(cuMemsetD8_v2);
    LOAD_FUNCPTR(cuMipmappedArrayCreate);
    LOAD_FUNCPTR(cuMipmappedArrayDestroy);
    LOAD_FUNCPTR(cuMipmappedArrayGetLevel);
    LOAD_FUNCPTR(cuModuleGetFunction);
    LOAD_FUNCPTR(cuModuleGetGlobal);
    LOAD_FUNCPTR(cuModuleGetGlobal_v2);
    LOAD_FUNCPTR(cuModuleGetSurfRef);
    LOAD_FUNCPTR(cuModuleGetTexRef);
    LOAD_FUNCPTR(cuModuleLoad);
    LOAD_FUNCPTR(cuModuleLoadData);
    LOAD_FUNCPTR(cuModuleLoadDataEx);
    LOAD_FUNCPTR(cuModuleLoadFatBinary);
    LOAD_FUNCPTR(cuModuleUnload);
    LOAD_FUNCPTR(cuParamSetSize);
    LOAD_FUNCPTR(cuParamSetTexRef);
    LOAD_FUNCPTR(cuParamSetf);
    LOAD_FUNCPTR(cuParamSeti);
    LOAD_FUNCPTR(cuParamSetv);
    LOAD_FUNCPTR(cuPointerGetAttribute);
    LOAD_FUNCPTR(cuPointerSetAttribute);
    LOAD_FUNCPTR(cuProfilerStart);
    LOAD_FUNCPTR(cuProfilerStop);
    LOAD_FUNCPTR(cuStreamAddCallback);
    LOAD_FUNCPTR(cuStreamAttachMemAsync);
    LOAD_FUNCPTR(cuStreamCreate);
    LOAD_FUNCPTR(cuStreamCreateWithPriority);
    LOAD_FUNCPTR(cuStreamDestroy);
    LOAD_FUNCPTR(cuStreamDestroy_v2);
    LOAD_FUNCPTR(cuStreamGetFlags);
    LOAD_FUNCPTR(cuStreamGetPriority);
    LOAD_FUNCPTR(cuStreamQuery);
    LOAD_FUNCPTR(cuStreamSynchronize);
    LOAD_FUNCPTR(cuStreamWaitEvent);
    LOAD_FUNCPTR(cuSurfObjectCreate);
    LOAD_FUNCPTR(cuSurfObjectDestroy);
    LOAD_FUNCPTR(cuSurfObjectGetResourceDesc);
    LOAD_FUNCPTR(cuSurfRefGetArray);
    LOAD_FUNCPTR(cuSurfRefSetArray);
    LOAD_FUNCPTR(cuTexObjectCreate);
    LOAD_FUNCPTR(cuTexObjectDestroy);
    LOAD_FUNCPTR(cuTexObjectGetResourceDesc);
    LOAD_FUNCPTR(cuTexObjectGetResourceViewDesc);
    LOAD_FUNCPTR(cuTexObjectGetTextureDesc);
    LOAD_FUNCPTR(cuTexRefCreate);
    LOAD_FUNCPTR(cuTexRefDestroy);
    LOAD_FUNCPTR(cuTexRefGetAddress);
    LOAD_FUNCPTR(cuTexRefGetAddressMode);
    LOAD_FUNCPTR(cuTexRefGetAddress_v2);
    LOAD_FUNCPTR(cuTexRefGetArray);
    LOAD_FUNCPTR(cuTexRefGetFilterMode);
    LOAD_FUNCPTR(cuTexRefGetFlags);
    LOAD_FUNCPTR(cuTexRefGetFormat);
    LOAD_FUNCPTR(cuTexRefGetMaxAnisotropy);
    LOAD_FUNCPTR(cuTexRefGetMipmapFilterMode);
    LOAD_FUNCPTR(cuTexRefGetMipmapLevelBias);
    LOAD_FUNCPTR(cuTexRefGetMipmapLevelClamp);
    LOAD_FUNCPTR(cuTexRefGetMipmappedArray);
    LOAD_FUNCPTR(cuTexRefSetAddress);
    LOAD_FUNCPTR(cuTexRefSetAddress2D);
    LOAD_FUNCPTR(cuTexRefSetAddress2D_v2);
    LOAD_FUNCPTR(cuTexRefSetAddress2D_v3);
    LOAD_FUNCPTR(cuTexRefSetAddressMode);
    LOAD_FUNCPTR(cuTexRefSetAddress_v2);
    LOAD_FUNCPTR(cuTexRefSetArray);
    LOAD_FUNCPTR(cuTexRefSetFilterMode);
    LOAD_FUNCPTR(cuTexRefSetFlags);
    LOAD_FUNCPTR(cuTexRefSetFormat);
    LOAD_FUNCPTR(cuTexRefSetMaxAnisotropy);
    LOAD_FUNCPTR(cuTexRefSetMipmapFilterMode);
    LOAD_FUNCPTR(cuTexRefSetMipmapLevelBias);
    LOAD_FUNCPTR(cuTexRefSetMipmapLevelClamp);
    LOAD_FUNCPTR(cuTexRefSetMipmappedArray);
    LOAD_FUNCPTR(cuLinkAddFile);

    /* CUDA 6.5 */
    LOAD_FUNCPTR(cuGLGetDevices_v2);
    LOAD_FUNCPTR(cuGraphicsResourceSetMapFlags_v2);
    LOAD_FUNCPTR(cuLinkAddData_v2);
    LOAD_FUNCPTR(cuLinkCreate_v2);
    LOAD_FUNCPTR(cuMemHostRegister_v2);
    LOAD_FUNCPTR(cuOccupancyMaxActiveBlocksPerMultiprocessor);
    LOAD_FUNCPTR(cuOccupancyMaxPotentialBlockSize);
    LOAD_FUNCPTR(cuLinkAddFile_v2);

    /* CUDA 7.0 */
    LOAD_FUNCPTR(cuCtxGetFlags);
    LOAD_FUNCPTR(cuDevicePrimaryCtxGetState);
    LOAD_FUNCPTR(cuDevicePrimaryCtxRelease);
    LOAD_FUNCPTR(cuDevicePrimaryCtxReset);
    LOAD_FUNCPTR(cuDevicePrimaryCtxRetain);
    LOAD_FUNCPTR(cuDevicePrimaryCtxSetFlags);
    LOAD_FUNCPTR(cuEventRecord_ptsz);
    LOAD_FUNCPTR(cuGLMapBufferObjectAsync_v2_ptsz);
    LOAD_FUNCPTR(cuGLMapBufferObject_v2_ptds);
    LOAD_FUNCPTR(cuGraphicsMapResources_ptsz);
    LOAD_FUNCPTR(cuGraphicsUnmapResources_ptsz);
    LOAD_FUNCPTR(cuLaunchKernel_ptsz);
    LOAD_FUNCPTR(cuMemcpy2DAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpy2DUnaligned_v2_ptds);
    LOAD_FUNCPTR(cuMemcpy2D_v2_ptds);
    LOAD_FUNCPTR(cuMemcpy3DAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpy3DPeerAsync_ptsz);
    LOAD_FUNCPTR(cuMemcpy3DPeer_ptds);
    LOAD_FUNCPTR(cuMemcpy3D_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyAsync_ptsz);
    LOAD_FUNCPTR(cuMemcpyAtoA_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyAtoD_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyAtoHAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpyAtoH_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyDtoA_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyDtoDAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpyDtoD_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyDtoHAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpyDtoH_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyHtoAAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpyHtoA_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyHtoDAsync_v2_ptsz);
    LOAD_FUNCPTR(cuMemcpyHtoD_v2_ptds);
    LOAD_FUNCPTR(cuMemcpyPeerAsync_ptsz);
    LOAD_FUNCPTR(cuMemcpyPeer_ptds);
    LOAD_FUNCPTR(cuMemcpy_ptds);
    LOAD_FUNCPTR(cuMemsetD16Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD16_v2_ptds);
    LOAD_FUNCPTR(cuMemsetD2D16Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD2D16_v2_ptds);
    LOAD_FUNCPTR(cuMemsetD2D32Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD2D32_v2_ptds);
    LOAD_FUNCPTR(cuMemsetD2D8Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD2D8_v2_ptds);
    LOAD_FUNCPTR(cuMemsetD32Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD32_v2_ptds);
    LOAD_FUNCPTR(cuMemsetD8Async_ptsz);
    LOAD_FUNCPTR(cuMemsetD8_v2_ptds);
    LOAD_FUNCPTR(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    LOAD_FUNCPTR(cuOccupancyMaxPotentialBlockSizeWithFlags);
    LOAD_FUNCPTR(cuPointerGetAttributes);
    LOAD_FUNCPTR(cuStreamAddCallback_ptsz);
    LOAD_FUNCPTR(cuStreamAttachMemAsync_ptsz);
    LOAD_FUNCPTR(cuStreamGetFlags_ptsz);
    LOAD_FUNCPTR(cuStreamGetPriority_ptsz);
    LOAD_FUNCPTR(cuStreamQuery_ptsz);
    LOAD_FUNCPTR(cuStreamSynchronize_ptsz);
    LOAD_FUNCPTR(cuStreamWaitEvent_ptsz);

    /* CUDA 8 */
    LOAD_FUNCPTR(cuDeviceGetP2PAttribute);
    LOAD_FUNCPTR(cuTexRefSetBorderColor);
    LOAD_FUNCPTR(cuTexRefGetBorderColor);
    LOAD_FUNCPTR(cuStreamWaitValue32);
    LOAD_FUNCPTR(cuStreamWriteValue32);
    LOAD_FUNCPTR(cuStreamWaitValue64);
    LOAD_FUNCPTR(cuStreamWriteValue64);
    LOAD_FUNCPTR(cuStreamBatchMemOp);
    LOAD_FUNCPTR(cuMemAdvise);
    LOAD_FUNCPTR(cuMemPrefetchAsync);
    LOAD_FUNCPTR(cuMemPrefetchAsync_ptsz);
    LOAD_FUNCPTR(cuMemRangeGetAttribute);
    LOAD_FUNCPTR(cuMemRangeGetAttributes);

    /* CUDA 9 */
    LOAD_FUNCPTR(cuLaunchCooperativeKernel);
    LOAD_FUNCPTR(cuLaunchCooperativeKernel_ptsz);
    LOAD_FUNCPTR(cuLaunchCooperativeKernelMultiDevice);
    LOAD_FUNCPTR(cuStreamGetCtx);
    LOAD_FUNCPTR(cuStreamGetCtx_ptsz);
    LOAD_FUNCPTR(cuStreamWaitValue32_ptsz);
    LOAD_FUNCPTR(cuStreamWriteValue32_ptsz);
    LOAD_FUNCPTR(cuStreamWaitValue64_ptsz);
    LOAD_FUNCPTR(cuStreamWriteValue64_ptsz);

    /* CUDA 10 */
    LOAD_FUNCPTR(cuDeviceGetUuid);
    LOAD_FUNCPTR(cuDeviceGetLuid);
    LOAD_FUNCPTR(cuStreamIsCapturing);
    LOAD_FUNCPTR(cuStreamIsCapturing_ptsz);
    LOAD_FUNCPTR(cuGraphCreate);
    LOAD_FUNCPTR(cuGraphAddMemcpyNode);
    LOAD_FUNCPTR(cuGraphAddMemsetNode);
    LOAD_FUNCPTR(cuGraphAddKernelNode);
    LOAD_FUNCPTR(cuGraphAddHostNode);
    LOAD_FUNCPTR(cuGraphGetNodes);
    LOAD_FUNCPTR(cuGraphInstantiate);
    LOAD_FUNCPTR(cuGraphClone);
    LOAD_FUNCPTR(cuGraphLaunch);
    LOAD_FUNCPTR(cuGraphLaunch_ptsz);
    LOAD_FUNCPTR(cuGraphExecKernelNodeSetParams);
    LOAD_FUNCPTR(cuStreamBeginCapture_v2);
    LOAD_FUNCPTR(cuStreamBeginCapture_v2_ptsz);
    LOAD_FUNCPTR(cuStreamEndCapture);
    LOAD_FUNCPTR(cuStreamEndCapture_ptsz);
    LOAD_FUNCPTR(cuGraphDestroyNode);
    LOAD_FUNCPTR(cuGraphDestroy);
    LOAD_FUNCPTR(cuGraphExecDestroy);
    LOAD_FUNCPTR(cuMemGetAllocationGranularity);
    LOAD_FUNCPTR(cuLaunchHostFunc);
    LOAD_FUNCPTR(cuLaunchHostFunc_ptsz);
    LOAD_FUNCPTR(cuImportExternalMemory);
    LOAD_FUNCPTR(cuExternalMemoryGetMappedBuffer);
    LOAD_FUNCPTR(cuExternalMemoryGetMappedMipmappedArray);
    LOAD_FUNCPTR(cuDestroyExternalMemory);
    LOAD_FUNCPTR(cuImportExternalSemaphore);
    LOAD_FUNCPTR(cuSignalExternalSemaphoresAsync);
    LOAD_FUNCPTR(cuSignalExternalSemaphoresAsync_ptsz);
    LOAD_FUNCPTR(cuWaitExternalSemaphoresAsync);
    LOAD_FUNCPTR(cuWaitExternalSemaphoresAsync_ptsz);
    LOAD_FUNCPTR(cuDestroyExternalSemaphore);
    LOAD_FUNCPTR(cuOccupancyAvailableDynamicSMemPerBlock);
    LOAD_FUNCPTR(cuGraphKernelNodeGetParams);
    LOAD_FUNCPTR(cuGraphKernelNodeSetParams);
    LOAD_FUNCPTR(cuGraphMemcpyNodeGetParams);
    LOAD_FUNCPTR(cuGraphMemcpyNodeSetParams);
    LOAD_FUNCPTR(cuGraphMemsetNodeGetParams);
    LOAD_FUNCPTR(cuGraphMemsetNodeSetParams);
    LOAD_FUNCPTR(cuGraphHostNodeGetParams);
    LOAD_FUNCPTR(cuGraphHostNodeSetParams);
    LOAD_FUNCPTR(cuGraphAddChildGraphNode);
    LOAD_FUNCPTR(cuGraphChildGraphNodeGetGraph);
    LOAD_FUNCPTR(cuGraphAddEmptyNode);
    LOAD_FUNCPTR(cuGraphNodeFindInClone);
    LOAD_FUNCPTR(cuGraphNodeGetType);
    LOAD_FUNCPTR(cuGraphGetRootNodes);
    LOAD_FUNCPTR(cuGraphGetEdges);
    LOAD_FUNCPTR(cuGraphNodeGetDependencies);
    LOAD_FUNCPTR(cuGraphNodeGetDependentNodes);
    LOAD_FUNCPTR(cuGraphAddDependencies);
    LOAD_FUNCPTR(cuGraphRemoveDependencies);
    LOAD_FUNCPTR(cuGraphExecMemcpyNodeSetParams);
    LOAD_FUNCPTR(cuGraphExecMemsetNodeSetParams);
    LOAD_FUNCPTR(cuGraphExecHostNodeSetParams);
    LOAD_FUNCPTR(cuThreadExchangeStreamCaptureMode);
    LOAD_FUNCPTR(cuGraphExecUpdate);
    LOAD_FUNCPTR(cuMemSetAccess);
    LOAD_FUNCPTR(cuStreamBeginCapture);
    LOAD_FUNCPTR(cuStreamBeginCapture_ptsz);
    LOAD_FUNCPTR(cuMemCreate);
    LOAD_FUNCPTR(cuMemRelease);
    LOAD_FUNCPTR(cuMemAddressFree);
    LOAD_FUNCPTR(cuMemGetAccess);
    LOAD_FUNCPTR(cuMemMap);
    LOAD_FUNCPTR(cuMemUnmap);
    LOAD_FUNCPTR(cuStreamGetCaptureInfo);
    LOAD_FUNCPTR(cuStreamGetCaptureInfo_ptsz);
    LOAD_FUNCPTR(cuMemAddressReserve);
    LOAD_FUNCPTR(cuMemExportToShareableHandle);
    LOAD_FUNCPTR(cuMemGetAllocationPropertiesFromHandle);
    LOAD_FUNCPTR(cuMemImportFromShareableHandle);

    #undef LOAD_FUNCPTR
    #undef TRY_LOAD_FUNCPTR

    return TRUE;
}

/* CUDA Function relays */
CUresult WINAPI wine_cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
    TRACE("(%p, %p)\n", pHandle, pAllocateArray);
    return pcuArray3DCreate(pHandle, pAllocateArray);
}

CUresult WINAPI wine_cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
    TRACE("(%p, %p)\n", pHandle, pAllocateArray);
    return pcuArray3DCreate_v2(pHandle, pAllocateArray);
}

CUresult WINAPI wine_cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
    TRACE("(%p, %p)\n", pArrayDescriptor, hArray);
    return pcuArray3DGetDescriptor(pArrayDescriptor, hArray);
}

CUresult WINAPI wine_cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
    TRACE("(%p, %p)\n", pArrayDescriptor, hArray);
    return pcuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
}

CUresult WINAPI wine_cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
    TRACE("(%p, %p)\n", pHandle, pAllocateArray);
    return pcuArrayCreate(pHandle, pAllocateArray);
}

CUresult WINAPI wine_cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
    TRACE("(%p, %p)\n", pHandle, pAllocateArray);
    return pcuArrayCreate_v2(pHandle, pAllocateArray);
}

CUresult WINAPI wine_cuArrayDestroy(CUarray hArray)
{
    TRACE("(%p)\n", hArray);
    return pcuArrayDestroy(hArray);
}

CUresult WINAPI wine_cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
    TRACE("(%p, %p)\n", pArrayDescriptor, hArray);
    return pcuArrayGetDescriptor(pArrayDescriptor, hArray);
}

CUresult WINAPI wine_cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
{
    TRACE("(%p, %p)\n", pArrayDescriptor, hArray);
    return pcuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
}

CUresult WINAPI wine_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    TRACE("(%p, %u)\n", pctx, flags);
    return pcuCtxAttach(pctx, flags);
}

CUresult WINAPI wine_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    TRACE("(%p, %u, %u)\n", pctx, flags, dev);
    return pcuCtxCreate(pctx, flags, dev);
}

CUresult WINAPI wine_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    TRACE("(%p, %u, %u)\n", pctx, flags, dev);
    return pcuCtxCreate_v2(pctx, flags, dev);
}

CUresult WINAPI wine_cuCtxDestroy(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxDestroy(ctx);
}

CUresult WINAPI wine_cuCtxDestroy_v2(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxDestroy_v2(ctx);
}

CUresult WINAPI wine_cuCtxDetach(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxDetach(ctx);
}

CUresult WINAPI wine_cuCtxDisablePeerAccess(CUcontext peerContext)
{
    TRACE("(%p)\n", peerContext);
    return pcuCtxDisablePeerAccess(peerContext);
}

CUresult WINAPI wine_cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
    TRACE("(%p, %u)\n", peerContext, Flags);
    return pcuCtxEnablePeerAccess(peerContext, Flags);
}

CUresult WINAPI wine_cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    TRACE("(%p, %p)\n", ctx, version);
    return pcuCtxGetApiVersion(ctx, version);
}

CUresult WINAPI wine_cuCtxGetCacheConfig(CUfunc_cache *pconfig)
{
    TRACE("(%p)\n", pconfig);
    return pcuCtxGetCacheConfig(pconfig);
}

CUresult WINAPI wine_cuCtxGetCurrent(CUcontext *pctx)
{
    TRACE("(%p)\n", pctx);
    return pcuCtxGetCurrent(pctx);
}

CUresult WINAPI wine_cuCtxGetDevice(CUdevice *device)
{
    TRACE("(%p)\n", device);
    return pcuCtxGetDevice(device);
}

CUresult WINAPI wine_cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    TRACE("(%p, %d)\n", pvalue, limit);
    return pcuCtxGetLimit(pvalue, limit);
}

CUresult WINAPI wine_cuCtxGetSharedMemConfig(CUsharedconfig *pConfig)
{
    TRACE("(%p)\n", pConfig);
    return pcuCtxGetSharedMemConfig(pConfig);
}

CUresult WINAPI wine_cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
    TRACE("(%p, %p)\n", leastPriority, greatestPriority);
    return pcuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
}

CUresult WINAPI wine_cuCtxPopCurrent(CUcontext *pctx)
{
    TRACE("(%p)\n", pctx);
    return pcuCtxPopCurrent(pctx);
}

CUresult WINAPI wine_cuCtxPopCurrent_v2(CUcontext *pctx)
{
    TRACE("(%p)\n", pctx);
    return pcuCtxPopCurrent_v2(pctx);
}

CUresult WINAPI wine_cuCtxPushCurrent(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxPushCurrent(ctx);
}

CUresult WINAPI wine_cuCtxPushCurrent_v2(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxPushCurrent_v2(ctx);
}

CUresult WINAPI wine_cuCtxSetCacheConfig(CUfunc_cache config)
{
    TRACE("(%d)\n", config);
    return pcuCtxSetCacheConfig(config);
}

CUresult WINAPI wine_cuCtxSetCurrent(CUcontext ctx)
{
    TRACE("(%p)\n", ctx);
    return pcuCtxSetCurrent(ctx);
}

CUresult WINAPI wine_cuCtxSetLimit(CUlimit limit, size_t value)
{
    TRACE("(%d, %lu)\n", limit, (SIZE_T)value);
    return pcuCtxSetLimit(limit, value);
}

CUresult WINAPI wine_cuCtxSetSharedMemConfig(CUsharedconfig config)
{
    TRACE("(%d)\n", config);
    return pcuCtxSetSharedMemConfig(config);
}

CUresult WINAPI wine_cuCtxSynchronize(void)
{
    TRACE("()\n");
    return pcuCtxSynchronize();
}

CUresult WINAPI wine_cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev)
{
    TRACE("(%p, %u, %u)\n", canAccessPeer, dev, peerDev);
    return pcuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);
}

CUresult WINAPI wine_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    CUresult ret;
    ret = pcuDeviceComputeCapability(major, minor, dev);
    TRACE("(Device: %d) SM Version: (%d.%d)\n", dev, *major, *minor);
    return ret;
}

CUresult WINAPI wine_cuDeviceGet(CUdevice *device, int ordinal)
{
    TRACE("(%p, %d)\n", device, ordinal);
    return pcuDeviceGet(device, ordinal);
}

CUresult WINAPI wine_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUresult ret = pcuDeviceGetAttribute(pi, attrib, dev);
    if(ret != CUDA_SUCCESS) TRACE("Attribute: %d, Returned error: %d\n", attrib, ret);
    else TRACE("(Device: %d, Attribute: %d) Value: (%d)\n", dev, attrib, *pi);
    return ret;
}

CUresult WINAPI wine_cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId)
{
    TRACE("(%p, %s)\n", dev, pciBusId);
    return pcuDeviceGetByPCIBusId(dev, pciBusId);
}

CUresult WINAPI wine_cuDeviceGetCount(int *count)
{
    TRACE("(%p)\n", count);
    return pcuDeviceGetCount(count);
}

CUresult WINAPI wine_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    CUresult ret;
    ret = pcuDeviceGetName(name, len, dev);
    TRACE("(Device: %d) Name: (%s)\n", dev, name);
    return ret;
}

CUresult WINAPI wine_cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev)
{
    TRACE("(%s, %d, %d)\n", pciBusId, len, dev);
    return pcuDeviceGetPCIBusId(pciBusId, len, dev);
}

CUresult WINAPI wine_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    TRACE("(%p, %d)\n", prop, dev);
    return pcuDeviceGetProperties(prop, dev);
}

CUresult WINAPI wine_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    TRACE("(%p, %d)\n", bytes, dev);
    return pcuDeviceTotalMem(bytes, dev);
}

CUresult WINAPI wine_cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    TRACE("(%p, %d)\n", bytes, dev);
    return pcuDeviceTotalMem_v2(bytes, dev);
}

CUresult WINAPI wine_cuDriverGetVersion(int *version)
{
    TRACE("(%p)\n", version);
    return pcuDriverGetVersion(version);
}

CUresult WINAPI wine_cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
    TRACE("(%p, %u)\n", phEvent, Flags);
    return pcuEventCreate(phEvent, Flags);
}

CUresult WINAPI wine_cuEventDestroy(CUevent hEvent)
{
    TRACE("(%p)\n", hEvent);
    return pcuEventDestroy(hEvent);
}

CUresult WINAPI wine_cuEventDestroy_v2(CUevent hEvent)
{
    TRACE("(%p)\n", hEvent);
    return pcuEventDestroy_v2(hEvent);
}

CUresult WINAPI wine_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    TRACE("(%p, %p, %p)\n", pMilliseconds, hStart, hEnd);
    return pcuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

CUresult WINAPI wine_cuEventQuery(CUevent hEvent)
{
    TRACE("(%p)\n", hEvent);
    return pcuEventQuery(hEvent);
}

CUresult WINAPI wine_cuEventRecord(CUevent hEvent, CUstream hStream)
{
    TRACE("(%p, %p)\n", hEvent, hStream);
    return pcuEventRecord(hEvent, hStream);
}

CUresult WINAPI wine_cuEventSynchronize(CUevent hEvent)
{
    TRACE("(%p)\n", hEvent);
    return pcuEventSynchronize(hEvent);
}

CUresult WINAPI wine_cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    TRACE("(%p, %d, %p)\n", pi, attrib, hfunc);
    return pcuFuncGetAttribute(pi, attrib, hfunc);
}

CUresult WINAPI wine_cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    TRACE("(%p, %d, %d)\n", hfunc, attrib, value);
    return pcuFuncSetAttribute(hfunc, attrib, value);
}

CUresult WINAPI wine_cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
    TRACE("(%p, %d, %d, %d)\n", hfunc, x, y, z);
    return pcuFuncSetBlockShape(hfunc, x, y, z);
}

CUresult WINAPI wine_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    TRACE("(%p, %d)\n", hfunc, config);
    return pcuFuncSetCacheConfig(hfunc, config);
}

CUresult WINAPI wine_cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
{
    TRACE("(%p, %d)\n", hfunc, config);
    return pcuFuncSetSharedMemConfig(hfunc, config);
}

CUresult WINAPI wine_cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
    TRACE("(%p, %u)\n", hfunc, bytes);
    return pcuFuncSetSharedSize(hfunc, bytes);
}

CUresult WINAPI wine_cuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, CUdevice device)
{
    TRACE("(%p, %u, %d)\n", pCtx, Flags, device);
    return pcuGLCtxCreate(pCtx, Flags, device);
}

CUresult WINAPI wine_cuGLCtxCreate_v2(CUcontext *pCtx, unsigned int Flags, CUdevice device)
{
    TRACE("(%p, %u, %d)\n", pCtx, Flags, device);
    return pcuGLCtxCreate_v2(pCtx, Flags, device);
}

CUresult WINAPI wine_cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                                    unsigned int cudaDeviceCount, CUGLDeviceList deviceList)
{
    TRACE("(%p, %p, %u, %d)\n", pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList);
    return pcuGLGetDevices(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList);
}

CUresult WINAPI wine_cuGLInit(void)
{
    TRACE("()\n");
    return pcuGLInit();
}

CUresult WINAPI wine_cuGLMapBufferObject(CUdeviceptr *dptr, size_t *size, GLuint buffer)
{
    TRACE("(%p, %p, %u)\n", dptr, size, buffer);
    return pcuGLMapBufferObject(dptr, size, buffer);
}

CUresult WINAPI wine_cuGLMapBufferObjectAsync(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream)
{
    TRACE("(%p, %p, %u, %p)\n", dptr, size,  buffer, hStream);
    return pcuGLMapBufferObjectAsync(dptr, size,  buffer, hStream);
}

CUresult WINAPI wine_cuGLMapBufferObjectAsync_v2(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream)
{
    TRACE("(%p, %p, %u, %p)\n", dptr, size,  buffer, hStream);
    return pcuGLMapBufferObjectAsync_v2(dptr, size,  buffer, hStream);
}

CUresult WINAPI wine_cuGLMapBufferObject_v2(CUdeviceptr *dptr, size_t *size, GLuint buffer)
{
    TRACE("(%p, %p, %u)\n", dptr, size, buffer);
    return pcuGLMapBufferObject_v2(dptr, size, buffer);
}

CUresult WINAPI wine_cuGLRegisterBufferObject(GLuint buffer)
{
    TRACE("(%u)\n", buffer);
    return pcuGLRegisterBufferObject(buffer);
}

CUresult WINAPI wine_cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags)
{
    TRACE("(%u, %u)\n", buffer, Flags);
    return pcuGLSetBufferObjectMapFlags(buffer, Flags);
}

CUresult WINAPI wine_cuGLUnmapBufferObject(GLuint buffer)
{
    TRACE("(%u)\n", buffer);
    return pcuGLUnmapBufferObject(buffer);
}

CUresult WINAPI wine_cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream)
{
    TRACE("(%u, %p)\n", buffer, hStream);
    return pcuGLUnmapBufferObjectAsync(buffer, hStream);
}

CUresult WINAPI wine_cuGLUnregisterBufferObject(GLuint buffer)
{
    TRACE("(%u)\n", buffer);
    return pcuGLUnregisterBufferObject(buffer);
}

CUresult WINAPI wine_cuGetErrorName(CUresult error, const char **pStr)
{
    TRACE("(%d, %p)\n", error, pStr);
    return pcuGetErrorName(error, pStr);
}

CUresult WINAPI wine_cuGetErrorString(CUresult error, const char **pStr)
{
    TRACE("(%d, %p)\n", error, pStr);
    return pcuGetErrorString(error, pStr);
}

CUresult WINAPI wine_cuGetExportTable(const void **table, const CUuuid *id)
{
    const void* orig_table = NULL;
    CUresult ret = pcuGetExportTable(&orig_table, id);
    return cuda_get_table(table, id, orig_table, ret);
}

CUresult WINAPI wine_cuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags)
{
    TRACE("(%p, %u, %u)\n", pCudaResource, buffer, Flags);
    return pcuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags);
}

CUresult WINAPI wine_cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags)
{
    TRACE("(%p, %d, %d, %d)\n", pCudaResource, image, target, Flags);
    return pcuGraphicsGLRegisterImage(pCudaResource, image, target, Flags);
}

CUresult WINAPI wine_cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
    TRACE("(%u, %p, %p)\n", count, resources, hStream);
    return pcuGraphicsMapResources(count, resources, hStream);
}

CUresult WINAPI wine_cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource)
{
    TRACE("(%p, %p)\n", pMipmappedArray, resource);
    return pcuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);
}

CUresult WINAPI wine_cuGraphicsResourceGetMappedPointer(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, CUgraphicsResource resource)
{
    TRACE("(%p, %p, %p)\n", pDevPtr, pSize, resource);
    return pcuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource);
}

CUresult WINAPI wine_cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource)
{
    TRACE("(%p, %p, %p)\n", pDevPtr, pSize, resource);
    return pcuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource);
}

CUresult WINAPI wine_cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags)
{
    TRACE("(%p, %u)\n", resource, flags);
    return pcuGraphicsResourceSetMapFlags(resource, flags);
}

CUresult WINAPI wine_cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource,
                                                         unsigned int arrayIndex, unsigned int mipLevel)
{
    TRACE("(%p, %p, %u, %u)\n", pArray, resource, arrayIndex, mipLevel);
    return pcuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
}

CUresult WINAPI wine_cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
    TRACE("(%u, %p, %p)\n", count, resources, hStream);
    return pcuGraphicsUnmapResources(count, resources, hStream);
}

CUresult WINAPI wine_cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    TRACE("(%p)\n", resource);
    return pcuGraphicsUnregisterResource(resource);
}

CUresult WINAPI wine_cuInit(unsigned int flags)
{
    TRACE("(%d)\n", flags);
    return pcuInit(flags);
}

CUresult WINAPI wine_cuIpcCloseMemHandle(CUdeviceptr_v2 dptr)
{
    TRACE("(" DEV_PTR ")\n", dptr);
    return pcuIpcCloseMemHandle(dptr);
}

CUresult WINAPI wine_cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event)
{
    TRACE("(%p, %p)\n", pHandle, event);
    return pcuIpcGetEventHandle(pHandle, event);
}

CUresult WINAPI wine_cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr_v2 dptr)
{
    TRACE("(%p, " DEV_PTR ")\n", pHandle, dptr);
    return pcuIpcGetMemHandle(pHandle, dptr);
}

CUresult WINAPI wine_cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle)
{
    TRACE("(%p, %p)\n", phEvent, &handle); /* FIXME */
    return pcuIpcOpenEventHandle(phEvent, handle);
}

CUresult WINAPI wine_cuIpcOpenMemHandle(CUdeviceptr_v2 *pdptr, CUipcMemHandle handle, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", pdptr, &handle, Flags); /* FIXME */
    return pcuIpcOpenMemHandle(pdptr, handle, Flags);
}

CUresult WINAPI wine_cuLaunch(CUfunction f)
{
    TRACE("(%p)\n", f);
    return pcuLaunch(f);
}

CUresult WINAPI wine_cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
    TRACE("(%p, %d, %d)\n", f, grid_width, grid_height);
    return pcuLaunchGrid(f, grid_width, grid_height);
}

CUresult WINAPI wine_cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
    TRACE("(%p, %d, %d, %p)\n", f, grid_width, grid_height, hStream);
    return pcuLaunchGridAsync(f, grid_width, grid_height, hStream);
}

CUresult WINAPI wine_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
    TRACE("(%p, %u, %u, %u, %u, %u, %u, %u, %p, %p, %p),\n", f, gridDimX, gridDimY, gridDimZ, blockDimX,
          blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

    return pcuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                           hStream, kernelParams, extra);
}

CUresult WINAPI wine_cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                                   unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    TRACE("(%p, %d, %p, %lu, %s, %u, %p, %p)\n", state, type, data, (SIZE_T)size, name, numOptions, options, optionValues);
    return pcuLinkAddData(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult WINAPI wine_cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut)
{
    TRACE("(%p, %p, %p)\n", state, cubinOut, sizeOut);
    return pcuLinkComplete(state, cubinOut, sizeOut);
}

CUresult WINAPI wine_cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                                  void **optionValues, CUlinkState *stateOut)
{
    TRACE("(%u, %p, %p, %p)\n", numOptions, options, optionValues, stateOut);
    return pcuLinkCreate(numOptions, options, optionValues, stateOut);
}

CUresult WINAPI wine_cuLinkDestroy(CUlinkState state)
{
    TRACE("(%p)\n", state);
    return pcuLinkDestroy(state);
}

CUresult WINAPI wine_cuMemAlloc(CUdeviceptr *dptr, unsigned int bytesize)
{
    TRACE("(%p, %u)\n", dptr, bytesize);
    return pcuMemAlloc(dptr, bytesize);
}

CUresult WINAPI wine_cuMemAllocHost(void **pp, unsigned int bytesize)
{
    TRACE("(%p, %u)\n", pp, bytesize);
    return pcuMemAllocHost(pp, bytesize);
}

CUresult WINAPI wine_cuMemAllocHost_v2(void **pp, size_t bytesize)
{
    TRACE("(%p, %lu)\n", pp, (SIZE_T)bytesize);
    return pcuMemAllocHost_v2(pp, bytesize);
}

CUresult WINAPI wine_cuMemAllocManaged(CUdeviceptr_v2 *dptr, size_t bytesize, unsigned int flags)
{
    TRACE("(%p, %lu, %u)\n", dptr, (SIZE_T)bytesize, flags);
    return pcuMemAllocManaged(dptr, bytesize, flags);
}

CUresult WINAPI wine_cuMemAllocPitch(CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes,
                                     unsigned int Height, unsigned int ElementSizeBytes)
{
    TRACE("(%p, %p, %u, %u, %u)\n", dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    return pcuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult WINAPI wine_cuMemAllocPitch_v2(CUdeviceptr_v2 *dptr, size_t *pPitch, size_t WidthInBytes,
                                        size_t Height, unsigned int ElementSizeBytes)
{
    TRACE("(%p, %p, %lu, %lu, %u)\n", dptr, pPitch, (SIZE_T)WidthInBytes, (SIZE_T)Height, ElementSizeBytes);
    return pcuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult WINAPI wine_cuMemAlloc_v2(CUdeviceptr_v2 *dptr, size_t bytesize)
{
    TRACE("(%p, %lu)\n", dptr, (SIZE_T)bytesize);
    return pcuMemAlloc_v2(dptr, bytesize);
}

CUresult WINAPI wine_cuMemFree(CUdeviceptr_v1 dptr)
{
    TRACE("(%u)\n", dptr);
    return pcuMemFree(dptr);
}

CUresult WINAPI wine_cuMemFreeHost(void *p)
{
    TRACE("(%p)\n", p);
    return pcuMemFreeHost(p);
}

CUresult WINAPI wine_cuMemFree_v2(CUdeviceptr_v2 dptr)
{
    TRACE("(" DEV_PTR ")\n", dptr);
    return pcuMemFree_v2(dptr);
}

CUresult WINAPI wine_cuMemGetAddressRange(CUdeviceptr_v1 *pbase, unsigned int *psize, CUdeviceptr_v1 dptr)
{
    TRACE("(%p, %p, %u)\n", pbase, psize, dptr);
    return pcuMemGetAddressRange(pbase, psize, dptr);
}

CUresult WINAPI wine_cuMemGetAddressRange_v2(CUdeviceptr_v2 *pbase, size_t *psize, CUdeviceptr_v2 dptr)
{
    TRACE("(%p, %p, " DEV_PTR ")\n", pbase, psize, dptr);
    return pcuMemGetAddressRange_v2(pbase, psize, dptr);
}

CUresult WINAPI wine_cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    TRACE("(%p, %p)\n", free, total);
    return pcuMemGetInfo(free, total);
}

CUresult WINAPI wine_cuMemGetInfo_v2(size_t *free, size_t *total)
{
    TRACE("(%p, %p)\n", free, total);
    return pcuMemGetInfo_v2(free, total);
}

CUresult WINAPI wine_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
    TRACE("(%p, %lu, %u)\n", pp, (SIZE_T)bytesize, Flags);
    return pcuMemHostAlloc(pp, bytesize, Flags);
}

CUresult WINAPI wine_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", pdptr, p, Flags);
    return pcuMemHostGetDevicePointer(pdptr, p, Flags);
}

CUresult WINAPI wine_cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", pdptr, p, Flags);
    return pcuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

CUresult WINAPI wine_cuMemHostGetFlags(unsigned int *pFlags, void *p)
{
    TRACE("(%p, %p)\n", pFlags, p);
    return pcuMemHostGetFlags(pFlags, p);
}

CUresult WINAPI wine_cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
{
    TRACE("(%p, %lu, %u)\n", p, (SIZE_T)bytesize, Flags);
    return pcuMemHostRegister(p, bytesize, Flags);
}

CUresult WINAPI wine_cuMemHostUnregister(void *p)
{
    TRACE("(%p)\n", p);
    return pcuMemHostUnregister(p);
}

CUresult WINAPI wine_cuMemcpy(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu)\n", dst, src, (SIZE_T)ByteCount);
    return pcuMemcpy(dst, src, ByteCount);
}

CUresult WINAPI wine_cuMemcpy2D(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2D(pCopy);
}

CUresult WINAPI wine_cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy2DAsync(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy2DAsync_v2(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2DUnaligned(pCopy);
}

CUresult WINAPI wine_cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2DUnaligned_v2(pCopy);
}

CUresult WINAPI wine_cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2D_v2(pCopy);
}

CUresult WINAPI wine_cuMemcpy3D(const CUDA_MEMCPY3D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy3D(pCopy);
}

CUresult WINAPI wine_cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy3DAsync(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy3DAsync_v2(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy3DPeer(pCopy);
}

CUresult WINAPI wine_cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy3DPeerAsync(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy3D_v2(pCopy);
}

CUresult WINAPI wine_cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu, %p)\n", dst, src, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyAsync(dst, src, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyAtoA(CUarray dstArray, unsigned int dstOffset, CUarray srcArray,
                                  unsigned int srcOffset, unsigned int ByteCount)
{
    TRACE("(%p, %u, %p, %u, %u)\n", dstArray, dstOffset, srcArray, srcOffset, ByteCount);
    return pcuMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                     size_t srcOffset, size_t ByteCount)
{
    TRACE("(%p, %lu, %p, %lu, %lu)\n", dstArray, (SIZE_T)dstOffset, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoD(CUdeviceptr_v1 dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount)
{
    TRACE("(%u, %p, %u, %u)\n", dstDevice, srcArray, srcOffset, ByteCount);
    return pcuMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoD_v2(CUdeviceptr_v2 dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, %lu, %lu)\n", dstDevice, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoH(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount)
{
    TRACE("(%p, %p, %u, %u)\n", dstHost, srcArray, srcOffset, ByteCount);
    return pcuMemcpyAtoH(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream)
{
    TRACE("(%p, %p, %u, %u, %p)\n", dstHost, srcArray, srcOffset, ByteCount, hStream);
    return pcuMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, %p, %lu, %lu, %p)\n", dstHost, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    TRACE("(%p, %p, %lu, %lu)\n", dstHost, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoA(CUarray dstArray, unsigned int dstOffset, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
{
    TRACE("(%p, %u, %u, %u)\n", dstArray, dstOffset, srcDevice, ByteCount);
    return pcuMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(%p, %lu, " DEV_PTR ", %lu)\n", dstArray, (SIZE_T)dstOffset, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoD(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
{
    TRACE("(%u, %u, %u)\n", dstDevice, srcDevice, ByteCount);
    return pcuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                       unsigned int ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %u, %p)\n", dstDevice, srcDevice, ByteCount, hStream);
    return pcuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                          size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu, %p)\n", dstDevice, srcDevice, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoD_v2(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu)\n", dstDevice, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoH(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
{
    TRACE("(%p, %u, %u)\n", dstHost, srcDevice, ByteCount);
    return pcuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream)
{
    TRACE("(%p, %u, %u, %p)\n", dstHost, srcDevice, ByteCount, hStream);
    return pcuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, " DEV_PTR ", %lu, %p)\n", dstHost, srcDevice, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(%p, " DEV_PTR ", %lu)\n", dstHost, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
{
    TRACE("(%p, %lu, %p, %lu)\n", dstArray, (SIZE_T)dstOffset, srcHost, (SIZE_T)ByteCount);
    return pcuMemcpyHtoA(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoAAsync(CUarray dstArray, unsigned int dstOffset, const void *srcHost,
                                       unsigned int ByteCount, CUstream hStream)
{
    TRACE("(%p, %u, %p, %u, %p)\n", dstArray, dstOffset, srcHost, ByteCount, hStream);
    return pcuMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                          size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, %lu, %p, %lu, %p)\n", dstArray, (SIZE_T)dstOffset, srcHost, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
{
    TRACE("(%p, %lu, %p, %lu)\n", dstArray, (SIZE_T)dstOffset, srcHost, (SIZE_T)ByteCount);
    return pcuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoD(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount)
{
    TRACE("(%u, %p, %u)\n", dstDevice, srcHost, ByteCount);
    return pcuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoDAsync(CUdeviceptr_v1 dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    TRACE("(%u, %p, %lu, %p)\n", dstDevice, srcHost, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %p, %lu, %p)\n", dstDevice, srcHost, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoD_v2(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, %lu)\n", dstDevice, srcHost, (SIZE_T)ByteCount);
    return pcuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                  CUcontext srcContext, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, " DEV_PTR ", %p, %lu)\n", dstDevice, dstContext, srcDevice, srcContext, (SIZE_T)ByteCount);
    return pcuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult WINAPI wine_cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                       CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %p, " DEV_PTR ", %p, %lu, %p)\n", dstDevice, dstContext, srcDevice, srcContext, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemsetD16(CUdeviceptr_v1 dstDevice, unsigned short us, unsigned int N)
{
    TRACE("(%u, %u, %u)\n", dstDevice, us, N);
    return pcuMemsetD16(dstDevice, us, N);
}

CUresult WINAPI wine_cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %u, %lu, %p)\n", dstDevice, us, (SIZE_T)N, hStream);
    return pcuMemsetD16Async(dstDevice, us, N, hStream);
}

CUresult WINAPI wine_cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    TRACE("(" DEV_PTR ", %u, %lu)\n", dstDevice, us, (SIZE_T)N);
    return pcuMemsetD16_v2(dstDevice, us, N);
}

CUresult WINAPI wine_cuMemsetD2D16(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height)
{
    TRACE("(%u, %u, %u, %u, %u)\n", dstDevice, dstPitch, us, Width, Height);
    return pcuMemsetD2D16(dstDevice, dstPitch, us, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                        size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, us, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D16_v2(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, us, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D32(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height)
{
    TRACE("(%u, %u, %u, %u, %u)\n", dstDevice, dstPitch, ui, Width, Height);
    return pcuMemsetD2D32(dstDevice, dstPitch, ui, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                        size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, ui, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, ui, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D8(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc,
                                  unsigned int Width, unsigned int Height)
{
    TRACE("(%u, %u, %x, %u, %u)\n", dstDevice, dstPitch, uc, Width, Height);
    return pcuMemsetD2D8(dstDevice, dstPitch, uc, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                       size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, uc, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D8_v2(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc,
                                     size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %x, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, uc, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
}

CUresult WINAPI wine_cuMemsetD32(CUdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N)
{
    TRACE("(%u, %u, %u)\n", dstDevice, ui, N);
    return pcuMemsetD32(dstDevice, ui, N);
}

CUresult WINAPI wine_cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %u, %lu, %p)\n", dstDevice, ui, (SIZE_T)N, hStream);
    return pcuMemsetD32Async(dstDevice, ui, N, hStream);
}

CUresult WINAPI wine_cuMemsetD32_v2(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N)
{
    TRACE("(" DEV_PTR ", %u, %lu)\n", dstDevice, ui, (SIZE_T)N);
    return pcuMemsetD32_v2(dstDevice, ui, N);
}

CUresult WINAPI wine_cuMemsetD8(CUdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N)
{
    TRACE("(%u, %x, %u)\n", dstDevice, uc, N);
    return pcuMemsetD8(dstDevice, uc, N);
}

CUresult WINAPI wine_cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %x, %lu, %p)\n", dstDevice, uc, (SIZE_T)N, hStream);
    return pcuMemsetD8Async(dstDevice, uc, N, hStream);
}

CUresult WINAPI wine_cuMemsetD8_v2(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N)
{
    TRACE("(" DEV_PTR ", %x, %lu)\n", dstDevice, uc, (SIZE_T)N);
    return pcuMemsetD8_v2(dstDevice, uc, N);
}

CUresult WINAPI wine_cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                            unsigned int numMipmapLevels)
{
    TRACE("(%p, %p, %u)\n", pHandle, pMipmappedArrayDesc, numMipmapLevels);
    return pcuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}

CUresult WINAPI wine_cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
    TRACE("(%p)\n", hMipmappedArray);
    return pcuMipmappedArrayDestroy(hMipmappedArray);
}

CUresult WINAPI wine_cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
{
    TRACE("(%p, %p, %u)\n", pLevelArray, hMipmappedArray, level);
    return pcuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
}

CUresult WINAPI wine_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    TRACE("(%p, %p, %s)\n", hfunc, hmod, name);
    return pcuModuleGetFunction(hfunc, hmod, name);
}

CUresult WINAPI wine_cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    TRACE("(%p, %p, %p, %s)\n", dptr, bytes, hmod, name);
    return pcuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult WINAPI wine_cuModuleGetGlobal_v2(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    TRACE("(%p, %p, %p, %s)\n", dptr, bytes, hmod, name);
    return pcuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

CUresult WINAPI wine_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    TRACE("(%p, %p, %s)\n", pSurfRef, hmod, name);
    return pcuModuleGetSurfRef(pSurfRef, hmod, name);
}

CUresult WINAPI wine_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    TRACE("(%p, %p, %s)\n", pTexRef, hmod, name);
    return pcuModuleGetTexRef(pTexRef, hmod, name);
}

CUresult WINAPI wine_cuModuleLoad(CUmodule *module, const char *fname)
{
    WCHAR filenameW[MAX_PATH];
    char *unix_name;

    TRACE("(%p, %s)\n", module, fname);

    if (!fname)
        return CUDA_ERROR_INVALID_VALUE;

    MultiByteToWideChar(CP_ACP, 0, fname, -1, filenameW, ARRAY_SIZE(filenameW));
    unix_name = wine_get_unix_file_name( filenameW );

    CUresult ret = pcuModuleLoad(module, unix_name);
    HeapFree(GetProcessHeap(), 0, unix_name);
    return ret;
}

CUresult WINAPI wine_cuModuleLoadData(CUmodule *module, const void *image)
{
    TRACE("(%p, %p)\n", module, image);
    return pcuModuleLoadData(module, image);
}

CUresult WINAPI wine_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                                        CUjit_option *options, void **optionValues)
{
    TRACE("(%p, %p, %u, %p, %p)\n", module, image, numOptions, options, optionValues);
    return pcuModuleLoadDataEx(module, image, numOptions, options, optionValues);
}

CUresult WINAPI wine_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    TRACE("(%p, %p)\n", module, fatCubin);
    return pcuModuleLoadFatBinary(module, fatCubin);;
}

CUresult WINAPI wine_cuModuleUnload(CUmodule hmod)
{
    TRACE("(%p)\n", hmod);
    return pcuModuleUnload(hmod);
}

CUresult WINAPI wine_cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    TRACE("(%p, %u)\n", hfunc, numbytes);
    return pcuParamSetSize(hfunc, numbytes);
}

CUresult WINAPI wine_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    TRACE("(%p, %d, %p)\n", hfunc, texunit, hTexRef);
    return pcuParamSetTexRef(hfunc, texunit, hTexRef);
}

CUresult WINAPI wine_cuParamSetf(CUfunction hfunc, int offset, float value)
{
    TRACE("(%p, %d, %f)\n", hfunc, offset, value);
    return pcuParamSetf(hfunc, offset, value);
}

CUresult WINAPI wine_cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
    TRACE("(%p, %d, %u)\n", hfunc, offset, value);
    return pcuParamSeti(hfunc, offset, value);
}

CUresult WINAPI wine_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    TRACE("(%p, %d, %p, %u)\n", hfunc, offset, ptr, numbytes);
    return pcuParamSetv(hfunc, offset, ptr, numbytes);
}

CUresult WINAPI wine_cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    TRACE("(%p, %d, " DEV_PTR ")\n", data, attribute, ptr);
    return pcuPointerGetAttribute(data, attribute, ptr);
}

CUresult WINAPI wine_cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    TRACE("(%p, %d, " DEV_PTR ")\n", value, attribute, ptr);
    return pcuPointerSetAttribute(value, attribute, ptr);
}

CUresult WINAPI wine_cuProfilerStart(void)
{
    TRACE("()\n");
    return pcuProfilerStart();
}

CUresult WINAPI wine_cuProfilerStop(void)
{
    TRACE("()\n");
    return pcuProfilerStop();
}

static DWORD WINAPI stream_callback_worker_thread(LPVOID parameter)
{
    struct stream_callback_entry *wrapper;
    struct list *ptr;
    pthread_mutex_lock(&stream_callback_mutex);

    for (;;)
    {
        while ((ptr = list_head(&stream_callbacks)))
        {
            wrapper = LIST_ENTRY(ptr, struct stream_callback_entry, entry);
            list_remove(&wrapper->entry);

            switch (wrapper->status)
            {
                case STREAM_CALLBACK_ABANDONED:
                    free(wrapper);
                    break;

                case STREAM_CALLBACK_PENDING:
                    pthread_mutex_unlock(&stream_callback_mutex);

                    TRACE("calling stream callback %p(%p, %d, %p)\n", wrapper->callback,
                          wrapper->args.stream, wrapper->args.status, wrapper->args.userdata);
                    wrapper->callback(wrapper->args.stream, wrapper->args.status, wrapper->args.userdata);
                    TRACE("stream callback %p returned\n", wrapper->callback);

                    wrapper->status = STREAM_CALLBACK_EXECUTED;
                    pthread_cond_broadcast(&stream_callback_reply);
                    pthread_mutex_lock(&stream_callback_mutex);
                    break;

                default:
                    assert(0); /* never reached */
            }

            if (!--num_stream_callbacks)
                goto end;
        }

        pthread_cond_wait(&stream_callback_request, &stream_callback_mutex);
    }

end:
    pthread_mutex_unlock(&stream_callback_mutex);
    return 0;
}

static void stream_callback_wrapper(CUstream hStream, CUresult status, void *userData)
{
    struct stream_callback_entry *wrapper = userData;
    wrapper->status         = STREAM_CALLBACK_PENDING;
    wrapper->args.stream    = hStream;
    wrapper->args.status    = status;
    pthread_mutex_lock(&stream_callback_mutex);

    list_add_tail(&stream_callbacks, &wrapper->entry);
    pthread_cond_signal(&stream_callback_request);
    while (wrapper->status == STREAM_CALLBACK_PENDING)
        pthread_cond_wait(&stream_callback_reply, &stream_callback_mutex);

    pthread_mutex_unlock(&stream_callback_mutex);
    free(wrapper);
}

static CUresult stream_add_callback(CUresult (*func)(CUstream, void *, void *, unsigned int),
                                    CUstream hStream, void *callback, void *userData, unsigned int flags)
{
    struct stream_callback_entry *wrapper;
    CUresult ret;

    wrapper = malloc(sizeof(*wrapper));
    if (!wrapper)
        return CUDA_ERROR_OUT_OF_MEMORY;
    wrapper->callback       = callback;
    wrapper->args.userdata  = userData;

    /* spawn a new worker thread if necessary */
    pthread_mutex_lock(&stream_callback_mutex);
    if (!num_stream_callbacks++)
    {
        HANDLE thread = CreateThread(NULL, 0, stream_callback_worker_thread, NULL, 0, NULL);
        if (!thread)
        {
            num_stream_callbacks--;
            pthread_mutex_unlock(&stream_callback_mutex);
            free(wrapper);
            return CUDA_ERROR_OUT_OF_MEMORY; /* FIXME */
        }
        CloseHandle(thread);
    }
    pthread_mutex_unlock(&stream_callback_mutex);

    ret = func(hStream, stream_callback_wrapper, wrapper, flags);
    if (ret)
    {
        pthread_mutex_lock(&stream_callback_mutex);
        if (num_stream_callbacks == 1)
        {
            wrapper->status = STREAM_CALLBACK_ABANDONED;
            list_add_tail(&stream_callbacks, &wrapper->entry);
            pthread_cond_signal(&stream_callback_request);
            wrapper = NULL;
        }
        else num_stream_callbacks--;
        pthread_mutex_unlock(&stream_callback_mutex);
        free(wrapper);
    }

    return ret;
}

CUresult WINAPI wine_cuStreamAddCallback(CUstream hStream, void *callback, void *userData, unsigned int flags)
{
    TRACE("(%p, %p, %p, %u)\n", hStream, callback, userData, flags);
    return stream_add_callback(pcuStreamAddCallback, hStream, callback, userData, flags);
}

CUresult WINAPI wine_cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", hStream, dptr, (SIZE_T)length, flags);
    return pcuStreamAttachMemAsync(hStream, dptr, length, flags);
}

CUresult WINAPI wine_cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
    TRACE("(%p, %u)\n", phStream, Flags);
    return pcuStreamCreate(phStream, Flags);
}

CUresult WINAPI wine_cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority)
{
    TRACE("(%p, %u, %d)\n", phStream, flags, priority);
    return pcuStreamCreateWithPriority(phStream, flags, priority);
}

CUresult WINAPI wine_cuStreamDestroy(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamDestroy(hStream);
}

CUresult WINAPI wine_cuStreamDestroy_v2(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamDestroy_v2(hStream);
}

CUresult WINAPI wine_cuStreamGetFlags(CUstream hStream, unsigned int *flags)
{
    TRACE("(%p, %p)\n", hStream, flags);
    return pcuStreamGetFlags(hStream, flags);
}

CUresult WINAPI wine_cuStreamGetPriority(CUstream hStream, int *priority)
{
    TRACE("(%p, %p)\n", hStream, priority);
    return pcuStreamGetPriority(hStream, priority);
}

CUresult WINAPI wine_cuStreamQuery(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamQuery(hStream);
}

CUresult WINAPI wine_cuStreamSynchronize(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamSynchronize(hStream);
}

CUresult WINAPI wine_cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", hStream, hEvent, Flags);
    return pcuStreamWaitEvent(hStream, hEvent, Flags);
}

CUresult WINAPI wine_cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc)
{
    TRACE("(%p, %p)\n", pSurfObject, pResDesc);
    return pcuSurfObjectCreate(pSurfObject, pResDesc);
}

CUresult WINAPI wine_cuSurfObjectDestroy(CUsurfObject surfObject)
{
    TRACE("(%llu)\n", surfObject);
    return pcuSurfObjectDestroy(surfObject);
}

CUresult WINAPI wine_cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject)
{
    TRACE("(%p, %llu)\n", pResDesc, surfObject);
    return pcuSurfObjectGetResourceDesc(pResDesc, surfObject);
}

CUresult WINAPI wine_cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef)
{
    TRACE("(%p, %p)\n", phArray, hSurfRef);
    return pcuSurfRefGetArray(phArray, hSurfRef);
}

CUresult WINAPI wine_cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", hSurfRef, hArray, Flags);
    return pcuSurfRefSetArray(hSurfRef, hArray, Flags);
}

CUresult WINAPI wine_cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                                       const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc)
{
    TRACE("(%p, %p, %p, %p)\n", pTexObject, pResDesc, pTexDesc, pResViewDesc);
    return pcuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

CUresult WINAPI wine_cuTexObjectDestroy(CUtexObject texObject)
{
    TRACE("(%llu)\n", texObject);
    return pcuTexObjectDestroy(texObject);
}

CUresult WINAPI wine_cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject)
{
    TRACE("(%p, %llu)\n", pResDesc, texObject);
    return pcuTexObjectGetResourceDesc(pResDesc, texObject);
}

CUresult WINAPI wine_cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject)
{
    TRACE("(%p, %llu)\n", pResViewDesc, texObject);
    return pcuTexObjectGetResourceViewDesc(pResViewDesc, texObject);
}

CUresult WINAPI wine_cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject)
{
    TRACE("(%p, %llu)\n", pTexDesc, texObject);
    return pcuTexObjectGetTextureDesc(pTexDesc, texObject);
}

CUresult WINAPI wine_cuTexRefCreate(CUtexref *pTexRef)
{
    TRACE("(%p)\n", pTexRef);
    return pcuTexRefCreate(pTexRef);
}

CUresult WINAPI wine_cuTexRefDestroy(CUtexref hTexRef)
{
    TRACE("(%p)\n", hTexRef);
    return pcuTexRefDestroy(hTexRef);
}

CUresult WINAPI wine_cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pdptr, hTexRef);
    return pcuTexRefGetAddress(pdptr, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim)
{
    TRACE("(%p, %p, %d)\n", pam, hTexRef, dim);
    return pcuTexRefGetAddressMode(pam, hTexRef, dim);
}

CUresult WINAPI wine_cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pdptr, hTexRef);
    return pcuTexRefGetAddress_v2(pdptr, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", phArray, hTexRef);
    return pcuTexRefGetArray(phArray, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pfm, hTexRef);
    return pcuTexRefGetFilterMode(pfm, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pFlags, hTexRef);
    return pcuTexRefGetFlags(pFlags, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef)
{
    TRACE("(%p, %p, %p)\n", pFormat, pNumChannels, hTexRef);
    return pcuTexRefGetFormat(pFormat, pNumChannels, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pmaxAniso, hTexRef);
    return pcuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pfm, hTexRef);
    return pcuTexRefGetMipmapFilterMode(pfm, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pbias, hTexRef);
    return pcuTexRefGetMipmapLevelBias(pbias, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef)
{
    TRACE("(%p, %p, %p)\n", pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
    return pcuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}

CUresult WINAPI wine_cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", phMipmappedArray, hTexRef);
    return pcuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);
}

CUresult WINAPI wine_cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr_v1 dptr, size_t bytes)
{
    TRACE("(%p, %p, %u, %lu)\n", ByteOffset, hTexRef, dptr, (SIZE_T)bytes);
    return pcuTexRefSetAddress(ByteOffset, hTexRef, dptr, bytes);
}

CUresult WINAPI wine_cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                          CUdeviceptr dptr, unsigned int Pitch)
{
    TRACE("(%p, %p, " DEV_PTR ", %u)\n", hTexRef, desc, dptr, Pitch);
    return pcuTexRefSetAddress2D(hTexRef, desc, dptr, Pitch);
}

CUresult WINAPI wine_cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                             CUdeviceptr dptr, size_t Pitch)
{
    TRACE("(%p, %p, " DEV_PTR ", %lu)\n", hTexRef, desc, dptr, (SIZE_T)Pitch);
    return pcuTexRefSetAddress2D_v2(hTexRef, desc, dptr, Pitch);
}

CUresult WINAPI wine_cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                             CUdeviceptr dptr, size_t Pitch)
{
    TRACE("(%p, %p, " DEV_PTR ", %lu)\n", hTexRef, desc, dptr, (SIZE_T)Pitch);
    return pcuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);
}

CUresult WINAPI wine_cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
    TRACE("(%p, %d, %u)\n", hTexRef, dim, am);
    return pcuTexRefSetAddressMode(hTexRef, dim, am);
}

CUresult WINAPI wine_cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
    TRACE("(%p, %p, " DEV_PTR ", %lu)\n", ByteOffset, hTexRef, dptr, (SIZE_T)bytes);
    return pcuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
}

CUresult WINAPI wine_cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", hTexRef, hArray, Flags);
    return pcuTexRefSetArray(hTexRef, hArray, Flags);
}

CUresult WINAPI wine_cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    TRACE("(%p, %u)\n", hTexRef, fm);
    return pcuTexRefSetFilterMode(hTexRef, fm);
}

CUresult WINAPI wine_cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
    TRACE("(%p, %u)\n", hTexRef, Flags);
    return pcuTexRefSetFlags(hTexRef, Flags);
}

CUresult WINAPI wine_cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
    TRACE("(%p, %d, %d)\n", hTexRef, fmt, NumPackedComponents);
    return pcuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
}

CUresult WINAPI wine_cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso)
{
    TRACE("(%p, %u)\n", hTexRef, maxAniso);
    return pcuTexRefSetMaxAnisotropy(hTexRef, maxAniso);
}

CUresult WINAPI wine_cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    TRACE("(%p, %u)\n", hTexRef, fm);
    return pcuTexRefSetMipmapFilterMode(hTexRef, fm);
}

CUresult WINAPI wine_cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
{
    TRACE("(%p, %f)\n", hTexRef, bias);
    return pcuTexRefSetMipmapLevelBias(hTexRef, bias);
}

CUresult WINAPI wine_cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
{
    TRACE("(%p, %f, %f)\n", hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
    return pcuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}

CUresult WINAPI wine_cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", hTexRef, hMipmappedArray, Flags);
    return pcuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);
}

CUresult WINAPI wine_cuProfilerInitialize(const char *configFile, const char *outputFile, void *outputMode)
{
    TRACE("(%s, %s, %p)\n", configFile, outputFile, outputMode);
    return pcuProfilerInitialize(configFile, outputFile, outputMode);
}

CUresult WINAPI wine_cuWGLGetDevice(CUdevice_v1 *pDevice, void *hGpu)
{
    // This function does not appear in the LINUX api version
    CUdevice dev;

    FIXME("(%p, %p) - semi-stub\n", pDevice, hGpu);
    CUresult ret = pcuDeviceGet(&dev, 0);
    if (ret) return ret;

    if (pDevice)
        *pDevice = dev;

    return CUDA_SUCCESS;
}

CUresult WINAPI wine_cuLinkAddFile(void *state, void *type, const char *path, unsigned int numOptions, void *options, void **optionValues)
{
    TRACE("(%p, %p, %s, %u, %p, %p)\n", state, type, path, numOptions, options, optionValues);
    return pcuLinkAddFile(state, type, path, numOptions, options, optionValues);
}

/*
 * Additions in CUDA 6.5
 */

CUresult WINAPI wine_cuGLGetDevices_v2(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                                       unsigned int cudaDeviceCount, CUGLDeviceList deviceList)
{
    TRACE("(%p, %p, %u, %d)\n", pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList);
    return pcuGLGetDevices_v2(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList);
}

CUresult WINAPI wine_cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags)
{
    TRACE("(%p, %u)\n", resource, flags);
    return pcuGraphicsResourceSetMapFlags_v2(resource, flags);
}

CUresult WINAPI wine_cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                                      unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    TRACE("(%p, %d, %p, %lu, %s, %u, %p, %p)\n", state, type, data, (SIZE_T)size, name, numOptions, options, optionValues);
    return pcuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult WINAPI wine_cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                                     void **optionValues, CUlinkState *stateOut)
{
    TRACE("(%u, %p, %p, %p)\n", numOptions, options, optionValues, stateOut);
    return pcuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult WINAPI wine_cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags)
{
    TRACE("(%p, %lu, %u)\n", p, (SIZE_T)bytesize, Flags);
    return pcuMemHostRegister_v2(p, bytesize, Flags);
}

CUresult WINAPI wine_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    TRACE("(%p, %p, %d, %lu)\n", numBlocks, func, blockSize, (SIZE_T)dynamicSMemSize);
    return pcuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult WINAPI wine_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                                      void *blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit)
{
    TRACE("(%p, %p, %p, %p, %lu, %d)\n", minGridSize, blockSize, func, blockSizeToDynamicSMemSize, (SIZE_T)dynamicSMemSize, blockSizeLimit);
    return pcuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

CUresult WINAPI wine_cuLinkAddFile_v2(void *state, void *type, const char *path, unsigned int numOptions, void *options, void **optionValues)
{
    TRACE("(%p, %p, %s, %u, %p, %p)\n", state, type, path, numOptions, options, optionValues);
    return pcuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

/*
 * Additions in CUDA 7.0
 */

CUresult WINAPI wine_cuCtxGetFlags(unsigned int *flags)
{
    TRACE("(%p)\n", flags);
    return pcuCtxGetFlags(flags);
}

CUresult WINAPI wine_cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
    TRACE("(%u, %p, %p)\n", dev, flags, active);
    return pcuDevicePrimaryCtxGetState(dev, flags, active);
}

CUresult WINAPI wine_cuDevicePrimaryCtxRelease(CUdevice dev)
{
    TRACE("(%u)\n", dev);
    return pcuDevicePrimaryCtxRelease(dev);
}

CUresult WINAPI wine_cuDevicePrimaryCtxReset(CUdevice dev)
{
    TRACE("(%u)\n", dev);
    return pcuDevicePrimaryCtxReset(dev);
}

CUresult WINAPI wine_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    TRACE("(%p, %u)\n", pctx, dev);
    return pcuDevicePrimaryCtxRetain(pctx, dev);
}

CUresult WINAPI wine_cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
    TRACE("(%u, %u)\n", dev, flags);
    return pcuDevicePrimaryCtxSetFlags(dev, flags);
}

CUresult WINAPI wine_cuEventRecord_ptsz(CUevent hEvent, CUstream hStream)
{
    TRACE("(%p, %p)\n", hEvent, hStream);
    return pcuEventRecord_ptsz(hEvent, hStream);
}

CUresult WINAPI wine_cuGLMapBufferObjectAsync_v2_ptsz(CUdeviceptr *dptr, size_t *size, GLuint buffer, CUstream hStream)
{
    TRACE("(%p, %p, %u, %p)\n", dptr, size,  buffer, hStream);
    return pcuGLMapBufferObjectAsync_v2_ptsz(dptr, size,  buffer, hStream);
}

CUresult WINAPI wine_cuGLMapBufferObject_v2_ptds(CUdeviceptr *dptr, size_t *size, GLuint buffer)
{
    TRACE("(%p, %p, %u)\n", dptr, size, buffer);
    return pcuGLMapBufferObject_v2_ptds(dptr, size, buffer);
}

CUresult WINAPI wine_cuGraphicsMapResources_ptsz(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
    TRACE("(%u, %p, %p)\n", count, resources, hStream);
    return pcuGraphicsMapResources_ptsz(count, resources, hStream);
}

CUresult WINAPI wine_cuGraphicsUnmapResources_ptsz(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
{
    TRACE("(%u, %p, %p)\n", count, resources, hStream);
    return pcuGraphicsUnmapResources_ptsz(count, resources, hStream);
}

CUresult WINAPI wine_cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                         unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
    TRACE("(%p, %u, %u, %u, %u, %u, %u, %u, %p, %p, %p),\n", f, gridDimX, gridDimY, gridDimZ, blockDimX,
          blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    return pcuLaunchKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                                hStream, kernelParams, extra);
}

CUresult WINAPI wine_cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy2DAsync_v2_ptsz(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2DUnaligned_v2_ptds(pCopy);
}

CUresult WINAPI wine_cuMemcpy2D_v2_ptds(const CUDA_MEMCPY2D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy2D_v2_ptds(pCopy);
}

CUresult WINAPI wine_cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy3DAsync_v2_ptsz(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)
{
    TRACE("(%p, %p)\n", pCopy, hStream);
    return pcuMemcpy3DPeerAsync_ptsz(pCopy, hStream);
}

CUresult WINAPI wine_cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy3DPeer_ptds(pCopy);
}

CUresult WINAPI wine_cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy)
{
    TRACE("(%p)\n", pCopy);
    return pcuMemcpy3D_v2_ptds(pCopy);
}

CUresult WINAPI wine_cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu, %p)\n", dst, src, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyAsync_ptsz(dst, src, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyAtoA_v2_ptds(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                          size_t srcOffset, size_t ByteCount)
{
    TRACE("(%p, %lu, %p, %lu, %lu)\n", dstArray, (SIZE_T)dstOffset, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoA_v2_ptds(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoD_v2_ptds(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, %lu, %lu)\n", dstDevice, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoD_v2_ptds(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyAtoHAsync_v2_ptsz(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, %p, %lu, %lu, %p)\n", dstHost, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyAtoHAsync_v2_ptsz(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyAtoH_v2_ptds(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    TRACE("(%p, %p, %lu, %lu)\n", dstHost, srcArray, (SIZE_T)srcOffset, (SIZE_T)ByteCount);
    return pcuMemcpyAtoH_v2_ptds(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoA_v2_ptds(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(%p, %lu, " DEV_PTR ", %lu)\n", dstArray, (SIZE_T)dstOffset, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoA_v2_ptds(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice,
                                               size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu, %p)\n", dstDevice, srcDevice, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyDtoDAsync_v2_ptsz(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoD_v2_ptds(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu)\n", dstDevice, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoD_v2_ptds(dstDevice, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyDtoHAsync_v2_ptsz(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, " DEV_PTR ", %lu, %p)\n", dstHost, srcDevice, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyDtoHAsync_v2_ptsz(dstHost, srcDevice, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyDtoH_v2_ptds(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount)
{
    TRACE("(%p, " DEV_PTR ", %lu)\n", dstHost, srcDevice, (SIZE_T)ByteCount);
    return pcuMemcpyDtoH_v2_ptds(dstHost, srcDevice, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                               size_t ByteCount, CUstream hStream)
{
    TRACE("(%p, %lu, %p, %lu, %p)\n", dstArray, (SIZE_T)dstOffset, srcHost, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyHtoAAsync_v2_ptsz(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoA_v2_ptds(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
{
    TRACE("(%p, %lu, %p, %lu)\n", dstArray, (SIZE_T)dstOffset, srcHost, (SIZE_T)ByteCount);
    return pcuMemcpyHtoA_v2_ptds(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %p, %lu, %p)\n", dstDevice, srcHost, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyHtoDAsync_v2_ptsz(dstDevice, srcHost, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, %lu)\n", dstDevice, srcHost, (SIZE_T)ByteCount);
    return pcuMemcpyHtoD_v2_ptds(dstDevice, srcHost, ByteCount);
}

CUresult WINAPI wine_cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                            CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %p, " DEV_PTR ", %p, %lu, %p)\n", dstDevice, dstContext, srcDevice, srcContext, (SIZE_T)ByteCount, hStream);
    return pcuMemcpyPeerAsync_ptsz(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult WINAPI wine_cuMemcpyPeer_ptds(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                       CUcontext srcContext, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", %p, " DEV_PTR ", %p, %lu)\n", dstDevice, dstContext, srcDevice, srcContext, (SIZE_T)ByteCount);
    return pcuMemcpyPeer_ptds(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult WINAPI wine_cuMemcpy_ptds(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount)
{
    TRACE("(" DEV_PTR ", " DEV_PTR ", %lu)\n", dst, src, (SIZE_T)ByteCount);
    return pcuMemcpy_ptds(dst, src, ByteCount);
}

CUresult WINAPI wine_cuMemsetD16Async_ptsz(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %u, %lu, %p)\n", dstDevice, us, (SIZE_T)N, hStream);
    return pcuMemsetD16Async_ptsz(dstDevice, us, N, hStream);
}

CUresult WINAPI wine_cuMemsetD16_v2_ptds(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    TRACE("(" DEV_PTR ", %u, %lu)\n", dstDevice, us, (SIZE_T)N);
    return pcuMemsetD16_v2_ptds(dstDevice, us, N);
}

CUresult WINAPI wine_cuMemsetD2D16Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                             size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, us, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D16Async_ptsz(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, us, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D16_v2_ptds(dstDevice, dstPitch, us, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D32Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                        size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, ui, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D32Async_ptsz(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, ui, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D32_v2_ptds(dstDevice, dstPitch, ui, Width, Height);
}

CUresult WINAPI wine_cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                            size_t Width, size_t Height, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %lu, %u, %lu, %lu, %p)\n", dstDevice, (SIZE_T)dstPitch, uc, (SIZE_T)Width, (SIZE_T)Height, hStream);
    return pcuMemsetD2D8Async_ptsz(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult WINAPI wine_cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                     size_t Width, size_t Height)
{
    TRACE("(" DEV_PTR ", %lu, %x, %lu, %lu)\n", dstDevice, (SIZE_T)dstPitch, uc, (SIZE_T)Width, (SIZE_T)Height);
    return pcuMemsetD2D8_v2_ptds(dstDevice, dstPitch, uc, Width, Height);
}

CUresult WINAPI wine_cuMemsetD32Async_ptsz(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %u, %lu, %p)\n", dstDevice, ui, (SIZE_T)N, hStream);
    return pcuMemsetD32Async_ptsz(dstDevice, ui, N, hStream);
}

CUresult WINAPI wine_cuMemsetD32_v2_ptds(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    TRACE("(" DEV_PTR ", %u, %lu)\n", dstDevice, ui, (SIZE_T)N);
    return pcuMemsetD32_v2_ptds(dstDevice, ui, N);
}

CUresult WINAPI wine_cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %x, %lu, %p)\n", dstDevice, uc, (SIZE_T)N, hStream);
    return pcuMemsetD8Async_ptsz(dstDevice, uc, N, hStream);
}

CUresult WINAPI wine_cuMemsetD8_v2_ptds(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    TRACE("(" DEV_PTR ", %x, %lu)\n", dstDevice, uc, (SIZE_T)N);
    return pcuMemsetD8_v2_ptds(dstDevice, uc, N);
}

CUresult WINAPI wine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize,
                                                                          size_t dynamicSMemSize, unsigned int flags)
{
    TRACE("(%p, %p, %d, %lu, %u)\n", numBlocks, func, blockSize, (SIZE_T)dynamicSMemSize, flags);
    return pcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult WINAPI wine_cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, void *blockSizeToDynamicSMemSize,
                                                               size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags)
{
    TRACE("(%p, %p, %p, %p, %lu, %d, %u)\n", minGridSize, blockSize, func, blockSizeToDynamicSMemSize, (SIZE_T)dynamicSMemSize, blockSizeLimit, flags);
    return pcuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                                                      dynamicSMemSize, blockSizeLimit, flags);
}

CUresult WINAPI wine_cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr)
{
    TRACE("(%u, %p, %p, " DEV_PTR ")\n", numAttributes, attributes, data, ptr);
    return pcuPointerGetAttributes(numAttributes, attributes, data, ptr);
}

CUresult WINAPI wine_cuStreamAddCallback_ptsz(CUstream hStream, void *callback, void *userData, unsigned int flags)
{
    TRACE("(%p, %p, %p, %u)\n", hStream, callback, userData, flags);
    return stream_add_callback(pcuStreamAddCallback_ptsz, hStream, callback, userData, flags);
}

CUresult WINAPI wine_cuStreamAttachMemAsync_ptsz(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", hStream, dptr, (SIZE_T)length, flags);
    return pcuStreamAttachMemAsync_ptsz(hStream, dptr, length, flags);
}

CUresult WINAPI wine_cuStreamGetFlags_ptsz(CUstream hStream, unsigned int *flags)
{
    TRACE("(%p, %p)\n", hStream, flags);
    return pcuStreamGetFlags_ptsz(hStream, flags);
}

CUresult WINAPI wine_cuStreamGetPriority_ptsz(CUstream hStream, int *priority)
{
    TRACE("(%p, %p)\n", hStream, priority);
    return pcuStreamGetPriority_ptsz(hStream, priority);
}

CUresult WINAPI wine_cuStreamQuery_ptsz(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamQuery_ptsz(hStream);
}

CUresult WINAPI wine_cuStreamSynchronize_ptsz(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamSynchronize_ptsz(hStream);
}

CUresult WINAPI wine_cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    TRACE("(%p, %p, %u)\n", hStream, hEvent, Flags);
    return pcuStreamWaitEvent_ptsz(hStream, hEvent, Flags);
}

/*
 * Additions in CUDA 8.0
 */

CUresult WINAPI wine_cuDeviceGetP2PAttribute(int *value, void *attrib, CUdevice_v1 srcDevice, CUdevice_v1 dstDevice)
{
    TRACE("(%p, %p, %d, %d)\n", value, attrib, srcDevice, dstDevice);
    return pcuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);
}

CUresult WINAPI wine_cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor)
{
    TRACE("(%p, %p)\n", hTexRef, pBorderColor);
    return pcuTexRefSetBorderColor(hTexRef, pBorderColor);
}

CUresult WINAPI wine_cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef)
{
    TRACE("(%p, %p)\n", pBorderColor, hTexRef);
    return pcuTexRefGetBorderColor(pBorderColor, hTexRef);
}

CUresult WINAPI wine_cuStreamWaitValue32(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %u, %u)\n", stream, addr, value, flags);
    return pcuStreamWaitValue32(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWriteValue32(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %u, %u)\n", stream, addr, value, flags);
    return pcuStreamWriteValue32(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWaitValue64(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", stream, addr, (SIZE_T)value, flags);
    return pcuStreamWaitValue64(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWriteValue64(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", stream, addr, (SIZE_T)value, flags);
    return pcuStreamWriteValue64(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamBatchMemOp(CUstream stream, unsigned int count, void *paramArray, unsigned int flags)
{
    TRACE("(%p, %u, %p, %u)\n", stream, count, paramArray, flags);
    return pcuStreamBatchMemOp(stream, count, paramArray, flags);
}

CUresult WINAPI wine_cuMemAdvise(CUdeviceptr_v2 devPtr, size_t count, void *advice, CUdevice_v1 device)
{
    TRACE("(" DEV_PTR ", %zd, %p, %d)\n", devPtr, count, advice, device);
    return pcuMemAdvise(devPtr, count, advice, device);
}

CUresult WINAPI wine_cuMemPrefetchAsync(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %zd, %d, %p)\n", devPtr, count, dstDevice, hStream);
    return pcuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
}

CUresult WINAPI wine_cuMemPrefetchAsync_ptsz(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream)
{
    TRACE("(" DEV_PTR ", %zd, %d, %p)\n", devPtr, count, dstDevice, hStream);
    return pcuMemPrefetchAsync_ptsz(devPtr, count, dstDevice, hStream);
}

CUresult WINAPI wine_cuMemRangeGetAttribute(void *data, size_t dataSize, void *attribute, CUdeviceptr_v2 devPtr, size_t count)
{
    TRACE("(%p, %lu, %p, " DEV_PTR ", %lu)\n", data, (SIZE_T)dataSize, attribute, devPtr, (SIZE_T)count);
    return pcuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
}

CUresult WINAPI wine_cuMemRangeGetAttributes(void **data, size_t *dataSizes, void *attributes, size_t numAttributes, CUdeviceptr_v2 devPtr, size_t count)
{
    TRACE("(%p, %p, %p, %lu, " DEV_PTR ", %lu)\n", data, dataSizes, attributes, (SIZE_T)numAttributes, devPtr, (SIZE_T)count);
    return pcuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
}

/*
 * Additions in CUDA 9.0
 */

CUresult WINAPI wine_cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
{
    TRACE("(%p, %u, %u, %u, %u, %u, %u, %u, %p, %p)\n", f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
    return pcuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult WINAPI wine_cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
{
    TRACE("(%p, %u, %u, %u, %u, %u, %u, %u, %p, %p)\n", f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
    return pcuLaunchCooperativeKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult WINAPI wine_cuLaunchCooperativeKernelMultiDevice(void *launchParamsList, unsigned int numDevices, unsigned int flags)
{
    TRACE("(%p, %u, %u)\n", launchParamsList, numDevices, flags);
    return pcuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
}

CUresult WINAPI wine_cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
    TRACE("(%p, %p)\n", hStream, pctx);
    return pcuStreamGetCtx(hStream, pctx);
}

CUresult WINAPI wine_cuStreamGetCtx_ptsz(CUstream hStream, CUcontext *pctx)
{
    TRACE("(%p, %p)\n", hStream, pctx);
    return pcuStreamGetCtx_ptsz(hStream, pctx);
}

CUresult WINAPI wine_cuStreamWaitValue32_ptsz(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %u, %u)\n", stream, addr, value, flags);
    return pcuStreamWaitValue32_ptsz(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWriteValue32_ptsz(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %u, %u)\n", stream, addr, value, flags);
    return pcuStreamWriteValue32_ptsz(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWaitValue64_ptsz(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", stream, addr, (SIZE_T)value, flags);
    return pcuStreamWaitValue64_ptsz(stream, addr, value, flags);
}

CUresult WINAPI wine_cuStreamWriteValue64_ptsz(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags)
{
    TRACE("(%p, " DEV_PTR ", %lu, %u)\n", stream, addr, (SIZE_T)value, flags);
    return pcuStreamWriteValue64_ptsz(stream, addr, value, flags);
}

/*
 * Additions in CUDA 10.0
 */

CUresult WINAPI wine_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
{
    char buffer[128];

    CUresult ret = pcuDeviceGetUuid(uuid, dev);
    if(ret == CUDA_SUCCESS)
    {
        TRACE("(UUID: %s, Device: %d)\n", cuda_print_uuid(uuid, buffer, sizeof(buffer)), dev);
        return ret;
    }
    else return CUDA_ERROR_INVALID_VALUE;
}

CUresult WINAPI wine_cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev)
{
    // In case we need to fake LUID creation possibly for emulated hardware situations
    CUresult ret;
    CUuuid uuid;
    const char *env = getenv("CUDA_FAKE_LUID");
    if (env && *env == '1')
    {
        uint64_t hash = 0;
        for (int k = 0; k < sizeof(uuid.bytes); k++)
            hash = (hash << 5) - hash + uuid.bytes[k];
        memcpy(luid, &hash, sizeof(LUID));

        *deviceNodeMask = 1;
        TRACE("LUID override! Returning fake LUID for device %d\n", dev);

        return CUDA_SUCCESS;
    }
    char buffer[1024];
    KEY_VALUE_PARTIAL_INFORMATION *value = (void *)buffer;
    KEY_BASIC_INFORMATION *key = (void *)buffer;
    HKEY pci, outerdev, innerdev, prop;
    DWORD size, i = 0;

    TRACE("(%p, %p, %d)\n", luid, deviceNodeMask, dev);

    if ((ret = pcuDeviceGetUuid(&uuid, dev)) != CUDA_SUCCESS)
    {
        ERR("Error: %d when retrieving UUID for device %d\n", ret, dev);
        return ret;
    }

    ret = CUDA_ERROR_UNKNOWN;

    if (!(pci = reg_open_ascii_key(NULL, "\\Registry\\Machine\\System\\CurrentControlSet\\Enum\\PCI")))
    {
        ERR("Error opening registry key for PCI device enumerator\n");
        return ret;
    }

    while (!NtEnumerateKey(pci, i++, KeyBasicInformation, key, sizeof(buffer), &size))
    {
        DWORD j = 0;

        if (!(outerdev = reg_open_key(pci, key->Name, key->NameLength)))
            continue;

        while (!NtEnumerateKey(outerdev, j++, KeyBasicInformation, key, sizeof(buffer), &size))
        {
            if (!(innerdev = reg_open_key(outerdev, key->Name, key->NameLength)))
                continue;

            if (!(prop = reg_open_ascii_key(innerdev, devpropkey_gpu_vulkan_uuidA)))
                goto next;

            if (query_reg_value(prop, NULL, value, sizeof(buffer)) != sizeof(CUuuid) || memcmp(value->Data, &uuid, sizeof(uuid)))
                goto cleanup;

            NtClose(prop);

            if (!(prop = reg_open_ascii_key(innerdev, devpropkey_gpu_luidA)))
                goto next;

            if (query_reg_value(prop, NULL, value, sizeof(buffer)) != sizeof(LUID))
                goto cleanup;

            memcpy(luid, value->Data, sizeof(LUID));
            *deviceNodeMask = 1;
            ret = CUDA_SUCCESS;

            TRACE("Found LUID: %02x%02x%02x%02x-%02x%02x%02x%02x\n",
                (unsigned char)luid[0], (unsigned char)luid[1], (unsigned char)luid[2], (unsigned char)luid[3],
                (unsigned char)luid[4], (unsigned char)luid[5], (unsigned char)luid[6], (unsigned char)luid[7]);

        cleanup:
            NtClose(prop);
        next:
            NtClose(innerdev);
        }

        NtClose(outerdev);
    }

    NtClose(pci);

    if (ret != CUDA_SUCCESS)
        ERR("Failed to retrieve LUID for device %d\n", dev);

    return ret;
}

CUresult WINAPI wine_cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus)
{
    TRACE("(%p, %p)\n", hStream, captureStatus);
    return pcuStreamIsCapturing(hStream, captureStatus);
}

CUresult WINAPI wine_cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus *captureStatus)
{
    TRACE("(%p, %p)\n", hStream, captureStatus);
    return pcuStreamIsCapturing_ptsz(hStream, captureStatus);
}

CUresult WINAPI wine_cuGraphCreate(CUgraph *phGraph, unsigned int flags)
{
    TRACE("(%p, %d)\n", phGraph, flags);
    return pcuGraphCreate(phGraph, flags);
}

CUresult WINAPI wine_cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx)
{
    TRACE("(%p, %p, %p, %zd, %p, %p)\n", phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
    return pcuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
}

CUresult WINAPI wine_cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *memsetParams, CUcontext ctx)
{
    TRACE("(%p, %p, %p, %zd, %p, %p)\n", phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
    return pcuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
}

CUresult WINAPI wine_cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *nodeParams)
{
    TRACE("(%p, %p, %p, %zd, %p)\n", phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    return pcuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult WINAPI wine_cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_NODE_PARAMS *nodeParams)
{
    TRACE("(%p, %p, %p, %zd, %p)\n", phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    return pcuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult WINAPI wine_cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes)
{
    TRACE("(%p, %p, %p)\n", hGraph, nodes, numNodes);
    return pcuGraphGetNodes(hGraph, nodes, numNodes);
}

CUresult WINAPI wine_cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize)
{
    TRACE("(%p, %p, %p, %p, %lu)\n", phGraphExec, hGraph, phErrorNode, logBuffer, (SIZE_T)bufferSize);
    return pcuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

CUresult WINAPI wine_cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph)
{
    TRACE("(%p, %p)\n", phGraphClone, originalGraph);
    return pcuGraphClone(phGraphClone, originalGraph);
}

CUresult WINAPI wine_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
    TRACE("(%p, %p)\n", hGraphExec, hStream);
    return pcuGraphLaunch(hGraphExec, hStream);
}

CUresult WINAPI wine_cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream)
{
    TRACE("(%p, %p)\n", hGraphExec, hStream);
    return pcuGraphLaunch_ptsz(hGraphExec, hStream);
}

CUresult WINAPI wine_cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_NODE_PARAMS *nodeParams)
{
    TRACE("(%p, %p, %p)\n", hGraphExec, hNode, nodeParams);
    return pcuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams);
}

CUresult WINAPI wine_cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
{
    TRACE("(%p, %d)\n", hStream, mode);
    return pcuStreamBeginCapture_v2(hStream, mode);
}

CUresult WINAPI wine_cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode)
{
    TRACE("(%p, %d)\n", hStream, mode);
    return pcuStreamBeginCapture_v2_ptsz(hStream, mode);
}

CUresult WINAPI wine_cuStreamEndCapture(CUstream hStream, CUgraph *phGraph)
{
    TRACE("(%p, %p)\n", hStream, phGraph);
    return pcuStreamEndCapture(hStream, phGraph);
}

CUresult WINAPI wine_cuStreamEndCapture_ptsz(CUstream hStream, CUgraph *phGraph)
{
    TRACE("(%p, %p)\n", hStream, phGraph);
    return pcuStreamEndCapture_ptsz(hStream, phGraph);
}

CUresult WINAPI wine_cuGraphDestroyNode(CUgraphNode hNode)
{
    TRACE("(%p)\n", hNode);
    return pcuGraphDestroyNode(hNode);
}

CUresult WINAPI wine_cuGraphDestroy(CUgraph hGraph)
{
    TRACE("(%p)\n", hGraph);
    return pcuGraphDestroy(hGraph);
}

CUresult WINAPI wine_cuGraphExecDestroy(CUgraphExec hGraphExec)
{
    TRACE("(%p)\n", hGraphExec);
    return pcuGraphExecDestroy(hGraphExec);
}

CUresult WINAPI wine_cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option)
{
    TRACE("(%p, %p, %d)\n", granularity, prop, option);
    return pcuMemGetAllocationGranularity(granularity, prop, option);
}

CUresult WINAPI wine_cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData)
{
    TRACE("(%p, %p, %p)\n", hStream, fn, userData);
    return pcuLaunchHostFunc(hStream, fn, userData);
}

CUresult WINAPI wine_cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void *userData)
{
    TRACE("(%p, %p, %p)\n", hStream, fn, userData);
    return pcuLaunchHostFunc_ptsz(hStream, fn, userData);
}

CUresult WINAPI wine_cuImportExternalMemory(void *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc)
{
    TRACE("(%p, %p)\n", extMem_out, memHandleDesc);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC linuxHandleDesc = *memHandleDesc;
    HANDLE handle;

    /* Linux libcuda does not work win32 handles */
    switch (linuxHandleDesc.type)
    {
        case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32:
            linuxHandleDesc.handle.fd = get_shared_resource_fd(linuxHandleDesc.handle.win32.handle);
            linuxHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
            break;
        case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            handle = get_shared_resource_kmt_handle(linuxHandleDesc.handle.win32.handle);
            linuxHandleDesc.handle.fd = get_shared_resource_fd(handle);
            linuxHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        default:
            break;
    }

    CUresult ret = pcuImportExternalMemory(extMem_out, &linuxHandleDesc);
    if(ret) ERR("Returned error: %d\n", ret);
    return ret;
}

CUresult WINAPI wine_cuExternalMemoryGetMappedBuffer(CUdeviceptr_v2 *devPtr, void *extMem, const void *bufferDesc)
{
    TRACE("(%p, %p, %p)\n", devPtr, extMem, bufferDesc);
    return pcuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
}

CUresult WINAPI wine_cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, void *extMem, const void *mipmapDesc)
{
    TRACE("(%p, %p, %p)\n", mipmap, extMem, mipmapDesc);
    return pcuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
}

CUresult WINAPI wine_cuDestroyExternalMemory(void *extMem)
{
    TRACE("(%p)\n", extMem);
    return pcuDestroyExternalMemory(extMem);
}

static pthread_mutex_t semMutex = PTHREAD_MUTEX_INITIALIZER;
static int semCounter = 0;
static CUexternalSemaphore semList[4] = {0};

static void semaphoreHandling(CUexternalSemaphore *extSem_out)
{
    pthread_mutex_lock(&semMutex);
    if(semCounter >= 4) semCounter = 0;
    if(semList[semCounter])
    {
        CUresult ret = pcuDestroyExternalSemaphore(semList[semCounter]);
        if(ret) FIXME("Failed to destroy semaphore: %p: %d\n", semList[semCounter], ret);
    }

    semList[semCounter] = *extSem_out;
    semCounter++;
    pthread_mutex_unlock(&semMutex);
}

CUresult WINAPI wine_cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc)
{
    TRACE("(%p, %p, %d)\n", extSem_out, semHandleDesc, semHandleDesc->type);

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC linuxHandleDesc = *semHandleDesc;
    HANDLE handle;

    switch (linuxHandleDesc.type)
    {
        case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32:
            linuxHandleDesc.handle.fd = get_shared_resource_fd(linuxHandleDesc.handle.win32.handle);
            linuxHandleDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
            break;
        case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            handle = get_shared_resource_kmt_handle(linuxHandleDesc.handle.win32.handle);
            linuxHandleDesc.handle.fd = get_shared_resource_fd(handle);
            linuxHandleDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
            break;
        case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32:
            linuxHandleDesc.handle.fd = get_shared_resource_fd(linuxHandleDesc.handle.win32.handle);
            linuxHandleDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
            break;
        default:
            break;
    }

    CUresult ret = pcuImportExternalSemaphore(extSem_out, &linuxHandleDesc);
    if(!ret) semaphoreHandling(extSem_out);
    else ERR("Returned error: %d\n", ret);

    return ret;
}

CUresult WINAPI wine_cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream)
{
    TRACE("(%p, %p, %u, %p)\n", extSemArray, paramsArray, numExtSems, stream);
    return pcuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
}

CUresult WINAPI wine_cuSignalExternalSemaphoresAsync_ptsz(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream)
{
    TRACE("(%p, %p, %u, %p)\n", extSemArray, paramsArray, numExtSems, stream);
    return pcuSignalExternalSemaphoresAsync_ptsz(extSemArray, paramsArray, numExtSems, stream);
}

CUresult WINAPI wine_cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream)
{
    TRACE("(%p, %p, %u, %p)\n", extSemArray, paramsArray, numExtSems, stream);
    return pcuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
}

CUresult WINAPI wine_cuWaitExternalSemaphoresAsync_ptsz(const CUexternalSemaphore *extSemArray, const void *paramsArray, unsigned int numExtSems, CUstream stream)
{
    TRACE("(%p, %p, %u, %p)\n", extSemArray, paramsArray, numExtSems, stream);
    return pcuWaitExternalSemaphoresAsync_ptsz(extSemArray, paramsArray, numExtSems, stream);
}

CUresult WINAPI wine_cuDestroyExternalSemaphore(CUexternalSemaphore extSem)
{
    TRACE("(%p)\n", extSem);
    return pcuDestroyExternalSemaphore(extSem);
}

CUresult WINAPI wine_cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
{
    TRACE("(%p, %p, %d, %d)\n", dynamicSmemSize, func, numBlocks, blockSize);
    return pcuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
}

CUresult WINAPI wine_cuGraphKernelNodeGetParams(CUgraphNode hNode, void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphKernelNodeGetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphKernelNodeSetParams(CUgraphNode hNode, const void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphKernelNodeSetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphMemcpyNodeGetParams(CUgraphNode hNode, void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphMemcpyNodeGetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphMemcpyNodeSetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphMemsetNodeGetParams(CUgraphNode hNode, void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphMemsetNodeGetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphMemsetNodeSetParams(CUgraphNode hNode, const void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphMemsetNodeSetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphHostNodeGetParams(CUgraphNode hNode, void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphHostNodeGetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphHostNodeSetParams(CUgraphNode hNode, const void *nodeParams)
{
    TRACE("(%p, %p)\n", hNode, nodeParams);
    return pcuGraphHostNodeSetParams(hNode, nodeParams);
}

CUresult WINAPI wine_cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph)
{
    TRACE("(%p, %p, %p, %zu, %p)\n", phGraphNode, hGraph, dependencies, numDependencies, childGraph);
    return pcuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
}

CUresult WINAPI wine_cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph)
{
    TRACE("(%p, %p)\n", hNode, phGraph);
    return pcuGraphChildGraphNodeGetGraph(hNode, phGraph);
}

CUresult WINAPI wine_cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies)
{
    TRACE("(%p, %p, %p, %zu)\n", phGraphNode, hGraph, dependencies, numDependencies);
    return pcuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);
}

CUresult WINAPI wine_cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
{
    TRACE("(%p, %p, %p)\n", phNode, hOriginalNode, hClonedGraph);
    return pcuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);
}

CUresult WINAPI wine_cuGraphNodeGetType(CUgraphNode hNode, void *type)
{
    TRACE("(%p, %p)\n", hNode, type);
    return pcuGraphNodeGetType(hNode, type);
}

CUresult WINAPI wine_cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes)
{
    TRACE("(%p, %p, %p)\n", hGraph, rootNodes, numRootNodes);
    return pcuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);
}

CUresult WINAPI wine_cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges)
{
    TRACE("(%p, %p, %p, %p)\n", hGraph, from, to, numEdges);
    return pcuGraphGetEdges(hGraph, from, to, numEdges);
}

CUresult WINAPI wine_cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies)
{
    TRACE("(%p, %p, %p)\n", hNode, dependencies, numDependencies);
    return pcuGraphNodeGetDependencies(hNode, dependencies, numDependencies);
}

CUresult WINAPI wine_cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes)
{
    TRACE("(%p, %p, %p)\n", hNode, dependentNodes, numDependentNodes);
    return pcuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
}

CUresult WINAPI wine_cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies)
{
    TRACE("(%p, %p, %p, %zu)\n", hGraph, from, to, numDependencies);
    return pcuGraphAddDependencies(hGraph, from, to, numDependencies);
}

CUresult WINAPI wine_cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies)
{
    TRACE("(%p, %p, %p, %zu)\n", hGraph, from, to, numDependencies);
    return pcuGraphRemoveDependencies(hGraph, from, to, numDependencies);
}

CUresult WINAPI wine_cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const void *copyParams, CUcontext ctx)
{
    TRACE("(%p, %p, %p, %p)\n", hGraphExec, hNode, copyParams, ctx);
    return pcuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);
}

CUresult WINAPI wine_cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const void *memsetParams, CUcontext ctx)
{
    TRACE("(%p, %p, %p, %p)\n", hGraphExec, hNode, memsetParams, ctx);
    return pcuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);
}

CUresult WINAPI wine_cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const void *nodeParams)
{
    TRACE("(%p, %p, %p)\n", hGraphExec, hNode, nodeParams);
    return pcuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);
}

CUresult WINAPI wine_cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode)
{
    TRACE("(%p)\n", mode);
    return pcuThreadExchangeStreamCaptureMode(mode);
}

CUresult WINAPI wine_cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, void *updateResult_out)
{
    TRACE("(%p, %p, %p, %p)\n", hGraphExec, hGraph, hErrorNode_out, updateResult_out);
    return pcuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}

CUresult WINAPI wine_cuMemSetAccess(CUdeviceptr_v2 ptr, size_t size, const void *desc, size_t count)
{
    TRACE("(" DEV_PTR ", %zu, %p, %zu)\n", ptr, size, desc, count);
    return pcuMemSetAccess(ptr, size, desc, count);
}

CUresult WINAPI wine_cuStreamBeginCapture(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamBeginCapture(hStream);
}

CUresult WINAPI wine_cuStreamBeginCapture_ptsz(CUstream hStream)
{
    TRACE("(%p)\n", hStream);
    return pcuStreamBeginCapture_ptsz(hStream);
}

CUresult WINAPI wine_cuMemCreate(CUmemGenericAllocationHandle_v1 *handle, size_t size, const void *prop, unsigned long long flags)
{
    TRACE("(%p, %lu, %p, %llu)\n", handle, (SIZE_T)size, prop, flags);
    return pcuMemCreate(handle, size, prop, flags);
}

CUresult WINAPI wine_cuMemRelease(CUmemGenericAllocationHandle_v1 handle)
{
    TRACE("(%lld)\n", handle);
    return pcuMemRelease(handle);
}

CUresult WINAPI wine_cuMemAddressFree(CUdeviceptr_v2 ptr, size_t size)
{
    TRACE("(" DEV_PTR ", %lu)\n", ptr, (SIZE_T)size);
    return pcuMemAddressFree(ptr, size);
}

CUresult WINAPI wine_cuMemGetAccess(unsigned long long *flags, const void *location, CUdeviceptr_v2 ptr)
{
    TRACE("(%p, %p, " DEV_PTR ")\n", flags, location, ptr);
    return pcuMemGetAccess(flags, location, ptr);
}

CUresult WINAPI wine_cuMemMap(CUdeviceptr_v2 ptr, size_t size, size_t offset, CUmemGenericAllocationHandle_v1 handle, unsigned long long flags)
{
    TRACE("(" DEV_PTR ", %lu, %lu, %lld, %llu)\n", ptr, (SIZE_T)size, (SIZE_T)offset, handle, flags);
    return pcuMemMap(ptr, size, offset, handle, flags);
}

CUresult WINAPI wine_cuMemUnmap(CUdeviceptr_v2 ptr, size_t size)
{
    TRACE("(" DEV_PTR ", %lu\n", ptr, (SIZE_T)size);
    return pcuMemUnmap(ptr, size);
}

CUresult WINAPI wine_cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out)
{
    TRACE("(%p, %p, %p)\n", hStream, captureStatus_out, id_out);
    return pcuStreamGetCaptureInfo(hStream, captureStatus_out, id_out);
}

CUresult WINAPI wine_cuStreamGetCaptureInfo_ptsz(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out)
{
    TRACE("(%p, %p, %p)\n", hStream, captureStatus_out, id_out);
    return pcuStreamGetCaptureInfo_ptsz(hStream, captureStatus_out, id_out);
}

CUresult WINAPI wine_cuMemAddressReserve(CUdeviceptr_v2 *ptr, size_t size, size_t alignment, CUdeviceptr_v2 addr, unsigned long long flags)
{
    TRACE("(%p, %zu, %zu, " DEV_PTR ", %llu)\n", ptr, size, alignment, addr, flags);
    return pcuMemAddressReserve(ptr, size, alignment, addr, flags);
}

CUresult WINAPI wine_cuMemExportToShareableHandle(void* shareableHandle, void* handle, void* handleType, unsigned long long flags)
{
    TRACE("(%p, %p, %p, %llu)\n", shareableHandle, handle, handleType, flags);
    return pcuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
}

CUresult WINAPI wine_cuMemGetAllocationPropertiesFromHandle(void* prop, CUmemGenericAllocationHandle_v1 handle)
{
    TRACE("(%p, %llu)\n", prop, handle);
    return pcuMemGetAllocationPropertiesFromHandle(prop, handle);
}

CUresult WINAPI wine_cuMemImportFromShareableHandle(void* handle, void* osHandle, void* shHandleType)
{
    TRACE("(%p, %p, %p)\n", handle, osHandle, shHandleType);
    return pcuMemImportFromShareableHandle(handle, osHandle, shHandleType);
}

#define CHECK_FUNCPTR(f) \
    do \
    { \
        if (p##f == NULL) \
        { \
            FIXME("not supported\n"); \
            return CUDA_ERROR_NOT_SUPPORTED; \
        } \
    } \
    while (0)

#undef CHECK_FUNCPTR


/*
 * Direct3D emulated functions
 */

CUresult WINAPI wine_cuD3D9CtxCreate(CUcontext *pCtx, CUdevice *pCudaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice)
{
    CUresult ret;
    CUdevice dev;

    FIXME("(%p, %p, %u, %p) - semi-stub\n", pCtx, pCudaDevice, Flags, pD3DDevice);

    ret = pcuDeviceGet(&dev, 0);
    if (ret) return ret;

    ret = pcuCtxCreate(pCtx, Flags, dev);
    if (ret) return ret;

    if (pCudaDevice)
        *pCudaDevice = dev;

    return CUDA_SUCCESS;
}

CUresult WINAPI wine_cuD3D9GetDevice(CUdevice *pCudaDevice, const char *pszAdapterName)
{
    FIXME("(%p, %s) - semi-stub\n", pCudaDevice, pszAdapterName);
    return pcuDeviceGet(pCudaDevice, 0);
}

CUresult WINAPI wine_cuGraphicsD3D9RegisterResource(CUgraphicsResource *pCudaResource, IDirect3DResource9 *pD3DResource, unsigned int Flags)
{
    FIXME("(%p, %p, %u) - semi-stub\n", pCudaResource, pD3DResource, Flags);
    /* Not able to handle spesific flags at this time */
    if(Flags > 0)
      return CUDA_ERROR_INVALID_VALUE;

    /* pD3DResource is unknown at this time and cannot be "registered" */
    return CUDA_ERROR_UNKNOWN;
}

CUresult WINAPI wine_cuD3D10GetDevice(CUdevice *pCudaDevice, IDXGIAdapter *pAdapter)
{
    FIXME("(%p, %p) - semi-stub\n", pCudaDevice, pAdapter);
    /* DXGI adapters don't have an OpenGL context assigned yet, otherwise we could use cuGLGetDevices */
    return pcuDeviceGet(pCudaDevice, 0);
}

CUresult WINAPI wine_cuGraphicsD3D10RegisterResource(CUgraphicsResource *pCudaResource, ID3D10Resource *pD3DResource, unsigned int Flags)
{
    FIXME("(%p, %p, %u) - semi-stub\n", pCudaResource, pD3DResource, Flags);
    /* Not able to handle spesific flags at this time */
    if(Flags > 0)
      return CUDA_ERROR_INVALID_VALUE;

    /* pD3DResource is unknown at this time and cannot be "registered" */
    return CUDA_ERROR_UNKNOWN;
}

CUresult WINAPI wine_cuD3D11GetDevice(CUdevice *pCudaDevice, IDXGIAdapter *pAdapter)
{
    FIXME("(%p, %p) - semi-stub\n", pCudaDevice, pAdapter);
    /* DXGI adapters don't have an OpenGL context assigned yet, otherwise we could use cuGLGetDevices */
    return pcuDeviceGet(pCudaDevice, 0);
}

CUresult WINAPI wine_cuGraphicsD3D11RegisterResource(CUgraphicsResource *pCudaResource, ID3D11Resource *pD3DResource, unsigned int Flags)
{
    FIXME("(%p, %p, %u) - semi-stub\n", pCudaResource, pD3DResource, Flags);
    /* Not able to handle spesific flags at this time */
    if(Flags > 0)
      return CUDA_ERROR_INVALID_VALUE;

    /* pD3DResource is unknown at this time and cannot be "registered" */
    return CUDA_ERROR_UNKNOWN;
}

CUresult WINAPI wine_cuD3D11CtxCreate(CUcontext* pCtx, CUdevice* pCudaDevice, unsigned int Flags, ID3D11Device* pD3DDevice)
{
    FIXME("(%p, %p, %u, %p) - semi-stub\n", pCtx, pCudaDevice, Flags, pD3DDevice);
    /* Create a cuda context on the device - This will NOT actually be registered to the D3D11 device. */
    CUresult ret;

    ret = pcuDeviceGet(pCudaDevice, 0);
    if (ret)
    {
        ERR("Failed to get CUDA device, error: %d\n", ret);
        return ret;
    }
    ret = pcuCtxCreate(pCtx, Flags, *pCudaDevice);
    if (ret) ERR("Failed to create cuda context, error: %d\n", ret);
    return ret;
}

CUresult WINAPI wine_cuD3D11CtxCreate_v2(CUcontext* pCtx, CUdevice* pCudaDevice, unsigned int Flags, ID3D11Device* pD3DDevice)
{
    FIXME("(%p, %p, %u, %p) - semi-stub\n", pCtx, pCudaDevice, Flags, pD3DDevice);
    /* Create a cuda context on the device - This will NOT actually be registered to the D3D11 device. */
    CUresult ret;

    ret = pcuDeviceGet(pCudaDevice, 0);
    if (ret)
    {
        ERR("Failed to get CUDA device, error: %d\n", ret);
        return ret;
    }
    ret = pcuCtxCreate_v2(pCtx, Flags, *pCudaDevice);
    if (ret) ERR("Failed to create cuda context, error: %d\n", ret);
    return ret;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    switch (reason)
    {
        case DLL_PROCESS_ATTACH:
            TRACE("Process attached: (%p)\n", instance);
            if (!load_functions()) return FALSE;
            break;
        case DLL_PROCESS_DETACH:
            if (reserved) break;
            if (cuda_handle) dlclose(cuda_handle);
            break;
        case DLL_THREAD_ATTACH:
            break;
        case DLL_THREAD_DETACH:
            cuda_process_tls_callbacks(reason);
            break;
    }

    return TRUE;
}
