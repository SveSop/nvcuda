/*
 * Copyright (C) 2015 Sebastian Lackner
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

#ifndef __WINE_CUDA_H
#define __WINE_CUDA_H

#include <stdint.h>
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;

#define CUDA_CB

#define CUDA_SUCCESS                 0
#define CUDA_ERROR_INVALID_VALUE     1
#define CUDA_ERROR_OUT_OF_MEMORY     2
#define CUDA_ERROR_INVALID_CONTEXT   201
#define CUDA_ERROR_NO_BINARY_FOR_GPU 209
#define CUDA_ERROR_FILE_NOT_FOUND    301
#define CUDA_ERROR_INVALID_HANDLE    400
#define CUDA_ERROR_NOT_SUPPORTED     801
#define CUDA_ERROR_UNKNOWN           999

#define CU_IPC_HANDLE_SIZE                 64
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID     33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID  34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID  50

typedef enum CUdriverProcAddressQueryResult_enum
{
    CU_GET_PROC_ADDRESS_SUCCESS                = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND       = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
}  CUdriverProcAddressQueryResult;

typedef struct CUextSemaphore_st *CUexternalSemaphore;
typedef unsigned long long CUmemGenericAllocationHandle_v1;
typedef CUmemGenericAllocationHandle_v1 CUmemGenericAllocationHandle;
typedef unsigned int CUdeviceptr_v1;
typedef unsigned long long CUdeviceptr_v2;
typedef CUdeviceptr_v2 CUdeviceptr;
typedef int CUGLDeviceList;
typedef int CUaddress_mode;
typedef int CUarray_format;
typedef int CUdevice_v1;
typedef CUdevice_v1 CUdevice;
typedef int CUdevice_attribute;
typedef int CUfilter_mode;
typedef int CUfunc_cache;
typedef int CUfunction_attribute;
typedef int CUipcMem_flags;
typedef int CUjitInputType;
typedef int CUjit_option;
typedef int CUlimit;
typedef int CUmemorytype;
typedef int CUpointer_attribute;
typedef int CUresourceViewFormat;
typedef int CUresourcetype;
typedef int CUresult;
typedef int CUsharedconfig;
typedef int CUstreamCaptureStatus;
typedef int CUstreamCaptureMode;
typedef int CUgraphMem_attribute;
typedef int CUmemPool_attribute;
typedef int CUmemAllocationGranularity_flags;
typedef int CUmemRangeHandleType;
typedef int CUtensorMapIm2ColWideMode;
typedef int CUtensorMapDataType;
typedef int CUtensorMapInterleave;
typedef int CUtensorMapSwizzle;
typedef int CUtensorMapL2promotion;
typedef int CUtensorMapFloatOOBfill;
typedef int CUprocessState;
typedef unsigned int CUlogIterator;

typedef void *CUDA_ARRAY_DESCRIPTOR;
typedef void *CUDA_MEMCPY3D_PEER;
typedef void *CUDA_RESOURCE_DESC;
typedef void *CUDA_RESOURCE_VIEW_DESC;
typedef void *CUDA_TEXTURE_DESC;
typedef void *CUDA_NODE_PARAMS;
typedef void *CUarray;
typedef void *CUcontext;
typedef void *CUdevprop;
typedef void *CUevent;
typedef void *CUfunction;
typedef void *CUgraphicsResource;
typedef void *CUlinkState;
typedef void *CUmipmappedArray;
typedef void *CUmodule;
typedef void *CUlibrary;
typedef void *CUstream;
typedef void *CUsurfref;
typedef void *CUtexref;
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef void *CUgraphNode;
typedef void *CUmemoryPool;
typedef void *CUmemAllocationProp;
typedef void *CUmoduleLoadingMode;
typedef void (CUDA_CB *CUhostFn)(void *userData);
typedef void *CUlaunchConfig;
typedef void *CUkernel;
typedef void *CUgreenCtx;
typedef void *CUmemcpyAttributes_v1;
typedef void *CUDA_MEMCPY3D_BATCH_OP_v1;
typedef void *CUtensorMap;
typedef void *CUmemDecompressParams;
typedef void *CUcheckpointLockArgs;
typedef void *CUcheckpointCheckpointArgs;
typedef void *CUcheckpointRestoreArgs;
typedef void *CUcheckpointUnlockArgs;
typedef void *CUlogsCallback;
typedef void *CUlogsCallbackHandle;
typedef void *CUmemLocation_v1;
typedef void *CUmemAllocationType;
typedef void *CUdevResource;

typedef unsigned long long CUsurfObject;
typedef unsigned long long CUtexObject;
typedef CUmemcpyAttributes_v1 CUmemcpyAttributes;
typedef CUDA_MEMCPY3D_BATCH_OP_v1 CUDA_MEMCPY3D_BATCH_OP;
typedef CUmemLocation_v1 CUmemLocation;

typedef enum CUexternalMemoryHandleType_enum
{
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD          = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32       = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   = 3,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP         = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE     = 5,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE     = 6,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} CUexternalMemoryHandleType;

typedef enum CUexternalSemaphoreHandleType_enum
{
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD             = 1,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32          = 2,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT      = 3,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE           = 4,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE           = 5,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC             = 6,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX     = 7,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
} CUexternalSemaphoreHandleType;

typedef enum CUctx_flags_enum
{
    CU_CTX_SCHED_AUTO          = 0x00,
    CU_CTX_SCHED_SPIN          = 0x01,
    CU_CTX_SCHED_YIELD         = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_BLOCKING_SYNC       = 0x04,
    CU_CTX_SCHED_MASK          = 0x07,
    CU_CTX_MAP_HOST            = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10,
    CU_CTX_COREDUMP_ENABLE     = 0x20,
    CU_CTX_USER_COREDUMP_ENABLE= 0x40,
    CU_CTX_SYNC_MEMOPS         = 0x80,
    CU_CTX_FLAGS_MASK          = 0xFF
} CUctx_flags;

typedef enum CUatomicOperation_enum
{
    CU_ATOMIC_OPERATION_INTEGER_ADD         = 0,
    CU_ATOMIC_OPERATION_INTEGER_MIN         = 1,
    CU_ATOMIC_OPERATION_INTEGER_MAX         = 2,
    CU_ATOMIC_OPERATION_INTEGER_INCREMENT   = 3,
    CU_ATOMIC_OPERATION_INTEGER_DECREMENT   = 4,
    CU_ATOMIC_OPERATION_AND                 = 5,
    CU_ATOMIC_OPERATION_OR                  = 6,
    CU_ATOMIC_OPERATION_XOR                 = 7,
    CU_ATOMIC_OPERATION_EXCHANGE            = 8,
    CU_ATOMIC_OPERATION_CAS                 = 9,
    CU_ATOMIC_OPERATION_FLOAT_ADD           = 10,
    CU_ATOMIC_OPERATION_FLOAT_MIN           = 11,
    CU_ATOMIC_OPERATION_FLOAT_MAX           = 12,
    CU_ATOMIC_OPERATION_MAX
} CUatomicOperation;

typedef enum {
    CU_DEV_RESOURCE_TYPE_INVALID = 0,
    CU_DEV_RESOURCE_TYPE_SM = 1,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE = 10000,
} CUdevResourceType;

typedef struct CUipcEventHandle_st
{
    char reserved[CU_IPC_HANDLE_SIZE];
} CUipcEventHandle;

typedef struct CUipcMemHandle_st
{
    char reserved[CU_IPC_HANDLE_SIZE];
} CUipcMemHandle;

typedef struct CUuuid_st
{
    char bytes[16];
} CUuuid;

typedef struct CUDA_MEMCPY2D_st
{
    size_t srcXInBytes;
    size_t srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2D_v2;
typedef CUDA_MEMCPY2D_v2 CUDA_MEMCPY2D;

typedef struct CUDA_MEMCPY3D_st
{
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    void *reserved0;
    size_t srcPitch;
    size_t srcHeight;
    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    void *reserved1;
    size_t dstPitch;
    size_t dstHeight;
    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
} CUDA_MEMCPY3D_v2;
typedef CUDA_MEMCPY3D_v2 CUDA_MEMCPY3D;

typedef struct CUDA_ARRAY3D_DESCRIPTOR_st
{
    size_t Width;
    size_t Height;
    size_t Depth;
    CUarray_format Format;
    unsigned int NumChannels;
    unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR_v2;
typedef CUDA_ARRAY3D_DESCRIPTOR_v2 CUDA_ARRAY3D_DESCRIPTOR;

typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
{
    CUexternalMemoryHandleType type;
    union
    {
        int fd;
        struct
        {
            void *handle;
            const void *name;
        } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;
typedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 CUDA_EXTERNAL_MEMORY_HANDLE_DESC;

typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
{
    CUexternalSemaphoreHandleType type;
    union
    {
        int fd;
        struct
        {
            void *handle;
            const void *name;
        } win32;
        const void* nvSciSyncObj;
    } handle;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;
typedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;

typedef struct CU_DEV_SM_RESOURCE_GROUP_PARAMS_st {
    unsigned int smCount;
    unsigned int coscheduledSmCount;
    unsigned int preferredCoscheduledSmCount;
    unsigned int flags;
    unsigned int reserved[12];
} CU_DEV_SM_RESOURCE_GROUP_PARAMS;

#endif /* __WINE_CUDA_H */
