/*
 * Copyright (C) 2026 Sveinar Søpler
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

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "wine/debug.h"
#include "cuda.h"
#include "nvcuda.h"
#include "nvcuda_d3d.h"
#include "d3d9.h"
#include "d3d11.h"
#include "function_mappings.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvcuda);

/*
 * Direct3D emulated functions
 */

CUresult WINAPI wine_cuD3D9CtxCreate(CUcontext *pCtx, CUdevice *pCudaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice)
{
    CUresult ret;
    CUdevice dev;

    FIXME("(%p, %p, %u, %p) - semi-stub\n", pCtx, pCudaDevice, Flags, pD3DDevice);

    ret = wine_cuDeviceGet(&dev, 0);
    if (ret) return ret;

    ret = wine_cuCtxCreate(pCtx, Flags, dev);
    if (ret) return ret;

    if (pCudaDevice)
        *pCudaDevice = dev;

    return CUDA_SUCCESS;
}

CUresult WINAPI wine_cuD3D9GetDevice(CUdevice *pCudaDevice, const char *pszAdapterName)
{
    FIXME("(%p, %s) - semi-stub\n", pCudaDevice, pszAdapterName);
    return wine_cuDeviceGet(pCudaDevice, 0);
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
    return wine_cuDeviceGet(pCudaDevice, 0);
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
    return wine_cuDeviceGet(pCudaDevice, 0);
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

    ret = wine_cuDeviceGet(pCudaDevice, 0);
    if (ret)
    {
        ERR("Failed to get CUDA device, error: %d\n", ret);
        return ret;
    }
    ret = wine_cuCtxCreate(pCtx, Flags, *pCudaDevice);
    if (ret) ERR("Failed to create cuda context, error: %d\n", ret);
    return ret;
}

CUresult WINAPI wine_cuD3D11CtxCreate_v2(CUcontext* pCtx, CUdevice* pCudaDevice, unsigned int Flags, ID3D11Device* pD3DDevice)
{
    FIXME("(%p, %p, %u, %p) - semi-stub\n", pCtx, pCudaDevice, Flags, pD3DDevice);
    /* Create a cuda context on the device - This will NOT actually be registered to the D3D11 device. */
    CUresult ret;

    ret = wine_cuDeviceGet(pCudaDevice, 0);
    if (ret)
    {
        ERR("Failed to get CUDA device, error: %d\n", ret);
        return ret;
    }
    ret = wine_cuCtxCreate_v2(pCtx, Flags, *pCudaDevice);
    if (ret) ERR("Failed to create cuda context, error: %d\n", ret);
    return ret;
}