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

#ifndef __NVCUDA_D3D_H
#define __NVCUDA_D3D_H

CUresult WINAPI wine_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult WINAPI wine_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult WINAPI wine_cuDeviceGet(CUdevice *device, int ordinal);

typedef enum CUd3d11DeviceList_enum {
    CU_D3D11_DEVICE_LIST_ALL            = 0x01, /**< The CUDA devices for all GPUs used by a D3D11 device */
    CU_D3D11_DEVICE_LIST_CURRENT_FRAME  = 0x02, /**< The CUDA devices for the GPUs used by a D3D11 device in its currently rendering frame */
    CU_D3D11_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The CUDA devices for the GPUs to be used by a D3D11 device in the next frame */
} CUd3d11DeviceList;

#endif