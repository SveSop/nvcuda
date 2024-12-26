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

#ifndef __WINE_OPTICALFLOW_H
#define __WINE_OPTICALFLOW_H

typedef int VkFormat;
typedef enum _NV_OF_STATUS
{
    NV_OF_SUCCESS,
    NV_OF_ERR_OF_NOT_AVAILABLE,
    NV_OF_ERR_UNSUPPORTED_DEVICE,
    NV_OF_ERR_DEVICE_DOES_NOT_EXIST,
    NV_OF_ERR_INVALID_PTR,
    NV_OF_ERR_INVALID_PARAM,
    NV_OF_ERR_INVALID_CALL,
    NV_OF_ERR_INVALID_VERSION,
    NV_OF_ERR_OUT_OF_MEMORY,
    NV_OF_ERR_NOT_INITIALIZED,
    NV_OF_ERR_UNSUPPORTED_FEATURE,
    NV_OF_ERR_GENERIC,
} NV_OF_STATUS;

typedef void *NvOFHandle;
typedef void *NvOFGPUBufferHandle;
typedef void *NvOFPrivDataHandle;

#endif /* __WINE_OPTICALFLOW_H */
