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
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "windef.h"
#include "winbase.h"
#include "winternl.h"
#include "winioctl.h"
#include "ntstatus.h"
#include "wine/debug.h"
#include "wine/list.h"
#include "wine/server.h"
#include "cuda.h"
#include "nvcuda.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvcuda);

struct tls_callback_entry
{
    struct list entry;
    void (CDECL *callback)(DWORD, void *);
    void *userdata;
    ULONG count;
};

static struct list tls_callbacks = LIST_INIT( tls_callbacks );

static RTL_CRITICAL_SECTION tls_callback_section;
static RTL_CRITICAL_SECTION_DEBUG critsect_debug =
{
    0, 0, &tls_callback_section,
    { &critsect_debug.ProcessLocksList, &critsect_debug.ProcessLocksList },
      0, 0, { (DWORD_PTR)(__FILE__ ": tls_callback_section") }
};
static RTL_CRITICAL_SECTION tls_callback_section = { &critsect_debug, -1, 0, 0, 0, 0 };

void cuda_process_tls_callbacks(DWORD reason)
{
    // Check if the list is empty before entering the critical section
    if (list_empty(&tls_callbacks))
        return;
    
    struct list *ptr;

    TRACE("(%d)\n", reason);

    EnterCriticalSection( &tls_callback_section );
    ptr = list_head( &tls_callbacks );
    while (ptr)
    {
        struct tls_callback_entry *callback = LIST_ENTRY( ptr, struct tls_callback_entry, entry );
        callback->count++;

        TRACE("calling handler %p(0, %p)\n", callback->callback, callback->userdata);
        callback->callback(0, callback->userdata);
        TRACE("handler %p returned\n", callback->callback);

        ptr = list_next( &tls_callbacks, ptr );
        if (!--callback->count)  /* removed during execution */
        {
            list_remove( &callback->entry );
            HeapFree( GetProcessHeap(), 0, callback );
        }
    }
    LeaveCriticalSection( &tls_callback_section );
}

#define IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE CTL_CODE(FILE_DEVICE_VIDEO, 3, METHOD_BUFFERED, FILE_READ_ACCESS)
#define IOCTL_SHARED_GPU_RESOURCE_GETKMT CTL_CODE(FILE_DEVICE_VIDEO, 2, METHOD_BUFFERED, FILE_READ_ACCESS)

HANDLE get_shared_resource_kmt_handle(HANDLE shared_resource)
{
    IO_STATUS_BLOCK iosb;
    obj_handle_t kmt_handle;

    if (NtDeviceIoControlFile(shared_resource, NULL, NULL, NULL, &iosb, IOCTL_SHARED_GPU_RESOURCE_GETKMT,
            NULL, 0, &kmt_handle, sizeof(kmt_handle)))
    {
        ERR("NtDeviceIoControlFile failed for kmt object %p\n", shared_resource);
        return INVALID_HANDLE_VALUE;
    }

    return wine_server_ptr_handle(kmt_handle);
}

int get_shared_resource_fd(HANDLE win32_handle)
{
    IO_STATUS_BLOCK iosb;
    obj_handle_t unix_resource;
    NTSTATUS status;
    int unix_fd = -1;
    HANDLE new_handle = NULL;

    if(NtDuplicateObject(NtCurrentProcess(), win32_handle, NtCurrentProcess(), &new_handle, 0, 0, DUPLICATE_SAME_ACCESS))
    {
       ERR("NtDuplicateObject failed to create handle for %p\n", win32_handle);
       return -1;
    }

    if (NtDeviceIoControlFile(new_handle, NULL, NULL, NULL, &iosb, IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE,
                              NULL, 0, &unix_resource, sizeof(unix_resource)))
    {
        ERR("NtDeviceIoControlFile failed for handle %p\n", new_handle);
        return -1;
    }

    status = wine_server_handle_to_fd(wine_server_ptr_handle(unix_resource), FILE_READ_DATA, &unix_fd, NULL);
    if (status != STATUS_SUCCESS || unix_fd < 0)
    {
        ERR("Failed to convert Unix resource to FD for handle %p - Status: 0x%x\n", win32_handle, status);
        return -1;
    }
    NtClose(wine_server_ptr_handle(unix_resource));

    return unix_fd;
}

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

struct cuda_table
{
    int size;
    void *functions[0];
};

/*
 * Relay1
 */
struct Relay1_table
{
    int size;
    void* (WINAPI *func0)(void *param0, void *param1);
    void* (WINAPI *func1)(void *param0, void *param1);
    void* (WINAPI *func2)(void *param0, void *param1);
    void* (WINAPI *func3)(void *param0, void *param1);
    void* (WINAPI *func4)(void *param0);
    void* (WINAPI *func5)(void *param0, void *param1, void *param2, void *param3, void *param4);
    void* (WINAPI *func6)(void *param0, void *param1);
    void* (WINAPI *func7)(void *param0, void *param1);
    void* (WINAPI *func8)(void *param0, void *param1);
    void* (WINAPI *func9)(void *param0, void *param1);
    void* (WINAPI *func10)(void *param0, void *param1);
    void* (WINAPI *func11)(void *param0, void *param1);
};
static const struct
{
    int size;
    void* (*func0)(void *param0, void *param1);
    void* (*func1)(void *param0, void *param1);
    void* (*func2)(void *param0, void *param1);
    void* (*func3)(void *param0, void *param1);
    void* (*func4)(void *param0);
    void* (*func5)(void *param0, void *param1, void *param2, void *param3, void *param4);
    void* (*func6)(void *param0, void *param1);
    void* (*func7)(void *param0, void *param1);
    void* (*func8)(void *param0, void *param1);
    void* (*func9)(void *param0, void *param1);
    void* (*func10)(void *param0, void *param1);
    void* (*func11)(void *param0, void *param1);
} *Relay1_orig = NULL;

/*
 * Relay2
 */
struct Relay2_table
{
    int size;
    void* (WINAPI *func0)(void *param0, void *param1);
    void* (WINAPI *func1)(void *param0, void *param1);
    void* (WINAPI *func2)(void *param0, void *param1, void *param2);
    void* (WINAPI *func3)(void *param0, void *param1);
    void* (WINAPI *func4)(void *param0, void *param1);
    void* (WINAPI *func5)(void *param0, void *param1);
};
static const struct
{
    int size;
    void* (*func0)(void *param0, void *param1);
    void* (*func1)(void *param0, void *param1);
    void* (*func2)(void *param0, void *param1, void *param2);
    void* (*func3)(void *param0, void *param1);
    void* (*func4)(void *param0, void *param1);
    void* (*func5)(void *param0, void *param1);
} *Relay2_orig = NULL;

/*
 * Relay3
 */
struct Relay3_table
{
    int size;
    void* (WINAPI *func0)(void *param0);
    void* (WINAPI *func1)(void *param0);
};
static const struct
{
    int size;
    void* (*func0)(void *param0);
    void* (*func1)(void *param0);
} *Relay3_orig = NULL;

/*
 * ContextStorage
 */
struct ContextStorage_table
{
    CUresult (WINAPI *Set)(CUcontext ctx, void *key, void *value, void *callback);
    CUresult (WINAPI *Remove)(CUcontext ctx, void *key);
    CUresult (WINAPI *Get)(void **value, CUcontext ctx, void *key);
};
static const struct
{
    CUresult (*Set)(CUcontext ctx, void *key, void *value, void *callback);
    CUresult (*Remove)(CUcontext ctx, void *key);
    CUresult (*Get)(void **value, CUcontext ctx, void *key);
} *ContextStorage_orig = NULL;

/*
 * Relay5
 */
struct Relay5_table
{
    int size;
    void* (WINAPI *func0)(void *param0, void *param1, void *param2);
};
static const struct
{
    int size;
    void* (*func0)(void *param0, void *param1, void *param2);
} *Relay5_orig = NULL;


/*
 * TlsNotifyInterface
 */
struct TlsNotifyInterface_table
{
    int size;
    CUresult (WINAPI *Set)(void **handle, void *callback, void *data);
    CUresult (WINAPI *Remove)(void *handle, void *param1);
};

static void* WINAPI Relay1_func0(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func0(param0, param1);
}

static void* WINAPI Relay1_func1(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func1(param0, param1);
}

static void* WINAPI Relay1_func2(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func2(param0, param1);
}

static void* WINAPI Relay1_func3(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func3(param0, param1);
}

static void* WINAPI Relay1_func4(void *param0)
{
    TRACE("(%p)\n", param0);
    return Relay1_orig->func4(param0);
}

static void* WINAPI Relay1_func5(void *param0, void *param1, void *param2, void *param3, void *param4)
{
    TRACE("(%p, %p, %p, %p, %p)\n", param0, param1, param2, param3, param4);
    return Relay1_orig->func5(param0, param1, param2, param3, param4);
}

static void* WINAPI Relay1_func6(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func6(param0, param1);
}

static void* WINAPI Relay1_func7(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func7(param0, param1);
}

static void* WINAPI Relay1_func8(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func8(param0, param1);
}

static void* WINAPI Relay1_func9(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func9(param0, param1);
}

static void* WINAPI Relay1_func10(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func10(param0, param1);
}

static void* WINAPI Relay1_func11(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay1_orig->func11(param0, param1);
}

static struct Relay1_table Relay1_Impl =
{
    sizeof(struct Relay1_table),
    Relay1_func0,
    Relay1_func1,
    Relay1_func2,
    Relay1_func3,
    Relay1_func4,
    Relay1_func5,
    Relay1_func6,
    Relay1_func7,
    Relay1_func8,
    Relay1_func9,
    Relay1_func10,
    Relay1_func11,
};

static void* WINAPI Relay2_func0(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay2_orig->func0(param0, param1);
}

static void* WINAPI Relay2_func1(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay2_orig->func1(param0, param1);
}

static void* WINAPI Relay2_func2(void *param0, void *param1, void *param2)
{
    TRACE("(%p, %p, %p)\n", param0, param1, param2);
    return Relay2_orig->func2(param0, param1, param2);
}

static void* WINAPI Relay2_func3(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay2_orig->func3(param0, param1);
}

static void* WINAPI Relay2_func4(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay2_orig->func4(param0, param1);
}

static void* WINAPI Relay2_func5(void *param0, void *param1)
{
    TRACE("(%p, %p)\n", param0, param1);
    return Relay2_orig->func5(param0, param1);
}

static struct Relay2_table Relay2_Impl =
{
    sizeof(struct Relay2_table),
    Relay2_func0,
    Relay2_func1,
    Relay2_func2,
    Relay2_func3,
    Relay2_func4,
    Relay2_func5,
};

static void* WINAPI Relay3_func0(void *param0)
{
    TRACE("(%p)\n", param0);
    return Relay3_orig->func0(param0);
}

static void* WINAPI Relay3_func1(void *param0)
{
    TRACE("(%p)\n", param0);
    return Relay3_orig->func1(param0);
}

static struct Relay3_table Relay3_Impl =
{
    sizeof(struct Relay3_table),
    Relay3_func0,
    Relay3_func1,
};

struct context_storage
{
    void *value;
    void (WINAPI *callback)(CUcontext ctx, void *key, void *value);
};

static void storage_destructor_callback(CUcontext ctx, void *key, void *value)
{
    struct context_storage *storage = value;

    TRACE("(%p, %p, %p)\n", ctx, key, value);

    if (storage->callback)
    {
        TRACE("calling destructor callback %p(%p, %p, %p)\n",
              storage->callback, ctx, key, storage->value);
        storage->callback(ctx, key, storage->value);
        TRACE("destructor callback %p returned\n", storage->callback);
    }

    HeapFree( GetProcessHeap(), 0, storage );
}

static CUresult WINAPI ContextStorage_Set(CUcontext ctx, void *key, void *value, void *callback)
{
    struct context_storage *storage;

    TRACE("(%p, %p, %p, %p)\n", ctx, key, value, callback);

    storage = HeapAlloc( GetProcessHeap(), 0, sizeof(*storage) );
    if (!storage)
        return CUDA_ERROR_OUT_OF_MEMORY;

    storage->callback = callback;
    storage->value = value;

    CUresult ret = ContextStorage_orig->Set(ctx, key, storage, storage_destructor_callback);
    if (ret) HeapFree( GetProcessHeap(), 0, storage );
    return ret;
}

static CUresult WINAPI ContextStorage_Remove(CUcontext ctx, void *key)
{
    struct context_storage *storage;

    TRACE("(%p, %p)\n", ctx, key);

    /* FIXME: This is not completely race-condition save, but using a mutex
     * could have a relatively big overhead. Can still be added later when it
     * turns out to be necessary. */
    if (!ContextStorage_orig->Get((void **)&storage, ctx, key))
        HeapFree( GetProcessHeap(), 0, storage );

    return ContextStorage_orig->Remove(ctx, key);
}

static CUresult WINAPI ContextStorage_Get(void **value, CUcontext ctx, void *key)
{
    struct context_storage *storage;

    TRACE("(%p, %p, %p)\n", value, ctx, key);

    CUresult ret = ContextStorage_orig->Get((void **)&storage, ctx, key);
    if (!ret) *value = storage->value;
    return ret;
}

static struct ContextStorage_table ContextStorage_Impl =
{
    ContextStorage_Set,
    ContextStorage_Remove,
    ContextStorage_Get,
};

static void* WINAPI Relay5_func0(void *param0, void *param1, void *param2)
{
    TRACE("(%p, %p, %p)\n", param0, param1, param2);
    return Relay5_orig->func0(param0, param1, param2);
}

static struct Relay5_table Relay5_Impl =
{
    sizeof(struct Relay5_table),
    Relay5_func0,
};

static CUresult WINAPI TlsNotifyInterface_Set(void **handle, void *callback, void *userdata)
{
    struct tls_callback_entry *new_entry;

    TRACE("(%p, %p, %p)\n", handle, callback, userdata);

    new_entry = HeapAlloc( GetProcessHeap(), 0, sizeof(*new_entry) );
    if (!new_entry)
        return CUDA_ERROR_OUT_OF_MEMORY;

    new_entry->callback = callback;
    new_entry->userdata = userdata;
    new_entry->count = 1;

    EnterCriticalSection( &tls_callback_section );
    list_add_tail( &tls_callbacks, &new_entry->entry );
    LeaveCriticalSection( &tls_callback_section );

    *handle = new_entry;
    return CUDA_SUCCESS;
}

static CUresult WINAPI TlsNotifyInterface_Remove(void *handle, void *param1)
{
    CUresult ret = CUDA_ERROR_INVALID_VALUE;
    struct tls_callback_entry *to_free = NULL;
    struct list *ptr;

    TRACE("(%p, %p)\n", handle, param1);

    if (param1)
        FIXME("semi stub: param1 != 0 not supported.\n");

    EnterCriticalSection( &tls_callback_section );
    LIST_FOR_EACH( ptr, &tls_callbacks )
    {
        struct tls_callback_entry *callback = LIST_ENTRY( ptr, struct tls_callback_entry, entry );
        if (callback == handle)
        {
            if (!--callback->count)
            {
                list_remove( ptr );
                to_free = callback;
            }
            ret = CUDA_SUCCESS;
            break;
        }
    }
    LeaveCriticalSection( &tls_callback_section );
    HeapFree( GetProcessHeap(), 0, to_free );
    return ret;
}

static struct TlsNotifyInterface_table TlsNotifyInterface_Impl =
{
    sizeof(struct TlsNotifyInterface_table),
    TlsNotifyInterface_Set,
    TlsNotifyInterface_Remove,
};

static BOOL cuda_check_table(const struct cuda_table *orig, struct cuda_table *impl, const char *name)
{
    if (!orig)
        return FALSE;

    /* FIXME: better check for size, verify that function pointers are != NULL */

    if (orig->size > impl->size)
    {
        FIXME("WARNING: Your CUDA version supports a newer interface for %s then the Wine implementation.\n", name);
        FIXME("WARNING: Driver implementation size: %d, Wine implementation size: %d\n", orig->size, impl->size);
    }
    else if (orig->size < impl->size)
    {
        FIXME("Your CUDA version supports only an older interface for %s, downgrading version.\n", name);
        FIXME("WARNING: Driver implementation size: %d, Wine implementation size: %d\n", orig->size, impl->size);
        impl->size = orig->size;
    }

    return TRUE;
}

static inline BOOL cuda_equal_uuid(const CUuuid *id1, const CUuuid *id2)
{
    return !memcmp(id1, id2, sizeof(CUuuid));
}

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

CUresult cuda_get_table(const void **table, const CUuuid *uuid, const void *orig_table, CUresult orig_result)
{
    char buffer[128];

    if (cuda_equal_uuid(uuid, &UUID_Relay1))
    {
        TRACE("(%p, Relay1_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        if (orig_result)
            return orig_result;
        if (!cuda_check_table(orig_table, (void *)&Relay1_Impl, "Relay1"))
            return CUDA_ERROR_UNKNOWN;

        Relay1_orig = orig_table;
        *table = (void *)&Relay1_Impl;
        return CUDA_SUCCESS;
    }
    else if (cuda_equal_uuid(uuid, &UUID_Relay2))
    {
        TRACE("(%p, Relay2_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        if (orig_result)
            return orig_result;
        if (!cuda_check_table(orig_table, (void *)&Relay2_Impl, "Relay2"))
            return CUDA_ERROR_UNKNOWN;

        Relay2_orig = orig_table;
        *table = (void *)&Relay2_Impl;
        return CUDA_SUCCESS;
    }
    else if (cuda_equal_uuid(uuid, &UUID_Relay3))
    {
        TRACE("(%p, Relay3_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        if (orig_result)
            return orig_result;
        if (!cuda_check_table(orig_table, (void *)&Relay3_Impl, "Relay3"))
            return CUDA_ERROR_UNKNOWN;

        Relay3_orig = orig_table;
        *table = (void *)&Relay3_Impl;
        return CUDA_SUCCESS;
    }
    else if (cuda_equal_uuid(uuid, &UUID_ContextStorage))
    {
        TRACE("(%p, ContextStorage_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        if (orig_result)
            return orig_result;
        if (!orig_table)
            return CUDA_ERROR_UNKNOWN;

        ContextStorage_orig = orig_table;
        *table = (void *)&ContextStorage_Impl;
        return CUDA_SUCCESS;
    }
    else if (cuda_equal_uuid(uuid, &UUID_Relay5))
    {
        TRACE("(%p, Relay5_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        if (orig_result)
            return orig_result;
        if (!cuda_check_table(orig_table, (void *)&Relay5_Impl, "Relay5"))
            return CUDA_ERROR_UNKNOWN;

        Relay5_orig = orig_table;
        *table = (void *)&Relay5_Impl;
        return CUDA_SUCCESS;
    }
    else if (cuda_equal_uuid(uuid, &UUID_TlsNotifyInterface))
    {
        TRACE("(%p, TlsNotifyInterface_UUID: %s)\n", table, cuda_print_uuid(uuid, buffer, sizeof(buffer)));
        /* the following interface is not implemented in the Linux
         * CUDA driver, we provide a replacement implementation */
        *table = (void *)&TlsNotifyInterface_Impl;
        return CUDA_SUCCESS;
    }

    FIXME("Unknown UUID: %s, error: %d\n", cuda_print_uuid(uuid, buffer, sizeof(buffer)), orig_result);
    return CUDA_ERROR_UNKNOWN;
}
