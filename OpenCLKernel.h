#pragma once

#include "Memory.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>

template< typename T >
struct InArray : Array< T >
{
    using Base = Array< T >;
    using Base::Base;
};

template< typename T >
struct OutArray : Array< T >
{
    using Base = Array< T >;
    using Base::Base;
};

template< typename T >
struct InOutArray : Array< T >
{
    using Base = Array< T >;
    using Base::Base;
};


template< typename T >
struct ClMemBridge
{
    using type = T;
};

template< typename T >
struct ClMemBridge< const T* >
{
    typedef InArray< T > type;
};

template< typename T >
struct ClMemBridge< T* >
{
    typedef InOutArray< T > type;
};

template< typename T >
struct ClMemBridge< T[] >
{
    typedef OutArray< T > type;
};

namespace detail
{
    cl_program create_program(
        cl_context          context,
        const char*         fileName,
        const cl_device_id* devices,
        cl_uint             numDevices,
        std::string*        pFirstKernel = nullptr );
}

template< typename... Args >
class OpenCLKernel
{
private:

    static constexpr size_t NUM_ARGS = sizeof...( Args );
    static constexpr size_t MAX_DEVICES = 3;

    cl_uint          _numDevices;
    cl_context       _context;
    cl_command_queue _queues[ MAX_DEVICES ];
    cl_program       _program;
    cl_kernel        _kernel;

    cl_mem _memBuffers[ NUM_ARGS ];

    std::mutex _mutex;

public:
    #pragma region ctors

    OpenCLKernel( cl_platform_id platform,
                  const char*    fileName,
                  cl_device_type deviceTypes = CL_DEVICE_TYPE_DEFAULT,
                  const char*    kernelName = NULL )
    {
        cl_int err;

        cl_device_id devices[ MAX_DEVICES ];
        err = clGetDeviceIDs( platform, deviceTypes, MAX_DEVICES, devices, &_numDevices );

        _context = clCreateContext( 0, _numDevices, devices, 0, 0, &err );
        for ( size_t i = 0; i < _numDevices; ++i )
            _queues[ i ] = clCreateCommandQueue( _context, devices[ i ], 0, &err );

        if ( kernelName != NULL )
        {
            _program = detail::create_program( _context, fileName, devices, _numDevices );
            _kernel = clCreateKernel( _program, kernelName, &err );
        }
        else
        {
            std::string str;
            _program = detail::create_program( _context, fileName, devices, _numDevices, &str );
            _kernel = clCreateKernel( _program, str.c_str(), &err );
        }
    }

    OpenCLKernel( cl_platform_id platform,
                  const char*    fileName,
                  const char*    kernelName )
        : OpenCLKernel( platform, fileName, CL_DEVICE_TYPE_DEFAULT, kernelName )
    {
    }

    explicit OpenCLKernel( const char*    fileName,
                           cl_device_type deviceTypes = CL_DEVICE_TYPE_DEFAULT,
                           const char*    kernelName = NULL,
                           cl_platform_id _unused = nullptr )
        : OpenCLKernel( (clGetPlatformIDs( 1, &_unused, 0 ), _unused),
                        fileName, deviceTypes, kernelName )
    {
    }

    OpenCLKernel( const OpenCLKernel& copy )
        : _numDevices( copy._numDevices )
        , _context( clRetainContext( copy._context ) )
        , _program( clRetainProgram( copy._program ) )
        , _kernel( clCloneKernel( copy._kernel ) )
        , workDim( copy.workDim )
    {
        for ( size_t i = 0; i < _numDevices; ++i )
            _queues[ i ] = clRetainCommandQueue( copy._queues[ i ] );

        for ( int i = 0; i < 3; ++i )
        {
            globalWorkSize[ i ] = copy.globalWorkSize[ i ];
            localWorkSize[ i ] = copy.localWorkSize[ i ];
        }
    }

    ~OpenCLKernel()
    {
        for ( int i = 0; i < NUM_ARGS; ++i )
            if ( _memBuffers[ i ] != nullptr )
                clReleaseMemObject( (cl_mem) _memBuffers[ i ] );

        clReleaseKernel( _kernel );
        clReleaseProgram( _program );
        for ( size_t i = 0; i < _numDevices; ++i )
            clReleaseCommandQueue( _queues[ i ] );
        clReleaseContext( _context );
    }

    #pragma endregion

    int    workDim = -1; // Negative workDim means pass NULL as localWorkSize.
    size_t globalWorkSize[ 3 ] = { 1, 1, 1 };
    size_t localWorkSize[ 3 ] = { 1, 1, 1 };

    void operator ()( typename ClMemBridge< Args >::type... args )
    {
        std::lock_guard< std::mutex > lck( _mutex );

        _setArgs( 0, args... );

        cl_int   err;
        cl_event waitFinish;

        err = clEnqueueNDRangeKernel(
            _queues[ 0 ], _kernel,
            std::abs( workDim ), nullptr /* global_work_offset */,
            globalWorkSize,
            (workDim > 0 ? localWorkSize : nullptr),
            0, nullptr, &waitFinish );

        if ( err )
            return;
        err = clWaitForEvents( 1, &waitFinish );

        _readArgs( 0, args... );

        for ( int i = 0; i < NUM_ARGS; ++i )
            _memBuffers[ i ] = nullptr;
    }

private:
    #pragma region private

    template< typename Arg, typename... Rest >
    void _setArgs( size_t idx, Arg arg, Rest... rest )
    {
        _setArgs( idx, arg );
        _setArgs( idx + 1, rest... );
    }

    template< typename Arg >
    void _setArgs( size_t idx, const Arg& arg )
    {
        cl_int err = clSetKernelArg( _kernel, idx, sizeof( Arg ), (void*) &arg );
    }

    template< typename T >
    void _setArgs( size_t idx, InArray< T > arr )
    {
        Blk blk = arr;
        cl_int err;

        cl_mem buffer = clCreateBuffer( _context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        blk.size, (void*) blk.ptr, &err );

        err = clSetKernelArg( _kernel, idx, sizeof( cl_mem ), (void*) &buffer );

        //err = clEnqueueWriteBuffer( _queues[ 0 ], buffer, CL_TRUE, 0,
        //                            blk.size, blk.ptr,
        //                            0, nullptr, nullptr );

        _memBuffers[ idx ] = buffer;
    }

    template< typename T >
    void _setArgs( size_t idx, OutArray< T > arr )
    {
        Blk blk = arr;
        cl_int err;

        cl_mem buffer = clCreateBuffer( _context, CL_MEM_WRITE_ONLY,
                                        blk.size, nullptr, &err );

        err = clSetKernelArg( _kernel, idx, sizeof( cl_mem ), (void*) &buffer );

        _memBuffers[ idx ] = buffer;
    }

    template< typename T >
    void _setArgs( size_t idx, InOutArray< T > arr )
    {
        Blk blk = arr;
        cl_int err;

        cl_mem buffer = clCreateBuffer( _context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        blk.size, (void*) blk.ptr, &err );

        err = clSetKernelArg( _kernel, idx, sizeof( cl_mem ), (void*) &buffer );

        //err = clEnqueueWriteBuffer( _queues[ 0 ], buffer, CL_TRUE, 0,
        //                            blk.size, blk.ptr,
        //                            0, nullptr, nullptr );

        _memBuffers[ idx ] = buffer;
    }


    template< typename Arg, typename... Rest >
    void _readArgs( size_t idx, Arg arg, Rest... rest )
    {
        _readArgs( idx, arg );
        _readArgs( idx + 1, rest... );
    }

    template< typename Arg >
    void _readArgs( size_t idx, const Arg& )
    {
        if ( _memBuffers[ idx ] != nullptr )
            clReleaseMemObject( (cl_mem) _memBuffers[ idx ] );
    }

    template< typename T >
    void _readArgs( size_t idx, OutArray< T > arr )
    {
        _readBlk( idx, arr );
    }

    template< typename T >
    void _readArgs( size_t idx, InOutArray< T > arr )
    {
        _readBlk( idx, arr );
    }

    void _readBlk( size_t idx, Blk blk )
    {
        cl_int err = clEnqueueReadBuffer(
            _queues[ 0 ], _memBuffers[ idx ], CL_TRUE, 0,
            blk.size, blk.ptr, 0, nullptr, nullptr );

        if ( !err )
            clReleaseMemObject( (cl_mem) _memBuffers[ idx ] );
    }

    #pragma endregion

};

inline cl_program detail::create_program(
    cl_context          context,
    const char*         fileName,
    const cl_device_id* devices, 
    cl_uint             numDevices,
    std::string*        pFirstKernel )
{
    cl_int err;
    cl_program program;

    std::ifstream kernelFile( fileName, std::ios::in );
    if ( !kernelFile.is_open() )
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string str = oss.str();
    const char* cstr = str.c_str();

    // Find the name of the first kernel.
    if ( pFirstKernel )
    {
        size_t kernel_pos = str.find( "__kernel void " ) + sizeof( "__kernel void" );
        kernel_pos = str.find_last_not_of( " \t\r\n", kernel_pos );
        size_t kernel_len = str.find_first_of( '(', kernel_pos ) - kernel_pos;
        *pFirstKernel = str.substr( kernel_pos, kernel_len );
    }

    program = clCreateProgramWithSource( context, 1, &cstr, NULL, NULL );
    if ( program == NULL )
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    err = clBuildProgram( program, numDevices, devices, NULL, NULL, NULL );
    if ( err != CL_SUCCESS )
    {
        // Determine the reason for the error
        char buildLog[ 16384 ];
        clGetProgramBuildInfo( program, *devices, CL_PROGRAM_BUILD_LOG,
                               sizeof( buildLog ), buildLog, NULL );

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram( program );
        return NULL;
    }

    return program;
}
