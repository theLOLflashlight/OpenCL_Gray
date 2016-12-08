// Andrew Meckling
#pragma once

#include <algorithm>
#include <cstring>
#include <malloc.h>
#include <vector>

using byte = unsigned __int8; // 8-bit bitfield.

// Represents a unit of allocated memory.
struct Blk
{
    void*  ptr;  // Pointer to allocated memory.
    size_t size; // Length of allocated memory in bytes.

    Blk() = default;

    // Direct initialization constructor.
    constexpr Blk( void* ptr, size_t size )
        : ptr( ptr ), size( size )
    {
    }

    // Conversion from nullptr (for convenience).
    constexpr Blk( std::nullptr_t )
        : ptr( nullptr ), size( 0 )
    {
    }

    // Conversion from typed pointer.
    template< typename T >
    constexpr Blk( T* ptr )
        : ptr( ptr ), size( sizeof( T ) )
    {
    }

    template< typename T, size_t N >
    Blk( T (&)[ N ] ) = delete;

    // std::memsets the block with val.
    void set( byte val )
    {
        std::memset( ptr, val, size );
    }
};

// Represents an array of units of allocated memory.
template< typename T >
struct Array
{
    static_assert( sizeof( T ) > 0, "struct Array<T>: T cannot be void" );

    T*     array; // Typed pointer to allocated array of elements.
    size_t count; // Length of array in number of elements.

    Array() = default;

    // Direct initialization constructor.
    constexpr Array( T* array, size_t count )
        : array( array ), count( count )
    {
    }

    // Conversion from static array.
    template< size_t N >
    constexpr Array( T (&array)[ N ] )
        : array( array ), count( N )
    {
    }

    // Conversion from std::vector (for convenience).
    Array( std::vector< T >& vector )
        : Array( vector.data(), vector.size() )
    {
    }

    // Conversion to Blk (for convenience).
    constexpr operator Blk() const
    {
        return { array, count * sizeof( T ) };
    }

    // Conversion to typed pointer.
    constexpr operator T*() const
    {
        return array;
    }

    // std::fills the array with val.
    void fill( const T& val )
    {
        std::fill( array, array + count, val );
    }
};
