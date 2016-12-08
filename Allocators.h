// Andrew Meckling
#pragma once

#include "Memory.h"

#include <cassert>

// usage: ALLOC( <allocator>, <type> )
//    or: ALLOC( <allocator>, <type> )( <direct-init> )
//    or: ALLOC( <allocator>, <type> ){ <list-init> }
#define ALLOC( ALLOCATOR, TYPE ) \
    new( (ALLOCATOR).allocate( sizeof( TYPE ) ).ptr ) (TYPE)

// usage: ALLOC_N( <allocator>, <type>, <length> )
//    or: ALLOC_N( <allocator>, <type>, <length> )( <direct-init?> )
//    or: ALLOC_N( <allocator>, <type>, <length> ){ <aggregate-init> }
#define ALLOC_N( ALLOCATOR, TYPE, LENGTH ) Array< TYPE >{ nullptr, LENGTH } \
    * new( (ALLOCATOR).allocate( sizeof( TYPE ) * (LENGTH) ).ptr ) (TYPE[ LENGTH ])

// usage: DEALLOC( <allocator>, <block> )
#define DEALLOC( ALLOCATOR, BLOCK ) (ALLOCATOR).deallocate( BLOCK )


// ALLOC_N macro required this operator to function as intended.
template< typename T >
constexpr Array< T > operator *( Array< T > arr, T* ptr )
{
    return { ptr, arr.count };
}

// Rounds a size (of memory) up to the smallest multiple of align.
constexpr size_t round_to_alignment( size_t size, size_t align )
{
    return size % align ? size + align - size % align : size;
}


// Default allocator in most cases. Calls malloc(size_t) and free(void*).
class Mallocator
{
public:

    Blk allocate( size_t size )
    {
        return { malloc( size ), size };
    }

    void deallocate( Blk blk )
    {
        free( blk.ptr );
    }
};

constexpr bool operator ==( Mallocator, Mallocator )
{
    return true;
}

// So-called "null" allocator. Always fails to allocate memory.
// Do not deallocate a Blk that wasn't returned by the associated
// allocate function.
class NullAllocator
{
public:

    constexpr Blk allocate( size_t )
    {
        return { nullptr, 0 };
    }

    void deallocate( Blk blk ) noexcept
    {
        assert( owns( blk.ptr ) );
    }

    constexpr bool owns( void* ptr )
    {
        return ptr == nullptr;
    }
};

constexpr bool operator ==( NullAllocator, NullAllocator )
{
    return true;
}


// Minimum interface for an allocator.
struct IAllocator
{
    virtual Blk allocate( size_t ) = 0;
    virtual void deallocate( Blk ) = 0;
};

// Polymorphism compatible type wrapper around a non-virtual allocator.
template< typename Allocator >
class PolymorphicAllocator
    : public IAllocator
    , protected Allocator
{
public:

    Blk allocate( size_t n )
    {
        return Allocator::allocate( n );
    }

    void deallocate( Blk b )
    {
        return Allocator::deallocate( b );
    }

    bool owns( void* p )
    {
        return Allocator::owns( p );
    }
};



// Provides a static instance of a default-initialized Allocator object.
template< typename Allocator, size_t Index = 0 >
struct StaticAllocator
{
    using Allocator = Allocator;
    static constexpr size_t INDEX = Index;

    static Allocator sInstance;

    Blk allocate( size_t n )
    {
        return sInstance.allocate( n );
    }

    void deallocate( Blk b )
    {
        return sInstance.deallocate( b );
    }

    bool owns( void* p )
    {
        return sInstance.owns( p );
    }
};

template< typename Allo, size_t I >
constexpr bool operator ==( StaticAllocator< Allo, I >, StaticAllocator< Allo, I > )
{
    return true;
}


template< typename Allocator, size_t Index = 0 >
struct ThreadStaticAllocator
{
    using Allocator = Allocator;
    static constexpr size_t INDEX = Index;

    thread_local static Allocator sInstance;

    Blk allocate( size_t n )
    {
        return sInstance.allocate( n );
    }

    void deallocate( Blk b )
    {
        return sInstance.deallocate( b );
    }

    bool owns( void* p )
    {
        return sInstance.owns( p );
    }
};

template< typename Allo, size_t I >
constexpr bool operator ==( ThreadStaticAllocator< Allo, I >,
                            ThreadStaticAllocator< Allo, I > )
{
    return true;
}



// Calls the Fallback allocator if the Primary allocator fails to allocate memory.
template< class Primary, class Fallback >
class FallbackAllocator
    : protected Primary
    , protected Fallback
{
    using P = Primary;
    using F = Fallback;
public:

    Blk allocate( size_t n )
    {
        Blk r = P::allocate( n );
        if ( !r.ptr ) r = F::allocate( n );
        return r;
    }

    void deallocate( Blk b )
    {
        if ( P::owns( b.ptr ) ) P::deallocate( b );
        else F::deallocate( b );
    }

    bool owns( void* p )
    {
        return P::owns( p ) || F::owns( p );
    }
};



// Allocates units of memory greater than Threshold in size with the 
// Larger allocator and the rest with the Smaller allocator.
template< size_t Threshold, class Smaller, class Larger >
class ThresholdAllocator
    : protected Smaller
    , protected Larger
{
    using S = Smaller;
    using L = Larger;
public:

    static constexpr size_t THRESHOLD = Threshold;

    Blk allocate( size_t size )
    {
        return size <= THRESHOLD ? S::allocate( size )
            : L::allocate( size );
    }

    void deallocate( Blk blk )
    {
        if ( blk.size <= THRESHOLD ) S::deallocate( blk );
        else L::deallocate( blk );
    }

    bool owns( void* ptr )
    {
        return S::owns( ptr ) || L::owns( ptr );
    }
};



// Allocates memory from this object to its users. Deallocation is ignored
// for all but the most recent unit of allocation.
template< size_t Bytes, size_t Align = sizeof( int ) >
class StackAllocator
{
public:

    static constexpr size_t SIZE = Bytes;
    static constexpr size_t ALIGNMENT = Align;

    byte* ptr = memory;  // Pointer to the top of the allocation stack.
    byte memory[ SIZE ]; // Local store of allocatable memory.

    Blk allocate( size_t size )
    {
        size_t rounded_size = _round_to_aligned( size );
        if ( rounded_size > memory + SIZE - ptr )
            return { nullptr, 0 };

        Blk result = { ptr, size };
        ptr += rounded_size;
        return result;
    }

    void deallocate( Blk blk )
    {
        assert( owns( blk.ptr ) );
        if ( blk.ptr + _round_to_aligned( blk.size ) == ptr )
            ptr = blk.ptr;
    }

    constexpr bool owns( void* ptr )
    {
        return memory <= ptr && ptr < memory + SIZE;
    }

private:

    static constexpr size_t _round_to_aligned( size_t size )
    {
        return round_to_alignment( size, ALIGNMENT );
    }
};


// Allocates memory from this object to its users. Maps individual allocation
// units of length Align to bits in a bitset. The overhead cost of this type
// in bytes is (Bytes / Align / 8) or O(n/8).
template< size_t Bytes, size_t Align = sizeof( int ) >
class BitsetAllocator
{
public:

    using Word = unsigned __int64;

    static constexpr size_t SIZE = Bytes;
    static constexpr size_t ALIGNMENT = Align;
    static constexpr size_t BITS_PER_WORD = 8 * sizeof( Word );

    static_assert( SIZE % ALIGNMENT == 0, "" );

    Word bitset[ SIZE / ALIGNMENT / BITS_PER_WORD + 1 ] = { 0 };
    byte memory[ SIZE ];

    Blk allocate( size_t size ) noexcept
    {
        size_t rounded_size = _round_to_aligned( size );
        size_t bitlen = rounded_size / ALIGNMENT;

        for ( size_t pos = 0; pos < SIZE - rounded_size; )
        {
            if ( _checkRange( pos, bitlen, false, &pos ) )
            {
                _setRange( pos, bitlen, true );
                #ifdef _DEBUG
                std::memset( memory + pos, 0xbb, size );
                #endif
                return { memory + pos, size };
            }
        }
        return { nullptr, size };
    }

    void deallocate( Blk blk ) noexcept
    {
        if ( blk.ptr == nullptr )
            return;

        assert( owns( blk.ptr ) );
        #ifdef _DEBUG
        blk.set( 0xdd );
        #endif
        size_t pos = (byte*) blk.ptr - memory;
        size_t bitlen = _round_to_aligned( blk.size ) / ALIGNMENT;
        assert( _checkRange( pos, bitlen, true ) );
        _setRange( pos, bitlen, false );
    }

    constexpr bool owns( void* ptr )
    {
        return memory <= ptr && ptr < memory + SIZE;
    }

private:

    // Sets a range in the bitset to bit.
    void _setRange( size_t pos, size_t len, bool bit ) noexcept
    {
        Word* pWord0 = &bitset[ pos / BITS_PER_WORD ];
        size_t idx0 = pos % BITS_PER_WORD;

        Word* pWord1 = &bitset[ (pos + len) / BITS_PER_WORD ];
        size_t idx1 = BITS_PER_WORD - (pos + len) % BITS_PER_WORD;

        for ( Word* pWord = pWord0; pWord <= pWord1; ++pWord )
        {
            Word mask = ~0;

            if ( pWord == pWord0 )
                mask = mask >> idx0 << idx0;

            if ( pWord == pWord1 )
                mask = mask << idx1 >> idx1;

            if ( bit )
                *pWord |= mask;
            else
                *pWord &= ~mask;
        }
    }

    // Returns true if all bits in the range in bitset are equal to bit.
    // Otherwise, sets outpos to the index of the first failing bit and
    // returns false.
    bool _checkRange( size_t pos, size_t len, bool bit,
                      size_t* outpos = nullptr ) const noexcept
    {
        const Word* pWord0 = &bitset[ pos / BITS_PER_WORD ];
        size_t idx0 = pos % BITS_PER_WORD;

        const Word* pWord1 = &bitset[ (pos + len) / BITS_PER_WORD ];
        size_t idx1 = BITS_PER_WORD - (pos + len) % BITS_PER_WORD;

        for ( const Word* pWord = pWord0; pWord <= pWord1; ++pWord )
        {
            Word mask = ~0;

            if ( pWord == pWord0 )
                mask = mask >> idx0 << idx0;

            if ( pWord == pWord1 )
                mask = mask << idx1 >> idx1;

            if ( (*pWord & mask) != (bit ? mask : 0) )
            {
                if ( outpos != nullptr )
                {
                    size_t offset = _skip( *pWord, mask, bit ) * ALIGNMENT;
                    size_t chunk = (pWord - bitset) * BITS_PER_WORD;
                    *outpos = _round_to_aligned( chunk + offset );
                }
                return false;
            }
        }
        return true;
    }

    static size_t _skip( Word word, Word mask, bool bit )
    {
        size_t offset;
        if ( bit ) {
            offset =  _first_0( word & mask );    // Skip 1s
            offset += _first_1( word >> offset ); // Skip 0s
        } else {
            offset =  _first_1( word & mask );    // Skip 0s
            offset += _first_0( word >> offset ); // Skip 1s
        }
        return std::min( offset, BITS_PER_WORD );
    }

    static size_t _first_1( Word word )
    {
        size_t pos = 0;
        while ( !(word & 1) && ++pos < BITS_PER_WORD )
            word >>= 1;
        return pos;
    }

    static size_t _first_0( Word word )
    {
        size_t pos = 0;
        while ( (word & 1) && ++pos < BITS_PER_WORD )
            word >>= 1;
        return pos;
    }

    static constexpr size_t _round_to_aligned( size_t size )
    {
        return round_to_alignment( size, ALIGNMENT );
    }
};
