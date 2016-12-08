
typedef unsigned char byte; // 8-bit bitfield.

__kernel void grayscale( __global byte* image )
{
    int i = get_global_id( 0 ) * 4;

    float gray = image[ i + 0 ] * 0.21
               + image[ i + 1 ] * 0.72
               + image[ i + 2 ] * 0.07;
    image[ i + 0 ] = gray;
    image[ i + 1 ] = gray;
    image[ i + 2 ] = gray;
}
