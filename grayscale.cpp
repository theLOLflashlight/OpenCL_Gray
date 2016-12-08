// Andrew Meckling
// Nav Bhatti
#include "lodepng.h"
#include "OpenCLKernel.h"

#include <stdlib.h>
#include <iostream>
#include <thread>

#include <chrono>

using std::cout;
using std::endl;

void serialGrayscale( std::vector< byte >& image )
{
    for ( int i = 0; i < image.size(); i += 4 )
    {
        byte gray = byte( image[ i + 0 ] * 0.21
                        + image[ i + 1 ] * 0.72
                        + image[ i + 2 ] * 0.07 );
        image[ i + 0 ] = gray;
        image[ i + 1 ] = gray;
        image[ i + 2 ] = gray;
    }
}

// Converts to grayscale on serially. Returns timing.
auto _serial( std::vector< byte >& image, size_t size )
{
    using namespace std::chrono;
    auto start = steady_clock::now();

    serialGrayscale( image );

    auto end = steady_clock::now();
    return end - start;
}

// Converts to grayscale on gpu. Returns timing.
auto _gpu( std::vector< byte >& image, size_t size )
{
    using namespace std::chrono;
    
    cl_platform_id platforms[ 3 ];
    clGetPlatformIDs( 3, platforms, 0 );

    OpenCLKernel< byte* > grayscale( platforms[ 0 ], "grayscale.cl" );
    grayscale.globalWorkSize[ 0 ] = size;

    auto start = steady_clock::now();
    
    grayscale( image );

    auto end = steady_clock::now();
    return end - start;
}

// Converts to grayscale on cpu. Returns timing.
auto _cpu( std::vector< byte >& image, size_t size )
{
    using namespace std::chrono;

    cl_platform_id platforms[ 3 ];
    clGetPlatformIDs( 3, platforms, 0 );

    OpenCLKernel< byte* > grayscale( platforms[ 2 ], "grayscale.cl" );
    grayscale.globalWorkSize[ 0 ] = size;

    auto start = steady_clock::now();
    
    grayscale( image );

    auto end = steady_clock::now();
    return end - start;
}


// Converts to grayscale on gpu and cpu. Returns timing.
auto _gcpu( std::vector< byte >& image, size_t size )
{
    using namespace std::chrono;

    size_t half_len = image.size() / 2;

    cl_platform_id platforms[ 3 ];
    clGetPlatformIDs( 3, platforms, 0 );

    OpenCLKernel< byte* > grayscaleGPU( platforms[ 0 ], "grayscale.cl" );
    grayscaleGPU.globalWorkSize[ 0 ] = size / 2;

    OpenCLKernel< byte* > grayscaleCPU( platforms[ 2 ], "grayscale.cl" );
    grayscaleCPU.globalWorkSize[ 0 ] = size / 2;

    auto start = steady_clock::now();

    std::thread gpu( [&]() {
        grayscaleGPU( { image.data(), half_len } );
    } );
    std::thread cpu( [&]() {
        grayscaleCPU( { image.data() + half_len, half_len } );
    } );
    gpu.join();
    cpu.join();

    auto end = steady_clock::now();
    return end - start;
}


int main( int argc, char** argv )
{
    std::vector< byte > image;
    unsigned width, height;

    if ( unsigned error = lodepng::decode( image, width, height, "input.png" ) ) {
        cout << "decoder error " << error << ": " << lodepng_error_text( error ) << endl;
        return error;
    }

    size_t size = width * height;

    printf( "starting programs\n" );
    using namespace std::chrono;

    auto serial_diff = _serial( image, size );
    cout << "serial took " << (duration_cast< nanoseconds >( serial_diff ).count() / 1'000'000.0) << " ms\n";

    auto gpu_diff = _gpu( image, size );
    cout << "gpu took " << (duration_cast< nanoseconds >( gpu_diff ).count() / 1'000'000.0) << " ms\n";

    auto cpu_diff = _cpu( image, size );
    cout << "cpu took " << (duration_cast< nanoseconds >( cpu_diff ).count() / 1'000'000.0) << " ms\n";

    auto gcpu_diff = _gcpu( image, size );
    cout << "gpu and cpu took " << (duration_cast< nanoseconds >( gcpu_diff ).count() / 1'000'000.0) << " ms\n";

    if ( unsigned error = lodepng::encode( "output.png", image, width, height ) ) {
        cout << "encoder error " << error << ": " << lodepng_error_text( error ) << endl;
        return error;
    }

    std::system( "pause" );
}


// vv   This didn't work out as planned   vv

/*template< typename T >
T reflect( T m, T x )
{
    if ( x < 0 )
        return -x - 1;
    if ( x >= m )
        return 2 * m - x - 1;
    return x;
}

double G( double x, double y, double sd )
{
    using namespace std;

    return exp( -((x*x + y*y) / (2 * sd*sd)) ) / (2 * CL_M_PI * sd*sd);
}

double G( double x, double sd )
{
    using namespace std;

    return exp( -(x*x / (2 * sd*sd)) ) / (sqrt( 2 * CL_M_PI ) * sd);
}


//auto gauss_kernel( double sigma )
//{
//
//    std::deque< double > kernel;
//
//    for ( int x =  )
//
//    return kernel;
//}


double gaussian( double x, double mu, double sigma )
{
    return std::exp( -(((x-mu)/(sigma))*((x-mu)/(sigma)))/2.0 );
}

using kernel_row = std::vector< double >;
using kernel_type = std::vector< kernel_row >;



kernel_type produce2dGaussianKernel( int kernelRadius )//, double sigma )
{
    double sigma = kernelRadius / 3.0;
    size_t kernel_size = 2 * kernelRadius + 1;

    kernel_type kernel2d( kernel_size, kernel_row( kernel_size ) );
    double sum = 0;
    double max = 0;
    // compute values
    for (int row = 0; row < kernel_size; row++)
        for (int col = 0; col < kernel_size; col++) {
            double x = G( col - kernelRadius,
                          row - kernelRadius,
                          sigma );
            //* gaussian(col, kernelRadius, sigma);
            kernel2d[row][col] = x;
            sum += x;
            if ( x > max )
                max = x;
        }

    //double G00 = kernel2d[ (kernel_size - 1) / 2 ][ (kernel_size - 1) / 2 ];
    double G00 = kernel2d[ (kernel_size - 1) / 2 ][ 0 ];
    // normalize
    for (int row = 0; row < kernel_size; row++)
        for (int col = 0; col < kernel_size; col++)
            kernel2d[row][col] /= sum;
    return kernel2d;
}

void gauss_blur( const std::vector< byte >& in,
                 std::vector< byte >&       out,
                 int width, int height, int radius )
{
    auto w = width;
    auto h = height;
    auto r = radius;

    double coeffs[] = {
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765
    };

    auto kern = produce2dGaussianKernel( r );
    size_t kern_size = kern.size();
    double kern_mid = kern[ (kern_size - 1) / 2 ][ (kern_size - 1) / 2 ];

    for ( int y = 0; y < h; y++ )
    {
        for ( int x = 0; x < w; x++ )
        {
            double sum[ 4 ] = { 0 };

            for ( int i = 0; i < r; i++ )
            {
                for ( int j = 0; j < r; j++ )
                {
                    int y1 = reflect( h, y - (r / 2) + i );
                    int x1 = reflect( w, x - (r / 2) + j );

                    int px = y1 * w + x1;
                    for ( int k = 0; k < 4; ++k )
                        sum[ k ] += in[ px * 4 + k ] / 255.0 * kern[ i ][ j ];// coeffs[ i * r + j ];
                                                                              //sum[ 3 ] += in[ px * 4 + 4 ];
                }
            }

            sum[ 3 ] = 1;
            for ( int k = 0; k < 4; k++ )
            {
                out[ (y * w + x) * 4 + k ] = sum[ k ] * 255 * r;// / (r / 2.);
            }
        }
    }
}*/
