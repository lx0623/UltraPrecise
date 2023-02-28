/*
 * test_nullable.cu
 *
 *  Created on: Jun 21, 2019
 *      Author: lichi
 */

#include "test_common.h"
#include <iostream>
using namespace std;

ARIES_HOST_DEVICE bool str_like2( const char* text, const char* regexp, int len )
{
    const char* star = nullptr;
    const char* ss = text;
    const char* end = ss + len;
    while( text < end && *text )
    {
        if( *regexp == '%' )
        {
            star = regexp++;
            ss = text;
            continue;
        }
        if( *regexp == '_' || *regexp == *text )
        {
            ++text;
            ++regexp;
            continue;
        }
        if( star )
        {
            regexp = star + 1;
            text = ++ss;
            continue;
        }
        return false;
    }
    while( *regexp == '%' )
        ++regexp;
    return !*regexp;
}
ARIES_HOST_DEVICE bool str_like( const char* text, const char* regexp, int len )
{
    const char* posStar = 0;
    const char* flagInS = 0;
    const char* end = text + len;
    while( *text && text < end )
    {
        if( *regexp && ( *text == *regexp || *regexp == '_' ) )
        {
            regexp++;
            text++;
        }
        else if( *regexp && *regexp == '%' )
        {
            flagInS = text;
            posStar = regexp;
            ++regexp;
        }
        else if( posStar )
        {
            text = ++flagInS;
            regexp = posStar + 1;
        }
        else
        {
            return false;
        }
    }
    while( *regexp && *regexp == '%' )
    {
        ++regexp;
    }

    return ( *text == 0 || text == end ) && *regexp == 0;
}

TEST( nullable, division )
{
    string str = "ab%xx";
    string reg = "ab%x";
    cout<<str_like2( str.c_str(), reg.c_str(), str.size() )<<endl;

    try {
            throw std::exception();
        } catch (std::exception e) {

        }
        nullable_type< aries_acc::Decimal > d( "56586554400.73" );
        nullable_type< aries_acc::Decimal > r(d);

        nullable_type< int64_t > v = 1478493;
        r /= v;
        nullable_type< aries_acc::Decimal > result = r;
        if( r.flag )
        {
            char result[64];
            cout << r.value.GetDecimal( result ) << endl;
        }
}

TEST( nullable, null_null_promotion )
{
    nullable_type< int32_t > a = 10000000;
    nullable_type< int8_t > b = -3;
    nullable_type< int64_t > c = 300;
    nullable_type< float > d = 100.0f;
    nullable_type< double > e = 200.0;
    ASSERT_TRUE( bool( std::is_same< decltype(a * c + b + d - e), nullable_type< double >>::value));
    ASSERT_TRUE( bool( std::is_same< decltype(a * c + b / d ), nullable_type< float >>::value));
    ASSERT_TRUE( bool( std::is_same< decltype(a * b /c), nullable_type< int64_t >>::value));
    ASSERT_EQ( a * b + c, -29999700);
    ASSERT_EQ( c / 3, 100);
    ASSERT_EQ( c % 3, 0);
    ASSERT_EQ( 3000 / c, 10);
    ASSERT_EQ( 3 % c, 3);
}

TEST( nullable, null_value_promotion )
{
    int32_t a = 10000000;
    nullable_type< int8_t > b = -3;
    int64_t c = 300;
    float d = 100.0f;
    double e = 200.0;
    ASSERT_TRUE( bool( std::is_same< decltype(a * c + b + d - e), nullable_type< double >>::value));
    ASSERT_TRUE( bool( std::is_same< std::common_type< int16_t, nullable_type< int8_t > >::type, nullable_type< int32_t > >::value));
    ASSERT_TRUE( bool( std::is_same< std::common_type< float, nullable_type< int > >::type, nullable_type< float > >::value));
    ASSERT_EQ( (a * b + c), -29999700);
}

TEST( nullable, null_calc )
{
    int32_t a = 10000000;
    nullable_type< int8_t > b
    { 0, -3 };
    int64_t c = 300;
    float d = 100.0f;
    double e = 200.0;
    ASSERT_TRUE( bool( std::is_same< decltype(a * c + b + d - e), nullable_type< double >>::value));
    ASSERT_TRUE( (a * c + b + d - e).is_null() );
}

TEST( nullable, cmp_same )
{
    nullable_type< int8_t > a
    { 0, 10 };
    nullable_type< int8_t > b
    { 0, 10 };
    ASSERT_TRUE( ( a == b ).is_unknown() );
    ASSERT_TRUE( ( a < b ).is_unknown() );
    ASSERT_TRUE( ( a > b ).is_unknown() );

    a.flag = 1;
    ASSERT_TRUE( a == 10 );
    ASSERT_TRUE( a > 0 );
    ASSERT_TRUE( a < 11 );

    ASSERT_TRUE( ( b == 10 ).is_unknown() );
    ASSERT_TRUE( ( b > 0 ).is_unknown() );
    ASSERT_TRUE( ( b < 11 ).is_unknown() );
    b.flag = 1;
    ASSERT_TRUE( a == b );
}

TEST( nullable, and_or_same )
{
    nullable_type< int8_t > a
    { 0, 10 };
    nullable_type< int8_t > b
    { 0, 0 };
    ASSERT_TRUE( ( a && b ).is_unknown() );
    ASSERT_TRUE( ( a || b ).is_unknown() );
    ASSERT_TRUE( AriesBool( a ).is_unknown() );
    ASSERT_TRUE( AriesBool( !a ).is_unknown() );
    ASSERT_TRUE( AriesBool( b ).is_unknown() );
    ASSERT_TRUE( AriesBool( !b ).is_unknown() );
    a.flag = 1;

    ASSERT_TRUE( ( a && b ).is_unknown() );
    ASSERT_TRUE( a || b );

    b.flag = 1;
    ASSERT_FALSE( a && b );
    ASSERT_TRUE( a || b );
}

TEST( nullable, and_or_diff )
{
    nullable_type< int8_t > a
    { 0, 10 };
    int32_t b = 0;
    ASSERT_FALSE( a && b );
    ASSERT_TRUE( ( a || b ).is_unknown() );
    a.flag = 1;

    ASSERT_FALSE( a && b );
    ASSERT_TRUE( a || b );

    b = 1;
    ASSERT_TRUE( a && b );
    ASSERT_TRUE( a || b );

    a.flag = 0;
    ASSERT_TRUE( ( a && b ).is_unknown() );
    ASSERT_TRUE( a || b );
}

TEST( nullable, cmp_diff )
{
    nullable_type< int8_t > a
    { 0, 10 };
    nullable_type< int64_t > b
    { 0, 10 };
    ASSERT_TRUE( ( a == b ).is_unknown() );
    ASSERT_TRUE( ( a < b ).is_unknown() );
    ASSERT_TRUE( ( a > b ).is_unknown() );
    a.flag = 1;
    ASSERT_TRUE( a == 10 );
    ASSERT_TRUE( a > 0 );
    ASSERT_TRUE( a < 11 );

    ASSERT_TRUE( ( b == 10 ).is_unknown() );
    ASSERT_TRUE( ( b > 0 ).is_unknown() );
    ASSERT_TRUE( ( b < 11 ).is_unknown() );
    b.flag = 1;
    ASSERT_TRUE( a == b );
}

TEST( nullable, cmp_diff2 )
{
    nullable_type< int8_t > a
    { 1, 10 };
    nullable_type< int64_t > b
    { 0, 10 };
    ASSERT_TRUE( 10 == a );
    ASSERT_TRUE( 0 < a );
    ASSERT_TRUE( 11 > a );

    ASSERT_TRUE( ( 10 == b ).is_unknown() );
    ASSERT_TRUE( ( 0 < b ).is_unknown() );
    ASSERT_TRUE( ( 11 > b ).is_unknown() );
    b.flag = 1;
    ASSERT_TRUE( a == b );
}

//int main( int argc, char** argv )
//{
//    int aa = 10000000;
//    int bb = -3;
//    int cc = 300;
//    long rr = aa * cc + bb;
//    cout<<"rr:"<<rr<<endl;
//    nullable_type< int32_t > a = 10000000;
//    nullable_type< int16_t > b = -3;
//    nullable_type< int32_t > c = 300;
//    nullable_type< float > d = 100.0f;
//    nullable_type< double > e = 200.0;
//    assert(bool(std::is_same<decltype(a * c + b + d), nullable_type< float >>::value));
////    ASSERT_TRUE(bool(std::is_same<decltype(a * a), nullable_type< int32_t >>::value));
//    standard_context_t context;
//    mem_t< nullable_type< int64_t > > result( 1, context );
//    nullable_type< int64_t >* pResult = result.data();
//    transform( [=] ARIES_DEVICE(int index)
//            {
//                pResult[ 0 ] = a * (nullable_type< int64_t >)c + b;
//                auto p = a * c + b;
//                printf( "p.flag=%d, p.value=%ld\n", p.flag, p.value );
//            }, 1, context );
//    context.synchronize();
//    cout << "result.flag:" << ( int )pResult[ 0 ].flag << ", result.value:" << pResult[ 0 ].value << endl;
//    //printf( "result.flag=%d, result.value=%ld\n", pResult[ 0 ].flag, pResult[ 0 ].value );
//    //a = 30000;
////    b = { 1, 10 };
////    c = 0;
////    transform( [=] ARIES_DEVICE(int index)
////            {
////                auto d = ( a * a / 1000 ) / c + b;
////                printf( "result.flag=%d, result.value=%d\n", d.flag, d.value );
////            }, 1, context );
////    context.synchronize();
//}
