/*
* test.cxx
*
*  Created on: 2019年6月24日
*      Author: david
*/

#include <string.h>
#include <string>
#include <gtest/gtest.h>
#include "decimal.hxx"

using namespace aries_acc;
using namespace std;


TEST(UT_decimal, compare)
{
Decimal d1 = Decimal( 10, 2, "99999999.99" );
Decimal d11 = Decimal( 10, 2, "99999999.99" );
ASSERT_TRUE((bool) d1);
ASSERT_TRUE(bool(d1));
ASSERT_TRUE(d1 == d11);
ASSERT_TRUE(d1 <= d11);
ASSERT_TRUE(d1 >= d11);
ASSERT_FALSE(d1 != d11);
ASSERT_FALSE(d1 < d11);
ASSERT_FALSE(d1 > d11);
Decimal d12 = Decimal( 15, 2, "99999999.99" );
ASSERT_TRUE(d1 == d12);
Decimal d13 = Decimal( 15, 3, "99999999.99" );
ASSERT_TRUE(d1 == d13);
Decimal d2 = Decimal( 15, 3, "99999999.999" );
ASSERT_TRUE(d2 > d1);
ASSERT_TRUE(d2 >= d1);
ASSERT_TRUE(d2 != d1);
ASSERT_FALSE(d2 < d1);
ASSERT_FALSE(d2 <= d1);
ASSERT_FALSE(d2 == d1);
Decimal d3 = Decimal( 25, 10, "99999999.999" );
ASSERT_TRUE(d3 > d1);
ASSERT_TRUE(d3 >= d1);
ASSERT_TRUE(d3 != d1);
ASSERT_FALSE(d3 < d1);
ASSERT_FALSE(d3 <= d1);
ASSERT_FALSE(d3 == d1);
ASSERT_TRUE(d3 == d2);
ASSERT_TRUE(d3 <= d2);
ASSERT_TRUE(d3 >= d2);
ASSERT_FALSE(d3 != d2);
ASSERT_FALSE(d3 < d2);
ASSERT_FALSE(d3 > d3);
Decimal d4 = Decimal( 25, 10, "99999999.0000000001" );
ASSERT_TRUE(d4 < d3);
ASSERT_TRUE(d4 <= d3);
ASSERT_TRUE(d4 != d3);
ASSERT_FALSE(d4 > d3);
ASSERT_FALSE(d4 >= d3);
ASSERT_FALSE(d4 == d3);

Decimal d5 = Decimal( 10, 2, "99999999" );
int i32 = 99999999;
ASSERT_TRUE(d5 == i32);
ASSERT_TRUE(d5 <= i32);
ASSERT_TRUE(d5 >= i32);
ASSERT_FALSE(d5 != i32);
ASSERT_FALSE(d5 < i32);
ASSERT_FALSE(d5 > i32);
int64_t i64 = 99999999;
ASSERT_TRUE(d5 == i64);
ASSERT_TRUE(d5 <= i64);
ASSERT_TRUE(d5 >= i64);
ASSERT_FALSE(d5 != i64);
ASSERT_FALSE(d5 < i64);
ASSERT_FALSE(d5 > i64);
}

TEST(UT_decimal, add_decimal_1)
{
char temp[128];

// printf("\n");
// Decimal decimal50 = Decimal(11, 4, "1635619.3155");
// printf("int main::被除数x :: x ::sign = %d prec = %d frac = %d ,v = %09d %09d %09d %09d %09d\n",GET_SIGN(decimal50.prec),GET_CALC_PREC(decimal50.prec),GET_CALC_FRAC(decimal50.frac),decimal50.v[4],decimal50.v[3],decimal50.v[2],decimal50.v[1],decimal50.v[0]); 
// Decimal decimal51 = Decimal(12, 4, "47497517.7405");
// printf("int main::  除数y :: y ::sign = %d prec = %d frac = %d ,v = %09d %09d %09d %09d %09d\n",GET_SIGN(decimal51.prec),GET_CALC_PREC(decimal51.prec),GET_CALC_FRAC(decimal51.frac),decimal51.v[4],decimal51.v[3],decimal51.v[2],decimal51.v[1],decimal51.v[0]); 
// decimal50 /= decimal51;
// printf("int main::    商z :: z ::sign = %d prec = %d frac = %d ,v = %09d %09d %09d %09d %09d\n",GET_SIGN(decimal50.prec),GET_CALC_PREC(decimal50.prec),GET_CALC_FRAC(decimal50.frac),decimal50.v[4],decimal50.v[3],decimal50.v[2],decimal50.v[1],decimal50.v[0]); 
// printf("GetDecimal = %s\n",decimal50.GetDecimal(temp));
// Decimal decimal = Decimal(12,2,"0.04");
// Decimal ans = 1 - decimal;
// printf("%s\n",ans.GetDecimal(temp));
// for(int i=0;i<5000000-1;i++){
//     Decimal decimal1 = Decimal(12,2,"21.33");
//     if(i%1000000==1){
//         Decimal decimal2 = Decimal(10,0,"0");
//         decimal += decimal2;
//     }
//     decimal += decimal1;
// }
// printf(" %09d %09d %09d %09d %09d\n",decimal.v[4],decimal.v[3],decimal.v[2],decimal.v[1],decimal.v[0]);
// Decimal decimal1 = Decimal("0.01");
// printf(" %09d %09d %09d %09d %09d\n",decimal.v[4],decimal.v[3],decimal.v[2],decimal.v[1],decimal.v[0]);
// printf(" %09d %09d %09d %09d %09d\n",decimal1.v[4],decimal1.v[3],decimal1.v[2],decimal1.v[1],decimal1.v[0]);
// decimal += decimal1;
// char temp[128];
// ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.01");
}

TEST(UT_decimal, add_decimal_2)
{
Decimal decimal = Decimal(0);
Decimal decimal1 = Decimal("-0.01");
// printf(" %09d %09d %09d %09d %09d\n",decimal.v[4],decimal.v[3],decimal.v[2],decimal.v[1],decimal.v[0]);
// printf(" %09d %09d %09d %09d %09d\n",decimal1.v[4],decimal1.v[3],decimal1.v[2],decimal1.v[1],decimal1.v[0]);
decimal += decimal1;
char temp[128];
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-0.01");
}

TEST(UT_decimal, add_decimal_3)
{
char temp[128];
Decimal decimal = Decimal(23, 12, "9123456789.123456789012");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "9123456789.123456789012");
Decimal decimal1 = Decimal(12, 5, "1234567.11111");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "1234567.11111");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetDecimal(temp)), "9124691356.234566789012");
}

// 此tast需要修改
TEST(UT_decimal, add_decimal_4)
{
char temp[128];
Decimal decimal = Decimal(3, 0, "1");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1");
Decimal decimal1 = Decimal(3, 0, "-2");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-2");
// printf(" %09d %09d %09d %09d %09d\n",decimal.v[4],decimal.v[3],decimal.v[2],decimal.v[1],decimal.v[0]);
// printf(" %09d %09d %09d %09d %09d\n",decimal1.v[4],decimal1.v[3],decimal1.v[2],decimal1.v[1],decimal1.v[0]);
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "1,0"); 
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-1");
}

TEST(UT_decimal, add_decimal_5)
{
char temp[128];
Decimal decimal = Decimal(36, 0, "999999999999999999999999999999999999");
Decimal decimal1 = Decimal(36, 0, "999999999999999999999999999999999999");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "37,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1999999999999999999999999999999999998");
}

TEST(UT_decimal, add_decimal_6)
{
char temp[128];
Decimal decimal = Decimal(36, 0, "-999999999999999999999999999999999999");
Decimal decimal1 = Decimal(36, 0, "-999999999999999999999999999999999999");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "37,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-1999999999999999999999999999999999998");
}

// 此tast需要修改
TEST(UT_decimal, add_decimal_7)
{
char temp[128];

Decimal decimal = Decimal(36, 0, "-999999999999999999999999999999999999");
Decimal decimal1 = Decimal(36, 0, "999999999999999999999999999999999999");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "1,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0");
}

TEST(UT_decimal, add_decimal_8)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "999999999999999999999999999999.111111");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999999999999999999999999999.111111");
Decimal decimal1 = Decimal(36, 5, "9999999999999999999999999900000.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "9999999999999999999999999900000.00001");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "38,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "10999999999999999999999999899999.111121");
}

TEST(UT_decimal, add_decimal_9)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "-999999999999999999999999999999.111111");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-999999999999999999999999999999.111111");
Decimal decimal1 = Decimal(36, 5, "-9999999999999999999999999900000.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-9999999999999999999999999900000.00001");
decimal += decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "38,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-10999999999999999999999999899999.111121");
}

TEST(UT_decimal, sub_decimal_0)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "999999999999999999999999999999.111111");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999999999999999999999999999.111111");
Decimal decimal1 = Decimal(36, 5, "9999999999999999999999999900000.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "9999999999999999999999999900000.00001");
decimal1 -= decimal;
ASSERT_EQ(string(decimal1.GetPrecisionScale(temp)), "37,6");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "8999999999999999999999999900000.888899");
}

TEST(UT_decimal, sub_decimal_1)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "999999999999999999999999999999.111111");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999999999999999999999999999.111111");
Decimal decimal1 = Decimal(36, 5, "9999999999999999999999999900000.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "9999999999999999999999999900000.00001");
decimal -= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "37,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-8999999999999999999999999900000.888899");
}

TEST(UT_decimal, sub_decimal_2)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "0.111111");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.111111");
Decimal decimal1 = Decimal(36, 5, "-1.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-1.00001");
decimal -= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "7,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1.111121");
}

TEST(UT_decimal, sub_decimal_3)
{
char temp[128];
Decimal decimal = Decimal(36, 6, "0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.000000");
Decimal decimal1 = Decimal(36, 5, "-1.00001");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-1.00001");
decimal -= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "7,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1.000010");
}

TEST(UT_decimal, sub_decimal_4) {
char temp[128];
Decimal d1 = Decimal( 10, 2, "0.08" );
ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "10,2" );
ASSERT_EQ( string( d1.GetDecimal( temp ) ), "0.08" );
Decimal d2 = Decimal("1") - d1;
ASSERT_EQ( string( d2.GetPrecisionScale( temp ) ), "3,2" );
ASSERT_EQ( string( d2.GetDecimal( temp ) ), "0.92" );
}

TEST(UT_decimal, mul_decimal_1)
{
char temp[128];
Decimal decimal = Decimal(9, 0, "999999999");
Decimal decimal1 = Decimal(9, 0, "999999999");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "18,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999998000000001");
}

TEST(UT_decimal, mul_decimal_2)
{
char temp[128];
Decimal decimal = Decimal(9, 0, "0");
Decimal decimal1 = Decimal(9, 0, "999999999");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "1,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0");
}

TEST(UT_decimal, mul_decimal_3)
{
char temp[128];
Decimal decimal = Decimal(9, 0, "999999999");
Decimal decimal1 = Decimal(9, 0, "-1");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "9,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-999999999");
}

TEST(UT_decimal, mul_decimal_4)
{
char temp[128];
Decimal decimal = Decimal(9, 3, "99.99");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "99.990");
Decimal decimal1 = Decimal(9, 2, "0.00");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "0.00");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,5");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.00000");
}

TEST(UT_decimal, mul_decimal_5)
{
char temp[128];
Decimal decimal = Decimal(18, 3, "1000000000000.222");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1000000000000.222");
Decimal decimal1 = Decimal(9, 2, "1.00");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "1.00");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "18,5");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "1000000000000.22200");
}

TEST(UT_decimal, mul_decimal_6)
{
char temp[128];
Decimal decimal = Decimal(18, 3, "999999999.222");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999999.222");
Decimal decimal1 = Decimal(9, 2, "0.09");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "0.09");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "13,5");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "89999999.92998");
}

TEST(UT_decimal, mul_decimal_7)
{
char temp[128];
Decimal decimal = Decimal(26, 11, "-99.99999999999");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-99.99999999999");
Decimal decimal1 = Decimal(12, 2, "9000000000");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "9000000000.00");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "25,13");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-899999999999.9100000000000");
}

TEST(UT_decimal, mul_decimal_8)
{
char temp[128];
Decimal decimal = Decimal(19, 6, "-9999999999999.999999");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-9999999999999.999999");
Decimal decimal1 = Decimal(17, 2, "-999999999999999");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-999999999999999.00");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "36,8");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "9999999999999989999000000000.00000100");
}

TEST(UT_decimal, mul_decimal_9)
{
char temp[128];
Decimal decimal = Decimal(19, 6, "-9999999999999.999999");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-9999999999999.999999");
Decimal decimal1 = Decimal(17, 2, "999999999999999");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "999999999999999.00");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "36,8");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-9999999999999989999000000000.00000100");
}

TEST(UT_decimal, mul_decimal_10)
{
char temp[128];
Decimal decimal = Decimal(26, 0, "99999999999999999999999999");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "99999999999999999999999999");
Decimal decimal1 = Decimal(21, 0, "999999999999999999999");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "999999999999999999999");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "45,0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "999999999999999999899999000000000000000000001");
}

TEST(UT_decimal, mul_decimal_11)
{
char temp[128];
Decimal decimal = Decimal(26, 6, "99999999999999999999.999999");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "99999999999999999999.999999");
Decimal decimal1 = Decimal(21, 10, "99999999999.9999999999");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "99999999999.9999999999");
decimal *= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "45,14");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "9999999999999999999989999900000.00000000000000");
}

TEST(UT_decimal, mul_decimal_12) {
char temp[128];
Decimal d1 = Decimal( 15, 2, "709.87" );
ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "15,2" );
ASSERT_EQ( string( d1.GetDecimal( temp ) ), "709.87" );
Decimal d2 = Decimal( 11, 0, "4540" );
Decimal d3 = d2 * d1;
ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "9,2" );
ASSERT_EQ( string( d3.GetDecimal( temp ) ), "3222809.80" );
}

TEST(UT_decimal, div_decimal_1)
{
char temp[128];
Decimal decimal = Decimal(5, 0, "100");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "100");
Decimal decimal1 = Decimal(3, 0, "10");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "10");
decimal /= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "8,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "10.000000");
}

TEST(UT_decimal, div_decimal_2)
{
char temp[128];
Decimal decimal = Decimal(5, 0, "-100");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-100");
Decimal decimal1 = Decimal(3, 0, "10");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "10");
decimal /= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "8,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-10.000000");
}

TEST(UT_decimal, div_decimal_3)
{
char temp[128];
Decimal decimal = Decimal(5, 0, "-100");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-100");
Decimal decimal1 = Decimal(3, 0, "-10");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-10");
decimal /= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "8,6");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "10.000000");
}

TEST(UT_decimal, div_decimal_4)
{
char temp[128];
Decimal decimal = Decimal(7, 5, "-9.12345");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-9.12345");
Decimal decimal1 = Decimal(6, 4, "-3.9876");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-3.9876");
decimal /= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "12,11");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "2.28795516100");
}

TEST(UT_decimal, div_decimal_5)
{
char temp[128];
Decimal decimal = Decimal(7, 5, "0");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.00000");
Decimal decimal1 = Decimal(6, 4, "-3.9876");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "-3.9876");
decimal /= decimal1;
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "12,11");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.00000000000");
}

TEST(UT_decimal, div_decimal_6)
{
char temp[128];
Decimal decimal = Decimal(7, 5, "-9.12345");
ASSERT_EQ(string(decimal.GetDecimal(temp)), "-9.12345");
Decimal decimal1 = Decimal(6, 4, "-0");
ASSERT_EQ(string(decimal1.GetDecimal(temp)), "0.0000");
}


TEST(UT_decimal, compact_transfer)
{
char temp[128];
char compactBuf[128];
Decimal decimal = Decimal( 1, 0, "1" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "1,0");
int total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
bool r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
Decimal transfered = Decimal((CompactDecimal *)compactBuf, 1, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 1, 1, "0.1" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "1,1");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 1, 1);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 6, 0, "1" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 6, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 6, 6, "0.1" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,6");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 6, 6);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 6, 0, "999999" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 6, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 6, 6, "0.999999" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,6");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 6, 6);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 9, 0, "123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "9,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 9, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 9, 9, "0.123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "9,9");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 9, 9);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 10, 0, "1123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "10,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 10, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 10, 10, "0.1123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "10,10");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 10, 10);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 10, 2, "11234567.89" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "10,2");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 10, 2);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 15, 2, "55511234567.89" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "15,2");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 15, 2);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 36, 0, "5551123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "36,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 36, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 36, 36, "0.5551123456789" );
// ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "30,30");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 36, 36);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

//此test到old中会自动截断为36，0所以无效
decimal = Decimal( 36, 0, "5551123456789" );
// ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "36,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 36, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 37, 37, "0.5551123456789" );
// ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "30,30");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 37, 37);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 37, 30, "12345.5551123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "37,30");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 37, 30);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 6, 0, "-999999" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "6,0");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
// ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 6, 0);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

//超过36他就自动变成36了所以此测试无效
decimal = Decimal( 36, 30, "-12345.5551123456789" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "36,30");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 36, 30);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 14, 7, "-9999999.9999999" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "14,7");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 14, 7);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));

decimal = Decimal( 32, 16, "-9999999999999999.9999999999999999" );
ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "32,16");
total = GetDecimalRealBytes(GET_CALC_PREC(decimal.prec), GET_CALC_FRAC(decimal.frac));
r = decimal.ToCompactDecimal(temp, total);
ASSERT_TRUE(r);
memcpy(compactBuf, temp, total);
transfered = Decimal((CompactDecimal *)compactBuf, 32, 16);
ASSERT_EQ(string(decimal.GetDecimal(temp)), string(transfered.GetDecimal(compactBuf)));
}

TEST(UT_decimal, abs) {
    Decimal d1 = Decimal( 10, 2, "-99999999.99" );
    Decimal d2 = Decimal( 10, 2, "99999999.99" );
    ASSERT_TRUE(d2 <= abs(d1));
}

TEST(UT_decimal, truncate)
{
    char temp[128];
    auto d = Decimal( 8, 3, "11111.227" );
    auto t = d.truncate(2);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "7,2");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "11111.22" );

    Decimal d2( 7, 2, "11111.22" );
    ASSERT_EQ( t, Decimal( 7, 2, "11111.22" ) );
    ASSERT_EQ( t * Decimal("2.2"), d2 * Decimal("2.2") );

    d = Decimal( 8, 3, "11111.227" );
    t = d.truncate(0);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "5,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "11111" );

    d = Decimal( 8, 3, "11111.227" );
    t = d.truncate(30);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "35,30");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "11111.227000000000000000000000000000" );

    d = Decimal( 8, 3, "11111.227" );
    t = d.truncate(-3);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "5,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "11000" );

    d = Decimal( 8, 3, "11111.227" );
    t = d.truncate(-5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "1,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "0" );

    d = Decimal( 8, 3, "11111.227" );
    t = d.truncate(-36);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "1,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "0" );

    d = Decimal( 10, 3, "1111111.227" );
    t = d.truncate(-5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "7,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "1100000" );

    d = Decimal( 20, 3, "1111111.227" );
    t = d.truncate(-5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "17,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "1100000" );

    d = Decimal( 20, 3, "-1111111.227" );
    t = d.truncate(-5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "17,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "-1100000" );

    d = Decimal( 20, 0, "-1111111" );
    t = d.truncate(-5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "20,0");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "-1100000" );

    d = Decimal( 20, 0, "-1111111" );
    t = d.truncate(5);
    ASSERT_EQ( string(t.GetPrecisionScale(temp)), "25,5");
    ASSERT_EQ( string( t.GetDecimal( temp ) ), "-1111111.00000" );

}


TEST(UT_decimal, initialize_1)
{
    Decimal decimal;
    char temp[128];
    ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "10,0");
}

TEST(UT_decimal, initialize_2)
{
    Decimal decimal = Decimal(45, 30);
    char temp[128];
    ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "45,30");
}

TEST(UT_decimal, initialize_3)
{
    Decimal decimal = Decimal(30, 20, 3);
    char temp[128];
    ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "30,20");
}

TEST(UT_decimal, initialize_8)
{
    char temp[128];
    int8_t i8 = 127;
    Decimal d81(i8);
    ASSERT_EQ( string( d81.GetDecimal( temp ) ), "127" );
    i8 = -128;
    Decimal d82(i8);
    ASSERT_EQ( string( d82.GetDecimal( temp ) ), "-128" ); 	
    int16_t i16 = 32767;
    Decimal d161(i16);
    ASSERT_EQ( string( d161.GetDecimal( temp ) ), "32767" );
    i16 = -32768;
    Decimal d162(i16);
    ASSERT_EQ( string( d162.GetDecimal( temp ) ), "-32768" );
    int32_t i32 = 2147483647;
    Decimal d321(i32);
    ASSERT_EQ( string( d321.GetDecimal( temp ) ), "2147483647" );
    i32 = -2147483648;
    Decimal d322(i32);
    ASSERT_EQ( string( d322.GetDecimal( temp ) ), "-2147483648" );
    int64_t i64 = 9223372036854775807;
    Decimal d641(i64);
    ASSERT_EQ( string( d641.GetDecimal( temp ) ), "9223372036854775807" );
    i64 = -9223372036854775807 - 1;
    Decimal d642(i64);
    ASSERT_EQ( string( d642.GetDecimal( temp ) ), "-9223372036854775808" );
}

TEST(UT_decimal, all_2) {
    char temp[128];
    Decimal d1("2316507072.1135");
    Decimal d2("100.00");
    Decimal d3("11347");
    Decimal r = d2 * d3 / d1;
    ASSERT_EQ( string( r.GetPrecisionScale( temp ) ), "9,8" );
    ASSERT_EQ( string( r.GetDecimal( temp ) ), "0.00048983" );
}

/*
* test.cxx
*
*  Created on: 2019年6月24日
*      Author: david
*/

// #include <string.h>
// #include <string>
// #include <gtest/gtest.h>
// #include "decimal.hxx"

// using namespace aries_acc;
// using namespace std;

// TEST(UT_decimal, cast) {
//     char temp[128];
//     Decimal d = Decimal(3, 0, "1234");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "3,0");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "999" );
//     d = Decimal(2, 0).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "2,0");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "99" );
//     d  = Decimal(5, 2, "123");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "5,2");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.00" );
//     d = Decimal(4, 2).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "4,2");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "99.99" );
//     d  = Decimal(6, 2, "123.123");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "6,2");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.12" );
//     d = Decimal(7, 3).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "7,3");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.120" );
//     d  = Decimal(6, 2, "123.12511");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "6,2");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.13" );
//     d = Decimal(2, 0).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "2,0");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "99" );
//     d  = Decimal(12, 8, "123.1251111190000");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "12,8");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.12511112" );
//     d = Decimal(12, 0).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "12,0");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123" );
//     d  = Decimal(12, 9, "123.12511111190000");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "12,9");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.125111112" );
//     d = Decimal(5, 2).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "5,2");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.13" );
//     d  = Decimal(29, 17, "123.1251111119999999990000");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "29,17");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.12511111200000000" );
//     d = Decimal(29, 10).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "29,10");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.1251111120" );
//     d  = Decimal(29, 18, "123.12511111199999999990000");
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "29,18");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.125111112000000000" );
//     d = Decimal(21, 18).cast(d);
//     ASSERT_EQ(string(d.GetPrecisionScale(temp)), "21,18");
//     ASSERT_EQ( string( d.GetDecimal( temp ) ), "123.125111112000000000" );
// }

// TEST(UT_decimal, div_decimal_roundup)
// {
//     char temp[128];
//     Decimal decimal = Decimal( 20, 0, "10000000000000000000" );
//     Decimal decimal1 = Decimal( 20, 0, "12800000000000000000" );
//     decimal /= decimal1;
//     ASSERT_EQ( string( decimal.GetInnerPrecisionScale( temp ) ), "7,6" );
//     ASSERT_EQ( string( decimal.GetInnerDecimal( temp ) ), "0.781250" );
//     ASSERT_EQ(string(decimal.GetPrecisionScale(temp)), "5,4");
//     ASSERT_EQ(string(decimal.GetDecimal(temp)), "0.7813");
// }

TEST(UT_decimal, calc_precision)
{
    char temp[128];
    Decimal decimal = Decimal( 5, 2, "1" );
    Decimal decimal1 = Decimal( 5, 2, "1" );
    decimal.CalcAddTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "6,2" );

    decimal = Decimal( 5, 2, "1" );
    decimal1 = Decimal( 5, 2, "1" );
    decimal.CalcSubTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "6,2" );

    decimal = Decimal( 5, 2, "1" );
    decimal1 = Decimal( 5, 2, "1" );
    decimal.CalcMulTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "10,4" );

    decimal = Decimal( 5, 2, "1" );
    decimal1 = Decimal( 5, 2, "1" );
    decimal.CalcDivTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "11,6" );

    decimal = Decimal( 15, 2, "1" );
    decimal1 = Decimal( 15, 2, "1" );
    decimal.CalcAddTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "16,2" );

    decimal = Decimal( 15, 2, "1" );
    decimal1 = Decimal( 15, 2, "1" );
    decimal.CalcSubTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "16,2" );

    decimal = Decimal( 15, 2, "1" );
    decimal1 = Decimal( 15, 2, "1" );
    decimal.CalcMulTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "30,4" );

    decimal = Decimal( 15, 2, "1" );
    decimal1 = Decimal( 15, 2, "1" );
    decimal.CalcDivTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "21,6" );

    decimal = Decimal( 36, 30, "1" );
    decimal1 = Decimal( 36, 30, "1" );
    decimal.CalcAddTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "37,30" );

    decimal = Decimal( 36, 30, "1" );
    decimal1 = Decimal( 36, 30, "1" );
    decimal.CalcSubTargetPrecision(decimal1);
    ASSERT_EQ( string( decimal.GetPrecisionScale( temp ) ), "37,30" );
}

// TEST(UT_decimal, transOverFlow)
// {
//     char temp[128];
//     Decimal decimal = Decimal( 20, 0, ARIES_MODE_STRICT_ALL_TABLES, "10000000000000000000" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "10000000000000000000");
//     decimal = Decimal( 10, 0, "10000000000000000000" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "9999999999");
//     decimal = Decimal( 10, 0, ARIES_MODE_STRICT_ALL_TABLES, "10000000000000000000" );
//     ASSERT_EQ(decimal.GetError(), ERR_OVER_FLOW);
//     decimal = Decimal( 10, 1, "1.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "1.2");
//     decimal = Decimal( 10, 1, ARIES_MODE_STRICT_ALL_TABLES, "1.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     decimal = Decimal( 4, 2, "12345.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "99.99");
//     decimal = Decimal( 4, 2, ARIES_MODE_STRICT_ALL_TABLES, "12345.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OVER_FLOW);
//     decimal = Decimal( 7, 2, "12345.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "12345.23");
//     decimal = Decimal( 7, 2, ARIES_MODE_STRICT_ALL_TABLES, "12345.23" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "12345.23");
//     decimal = Decimal( 7, 2, "99999.996" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "99999.99");
//     decimal = Decimal( 7, 2, ARIES_MODE_STRICT_ALL_TABLES, "99999.996" );
//     ASSERT_EQ(decimal.GetError(), ERR_OVER_FLOW);
//     decimal = Decimal( 7, 1, "999999.96" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "999999.9");
//     decimal = Decimal( 7, 1, ARIES_MODE_STRICT_ALL_TABLES, "999999.96" );
//     ASSERT_EQ(decimal.GetError(), ERR_OVER_FLOW);
//     decimal = Decimal( 8, 1, "-999999.96" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
//     ASSERT_EQ(string( decimal.GetDecimal( temp ) ), "-1000000.0");
//     decimal = Decimal( 8, 1, ARIES_MODE_STRICT_ALL_TABLES, "-1000000.0" );
//     ASSERT_EQ(decimal.GetError(), ERR_OK);
// }


// TEST(UT_decimal, mod)
// {
//     char temp[128];
//     Decimal d1 = Decimal( 10, 2, "99999999.99" );
//     Decimal d2 = Decimal( 10, 0, "2" );
//     Decimal d3 = d1 % d2;
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "5,2" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "1.99" );
//     float mod1 = 2.0;
//     double dd = d1 % mod1;
//     ASSERT_EQ(to_string(dd), "1.990000");
//     double mod2 = 3.0;
//     dd = d1 % mod2;
//     ASSERT_EQ(to_string(dd), "0.990000");
//     int mod3 = 3;
//     Decimal d4 = d1 % mod3;
//     ASSERT_EQ(string( d4.GetDecimal( temp ) ), "0.99");
//     d2 %= d1;
//     ASSERT_EQ( string( d2.GetInnerPrecisionScale( temp ) ), "10,2" );
//     ASSERT_EQ( string( d2.GetInnerDecimal( temp ) ), "2.00" );
//     ASSERT_EQ( string( d2.GetPrecisionScale( temp ) ), "10,2" );
//     ASSERT_EQ( string( d2.GetDecimal( temp ) ), "2.00" );
// }

// TEST(UT_decimal, all_1)
// {
//     char temp[128];
//     Decimal d1 = Decimal( 10, 2, "99999999.99" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "10,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "99999999.99" );
//     Decimal d2 = Decimal( 13, 4, "123456789.0129" );
//     ASSERT_EQ( string( d2.GetPrecisionScale( temp ) ), "13,4" );
//     ASSERT_EQ( string( d2.GetDecimal( temp ) ), "123456789.0129" );
//     Decimal d3 = Decimal( 12, 3, "987654321.019" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "12,3" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "987654321.019" );
//     Decimal d4 = 2000;
//     ASSERT_EQ( string( d4.GetPrecisionScale( temp ) ), "10,0" );

//     Decimal res = (d1 + d2 - d4) * d4;
//     ASSERT_EQ( string( res.GetInnerPrecisionScale( temp ) ), "17,4" );
//     ASSERT_EQ( string( res.GetInnerDecimal( temp ) ), "446909578005.8000" );
//     ASSERT_EQ( string( res.GetPrecisionScale( temp ) ), "17,4" );
//     ASSERT_EQ( string( res.GetDecimal( temp ) ), "446909578005.8000" );
//     uint32_t t = 4294967295;
//     bool r = res <= t;
//     ASSERT_TRUE( !r );
//     Decimal res1 = (d1 * d2 - d3 / d4) + d1;
//     ASSERT_EQ( string( res1.GetInnerPrecisionScale( temp ) ), "26,9" );
//     ASSERT_EQ( string( res1.GetInnerDecimal( temp ) ), "12345678999561604.939361500" );
//     ASSERT_EQ( string( res1.GetPrecisionScale( temp ) ), "24,7" );
//     ASSERT_EQ( string( res1.GetDecimal( temp ) ), "12345678999561604.9393615" );
//     bool b = res1 > 0;
//     ASSERT_TRUE( b );
// }

// TEST(UT_decimal, all_3) {
//     char temp[128];
//     Decimal d1 = Decimal( 10, 2, "99999999.99" );
//     d1.error = ERR_OK;
//     d1.mode = ARIES_MODE_EMPTY;
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "10,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "99999999.99" );
//     Decimal d2 = Decimal( 10, 0, "2" );
//     d2.error = ERR_OK;
//     d2.mode = ARIES_MODE_EMPTY;
//     ASSERT_EQ( string( d2.GetPrecisionScale( temp ) ), "10,0" );
//     ASSERT_EQ( string( d2.GetDecimal( temp ) ), "2" );
//     Decimal d3 = d1 * d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "199999999.98" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "199999999.98" );
//     d3 = d1 + d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "100000001.99" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "100000001.99" );
//     d3 = d1 - d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "99999997.99" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "99999997.99" );
//     d3 = d1 / d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "16,8" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "49999999.99500000" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "14,6" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "49999999.995000" );
//     d3 = d1 % d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "5,2" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "1.99" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "5,2" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "1.99" );

//     d1 *= d2;
//     ASSERT_EQ( string( d1.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetInnerDecimal( temp ) ), "199999999.98" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "199999999.98" );

//     d1 = Decimal( 10, 2, "99999999.99" );
//     d1.error = ERR_OK;
//     d1.mode = ARIES_MODE_EMPTY;
//     d2 = Decimal( 10, 0, "2" );
//     d2.error = ERR_OK;
//     d2.mode = ARIES_MODE_EMPTY;
//     d1 += d2;
//     ASSERT_EQ( string( d1.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetInnerDecimal( temp ) ), "100000001.99" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "100000001.99" );
//     d1 = Decimal( 10, 2, "99999999.99" );
//     d1.error = ERR_OK;
//     d1.mode = ARIES_MODE_EMPTY;
//     d2 = Decimal( 10, 0, "2" );
//     d2.error = ERR_OK;
//     d2.mode = ARIES_MODE_EMPTY;
//     d1 -= d2;
//     ASSERT_EQ( string( d1.GetInnerPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetInnerDecimal( temp ) ), "99999997.99" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "11,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "99999997.99" );
//     d1 = Decimal( 10, 2, "99999999.99" );
//     d1.error = ERR_OK;
//     d1.mode = ARIES_MODE_EMPTY;
//     d2 = Decimal( 10, 0, "2" );
//     d2.error = ERR_OK;
//     d2.mode = ARIES_MODE_EMPTY;
//     d1 /= d2;
//     ASSERT_EQ( string( d1.GetInnerPrecisionScale( temp ) ), "16,8" );
//     ASSERT_EQ( string( d1.GetInnerDecimal( temp ) ), "49999999.99500000" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "14,6" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "49999999.995000" );
//     d1 = Decimal( 10, 2, "99999999.99" );
//     d1.error = ERR_OK;
//     d1.mode = ARIES_MODE_EMPTY;
//     d2 = Decimal( 10, 0, "2" );
//     d2.error = ERR_OK;
//     d2.mode = ARIES_MODE_EMPTY;
//     d1 %= d2;
//     ASSERT_EQ( string( d1.GetInnerPrecisionScale( temp ) ), "5,2" );
//     ASSERT_EQ( string( d1.GetInnerDecimal( temp ) ), "1.99" );
//     ASSERT_EQ( string( d1.GetPrecisionScale( temp ) ), "5,2" );
//     ASSERT_EQ( string( d1.GetDecimal( temp ) ), "1.99" );
// }

// TEST(UT_decimal, all_4) {
//     char temp[128];
//     Decimal d1("25");
//     Decimal d2("800");
//     Decimal d3 = 1 - d1 / d2;
//     ASSERT_EQ( string( d3.GetInnerPrecisionScale( temp ) ), "7,6" );
//     ASSERT_EQ( string( d3.GetInnerDecimal( temp ) ), "0.968750" );
//     ASSERT_EQ( string( d3.GetPrecisionScale( temp ) ), "5,4" );
//     ASSERT_EQ( string( d3.GetDecimal( temp ) ), "0.9688" );
// }
