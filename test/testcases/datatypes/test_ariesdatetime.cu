//
// Created by david on 19-7-26.
//

#include <gtest/gtest.h>
#include "AriesDefinition.h"
#include "AriesMysqlInnerTime.h"
#include "AriesDatetimeTrans.h"
#include "AriesTimeCalc.hxx"
#include "timefunc.h"
#include "CudaAcc/AriesSqlOperator.h"
using namespace std;
using namespace aries_acc;

__global__ void TestString2DateAndDateSub()
{
    aries_acc::AriesDatetime data;
    STRING_TO_DATE( "2019-06-01", 10, data );
    AriesDatetime  Cuda_Dyn_resultValueName = DATE_SUB( data, DayInterval( 10, 1 ) );
}

TEST(UT_ariesdatetime, date) {
    string date = "2018-12-21";
    int was_cut;
    MYSQL_TIME d;
    enum_mysql_timestamp_type type = str_to_datetime(date.c_str(), date.size(), &d, TIME_FUZZY_DATE, &was_cut);
    ASSERT_EQ(type, MYSQL_TIMESTAMP_DATE);
    AriesDate r = AriesDatetimeTrans::GetInstance().ToAriesDate(d);
    ASSERT_EQ(r.getYear(), 2018);
    ASSERT_EQ(r.getMonth(), 12);
    ASSERT_EQ(r.getDay(), 21);
}

TEST(UT_ariesdatetime, year) {
    string date = "2018";
    int mode = ARIES_DATE_NOT_STRICT_MODE;
    AriesYear y = AriesDatetimeTrans::GetInstance().ToAriesYear(date, mode);
    ASSERT_EQ(y.getYear(), 2018);
}

TEST(UT_ariesdatetime, datetime) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d;
    enum_mysql_timestamp_type type = str_to_datetime(date.c_str(), date.size(), &d, TIME_DATETIME_ONLY, &was_cut);
    ASSERT_EQ(type, MYSQL_TIMESTAMP_DATETIME);
    AriesDatetime r = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d);
    ASSERT_EQ(r.getYear(), 2018);
    ASSERT_EQ(r.getMonth(), 12);
    ASSERT_EQ(r.getDay(), 21);
    ASSERT_EQ(r.getHour(), 8);
    ASSERT_EQ(r.getMinute(), 8);
    ASSERT_EQ(r.getSecond(), 8);
    ASSERT_EQ(r.getMicroSec(), 888888);
    date = "2018-12-21";
    type = str_to_datetime(date.c_str(), date.size(), &d, TIME_FUZZY_DATE, &was_cut);
    ASSERT_EQ(type, MYSQL_TIMESTAMP_DATE);
    r = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d);
    ASSERT_EQ(r.getYear(), 2018);
    ASSERT_EQ(r.getMonth(), 12);
    ASSERT_EQ(r.getDay(), 21);
    ASSERT_EQ(r.getHour(), 0);
    ASSERT_EQ(r.getMinute(), 0);
    ASSERT_EQ(r.getSecond(), 0);
    ASSERT_EQ(r.getMicroSec(), 0);
    r = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    ASSERT_EQ(r.getYear(), 2018);
    ASSERT_EQ(r.getMonth(), 12);
    ASSERT_EQ(r.getDay(), 21);
    ASSERT_EQ(r.getHour(), 0);
    ASSERT_EQ(r.getMinute(), 0);
    ASSERT_EQ(r.getSecond(), 0);
    ASSERT_EQ(r.getMicroSec(), 0);
}

TEST(UT_ariesdatetime, date_cmp1) {
    string date = "2019-7-26";
    int was_cut;
    MYSQL_TIME d1,d2;
    AriesDate r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_FUZZY_DATE, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDate(d1);
    date = "2019-7-27";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_FUZZY_DATE, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDate(d2);
    ASSERT_EQ(r1 < r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 != r2, true);
    ASSERT_EQ(r1 > r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, date_cmp2) {
    string date = "2019-7-26";
    int was_cut;
    MYSQL_TIME d1,d2;
    AriesDate r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_FUZZY_DATE, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDate(d1);
    date = "2019-8-27";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_FUZZY_DATE, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDate(d2);
    ASSERT_EQ(r1 < r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 > r2, false);
    ASSERT_EQ(r1 >= r2, false);
    ASSERT_EQ(r1 != r2, true);
}

TEST(UT_ariesdatetime, date_cmp3) {
    string date = "2019-7-26";
    int was_cut;
    MYSQL_TIME d1,d2;
    AriesDate r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_FUZZY_DATE, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDate(d1);
    date = "2019-6-27";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_FUZZY_DATE, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDate(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 == r2, false);
    ASSERT_EQ(r1 < r2, false);
}

TEST(UT_ariesdatetime, date_cmp4) {
    string date = "2019-7-26";
    int was_cut;
    MYSQL_TIME d1,d2;
    AriesDate r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_FUZZY_DATE, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDate(d1);
    date = "2019-7-26";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_FUZZY_DATE, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDate(d2);
    ASSERT_EQ(r1 > r2, false);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 == r2, true);
    ASSERT_EQ(r1 < r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp1) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2019-1-1 08:08:08.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 < r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 != r2, true);
    ASSERT_EQ(r1 > r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp2) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-29 08:08:08.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 < r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 > r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp3) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 08:08:08.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 == r2, true);
}

TEST(UT_ariesdatetime, datetime_cmp4) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 07:08:08.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp5) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 08:07:08.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp6) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 08:08:07.888888";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp7) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 08:08:08.888881";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp8) {
    string date = "2018-12-21 08:08:08.888888";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDatetime r1, r2;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_DATETIME_ONLY, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d1);
    date = "2018-12-21 08:08:08.87";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);
    ASSERT_EQ(r1 > r2, true);
    ASSERT_EQ(r1 >= r2, true);
    ASSERT_EQ(r1 < r2, false);
    ASSERT_EQ(r1 == r2, false);
}

TEST(UT_ariesdatetime, datetime_cmp9) {
    string date = "2018-12-21";
    int was_cut;
    MYSQL_TIME d1, d2;
    AriesDate r1;
    str_to_datetime(date.c_str(), date.size(), &d1, TIME_FUZZY_DATE, &was_cut);
    r1 = AriesDatetimeTrans::GetInstance().ToAriesDate(d1);
    date = "2018-12-21 1:01:01";
    str_to_datetime(date.c_str(), date.size(), &d2, TIME_DATETIME_ONLY, &was_cut);
    AriesDatetime r2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(d2);

    ASSERT_EQ(r1 < r2, true);
    ASSERT_EQ(r1 <= r2, true);
    ASSERT_EQ(r1 > r2, false);
    ASSERT_EQ(r1 == r2, false);

    ASSERT_EQ(r2 > r1, true);
    ASSERT_EQ(r2 >= r1, true);
    ASSERT_EQ(r2 < r1, false);
    ASSERT_EQ(r2 == r1, false);

}

TEST(UT_ariesdatetime, to_ariesdate) {
    string s = "2019-8-15 14:01:38";
    AriesDatetime datetime = AriesDatetimeTrans::GetInstance().ToAriesDatetime(s);
    AriesDatetime e(2019, 8, 15, 14, 1, 38, 0);
    ASSERT_EQ(datetime == e, true);
    AriesDate d = DATE(datetime);
    AriesDate e1(2019, 8, 15);
    ASSERT_EQ(d == e1, true);
}

TEST(UT_ariesdatetime, interval_year) {
    string date = "2019-7-26";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string interval = "1";
    string type = "year";
    YearInterval yi = AriesDatetimeTrans::GetInstance().ToYearInterval(interval, type);
    AriesDate r = DATE_SUB(d, yi);
    AriesDate e11(2018, 7, 26);
    ASSERT_EQ(r == e11, true);
    r = DATE_ADD(d, yi);
    AriesDate e12(2020, 7, 26);
    ASSERT_EQ(r == e12, true);
    date = "2016-2-29";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    yi = AriesDatetimeTrans::GetInstance().ToYearInterval(interval, type);
    r = DATE_SUB(d, yi);
    AriesDate e2(2015, 2, 28);
    ASSERT_EQ(r == e2, true);
}

TEST(UT_ariesdatetime, interval_month) {
    string date = "2019-1-26";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string interval = "1";
    string type = "month";
    MonthInterval mi = AriesDatetimeTrans::GetInstance().ToMonthInterval(interval, type);
    AriesDate r = DATE_SUB(d, mi);
    AriesDate e11(2018, 12, 26);
    ASSERT_EQ(r == e11, true);
    r = DATE_ADD(d, mi);
    AriesDate e12(2019, 2, 26);
    ASSERT_EQ(r == e12, true);
    date = "2016-3-31";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = DATE_SUB(d, mi);
    AriesDate e2(2016, 2, 29);
    ASSERT_EQ(r == e2, true);
    date = "2015-3-31";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = DATE_SUB(d, mi);
    AriesDate e3(2015, 2, 28);
    ASSERT_EQ(r == e3, true);
    interval = "61";
    mi = AriesDatetimeTrans::GetInstance().ToMonthInterval(interval, type);
    r = DATE_SUB(d, mi);
    AriesDate e4(2010, 2, 28);
    ASSERT_EQ(r == e4, true);
}

TEST(UT_ariesdatetime, interval_dayweek) {
    string date = "2019-1-26";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string interval = "1";
    string type = "week";
    DayInterval di = AriesDatetimeTrans::GetInstance().ToDayInterval(interval, type);
    AriesDate r = DATE_SUB(d, di);
    AriesDate e1(2019, 1, 19);
    ASSERT_EQ(r == e1, true);
}

TEST(UT_ariesdatetime, interval_second) {
    string date = "2019-1-26";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string interval = "1 1";
    string type = "DAY_HOUR";
    SecondInterval si = AriesDatetimeTrans::GetInstance().ToSecondInterval(interval, type);
    AriesDatetime r = DATE_SUB(d, si);
    AriesDatetime e1(2019, 1, 24, 23, 0, 0, 0);
    ASSERT_EQ(r == e1, true);
    interval = "100 100:59:59";
    type = "DAY_SECOND";
    si = AriesDatetimeTrans::GetInstance().ToSecondInterval(interval, type);
    r = DATE_SUB(d, si);
    AriesDatetime e2(2018, 10, 13, 19, 0, 1, 0);
    ASSERT_EQ(r == e2, true);
    interval = "-100 100:59:59";
    type = "DAY_SECOND";
    si = AriesDatetimeTrans::GetInstance().ToSecondInterval(interval, type);
    r = DATE_SUB(d, si);
    AriesDatetime e3(2019, 5, 10, 4, 59, 59, 0);
    ASSERT_EQ(r == e3, true);
}

TEST(UT_ariesdatetime, interval_secondpart) {
    string date = "2019-1-26";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string interval = "1 1000:1000:100000.385459";
    string type = "DAY_MICROSECOND";
    SecondPartInterval si = AriesDatetimeTrans::GetInstance().ToSecondPartInterval(interval, type);
    AriesDatetime r = DATE_SUB(d, si);
    AriesDatetime e1(2019, 3, 11, 12, 26, 40, 385459);
    ASSERT_EQ(r == e1, true);
    interval = "-1 1000:1000:100000.385459";
    type = "DAY_MICROSECOND";
    si = AriesDatetimeTrans::GetInstance().ToSecondPartInterval(interval, type);
    r = DATE_SUB(d, si);
    AriesDatetime e2(2018, 12, 12, 11, 33, 19, 614541);
    ASSERT_EQ(r == e2, true);
}

TEST(UT_ariesdatetime, datediff) {
    string date = "2019-1-26";
    AriesDate d1 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    date = "2019-1-25";
    AriesDate d2 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    int diff = DATEDIFF(d1, d2);
    ASSERT_EQ(diff == 1, true);
    date = "2019-1-25 10:12:25";
    d2 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    diff = DATEDIFF(d1, d2);
    ASSERT_EQ(diff == 1, true);
    date = "2019-2-25";
    d2 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    diff = DATEDIFF(d1, d2);
    ASSERT_EQ(diff == -30, true);
    date = "2019-2-25 10:12:25";
    d2 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    diff = DATEDIFF(d1, d2);
    ASSERT_EQ(diff == -30, true);
}

TEST(UT_ariesdatetime, timediff) {
    string date = "2019-1-26";
    AriesDate d1 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    AriesDate d2 = d1;
    AriesTime ariesTime = TIMEDIFF(d1, d2);
    ASSERT_EQ(ariesTime.sign == 1, true);
    ASSERT_EQ(ariesTime.hour == 0, true);
    ASSERT_EQ(ariesTime.minute == 0, true);
    ASSERT_EQ(ariesTime.second == 0, true);
    ASSERT_EQ(ariesTime.second_part == 0, true);
    date = "2019-1-25 10:12:25";
    AriesDatetime dt1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    date = "2019-1-26 10:12:25";
    AriesDatetime dt2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    ariesTime = TIMEDIFF(dt1, dt2);
    ASSERT_EQ(ariesTime.sign == -1, true);
    ASSERT_EQ(ariesTime.hour == 24, true);
    ASSERT_EQ(ariesTime.minute == 0, true);
    ASSERT_EQ(ariesTime.second == 0, true);
    ASSERT_EQ(ariesTime.second_part == 0, true);
    date = "2008-12-31 23:59:59.000001";
    dt1 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    date = "2008-12-30 01:01:01.000002";
    dt2 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    ariesTime = TIMEDIFF(dt1, dt2);
    ASSERT_EQ(ariesTime.sign == 1, true);
    ASSERT_EQ(ariesTime.hour == 46, true);
    ASSERT_EQ(ariesTime.minute == 58, true);
    ASSERT_EQ(ariesTime.second == 57, true);
    ASSERT_EQ(ariesTime.second_part == 999999, true);
    date = "1 10:12:25";
    AriesTime time1 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "10:12:25.900000";
    AriesTime time2 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ariesTime = TIMEDIFF(time1, time2);
    ASSERT_EQ(ariesTime.sign == 1, true);
    ASSERT_EQ(ariesTime.hour == 23, true);
    ASSERT_EQ(ariesTime.minute == 59, true);
    ASSERT_EQ(ariesTime.second == 59, true);
    ASSERT_EQ(ariesTime.second_part == 100000, true);
    date = "-1 10:12:25";
    AriesTime time21 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "-10:12:25.900000";
    AriesTime time22 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ariesTime = TIMEDIFF(time21, time22);
    ASSERT_EQ(ariesTime.sign == -1, true);
    ASSERT_EQ(ariesTime.hour == 23, true);
    ASSERT_EQ(ariesTime.minute == 59, true);
    ASSERT_EQ(ariesTime.second == 59, true);
    ASSERT_EQ(ariesTime.second_part == 100000, true);
}

TEST(UT_ariestimecalc, ariestime_cmp) {
    string date = "-72:00:00";
    AriesTime time11 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "-10:12:25";
    AriesTime time12 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ASSERT_TRUE(time11 < time12);
    ASSERT_TRUE(time11 <= time12);
    ASSERT_TRUE(time11 != time12);
    ASSERT_TRUE(time12 > time11);
    ASSERT_TRUE(time12 >= time11);
    date = "72:00:00";
    AriesTime time21 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "10:12:25";
    AriesTime time22 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ASSERT_TRUE(time21 > time22);
    ASSERT_TRUE(time21 >= time22);
    ASSERT_TRUE(time21 != time22);
    ASSERT_TRUE(time22 < time21);
    ASSERT_TRUE(time22 <= time21);
    date = "72:00:00";
    AriesTime time31 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "-10:12:25";
    AriesTime time32 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ASSERT_TRUE(time31 > time32);
    ASSERT_TRUE(time31 >= time32);
    ASSERT_TRUE(time31 != time32);
    ASSERT_TRUE(time32 < time31);
    ASSERT_TRUE(time32 <= time31);
    date = "-72:00:00";
    AriesTime time41 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    date = "10:12:25";
    AriesTime time42 = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    ASSERT_TRUE(time41 < time42);
    ASSERT_TRUE(time41 <= time42);
    ASSERT_TRUE(time41 != time42);
    ASSERT_TRUE(time42 > time41);
    ASSERT_TRUE(time42 >= time41);
}

TEST(UT_ariesdatetime, extract_date) {
    string date = "2019-8-20";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    int year = EXTRACT(INTERVAL_YEAR, d);
    ASSERT_EQ(year == 2019, true);
    int month = EXTRACT(INTERVAL_MONTH, d);
    ASSERT_EQ(month == 8, true);
    int week = EXTRACT(INTERVAL_WEEK, d);
    ASSERT_EQ(week == 33, true);
    int day = EXTRACT(INTERVAL_DAY, d);
    ASSERT_EQ(day == 20, true);
    int hour = EXTRACT(INTERVAL_HOUR, d);
    ASSERT_EQ(hour == 0, true);
    int ym = EXTRACT(INTERVAL_YEAR_MONTH, d);
    ASSERT_EQ(ym == 201908, true);
    int last = EXTRACT(INTERVAL_LAST, d);
    ASSERT_EQ(last == -1, true);
}

TEST(UT_ariesdatetime, extract_datetime) {
    string date = "2019-8-20 12:28:58.999999";
    AriesDatetime datetime = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    int year = EXTRACT(INTERVAL_YEAR, datetime);
    ASSERT_EQ(year == 2019, true);
    int month = EXTRACT(INTERVAL_MONTH, datetime);
    ASSERT_EQ(month == 8, true);
    int day = EXTRACT(INTERVAL_DAY, datetime);
    ASSERT_EQ(day == 20, true);
    int hour = EXTRACT(INTERVAL_HOUR, datetime);
    ASSERT_EQ(hour == 12, true);
    int minute = EXTRACT(INTERVAL_MINUTE, datetime);
    ASSERT_EQ(minute == 28, true);
    int second = EXTRACT(INTERVAL_SECOND, datetime);
    ASSERT_EQ(second == 58, true);
    int sp = EXTRACT(INTERVAL_MICROSECOND, datetime);
    ASSERT_EQ(sp == 999, true);
    int ym = EXTRACT(INTERVAL_YEAR_MONTH, datetime);
    ASSERT_EQ(ym == 201908, true);
    int dh = EXTRACT(INTERVAL_DAY_HOUR, datetime);
    ASSERT_EQ(dh == 2012, true);
    int dm = EXTRACT(INTERVAL_DAY_MINUTE, datetime);
    ASSERT_EQ(dm == 201228, true);
    int ds = EXTRACT(INTERVAL_DAY_SECOND, datetime);
    ASSERT_EQ(ds == 20122858, true);
    int hm = EXTRACT(INTERVAL_HOUR_MINUTE, datetime);
    ASSERT_EQ(hm == 1228, true);
    int hs = EXTRACT(INTERVAL_HOUR_SECOND, datetime);
    ASSERT_EQ(hs == 122858, true);
    int ms = EXTRACT(INTERVAL_MINUTE_SECOND, datetime);
    ASSERT_EQ(ms == 2858, true);
    int dms = EXTRACT(INTERVAL_DAY_MICROSECOND, datetime);
    ASSERT_EQ(dms == -1, true);
    int hms = EXTRACT(INTERVAL_HOUR_MICROSECOND, datetime);
    ASSERT_EQ(hms == 122858999, true);
    int mms = EXTRACT(INTERVAL_MINUTE_MICROSECOND, datetime);
    ASSERT_EQ(mms == 2858999, true);
    int sms = EXTRACT(INTERVAL_SECOND_MICROSECOND, datetime);
    ASSERT_EQ(sms == 58999, true);
    int last = EXTRACT(INTERVAL_LAST, datetime);
    ASSERT_EQ(last == -1, true);
}

TEST(UT_ariesdatetime, extract_time) {
    string date = "1 12:28:58.999999";
    AriesTime time = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    int year = EXTRACT(INTERVAL_YEAR, time);
    ASSERT_EQ(year == -1, true);
    int month = EXTRACT(INTERVAL_MONTH, time);
    ASSERT_EQ(month == -1, true);
    int day = EXTRACT(INTERVAL_DAY, time);
    ASSERT_EQ(day == -1, true);
    int hour = EXTRACT(INTERVAL_HOUR, time);
    ASSERT_EQ(hour == 36, true);
    int minute = EXTRACT(INTERVAL_MINUTE, time);
    ASSERT_EQ(minute == 28, true);
    int second = EXTRACT(INTERVAL_SECOND, time);
    ASSERT_EQ(second == 58, true);
    int sp = EXTRACT(INTERVAL_MICROSECOND, time);
    ASSERT_EQ(sp == 999, true);
    int ym = EXTRACT(INTERVAL_YEAR_MONTH, time);
    ASSERT_EQ(ym == -1, true);
    int dh = EXTRACT(INTERVAL_DAY_HOUR, time);
    ASSERT_EQ(dh == 36, true);
    int dm = EXTRACT(INTERVAL_DAY_MINUTE, time);
    ASSERT_EQ(dm == 3628, true);
    int ds = EXTRACT(INTERVAL_DAY_SECOND, time);
    ASSERT_EQ(ds == 362858, true);
    int hm = EXTRACT(INTERVAL_HOUR_MINUTE, time);
    ASSERT_EQ(hm == 3628, true);
    int hs = EXTRACT(INTERVAL_HOUR_SECOND, time);
    ASSERT_EQ(hs == 362858, true);
    int ms = EXTRACT(INTERVAL_MINUTE_SECOND, time);
    ASSERT_EQ(ms == 2858, true);
    int dms = EXTRACT(INTERVAL_DAY_MICROSECOND, time);
    ASSERT_EQ(dms == 362858999, true);
    int hms = EXTRACT(INTERVAL_HOUR_MICROSECOND, time);
    ASSERT_EQ(hms == 362858999, true);
    int mms = EXTRACT(INTERVAL_MINUTE_MICROSECOND, time);
    ASSERT_EQ(mms == 2858999, true);
    int sms = EXTRACT(INTERVAL_SECOND_MICROSECOND, time);
    ASSERT_EQ(sms == 58999, true);
    int last = EXTRACT(INTERVAL_LAST, time);
    ASSERT_EQ(last == -1, true);

    date = "-100 12:28:58.999999";
    time = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    hour = EXTRACT(INTERVAL_HOUR, time);
    ASSERT_EQ(hour == -838, true);
    minute = EXTRACT(INTERVAL_MINUTE, time);
    ASSERT_EQ(minute == -59, true);
    second = EXTRACT(INTERVAL_SECOND, time);
    ASSERT_EQ(second == -59, true);
    ms = EXTRACT(INTERVAL_SECOND_MICROSECOND, time);
    ASSERT_EQ(ms == -59000, true);
}

TEST(UT_ariesdatetime, subtime) {
    string date = "0001-01-01 00:00:00";
    AriesDatetime datetime = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    date = "-1 1:1:1.000002";
    AriesTime time = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    AriesDatetime r = SUBTIME(datetime, time);
    ASSERT_EQ(r.year == 1, true);
    ASSERT_EQ(r.month == 1, true);
    ASSERT_EQ(r.day == 2, true);
    ASSERT_EQ(r.hour == 1, true);
    ASSERT_EQ(r.minute == 1, true);
    ASSERT_EQ(r.second == 1, true);
    ASSERT_EQ(r.second_part == 2, true);

    datetime.year = 0;
    date = "1 1:1:1.000002";
    time = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    r = SUBTIME(datetime, time);
    ASSERT_EQ(r.isValid() == false, true);
}

TEST(UT_ariesdatetime, timestamp) {
    string date = "2019-8-21";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    date = "1 1:1:1.00001";
    AriesTime t = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    AriesDatetime r = TIMESTAMP(d);
    AriesDatetime e(2019, 8, 21, 0, 0, 0, 0);
    ASSERT_EQ(r == e, true);
    r = TIMESTAMP(d, t);
    AriesDatetime e1(2019, 8, 22, 1, 1, 1, 10);
    ASSERT_EQ(r == e1, true);
}

TEST(UT_ariesdatetime, timestampadd) {
    string date = "2019-8-21";
    AriesDate d1 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    string type = "minute";
    date = "1";
    SecondInterval interval = AriesDatetimeTrans::GetInstance().ToSecondInterval(date, type);
    AriesDatetime r = TIMESTAMPADD(d1, interval);
    AriesDatetime e(2019, 8, 21, 0, 1, 0, 0);
    ASSERT_EQ(r == e, true);
    type = "microsecond";
    date = "1";
    SecondPartInterval interval1 = AriesDatetimeTrans::GetInstance().ToSecondPartInterval(date, type);
    r = TIMESTAMPADD(d1, interval1);
    AriesDatetime e1(2019, 8, 21, 0, 0, 0, 1);
    ASSERT_EQ(r == e1, true);
    type = "day";
    date = "1";
    DayInterval interval2 = AriesDatetimeTrans::GetInstance().ToDayInterval(date, type);
    r = TIMESTAMPADD(d1, interval2);
    AriesDatetime e2(2019, 8, 22, 0, 0, 0, 0);
    ASSERT_EQ(r == e2, true);
}

TEST(UT_ariesdatetime, timestampdiff) {
    string type = "year";
    interval_type  t = get_interval_type(type);
    string date = "2019-8-21";
    AriesDate d1 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    date = "2018-10-21";
    AriesDate d2 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    int64_t r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == 0, true);
    type = "quarter";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -3, true);
    type = "month";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -10, true);
    type = "week";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -43, true);
    type = "day";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -304, true);
    type = "hour";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -7296, true);
    type = "minute";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -437760, true);
    type = "second";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -26265600, true);
    type = "microsecond";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d1, d2, t);
    ASSERT_EQ(r == -26265600000L, true);

    date = "2020-2-29 18:18:18.888888";
    AriesDatetime d3 = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    date = "2021-2-28";
    AriesDate d4 = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    type = "year";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 0, true);
    type = "quarter";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 3, true);
    type = "month";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 11, true);
    type = "week";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 52, true);
    type = "day";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 364, true);
    type = "hour";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 8741, true);
    type = "minute";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 524501, true);
    type = "second";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 31470101, true);
    type = "microsecond";
    t = get_interval_type(type);
    r = TIMESTAMPDIFF(d3, d4, t);
    ASSERT_EQ(r == 31470101111112, true);
}

TEST(UT_ariesdatetime, datetime_week) {
    string date = "2020-12-29 18:18:18.888888";
    AriesDatetime d = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    ASSERT_EQ(YEAR(d) == 2020, true);
    ASSERT_EQ(QUARTER(d) == 4, true);
    ASSERT_EQ(WEEK(d) == 52, true);
}

int64_t getTimestamp(string date) {
    struct tm timeStruct;
    memset(&timeStruct, 0x00, sizeof(tm));
    char *left=strptime(date.c_str(), "%Y-%m-%d %H:%M:%S", &timeStruct);
    char *tail;
    double um = strtod(left, &tail);
    int64_t ts = timegm(&timeStruct) * 1000000;
    if (tail != left) {
        ts += static_cast<long>(um * 1000000.0);
    }
    return ts;
}

TEST(UT_ariesdatetime, datetime_timestamp) {
    string date = "2013-10-10 23:40:00";
    AriesTimestamp d = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(date);
    ASSERT_EQ(d.getTimeStamp(), getTimestamp(date));

    date = "2013-10-10 23:40:00.123456";
    d = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(date);
    ASSERT_EQ(d.getTimeStamp(), getTimestamp(date));

    date = "1970-1-1 00:00:00";
    d = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(date);
    ASSERT_EQ(d.getTimeStamp(), getTimestamp(date));

    date = "1969-1-1 00:00:00";
    d = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(date);
    ASSERT_EQ(d.getTimeStamp(), getTimestamp(date));

    date = "2013-10-10 23:40:00.9999999";
    d = AriesDatetimeTrans::GetInstance().ToAriesTimestamp(date);
    ASSERT_EQ(d.getTimeStamp(), getTimestamp(date));
}

TEST(UT_ariesdatetime, unix_timestamp) {
    char tmp[64];
    string date = "1970-1-1";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    aries_acc::Decimal r = UNIX_TIMESTAMP(d, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "0");
    date = "1970-1-2";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = UNIX_TIMESTAMP(d, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "86400");
    date = "2038-1-19";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = UNIX_TIMESTAMP(d, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "2147472000");
    //for bad
    date = "1969-12-31";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = UNIX_TIMESTAMP(d, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "0");
    date = "2038-1-20";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    r = UNIX_TIMESTAMP(d, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "0");
    date = "1970-1-1 1:1:1.999";
    AriesDatetime dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "3661.999000");
    date = "1970-1-1 1:1:1.00099";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "3661.000990");
    date = "1970-1-1 00:00:00";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "0.000000");
    //ms
    date = "1970-1-1 1:1:1.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "3661.999000");
    date = "2038-1-19 3:14:07.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "2147483647.999000");
    date = "2038-1-19 3:14:08.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, 0);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "0");
    date = "1970-1-1 1:1:1.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, -600);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "3061.999000");
    date = "2038-1-19 3:14:07.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, -600);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "2147483047.999000");
    date = "2038-1-19 3:14:08.999";
    dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    r = UNIX_TIMESTAMP(dt, -600);
    ASSERT_EQ( string(r.GetDecimal(tmp)), "2147483048.999000");

    dt = FROM_UNIXTIME(r, -600);
    AriesDatetime e1(2038, 1, 19, 3, 14, 8, 999000);
    ASSERT_TRUE(dt == e1);
    dt = FROM_UNIXTIME(r, 0);
    AriesDatetime e2(2038, 1, 19, 3, 4, 8, 999000);
    ASSERT_TRUE(dt == e2);
    AriesTimestamp t(2147472000000000);
    dt = FROM_UNIXTIME(t, 0);
    AriesDatetime e3(2038, 1, 19, 0, 0, 0, 0);
    ASSERT_TRUE(dt == e3);
}

#define FORMAT_MAX_LEN 64

TEST(UT_ariesdatetime, format) {
    char to[FORMAT_MAX_LEN];
    string date = "2018-12-21";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    LOCALE_LANGUAGE target_l = en_US;
    string format;
    uint32_t l;
    format = "%b %d %Y %h:%i %p";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "Dec 21 2018 12:00 AM" == string(to), true);

    format = "%m-%d-%Y";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "12-21-2018" == string(to), true);

    format = "%d %b %y";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "21 Dec 18" == string(to), true);

    format = "%d %b %Y %T:%f";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "21 Dec 2018 00:00:00:000000" == string(to), true);

    date = "2019-12-16";
    d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    format = "%x %v";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "2019 51" == string(to), true);

    format = "%x %V";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "2019 50" == string(to), true);

    format = "%U";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "50" == string(to), true);

    format = "%u";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), d, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "51" == string(to), true);

    format = "%Y-%m";
    AriesTimestamp temp(1576834654494944);
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), temp, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "2019-12" == string(to), true);

    auto tmp = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), tmp, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "2019-12" == string(to), true);

    format = "%Y-%m-%d %H:%i:%s.%f";
    AriesTime t(-1, 838, 59, 59, 999999);
    AriesDatetime dt(2019, 12, 20, 0, 0, 0, 0);
    dt += t;
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), dt, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "2019-11-15 01:00:00.000001" == string(to), true);

    AriesTimestamp tm(0);
    format = "%Y-%m-%d %H:%i:%s";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), tm, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "0000-00-00 00:00:00" == string(to), true);

    format = "%Y-%m-%d %h:%i:%s";
    l = get_format_length(format.data(), target_l);
    ASSERT_EQ( l < FORMAT_MAX_LEN, true);
    DATE_FORMAT(to, format.data(), tm, target_l);
    ASSERT_EQ( l >= strlen(to), true);
    ASSERT_EQ( "0000-00-00 12:00:00" == string(to), true);
}

TEST(UT_ariesdatetime, strict) {
    int mode = ARIES_DATE_STRICT_MODE;
    string date = "2019-00-16";
    AriesDate d;
    bool exceptionHappened = false;
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    date = "abc";
    AriesYear y;
    try {
        y = AriesDatetimeTrans::GetInstance().ToAriesYear(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    date = "2019-00-16 00:00:00";
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    date = "abc";
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    AriesDatetime dt;
    date = "2019-00-16 00:00:00";
    try {
        dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    date = "abddd";
    try {
        dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);

    exceptionHappened = false;
    AriesTime time;
    date = "aaaaa";
    try {
        time = AriesDatetimeTrans::GetInstance().ToAriesTime(date, mode);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( exceptionHappened);
}

TEST(UT_ariesdatetime, not_strict) {
    string date = "2019-00-16";
    AriesDate d;
    bool exceptionHappened = false;
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);

    exceptionHappened = false;
    date = "abc";
    AriesYear y;
    try {
        y = AriesDatetimeTrans::GetInstance().ToAriesYear(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);

    exceptionHappened = false;
    date = "2019-00-16";
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);
    ASSERT_TRUE(AriesDatetimeTrans::GetInstance().ToString(d) == date);

    exceptionHappened = false;
    date = "2019-00-16 00:00:00";
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);
    ASSERT_TRUE(AriesDatetimeTrans::GetInstance().ToString(d) == "2019-00-16");

    exceptionHappened = false;
    date = "abc";
    try {
        d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);

    exceptionHappened = false;
    AriesDatetime dt;
    date = "2019-00-16 00:00:00";
    try {
        dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);
    ASSERT_TRUE(AriesDatetimeTrans::GetInstance().ToString(dt) == date);

    exceptionHappened = false;
    date = "abddd";
    try {
        dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);

    exceptionHappened = false;
    AriesTime time;
    date = "aaaaa";
    try {
        time = AriesDatetimeTrans::GetInstance().ToAriesTime(date);
    } catch (...) {
        exceptionHappened = true;
    }
    ASSERT_TRUE( !exceptionHappened);
}

TEST(UT_ariesdatetime, string_date) {
    string dateString = "2020-12-1";
    AriesDate date;
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    AriesDatetime datetime;
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    AriesTimestamp timestamp;
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString.resize(20, 0);
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-12-32";
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-2-29";
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2021-2-29";
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-2-29 00:00:00.000000";
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-2-29 00:00:00.999999";
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_TRUE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-2-29 00:00:999999";
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020/2/29 00:00:00.999999";
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));

    dateString = "2020-2-29T00:00:00.999999";
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), date));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), datetime));
    ASSERT_FALSE(STRING_TO_DATE(dateString.data(), dateString.size(), timestamp));
}

TEST(UT_ariesdatetime, other_1) {
    string date = "2020-12-1";
    AriesDate d = AriesDatetimeTrans::GetInstance().ToAriesDate(date);
    date = "2020-12-1 12:23:35";
    AriesDatetime dt = AriesDatetimeTrans::GetInstance().ToAriesDatetime(date);
    uint8_t  day = DAYOFMONTH(d);
    ASSERT_TRUE(day == 1);
    day = DAYOFMONTH(dt);
    ASSERT_TRUE(day == 1);
    uint8_t  w = DAYOFWEEK(d);
    ASSERT_TRUE(w == 3);
    w = DAYOFWEEK(dt);
    ASSERT_TRUE(w == 3);
    uint16_t daysofyear = DAYOFYEAR(d);
    ASSERT_TRUE(daysofyear == 336);
    daysofyear = DAYOFYEAR(dt);
    ASSERT_TRUE(daysofyear == 336);
    AriesDate rd = LAST_DAY(d);
    AriesDate ep(2020, 12, 31);
    ASSERT_TRUE(rd == ep);
    rd = LAST_DAY(dt);
    ASSERT_TRUE(rd == ep);
    uint8_t wd = WEEKDAY(d);
    ASSERT_TRUE(wd == 1);
    wd = WEEKDAY(dt);
    ASSERT_TRUE(wd == 1);
    uint8_t wy = WEEKOFYEAR(d);
    ASSERT_TRUE(wy == 49);
    wy = WEEKOFYEAR(dt);
    ASSERT_TRUE(wy == 49);
    uint32_t yw = YEARWEEK(d);
    ASSERT_TRUE(yw == 202048);
    yw = YEARWEEK(dt);
    ASSERT_TRUE(yw == 202048);
    AriesDate dd = FROM_DAYS(999999);
    AriesDate exd(2737, 11, 27);
    ASSERT_TRUE(dd == exd);
}

TEST(UT_ariesdatetime, cuda) {
    TestString2DateAndDateSub<<<1,1>>>();
    cudaDeviceSynchronize();
}
