//
// Created by david.shen on 2019/12/18.
//

#include <cassert>
#include "AriesDateFormat.hxx"
#include "AriesDateTimeLocales.h"
#include "AriesInnerHelper.hxx"
#include "AriesDataTypeUtil.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE void AriesDateFormat::prefill_strcpy(char *dest, char *input, int length, int &fullLen, char fillCh) {
        int targetLen = fullLen - length;
        char * t = dest;
        while (targetLen-- > 0) {
            *t++ = fillCh;
        }
        aries_strcpy(t, input);
        if (fullLen < length) {
            fullLen = length;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void AriesDateFormat::get_weekday_name(char *des, int weekIndex, int &len) {
        switch (localeLanguage) {
            case en_US:
                en_US_get_weekday_name(des, weekIndex, len);
                break;
            case zh_CN:
            default:
                assert(0);
                break;
        }
    }
    ARIES_HOST_DEVICE_NO_INLINE void AriesDateFormat::get_weekday_ab_name(char *des, int weekIndex, int &len) {
        switch (localeLanguage) {
            case en_US:
                en_US_get_weekday_ab_name(des, weekIndex, len);
                break;
            case zh_CN:
            default:
                assert(0);
                break;
        }
    }
    ARIES_HOST_DEVICE_NO_INLINE void AriesDateFormat::get_month_name(char *des, int weekIndex, int &len) {
        switch (localeLanguage) {
            case en_US:
                en_US_get_month_name(des, weekIndex, len);
                break;
            case zh_CN:
            default:
                assert(0);
                break;
        }
    }
    ARIES_HOST_DEVICE_NO_INLINE void AriesDateFormat::get_month_ab_name(char *des, int weekIndex, int &len) {
        switch (localeLanguage) {
            case en_US:
                en_US_get_month_ab_name(des, weekIndex, len);
                break;
            case zh_CN:
            default:
                assert(0);
                break;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::make_date_time(char *to, const char *format, const AriesDate &date)
    {
        AriesFormatDatetime tmp( 0, date.year, date.month, date.day, 0, 0, 0, 0 );
        return make_date_time( to, format, tmp, false );
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::make_date_time(char *to, const char *format, const AriesDatetime &datetime)
    {
        AriesFormatDatetime tmp( 0, datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute, datetime.second, datetime.second_part );
        return make_date_time( to, format, tmp, false );
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::make_date_time(char *to, const char *format, const AriesTimestamp &timestamp)
    {
        auto datetime = AriesDatetime( timestamp );
        AriesFormatDatetime tmp( 0, datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute, datetime.second, datetime.second_part );
        return make_date_time( to, format, tmp, true );
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::make_date_time(char *to, const char *format, const AriesFormatDatetime &formatDatetime, bool isTimeType)
    {
        char intBuff[16];
        uint32_t hours_i;
        uint32_t weekday;
        uint32_t length;
        const char *ptr, *end;
        int pos = 0;
        int usedLen = 0;

        if (formatDatetime.sign) {
            to[pos++] = '-';
        }

        end= (ptr= format) + aries_strlen(format);
        for (; ptr != end ; ptr++)
        {
            if (*ptr != '%' || ptr+1 == end)
            {
                to[pos++] = *ptr;
            }
            else
            {
                switch (*++ptr) {
                    case 'M':
                        if (!formatDatetime.month)
                            return false;
                        get_month_name(to + pos, formatDatetime.month, usedLen);
                        pos += usedLen;
                        break;
                    case 'b':
                        if (!formatDatetime.month)
                            return false;
                        get_month_ab_name(to + pos, formatDatetime.month, usedLen);
                        pos += usedLen;
                        break;
                    case 'W':
                        if (isTimeType || !(formatDatetime.month || formatDatetime.year))
                            return false;
                        weekday= calc_weekday(calc_daynr(formatDatetime.year,formatDatetime.month,
                                                         formatDatetime.day),0);
                        get_weekday_name(to + pos, weekday + 1, usedLen);
                        pos += usedLen;
                        break;
                    case 'a':
                        if (isTimeType || !(formatDatetime.month || formatDatetime.year))
                            return false;
                        weekday= calc_weekday(calc_daynr(formatDatetime.year,formatDatetime.month,
                                                         formatDatetime.day),0);
                        get_weekday_ab_name(to + pos, weekday + 1, usedLen);
                        pos += usedLen;
                        break;
                    case 'D':
                        if (isTimeType)
                            return false;
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.day, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        if (formatDatetime.day >= 10 &&  formatDatetime.day <= 19) {
                            aries_strcpy(to + pos, "th");
                        }
                        else {
                            switch (formatDatetime.day % 10) {
                                case 1:
                                    aries_strcpy(to + pos, "st");
                                    break;
                                case 2:
                                    aries_strcpy(to + pos, "nd");
                                    break;
                                case 3:
                                    aries_strcpy(to + pos, "rd");
                                    break;
                                default:
                                    aries_strcpy(to + pos, "th");
                                    break;
                            }
                        }
                        pos += 2;
                        break;
                    case 'Y':
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.year, intBuff, 10) - intBuff);
                        usedLen = 4;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'y':
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.year%100, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'm':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.month, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'c':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.month, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'd':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.day, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'e':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.day, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'f':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.second_part, intBuff, 10) - intBuff);
                        usedLen = 6;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'H':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.hour, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'h':
                    case 'I':
                        hours_i= (formatDatetime.hour%24 + 11)%12+1;
                        length= (uint32_t) (aries_int10_to_str(hours_i, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'i':					/* minutes */
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.minute, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'j':
                        if (isTimeType)
                            return false;
                        length= (uint32_t) (aries_int10_to_str(calc_daynr(formatDatetime.year,formatDatetime.month,
                                                                          formatDatetime.day) -
                                                     calc_daynr(formatDatetime.year,1,1) + 1, intBuff, 10) - intBuff);
                        usedLen = 3;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'k':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.hour, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'l':
                        hours_i= (formatDatetime.hour%24 + 11)%12+1;
                        length= (uint32_t) (aries_int10_to_str(hours_i, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'p':
                        hours_i= formatDatetime.hour%24;
                        aries_strcpy(to + pos, hours_i < 12 ? "AM" : "PM");
                        pos += 2;
                        break;
                    case 'r':
                        // handle format "%02d:%02d:%02d AM" or "%02d:%02d:%02d PM" for hour, minute, second
                        length = (uint32_t) (aries_int10_to_str((formatDatetime.hour+11)%12+1, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        to[pos++] = ':';
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.minute, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        to[pos++] = ':';
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.second, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        aries_strcpy(to + pos, ((formatDatetime.hour % 24) < 12) ? " AM" : " PM");
                        pos += 3;
                        break;
                    case 'S':
                    case 's':
                        length= (uint32_t) (aries_int10_to_str(formatDatetime.second, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'T':
                        // handle format "%02d:%02d:%02d" for hour, minute, second
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.hour, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        to[pos++] = ':';
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.minute, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        to[pos++] = ':';
                        length = (uint32_t) (aries_int10_to_str(formatDatetime.second, intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    case 'U':
                    case 'u':
                    {
                        if (isTimeType)
                            return false;
                        uint32_t year;
                        length= (uint32_t) (aries_int10_to_str(calc_week(formatDatetime.year, formatDatetime.month, formatDatetime.day,
                                                               (*ptr) == 'U' ?
                                                               WEEK_FIRST_WEEKDAY : WEEK_MONDAY_FIRST,
                                                               &year),
                                                     intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                    }
                        break;
                    case 'v':
                    case 'V':
                    {
                        if (isTimeType)
                            return false;
                        uint32_t year;
                        length= (uint32_t) (aries_int10_to_str(calc_week(formatDatetime.year, formatDatetime.month, formatDatetime.day,
                                                               ((*ptr) == 'V' ?
                                                                (WEEK_YEAR | WEEK_FIRST_WEEKDAY) :
                                                                (WEEK_YEAR | WEEK_MONDAY_FIRST)),
                                                               &year),
                                                     intBuff, 10) - intBuff);
                        usedLen = 2;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                    }
                        break;
                    case 'x':
                    case 'X':
                    {
                        if (isTimeType)
                            return false;
                        uint32_t year;
                        (void) calc_week(formatDatetime.year, formatDatetime.month, formatDatetime.day,
                                         ((*ptr) == 'X' ?
                                          WEEK_YEAR | WEEK_FIRST_WEEKDAY :
                                          WEEK_YEAR | WEEK_MONDAY_FIRST),
                                         &year);
                        length= (uint32_t) (aries_int10_to_str(year, intBuff, 10) - intBuff);
                        usedLen = 4;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                    }
                        break;
                    case 'w':
                        if (isTimeType || !(formatDatetime.month || formatDatetime.year))
                            return false;
                        weekday=calc_weekday(calc_daynr(formatDatetime.year,formatDatetime.month,
                                                        formatDatetime.day),1);
                        length= (uint32_t) (aries_int10_to_str(weekday, intBuff, 10) - intBuff);
                        usedLen = 1;
                        prefill_strcpy(to + pos, intBuff, length, usedLen, '0');
                        pos += usedLen;
                        break;
                    default:
                        to[pos++] = *ptr;
                        break;
                }
            }
        }
        to[pos] = 0;
        return true;
    }

    uint32_t AriesDateFormat::get_format_length(const char *format)
    {
        uint32_t size = 0;
        const char *ptr = format;
        const char *end = ptr + aries_strlen(format);

        for (; ptr != end ; ptr++)
        {
            if (*ptr != '%' || ptr == end-1)
                size++;
            else
            {
                switch(*++ptr) {
                    case 'M': /* month, textual */
                        if ( localeLanguage == en_US) {
                            size += en_US_get_month_name_max_len();
                            break;
                        }
                        size += 64; /* large for UTF8 locale data */
                        break;
                    case 'W': /* day (of the week), textual */
                        if ( localeLanguage == en_US) {
                            size += en_US_get_weekday_name_max_len();
                            break;
                        }
                        size += 64; /* large for UTF8 locale data */
                        break;
                    case 'D': /* day (of the month), numeric plus english suffix */
                    case 'Y': /* year, numeric, 4 digits */
                    case 'x': /* Year, used with 'v' */
                    case 'X': /* Year, used with 'v, where week starts with Monday' */
                        size += 4;
                        break;
                    case 'a': /* locale's abbreviated weekday name (Sun..Sat) */
                        if ( localeLanguage == en_US) {
                            size += en_US_get_weekday_ab_name_max_len();
                            break;
                        }
                        size += 32; /* large for UTF8 locale data */
                        break;
                    case 'b': /* locale's abbreviated month name (Jan.Dec) */
                        if ( localeLanguage == en_US) {
                            size += en_US_get_month_ab_name_max_len();
                            break;
                        }
                        size += 32; /* large for UTF8 locale data */
                        break;
                    case 'j': /* day of year (001..366) */
                        size += 3;
                        break;
                    case 'U': /* week (00..52) */
                    case 'u': /* week (00..52), where week starts with Monday */
                    case 'V': /* week 1..53 used with 'x' */
                    case 'v': /* week 1..53 used with 'x', where week starts with Monday */
                    case 'y': /* year, numeric, 2 digits */
                    case 'm': /* month, numeric */
                    case 'd': /* day (of the month), numeric */
                    case 'h': /* hour (01..12) */
                    case 'I': /* --||-- */
                    case 'i': /* minutes, numeric */
                    case 'l': /* hour ( 1..12) */
                    case 'S': /* second (00..61) */
                    case 's': /* seconds, numeric */
                    case 'c': /* month (0..12) */
                    case 'e': /* day (0..31) */
                        size += 2;
                        break;
                    case 'p': /* locale's AM or PM */
                        size += 2; /* TODO should handle locale*/
                        break;
                    case 'k': /* hour ( 0..23) */
                    case 'H': /* hour (00..23; value > 23 OK, padding always 2-digit) */
                        //size += 7; /* docs allow > 23, range depends on sizeof(unsigned int) */
                        //TODO: hard code 2
                        size += 2; /* docs allow > 23, range depends on sizeof(unsigned int) */
                        break;
                    case 'r': /* time, 12-hour (hh:mm:ss [AP]M) */
                        size += 11;
                        break;
                    case 'T': /* time, 24-hour (hh:mm:ss) */
                        size += 8;
                        break;
                    case 'f': /* microseconds */
                        size += 6;
                        break;
                    case 'w': /* day (of the week), numeric */
                    case '%':
                    default:
                        size++;
                        break;
                }
            }
        }
        return size + 1;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDateFormat::handle_digit_part(const char *start, const char *end, int &partLen, bool &ok) {
        const char *p = start;
        for (; p != end && aries_is_digit(*p); ++p);
        int n = 0;
        if ((ok = (partLen >= p - start))) {
            partLen = p - start;
            for(; start != p; ++start ) {
                n = 10 * n + ( *start - '0' );
            }
        }
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::check_datetime(AriesFormatDatetime &formatDatetime) {
        return formatDatetime.month <= 12 && formatDatetime.day <= get_last_day( formatDatetime.year, formatDatetime.month ) &&
               formatDatetime.hour < 24 && formatDatetime.minute < 60 && formatDatetime.second < 60;
    }
    /*
     * only handle format such as "1001-01-01 01:01:01.000"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::str_to_datetime(const char *str, int len, AriesFormatDatetime &formatDatetime) {
        const char *ptr = str;
        const char *end = ptr + len;
        //find the actual end pos
        --end;
        while (*end == 0 && end != str) --end;
        ++end;
        if (end == str) {
            return false;
        }
        /* Skip space at start */
        for (; ptr != end && aries_is_space(*ptr); ++ptr);
        if (ptr == end || !aries_is_digit(*ptr)) {
            return false;
        }

        bool ok = true;
        //handle year
        int partLen = 4;
        formatDatetime.year = handle_digit_part(ptr, end, partLen, ok);
        if (!ok || partLen != 4) {
            return false;
        }
        ptr += partLen;
        if (ptr == end) {
            formatDatetime.month = 1;
            formatDatetime.day = 1;
            return true;
        }
        if (*ptr != '-') {
            return false;
        }

        //handle month
        partLen = 2;
        ++ptr;
        formatDatetime.month = handle_digit_part(ptr, end, partLen, ok);
        ptr += partLen;
        if (!ok || ptr == end || *ptr != '-') {
            return false;
        }

        //handle day
        partLen = 2;
        ++ptr;
        formatDatetime.day = handle_digit_part(ptr, end, partLen, ok);
        if (!ok || !check_datetime(formatDatetime)) {
            return false;
        }
        ptr += partLen;
        if (ptr == end) {
            return true;
        }
        if (*ptr != ' ') {
            return false;
        }

        //handle hour
        partLen = 2;
        ++ptr;
        formatDatetime.hour = handle_digit_part(ptr, end, partLen, ok);
        ptr += partLen;
        if (!ok || ptr == end || *ptr != ':') {
            return false;
        }

        //handle minute
        partLen = 2;
        ++ptr;
        formatDatetime.minute = handle_digit_part(ptr, end, partLen, ok);
        ptr += partLen;
        if (!ok || ptr == end || *ptr != ':') {
            return false;
        }

        //handle second
        partLen = 2;
        ++ptr;
        formatDatetime.second = handle_digit_part(ptr, end, partLen, ok);
        if (!ok) {
            return false;
        }
        ptr += partLen;
        if (ptr == end) {
            return true;
        }
        if (*ptr != '.') {
            return false;
        }

        //handle microseconds
        partLen = 6;
        ++ptr;
        formatDatetime.second_part = handle_digit_part(ptr, end, partLen, ok);
        if (!ok) {
            return false;
        }
        ptr += partLen;
        /* Skip space at end */
        for (; ptr != end && aries_is_space(*ptr); ++ptr);
        return ptr == end;
    }

    /*
     * only handle format such as "1001-01-01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::str_to_date(const char *str, int len, AriesDate &date) {
        AriesFormatDatetime formatDatetime;
        bool success = str_to_datetime(str, len, formatDatetime);
        if (success) {
            date.year = formatDatetime.year;
            date.month = formatDatetime.month;
            date.day = formatDatetime.day;
        }
        return success;
    }

    /*
     * only handle format such as "1001-01-01 01:01:01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::str_to_datetime(const char *str, int len, AriesDatetime &datetime) {
        AriesFormatDatetime formatDatetime;
        bool success = str_to_datetime(str, len, formatDatetime);
        if (success) {
            datetime.year = formatDatetime.year;
            datetime.month = formatDatetime.month;
            datetime.day = formatDatetime.day;
            datetime.hour = formatDatetime.hour;
            datetime.minute = formatDatetime.minute;
            datetime.second = formatDatetime.second;
        }
        return success;
    }

    /*
     * only handle format such as "1001-01-01 01:01:01.000"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool AriesDateFormat::str_to_timestamp(const char *str, int len, AriesTimestamp &timestamp) {
        AriesFormatDatetime formatDatetime;
        bool success = str_to_datetime(str, len, formatDatetime);
        if (success) {
            AriesDatetime datetime{ formatDatetime.year, formatDatetime.month, formatDatetime.day, formatDatetime.hour, formatDatetime.minute,
                                    formatDatetime.second, formatDatetime.second_part };
            timestamp.timestamp = datetime.toTimestamp();
        }
        return success;
    }


END_ARIES_ACC_NAMESPACE
