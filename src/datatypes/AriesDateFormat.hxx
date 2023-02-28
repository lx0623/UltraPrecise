//
// Created by david.shen on 2019/12/18.
//

#ifndef DATATYPELIB_ARIESDATEFORMAT_HXX
#define DATATYPELIB_ARIESDATEFORMAT_HXX

#include "AriesDefinition.h"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"

BEGIN_ARIES_ACC_NAMESPACE
    struct ARIES_PACKED AriesFormatDatetime
    {
        int8_t sign;
        uint16_t year;
        uint8_t month;
        uint8_t day;
        uint8_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t second_part;
        ARIES_HOST_DEVICE_NO_INLINE AriesFormatDatetime(int8_t s, uint16_t y, uint8_t mon, uint8_t d, uint8_t h, uint8_t min, uint8_t sec,
                                                        uint32_t microsec) : sign( s ), year( y ), month( mon ), day( d ), hour( h ), minute( min ),
                                                                             second( sec ), second_part( microsec )
        {
        }

        ARIES_HOST_DEVICE_NO_INLINE AriesFormatDatetime(): sign(0), year(0), month(0), day(0), hour(0), minute(0), second(0), second_part(0) {
        }
    };

    enum LOCALE_LANGUAGE: int32_t {
        en_US = 0
        , zh_CN
    };


    class AriesDateFormat
    {
    public:
        ARIES_HOST_DEVICE_NO_INLINE AriesDateFormat(const LOCALE_LANGUAGE &language) {
            localeLanguage = language;
        }
        /*
         * calculate target buffer length by format string
         * */
        ARIES_HOST_DEVICE_NO_INLINE uint32_t get_format_length(const char *format);
        ARIES_HOST_DEVICE_NO_INLINE bool make_date_time(char *to, const char *format, const AriesDate &date);
        ARIES_HOST_DEVICE_NO_INLINE bool make_date_time(char *to, const char *format, const AriesDatetime &datetime);
        ARIES_HOST_DEVICE_NO_INLINE bool make_date_time(char *to, const char *format, const AriesTimestamp &timestamp);

        /*
         * only handle format such as "1001-01-01"
         * */
        ARIES_HOST_DEVICE_NO_INLINE bool str_to_date(const char *str, int len, AriesDate &date);

        /*
         * only handle format such as "1001-01-01 01:01:01"
         * */
        ARIES_HOST_DEVICE_NO_INLINE bool str_to_datetime(const char *str, int len, AriesDatetime &datetime);

        /*
         * only handle format such as "1001-01-01 01:01:01.000"
         * */
        ARIES_HOST_DEVICE_NO_INLINE bool str_to_timestamp(const char *str, int len, AriesTimestamp &timestamp);

    private:
        LOCALE_LANGUAGE localeLanguage;
        ARIES_HOST_DEVICE_NO_INLINE void prefill_strcpy(char *dest, char *input, int length, int &fullLen, char fillCh);
        ARIES_HOST_DEVICE_NO_INLINE void get_weekday_name(char *des, int weekIndex, int &len);
        ARIES_HOST_DEVICE_NO_INLINE void get_weekday_ab_name(char *des, int weekIndex, int &len);
        ARIES_HOST_DEVICE_NO_INLINE void get_month_name(char *des, int weekIndex, int &len);
        ARIES_HOST_DEVICE_NO_INLINE void get_month_ab_name(char *des, int weekIndex, int &len);
        ARIES_HOST_DEVICE_NO_INLINE bool make_date_time(char *str, const char *format, const AriesFormatDatetime &formatDatetime, bool isTimeType);
        ARIES_HOST_DEVICE_NO_INLINE bool str_to_datetime(const char *str, int len, AriesFormatDatetime &formatDatetime);
        ARIES_HOST_DEVICE_NO_INLINE int handle_digit_part(const char *start, const char *end, int &oartLen, bool &ok);
        ARIES_HOST_DEVICE_NO_INLINE bool check_datetime(AriesFormatDatetime &formatDatetime);
    };

END_ARIES_ACC_NAMESPACE

#endif //DATATYPELIB_ARIESDATEFORMAT_HXX
