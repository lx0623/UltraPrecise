//
// Created by tengjp on 19-8-14.
//

#include "include/sql_time.h"
Known_date_time_format known_date_time_formats[6]=
        {
                {"USA", "%m.%d.%Y", "%Y-%m-%d %H.%i.%s", "%h:%i:%s %p" },
                {"JIS", "%Y-%m-%d", "%Y-%m-%d %H:%i:%s", "%H:%i:%s" },
                {"ISO", "%Y-%m-%d", "%Y-%m-%d %H:%i:%s", "%H:%i:%s" },
                {"EUR", "%d.%m.%Y", "%Y-%m-%d %H.%i.%s", "%H.%i.%s" },
                {"INTERNAL", "%Y%m%d",   "%Y%m%d%H%i%s", "%H%i%s" },
                { 0, 0, 0, 0 }
        };

