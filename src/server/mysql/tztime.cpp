/*
   Copyright (c) 2004, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   51 Franklin Street, Suite 500, Boston, MA 02110-1335 USA */

/*
   Most of the following code and structures were derived from
   public domain code from ftp://elsie.nci.nih.gov/pub
   (We will refer to this code as to elsie-code further.)
*/

#include <algorithm>
#include "./include/my_global.h"
#include <server/mysql/include/tztime.h>

/*
  Portable time_t replacement.
  Should be signed and hold seconds for 1902 -- 2038-01-19 range
  i.e at least a 32bit variable

  Using the system built in time_t is not an option as
  we rely on the above requirements in the time functions
*/
typedef int32_t my_time_t;
/*
  Instance of this class represents local time zone used on this system
  (specified by TZ environment variable or via any other system mechanism).
  It uses system functions (localtime_r, my_system_gmt_sec) for conversion
  and is always available. Because of this it is used by default - if there
  were no explicit time zone specified. On the other hand because of this
  conversion methods provided by this class is significantly slower and
  possibly less multi-threaded-friendly than corresponding Time_zone_db
  methods so the latter should be preffered there it is possible.
*/
class Time_zone_system : public Time_zone
{
public:
    Time_zone_system() {
      name = "SYSTEM";
    }                       /* Remove gcc warning */
    virtual const std::string& get_name() const {
        return name;
    }
};

/*
  Instance of this class represents UTC time zone. It uses system gmtime_r
  function for conversions and is always available. It is used only for
  my_time_t -> MYSQL_TIME conversions in various UTC_...  functions, it is not
  intended for MYSQL_TIME -> my_time_t conversions and shouldn't be exposed to user.
*/
class Time_zone_utc : public Time_zone
{
public:
    Time_zone_utc() {}                          /* Remove gcc warning */
    virtual const std::string& get_name() const {
        DBUG_ASSERT(0);
        return name;
    }
};

/*
  Instance of this class represents time zone which
  was specified as offset from UTC.
*/
class Time_zone_offset : public Time_zone
{
public:
    Time_zone_offset(long tz_offset_arg);
    virtual const std::string& get_name() const;
    /*
      This have to be public because we want to be able to access it from
      my_offset_tzs_get_key() function
    */
    long offset;
private:
    /* Extra reserve because of snprintf */
    char name_buff[7+16];
};

#define SECS_PER_MIN	60
#define MINS_PER_HOUR	60
#define HOURS_PER_DAY	24
#define DAYS_PER_WEEK	7
#define DAYS_PER_NYEAR	365
#define DAYS_PER_LYEAR	366
#define SECS_PER_HOUR	(SECS_PER_MIN * MINS_PER_HOUR)
#define SECS_PER_DAY	((long) SECS_PER_HOUR * HOURS_PER_DAY)
#define MONS_PER_YEAR	12
/*
  Initializes object representing time zone described by its offset from UTC.

  SYNOPSIS
    Time_zone_offset()
      tz_offset_arg - offset from UTC in seconds.
                      Positive for direction to east.
*/
Time_zone_offset::Time_zone_offset(long tz_offset_arg):
        offset(tz_offset_arg)
{
    uint hours= abs((int)(offset / SECS_PER_HOUR));
    uint minutes= abs((int)(offset % SECS_PER_HOUR / SECS_PER_MIN));
    size_t length= snprintf(name_buff, sizeof(name_buff), "%s%02d:%02d",
                            (offset>=0) ? "+" : "-", hours, minutes);
    name.assign(name_buff, length);
}

/*
  Get name of time zone

  SYNOPSIS
    get_name()

  RETURN VALUE
    Name of time zone as pointer to String object
*/
const std::string&
Time_zone_offset::get_name() const
{
    return name;
}


static Time_zone_utc tz_UTC;
static Time_zone_system tz_SYSTEM;
static Time_zone_offset tz_OFFSET0(0);

Time_zone *my_tz_OFFSET0= &tz_OFFSET0;
Time_zone *my_tz_UTC= &tz_UTC;
Time_zone *my_tz_SYSTEM= &tz_SYSTEM;

