#ifndef TZTIME_INCLUDED
#define TZTIME_INCLUDED

/* Copyright (c) 2004, 2011, Oracle and/or its affiliates. All rights reserved.

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
#include <string>
#include "protocol.h"

class THD;
/**
  This class represents abstract time zone and provides
  basic interface for MYSQL_TIME <-> my_time_t conversion.
  Actual time zones which are specified by DB, or via offset
  or use system functions are its descendants.
*/
class Time_zone
{
public:
    Time_zone() {}                              /* Remove gcc warning */
    /**
      Because of constness of String returned by get_name() time zone name
      have to be already zeroended to be able to use String::ptr() instead
      of c_ptr().
    */
    virtual const std::string& get_name() const = 0;

    /**
      We need this only for surpressing warnings, objects of this type are
      allocated on MEM_ROOT and should not require destruction.
    */
    virtual ~Time_zone() {};

protected:
    std::string name;
};

extern Time_zone * my_tz_UTC;
extern Time_zone * my_tz_SYSTEM;
extern Time_zone * my_tz_OFFSET0;
extern Time_zone * my_tz_find(THD *thd, const String *name);
extern my_bool     my_tz_init(THD *org_thd, const char *default_tzname, my_bool bootstrap);
extern void        my_tz_free();

#endif /* TZTIME_INCLUDED */