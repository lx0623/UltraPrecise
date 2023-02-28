//
// Created by tengjp on 19-10-24.
//

#ifndef _my_time_h_
#define _my_time_h_

#include "datatypes/AriesTime.hxx"
#include "datatypes/AriesDatetime.hxx"
int mysql_datetime_to_str(const aries_acc::AriesDatetime *l_time, char *to, uint dec);
int mysql_time_to_str(const aries_acc::AriesTime *l_time, char *to, uint dec);
#endif //_my_time_h_
