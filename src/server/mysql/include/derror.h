//
// Created by tengjp on 19-7-15.
//

#ifndef AIRES_DERROR_H
#define AIRES_DERROR_H
#include <unordered_map>
#include "my_global.h"

class THD;

/* Max length of a error message. Should be kept in sync with MYSQL_ERRMSG_SIZE. */
#define ERRMSGSIZE      (512)

extern const char* ERRMSG_SYNTAX_ERROR;

const char *get_global_errmsg(int nr);
const char* ER_THD(const THD *thd, int mysql_errno);
void my_error(int nr, myf MyFlags, ...);
void my_error2(int nr, myf MyFlags, ...);
std::string format_err_msg(const char *format, ...);
std::string format_mysql_err_msg(int nr, ...);
void my_message(uint error, const char *str, myf MyFlags);
void my_printf_error(uint error, const char *format, myf MyFlags, ...);

class MY_LOCALE_ERRMSGS {
public:
    static bool Init();
    static MY_LOCALE_ERRMSGS* getInstance() {
        assert(instance);
        return instance;
    };
    const char* lookup(int mysql_errno);
    bool read_texts();
private:
    // static std::unordered_map<int, const char*> errMsgs;
    static MY_LOCALE_ERRMSGS* instance;
    const char **errmsgs;
};

#endif //AIRES_DERROR_H

