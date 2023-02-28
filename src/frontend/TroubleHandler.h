#ifndef ARIES_H_TROUBLE_HANDLER
#define ARIES_H_TROUBLE_HANDLER


/*comiling options: -g -DENABLE_STACKTRACE -ldl -lbacktrace -DBOOST_STACKTRACE_USE_BACKTRACE*/

#ifdef ENABLE_STACKTRACE
#include <boost/stacktrace.hpp>
#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>

#include <signal.h>
#endif


namespace aries {

#ifdef ENABLE_STACKTRACE
typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;
#endif

class TroubleHandler {

private:
    TroubleHandler();

    TroubleHandler(const TroubleHandler &arg);

    TroubleHandler &operator=(const TroubleHandler &arg);

public:

    // static void showMessage(std::string astr);

    // static void errorMessage(int code, std::string astr);

    // static void handleStdException(const std::exception &e);

    // static void handleEverything();


    static void InitTroubleHandlerForCrash();


    /*
    template<typename E>
    static void throwWithTrace(const E &e) {
#ifdef ENABLE_STACKTRACE
        throw boost::enable_error_info(e) << traced(boost::stacktrace::stacktrace());
#else
        throw e;
#endif

    }
    */


};


}

#endif
