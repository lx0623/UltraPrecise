#include <glog/logging.h>
#include "TroubleHandler.h"

#include "iostream"

#ifdef ENABLE_STACKTRACE

void my_signal_handler(int signum)
{
    ::signal(signum, SIG_DFL);
    boost::stacktrace::safe_dump_to("./backtrace.dump");
    ::raise(SIGABRT);
}

#endif


namespace aries {
// void TroubleHandler::showMessage(std::string astr) {
//     LOG(INFO) << astr << std::endl;
// }


// void TroubleHandler::errorMessage(int code, std::string astr) {
//     LOG(ERROR) << "ERROR: --> (" << astr << ")";
// 
// #ifdef ENABLE_STACKTRACE
//     std::cerr << "------------STACK--------------\n";
//     std::cerr << boost::stacktrace::stacktrace() << std::endl;
// #endif
// 
// }

// void TroubleHandler::handleStdException(const std::exception &e) {
//     LOG(ERROR) << "Standard Exception: " << e.what();
// 
// #ifdef ENABLE_STACKTRACE
//     const boost::stacktrace::stacktrace* st = boost::get_error_info<traced>(e);
//     if (st)
//     {
//         std::cerr << "------------STACK--------------\n";
// 
//         std::cerr << *st << std::endl;
//     }
// #endif
// 
// }

// void TroubleHandler::handleEverything() {
//     TroubleHandler::errorMessage(1, "We have a non-std exception");
// }


void TroubleHandler::InitTroubleHandlerForCrash() {

#ifdef ENABLE_STACKTRACE


    ::signal(SIGSEGV, &my_signal_handler);
    ::signal(SIGABRT, &my_signal_handler);


    if(boost::filesystem::exists("./backtrace.dump"))
    {
        std::ifstream ifs("./backtrace.dump");

        boost::stacktrace::stacktrace st = boost::stacktrace::stacktrace::from_dump(ifs);

        LOG(INFO) << "Previously run crashed:\n" << st << std::endl;

        ifs.close();
        boost::filesystem::remove("./backtrace.dump");
    }

#endif

}

}
