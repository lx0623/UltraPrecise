#include "AriesAssert.h"
#include "version.h"

namespace aries
{

std::string GetVersionInfo() {
    return std::string(VERSION_INFO_STRING);
}

}