#pragma once

#include <string>
#include "AriesEngineWrapper/AriesMemTable.h"
#include "AriesEngineWrapper/AbstractMemTable.h"

namespace aries
{

class JsonExecutor
{
public:
    bool Load( const std::string& path);
    AbstractMemTablePointer Run();
};

}
