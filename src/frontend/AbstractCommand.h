#ifndef ARIES_ABSTARCT_COMMAND
#define ARIES_ABSTARCT_COMMAND

#include <string>
#include <memory>
#include "VariousEnum.h"

namespace aries {

class AbstractCommand {
public:
    virtual ~AbstractCommand() = default;

    virtual std::string ToString() = 0;

    CommandType GetCommandType()
    {
        return command_type;
    }

    void SetCommandType(CommandType arg_value)
    {
        command_type = arg_value;
    }

    void SetCommandString(const string& s) {
        command_string = s;
    }
    string GetCommandString() {
        return command_string;
    }

public:
    bool ifNotExists = false;
    bool ifExists = false;

protected:
    CommandType command_type;

    // the original command string
    std::string command_string;

};

typedef std::shared_ptr<AbstractCommand> AbstractCommandPointer;

}//namespace

#endif

