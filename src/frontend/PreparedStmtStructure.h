//
// Created by tengjp on 19-9-5.
//

#ifndef AIRES_PREPAREDSTMTSTRUCTURE_H
#define AIRES_PREPAREDSTMTSTRUCTURE_H

#include <string>
#include <vector>
#include <memory>
#include <AriesDefinition.h>

using std::string;

NAMESPACE_ARIES_START
enum PREPARED_STMT_CMD : std::uint8_t {
    PREPARE,
    EXECUTE,
    DEALLOCATE
};
struct PrepareSrc {
    bool isVarRef = false;
    string stmtCode;
};
using PrepareSrcPtr = std::shared_ptr<PrepareSrc>;
class PreparedStmtStructure {
public:
    PREPARED_STMT_CMD stmtCmd;
    string stmtName;
    PrepareSrcPtr prepareSrcPtr;
    std::vector<std::string> executeVars; // execute stmtName (using @var1, @var2)?
};
using PreparedStmtStructurePtr = std::shared_ptr<PreparedStmtStructure>;

NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_PREPAREDSTMTSTRUCTURE_H
