//
// Created by tengjp on 19-8-18.
//

#ifndef AIRES_SHOWSTRUCTURE_H
#define AIRES_SHOWSTRUCTURE_H

#include <string>
#include <memory>
#include "AbstractBiaodashi.h"
#include "AriesDefinition.h"
#include "TableNameStructure.h"
#include "LimitStructure.h"

using std::string;

NAMESPACE_ARIES_START
// https://dev.mysql.com/doc/refman/5.7/en/show.html
enum class SHOW_CMD : std::uint8_t {
    SHOW_BIN_LOGS,
    SHOW_BIN_LOG_EVENTS,
    SHOW_CHAR_SET,
    SHOW_COLLATION,
    SHOW_COLUMNS,
    SHOW_CREATE_DB,
    SHOW_CREATE_EVENT,
    SHOW_CREATE_FUNCTION,
    SHOW_CREATE_PROCEDURE,
    SHOW_CREATE_TABLE,
    SHOW_CREATE_TRIGGER,
    SHOW_CREATE_USER,
    SHOW_CREATE_VIEW,
    SHOW_DATABASES,
    SHOW_ENGINE_STATUS,
    SHOW_ENGINE_MUTEX,
    SHOW_ENGINE_LOGS,
    SHOW_ENGINES,
    SHOW_ERRORS,
    SHOW_EVENTS,
    SHOW_FUNC_CODE,
    SHOW_FUNC_STATUS,
    SHOW_GRANTS,
    SHOW_INDEX,
    SHOW_MASTER_STATUS,
    SHOW_OPEN_TABLES,
    SHOW_PLUGINS,
    SHOW_PRIVILEGES,
    SHOW_PROCEDURE_CODE,
    SHOW_PROCEDURE_STATUS,
    SHOW_PROCESS_LIST,
    SHOW_PROFILE,
    SHOW_PROFILES,
    SHOW_RELAYLOG_EVENTS,
    SHOW_SLAVE_HOSTS,
    SHOW_SLAVE_STATUS,
    SHOW_STATUS,
    SHOW_TABLE_STATUS,
    SHOW_TABLES,
    SHOW_TRIGGERS,
    SHOW_VARIABLES,
    SHOW_WARNINGS,
    SHOW_MAX
};
string ToString(const SHOW_CMD& showCmd);

// https://dev.mysql.com/doc/refman/5.7/en/extended-show.html

class ShowStructure {
public:
    ShowStructure() = default;
    virtual ~ShowStructure() = default;
    SHOW_CMD showCmd;
    std::string wild;
    std::string wildOrWhereStr;
    aries::BiaodashiPointer where;
    LimitStructurePointer limitExpr;
    bool full = false;

};
using ShowStructurePtr = std::shared_ptr<ShowStructure>;

class ShowSchemaInfoStructure : public ShowStructure {
public:
    string id;
    TableNameStructureSPtr tableNameStructureSPtr = std::make_shared<TableNameStructure>();
};
using ShowSchemaInfoStructurePtr = std::shared_ptr<ShowSchemaInfoStructure>;
NAMESPACE_ARIES_END // namespace aries


#endif //AIRES_SHOWSTRUCTURE_H
