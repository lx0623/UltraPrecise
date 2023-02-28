//
// Created by tengjp on 19-10-14.
//

#ifndef AIRES_SQLIDENT_H
#define AIRES_SQLIDENT_H

#include <string>
#include <memory>
#include "AriesDefinition.h"
NAMESPACE_ARIES_START

class SQLIdent {
public:
    SQLIdent(const std::string& db_arg, const std::string& table_arg, const std::string& id_arg) {
        db = db_arg;
        table = table_arg;
        id = id_arg;
    }
    ~SQLIdent();
    std::string ToString() {
        std::string ret;
        if (!db.empty()) {
            ret.append(db).append(".").append(table).append(".").append(id);
        } else if (!table.empty()) {
            ret.append(table).append(".").append(id);
        } else {
            ret = id;
        }
        return ret;
    }
    std::string db;
    std::string table;
    std::string id;
};
using SQLIdentPtr = std::shared_ptr<SQLIdent>;

NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_SQLIDENT_H
