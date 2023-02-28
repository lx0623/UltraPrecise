#ifndef ARIES_SQL_PARSER_PORTAL
#define ARIES_SQL_PARSER_PORTAL

#include "string"
#include "memory"
#include "SelectStructure.h"
#include "AriesSQLStatement.h"

namespace aries {

class SQLParserPortal {
public:
    SQLParserPortal();

    // SelectStructurePointer ParseSQLFile(std::string arg_file_name);

    std::vector<AriesSQLStatementPointer> ParseSQLFile4Statements(std::string arg_file_name);

    std::vector<AriesSQLStatementPointer> ParseSQLString4Statements(const std::string& arg_sql_str);
};

}


#endif
