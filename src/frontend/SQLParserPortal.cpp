#include "SQLParserPortal.h"

#include <iostream>
#include <sstream>


#include "AriesSQLStatement.h"
#include "CommandStructure.h"
#include "parserv2/driver.h"

using namespace aries_parser;

namespace aries {


SQLParserPortal::SQLParserPortal() {
}

std::vector<AriesSQLStatementPointer> SQLParserPortal::ParseSQLString4Statements(const std::string& arg_sql_str) {
    Driver driver;
    driver.parse_string(arg_sql_str);
    return driver.statements;
}

std::vector<AriesSQLStatementPointer> SQLParserPortal::ParseSQLFile4Statements(std::string arg_file_name) {
    Driver driver;
    bool file_exist = driver.parse_file(arg_file_name);
    if (! file_exist){
        std::cerr << "ERROR: sql file not found:"<<arg_file_name<<endl;
        throw "sql file not found:" + arg_file_name;
    }
    std::vector<aries::AriesSQLStatementPointer> statement_array = driver.statements;
    return statement_array;
}

}
