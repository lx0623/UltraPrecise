#include <iostream>

#include "CommandStructure.h"
#include "utils/string_util.h"
#include "AriesException.h"

namespace aries {


CommandStructure::CommandStructure() {
    this->command_type = CommandType::NullCommand;
}

std::string CommandStructure::GetDatabaseName() {
    return this->database_name;
}

void CommandStructure::SetDatabaseName(std::string arg_value) {
    this->database_name = arg_value;
}

std::string CommandStructure::GetTableName() {
    return this->table_name;
}

void CommandStructure::SetTableName(std::string arg_value) {
    this->table_name = arg_value;
}


bool CommandStructure::GetMemMark() {
    return this->mem_mark;
}

void CommandStructure::SetMemMark(bool arg_value) {
    this->mem_mark = arg_value;
}

std::shared_ptr<std::vector<TableElementDescriptionPtr>> CommandStructure::GetColumns() {
    return columns;
}

void CommandStructure::SetColumns(const std::shared_ptr<std::vector<TableElementDescriptionPtr>>& arg_value) {
    columns = arg_value;
}

int CommandStructure::GetCopyDirection() {
    return this->direction;
}

void CommandStructure::SetCopyDirection(int arg_value) {
    this->direction = arg_value;
}

std::string CommandStructure::GetCopyFileLocation() {
    return this->file_location;
}

void CommandStructure::SetCopyFileLocation(std::string arg_value) {
    this->file_location = arg_value;
}


std::string CommandStructure::GetCopyFormatReq() {
    return this->format_req;
}

void CommandStructure::SetCopyFormatReq(std::string arg_value) {
    this->format_req = arg_value;
}


AbstractQueryPointer CommandStructure::GetQuery() {
    return this->the_query;
}

void CommandStructure::SetQuery(AbstractQueryPointer arg_value) {
    this->the_query = arg_value;
}


std::string CommandStructure::ToString() {
    std::string ret = "";

    switch (this->command_type) {
        case CommandType::CreateDatabase:
            if (1 > 0) {
                ret = "CREATE DATABASE " + this->database_name + " ;";
            }
            break;

        case CommandType::DropDatabase:
            if (1 > 0) {
                ret = "DROP DATABASE " + this->database_name + " ;";
            }
            break;

        case CommandType::CreateTable:
            if (1 > 0) {
                ret += "CREATE TABLE " + this->table_name + " (\n";

                for (size_t i = 0; i < columns->size(); i++) {
                    auto elementDesc = columns->at(i);
                    if (elementDesc->IsColumnDesc()) {
                        ret += "\t{";
                        ret += elementDesc->ToString();
                        ret += "}\n";
                    }
                }

                ret += " );";
            }
            break;

        case CommandType::DropTable:
            if (1 > 0) {
                ret = "DROP TABLE ";
                for (std::shared_ptr<BasicRel>& tableIdent : *table_list) {
                    string dbName = tableIdent->GetDb();
                    string tableName = tableIdent->GetID();
                    string fullName;
                    if (!dbName.empty()) {
                        fullName.append(dbName).append(".");
                    }
                    fullName.append(tableName);
                    ret.append(fullName).append(", ");
                }
                ret = aries_utils::rtrim(ret, ", ");
                ret.append(";");
            }
            break;

        case CommandType::CopyTable:
            if (1 > 0) {
                ret = "COPY TABLE " + this->table_name + " " + ((this->direction == -1) ? "INTO" : "FROM") + " " +
                      this->file_location + " AS " + this->format_req + ")";
            }
            break;

        case CommandType::InsertQuery:
            if (1 > 0) {
                ret = "INSERT INTO " + this->table_name + " (query);";
            }
            break;

        case CommandType::ChangeDatabase:
            ret = " USE " + database_name;
            break;

        case CommandType::CreateView:
            ret = "CREATE VIEW " + this->table_name + ";";
            break;

        case CommandType::DropView:
            ret = "Drop VIEW " + table_name + ";";
            break;

        default:
            ThrowNotSupportedException( "command type:" + std::to_string((int) this->command_type));
    }


    return ret;
}

}
