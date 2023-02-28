#ifndef ARIES_COMMAND_STRUCTURE
#define ARIES_COMMAND_STRUCTURE

#include <vector>
#include <memory>

#include "AbstractCommand.h"
#include "AbstractQuery.h"
#include "PartitionStructure.h"
#include "ColumnDescription.h"
#include "BasicRel.h"

namespace aries {

using TABLE_LIST = std::shared_ptr<vector<std::shared_ptr<BasicRel>>>;

struct CreateTableOptions
{
    PartitionStructureSPtr m_partitionStructure;
};

class CommandStructure : public AbstractCommand {
private:
    CommandStructure(const CommandStructure &arg);

    CommandStructure &operator=(const CommandStructure &arg);

    //create or drop db
    std::string database_name;

    //create or drop table
    std::string table_name;
    TABLE_LIST table_list;
    bool mem_mark = false;
    std::shared_ptr<std::vector<TableElementDescriptionPtr>> columns = nullptr;
    CreateTableOptions create_table_options;

    //copy
    int direction = 0; //-1 means table-to-file, 1 means file-to-table
    std::string file_location;
    std::string format_req;

    //insert_into_query
    AbstractQueryPointer the_query = nullptr;

public:

    CommandStructure();

    std::string ToString();

    std::string GetDatabaseName();

    void SetDatabaseName(std::string arg_value);

    std::string GetTableName();

    void SetTableName(std::string arg_value);

    void SetTableList(const TABLE_LIST& arg_value) {
        table_list = arg_value;
    }
    const TABLE_LIST GetTableList() const {
        return table_list;
    }


    bool GetMemMark();

    void SetMemMark(bool arg_value);

    std::shared_ptr<std::vector<TableElementDescriptionPtr>> GetColumns();

    void SetColumns(const std::shared_ptr<std::vector<TableElementDescriptionPtr>>& arg_value);

    void SetCreateTableOptions( const CreateTableOptions &arg_options )
    {
        create_table_options = arg_options;
    }
    const CreateTableOptions &GetCreateTableOptions() const
    {
        return create_table_options;
    }

    int GetCopyDirection();

    void SetCopyDirection(int arg_value);

    std::string GetCopyFileLocation();

    void SetCopyFileLocation(std::string arg_value);


    std::string GetCopyFormatReq();

    void SetCopyFormatReq(std::string arg_value);

    AbstractQueryPointer GetQuery();

    void SetQuery(AbstractQueryPointer arg_value);


};


typedef std::shared_ptr<CommandStructure> CommandStructurePointer;


}//namespace


#endif
