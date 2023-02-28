#ifndef ARIES_COMMAND_EXECUTOR
#define ARIES_COMMAND_EXECUTOR

#include "AriesServerSession.h"

namespace aries {

class AbstractCommand;
class CommandStructure;
class AccountMgmtStructure;
class CreateTableOptions;

schema::TableEntrySPtr CommandToTableEntry( CommandStructure *arg_command_p, const schema::DatabaseEntrySPtr& database );

class CommandExecutor {
public:
    CommandExecutor();

    std::string ExecuteCommand(AbstractCommand* arg_command_p, AriesServerSession *arg_session_p, const string& dbName);

    std::string ExecuteCopyTable(CommandStructure *arg_command_p, AriesServerSession *arg_session_p, const string& dbName);

    std::string ExecuteInsertQuery(CommandStructure *arg_command_p);

    std::string ExecuteCreateTable(CommandStructure *arg_command_p, const string& dbName);

    std::string ExecuteCreateDatabase(CommandStructure *arg_command_p);

    std::string ExecuteDropTable(CommandStructure *arg_command_p, const string& dbName);

    std::string ExecuteDropDatabase(CommandStructure *arg_command_p, const string& currentDbName);

    std::string ExecuteCreateView(CommandStructure *arg_command_p, const string& currentDbName);

    std::string ExecuteDropView(CommandStructure *arg_command_p, const string& currentDbName);

    void ExecuteCreateUser(AccountMgmtStructure *arg_command_p);

    void ExecuteDropUser(AccountMgmtStructure *arg_command_p);

    void ExecuteSetPassword( std::string& user, std::string& host, std::string password );

private:
    char GetSepCh(const std::string &param);

    void CheckTablePartitionOptions(
        schema::TableEntrySPtr &tableEntry,
        const CreateTableOptions &options,
        const schema::DatabaseEntrySPtr& database );
};

}//ns

#endif
