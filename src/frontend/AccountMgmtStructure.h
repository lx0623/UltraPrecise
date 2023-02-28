#pragma once

#include <vector>

#include "Account.h"
#include "AbstractCommand.h"

namespace aries {
class AccountMgmtStructure : public AbstractCommand
{
public:
    AccountMgmtStructure ( CommandType cmdType )
    {
        command_type = cmdType;
    }

    void SetAccounts( const vector< AccountSPtr >& userList ) { m_accounts = userList; }
    const std::vector< AccountSPtr >& GetAccounts() const { return m_accounts; }

    std::string ToString()
    {
        std::string ret( "create user " );
        for ( auto& a : m_accounts )
        {
            ret.append( a->m_user ).append( "@" ).append( a->m_host ).append( ", " );
        }
        return ret;
    }
private:
    std::vector< AccountSPtr > m_accounts;

};

using AccountMgmtStructureSPtr = std::shared_ptr< AccountMgmtStructure >;

}