#include "server/mysql/include/sql_auth_cache.h"
#include "server/mysql/include/sql_authentication.h"

#include "frontend/SQLExecutor.h"
#include "AriesEngineWrapper/AriesMemTable.h"

using namespace aries;
using namespace aries_engine;

std::unordered_map< string, ACL_USER_SPTR > acl_users;
std::mutex acl_users_mutex;

/**
  Convert scrambled password to binary form, according to scramble type, 
  Binary form is stored in user.salt.
  
  @param acl_user The object where to store the salt
   
  Despite the name of the function it is used when loading ACLs from disk
  to store the password hash in the ACL_USER object.
  Note that it works only for native and "old" mysql authentication built-in
  plugins.
  
  Assumption : user's authentication plugin information is available.

  @return Password hash validation
    @retval false Hash is of suitable length
    @retval true Hash is of wrong length or format
*/

bool set_user_salt( ACL_USER_SPTR& acl_user )
{
  return set_native_salt( acl_user->auth_string.data(),
                          acl_user->auth_string.length(),
                          acl_user->salt,
                          &acl_user->salt_len );
}

void LoadAclUsers()
{
    std::lock_guard< mutex > lock( acl_users_mutex );
    std::string sql = "select Host, User, authentication_string from mysql.user";
    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    if ( !sqlResult->IsSuccess() )
    {
        const auto& errMsg = sqlResult->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), errMsg.data() );
    }

    const std::vector<aries::AbstractMemTablePointer>& results = sqlResult->GetResults();
    auto amtp = results[0];
    auto table = ( ( AriesMemTable * )amtp.get() )->GetContent();
    auto tupleNum = table->GetRowCount();
    int columnCount = table->GetColumnCount();

    std::vector< AriesDataBufferSPtr > columns;
    for (int col = 1; col < columnCount + 1; col++) {
        columns.push_back( table->GetColumnBuffer( col ) );
    }

    for ( int tid = 0; tid < tupleNum; ++tid )
    {
        auto host = columns[ 0 ]->GetString( tid );
        auto user = columns[ 1 ]->GetString( tid );
        auto authStr = columns[ 2 ]->GetNullableString( tid ); // handle NULL
        auto aclUser = std::make_shared< ACL_USER >();
        aclUser->host = host;
        aclUser->user = user;
        aclUser->auth_string = authStr;
        set_user_salt( aclUser );

        acl_users[ user ] = aclUser;
    }
}

ACL_USER_SPTR FindAclUser( const string& user, const string& host )
{
    std::lock_guard< mutex > lock( acl_users_mutex );
    auto it = acl_users.find( user );
    if ( acl_users.end() != it )
        return it->second;
    return nullptr;
}

void AddAclUser( const string& user, const string& host, const string& authStr )
{
    std::lock_guard< mutex > lock( acl_users_mutex );
    auto aclUser = std::make_shared< ACL_USER >();
    aclUser->host = host;
    aclUser->user = user;
    aclUser->auth_string = authStr;
    set_user_salt( aclUser );

    acl_users[ user ] = aclUser;
}


void UpdateAclUser( const string& user, const string& host, const string& authStr )
{
    std::lock_guard< mutex > lock( acl_users_mutex );
    auto it = acl_users.find( user );
    if ( acl_users.end() != it && it->second->auth_string != authStr )
    {
        auto& aclUser = it->second;
        aclUser->host = host;
        aclUser->user = user;
        aclUser->auth_string = authStr;
        set_user_salt( aclUser );
    }
}

void RemoveAclUser( const string& user, const string& host )
{
    std::lock_guard< mutex > lock( acl_users_mutex );
    acl_users.erase( user );
}
