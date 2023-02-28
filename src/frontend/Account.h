#pragma once

#include <string>
#include <memory>

#include "server/mysql/include/m_string.h"
#include "AriesException.h"

extern LEX_CSTRING default_auth_plugin_name;
namespace aries {
class Account
{
public:
    Account( ) : m_currentUser( true ), m_authStrHashed( false ) {}
    Account( const std::string& user,
             const std::string& host,
             const std::string& authPlugin,
             const std::string& authStr )
    : m_user( user ),
      m_host( host ),
      m_authStr( authStr ),
      m_currentUser( false ),
      m_authStrHashed( false )
    {
      CheckAuthPlugin( authPlugin );
    }

    void SetAuthPlugin( const string& authPlugin )
    {
      CheckAuthPlugin( authPlugin );
    }
  
private:
    void CheckAuthPlugin( const std::string& authPlugin )
    {
        if ( !authPlugin.empty() && 0 != authPlugin.compare( default_auth_plugin_name.str ) ) 
        {
            string msg( "auth plugin " );
            msg.append( authPlugin );
            ThrowNotSupportedException( msg );
        }
    }

public:
    std::string m_user;
    std::string m_host;
    std::string m_authStr;
    bool m_currentUser;
    bool m_authStrHashed;

private:
    std::string m_authPlugin = default_auth_plugin_name.str;
};

using AccountSPtr = std::shared_ptr< Account >;
}