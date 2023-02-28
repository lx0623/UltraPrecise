#ifndef SQL_USER_CACHE_INCLUDED
#define SQL_USER_CACHE_INCLUDED

#include <string.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

#include "mysql_com.h"                  // SCRAMBLE_LENGTH

extern char wild_many;
extern char wild_one;

void LoadAclUsers();
void AddAclUser( const std::string& host, const std::string& user, const std::string& authStr );
void UpdateAclUser( const std::string& user, const std::string& host, const std::string& authStr );
void RemoveAclUser( const std::string& host, const std::string& user );

/*
class ACL_HOST_AND_IP
{
  char *hostname;
  size_t hostname_length;
  long ip, ip_mask; // Used with masked ip:s

  const char *calc_ip(const char *ip_arg, long *val, char end);

public:
  const char *get_host() const { return hostname; }
  size_t get_host_len() { return hostname_length; }

  bool has_wildcard()
  {
    return (strchr(hostname,wild_many) ||
            strchr(hostname,wild_one)  || ip_mask );
  }

  bool check_allow_all_hosts()
  {
    return (!hostname ||
            (hostname[0] == wild_many && !hostname[1]));
  }

  void update_hostname(const char *host_arg);

  bool compare_hostname(const char *host_arg, const char *ip_arg);

};
*/

class ACL_ACCESS {
public:
  // ACL_HOST_AND_IP host;
  std::string host;
  ulong sort;
  ulong access;
};

/* ACL_HOST is used if no host is specified */

class ACL_HOST :public ACL_ACCESS
{
public:
  char *db;
};

class ACL_USER :public ACL_ACCESS
{
public:
  // USER_RESOURCES user_resource;
  std::string user;
  /**
    The salt variable is used as the password hash for
    native_password_authetication.
  */
  uint8 salt[SCRAMBLE_LENGTH + 1];       // scrambled password in binary form
  /**
    In the old protocol the salt_len indicated what type of autnetication
    protocol was used: 0 - no password, 4 - 3.20, 8 - 4.0,  20 - 4.1.1
  */
  uint8 salt_len;
  // enum SSL_type ssl_type;
  // const char *ssl_cipher, *x509_issuer, *x509_subject;
  std::string plugin;
  std::string auth_string;
  bool password_expired;
  bool can_authenticate = true;
  // MYSQL_TIME password_last_changed;
  uint password_lifetime;
  bool use_default_password_lifetime;
  /**
    Specifies whether the user account is locked or unlocked.
  */
  bool account_locked;

  // ACL_USER *copy(MEM_ROOT *root);
};

using ACL_USER_SPTR = std::shared_ptr< ACL_USER >;

ACL_USER_SPTR FindAclUser( const std::string& user, const std::string& host );

extern std::unordered_map< std::string, ACL_USER_SPTR > acl_users;
extern std::mutex acl_users_mutex;
#endif