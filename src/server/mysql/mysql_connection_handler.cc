#include <glog/logging.h>
#include <server/mysql/include/mysqld.h>
#include <server/mysql/include/set_var.h>
#include <server/mysql/include/sys_vars_shared.h>
#include <server/mysql/include/sys_vars.h>
#include <server/mysql/include/mysqld_thd_manager.h>
#include <schema/SchemaManager.h>
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/vio.h"
#include "server/mysql/include/mysql_com.h"
#include "server/mysql/include/sql_authentication.h"
#include "utils/mutex_lock.h"
#include "AriesEngine/transaction/AriesTransManager.h"
#include "AriesEngine/transaction/AriesVacuum.h"

int check_connection(THD *thd);
void init_connection_handler()
{
    pthread_mutex_init(&LOCK_thread_ids, &fast_mutexattr);
}

THD *createThd(int connFd, bool isUnix)
{
    Vio *vio_tmp = mysql_socket_vio_new(connFd, isUnix ? VIO_TYPE_SOCKET : VIO_TYPE_TCPIP, isUnix ? VIO_LOCALHOST : 0);
    if (vio_tmp == NULL)
        return NULL;

    THD *thd = new (std::nothrow) THD;
    if (thd == NULL)
    {
        vio_delete(vio_tmp);
        return NULL;
    }
    thd->get_protocol_classic()->init_net(vio_tmp);
    if (thd->store_globals())
    {
        // close_connection(thd, ER_OUT_OF_RESOURCES);
        delete thd;
        return NULL;
    }

    return thd;
}

bool thd_prepare_connection(THD *thd);
bool do_command(THD *thd);
bool connection_alive(THD *thd);
void close_connection(THD *thd, uint sql_errno,
                      bool server_shutdown, bool generate_event);
void* handle_connection(void* arg) {
    aries_engine::AriesVacuum::GetInstance().WaitVacuumDone();
    CONN_ARG* connArg = (CONN_ARG*) arg;
    bool isUnix = connArg->unix_sock;
    int connFd = connArg->client_fd;
    // int port = connArg->client_port;
    // char client_addr[INET6_ADDRSTRLEN] = {0};
    // memcpy(client_addr, connArg->client_addr, INET6_ADDRSTRLEN);
    // std::string addr;
    // if ( client_addr[0] )
    // {
    //     addr.append(client_addr).append(":").append(std::to_string( port ));
    // }

    delete connArg;

    THD* thd = createThd(connFd, isUnix);
    if (thd) {
        Global_THD_manager::get_instance()->add_thd(thd);

        bool rc = thd_prepare_connection(thd);
        string host;
        if (!rc)
        {
            string user;
            if ( thd->get_user_name() )
            {
                user.append(thd->get_user_name());
            }
            if ( isUnix )
            {
                host.assign("localhost");
            }
            else
            {
                host.append( thd->peer_host ).append(":").append(std::to_string( thd->peer_port ));
            }
            /*
            string emptyStr;
            aries::schema::SchemaManager::GetInstance()->GetSchema()->InsertProcess(
                    thd->m_connection_id,
                    user, host, thd->db(),
                    emptyStr, 0, emptyStr, emptyStr);
            */

            while (connection_alive(thd))
            {
                if (do_command(thd))
                    break;
            }
            // aries::schema::SchemaManager::GetInstance()->GetSchema()->DeleteProcess(thd->m_connection_id);
        }

        if ( thd->m_tx )
        {
            aries_engine::TxId txId = thd->m_tx->GetTxId();
            bool explicitTx = thd->m_explicitTx;
            if ( explicitTx )
                LOG( INFO ) << "end explicit transaction " <<  txId << ": abort";
            else
                LOG( INFO ) << "end implicit transaction " << txId << ": abort";
            aries_engine::AriesTransManager::GetInstance().EndTransaction( thd->m_tx, aries_engine::TransactionStatus::ABORTED );
        }
        close_connection(thd, 0, false, false);

        LOG(INFO) << "connection end: " << thd->thread_id() << ", " << host;
        thd->get_stmt_da()->reset_diagnostics_area();
        thd->release_resources();
        Global_THD_manager::get_instance()->remove_thd(thd);
        delete thd;

    }
    return nullptr;
}
