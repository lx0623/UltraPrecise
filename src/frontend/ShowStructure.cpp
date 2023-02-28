//
// Created by tengjp on 19-8-18.
//

#include "ShowStructure.h"
NAMESPACE_ARIES_START
string ToString(const SHOW_CMD& showCmd) {
    switch (showCmd) {
        case SHOW_CMD::SHOW_BIN_LOGS:
            return "show binary logs";
            break;
        case SHOW_CMD::SHOW_BIN_LOG_EVENTS:
            return "show binary log events";
            break;
        case SHOW_CMD::SHOW_CHAR_SET:
            return "show character set";
            break;
        case SHOW_CMD::SHOW_COLLATION:
            return "show clollation";
            break;
        case SHOW_CMD::SHOW_COLUMNS:
            return "show columns";
            break;
        case SHOW_CMD::SHOW_CREATE_DB:
            return "show create database";
            break;
        case SHOW_CMD::SHOW_CREATE_EVENT:
            return "show create event";
            break;
        case SHOW_CMD::SHOW_CREATE_FUNCTION:
            return "show create function";
            break;
        case SHOW_CMD::SHOW_CREATE_PROCEDURE:
            return "show create procedure";
            break;
        case SHOW_CMD::SHOW_CREATE_TABLE:
            return "show create table";
            break;
        case SHOW_CMD::SHOW_CREATE_TRIGGER:
            return "show create trigger";
            break;
        case SHOW_CMD::SHOW_CREATE_USER:
            return "show create user";
            break;
        case SHOW_CMD::SHOW_CREATE_VIEW:
            return "show create view";
            break;
        case SHOW_CMD::SHOW_DATABASES:
            return "show databases";
            break;
        case SHOW_CMD::SHOW_ENGINE_STATUS:
            return "show engine status";
            break;
        case SHOW_CMD::SHOW_ENGINE_LOGS:
            return "show engine logs";
            break;
        case SHOW_CMD::SHOW_ENGINE_MUTEX:
            return "show engine mutex";
            break;
        case SHOW_CMD::SHOW_ENGINES:
            return "show engines";
            break;
        case SHOW_CMD::SHOW_ERRORS:
            return "show errors";
            break;
        case SHOW_CMD::SHOW_EVENTS:
            return "show events";
            break;
        case SHOW_CMD::SHOW_FUNC_CODE:
            return "show function code";
            break;
        case SHOW_CMD::SHOW_FUNC_STATUS:
            return "show function status";
            break;
        case SHOW_CMD::SHOW_GRANTS:
            return "show grants";
            break;
        case SHOW_CMD::SHOW_INDEX:
            return "show index";
            break;
        case SHOW_CMD::SHOW_MASTER_STATUS:
            return "show master status";
            break;
        case SHOW_CMD::SHOW_OPEN_TABLES:
            return "show open tables";
            break;
        case SHOW_CMD::SHOW_PLUGINS:
            return "show plugins";
            break;
        case SHOW_CMD::SHOW_PRIVILEGES:
            return "show privileges";
            break;
        case SHOW_CMD::SHOW_PROCEDURE_CODE:
            return "show procedure code";
            break;
        case SHOW_CMD::SHOW_PROCEDURE_STATUS:
            return "show procedure status";
            break;
        case SHOW_CMD::SHOW_PROCESS_LIST:
            return "show process list";
            break;
        case SHOW_CMD::SHOW_PROFILE:
            return "show profile";
            break;
        case SHOW_CMD::SHOW_PROFILES:
            return "show profiles";
            break;
        case SHOW_CMD::SHOW_RELAYLOG_EVENTS:
            return "show relaylog events";
            break;
        case SHOW_CMD::SHOW_SLAVE_HOSTS:
            return "show slave hosts";
            break;
        case SHOW_CMD::SHOW_SLAVE_STATUS:
            return "show slave status";
            break;
        case SHOW_CMD::SHOW_STATUS:
            return "show status";
            break;
        case SHOW_CMD::SHOW_TABLE_STATUS:
            return "show table status";
            break;
        case SHOW_CMD::SHOW_TABLES:
            return "show tables";
            break;
        case SHOW_CMD::SHOW_TRIGGERS:
            return "show triggers";
            break;
        case SHOW_CMD::SHOW_VARIABLES:
            return "show variables";
            break;
        case SHOW_CMD::SHOW_WARNINGS:
            return "show warnings";
            break;

        default:
            return "unknown show command";
            break;
    }
}
NAMESPACE_ARIES_END // namespace aries

