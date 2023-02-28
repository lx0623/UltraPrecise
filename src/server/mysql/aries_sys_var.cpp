//
// Created by tengjp on 19-8-12.
//
#include <memory>
#include "server/mysql/include/aries_sys_var.h"
unordered_map<string, std::shared_ptr<aries_sys_var>> global_sys_var_map;

void init_sys_var_map() {
    string value;

    int intVal = 1;
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["auto_increment_increment"] = std::make_shared<aries_sys_var_int>("auto_increment_increment", value);

    global_sys_var_map["version_comment"] = std::make_shared<aries_sys_var_string>("version_comment", "Aries database distribution");
    global_sys_var_map["character_set_client"] = std::make_shared<aries_sys_var_string>("character_set_client", "utf8");
    global_sys_var_map["character_set_connection"] = std::make_shared<aries_sys_var_string>("character_set_connection", "utf8");
    global_sys_var_map["character_set_results"] = std::make_shared<aries_sys_var_string>("character_set_results", "utf8");
    global_sys_var_map["character_set_server"] = std::make_shared<aries_sys_var_string>("character_set_server", "utf8");

    global_sys_var_map["collation_server"] = std::make_shared<aries_sys_var_string>("collation_server", "latin1_swedish_ci");
    global_sys_var_map["collation_connection"] = std::make_shared<aries_sys_var_string>("collation_connection", "latin1_swedish_ci");

    global_sys_var_map["init_connect"] = std::make_shared<aries_sys_var_string>("init_connect", "");

    intVal = 288800;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["interactive_timeout"] = std::make_shared<aries_sys_var_int>("interactive_timeout", value);

    global_sys_var_map["license"] = std::make_shared<aries_sys_var_string>("license", "GPL");
    global_sys_var_map["language"] = std::make_shared<aries_sys_var_string>("language", "en");
    global_sys_var_map["lower_case_table_names"] = std::make_shared<aries_sys_var_string>("lower_case_table_names", "0");

    intVal = 4194304;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["max_allowed_packet"] = std::make_shared<aries_sys_var_int>("max_allowed_packet", value);
    intVal = 16384;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["net_buffer_length"] = std::make_shared<aries_sys_var_int>("net_buffer_length", value);
    intVal = 60;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["net_write_timeout"] = std::make_shared<aries_sys_var_int>("net_write_timeout", value);
    intVal = 1;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["performance_schema"] = std::make_shared<aries_sys_var_int>("performance_schema", value);
    intVal = 1048576;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["query_cache_size"] = std::make_shared<aries_sys_var_int>("query_cache_size", value);
    global_sys_var_map["query_cache_type"] = std::make_shared<aries_sys_var_string>("query_cache_type", "OFF");

    global_sys_var_map["sql_mode"] = std::make_shared<aries_sys_var_string>("sql_mode", "ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION");
    global_sys_var_map["system_time_zone"] = std::make_shared<aries_sys_var_string>("system_time_zone", "CST");
    global_sys_var_map["time_zone"] = std::make_shared<aries_sys_var_string>("time_zone", "SYSTEM");
    global_sys_var_map["transaction_isolation"] = std::make_shared<aries_sys_var_string>("transaction_isolation", "REPEATABLE-READ");
    global_sys_var_map["tx_isolation"] = std::make_shared<aries_sys_var_string>("tx_isolation", "REPEATABLE-READ");

    intVal = 28800;
    value.clear();
    value.assign((char*)&intVal, sizeof(int));
    global_sys_var_map["wait_timeout"] = std::make_shared<aries_sys_var_int>("wait_timeout", value);

}
