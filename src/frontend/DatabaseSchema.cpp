#include "DatabaseSchema.h"
#include <glog/logging.h>

namespace aries {
DatabaseSchema::DatabaseSchema(std::string schema_name) {
    assert(schema_name.empty() == false);
    this->schema_name = schema_name;
}

std::string DatabaseSchema::GetName() {
    return this->schema_name;
}

bool DatabaseSchema::AddPhysicalTable(PhysicalTablePointer arg_table) {

    assert(arg_table != nullptr);

    /*Fuck C++!!!*/
//	std::pair<std::map<std::string, PhysicalTablePointer>::iterator, bool> insert_ret;
//	std::cout << arg_table->ToString() << "--\n";
//	this->name_table_map.insert(std::pair<std::string, PhysicalTablePointer>(arg_table->GetName(), arg_table));
//	if(insert_ret.second == false)
//	{
//	    aries::TroubleHandler::throwWithTrace(std::invalid_argument(std::string("duplicated physical table name:") + arg_table->GetName()));
//    
//	    return false;
//	}


    if (this->FindPhysicalTable(arg_table->GetName()) != nullptr) {
        LOG(ERROR) << "duplicated physical table name:" + arg_table->GetName();

        return false;

    }

    /*ok*/

    this->name_table_map[arg_table->GetName()] = arg_table;

    this->tables.push_back(arg_table);

    return true;

}

// PhysicalTablePointer DatabaseSchema::GetPhysicalTable(int arg_index) {
// 
//     if (arg_index < 0 || arg_index >= this->tables.size()) {
//         aries::TroubleHandler::throwWithTrace(std::out_of_range("GetPhysicalTable:" + std::to_string(arg_index)));
//         return nullptr;
//     }
// 
//     return this->tables[arg_index];
// 
// }

PhysicalTablePointer DatabaseSchema::FindPhysicalTable(std::string arg_table_name) {

#ifndef NDEBUG
    LOG(INFO) << "FindPhyscalTable " << arg_table_name << "\n";
#endif

    PhysicalTablePointer ret = nullptr;

    auto it = this->name_table_map.find(arg_table_name);
    if (it != this->name_table_map.end()) {
        ret = it->second;
    }

    if (ret == nullptr) {
        //std::cout << "nothing\n";
    }

    return ret;
}

}
