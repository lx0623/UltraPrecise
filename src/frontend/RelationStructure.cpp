#include "RelationStructure.h"

namespace aries {
RelationStructure::RelationStructure() {
}


/*simple clone just re-use existing columns!*/
std::shared_ptr<RelationStructure> RelationStructure::SimpleClone() {

    std::shared_ptr<RelationStructure> new_one = std::make_shared<RelationStructure>();

    for (size_t i = 0; i < this->columns.size(); i++) {
        new_one->AddColumn(this->columns[i]);
    }

    new_one->SetName(this->GetName());
    new_one->SetConstraints( constraints );

    return new_one;

}


void RelationStructure::SetName(std::string arg_name) {
    this->name = arg_name;
}

std::string RelationStructure::GetName() {
    return this->name;
}

size_t RelationStructure::GetColumnCount() {
    return this->columns.size();
}


bool RelationStructure::AddColumn(ColumnStructurePointer arg_column_p) {
    assert(arg_column_p != nullptr);

    /*check name*/
    ColumnStructurePointer existing_one = this->FindColumn(arg_column_p->GetName());

    #if 0
    if (existing_one) {
//	    std::cout << "duplicated column:" + arg_column_p->ToString();
        aries::TroubleHandler::throwWithTrace(
                std::invalid_argument(std::string("duplicated column:") + arg_column_p->ToString()));
        return false;
    }
    #endif


//     /*check primary*/
//     if (arg_column_p->GetIsPrimary() && schema::DBEntry::ROWID_COLUMN_NAME != arg_column_p->GetName()) {
//         if (this->primary_column == nullptr) {
//             this->primary_column = arg_column_p;
//         } else {
// //		std::cout << "duplicated primary column:" + arg_column_p->ToString();
//             aries::TroubleHandler::throwWithTrace(
//                     std::invalid_argument(std::string("duplicated primary column:") + arg_column_p->ToString()));
//             return false;
//         }

//     }

    /*now we can add*/

    this->name_column_map.insert(std::make_pair(arg_column_p->GetName(), arg_column_p));
    this->columns.push_back(arg_column_p);

    /*todo: what if I remove a column?*/
    this->name_location_map.insert(std::make_pair(arg_column_p->GetName(), this->column_count));

    this->column_count += 1;


    return true;
}


ColumnStructurePointer RelationStructure::GetColumn(size_t arg_index) {

    assert( arg_index < columns.size() );
    return this->columns[arg_index];

}


/*we assume this function is called after FindColumn return true!!!*/
int RelationStructure::LocateColumn(std::string arg_column_name) {

    return this->name_location_map[arg_column_name];

}

ColumnStructurePointer RelationStructure::FindColumn(std::string arg_column_name) {

    ColumnStructurePointer ret = nullptr;

    auto it = this->name_column_map.find(arg_column_name);
    if (it != this->name_column_map.end()) {
        ret = it->second;

    }

    return ret;
}

std::string RelationStructure::ToString() {

    std::string ret = "";
    for (size_t i = 0; i < this->columns.size(); i++) {
        ret += this->columns[i]->ToString();
        ret += "\n";
    }
    return ret;

}

void RelationStructure::ResetNameForTheOnlyColumn(std::string agg_column_name) {
    /*we know there is only one column there. To change its name, we remove it and then re-add it!*/

    assert(this->columns.size() == 1);

    ColumnStructurePointer csp = this->columns[0];

    csp->ResetName(agg_column_name);


    this->name_column_map.clear();
    this->name_location_map.clear();
    this->columns.clear();
    this->column_count = 0;

    this->AddColumn(this->columns[0]);
}

void RelationStructure::SetConstraints( const std::map< std::string, schema::TableConstraintSPtr >& constraints )
{
    this->constraints.clear();

    for ( const auto& it : constraints )
    {
        this->constraints[ it.first ] = it.second;
    }
}

 const std::map< std::string, schema::TableConstraintSPtr >&
 RelationStructure::GetConstraints() const
 {
     return constraints;
 }

}
