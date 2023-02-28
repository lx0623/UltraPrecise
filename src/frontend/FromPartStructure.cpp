#include "FromPartStructure.h"

namespace aries {
FromPartStructure::FromPartStructure() {
}

void FromPartStructure::AddFromItem(JoinStructurePointer arg_jsp) {
    this->from_list.push_back((arg_jsp));
}

void FromPartStructure::SetFromList(std::vector<JoinStructurePointer> arg_vjsp) {
    this->from_list = (arg_vjsp);
}

std::vector<JoinStructurePointer> FromPartStructure::GetFromList() {
    return this->from_list;
}


std::string FromPartStructure::ToString() {
    std::string result = "";

    for (size_t i = 0; i != this->from_list.size(); i++) {
        result += this->from_list[i]->ToString();

        result += ",";
    }

    return result;
}

}
