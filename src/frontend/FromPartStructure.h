#ifndef ARIES_FROM_PART_STRUCTURE
#define ARIES_FROM_PART_STRUCTURE

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "JoinStructure.h"

namespace aries {

class FromPartStructure {
private:

    std::vector<JoinStructurePointer> from_list;

    FromPartStructure(const FromPartStructure &arg);

    FromPartStructure &operator=(const FromPartStructure &arg);

public:


    FromPartStructure();

    void AddFromItem(JoinStructurePointer arg_jsp);

    void SetFromList(std::vector<JoinStructurePointer> arg_vjsp);

    std::vector<JoinStructurePointer> GetFromList();

    std::string ToString();


};

typedef std::shared_ptr<FromPartStructure> FromPartStructurePointer;


}//namespace
#endif
