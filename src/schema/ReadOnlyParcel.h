#pragma once

#include <string>
#include <stdexcept>

#include "Parcel.h"

using namespace std;

namespace aries {

class ReadOnlyParcel: public Parcel {

public:
    ReadOnlyParcel(uint8_t* buffer, size_t size) : Parcel(false){
        Parcel::readOffset = 0;
        Parcel::needToFree = false;

        data = buffer;
        capacity = size;
        offset = size;
    }

    template <typename T>
    void Write(T& value) {
        throw new std::runtime_error("read only parcel");
    }
};
}