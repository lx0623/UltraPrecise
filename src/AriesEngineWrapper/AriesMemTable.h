/*
 * AriesMemTable.h
 *
 *  Created on: Mar 16, 2019
 *      Author: lichi
 */

#pragma once

#include "AbstractMemTable.h"
#include "AriesEngine/AriesDataDef.h"
#include "AriesEngine/AriesUtil.h"

using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesMemTable: public AbstractMemTable, public DisableOtherConstructors
    {
    public:
        AriesMemTable();
        ~AriesMemTable();

    public:
        void Dump();
        void SetContent( AriesTableBlockUPtr content );
        AriesTableBlockUPtr GetContent();
    private:
        AriesTableBlockUPtr m_content;
    };

    using AriesMemTableSPtr = shared_ptr< AriesMemTable >;

END_ARIES_ENGINE_NAMESPACE
