/*
 * AriesMemTable.cpp
 *
 *  Created on: Mar 16, 2019
 *      Author: lichi
 */

#include "AriesMemTable.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesMemTable::AriesMemTable()
    {
        // TODO Auto-generated constructor stub

    }

    AriesMemTable::~AriesMemTable()
    {
        // TODO Auto-generated destructor stub
    }

    void AriesMemTable::SetContent( AriesTableBlockUPtr content )
    {
        m_content = std::move( content );
    }

    AriesTableBlockUPtr AriesMemTable::GetContent()
    {
        return std::move( m_content );
    }
    END_ARIES_ENGINE_NAMESPACE
