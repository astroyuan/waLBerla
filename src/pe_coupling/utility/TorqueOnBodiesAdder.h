//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can 
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of 
//  the License, or (at your option) any later version.
//  
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT 
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License 
//  for more details.
//  
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file TorqueOnBodiesAdder.h
//! \ingroup pe_coupling
//! \author Hang Yuan <johannyuan@gmail.com>
//
//======================================================================================================================

#pragma once

#include "core/math/Vector3.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include "BodySelectorFunctions.h"

namespace walberla {
namespace pe_coupling {

class TorqueOnBodiesAdder
{  
public:

   TorqueOnBodiesAdder( const shared_ptr<StructuredBlockStorage> & blockStorage, const BlockDataID & bodyStorageID,
                       const Vector3<real_t> & torque, const std::function<bool(pe::BodyID)> &bodySelectorFct = selectRegularBodies )
   : blockStorage_( blockStorage ), bodyStorageID_( bodyStorageID ), torque_( torque ), bodySelectorFct_( bodySelectorFct )
     { }

   // set a constant torque on all (only local, to avoid torque duplication) bodies
   void operator()();

   void updateTorque( const Vector3<real_t> & newTorque );

private:

   shared_ptr<StructuredBlockStorage> blockStorage_;
   const BlockDataID bodyStorageID_;
   Vector3<real_t> torque_;
   const std::function<bool(pe::BodyID)> bodySelectorFct_;
};

} // namespace pe_coupling
} // namespace walberla
