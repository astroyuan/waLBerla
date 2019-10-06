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
//! \file VelocityCorrectionNotification.h
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/mpi/notifications/NotificationType.h>
#include <mesa_pd/mpi/notifications/reset.h>

#include <core/mpi/Datatype.h>
#include <core/mpi/RecvBuffer.h>
#include <core/mpi/SendBuffer.h>

namespace walberla {
namespace mesa_pd {

/**
 * Transmits corrections of the linear and angular velocity (dv / dw) that were generated by the impulses acting on the
 * ghost particles during application of the Hard-Contact-Solvers (HCSITS) to the main particle and adds them up.
 * Use this Notification with ReduceProperty only.
 */
class VelocityCorrectionNotification
{
public:

   struct Parameters
   {
      id_t uid_;
      Vec3 dv_; /* Linear velocity correction */
      Vec3 dw_; /* Angular velocity correction */
   };

   inline explicit VelocityCorrectionNotification( const data::Particle& p ) : p_(p)  {}

   const data::Particle& p_;
};


// Reduce method for reduction (add up the velocity corrections)
void reduce(data::Particle&& p, const VelocityCorrectionNotification::Parameters& objparam)
{
   p.getDvRef() += objparam.dv_;
   p.getDwRef() += objparam.dw_;
}

template<>
void reset<VelocityCorrectionNotification>(data::Particle& p )
{
   p.setDv( Vec3(real_t(0)) );
   p.setDw( Vec3(real_t(0)) );
}

}  // namespace mesa_pd
}  // namespace walberla

//======================================================================================================================
//
//  Send/Recv Buffer Serialization Specialization
//
//======================================================================================================================

namespace walberla {
namespace mpi {

template< typename T,    // Element type of SendBuffer
          typename G>    // Growth policy of SendBuffer
mpi::GenericSendBuffer<T,G>& operator<<( mpi::GenericSendBuffer<T,G> & buf, const mesa_pd::VelocityCorrectionNotification& obj )
{
   buf.addDebugMarker( "ft" );
   buf << obj.p_.getUid();
   buf << obj.p_.getDv();
   buf << obj.p_.getDw();
   return buf;
}

template< typename T>    // Element type  of RecvBuffer
mpi::GenericRecvBuffer<T>& operator>>( mpi::GenericRecvBuffer<T> & buf, mesa_pd::VelocityCorrectionNotification::Parameters& objparam )
{
   buf.readDebugMarker( "ft" );
   buf >> objparam.uid_;
   buf >> objparam.dv_;
   buf >> objparam.dw_;
   return buf;
}

template< >
struct BufferSizeTrait< mesa_pd::VelocityCorrectionNotification > {
   static const bool constantSize = true;
   static const uint_t size = BufferSizeTrait<id_t>::size +
                              BufferSizeTrait<mesa_pd::Vec3>::size +
                              BufferSizeTrait<mesa_pd::Vec3>::size +
                              mpi::BUFFER_DEBUG_OVERHEAD;
};

} // mpi
} // walberla