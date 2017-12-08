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
//! \file PODPhantomData.h
//! \ingroup blockforest
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#pragma once

#include "blockforest/PhantomBlock.h"
#include "core/DataTypes.h"
#include "core/mpi/RecvBuffer.h"
#include "core/mpi/SendBuffer.h"


namespace walberla {
namespace blockforest {

template <typename T>
class PODPhantomWeight
{
public:

   typedef T weight_t;

   PODPhantomWeight( const T _weight ) : weight_( _weight ) {}

   T weight() const { return weight_; }

private:
   T weight_;
};

template <typename T>
struct PODPhantomWeightPackUnpack
{
   void operator()( mpi::SendBuffer & buffer, const PhantomBlock & block )
   {
      buffer << block.getData< PODPhantomWeight<T> >().weight();
   }

   void operator()( mpi::RecvBuffer & buffer, const PhantomBlock &, boost::any & data )
   {
      typename PODPhantomWeight<T>::weight_t w;
      buffer >> w;
      data = PODPhantomWeight<T>( w );
   }
};


} // namespace blockforest
} // namespace walberla
