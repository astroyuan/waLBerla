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
//! \file Ellipsoid.cpp
//! \author Tobias Preclik
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//! \brief Marshalling of objects for data transmission or storage.
//
//======================================================================================================================

#include "Ellipsoid.h"

namespace walberla {
namespace pe {
namespace communication {

//*************************************************************************************************
/*!\brief Marshalling a Ellipsoid primitive.
 *
 * \param buffer The buffer to be filled.
 * \param obj The object to be marshalled.
 * \return void
 */
void marshal( mpi::SendBuffer& buffer, const Ellipsoid& obj ) {
   marshal( buffer, static_cast<const GeomPrimitive&>( obj ) );
   buffer << obj.getSemiAxes();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unmarshalling a Ellipsoid primitive.
 *
 * \param buffer The buffer from where to read.
 * \param objparam The object to be reconstructed.
 * \param hasSuperBody False if body is not part of a union. Passed on to rigid body unmarshalling.
 * \return void
 */
void unmarshal( mpi::RecvBuffer& buffer, EllipsoidParameters& objparam ) {
   unmarshal( buffer, static_cast<GeomPrimitiveParameters&>( objparam ) );
   buffer >> objparam.semiAxes_;
}
//*************************************************************************************************

}  // namespace communication
}  // namespace pe
}  // namespace walberla
