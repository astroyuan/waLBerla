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
//! \file DumpBlockStructureProcess.h
//! \ingroup vtk
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================

#pragma once

#include "BlockCellDataWriter.h"
#include "core/mpi/MPIManager.h"


namespace walberla {
namespace vtk {



class DumpBlockStructureProcess : public ::walberla::vtk::BlockCellDataWriter< int > {

public:

   DumpBlockStructureProcess( const std::string& id ) : ::walberla::vtk::BlockCellDataWriter<int>( id ) {}

protected:

   void configure() override {}

   int evaluate( const cell_idx_t, const cell_idx_t, const cell_idx_t, const cell_idx_t ) override { return MPIManager::instance()->rank(); }

}; // DumpBlockStructureProcess



} // namespace vtk
} // namespace walberla
