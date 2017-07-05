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
//! \file CreateWorld.cpp
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include "CreateWorld.h"

#include <pe/Types.h>

#include <core/DataTypes.h>
#include <core/logging/Logging.h>
#include <core/math/AABB.h>

namespace walberla {
namespace pe {

shared_ptr<BlockForest> createBlockForest(const math::AABB simulationDomain,
                                          Vector3<uint_t> blocks,
                                          const Vector3<bool> isPeriodic)
{
   if (isPeriodic[0] && blocks[0]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic x direction (" << blocks[0] << ")! Setting to 2..." );
      blocks[0] = 2;
   }
   if (isPeriodic[1] && blocks[1]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic y direction (" << blocks[1] << ")! Setting to 2..." );
      blocks[1] = 2;
   }
   if (isPeriodic[2] && blocks[2]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic z direction (" << blocks[2] << ")! Setting to 2..." );
      blocks[2] = 2;
   }

   return blockforest::createBlockForest( simulationDomain,
                                          blocks[0], blocks[1], blocks[2],
         blocks[0], blocks[1], blocks[2],
         isPeriodic[0],isPeriodic[1],isPeriodic[2],
         false );
}

shared_ptr<BlockForest> createBlockForest(const math::AABB simulationDomain,
                                          Vector3<uint_t> blocks,
                                          const Vector3<bool> isPeriodic,
                                          const bool setupRun,
                                          const std::string sbffile)
{
   if (isPeriodic[0] && blocks[0]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic x direction (" << blocks[0] << ")! Setting to 2..." );
      blocks[0] = 2;
   }
   if (isPeriodic[1] && blocks[1]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic y direction (" << blocks[1] << ")! Setting to 2..." );
      blocks[1] = 2;
   }
   if (isPeriodic[2] && blocks[2]<2)
   {
      WALBERLA_LOG_WARNING_ON_ROOT( "To few blocks in periodic z direction (" << blocks[2] << ")! Setting to 2..." );
      blocks[2] = 2;
   }

   WALBERLA_LOG_INFO_ON_ROOT( "Setup file specified: Using " << sbffile );

   if (setupRun)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Setup run. For production run specify 'setupRun = false'");

      if( MPIManager::instance()->numProcesses() > 1 )
         WALBERLA_LOG_WARNING_ON_ROOT( "Setup run with more than one process! Only root is doing work! I hope you know what you are doing!" );

      WALBERLA_ROOT_SECTION()
      {
         WALBERLA_LOG_INFO_ON_ROOT( "Creating the block structure ..." );

         SetupBlockForest sforest;

         sforest.addWorkloadMemorySUIDAssignmentFunction( blockforest::uniformWorkloadAndMemoryAssignment );

         sforest.init( simulationDomain, blocks[0], blocks[1], blocks[2], isPeriodic[0], isPeriodic[1], isPeriodic[2] );

         // calculate process distribution
         sforest.balanceLoad( blockforest::StaticLevelwiseCurveBalance(true), blocks[0] * blocks[1] * blocks[2] );

         sforest.saveToFile( sbffile.c_str() );

         WALBERLA_LOG_INFO_ON_ROOT( "SetupBlockForest successfully saved to file!" );
      }

      return shared_ptr<BlockForest>();
   }

   WALBERLA_LOG_INFO_ON_ROOT( "Production Run!" );
   WALBERLA_LOG_INFO_ON_ROOT( "Creating the block structure: loading from file \'" << sbffile << "\' ..." );

   MPIManager::instance()->useWorldComm();

   return shared_ptr< BlockForest >( new BlockForest( uint_c( MPIManager::instance()->rank() ), sbffile.c_str(), true, false ) );
}

shared_ptr<BlockForest> createBlockForestFromConfig(const Config::BlockHandle& mainConf)
{
   Vec3 simulationCorner         = mainConf.getParameter<Vec3>("simulationCorner", Vec3(0, 0, 0));
   Vec3 simulationSize           = mainConf.getParameter<Vec3>("simulationDomain", Vec3(10, 10, 10));
   math::AABB simulationDomain   = math::AABB( simulationCorner, simulationCorner + simulationSize );
   Vector3<uint_t> blocks        = mainConf.getParameter<Vector3<uint_t>>("blocks", Vector3<uint_t>(3, 3, 3));
   Vector3<bool> isPeriodic      = mainConf.getParameter<Vector3<bool>>("isPeriodic", Vector3<bool>(true, true, true));

   if( !mainConf.isDefined( "sbfFile" ) )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "No setup file specified: Creation without setup file!" );

      return createBlockForest( simulationDomain, blocks, isPeriodic);
   }

   // sbf file given -> try to load or save domain decomposition
   std::string sbffile = mainConf.getParameter< std::string >( "sbfFile" );
   WALBERLA_LOG_INFO_ON_ROOT( "Setup file specified: Using " << sbffile );

   bool setupRun = mainConf.getParameter< bool >( "setupRun", true );
   return createBlockForest(simulationDomain, blocks, isPeriodic, setupRun, sbffile);
}

} // namespace pe
} // namespace walberla
