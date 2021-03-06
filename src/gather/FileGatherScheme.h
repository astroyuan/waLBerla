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
//! \file FileGatherScheme.h
//! \ingroup gather
//! \author Martin Bauer <martin.bauer@fau.de>
//! \brief Gathering Data using Files
//
//======================================================================================================================

#pragma once

#include "GatherPackInfo.h"
#include "core/debug/Debug.h"

#include <fstream>
#include <vector>


namespace walberla {

namespace domain_decomposition {
class BlockStorage;
}

namespace gather
{


   /**
    * Collects / Gathers data from multiple blocks to a single block, using temporary files.
    *
    * This scheme is only suitable, if the collected data have to be available not until
    * the end of the simulation. After every timestep writeToFile() has to be called, which
    * collects the data and writes them to a file (one file per "sending" process).
    * After the simulation has finished, collectFromFiles() has to be called. This function
    * is executed on only one process, which reads all the files and unpacks the data
    * (using the pack info's). So the unpack() function's of the pack info's are only called
    * AFTER the simulation.
    *
    * If the data has to be available after each timestep MPIGatherScheme should be used.
    */
   class FileGatherScheme
   {
      public:

         //**Construction and Destruction*************************************************************
         /*!\name Construction and Destruction*/
         //@{
         FileGatherScheme( domain_decomposition::BlockStorage & blockStorage, uint_t everyNTimestep = 1 );
         ~FileGatherScheme();
         //@}
         //*************************************************************************************************************


         /**
          * Registering PackInfo's
          * The ownership of the passed pack info, is transferred to the FileCollectorScheme,
          * i.e. the pack info is deleted by the scheme
          */
         void addPackInfo( const shared_ptr<GatherPackInfo>  & pi ) { packInfos_.push_back(pi); }


         /**
          * Each process writes the data that has to be collected to a temporary file.
          * This function is similar to the "communicate()" function of other scheme's
          * and should be called after every timestep.
          * The difference is, that the unpacking occurs only once at the end of the simulation
          * when collectFromFiles() is called.
          */
         void writeToFile();

         /// Same as communicate
          void operator() () {
             static uint_t timestep = 0;
             WALBERLA_ASSERT_UNEQUAL( everyNTimestep_, 0 );
             if ( timestep % everyNTimestep_ == 0)
                writeToFile();
             timestep++;
          }

         /**
          * Reads the temporary files generated by all collecting processes,
          * and calls the unpack() methods of the PackInfo's
          * Is usually executed only once at the end of the simulation.
          */
         void collectFromFiles();

      private:

         /// Temporary files are generated by collecting processes and written
         /// in the writeToFile() member. These files are deleted by this function.
         void deleteTemporaryFiles();

         domain_decomposition::BlockStorage  & blocks_;

         using PackInfoVector = std::vector<shared_ptr<GatherPackInfo>>;
         PackInfoVector  packInfos_; ///< all registered PackInfos

         /// To generated unique filenames for every FileCollectorScheme
         /// the schemes are numbered
         size_t        myId_;   ///< number of current scheme
         static size_t nextId_; ///< number of existing FileCollectorScheme's (to generated id's)

         /// Output file stream, where collected data is written to
         std::ofstream fileStream_;

         /// if operator() is called N times, writeToFile is called N / everyNTimestep_ times
         uint_t        everyNTimestep_;
   };

} // namespace gather
} // namespace walberla

