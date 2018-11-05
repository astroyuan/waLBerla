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
//! \file UniformGPUScheme.h
//! \ingroup cuda
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#pragma once

#include "blockforest/StructuredBlockForest.h"
#include "core/mpi/MPIWrapper.h"
#include "core/mpi/BufferSystem.h"
#include "core/WeakPtrWrapper.h"
#include "domain_decomposition/IBlock.h"
#include "stencil/Directions.h"

#include "cuda/CudaRAII.h"
#include "cuda/communication/GeneratedGPUPackInfo.h"
#include "cuda/communication/CustomMemoryBuffer.h"

#include <chrono>
#include <thread>

namespace walberla {
namespace cuda {
namespace communication {



   template<typename Stencil>
   class UniformGPUScheme
   {
   public:
       explicit UniformGPUScheme( weak_ptr_wrapper<StructuredBlockForest> bf,
                                  const shared_ptr<cuda::EventRAII> & startWaitEvent,
                                  bool sendDirectlyFromGPU = false,
                                  const int tag = 5432 );

       void addPackInfo( const shared_ptr<GeneratedGPUPackInfo> &pi );

       void startCommunication();
       void wait();

      void operator()()         { communicate(); }
      inline void communicate() { startCommunication(); wait(); }

   private:
       void setupCommunication();

       weak_ptr_wrapper<StructuredBlockForest> blockForest_;
       shared_ptr<cuda::EventRAII> startWaitEvent_;
       uint_t forestModificationStamp_;

       bool setupBeforeNextCommunication_;
       bool communicationInProgress_;
       bool sendFromGPU_;

       using CpuBuffer_T = cuda::communication::PinnedMemoryBuffer;
       using GpuBuffer_T = cuda::communication::GPUMemoryBuffer;

       mpi::GenericBufferSystem<CpuBuffer_T, CpuBuffer_T> bufferSystemCPU_;
       mpi::GenericBufferSystem<GpuBuffer_T, GpuBuffer_T> bufferSystemGPU_;

       std::vector<shared_ptr<GeneratedGPUPackInfo> > packInfos_;
       std::map<stencil::Direction, cuda::StreamRAII> streams_;

       struct Header
       {
           BlockID blockId;
           stencil::Direction dir;
       };
       std::map<mpi::MPIRank, std::vector<Header> > headers_;
   };


} // namespace communication
} // namespace cuda
} // namespace walberla

#include "UniformGPUScheme.impl.h"