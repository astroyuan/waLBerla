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
//! \file DragForceSphere.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/Debug.h"
#include "core/debug/TestSubsystem.h"
#include "core/logging/Logging.h"
#include "core/math/all.h"
#include "core/SharedFunctor.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/mpi/MPIManager.h"
#include "core/mpi/Reduce.h"

#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/communication/PackInfo.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/MacroscopicValueCalculation.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/lattice_model/ForceModel.h"
#include "lbm/sweeps/CellwiseSweep.h"
#include "lbm/sweeps/SweepWrappers.h"

#include "lbm_mesapd_coupling/momentum_exchange_method/MovingParticleMapping.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/boundary/SimpleBB.h"
#include "lbm_mesapd_coupling/momentum_exchange_method/boundary/CurvedLinear.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/utility/OmegaBulkAdaption.h"

#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "stencil/D3Q27.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/all.h"
#include "field/vtk/all.h"
#include "lbm/vtk/all.h"

#include <iomanip>
#include <iostream>

#ifdef WALBERLA_BUILD_WITH_CODEGEN
#include "GeneratedLBMWithForce.h"

#define USE_TRTLIKE
//#define USE_D3Q27TRTLIKE
//#define USE_CUMULANTTRT
//#define USE_CUMULANT
#endif

namespace drag_force_sphere_mem
{

///////////
// USING //
///////////

using namespace walberla;
using walberla::uint_t;


#ifdef WALBERLA_BUILD_WITH_CODEGEN
using LatticeModel_T = lbm::GeneratedLBMWithForce;
#else
using ForceModel_T = lbm::force_model::LuoConstant;
using LatticeModel_T = lbm::D3Q19< lbm::collision_model::D3Q19MRT, false, ForceModel_T>;
#endif

using Stencil_T = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField<LatticeModel_T>;

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;

using ScalarField_T = GhostLayerField< real_t, 1>;

const uint_t FieldGhostLayers = 1;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag   ( "fluid" );
const FlagUID MO_BB_Flag   ( "moving obstacle BB" );
const FlagUID MO_CLI_Flag  ( "moving obstacle CLI" );

////////////////
// PARAMETERS //
////////////////

struct Setup{
   uint_t checkFrequency;
   real_t visc;
   real_t tau;
   real_t radius;
   uint_t length;
   real_t chi;
   real_t extForce;
   real_t analyticalDrag;
};

enum MEMVariant { BB, CLI };

MEMVariant to_MEMVariant( const std::string& s )
{
   if( s == "BB"  ) return MEMVariant::BB;
   if( s == "CLI" ) return MEMVariant::CLI;
   throw std::runtime_error("invalid conversion from text to MEMVariant");
}

std::string MEMVariant_to_string ( const MEMVariant& m )
{
   if( m == MEMVariant::BB  ) return "BB";
   if( m == MEMVariant::CLI ) return "CLI";
   throw std::runtime_error("invalid conversion from MEMVariant to string");
}

/////////////////////////////////////
// BOUNDARY HANDLING CUSTOMIZATION //
/////////////////////////////////////

template <typename ParticleAccessor_T>
class MyBoundaryHandling
{
public:

   using SBB_T = lbm_mesapd_coupling::SimpleBB< LatticeModel_T, FlagField_T, ParticleAccessor_T >;
   using CLI_T = lbm_mesapd_coupling::CurvedLinear< LatticeModel_T, FlagField_T, ParticleAccessor_T >;
   using Type = BoundaryHandling< FlagField_T, Stencil_T, SBB_T, CLI_T >;

   MyBoundaryHandling( const BlockDataID & flagFieldID, const BlockDataID & pdfFieldID,
                       const BlockDataID & particleFieldID, const shared_ptr<ParticleAccessor_T>& ac) :
         flagFieldID_( flagFieldID ), pdfFieldID_( pdfFieldID ), particleFieldID_( particleFieldID ), ac_( ac ) {}

   Type * operator()( IBlock* const block, const StructuredBlockStorage* const storage ) const
   {
      WALBERLA_ASSERT_NOT_NULLPTR( block );
      WALBERLA_ASSERT_NOT_NULLPTR( storage );

      auto * flagField     = block->getData< FlagField_T >( flagFieldID_ );
      auto *  pdfField     = block->getData< PdfField_T > ( pdfFieldID_ );
      auto * particleField = block->getData< lbm_mesapd_coupling::ParticleField_T > ( particleFieldID_ );

      const auto fluid = flagField->flagExists( Fluid_Flag ) ? flagField->getFlag( Fluid_Flag ) : flagField->registerFlag( Fluid_Flag );

      Type * handling = new Type( "moving obstacle boundary handling", flagField, fluid,
                                  SBB_T("SBB_BB", MO_BB_Flag,  pdfField, flagField, particleField, ac_, fluid, *storage, *block ),
                                  CLI_T("CLI_BB", MO_CLI_Flag, pdfField, flagField, particleField, ac_, fluid, *storage, *block ) );

      handling->fillWithDomain( FieldGhostLayers );

      return handling;
   }

private:

   const BlockDataID flagFieldID_;
   const BlockDataID pdfFieldID_;
   const BlockDataID particleFieldID_;

   shared_ptr<ParticleAccessor_T> ac_;
};


template< typename ParticleAccessor_T>
class DragForceEvaluator
{
public:
   DragForceEvaluator( SweepTimeloop* timeloop, Setup* setup, const shared_ptr< StructuredBlockStorage > & blocks,
                       const BlockDataID & flagFieldID, const BlockDataID & pdfFieldID,
                       const shared_ptr<ParticleAccessor_T>& ac, walberla::id_t sphereID )
:     timeloop_( timeloop ), setup_( setup ), blocks_( blocks ),
      flagFieldID_( flagFieldID ), pdfFieldID_( pdfFieldID ),
      ac_( ac ), sphereID_( sphereID ),
      normalizedDragOld_( 0.0 ), normalizedDragNew_( 0.0 )
   {
      // calculate the analytical drag force value based on the series expansion of chi
      // see also Sangani - Slow flow through a periodic array of spheres, IJMF 1982. Eq. 60 and Table 1
      real_t analyticalDrag = real_c(0);
      real_t tempChiPowS = real_c(1);

      // coefficients to calculate the drag in a series expansion
      real_t dragCoefficients[31] = {  real_c(1.000000),  real_c(1.418649),  real_c(2.012564),  real_c(2.331523),  real_c(2.564809),   real_c(2.584787),
                                       real_c(2.873609),  real_c(3.340163),  real_c(3.536763),  real_c(3.504092),  real_c(3.253622),   real_c(2.689757),
                                       real_c(2.037769),  real_c(1.809341),  real_c(1.877347),  real_c(1.534685), real_c(0.9034708),  real_c(0.2857896),
                                       real_c(-0.5512626), real_c(-1.278724),  real_c(1.013350),  real_c(5.492491),  real_c(4.615388), real_c(-0.5736023),
                                       real_c(-2.865924), real_c(-4.709215), real_c(-6.870076), real_c(0.1455304),  real_c(12.51891),   real_c(9.742811),
                                       real_c(-5.566269)};

      for(uint_t s = 0; s <= uint_t(30); ++s){
         analyticalDrag += dragCoefficients[s] * tempChiPowS;
         tempChiPowS *= setup->chi;
      }
      setup_->analyticalDrag = analyticalDrag;
   }

   // evaluate the acting drag force
   void operator()()
   {
      const uint_t timestep (timeloop_->getCurrentTimeStep()+1);

      if( timestep % setup_->checkFrequency != 0) return;

      // get force in x-direction acting on the sphere
      real_t forceX = computeDragForce();
      // get average volumetric flowrate in the domain
      real_t uBar = computeAverageVel();

      averageVel_ = uBar;

      // f_total = f_drag + f_buoyancy
      real_t totalForce = forceX  + real_c(4.0/3.0) * math::pi * setup_->radius * setup_->radius * setup_->radius * setup_->extForce ;

      real_t normalizedDragForce = totalForce / real_c( 6.0 * math::pi * setup_->visc * setup_->radius * uBar );

      // update drag force values
      normalizedDragOld_ = normalizedDragNew_;
      normalizedDragNew_ = normalizedDragForce;
   }

   // return the relative temporal change in the normalized drag
   real_t getDragForceDiff() const
   {
      return std::fabs( ( normalizedDragNew_ - normalizedDragOld_ ) / normalizedDragNew_ );
   }

   // return the drag force
   real_t getDragForce() const
   {
      return normalizedDragNew_;
   }

   real_t getAverageVel()
   {
      return averageVel_;
   }

   void logResultToFile( const std::string & filename ) const
   {
      // write to file if desired
      // format: length tau viscosity simulatedDrag analyticalDrag\n
      WALBERLA_ROOT_SECTION()
      {
         std::ofstream file;
         file.open( filename.c_str(), std::ofstream::app );
         file.precision(8);
         file << setup_->length << " " << setup_->tau << " " << setup_->visc << " " << normalizedDragNew_ << " " << setup_->analyticalDrag << "\n";
         file.close();
      }
   }

private:

   // obtain the drag force acting on the sphere by summing up all the process local parts of fX
   real_t computeDragForce()
   {
      size_t idx = ac_->uidToIdx(sphereID_);
      real_t force = real_t(0);
      if( idx!= ac_->getInvalidIdx())
      {
         force = ac_->getHydrodynamicForce(idx)[0];
      }

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace( force, mpi::SUM );
      }

      return force;
   }

   // calculate the average velocity in forcing direction (here: x) inside the domain (assuming dx=1)
   real_t computeAverageVel()
   {
      auto velocity_sum = real_t(0);
      // iterate all blocks stored locally on this process
      for( auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt )
      {
         // retrieve the pdf field and the flag field from the block
         PdfField_T  *  pdfField = blockIt->getData< PdfField_T >( pdfFieldID_ );
         FlagField_T * flagField = blockIt->getData< FlagField_T >( flagFieldID_ );

         // get the flag that marks a cell as being fluid
         auto fluid = flagField->getFlag( Fluid_Flag );

         auto xyzField = pdfField->xyzSize();
         for (auto cell : xyzField) {
            if( flagField->isFlagSet( cell.x(), cell.y(), cell.z(), fluid ) )
            {
             velocity_sum += pdfField->getVelocity(cell)[0];
            }
         }
      }

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace( velocity_sum, mpi::SUM );
      }

      return velocity_sum / real_c( setup_->length * setup_->length * setup_->length );
   }

   SweepTimeloop* timeloop_;

   Setup* setup_;

   shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID flagFieldID_;
   const BlockDataID pdfFieldID_;

   shared_ptr<ParticleAccessor_T> ac_;
   const walberla::id_t sphereID_;

   // drag coefficient
   real_t normalizedDragOld_;
   real_t normalizedDragNew_;

   real_t averageVel_;

};

//////////
// MAIN //
//////////


//*******************************************************************************************************************
/*!\brief Testcase that checks the drag force acting on a fixed sphere in the center of a cubic domain in Stokes flow
 *
 * The drag force for this problem (often denoted as Simple Cubic setup) is given by a semi-analytical series expansion.
 * The cubic domain is periodic in all directions, making it a physically infinite periodic array of spheres.
 *         _______________
 *      ->|               |->
 *      ->|      ___      |->
 *    W ->|     /   \     |-> E
 *    E ->|    |  x  |    |-> A
 *    S ->|     \___/     |-> S
 *    T ->|               |-> T
 *      ->|_______________|->
 *
 * The collision model used for the LBM is TRT with a relaxation parameter tau=1.5 and the magic parameter 3/16.
 * The Stokes approximation of the equilibrium PDFs is used.
 * The flow is driven by a constant body force of 1e-5.
 * The domain is length x length x length, and the sphere has a diameter of chi * length cells
 * The simulation is run until the relative change in the dragforce between 100 time steps is less than 1e-5.
 * The RPD is not used since the sphere is kept fixed and the force is explicitly reset after each time step.
 * To avoid periodicity constrain problems, the sphere is set as global.
 *
 */
//*******************************************************************************************************************


int main( int argc, char **argv )
{
   debug::enterTestMode();

   mpi::Environment env( argc, argv );

   auto processes = MPIManager::instance()->numProcesses();

   if( processes != 1 && processes != 2 && processes != 4 && processes != 8)
   {
      std::cerr << "Number of processes must be equal to either 1, 2, 4, or 8!" << std::endl;
      return EXIT_FAILURE;
   }

   ///////////////////
   // Customization //
   ///////////////////

   bool shortrun  = false;
   bool funcTest  = false;
   bool logging   = true;
   uint_t vtkIOFreq = 0;
   std::string baseFolder = "vtk_out_DragForceSphere";

   real_t tau     = real_c( 1.5 );
   real_t externalForcing = real_t(1e-8);
   uint_t length  = uint_c( 40 );

   MEMVariant method = MEMVariant::CLI;
   real_t bulkViscRateFactor = real_t(1); // ratio between bulk and shear viscosity related relaxation factors, 1 = TRT, > 1 for stability with MRT
   // see Khirevich - Coarse- and fine-grid numerical behavior of MRT/TRT lattice-Boltzmann schemes in regular and random sphere packings
   bool useSRT = false;
   real_t magicNumber = real_t(3) / real_t(16);
   bool useOmegaBulkAdaption = false;
   real_t adaptionLayerSize = real_t(2);


   for( int i = 1; i < argc; ++i )
   {
      if( std::strcmp( argv[i], "--shortrun" )             == 0 ) { shortrun = true; continue; }
      if( std::strcmp( argv[i], "--funcTest" )             == 0 ) { funcTest = true; continue; }
      if( std::strcmp( argv[i], "--noLogging" )            == 0 ) { logging = false; continue; }
      if( std::strcmp( argv[i], "--vtkIOFreq" )            == 0 ) { vtkIOFreq = uint_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--MEMVariant" )           == 0 ) { method = to_MEMVariant( argv[++i] ); continue; }
      if( std::strcmp( argv[i], "--tau" )                  == 0 ) { tau = real_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--extForce" )             == 0 ) { externalForcing = real_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--length" )               == 0 ) { length = uint_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--bulkViscRateFactor" )   == 0 ) { bulkViscRateFactor = real_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--useSRT" )               == 0 ) { useSRT = true; continue; }
      if( std::strcmp( argv[i], "--magicNumber" )          == 0 ) { magicNumber = real_c( std::atof( argv[++i] ) ); continue; }
      if( std::strcmp( argv[i], "--useOmegaBulkAdaption" ) == 0 ) { useOmegaBulkAdaption = true; continue; }
      if( std::strcmp( argv[i], "--adaptionLayerSize" )    == 0 ) { adaptionLayerSize = real_c(std::atof( argv[++i] )); continue; }
      WALBERLA_ABORT("Unrecognized command line argument found: " << argv[i]);
   }


   ///////////////////////////
   // SIMULATION PROPERTIES //
   ///////////////////////////

   Setup setup;

   setup.length         = length;                            // length of the cubic domain in lattice cells
   setup.chi            = real_c( 0.5 );                     // porosity parameter: diameter / length
   setup.tau            = tau;                               // relaxation time
   setup.extForce       = externalForcing;                   // constant body force in lattice units
   setup.checkFrequency = uint_t( 100 );                     // evaluate the drag force only every checkFrequency time steps
   setup.radius         = real_c(0.5) * setup.chi * real_c( setup.length );  // sphere radius
   setup.visc           = ( setup.tau - real_c(0.5) ) / real_c(3);   // viscosity in lattice units
   const real_t omega      = real_c(1) / setup.tau;          // relaxation rate
   const real_t dx         = real_c(1);                      // lattice dx
   const real_t convergenceLimit = real_t(0.1) * setup.extForce;           // tolerance for relative change in drag force
   const uint_t timesteps  =  funcTest ? 1 : ( shortrun ? uint_c(150) : uint_c( 100000 ) );  // maximum number of time steps for the whole simulation

   WALBERLA_LOG_INFO_ON_ROOT("tau = " << tau);
   WALBERLA_LOG_INFO_ON_ROOT("diameter = " << real_t(2) * setup.radius);
   WALBERLA_LOG_INFO_ON_ROOT("viscosity = " << setup.visc);
   WALBERLA_LOG_INFO_ON_ROOT("MEM variant = " << MEMVariant_to_string(method));
   WALBERLA_LOG_INFO_ON_ROOT("external forcing = " << setup.extForce);

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   const uint_t XBlocks = (processes >= 2) ? uint_t( 2 ) : uint_t( 1 );
   const uint_t YBlocks = (processes >= 4) ? uint_t( 2 ) : uint_t( 1 );
   const uint_t ZBlocks = (processes == 8) ? uint_t( 2 ) : uint_t( 1 );
   const uint_t XCells = setup.length / XBlocks;
   const uint_t YCells = setup.length / YBlocks;
   const uint_t ZCells = setup.length / ZBlocks;

   // create fully periodic domain
   auto blocks = blockforest::createUniformBlockGrid( XBlocks, YBlocks, ZBlocks, XCells, YCells, ZCells, dx, true,
                                                      true, true, true );


   /////////
   // RPD //
   /////////

   mesa_pd::domain::BlockForestDomain domain(blocks->getBlockForestPointer());

   //init data structures
   auto ps = std::make_shared<mesa_pd::data::ParticleStorage>(1);
   auto ss = std::make_shared<mesa_pd::data::ShapeStorage>();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor = make_shared<ParticleAccessor_T >(ps, ss);
   auto sphereShape = ss->create<mesa_pd::data::Sphere>( setup.radius );

   //////////////////
   // RPD COUPLING //
   //////////////////

   // connect to pe
   const real_t overlap = real_t( 1.5 ) * dx;

   if( setup.radius > real_c( setup.length ) * real_t(0.5) - overlap )
   {
      std::cerr << "Periodic sphere is too large and would lead to incorrect mapping!" << std::endl;
      // solution: create the periodic copies explicitly
      return EXIT_FAILURE;
   }

   // create the sphere in the middle of the domain
   // it is global and thus present on all processes

   Vector3<real_t> position (real_c(setup.length) * real_c(0.5));
   walberla::id_t sphereID;
   {
      mesa_pd::data::Particle&& p = *ps->create(true);
      p.setPosition(position);
      p.setInteractionRadius(setup.radius);
      p.setOwner(mpi::MPIManager::instance()->rank());
      p.setShapeID(sphereShape);
      sphereID = p.getUid();
   }

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////
   
   real_t lambda_e = lbm::collision_model::TRT::lambda_e( omega );
   real_t lambda_d = (useSRT) ? lambda_e : lbm::collision_model::TRT::lambda_d( omega, magicNumber );
   real_t omegaBulk = (useSRT) ? lambda_e : lbm_mesapd_coupling::omegaBulkFromOmega(omega, bulkViscRateFactor);

   // add omega bulk field
   BlockDataID omegaBulkFieldID = field::addToStorage<ScalarField_T>( blocks, "omega bulk field", omegaBulk, field::fzyx);

   // create the lattice model
#ifdef WALBERLA_BUILD_WITH_CODEGEN

#if defined(USE_TRTLIKE) || defined(USE_D3Q27TRTLIKE)
   WALBERLA_LOG_INFO_ON_ROOT("Using generated TRT-like lattice model!");
   WALBERLA_LOG_INFO_ON_ROOT(" - magic number " << magicNumber);
   WALBERLA_LOG_INFO_ON_ROOT(" - omegaBulk = " << omegaBulk << ", bulk visc. = " << lbm_mesapd_coupling::bulkViscosityFromOmegaBulk(omegaBulk) << " (bvrf " << bulkViscRateFactor << ")");
   WALBERLA_LOG_INFO_ON_ROOT(" - lambda_e " << lambda_e << ", lambda_d " << lambda_d << ", omegaBulk " << omegaBulk );
   WALBERLA_LOG_INFO_ON_ROOT(" - use omega bulk adaption = " << useOmegaBulkAdaption << " (adaption layer size = " << adaptionLayerSize << ")");
   LatticeModel_T latticeModel = LatticeModel_T(omegaBulkFieldID, setup.extForce, lambda_d, lambda_e);
#elif defined(USE_CUMULANTTRT)
   WALBERLA_LOG_INFO_ON_ROOT("Using generated cumulant TRT lattice model!");
   WALBERLA_LOG_INFO_ON_ROOT(" - magic number " << magicNumber);
   WALBERLA_LOG_INFO_ON_ROOT(" - lambda_e " << lambda_e << ", lambda_d " << lambda_d );
   LatticeModel_T latticeModel = LatticeModel_T(setup.extForce, lambda_d, lambda_e);
#elif defined(USE_CUMULANT)
   LatticeModel_T latticeModel = LatticeModel_T(setup.extForce, omega);
#endif

#else
   WALBERLA_LOG_INFO_ON_ROOT("Using waLBerla built-in MRT lattice model and ignoring omega bulk field since not supported!");
   WALBERLA_LOG_INFO_ON_ROOT(" - magic number " << magicNumber);
   WALBERLA_LOG_INFO_ON_ROOT(" - omegaBulk = " << omegaBulk << ", bulk visc. = " << lbm_mesapd_coupling::bulkViscosityFromOmegaBulk(omegaBulk) << " (bvrf " << bulkViscRateFactor << ")");
   WALBERLA_LOG_INFO_ON_ROOT(" - lambda_e " << lambda_e << ", lambda_d " << lambda_d << ", omegaBulk " << omegaBulk );

   LatticeModel_T latticeModel = LatticeModel_T(lbm::collision_model::D3Q19MRT( omegaBulk, omegaBulk, lambda_d, lambda_e, lambda_e, lambda_d ), ForceModel_T( Vector3<real_t> ( setup.extForce, 0, 0 ) ));
#endif


   // add PDF field
   BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (fzyx)", latticeModel,
                                                                         Vector3< real_t >( real_t(0) ), real_t(1),
                                                                         uint_t(1), field::fzyx );

   // add flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage<FlagField_T>( blocks, "flag field" );

   // add particle field
   BlockDataID particleFieldID = field::addToStorage<lbm_mesapd_coupling::ParticleField_T>( blocks, "particle field", accessor->getInvalidUid(), field::fzyx, FieldGhostLayers );

   // add boundary handling
   using BoundaryHandling_T = MyBoundaryHandling<ParticleAccessor_T>::Type;
   BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(MyBoundaryHandling<ParticleAccessor_T>( flagFieldID, pdfFieldID, particleFieldID, accessor), "boundary handling" );

   ///////////////
   // TIME LOOP //
   ///////////////

   // create the timeloop
   SweepTimeloop timeloop( blocks->getBlockStorage(), timesteps );

   // setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   blockforest::communication::UniformBufferedScheme< Stencil_T > optimizedPDFCommunicationScheme( blocks );
   optimizedPDFCommunicationScheme.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldID ) ); // optimized sync

   //blockforest::communication::UniformBufferedScheme< stencil::D3Q27 > optimizedPDFCommunicationScheme( blocks );
   //optimizedPDFCommunicationScheme.addPackInfo( make_shared< field::communication::PackInfo< PdfField_T > >( pdfFieldID ) );

   // initially map particles into the LBM simulation
   lbm_mesapd_coupling::MovingParticleMappingKernel<BoundaryHandling_T> movingParticleMappingKernel(blocks, boundaryHandlingID, particleFieldID);
   if( method == MEMVariant::CLI )
   {
      // uses a higher order boundary condition (CLI)
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, movingParticleMappingKernel, *accessor, MO_CLI_Flag);
   }else{
      // uses standard bounce back boundary conditions
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, movingParticleMappingKernel, *accessor, MO_BB_Flag);
   }

   // update bulk omega in all cells to adapt to changed particle position
   if(useOmegaBulkAdaption)
   {
      using OmegaBulkAdapter_T = lbm_mesapd_coupling::OmegaBulkAdapter<ParticleAccessor_T, mesa_pd::kernel::SelectAll>;
      real_t defaultOmegaBulk = lbm_mesapd_coupling::omegaBulkFromOmega(omega, real_t(1));
      shared_ptr<OmegaBulkAdapter_T> omegaBulkAdapter = make_shared<OmegaBulkAdapter_T>(blocks, omegaBulkFieldID, accessor, defaultOmegaBulk, omegaBulk, adaptionLayerSize, mesa_pd::kernel::SelectAll());
      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt) {
         (*omegaBulkAdapter)(blockIt.get());
      }
   }

   if( vtkIOFreq != uint_t(0) )
   {

      // spheres
      auto particleVtkOutput = make_shared<mesa_pd::vtk::ParticleVtkOutput>(ps);
      particleVtkOutput->addOutput<mesa_pd::data::SelectParticleLinearVelocity>("velocity");
      auto particleVtkWriter = vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkIOFreq, baseFolder, "simulation_step");
      timeloop.addFuncBeforeTimeStep( vtk::writeFiles( particleVtkWriter ), "VTK (sphere data)" );

      // pdf field
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData( blocks, "fluid_field", vtkIOFreq, 0, false, baseFolder );

      field::FlagFieldCellFilter< FlagField_T > fluidFilter( flagFieldID );
      fluidFilter.addFlag( Fluid_Flag );
      pdfFieldVTK->addCellInclusionFilter( fluidFilter );

      pdfFieldVTK->addCellDataWriter( make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >( pdfFieldID, "VelocityFromPDF" ) );
      pdfFieldVTK->addCellDataWriter( make_shared< lbm::DensityVTKWriter < LatticeModel_T, float > >( pdfFieldID, "DensityFromPDF" ) );

      timeloop.addFuncBeforeTimeStep( vtk::writeFiles( pdfFieldVTK ), "VTK (fluid field data)" );

      // omega bulk field
      timeloop.addFuncBeforeTimeStep( field::createVTKOutput<ScalarField_T, float>( omegaBulkFieldID, *blocks, "omega_bulk_field", vtkIOFreq, uint_t(0), false, baseFolder ), "VTK (omega bulk field)" );
   }


   // since external forcing is applied, the evaluation of the velocity has to be carried out directly after the streaming step
   // however, the default sweep is a  stream - collide step, i.e. after the sweep, the velocity evaluation is not correct
   // solution: split the sweep explicitly into collide and stream

   using DragForceEval_T = DragForceEvaluator<ParticleAccessor_T>;
   auto forceEval = make_shared< DragForceEval_T >( &timeloop, &setup, blocks, flagFieldID, pdfFieldID, accessor, sphereID );

#ifdef WALBERLA_BUILD_WITH_CODEGEN

   auto lbmSweep = LatticeModel_T::Sweep( pdfFieldID );
   // splitting stream and collide is currently buggy in lbmpy with vectorization -> build without optimize for local host and vectorization
   // collide LBM step
   timeloop.add() << Sweep([&lbmSweep](IBlock * const block){lbmSweep.collide(block);}, "collide LB sweep" );

   // add LBM communication function and boundary handling sweep (does the hydro force calculations and the no-slip treatment)
   timeloop.add() << BeforeFunction( optimizedPDFCommunicationScheme, "LBM Communication" )
                  << Sweep( BoundaryHandling_T::getBlockSweep( boundaryHandlingID ), "Boundary Handling" );

   // stream LBM step
   timeloop.add() << Sweep([&lbmSweep](IBlock * const block){lbmSweep.stream(block);}, "stream LB sweep" )
                  << AfterFunction( SharedFunctor< DragForceEval_T >( forceEval ), "drag force evaluation" );
#else
   auto lbmSweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldID, flagFieldID, Fluid_Flag );

   // collision sweep
   timeloop.add() << Sweep( lbm::makeCollideSweep( lbmSweep ), "cell-wise LB sweep (collide)" );

   // add LBM communication function and boundary handling sweep (does the hydro force calculations and the no-slip treatment)
   timeloop.add() << BeforeFunction( optimizedPDFCommunicationScheme, "LBM Communication" )
                  << Sweep( BoundaryHandling_T::getBlockSweep( boundaryHandlingID ), "Boundary Handling" );

   // streaming & force evaluation
   timeloop.add() << Sweep( lbm::makeStreamSweep( lbmSweep ), "cell-wise LB sweep (stream)" )
                  << AfterFunction( SharedFunctor< DragForceEval_T >( forceEval ), "drag force evaluation" );
#endif

   // resetting force
   timeloop.addFuncAfterTimeStep( [ps,accessor](){ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel(),*accessor);}, "reset force on sphere");

   timeloop.addFuncAfterTimeStep( RemainingTimeLogger( timeloop.getNrOfTimeSteps() ), "Remaining Time Logger" );

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;

   // time loop
   for (uint_t i = 0; i < timesteps; ++i )
   {
      // perform a single simulation step
      timeloop.singleStep( timeloopTiming );

      // check if the relative change in the normalized drag force is below the specified convergence criterion
      if ( i > setup.checkFrequency && forceEval->getDragForceDiff() < convergenceLimit )
      {
         // if simulation has converged, terminate simulation
         break;
      }

      if( std::isnan(forceEval->getDragForce()) ) WALBERLA_ABORT("Nan found!");

      if(i%1000 == 0)
      {
         WALBERLA_LOG_INFO_ON_ROOT("Current drag force: " << forceEval->getDragForce());
      }

   }

   WALBERLA_LOG_INFO_ON_ROOT("Final drag force: " << forceEval->getDragForce());
   WALBERLA_LOG_INFO_ON_ROOT("Re = " << forceEval->getAverageVel() * setup.radius * real_t(2) / setup.visc);

   timeloopTiming.logResultOnRoot();

   if ( !funcTest && !shortrun ){
      // check the result
      real_t relErr = std::fabs( ( setup.analyticalDrag - forceEval->getDragForce() ) / setup.analyticalDrag );
      if ( logging )
      {
         WALBERLA_ROOT_SECTION()
         {
            std::cout << "Analytical drag: " << setup.analyticalDrag << "\n"
                      << "Simulated drag: " << forceEval->getDragForce() << "\n"
                      << "Relative error: " << relErr << "\n";
         }

         std::string fileName( MEMVariant_to_string(method) );
         fileName += "_bvrf" + std::to_string(uint_c(bulkViscRateFactor));
         fileName += "_mn" + std::to_string(float(magicNumber));
         if(useOmegaBulkAdaption ) fileName += "_uOBA" + std::to_string(uint_c(adaptionLayerSize));
         if(useSRT) fileName += "_SRT";

         forceEval->logResultToFile( "log_DragForceSphereMEM_Generated_"+fileName+".txt" );
      }
   }

   return 0;

}

} //namespace drag_force_sphere_mem

int main( int argc, char **argv ){
   drag_force_sphere_mem::main(argc, argv);
}
