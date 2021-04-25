#include "core/all.h"

#include "blockforest/Initialization.h"
#include "blockforest/communication/all.h"

#include "core/logging/all.h"

#include "pe/basic.h"
#include "pe/Types.h"
#include "pe/synchronization/SyncShadowOwners.h"
#include "pe/utility/DestroyBody.h"
#include "pe/vtk/SphereVtkOutput.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D2Q9.h"
#include "lbm/sweeps/CellwiseSweep.h"
#include "lbm/sweeps/SweepWrappers.h"
#include "lbm/vtk/all.h"

#include "pe_coupling/mapping/all.h"
#include "pe_coupling/momentum_exchange_method/all.h"
#include "pe_coupling/utility/all.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/VTKOutput.h"

#include <tuple>

namespace spinners_suspension
{
// using
using namespace walberla;
using walberla::uint_t;

// typedefs
using LatticeModel_T = lbm::D2Q9<lbm::collision_model::TRT, false>;
const real_t magicNumberTRT = lbm::collision_model::TRT::threeSixteenth;

using Stencil_T = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField<LatticeModel_T>;

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;
using BodyField_T = GhostLayerField<pe::BodyID, 1>;

const uint_t FieldGhostLayers = 1;

// boundary handling
using NoSlip_T = lbm::NoSlip<LatticeModel_T, flag_t>;
using UBB_T = lbm::SimpleUBB<LatticeModel_T, flag_t>;
using Pressure_T = lbm::SimplePressure<LatticeModel_T, flag_t>;

using MEM_BB_T = pe_coupling::SimpleBB<LatticeModel_T, FlagField_T>;
using MEM_CLI_T = pe_coupling::CurvedLinear<LatticeModel_T, FlagField_T>;

using BoundaryHandling_T = BoundaryHandling<FlagField_T, Stencil_T, UBB_T, Pressure_T, MEM_BB_T, MEM_CLI_T>;

// body types
using BodyTypeTuple = std::tuple< pe::Plane, pe::Sphere>;

// flags

const FlagUID Fluid_Flag ( "fluid" );

const FlagUID NoSlip_Flag ("no slip");
const FlagUID UBB_Flag ( "velocity bounce back" );
const FlagUID Pressure_Flag ( "pressure" );

const FlagUID MEM_BB_Flag ("moving obstacle BB");
const FlagUID MEM_CLI_Flag ("moving obstable CLI");

const FlagUID FormerMEM_Flag ( "former moving obstacle" );

// coupling algorithm
enum MEMVariant {BB, CLI};

/////////////////////
// parameters
/////////////////////
struct Setup
{
    // domain parameters
    Vector3< uint_t > numBlocks; // number of blocks in x,y,z direction
    Vector3< uint_t > numCells; // number of cells in x,y,z direction
    Vector3< uint_t > domainSize; // domian size in x,y,z direction
    Vector3< bool > isPeriodic; // whether periodic in x,y,z direction
    bool oneBlockPerProcess;

    bool boundX; // bounding walls in x-axis
    bool boundY; // bounding walls in y-axis
    bool boundZ; // bounding walls in z-axis

    // simulation parameters
    uint_t timesteps; // simulation time steps
    real_t dt; // time interval
    real_t dx; // lattice spacing

    bool resolve_overlapping; // whether resolve initial particle overlappings
    uint_t resolve_maxsteps; // max resolve timesteps
    real_t resolve_dt; // time interval for resolving particle overlappings

    std::string coupling_method;
    MEMVariant memVariant; // momentum exchange method

    uint_t substepsPE; // number of PE calls in each subcycle

    // fluid properties
    real_t fluid_density; // density of fluid
    real_t viscosity; // viscosity of fluid
    real_t omega;

    // particle properties
    real_t particle_density; // density of particles

    uint_t numParticles; // number of particles

    uint_t particle_number_1; // number of type 1 particles
    uint_t particle_number_2; // number of type 2 particles
    real_t particle_diameter_1; // particle diameter of first type
    real_t particle_diameter_2; // particle diameter of second type
    real_t particle_diameter_max; // maxium possible particle diameters
    real_t particle_diameter_avg; // average particle diameter
    real_t particle_volume_avg; // average particle volume
    real_t particle_mass_avg; // average particle mass

    real_t restitutionCoeff; // Coefficient of restitution
    real_t frictionSCoeff; // Coefficient of static friction
    real_t frictionDCoeff; // Coefficient of dynamic friction
    real_t poisson; // Poisson's ratio
    real_t young; // Young's modulus
    real_t contactT; // contact time
    real_t stiffnessCoeff; // contact stiffness
    real_t dampingNCoeff; // damping coefficient in the normal direction
    real_t dampingTCoeff; // damping coefficient in the tangential direction

    real_t density_ratio;

    // output parameters
    std::string vtkBaseFolder;
    uint_t vtkWriteFrequency;

    std::string logBaseFolder;
    uint_t logInfoFrequency;

    void printSetup()
    {
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numCells);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(isPeriodic);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundX);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundY);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundZ);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(fluid_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(density_ratio);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_1);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_2);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_volume_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_mass_avg);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(vtkBaseFolder);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(vtkWriteFrequency);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(logInfoFrequency);
    }

    void sanity_check()
    {
        if ( isPeriodic[0] == true && numBlocks[0] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");
        if ( isPeriodic[1] == true && numBlocks[1] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");
        if ( isPeriodic[2] == true && numBlocks[2] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");

        if (numBlocks[0] * numCells[0] != domainSize[0] || numBlocks[1] * numCells[1] != domainSize[1] || numBlocks[2] * numCells[2] != domainSize[2])
            WALBERLA_ABORT("Domain decomposition does not fit the simulation domain.");


        WALBERLA_CHECK(particle_diameter_1 > 0 && particle_diameter_2 > 0, "particle diamters should be positive.");
    }
};

/////////////////////////
// Boundary conditions
/////////////////////////
class MyBoundaryHandling
{
public:

    MyBoundaryHandling( const BlockDataID & flagFieldID, const BlockDataID & pdfFieldID, const BlockDataID & bodyFieldID, const Vector3<real_t> & uInfty):
        flagFieldID_( flagFieldID ), pdfFieldID_( pdfFieldID ), bodyFieldID_( bodyFieldID ), velocity_( uInfty )
    {}

    BoundaryHandling_T * operator()( IBlock* const block, const StructuredBlockStorage* const storage) const;

private:
    const BlockDataID flagFieldID_;
    const BlockDataID pdfFieldID_;
    const BlockDataID bodyFieldID_;
    const Vector3<real_t> & velocity_;
};

BoundaryHandling_T * MyBoundaryHandling::operator()( IBlock* const block, const StructuredBlockStorage* const storage) const
{
    WALBERLA_ASSERT_NOT_NULLPTR( block );
    WALBERLA_ASSERT_NOT_NULLPTR( storage );

    FlagField_T * flagField = block->getData< FlagField_T >( flagFieldID_ );
    PdfField_T * pdfField = block->getData< PdfField_T >( pdfFieldID_ );
    BodyField_T * bodyField = block->getData< BodyField_T >( bodyFieldID_ );

    const auto fluid = flagField->getOrRegisterFlag(Fluid_Flag);

    BoundaryHandling_T * handling = new BoundaryHandling_T("moving obstacle boundary handling", flagField, fluid,
                                                            UBB_T("UBB", UBB_Flag, pdfField, velocity_),
                                                            Pressure_T("Pressure", Pressure_Flag, pdfField, real_c(1.0)),
                                                            MEM_BB_T("MEM_BB", MEM_BB_Flag, pdfField, flagField, bodyField, fluid, *storage, *block),
                                                            MEM_CLI_T("MEM_CLI", MEM_CLI_Flag, pdfField, flagField, bodyField, fluid, *storage, *block));
    
    const auto ubb = flagField->getFlag( UBB_Flag );
    const auto pressure = flagField->getFlag( Pressure_Flag );

    CellInterval domainBB = storage->getDomainCellBB();

    domainBB.xMin() -= cell_idx_c( FieldGhostLayers );
    domainBB.xMax() += cell_idx_c( FieldGhostLayers );

    domainBB.yMin() -= cell_idx_c( FieldGhostLayers );
    domainBB.yMax() += cell_idx_c( FieldGhostLayers );

    // 2D system skip z direction

    // south
    CellInterval south(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(), domainBB.xMax(), domainBB.yMin(), domainBB.zMin());
    storage->transformGlobalToBlockLocalCellInterval( south, *block);
    handling->forceBoundary( ubb, south);

    // north
    CellInterval north(domainBB.xMin(), domainBB.yMax(), domainBB.zMin(), domainBB.xMax(), domainBB.yMax(), domainBB.zMin());
    storage->transformGlobalToBlockLocalCellInterval( south, *block);
    handling->forceBoundary( pressure, north);

    // West and East are periodic

    // fill rest cells as fluid
    handling->fillWithDomain( domainBB );

    return handling;
}

/////////////////////////
// auxiliary functions
/////////////////////////
uint_t generateSingleSphere(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                            pe::MaterialID & material, const Vector3<real_t> pos, const real_t diameter);
uint_t generateRandomSpheresLayer(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                                  const Setup & setup, const AABB & domainGeneration, pe::MaterialID & material, real_t layer_zpos);
void resolve_particle_overlaps(const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & bodyStorageID,
                               const shared_ptr<pe::cr::ICR> & cr, const std::function<void (void)> & syncFunc, const Setup & setup);

class ForceOnBodiesAdder
{
public:

   ForceOnBodiesAdder( const shared_ptr<StructuredBlockStorage> & blockStorage, const BlockDataID & bodyStorageID,
                       const Vector3<real_t> & forcePerVolume )
         : blockStorage_( blockStorage ), bodyStorageID_( bodyStorageID ), forcePerVolume_( forcePerVolume )
   { }

   // set a force on all (only local, to avoid force duplication) bodies
   void operator()()
   {
      for( auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt )
      {
         for( auto bodyIt = pe::LocalBodyIterator::begin( *blockIt, bodyStorageID_); bodyIt != pe::LocalBodyIterator::end(); ++bodyIt )
         {
            real_t volume = bodyIt->getVolume();
            bodyIt->addForce ( forcePerVolume_ * volume );
         }
      }
   }

private:

   shared_ptr<StructuredBlockStorage> blockStorage_;
   const BlockDataID bodyStorageID_;
   Vector3<real_t> forcePerVolume_;
};

class CollisionPropertiesEvaluator
{
public:
   CollisionPropertiesEvaluator( pe::cr::ICR & collisionResponse ) : collisionResponse_( collisionResponse ), maximumPenetration_(real_t(0))
   {}

   void operator()()
   {
      real_t maxPen = collisionResponse_.getMaximumPenetration();
      maximumPenetration_ = std::max( maximumPenetration_, maxPen );
      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace( maximumPenetration_, mpi::MAX );
      }
   }
   real_t getMaximumPenetration()
   {
      return maximumPenetration_;
   }
   void resetMaximumPenetration()
   {
      maximumPenetration_ = real_t(0);
   }
private:
   pe::cr::ICR & collisionResponse_;
   real_t maximumPenetration_;
};

class RestingSphereForceEvaluator
{
public:
    RestingSphereForceEvaluator( SweepTimeloop* timeloop, const shared_ptr< StructuredBlockStorage > & blocks,
                                 const BlockDataID & bodyStorageID, uint_t averageFreq, const std::string & basefolder):
    timeloop_( timeloop ), blocks_( blocks ), bodyStorageID_( bodyStorageID ), averageFreq_( averageFreq )
    {
        std::ofstream file;
        filename_ = basefolder;
        filename_ += "/log_fixed.txt";
        file.open( filename_.c_str() );
        file << "# f_drag_curr f_drag_ave\n";
        file.close();
    }

    // evaluate forces
    void operator()()
    {
        const uint_t timestep (timeloop_->getCurrentTimeStep()+1);

        real_t currentForce = calcForce();
        currentAverage_ += currentForce;

        WALBERLA_ROOT_SECTION()
        {
            std::ofstream file;
            file.open( filename_.c_str(), std::ofstream::app);
            file.precision(8);
            file << timestep << "\t" << currentForce << "\t" << dragForceNew_ << std::endl;
            file.close();
        }

        if ( timestep % averageFreq_ != 0) return;

        dragForceOld_ = dragForceNew_;
        dragForceNew_ = currentAverage_ / real_c( averageFreq_ );
        currentAverage_ = real_t(0);
    }

    real_t getForceDiff() const
    {
        return std::fabs( (dragForceNew_ - dragForceOld_ ) / dragForceOld_ );
    }

    real_t getForce() const
    {
        return dragForceNew_;
    }

private:
    std::string filename_;

    SweepTimeloop* timeloop_;

    shared_ptr< StructuredBlockStorage > blocks_;
    const BlockDataID bodyStorageID_;

    uint_t averageFreq_;

    real_t currentAverage_ = real_t(0);
    real_t dragForceOld_ = real_t(0);
    real_t dragForceNew_ = real_t(0);

    real_t calcForce()
    {
        real_t force( real_t(0.0) );
        for( auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt )
            for( auto bodyIt = pe::BodyIterator::begin<pe::Sphere>(*blockIt, bodyStorageID_); bodyIt != pe::BodyIterator::end<pe::Sphere>(); ++bodyIt )
                force += bodyIt->getForce()[1];
        
        WALBERLA_MPI_SECTION()
        {
            mpi::allReduceInplace( force, mpi::SUM);
        }
        return force;
    }
};

//*******************************************************************************************************************
/*!\brief Simulation of a collection rotating colloids with the discrete particle method.
 *
 * The simulation features a fluidized bed with spherical particles inside a rectangular column.
 * The domain size is [32 x 16 x 256] * d_avg
 *
 */
//*******************************************************************************************************************
int main(int argc, char** argv)
{
    Environment env(argc, argv);

    if (argc < 2) { WALBERLA_ABORT("Please specify a parameter file as input argument."); }

    Setup setup;

    //////////////////////////////////
    // Parse parameters
    //////////////////////////////////

    //--------------------domain parameters------------------------------//
    auto domainParameters = env.config()->getOneBlock("DomainParameters");
    setup.numBlocks = domainParameters.getParameter< Vector3< uint_t > >("numBlocks");
    setup.domainSize = domainParameters.getParameter< Vector3< uint_t > >("domainSize");
    setup.isPeriodic = domainParameters.getParameter < Vector3<  bool > >("isPeriodic");
    setup.oneBlockPerProcess = domainParameters.getParameter< bool >("oneBlockPerProcess");
    setup.boundX = domainParameters.getParameter< bool >("boundX");
    setup.boundY = domainParameters.getParameter< bool >("boundY");
    setup.boundZ = domainParameters.getParameter< bool >("boundZ");

    //--------------------simulation parameters------------------------------//
    auto simulationParameters = env.config()->getOneBlock("SimulationParameters");
    setup.timesteps = simulationParameters.getParameter< uint_t >("timesteps");
    setup.dt = simulationParameters.getParameter< real_t >("dt");
    setup.dx = simulationParameters.getParameter< real_t >("dx");

    setup.resolve_overlapping = simulationParameters.getParameter< bool >("resolve_overlapping");
    setup.resolve_maxsteps = simulationParameters.getParameter< uint_t >("resolve_maxsteps");
    setup.resolve_dt = simulationParameters.getParameter< real_t >("resolve_dt");

    setup.substepsPE = simulationParameters.getParameter< uint_t >("substepsPE");
    setup.coupling_method = simulationParameters.getParameter< std::string >("coupling_method");
    if (setup.coupling_method == "MEM_BB")
        setup.memVariant = MEMVariant::BB;
    else if (setup.coupling_method == "MEM_CLI")
        setup.memVariant = MEMVariant::CLI;
    else
        WALBERLA_ABORT("Unsupported Coupling Method.");

    //--------------------particle parameters------------------------------//
    auto particleProperties = env.config()->getOneBlock("ParticleProperties");
    setup.particle_density = particleProperties.getParameter< real_t >("particle_density");

    setup.particle_number_1 = particleProperties.getParameter< uint_t >("particle_number_1");
    setup.particle_number_2 = particleProperties.getParameter< uint_t >("particle_number_2");
    setup.particle_diameter_1 = particleProperties.getParameter< real_t >("particle_diameter_1");
    setup.particle_diameter_2 = particleProperties.getParameter< real_t >("particle_diameter_2");
    setup.particle_diameter_max = std::max(setup.particle_diameter_1, setup.particle_diameter_2);
    setup.particle_diameter_avg = (setup.particle_diameter_1 + setup.particle_diameter_2) / real_t(2.0);
    setup.particle_volume_avg = setup.particle_diameter_avg * setup.particle_diameter_avg * setup.particle_diameter_avg * math::pi / real_t(6.0);
    setup.particle_mass_avg = setup.particle_density * setup.particle_volume_avg;

    // collision related material properties
    setup.restitutionCoeff = particleProperties.getParameter< real_t >("restitutionCoeff");
    setup.frictionSCoeff = particleProperties.getParameter< real_t >("frictionSCoeff");
    setup.frictionDCoeff = particleProperties.getParameter< real_t >("frictionDCoeff");
    setup.poisson = particleProperties.getParameter< real_t >("poisson");
    setup.young = particleProperties.getParameter< real_t >("young");
    setup.stiffnessCoeff = particleProperties.getParameter< real_t >("stiffnessCoeff");
    setup.dampingNCoeff = particleProperties.getParameter< real_t >("dampingNCoeff");
    setup.dampingTCoeff = particleProperties.getParameter< real_t >("dampingTCoeff");
    setup.contactT = particleProperties.getParameter< real_t >("contactT");

    real_t mij = setup.particle_mass_avg * setup.particle_mass_avg / (setup.particle_mass_avg + setup.particle_mass_avg);

    //setup.contactT = real_t(2.0) * math::pi * mij / (std::sqrt(real_t(4) * mij * setup.stiffnessCoeff - setup.dampingNCoeff * setup.dampingNCoeff)); //formula from Uhlman

    // estimate from contact time

    setup.stiffnessCoeff = math::pi * math::pi * mij / (setup.contactT * setup.contactT * (real_t(1.0) - 
    std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff) / 
    (math::pi * math::pi + std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff))));

    setup.dampingNCoeff = -real_t(2.0) * std::sqrt(mij * setup.stiffnessCoeff) * (std::log(setup.restitutionCoeff) / 
    std::sqrt(math::pi * math::pi + (std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff))));
    setup.dampingTCoeff = setup.dampingNCoeff;

    //--------------------fluid parameters------------------------------//
    auto fluidProperties = env.config()->getOneBlock("FluidProperties");
    setup.fluid_density = fluidProperties.getParameter< real_t >("fluid_density");
    setup.viscosity = fluidProperties.getParameter< real_t >("viscosity");
    setup.omega = lbm::collision_model::omegaFromViscosity(setup.viscosity);

    real_t Re_target = real_c(1.0);
    real_t uIn = Re_target * setup.viscosity / setup.particle_diameter_avg;
    Vector3< real_t > uInfty( real_t(0.0), uIn, real_t(0.0));
    WALBERLA_LOG_INFO_ON_ROOT("viscosity: " << setup.viscosity);
    WALBERLA_LOG_INFO_ON_ROOT("diameter: " << setup.particle_diameter_avg);
    WALBERLA_LOG_INFO_ON_ROOT("u: " << uInfty);

    setup.density_ratio = setup.particle_density / setup.fluid_density;

    //--------------------output parameters------------------------------//
    auto outputParameters = env.config()->getOneBlock("OutputParameters");
    setup.vtkBaseFolder = outputParameters.getParameter< std::string >("vtkBaseFolder");
    setup.vtkWriteFrequency = outputParameters.getParameter< uint_t >("vtkWriteFrequency");
    setup.logBaseFolder = outputParameters.getParameter< std::string >("logBaseFolder");
    setup.logInfoFrequency = outputParameters.getParameter< uint_t >("logInfoFrequency");

    // configure block decomposition
    const uint_t XBlocks = setup.numBlocks[0];
    const uint_t YBlocks = setup.numBlocks[1];
    const uint_t ZBlocks = setup.numBlocks[2];

    const uint_t Lx = setup.domainSize[0];
    const uint_t Ly = setup.domainSize[1];
    const uint_t Lz = setup.domainSize[2];

    const bool xPeriodic = setup.isPeriodic[0];
    const bool yPeriodic = setup.isPeriodic[1];
    const bool zPeriodic = setup.isPeriodic[2];

    const uint_t XCells = Lx / XBlocks;
    const uint_t YCells = Ly / YBlocks;
    const uint_t ZCells = Lz / ZBlocks;

    setup.numCells = Vector3<uint_t>(XCells, YCells, ZCells);

    // check sanity of input parameters
    setup.sanity_check();
    
    setup.printSetup();

    ////////////////////////////
    // simulation setup
    ////////////////////////////

    // simulation domain
    const auto domainSimulation = AABB(real_c(0.0), real_c(0.0), real_c(0.0),
    real_c(setup.domainSize[0]), real_c(setup.domainSize[1]), real_c(setup.domainSize[2]) );

    // create blockforect
    //auto forest = blockforest::createBlockForest(domainSimulation, setup.numBlocks, setup.isPeriodic);
    auto blocks = blockforest::createUniformBlockGrid(XBlocks, YBlocks, ZBlocks, 
                                                      XCells, YCells, ZCells,
                                                      setup.dx,
                                                      setup.oneBlockPerProcess,
                                                      xPeriodic, yPeriodic, zPeriodic);

    //--------------
    // PE setup
    //--------------

    // generate IDs of specified PE body types
    pe::SetBodyTypeIDs< BodyTypeTuple >::execute();

    // add global body storage for PE bodies
    shared_ptr< pe::BodyStorage > globalBodyStorage = make_shared< pe::BodyStorage >();

    // add block-local body storage
    const auto bodyStorageID = blocks->addBlockData(pe::createStorageDataHandling< BodyTypeTuple >(), "bodyStorage");

    // add data-handling for coarse collision dection
    const auto ccdID = blocks->addBlockData(pe::ccd::createHashGridsDataHandling(globalBodyStorage, bodyStorageID), "ccd");

    // add data-handling for fine collision dection
    const auto fcdID = blocks->addBlockData(pe::fcd::createGenericFCDDataHandling< BodyTypeTuple, pe::fcd::AnalyticCollideFunctor >(), "fcd");

    // add contact solver - using soft contact model
    //const auto cr = make_shared< pe::cr::DEM >(globalBodyStorage, blocks->getBlockStoragePointer(), bodyStorageID, ccdID, fcdID);
    // add contact solver - using hard contact model
    const auto cr = make_shared< pe::cr::HCSITS >(globalBodyStorage, blocks->getBlockStoragePointer(), bodyStorageID, ccdID, fcdID);
    cr->setMaxIterations(10);
    cr->setRelaxationParameter(0.75);
    cr->setRelaxationModel( pe::cr::HardContactSemiImplicitTimesteppingSolvers::ApproximateInelasticCoulombContactByDecoupling );

    // set up synchronization procedure
    const real_t overlap = real_c( 1.5 ) * setup.dx;

    std::function<void(void)> PEsyncCall;
    if ( XBlocks <= uint_t(4) )
        PEsyncCall = std::bind( pe::syncNextNeighbors<BodyTypeTuple>, std::ref(blocks->getBlockForest()), bodyStorageID, static_cast<WcTimingTree*>(nullptr), overlap, false);
    else
        PEsyncCall = std::bind( pe::syncShadowOwners<BodyTypeTuple>, std::ref(blocks->getBlockForest()), bodyStorageID, static_cast<WcTimingTree*>(nullptr), overlap, false);

    // define particle material
    auto peMaterial = pe::createMaterial("particleMat", setup.particle_density, setup.restitutionCoeff, 
    setup.frictionSCoeff, setup.frictionDCoeff, setup.poisson, setup.young,
    setup.stiffnessCoeff, setup.dampingNCoeff, setup.dampingTCoeff);

    WALBERLA_LOG_INFO_ON_ROOT("Summary of particle material properties:\n"
                              << " - coefficient of restitution = " << setup.restitutionCoeff << "\n"
                              << " - coefficient of static friction = " << setup.frictionSCoeff << "\n"
                              << " - coefficient of dynamic friction = " << setup.frictionDCoeff << "\n"
                              << " - stiffness coefficient kn = " << setup.stiffnessCoeff << "\n"
                              << " - normal damping coefficient = " << setup.dampingNCoeff << "\n"
                              << " - tangential damping coefficient = " << setup.dampingTCoeff << "\n"
                              << " - contact time Tc = " << setup.contactT);

    // create bounding walls
    // top and bottom - z
    if (setup.boundZ)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainSimulation.zMax()), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0,  1), Vector3<real_t>(0, 0, domainSimulation.zMin()), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the z direction created.");
    }
    // front and back - y
    if (setup.boundY)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0,  1, 0), Vector3<real_t>(0, domainSimulation.yMin(), 0), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, -1, 0), Vector3<real_t>(0, domainSimulation.yMax(), 0), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the y direction created.");
    }
    // left and right - x
    if (setup.boundX)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>( 1, 0, 0), Vector3<real_t>(domainSimulation.xMin(), 0, 0), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(-1, 0, 0), Vector3<real_t>(domainSimulation.xMax(), 0, 0), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the x direction created.");
    }

    ////////////////////////////
    // Initialization
    ////////////////////////////

    // generate initial spherical particles

    real_t layer_thickness = setup.dx;
    real_t layer_zpos = real_c(0.5) * setup.dx;
    real_t radius_max = setup.particle_diameter_max / real_c(2.0);
    const auto domainGeneration = AABB(domainSimulation.xMin() + radius_max, domainSimulation.yMin() + radius_max, domainSimulation.zMin(), 
                                       domainSimulation.xMax() - radius_max, domainSimulation.yMax() - radius_max, domainSimulation.zMin() + layer_thickness);

    // random generation of spherical particles
    //setup.numParticles = generateRandomSpheresLayer(blocks, globalBodyStorage, bodyStorageID, setup, domainGeneration, peMaterial, layer_zpos);

    setup.numParticles = generateSingleSphere(blocks, globalBodyStorage, bodyStorageID, peMaterial, Vector3<real_t>(real_t(Lx/2.0), real_t(Ly/2.0), real_t(0.0)), setup.particle_diameter_1);

    // sync the created particles between processes
    PEsyncCall();

    WALBERLA_LOG_INFO_ON_ROOT(setup.numParticles << " spheres created.");

    //---------------------
    // Add data to blocks
    //---------------------

    // create the lattice model
    LatticeModel_T latticeModel = LatticeModel_T( lbm::collision_model::TRT::constructWithMagicNumber( setup.omega, magicNumberTRT) );

    // add pdf field
    BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (zyxf)", latticeModel, uInfty, real_t(1.0), FieldGhostLayers, field::zyxf);

    // add flag field
    BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field" );

    // add body field
    BlockDataID bodyFieldID = field::addToStorage< BodyField_T >( blocks, "body field", nullptr, field::zyxf);

    // add boundary handling
    BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >( MyBoundaryHandling( flagFieldID, pdfFieldID, bodyFieldID, uInfty) );

    // mapping rigid bodies into LBM grids
    if ( setup.memVariant == MEMVariant::BB)
        pe_coupling::mapMovingBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, MEM_BB_Flag, pe_coupling::selectRegularBodies );
    else if ( setup.memVariant == MEMVariant::CLI)
        pe_coupling::mapMovingBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, MEM_CLI_Flag, pe_coupling::selectRegularBodies );
    else
        WALBERLA_ABORT("Unsupported coupling method.");

    // mapping bound walls into LBM grids
    //pe_coupling::mapMovingBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, NoSlip_Flag, pe_coupling::selectGlobalBodies );

    //---------------------
    // setup LBM communication scheme
    //---------------------
    blockforest::communication::UniformBufferedScheme< Stencil_T > pdfCommunicationScheme( blocks );
    pdfCommunicationScheme.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldID ) );

    //-------------
    // PE-only initial simulations
    //-------------

    // resolve particle overlaps
    if(setup.resolve_overlapping)
    {
        WALBERLA_LOG_INFO_ON_ROOT("-----Resolving Particle Overlaps Start-----");
        resolve_particle_overlaps(blocks, bodyStorageID, cr, PEsyncCall, setup);
        WALBERLA_LOG_INFO_ON_ROOT("-----Resolving Particle Overlaps End-----");
    }

    //-------------
    // Output setup
    //-------------

    // add vtk output for the domain decomposition
    vtk::writeDomainDecomposition(blocks, "domain_decomposition", setup.vtkBaseFolder);

    // add vtk output for particles
    const auto bodyVTKOutput = make_shared< pe::SphereVtkOutput >(bodyStorageID, blocks->getBlockStorage());
    const auto bodyVTKWriter = vtk::createVTKOutput_PointData(bodyVTKOutput, "bodies", setup.vtkWriteFrequency, setup.vtkBaseFolder);
    bodyVTKWriter->write();

    // add vtk output for fluid
    auto pdfFieldVTK = vtk::createVTKOutput_BlockData( blocks, "fluid_field", setup.vtkWriteFrequency, 0, false, setup.vtkBaseFolder);

    field::FlagFieldCellFilter< FlagField_T > fluidFilter( flagFieldID );
    fluidFilter.addFlag( Fluid_Flag );
    pdfFieldVTK->addCellInclusionFilter( fluidFilter );

    pdfFieldVTK->addCellDataWriter( make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >( pdfFieldID, "VelocityFromPDF") );
    pdfFieldVTK->addCellDataWriter( make_shared< lbm::DensityVTKWriter<LatticeModel_T, float> >( pdfFieldID, "DensityFromPDF") );

    //pdfFieldVTK->write();

    /////////////////////////
    //  Time loop
    /////////////////////////

    SweepTimeloop timeloopInit( blocks->getBlockStorage(), 10000);

    auto sweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldID, flagFieldID, Fluid_Flag );

    // collision sweep
    timeloopInit.add() << Sweep( lbm::makeCollideSweep( sweep ), "cell-wise LB collide sweep");

    // LBM communication and boundary hanlding sweep
    timeloopInit.add() << BeforeFunction( pdfCommunicationScheme, "LBM communication")
                       << Sweep( BoundaryHandling_T::getBlockSweep( boundaryHandlingID ), "Boundary Handling");
    
    // streaming
    timeloopInit.add() << Sweep( lbm::makeStreamSweep( sweep ), "cell-wise LB stream sweep");

    // evaluate the drag force
    shared_ptr< RestingSphereForceEvaluator > forceEval = make_shared< RestingSphereForceEvaluator >( &timeloopInit, blocks, bodyStorageID, 10, setup.vtkBaseFolder);
    timeloopInit.addFuncAfterTimeStep( SharedFunctor< RestingSphereForceEvaluator >( forceEval ), "Evaluating drag force");

    // reset all force and torque
    timeloopInit.addFuncAfterTimeStep( pe_coupling::ForceTorqueOnBodiesResetter( blocks, bodyStorageID ), "Resetting force on body");

    // output
    timeloopInit.addFuncAfterTimeStep( vtk::writeFiles( pdfFieldVTK ), "VTK pdf field");

    // timer
    timeloopInit.addFuncAfterTimeStep( RemainingTimeLogger( timeloopInit.getNrOfTimeSteps(), real_t(30) ), "Remaining Time");

    // execute initial simulation

    WcTimingPool timeloopInitTiming;

    for( uint_t i = 1; i < 10000; ++i)
    {
        timeloopInit.singleStep( timeloopInitTiming );

        WALBERLA_LOG_INFO_ON_ROOT(" current force: " << forceEval->getForce() );
    }

    // create sediments
    /*
    cr->setGlobalLinearAcceleration(Vector3< real_t >(real_t(0), real_t(-0.0001), real_t(0.0)));
    for (uint_t pestep = uint_c(0); pestep < 100000; ++pestep)
    {
        for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
        {
            for ( auto bodyIt = pe::BodyIterator::begin( *blockIt, bodyStorageID); bodyIt != pe::BodyIterator::end(); ++bodyIt)
            {
                Vector3<real_t> vel = bodyIt->getLinearVel();
                vel[2] = real_c(0.0);
                bodyIt->setLinearVel(vel);

                Vector3<real_t> ang(real_t(0.0), real_t(0.0), real_t(0.001));
                bodyIt->setAngularVel(ang);
            }
        }

        cr->timestep(real_t(0.25));
        PEsyncCall();

        bodyVTKWriter->write();

        if (pestep % setup.logInfoFrequency == uint_c(0))
            WALBERLA_LOG_INFO_ON_ROOT("sediment timestep: " << pestep);
    }
    */

    return EXIT_SUCCESS;
}

uint_t generateSingleSphere(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                            pe::MaterialID & material, const Vector3<real_t> pos, const real_t diameter)
{
    //generate a single sphere at specified location (x,y,z)

    WALBERLA_LOG_INFO_ON_ROOT("Creating a sphere with diameter = " << diameter << " at location " << pos);

    pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, pos, diameter * real_c(0.5), material);

    return uint_t(1);
}

uint_t generateRandomSpheresLayer(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                                const Setup & setup, const AABB & domainGeneration, pe::MaterialID & material, real_t layer_zpos)
{
    //Randomly generate certain number of bidisperse spheres inside a specified domain

    uint_t N1 = 0;
    uint_t N2 = 0;

    real_t xpos, ypos, zpos, diameter;

    math::seedRandomGenerator( std::mt19937::result_type(std::time(nullptr)));

    WALBERLA_LOG_INFO_ON_ROOT("Creating " << setup.particle_number_1 << " type 1 spheres with diameter = " << setup.particle_diameter_1);

    while (N1 < setup.particle_number_1)
    {
        WALBERLA_ROOT_SECTION()
        {
            xpos = math::realRandom<real_t>(domainGeneration.xMin(), domainGeneration.xMax());
            ypos = math::realRandom<real_t>(domainGeneration.yMin(), domainGeneration.yMax());
            //zpos = math::realRandom<real_t>(domainGeneration.zMin(), domainGeneration.zMax());
            zpos = layer_zpos;
            diameter = setup.particle_diameter_1;
        }

        WALBERLA_MPI_SECTION()
        {
            mpi::broadcastObject(xpos);
            mpi::broadcastObject(ypos);
            mpi::broadcastObject(zpos);
            mpi::broadcastObject(diameter);
        }

        //pe::SphereID sp = pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);

        //if (sp != nullptr) ++N1;

        pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);
        ++N1;
    }

    WALBERLA_LOG_INFO_ON_ROOT("Creating " << setup.particle_number_2 << " type 2 spheres with diameter = " << setup.particle_diameter_2);

    while (N2 < setup.particle_number_2)
    {
        WALBERLA_ROOT_SECTION()
        {
            xpos = math::realRandom<real_t>(domainGeneration.xMin(), domainGeneration.xMax());
            ypos = math::realRandom<real_t>(domainGeneration.yMin(), domainGeneration.yMax());
            //zpos = math::realRandom<real_t>(domainGeneration.zMin(), domainGeneration.zMax());
            zpos = layer_zpos;
            diameter = setup.particle_diameter_2;
        }

        WALBERLA_MPI_SECTION()
        {
            mpi::broadcastObject(xpos);
            mpi::broadcastObject(ypos);
            mpi::broadcastObject(zpos);
            mpi::broadcastObject(diameter);
        }

        //pe::SphereID sp = pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);

        //if (sp != nullptr) ++N2;

        pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);
        ++N2;
    }

    return N1 + N2;
}

void resolve_particle_overlaps(const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & bodyStorageID, 
                               const shared_ptr<pe::cr::ICR> & cr, const std::function<void (void)> & syncFunc, const Setup & setup)
{
    // collision properties evaluator
    auto OverlapEvaluator = make_shared<CollisionPropertiesEvaluator>(*cr);

    const uint_t PEsteps = setup.resolve_maxsteps;
    const real_t dt = setup.resolve_dt;

    // temperary bounding plane
    //auto boundingPlane = pe::createPlane(*globalBodyStorage, 1, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainGeneration.zMax() + radius_max), peMaterial);

    for (uint_t pestep = uint_c(0); pestep < PEsteps; ++pestep)
    {
        cr->timestep(dt);
        syncFunc();

        // reset all velocities to zero
        for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
        {
            for ( auto bodyIt = pe::BodyIterator::begin( *blockIt, bodyStorageID); bodyIt != pe::BodyIterator::end(); ++bodyIt)
            {
                bodyIt->setLinearVel(Vector3<real_t>(real_t(0.0)));
                bodyIt->setAngularVel(Vector3<real_t>(real_t(0.0)));
            }
        }

        OverlapEvaluator->operator()();

        real_t maxOverlap = OverlapEvaluator->getMaximumPenetration();

        WALBERLA_LOG_INFO_ON_ROOT("timestep: " << pestep << " - current max overlap = " << maxOverlap / setup.particle_diameter_avg * real_c(100) << "%");

        OverlapEvaluator->resetMaximumPenetration();
    }

    // destroy temperary bounding plane
    //pe::destroyBodyBySID(*globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, boundingPlane->getSystemID());

    // comunication
    syncFunc();
}

} //namespace spinners_suspension

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}