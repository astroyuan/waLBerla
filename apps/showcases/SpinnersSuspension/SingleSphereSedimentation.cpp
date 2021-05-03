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

using BoundaryHandling_T = BoundaryHandling<FlagField_T, Stencil_T, NoSlip_T, UBB_T, Pressure_T, MEM_BB_T, MEM_CLI_T>;

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
    bool oneBlockPerProcess; // whether assign one block per process

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

    uint_t numPEsubsteps; // number of PE calls in each time step
    uint_t numLBMsubsteps; // number of LBM calls in each time step

    // fluid properties
    real_t fluid_density; // density of fluid
    real_t viscosity; // viscosity of fluid
    real_t omega;
    real_t tau;

    real_t u_ref; // characteristic velocity
    real_t x_ref; // characteristic length scale
    real_t t_ref; // characteristic time scale
    real_t Re; // Reynolds number

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
    real_t accg; // gravitational acceleration

    // output parameters
    std::string BaseFolder;

    std::string vtkBaseFolder;
    uint_t vtkWriteFrequency;

    std::string logBaseFolder;
    uint_t logInfoFrequency;

    void printSetup()
    {
        WALBERLA_LOG_INFO_ON_ROOT("Below is a summary of parameters setup:");

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numCells);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(isPeriodic);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(oneBlockPerProcess);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundX);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundY);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(boundZ);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(timesteps);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(dt);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(dx);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numLBMsubsteps);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numPEsubsteps);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(coupling_method);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(fluid_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(density_ratio);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_number_1);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_1);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_number_2);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_2);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_volume_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_mass_avg);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(BaseFolder);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(vtkBaseFolder);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(vtkWriteFrequency);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(logBaseFolder);
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

        if ( particle_density < fluid_density )
            WALBERLA_ABORT("The case where particle density is smaller than fluid density needs special treatment.")

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
                                                            NoSlip_T("NoSlip", NoSlip_Flag, pdfField),
                                                            UBB_T("UBB", UBB_Flag, pdfField, velocity_),
                                                            Pressure_T("Pressure", Pressure_Flag, pdfField, real_c(1.0)),
                                                            MEM_BB_T("MEM_BB", MEM_BB_Flag, pdfField, flagField, bodyField, fluid, *storage, *block),
                                                            MEM_CLI_T("MEM_CLI", MEM_CLI_Flag, pdfField, flagField, bodyField, fluid, *storage, *block));
    
    const auto noslip = flagField->getFlag( NoSlip_Flag );
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
    storage->transformGlobalToBlockLocalCellInterval( south, *block );
    handling->forceBoundary( noslip, south );

    // north
    CellInterval north(domainBB.xMin(), domainBB.yMax(), domainBB.zMin(), domainBB.xMax(), domainBB.yMax(), domainBB.zMin());
    storage->transformGlobalToBlockLocalCellInterval( north, *block );
    handling->forceBoundary( noslip, north );

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
                               pe::cr::ICR & cr, const std::function<void (void)> & syncFunc, const Setup & setup);

class Enforce2D
{
public:

   Enforce2D( const shared_ptr<StructuredBlockStorage> & blockStorage, const BlockDataID & bodyStorageID, const uint_t & direction )
         : blockStorage_( blockStorage ), bodyStorageID_( bodyStorageID ), direction_( direction )
   { }

   // set a force on all (only local, to avoid force duplication) bodies
   void operator()()
   {
      for( auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt )
      {
         for( auto bodyIt = pe::LocalBodyIterator::begin( *blockIt, bodyStorageID_); bodyIt != pe::LocalBodyIterator::end(); ++bodyIt )
         {
            force_ = bodyIt->getForce();
            force_[direction_] = real_t(0.0);
            bodyIt->setForce ( force_ );

            velocity_ = bodyIt->getLinearVel();
            velocity_[direction_] = real_t(0.0);
            bodyIt->setLinearVel(velocity_);
            //bodyIt->resetForceAndTorque();
         }
      }
   }

private:

   shared_ptr<StructuredBlockStorage> blockStorage_;
   const BlockDataID bodyStorageID_;
   const uint_t direction_;
   Vector3<real_t> force_;
   Vector3<real_t> velocity_;
};

class PrescribeAngularVel
{
public:

   PrescribeAngularVel( const shared_ptr<StructuredBlockStorage> & blockStorage, const BlockDataID & bodyStorageID, const Vector3<real_t> & angular_velocity )
         : blockStorage_( blockStorage ), bodyStorageID_( bodyStorageID ), angular_velocity_( angular_velocity )
   { }

   void operator()()
   {
      for( auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt )
      {
         for( auto bodyIt = pe::LocalBodyIterator::begin( *blockIt, bodyStorageID_); bodyIt != pe::LocalBodyIterator::end(); ++bodyIt )
         {
            bodyIt->setAngularVel( angular_velocity_ );
         }
      }
   }

private:

   shared_ptr<StructuredBlockStorage> blockStorage_;
   const BlockDataID bodyStorageID_;
   Vector3<real_t> angular_velocity_;
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

class SpherePropertiesLogger
{
public:
    SpherePropertiesLogger( const shared_ptr< SweepTimeloop > & timeloop, const shared_ptr< StructuredBlockStorage > & blocks,
                            const BlockDataID & bodyStorageID, const uint_t logFreq, const std::string & basefolder, const std::string & filename):
    timeloop_( timeloop ), blocks_( blocks ), bodyStorageID_( bodyStorageID ), logFreq_(logFreq)
    {
        WALBERLA_ROOT_SECTION()
        {
            // write the file header
            std::ofstream file;
            filename_ = basefolder + '/' + filename;
            file.open( filename_.c_str() );
            file << "# timestep posX posY posZ velX velY velZ forceX forceY forceZ\n";
            file.close();
        }
    }

    void operator()()
    {
        timestep_ = timeloop_->getCurrentTimeStep();

        Vector3<real_t> pos(real_t(0.0));
        Vector3<real_t> vel(real_t(0.0));
        Vector3<real_t> force(real_t(0.0));

        for( auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt )
        {
            for( auto bodyIt = pe::LocalBodyIterator::begin( *blockIt, bodyStorageID_); bodyIt != pe::LocalBodyIterator::end(); ++bodyIt )
            {
                pos = bodyIt->getPosition();
                vel = bodyIt->getLinearVel();
                force = bodyIt->getForce();
            }
        }

        WALBERLA_MPI_SECTION()
        {
            mpi::allReduceInplace( pos[0], mpi::SUM);
            mpi::allReduceInplace( pos[1], mpi::SUM);
            mpi::allReduceInplace( pos[2], mpi::SUM);

            mpi::allReduceInplace( vel[0], mpi::SUM);
            mpi::allReduceInplace( vel[1], mpi::SUM);
            mpi::allReduceInplace( vel[2], mpi::SUM);

            mpi::allReduceInplace( force[0], mpi::SUM);
            mpi::allReduceInplace( force[1], mpi::SUM);
            mpi::allReduceInplace( force[2], mpi::SUM);
        }

        pos_ = pos;
        vel_ = vel;
        force_ = force;

        if( timestep_ % logFreq_ == 0)
        {
            writeFiles();
        }
    }

    void writeFiles()
    {
        WALBERLA_ROOT_SECTION()
        {
            // write quantites in current time step to files
            std::ofstream file;
            file.open( filename_.c_str(), std::ofstream::app );

            file << timestep_ << ' '
                 << pos_[0] << ' ' << pos_[1] << ' ' << pos_[2] << ' '
                 << vel_[0] << ' ' << vel_[1] << ' ' << vel_[2] << ' '
                 << force_[0] << ' ' << force_[1] << ' ' << force_[2] << ' '
                 << "\n";
            
            file.close();
        }
    }

private:
    std::string filename_;

    shared_ptr< SweepTimeloop > timeloop_;

    shared_ptr< StructuredBlockStorage > blocks_;
    const BlockDataID bodyStorageID_;

    const uint_t logFreq_;

    uint_t timestep_;
    Vector3<real_t> pos_;
    Vector3<real_t> vel_;
    Vector3<real_t> force_;
};

//*******************************************************************************************************************
/*!\brief Simulation of a population of rotating colloids in a 2D plane.
 *
 * pending
 *
 *
 */
//*******************************************************************************************************************
int main(int argc, char** argv)
{
    Environment env(argc, argv);

    if (argc < 2) { WALBERLA_ABORT("Please specify a parameter file as input argument."); }

    //logging::Logging::instance()->setLogLevel(logging::Logging::LogLevel::DETAIL);

    Setup setup;

    //////////////////////////////////
    /////   Parse Parameters  ////////
    //////////////////////////////////

    //-----------------------domain parameters------------------------------//
    auto domainParameters = env.config()->getOneBlock("DomainParameters");
    setup.numBlocks = domainParameters.getParameter< Vector3< uint_t > >("numBlocks");
    setup.domainSize = domainParameters.getParameter< Vector3< uint_t > >("domainSize");
    setup.isPeriodic = domainParameters.getParameter < Vector3<  bool > >("isPeriodic");
    setup.oneBlockPerProcess = domainParameters.getParameter< bool >("oneBlockPerProcess");
    setup.boundX = domainParameters.getParameter< bool >("boundX");
    setup.boundY = domainParameters.getParameter< bool >("boundY");
    setup.boundZ = domainParameters.getParameter< bool >("boundZ");

    //----------------------simulation parameters----------------------------//
    auto simulationParameters = env.config()->getOneBlock("SimulationParameters");
    setup.timesteps = simulationParameters.getParameter< uint_t >("timesteps");
    setup.dt = simulationParameters.getParameter< real_t >("dt");
    setup.dx = simulationParameters.getParameter< real_t >("dx");

    setup.resolve_overlapping = simulationParameters.getParameter< bool >("resolve_overlapping");
    setup.resolve_maxsteps = simulationParameters.getParameter< uint_t >("resolve_maxsteps");
    setup.resolve_dt = simulationParameters.getParameter< real_t >("resolve_dt");

    setup.numPEsubsteps = simulationParameters.getParameter< uint_t >("numPEsubsteps");
    setup.numLBMsubsteps = simulationParameters.getParameter< uint_t >("numLBMsubsteps");
    setup.coupling_method = simulationParameters.getParameter< std::string >("coupling_method");
    if (setup.coupling_method == "MEM_BB")
        setup.memVariant = MEMVariant::BB;
    else if (setup.coupling_method == "MEM_CLI")
        setup.memVariant = MEMVariant::CLI;
    else
        WALBERLA_ABORT("Unsupported Coupling Method.");

    //----------------------particle properties----------------------------//
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

    //--------------------fluid properties------------------------------//
    auto fluidProperties = env.config()->getOneBlock("FluidProperties");
    setup.fluid_density = fluidProperties.getParameter< real_t >("fluid_density");
    setup.density_ratio = setup.particle_density / setup.fluid_density;
    setup.viscosity = fluidProperties.getParameter< real_t >("viscosity");

    setup.u_ref = real_c(0.01); // settling speed
    setup.x_ref = setup.particle_diameter_avg;
    setup.t_ref = setup.x_ref / setup.u_ref;

    setup.Re = real_c(1.0);
    setup.viscosity = setup.u_ref * setup.x_ref / setup.Re;
    setup.accg = setup.u_ref * setup.u_ref / ( (setup.density_ratio-real_c(1.0)) * setup.x_ref );
    setup.omega = lbm::collision_model::omegaFromViscosity(setup.viscosity);
    setup.tau = real_t(1.0) / setup.omega;

    real_t uIn = 0.00;
    Vector3< real_t > uInfty( real_t(0.0), uIn, real_t(0.0) );

    WALBERLA_LOG_INFO_ON_ROOT("Summary of LB fluid properties:\n"
                              << " - fluid density = " << setup.fluid_density << "\n"
                              << " - Reynolds number = " << setup.Re << "\n"
                              << " - characteristic velocity = " << setup.u_ref << "\n"
                              << " - characteristic length scale = " << setup.x_ref << "\n"
                              << " - characteristic time scale = " << setup.t_ref << "\n"
                              << " - kinetic viscosity = " << setup.viscosity << "\n"
                              << " - relaxation time = " << setup.tau << " omega = " << setup.omega << "\n"
                              << " - gravitational acceleration = " << setup.accg << "\n");

    //--------------------output parameters------------------------------//
    auto outputParameters = env.config()->getOneBlock("OutputParameters");
    setup.BaseFolder = outputParameters.getParameter< std::string >("BaseFolder");
    WALBERLA_ROOT_SECTION()
    {
        // create base directory if it does not yet exist
        filesystem::path basepath( setup.BaseFolder );
        if( !filesystem::exists( basepath ) )
            filesystem::create_directory( basepath );
    }

    setup.vtkBaseFolder = outputParameters.getParameter< std::string >("vtkBaseFolder");
    setup.vtkBaseFolder = setup.BaseFolder + '/' + setup.vtkBaseFolder;
    setup.vtkWriteFrequency = outputParameters.getParameter< uint_t >("vtkWriteFrequency");
    WALBERLA_ROOT_SECTION()
    {
        // create base directory if it does not yet exist
        filesystem::path vtkbasepath( setup.vtkBaseFolder );
        if( !filesystem::exists( vtkbasepath ) )
            filesystem::create_directory( vtkbasepath );
    }

    setup.logBaseFolder = outputParameters.getParameter< std::string >("logBaseFolder");
    setup.logBaseFolder = setup.BaseFolder + '/' + setup.logBaseFolder;
    setup.logInfoFrequency = outputParameters.getParameter< uint_t >("logInfoFrequency");
    WALBERLA_ROOT_SECTION()
    {
        // create base directory if it does not yet exist
        filesystem::path logbasepath( setup.logBaseFolder );
        if( !filesystem::exists( logbasepath ) )
            filesystem::create_directory( logbasepath );
    }

    //--------------------block decomposition------------------------------//
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

    // output parameters
    setup.printSetup();

    //////////////////////////////////
    /////   Simulation Setup  ////////
    //////////////////////////////////

    // simulation domain
    const auto domainSimulation = AABB(real_c(0.0), real_c(0.0), real_c(0.0),
    real_c(setup.domainSize[0]), real_c(setup.domainSize[1]), real_c(setup.domainSize[2]) );

    // create blockforect
    auto blocks = blockforest::createUniformBlockGrid(XBlocks, YBlocks, ZBlocks, 
                                                      XCells, YCells, ZCells,
                                                      setup.dx,
                                                      setup.oneBlockPerProcess,
                                                      xPeriodic, yPeriodic, zPeriodic);

    //--------------//
    //   PE setup   //
    //--------------//

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

    //--------------------//
    //   Initialization   //
    //--------------------//

    // generate initial spherical particles

    setup.numParticles = generateSingleSphere(blocks, globalBodyStorage, bodyStorageID, peMaterial, Vector3<real_t>(real_c(Lx)/real_c(2.0), real_c(Ly)/real_c(2.0), real_c(0.5) * setup.dx), setup.particle_diameter_1);

    // sync the created particles between processes
    PEsyncCall();

    WALBERLA_LOG_INFO_ON_ROOT(setup.numParticles << " spheres created.");

    //---------------------//
    // Add data to blocks  //
    //---------------------//

    // create the lattice model
    LatticeModel_T latticeModel = LatticeModel_T( lbm::collision_model::TRT::constructWithMagicNumber( setup.omega, magicNumberTRT) );

    // add pdf field
    BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (zyxf)", latticeModel, uInfty, setup.fluid_density, FieldGhostLayers, field::zyxf );

    // add flag field
    BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field" );

    // add body field
    BlockDataID bodyFieldID = field::addToStorage< BodyField_T >( blocks, "body field", nullptr, field::zyxf );

    // add boundary handling
    BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >( MyBoundaryHandling( flagFieldID, pdfFieldID, bodyFieldID, uInfty) );

    // initial mapping from rigid bodies into LBM grids
    if ( setup.memVariant == MEMVariant::BB)
        pe_coupling::mapMovingBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, MEM_BB_Flag, pe_coupling::selectRegularBodies );
    else if ( setup.memVariant == MEMVariant::CLI)
        pe_coupling::mapMovingBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, MEM_CLI_Flag, pe_coupling::selectRegularBodies );
    else
        WALBERLA_ABORT("Unsupported coupling method.");

    // mapping bound walls into LBM grids
    //pe_coupling::mapBodies< BoundaryHandling_T >( *blocks, boundaryHandlingID, bodyStorageID, *globalBodyStorage, bodyFieldID, NoSlip_Flag, pe_coupling::selectGlobalBodies );

    //---------------------------------//
    // setup LBM communication scheme  //
    //---------------------------------//
    blockforest::communication::UniformBufferedScheme< Stencil_T > pdfCommunicationScheme( blocks );
    pdfCommunicationScheme.addPackInfo( make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >( pdfFieldID ) );

    //---------------//
    // Output Setup  //
    //---------------//

    // write vtk output for the domain decomposition
    vtk::writeDomainDecomposition(blocks, "domain_decomposition", setup.vtkBaseFolder);

    // add vtk output for particles
    const auto bodyVTKOutput = make_shared< pe::SphereVtkOutput >( bodyStorageID, blocks->getBlockStorage() );
    const auto bodyVTK = vtk::createVTKOutput_PointData( bodyVTKOutput, "bodies", setup.vtkWriteFrequency, setup.vtkBaseFolder );

    bodyVTK->write();

    // add vtk output for flag field
    auto flagFieldVTK = vtk::createVTKOutput_BlockData( blocks, "flag_field", setup.vtkWriteFrequency, 0, false, setup.vtkBaseFolder );
    flagFieldVTK->addCellDataWriter( make_shared< field::VTKWriter< FlagField_T > >( flagFieldID, "FlagField" ) );

    flagFieldVTK->write();

    // add vtk output for fluid field
    auto pdfFieldVTK = vtk::createVTKOutput_BlockData( blocks, "fluid_field", setup.vtkWriteFrequency, 0, false, setup.vtkBaseFolder );
    //pdfFieldVTK->addBeforeFunction( pdfCommunicationScheme );

    field::FlagFieldCellFilter< FlagField_T > fluidFilter( flagFieldID );
    fluidFilter.addFlag( Fluid_Flag );
    pdfFieldVTK->addCellInclusionFilter( fluidFilter );

    pdfFieldVTK->addCellDataWriter( make_shared< lbm::VelocityVTKWriter< LatticeModel_T, float > >( pdfFieldID, "VelocityFromPDF") );
    pdfFieldVTK->addCellDataWriter( make_shared< lbm::DensityVTKWriter<LatticeModel_T, float> >( pdfFieldID, "DensityFromPDF") );

    pdfFieldVTK->write();

    ///////////////////////////
    /////   Time Loop  ////////
    ///////////////////////////

    shared_ptr<SweepTimeloop> timeloop = make_shared<SweepTimeloop>( blocks->getBlockStorage(), setup.timesteps );

    // update mapping from rigid bodies to LBM grids
    if( setup.memVariant == MEMVariant::BB )
        timeloop->add() << Sweep( pe_coupling::BodyMapping< LatticeModel_T, BoundaryHandling_T >( blocks, pdfFieldID, boundaryHandlingID, bodyStorageID, globalBodyStorage, bodyFieldID, MEM_BB_Flag, FormerMEM_Flag, pe_coupling::selectRegularBodies), "Body Mapping");
    else if ( setup.memVariant == MEMVariant::CLI )
        timeloop->add() << Sweep( pe_coupling::BodyMapping< LatticeModel_T, BoundaryHandling_T >( blocks, pdfFieldID, boundaryHandlingID, bodyStorageID, globalBodyStorage, bodyFieldID, MEM_CLI_Flag, FormerMEM_Flag, pe_coupling::selectRegularBodies), "Body Mapping");
    
    // reconstruct missing pdfs
    using ExtrapolationFinder_T = pe_coupling::SphereNormalExtrapolationDirectionFinder;
    ExtrapolationFinder_T extrapolationFinder( blocks, bodyFieldID );
    using Reconstructor_T = pe_coupling::ExtrapolationReconstructor<LatticeModel_T, BoundaryHandling_T, ExtrapolationFinder_T>;
    Reconstructor_T reconstructor( blocks, boundaryHandlingID, bodyFieldID, extrapolationFinder, false );
    timeloop->add() << Sweep( pe_coupling::PDFReconstruction< LatticeModel_T, BoundaryHandling_T, Reconstructor_T >( blocks, pdfFieldID, boundaryHandlingID, bodyStorageID, globalBodyStorage, bodyFieldID, reconstructor, FormerMEM_Flag, Fluid_Flag, pe_coupling::selectRegularBodies), "pdf Reconstruction");

    // --------------------------------------//
    // excecute LBM collision and streaming  //
    // --------------------------------------//
    for( uint_t lbmstep = 0; lbmstep < setup.numLBMsubsteps; ++lbmstep)
    {
        auto sweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( pdfFieldID, flagFieldID, Fluid_Flag );

        // collision sweep
        timeloop->add() << Sweep( lbm::makeCollideSweep( sweep ), "cell-wise LB collide sweep" );

        // LBM communication and boundary hanlding sweep
        timeloop->add() << BeforeFunction( pdfCommunicationScheme, "LBM communication" )
                        << Sweep( BoundaryHandling_T::getBlockSweep( boundaryHandlingID ), "Boundary Handling" );

        // streaming
        timeloop->add() << Sweep( lbm::makeStreamSweep( sweep ), "cell-wise LB stream sweep" );
    }

    if ( setup.numLBMsubsteps != uint_t(1) )
        // average hydrodynamic forces if more than one substep
        timeloop->addFuncAfterTimeStep( pe_coupling::ForceTorqueOnBodiesScaler( blocks, bodyStorageID, real_t(1.0) / real_c(setup.numLBMsubsteps) ), "Average forces over LBM substeps" );

    // prescribe particle angular velocity
    //Vector3<real_t> angular_vel( real_t(0.0), real_t(0.0), setup.u_ref / setup.x_ref );
    //timeloop->addFuncBeforeTimeStep( PrescribeAngularVel( blocks, bodyStorageID, angular_vel ), "Prescribe angular velocity of particles");

    // add external forces (gravity)
    Vector3<real_t> forces_ext( real_t(0.0), - setup.accg * (setup.density_ratio - real_t(1.0)) * setup.particle_volume_avg, real_t(0.0) );
    timeloop->addFuncAfterTimeStep( pe_coupling::ForceOnBodiesAdder( blocks, bodyStorageID, forces_ext ), "Add gravity and buoyancy forces");

    // confine to 2D plane - remove z-component force and velocity
    timeloop->addFuncAfterTimeStep( Enforce2D( blocks, bodyStorageID, uint_t(2) ), "Enforce 2D simulation");

    // log sphere properties
    timeloop->addFuncAfterTimeStep( SpherePropertiesLogger( timeloop, blocks, bodyStorageID, uint_c(1), setup.logBaseFolder, "log_single_sphere.txt"), "Sphere Properties Logger");

    // advance pe rigid body simulation
    timeloop->addFuncAfterTimeStep( pe_coupling::TimeStep( blocks, bodyStorageID, *cr, PEsyncCall, real_c(setup.numLBMsubsteps) * setup.dt, setup.numPEsubsteps ), "PE steps" );

    // VTK output
    timeloop->addFuncAfterTimeStep( vtk::writeFiles( bodyVTK ), "VTK (particles data)");
    timeloop->addFuncAfterTimeStep( vtk::writeFiles( flagFieldVTK ), "VTK (flag field data)");
    timeloop->addFuncAfterTimeStep( vtk::writeFiles( pdfFieldVTK ), "VTK (fluid field data)");

    // timer
    timeloop->addFuncAfterTimeStep( RemainingTimeLogger( timeloop->getNrOfTimeSteps(), real_t(30) ), "Remaining Time Logger");

    ////////////////////////
    // execute simulation //
    ////////////////////////

    WcTimingPool timeloopTiming;
    timeloop->run( timeloopTiming );
    timeloopTiming.logResultOnRoot();

    return EXIT_SUCCESS;
}

uint_t generateSingleSphere(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                            pe::MaterialID & material, const Vector3<real_t> pos, const real_t diameter)
{
    //generate a single sphere at specified location (x,y,z)

    WALBERLA_LOG_INFO_ON_ROOT("Creating a sphere with diameter = " << diameter << " at location " << pos);

    auto sphere = pe::createSphere( *globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, 0, pos, diameter * real_c(0.5), material);

    if (sphere != nullptr)
    {
        sphere->setLinearVel( real_t(0.0), -real_t(0.01) * real_t(0.01), real_t(0.0) );
    }

    return uint_t(1);
}

} //namespace spinners_suspension

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}