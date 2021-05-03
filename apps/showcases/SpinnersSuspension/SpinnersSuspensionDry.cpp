#include "core/all.h"

#include "blockforest/Initialization.h"
#include "blockforest/communication/all.h"

#include "core/logging/all.h"

#include "pe/basic.h"
#include "pe/Types.h"
#include "pe/synchronization/SyncShadowOwners.h"
#include "pe/utility/DestroyBody.h"
#include "pe/vtk/SphereVtkOutput.h"

#include "pe_coupling/utility/all.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/VTKOutput.h"

#include <tuple>

namespace spinners_suspension
{
// using
using namespace walberla;
using walberla::uint_t;

// body types
using BodyTypeTuple = std::tuple< pe::Plane, pe::Sphere>;

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

    real_t u_ref; // characteristic velocity
    real_t x_ref; // characteristic length scale
    real_t t_ref; // characteristic time scale
    real_t Re; // Reynolds number
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

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_density);

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

        WALBERLA_CHECK(particle_diameter_1 > 0 && particle_diameter_2 > 0, "particle diamters should be positive.");
    }
};

/////////////////////////
// auxiliary functions
/////////////////////////
uint_t generateSingleSphere(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                            pe::MaterialID & material, const Vector3<real_t> pos, const real_t diameter);
uint_t generateRandomSpheresLayer(const shared_ptr< StructuredBlockForest > & blocks, const shared_ptr<pe::BodyStorage> & globalBodyStorage, const BlockDataID & bodyStorageID,
                                  const Setup & setup, const AABB & domainGeneration, pe::MaterialID & material, real_t layer_zpos);
void resolve_particle_overlaps(const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & bodyStorageID,
                               pe::cr::ICR & cr, const std::function<void (void)> & syncFunc, const Setup & setup);

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

class DummySweep
{
public:
    DummySweep() = default;

    void operator()(IBlock* const /*block*/)
    {}
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

    setup.u_ref = real_c(0.01); // settling speed
    setup.x_ref = setup.particle_diameter_avg;
    setup.t_ref = setup.x_ref / setup.u_ref;

    setup.accg = setup.u_ref * setup.u_ref / ( setup.particle_density * setup.x_ref );

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

    real_t layer_thickness = setup.dx;
    real_t layer_zpos = real_c(0.5) * setup.dx;
    real_t radius_max = setup.particle_diameter_max / real_c(2.0);
    const auto domainGeneration = AABB(domainSimulation.xMin() + radius_max, domainSimulation.yMin() + radius_max, domainSimulation.zMin(), 
                                       domainSimulation.xMax() - radius_max, domainSimulation.yMax() - radius_max, domainSimulation.zMin() + layer_thickness);

    // random generation of spherical particles
    setup.numParticles = generateRandomSpheresLayer(blocks, globalBodyStorage, bodyStorageID, setup, domainGeneration, peMaterial, layer_zpos);

    //setup.numParticles = generateSingleSphere(blocks, globalBodyStorage, bodyStorageID, peMaterial, Vector3<real_t>(real_c(Lx)/real_c(2.0), real_c(Ly)/real_c(2.0), real_c(0.5) * setup.dx), setup.particle_diameter_1);

    // sync the created particles between processes
    PEsyncCall();

    WALBERLA_LOG_INFO_ON_ROOT(setup.numParticles << " spheres created.");

    //------------------------------//
    // PE-only initial simulations  //
    //------------------------------//

    // resolve particle overlaps
    if(setup.resolve_overlapping)
    {
        WALBERLA_LOG_INFO_ON_ROOT("-----Resolving Particle Overlaps Start-----");
        resolve_particle_overlaps(blocks, bodyStorageID, *cr, PEsyncCall, setup);
        WALBERLA_LOG_INFO_ON_ROOT("-----Resolving Particle Overlaps End-----");
    }

    //---------------//
    // Output Setup  //
    //---------------//

    // write vtk output for the domain decomposition
    vtk::writeDomainDecomposition(blocks, "domain_decomposition", setup.vtkBaseFolder);

    // add vtk output for particles
    const auto bodyVTKOutput = make_shared< pe::SphereVtkOutput >( bodyStorageID, blocks->getBlockStorage() );
    const auto bodyVTK = vtk::createVTKOutput_PointData( bodyVTKOutput, "bodies", setup.vtkWriteFrequency, setup.vtkBaseFolder );

    ///////////////////////////
    /////   Time Loop  ////////
    ///////////////////////////

    shared_ptr<SweepTimeloop> timeloop = make_shared<SweepTimeloop>( blocks->getBlockStorage(), setup.timesteps );

    // add external forces (gravity)
    Vector3<real_t> forces_ext( real_t(0.0), - setup.accg * setup.particle_density * setup.particle_volume_avg, real_t(0.0) );
    timeloop->addFuncBeforeTimeStep( pe_coupling::ForceOnBodiesAdder( blocks, bodyStorageID, forces_ext ), "Add gravity and buoyancy forces");

    // prescribe particle angular velocity
    Vector3<real_t> angular_vel( real_t(0.0), real_t(0.0), real_t(2.0) * setup.u_ref / setup.x_ref );
    timeloop->addFuncBeforeTimeStep( PrescribeAngularVel( blocks, bodyStorageID, angular_vel ), "Prescribe angular velocity of particles");

    // advance pe simulation
    timeloop->add() << Sweep( DummySweep(), "Dummy Sweep" )
                    << AfterFunction( pe_coupling::TimeStep( blocks, bodyStorageID, *cr, PEsyncCall, setup.dt, uint_t(1) ), "PE steps");

    // confine to 2D plane - remove z-component force and velocity
    timeloop->addFuncAfterTimeStep( Enforce2D( blocks, bodyStorageID, uint_t(2) ), "Enforce 2D simulation");

    // VTK output
    timeloop->addFuncAfterTimeStep( vtk::writeFiles( bodyVTK ), "VTK (particles data)");

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
                               pe::cr::ICR & cr, const std::function<void (void)> & syncFunc, const Setup & setup)
{
    // collision properties evaluator
    auto OverlapEvaluator = make_shared<CollisionPropertiesEvaluator>(cr);

    const uint_t PEsteps = setup.resolve_maxsteps;
    const real_t dt = setup.resolve_dt;

    // temperary bounding plane
    //auto boundingPlane = pe::createPlane(*globalBodyStorage, 1, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainGeneration.zMax() + radius_max), peMaterial);

    for (uint_t pestep = uint_c(0); pestep < PEsteps; ++pestep)
    {
        cr.timestep(dt);
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

    // communication
    syncFunc();
}

} //namespace spinners_suspension

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}