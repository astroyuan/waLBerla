waLBerla_link_files_to_builddir( *.cfg )
waLBerla_link_files_to_builddir( *.txt )

waLBerla_add_executable (
        NAME SpinnersSuspension
        FILES SpinnersSuspension.cpp
        DEPENDS blockforest core field geometry timeloop lbm pe pe_coupling vtk)

waLBerla_add_executable (
        NAME SingleSphereSedimentation
        FILES SingleSphereSedimentation.cpp
        DEPENDS blockforest core field geometry timeloop lbm pe pe_coupling vtk)

waLBerla_add_executable (
        NAME SpinnersSuspensionDry
        FILES SpinnersSuspensionDry.cpp
        DEPENDS blockforest core field geometry timeloop pe pe_coupling vtk)