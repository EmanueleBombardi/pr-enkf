/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    EnKF_Foam

Description
    Steady-state solver for incompressible, turbulent flow, using the SIMPLE
    algorithm.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "EnKF_multiVariable_multiDomain_clustering.H"
#include "EnKFControls.H"

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    EnKFControls enkfControls(runTime);

    EnKF_multiVariable_multiDomain_clustering enkf
    (
        runTime, mesh, enkfControls
    );

    const label currentIteration = enkfControls.iterationCount();

    if (currentIteration == 1)
    {
        enkf.setupEnsembleCases();
        enkf.generateAndStorePerturbations();
        enkf.generateMemberSubmissionScript();

        Info<< "Member simulations need to be run. Please execute the "
            << "generated script,\nwait for all simulations to complete, "
            << "then restart this application.\n" << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}


// ************************************************************************* //
