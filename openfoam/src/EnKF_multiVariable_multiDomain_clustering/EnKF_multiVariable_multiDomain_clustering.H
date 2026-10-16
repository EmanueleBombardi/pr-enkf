/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2018 OpenFOAM Foundation
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

Class

Description
    Specialisation for RAS of the generic EnKF_multiVariableBase base class.
    For more information, see Description of EnKF_multiVariableBase.H

See also
    Foam::EnKF_multiVariable_multiDomain_clustering

SourceFiles
    EnKF_multiVariable_multiDomain_clustering.C

\*---------------------------------------------------------------------------*/
#ifndef EnKF_multiVariable_multiDomain_clustering_H
#define EnKF_multiVariable_multiDomain_clustering_H

#include "fvCFD.H"
#include "EnKFControls.H"


class EnKF_multiVariable_multiDomain_clustering
{
    // Private data
    const Time& runTime_;
    const fvMesh& mesh_;
    label nEnsembles_;

    scalar obsUncertainty_;
    scalarList sigmaRel_;
    scalarList sigmaAbs_;

    scalarField priorUncertainty_;
    
    scalarField paramPerturbRange_; 
    scalar inflationFactor_; 
    bool parallelRun_;
    word  clusterPartition_;
    word clusterAccount_;
    fileName clusterEnvScript_;
    word  clusterTimeLimit_;
    label clusterTasksPerNode_;
    word  clusterSolverExecutable_;
    word  clusterMemPerCpu_;
    // Per-domain update interval. Read from EnKFDict as a label or a list;
    // expanded to length nSimulations_ by resolveUpdateIntervals() once
    // simulation directories are detected.
    scalarList updateInterval_;

    //- true: every cycle restarts from initialTime_ (restartMode firstTime)
    bool   restartFromFirstTime_;

    //- Initial time T0 of the members
    scalar initialTime_;
    label iterationCount_;
    label nCycles_;                  // number of EnKF analysis cycles
    label nClusters_;
    List<word> paramNames_;
    scalarField paramDefaults_;
    fileName parameterFile_;
    fileNameList memberFiles_;       
    label nBIACells_; 
    label nModelParams_;             // Numero di parametri
    label nObservables_;         // Dimension of observations (Nq)
    scalarSquareMatrix C_pp_;       // Parameter regularization matrix
    scalarField perturbedObsInBIA_;    // Store perturbed observations
    
    // Distance-based cluster localisation (optional).
    // The solver writes <clusterDistanceField_>0 ... <prefix>(nClusters-1),
    // each holding the per-cell distance d (not squared) to that cluster.
    bool                    featureSpaceLocalisation_;
    scalarList              featureSpaceLocalisationRadius_;  // L per domain
    word                    clusterDistanceField_;            // prefix
    wordList                clusterDistFieldNames_;           // generated names
    

    // ------------------------------------------------------------------
    // Observation field types, from EnKFDict's "sampledFieldTypes":
    // scalar/vector and, for vectors, the component assimilated.
    // Order is irrelevant — columns are located by name in the .xy header.
    // ------------------------------------------------------------------
    List<word>  sampledFieldNames_;
    List<bool>  sampledFieldIsVector_;
    List<label> sampledFieldComponent_;

    // Sample set name: every member writes
    // postProcessing/sampleDict/<time>/<featureSampleSetName_>.xy
    word        featureSampleSetName_;

    fileNameList memberCasePaths_;   // Paths to each member case directory
    bool casesSetup_;                // Flag to track if ensemble cases have been set up



    scalarField globalHx_;
    scalarField localPHt_;

    // Parametri del modello di turbolenza
    scalarField modelParams_;        // Valori attuali dei parametri
    scalarField expValues_;

    label nActiveObs_;
    labelList activeToGlobal_;
    labelList globalToActive_;


    // obsLocWeights_[simI][obsI][clusterC] — weight in [0,1]
    // built once per EnKF update by computeLocalisationWeights()
    List<List<scalarList>>  obsLocWeights_;

    scalarField paramMin_;
    scalarField paramMax_;

    List<label> simNProcs_;
    // ------------------------------------------------------------------
    // Member failure handling (restartMode firstTime only)
    // ------------------------------------------------------------------
    scalar            maxFailedFraction_;
    label             nValid_;          // members entering this analysis
    labelList         validMembers_;    // analysis column k -> member index
    labelList         failedMembers_;   // members excluded this cycle
    List<std::string> failReasons_;     // parallel to failedMembers_

    //- Mark members whose output is unusable in any sim; fills
    //  validMembers_/failedMembers_. Fatal above maxFailedFraction_
    //  or in latestTime mode.
    void detectFailedMembers();

    //- Keep only the valid columns of perturbedObsInBIA_ and paramPrior
    //  (stride nEnsembles_ -> nValid_)
    void compactStoredPerturbations();

    //- Non-fatal resolveTimeDirectory: false + reason if no usable time
    bool tryResolveTimeDirectory
    (
        const fileName& caseDir,
        const label     simI,
        fileName&       timeDir,
        std::string&    why
    );

    //- false + reason if the .xy file has non-finite values or != nObs rows
    bool xyFileIsFinite
    (
        const fileName& xyFile,
        const label     nObs,
        std::string&    why
    ) const;

    void readPerSimDecomposition();

    //- Expand a single-entry updateInterval_ to nSimulations_ entries, or
    //  fatal-error if its length is neither 1 nor nSimulations_.
    //  Must be called after detectSimulationDirectories().
    void resolveUpdateIntervals();
    
    //- Solver time at which domain simI's members stop in the current
    //  cycle: iterationCount_ (cycle index) x updateInterval_[simI].
    //  Single place for this arithmetic — used by createEnsemble,
    //  computeLocalisationWeights and modifyControlDict.
    scalar expectedTimestep(const label simI) const
    {
        // Checkpoint of the current cycle:
        //   latestTime: T0 + iterationCount*interval
        //   firstTime : T0 + interval (every cycle restarts from T0)
        const scalar n =
            restartFromFirstTime_ ? scalar(1) : scalar(iterationCount_);

        return initialTime_ + n*updateInterval_[simI];
    }

    scalar obsSigma(const label fieldI, const scalar y) const
    {
        return max(sigmaRel_[fieldI]*mag(y), sigmaAbs_[fieldI]);
    }

    //- Type (vector/scalar) and assimilated component of an observation
    //  field, from EnKFDict's sampledFieldTypes. FatalError if absent.
    void getFieldType
    (
        const word& fieldName,
        bool&       isVector,
        label&      component
    ) const;

    //- Parse the header line "# x y z <col> <col> ..." of a raw .xy
    //  sample file into column name -> absolute column index
    //  (x = 0, y = 1, z = 2, first sampled column = 3, ...).
    //  FatalError if the header is missing or does not start with x y z
    //  (sample sets must use 'axis xyz').
    HashTable<label> readXYHeader(const fileName& xyFile) const;

    //- Absolute column index of one sampled quantity in a .xy file.
    //  Scalar field f            -> column "f"
    //  Vector field f, comp. c   -> column "f_x" / "f_y" / "f_z"
    //  FatalError listing the available columns if not found.
    label xyColumnIndex
    (
        const fileName& xyFile,
        const word&     fieldName,
        const bool      isVector,
        const label     component
    ) const;

    // Changed to scalarField to support per-parameter values
    
    scalarField paramPrior;

    List<label> nObservationPointsPerSim_;  // per simulation

    //- Path (relative to each member case directory) to the file storing
    //  the model parameters to be perturbed/read/written by the EnKF.
    //  Sourced from EnKFDict's "parameterFile" (see EnKFControls).


    wordList observationFields_;     // from sampleDict: (U k)
    


    // Multi-domain support
    List<fileName> simulationDirs_;      // paths to sim1, sim2, sim3, ...
    label nSimulations_;                  // number of simulation domains
    List<label> domainCells_;            // cells per domain


    void modifyControlDict(const label memberID);

    void readStoredPerturbations();
    std::size_t findMatchingParenthesis(const std::string& text, std::size_t start);
    void updateMemberParameters(const scalarField& ensemble);



    // Private Member Functions
    void updateEnsemble(scalarField& ensemble, const scalarField& K);

    //- Cluster localisation weights obsLocWeights_[simI][obsI][c] =
    //  exp(-dbar_c^2 / 2L^2), dbar_c = ensemble mean of the solver-written
    //  distance to cluster c at observation obsI. Only when enabled.
    void computeLocalisationWeights();

    //- Read parameter values from the configured parameter file
    void readParameterValues(const fileName& memberDir, scalarField& paramValues);
    
    fileName resolveTimeDirectory(const fileName& caseDir, const label simI);
    label    readCellCountFromUField(const fileName& uFieldFile);

    //- Check that a freshly set-up member has every file the workflow needs.
    void checkMemberCase(const fileName& memberPath, const word& simName) const;

    //- Phi block reader — raw OpenFOAM field dump files (one field per
    //  file already, no grouping issue). Replaces the old
    //  readUxFromField/readScalarFromField pair.
    void readFieldComponent
    (
        const fileName& fieldFile,
        const label     nCells,
        scalarField&    ensemble,
        const label     rowBase,
        const label     totalCols,
        const label     ensI,
        const bool      isVector,
        const label     component
    );

    //- q-block reader: read absolute column 'column' of a raw .xy file
    //  (nObs data rows) directly into the ensemble matrix at
    //  rows [rowBase, rowBase + nObs), column ensI.
    void readXYColumn
    (
        const fileName& xyFile,
        const label     nObs,
        const label     column,
        scalarField&    ensemble,
        const label     rowBase,
        const label     totalCols,
        const label     ensI
    );

    //- Same, into a standalone scalarField of size nObs
    //  (used for the cluster distance fields).
    void readXYColumn
    (
        const fileName& xyFile,
        const label     nObs,
        const label     column,
        scalarField&    output
    );

    label parseScalarEntry(const std::string& content, const std::string& keyword);
    void cacheExperimentalValues();



public:
    EnKF_multiVariable_multiDomain_clustering
    (
        const Time& runTime,
        const fvMesh& mesh,
        const EnKFControls& controls
    );
    // Constructors
    //EnKF_multiVariable_multiDomain_clustering(const fvMesh& mesh, const volVectorField& U);

    // Destructor
    ~EnKF_multiVariable_multiDomain_clustering() = default;

    // Member Functions
    void update(scalarField& ensemble);

    void createEnsemble(scalarField& ensemble);
    
    void computeEnsembleMean(scalarField& ensemble, scalarField& mean);

    void computeEnsemblePerturbations(const scalarField& ensemble, const scalarField& ensembleMean, scalarField& ensemblePerturbation);

    void computeKalmanGain(scalarField& ensemble, const scalarField& ensembleMean, scalarField& ensemblePerturbation, scalarField& K);

    void detectSimulationDirectories();
    void validateEnKFConsistency();
    void perturbParametersForSimulation(const label simI);
    void copyParametersAcrossSimulations();
    void decomposeSimulationMembers(const label simI);
    void perturbMemberParameters(const label globalMemberIndex, const fileName& paramFile, Random& rnd);

    int getNEnsembles() const { return nEnsembles_; }
    const scalarList& getUpdateInterval() const { return updateInterval_; }
    scalar getUpdateInterval(const label simI) const { return updateInterval_[simI]; }
    label getNCycles() const { return nCycles_; }

    void generateAndStorePerturbations();

    // Add to public member functions in the class definition
    //- Set up ensemble case directories by copying base case structure
    void setupEnsembleCases();

    //- Check if ensemble cases have been set up
    bool casesAreSetup() const { return casesSetup_; }

    //- Get path to a specific ensemble member case
    fileName memberCasePath(const label memberID) const
    {
        return memberCasePaths_[memberID];
    }

    // Add to EnKF_multiVariable_multiDomain_clustering.H in public section:
    //- Generate a script to submit all ensemble simulations
    void generateMemberSubmissionScript();

};



#endif