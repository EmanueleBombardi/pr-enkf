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
    Specialisation for RAS of the generic EnKF_multiCaseBase base class.
    For more information, see Description of EnKF_multiCaseBase.H

See also
    Foam::EnKF_multiVariable_multiDomain_clustering

SourceFiles
    EnKF_multiVariable_multiDomain_clustering.C

\*---------------------------------------------------------------------------*/
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include "EnKF_multiVariable_multiDomain_clustering.H"
#include "Random.H"
#include "scalarMatrices.H"
#include <cmath>   
#include "Pstream.H"
#include "IPstream.H"
#include "OPstream.H"
#include <algorithm>
#include <numeric> 
#include "ops.H"
#include "List.H"
#include "scalarField.H"
#include <limits>
#include "volFields.H"
#include "labelList.H"
#include "fvCFD.H"
#include <vector>

#include <cstdlib>
#include <cctype>

namespace
{
    // Returns true and sets t if s is a numeric time directory name.
    // "0.0001", "1e-05", "12" -> true; "constant", "system", "processor0" -> false
    bool parseTimeName(const std::string& s, scalar& t)
    {
        if (s.empty()) return false;

        const char c = s[0];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-'))
        {
            return false;
        }

        char* end = nullptr;
        t = std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }


    // ------------------------------------------------------------------
    // Hard bounds. A value inside [lo, hi] is returned unchanged.
    // Outside, it is re-drawn just inside the violated bound:
    //   below lo  ->  lo + range*|lo|*absN   (= lo*(1 + range*|N|) for lo > 0)
    //   above hi  ->  hi - range*|hi|*absN   (= hi*(1 - range*|N|) for hi > 0)
    // then clipped to [lo, hi]. absN = |N(0,1)| is drawn by the caller
    // (so the parallel master/reduce logic stays where it is).
    // A zero bound uses the width (hi - lo) as scale instead of |bound|.
    // ------------------------------------------------------------------
    inline Foam::scalar boundParameter
    (
        Foam::scalar v,
        const Foam::scalar lo,
        const Foam::scalar hi,
        const Foam::scalar range,
        const Foam::scalar absN
    )
    {
        if (v >= lo && v <= hi)
        {
            return v;
        }

        const Foam::scalar width = hi - lo;

        if (v < lo)
        {
            const Foam::scalar s = (Foam::mag(lo) > Foam::SMALL) ? Foam::mag(lo) : width;
            v = lo + range*s*absN;
        }
        else
        {
            const Foam::scalar s = (Foam::mag(hi) > Foam::SMALL) ? Foam::mag(hi) : width;
            v = hi - range*s*absN;
        }

        return Foam::min(Foam::max(v, lo), hi);
    }
}
namespace
{
    // Same formatting as awk "%.10g" in submit_and_update.sh
    std::string fmtNum(const scalar v)
    {
        std::ostringstream oss;
        oss << std::setprecision(10) << v;
        return oss.str();
    }

    bool parseBoolWord(std::string v)
    {
        v.erase(0, v.find_first_not_of(" \t\r"));
        v.erase(v.find_last_not_of(" \t\r") + 1);
        for (char& c : v) c = std::tolower(static_cast<unsigned char>(c));
        return v == "yes" || v == "true" || v == "on" || v == "y"
            || v == "t" || v == "1";
    }

    // Finds the top-level (brace depth 0) single-line entry "key value;".
    //  - oldValue != nullptr : receives the current value (trimmed)
    //  - newValue != nullptr : replaces it, or appends the entry if absent
    // Returns true if the key was found. Entries inside functions{} or
    // FoamFile{} are never touched.
    bool topLevelEntry
    (
        std::string& content,
        const std::string& key,
        const std::string* newValue,
        std::string* oldValue
    )
    {
        const std::size_t npos = std::string::npos;

        auto padded = [&](const std::string& k)
        {
            std::string s = k;
            while (s.length() < 16) s += ' ';
            return s;
        };

        std::istringstream in(content);
        std::ostringstream out;
        std::string line;
        label depth = 0;
        bool found = false;

        while (std::getline(in, line))
        {
            const std::size_t cpos = line.find("//");
            const std::string code =
                (cpos == npos) ? line : line.substr(0, cpos);

            if (!found && depth == 0)
            {
                const std::size_t first = code.find_first_not_of(" \t\r");
                if (first != npos && code.compare(first, key.size(), key) == 0)
                {
                    const std::size_t after = first + key.size();
                    const std::size_t semi  = code.find(';', after);

                    if
                    (
                        after < code.size()
                     && std::isspace(static_cast<unsigned char>(code[after]))
                     && semi != npos
                    )
                    {
                        found = true;

                        if (oldValue)
                        {
                            std::string v = code.substr(after, semi - after);
                            v.erase(0, v.find_first_not_of(" \t\r"));
                            v.erase(v.find_last_not_of(" \t\r") + 1);
                            *oldValue = v;
                        }
                        if (newValue)
                        {
                            line = padded(key) + *newValue + ";";
                        }
                    }
                }
            }

            for (const char c : code)
            {
                if (c == '{') depth++;
                else if (c == '}') depth--;
            }
            out << line << '\n';
        }

        if (newValue)
        {
            if (!found) out << padded(key) << *newValue << ";\n";
            content = out.str();
        }
        return found;
    }
}

// Refactored EnKF_multiVariable_multiDomain_clustering class
EnKF_multiVariable_multiDomain_clustering::EnKF_multiVariable_multiDomain_clustering(
    const Time& runTime,
    const fvMesh& mesh,
    const EnKFControls& controls
)
:
    runTime_(runTime),
    mesh_(mesh),
    nEnsembles_(controls.nEnsembles()),  
    obsUncertainty_(controls.obsUncertainty()),
    priorUncertainty_(controls.priorUncertainty()),
    paramPerturbRange_(controls.paramPerturbRange()),
    inflationFactor_(controls.inflationFactor()),
    parallelRun_(controls.parallelRun()),
    clusterPartition_(controls.clusterPartition()),
    clusterAccount_(controls.clusterAccount()),
    clusterEnvScript_(controls.clusterEnvScript()),
    clusterTimeLimit_(controls.clusterTimeLimit()),
    clusterTasksPerNode_(controls.clusterTasksPerNode()),
    clusterSolverExecutable_(controls.clusterSolverExecutable()),
    clusterMemPerCpu_(controls.clusterMemPerCpu()),
    updateInterval_(controls.updateInterval()),
    iterationCount_(controls.iterationCount()),
    nCycles_(controls.nCycles()),
    nClusters_(controls.nClusters()),
    paramNames_(controls.paramNames()),
    paramDefaults_(controls.paramDefaults()),
    parameterFile_(controls.parameterFile()),
    memberFiles_(controls.memberFiles()),
    nBIACells_(0),
    nModelParams_(controls.nModelParams()),
    nObservables_(0),
    C_pp_(nModelParams_* nClusters_, 0.0),
    //paramRegWeight_(0.1),
    perturbedObsInBIA_(),   
    featureSpaceLocalisation_(controls.featureSpaceLocalisation()),
    featureSpaceLocalisationRadius_(controls.featureSpaceLocalisationRadius()),
    clusterDistanceField_(controls.clusterDistanceField()),
    clusterDistFieldNames_(),
    sampledFieldNames_(controls.sampledFieldNames()),
    sampledFieldIsVector_(controls.sampledFieldIsVector()),
    sampledFieldComponent_(controls.sampledFieldComponent()),
    featureSampleSetName_(controls.featureSampleSetName()),
    memberCasePaths_(nEnsembles_),
    casesSetup_(false)
    
{
    // Restart strategy: must be set before resolveUpdateIntervals(),
    // which calls expectedTimestep()
    restartFromFirstTime_ = (controls.restartMode() == "firstTime");
    initialTime_          = controls.initialTime();

    Info << "Restart mode: " << controls.restartMode()
         << "  (initial time T0 = " << initialTime_ << ")" << endl;

    // Read observation fields from sampleDict
    IOdictionary sampleDict
    (
        IOobject
        (
            "sampleDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    observationFields_ = wordList(sampleDict.lookup("fields"));

    forAll(observationFields_, fI)
    {
        if (findIndex(sampledFieldNames_, observationFields_[fI]) < 0)
        {
            FatalErrorInFunction
                << "Observation field '" << observationFields_[fI]
                << "' (from sampleDict) has no entry in EnKFDict's "
                << "'sampledFieldTypes'." << abort(FatalError);
        }
    }


    // ------------------------------------------------------------------
    // Observation error model per observation field, looked up by NAME
    // in sampledFieldTypes and stored in sampleDict 'fields' order.
    // ------------------------------------------------------------------
    sigmaRel_.setSize(observationFields_.size());
    sigmaAbs_.setSize(observationFields_.size());

    Info << "\nObservation error model: sigma = max(sigmaRel*|y|, sigmaAbs)"
         << endl;

    forAll(observationFields_, fI)
    {
        const label  idx = findIndex(sampledFieldNames_, observationFields_[fI]);
        const scalar sr  = controls.sampledFieldSigmaRel()[idx];
        const scalar sa  = controls.sampledFieldSigmaAbs()[idx];

        if (sr < 0 || sa <= 0)
        {
            FatalErrorInFunction
                << "Observation field '" << observationFields_[fI]
                << "' needs 'sigmaRel' (>= 0) and 'sigmaAbs' (> 0, physical "
                << "units) in its sampledFieldTypes entry." << nl
                << "Got sigmaRel=" << sr << "  sigmaAbs=" << sa
                << abort(FatalError);
        }

        sigmaRel_[fI] = sr;
        sigmaAbs_[fI] = sa;

        Info << "  [" << fI << "] " << observationFields_[fI]
             << ": sigmaRel=" << sr << "  sigmaAbs=" << sa << endl;
    }

    // Generate the per-cluster distance field names: <prefix>0 ... <prefix>(N-1)
    if (featureSpaceLocalisation_)
    {
        clusterDistFieldNames_.setSize(nClusters_);
        for (label c = 0; c < nClusters_; c++)
        {
            clusterDistFieldNames_[c] =
                word(clusterDistanceField_ + Foam::name(c));
        }
        Info << "Cluster distance fields: " << clusterDistFieldNames_ << endl;
    }


    Info << "Found " << observationFields_.size() << " observation fields: "
         << observationFields_ << endl;

    // Detect simulation directories first so we can read per-sim data
    detectSimulationDirectories();  
    resolveUpdateIntervals();
    readPerSimDecomposition();


    
    // Validate and expand per-domain localisation radius
   if (featureSpaceLocalisation_)
    {
        if (featureSpaceLocalisationRadius_.size() == 1)
        {
            scalar L = featureSpaceLocalisationRadius_[0];
            featureSpaceLocalisationRadius_.setSize(nSimulations_, L);
            Info << "featureSpaceLocalisationRadius: scalar " << L
                 << " replicated to " << nSimulations_ << " domains." << endl;
        }
        else if (featureSpaceLocalisationRadius_.size() != nSimulations_)
        {
            FatalErrorInFunction
                << "featureSpaceLocalisationRadius has "
                << featureSpaceLocalisationRadius_.size()
                << " entries but nSimulations=" << nSimulations_
                << ". Must be 1 or nSimulations."
                << abort(FatalError);
        }
        else
        {
            Info << "featureSpaceLocalisationRadius per domain: "
                 << featureSpaceLocalisationRadius_ << endl;
        }
    }

    if (simulationDirs_.empty())
    {
        FatalErrorInFunction
            << "No simulation directories found. Cannot initialize experimental data."
            << abort(FatalError);
    }

    // Read per-simulation observation point counts using raw std::ifstream
    // to avoid OpenFOAM registry issues with files outside the master case.
    // Different simulations may have different numbers of observation points,
    // but all fields within the same simulation must have the same count.
    nObservationPointsPerSim_.setSize(nSimulations_, 0);
    nObservables_ = 0;

    for (label simI = 0; simI < nSimulations_; simI++)
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName = simDir.name();

        Info << "Reading experimental data for simulation: " << simName << endl;

        label nPointsThisSim = -1;

        forAll(observationFields_, fieldI)
        {
            const word& fieldName = observationFields_[fieldI];

            fileName expDataFile =
                simDir/"constant"/("experimentalData_" + fieldName);

            if (!isFile(expDataFile))
            {
                FatalErrorInFunction
                    << "Experimental data file not found: " << expDataFile
                    << " for field: " << fieldName
                    << " in simulation: " << simName
                    << abort(FatalError);
            }

            // Read line by line with std::ifstream to avoid dictionary
            // parser hangs on files with blank lines or missing newlines
            std::ifstream rawFile(expDataFile.c_str());
            if (!rawFile.is_open())
            {
                FatalErrorInFunction
                    << "Cannot open file: " << expDataFile
                    << abort(FatalError);
            }

            label nPointsInFile = 0;
            bool insideList = false;
            std::string line;

            while (std::getline(rawFile, line))
            {
                // Trim leading whitespace
                std::string trimmed = line;
                trimmed.erase(0, trimmed.find_first_not_of(" \t\r"));

                if (trimmed.empty()) continue;

                if (!insideList)
                {
                    if (trimmed[0] == '(')
                    {
                        insideList = true;
                    }
                    continue;
                }

                // Closing parenthesis ends the data block
                if (trimmed[0] == ')' || trimmed == ");")
                {
                    break;
                }

                // Try to parse 4 scalar values (x y z value)
                std::istringstream iss(line);
                scalar x, y, z, val;
                if (iss >> x >> y >> z >> val)
                {
                    nPointsInFile++;
                }
            }

            rawFile.close();

            if (nPointsInFile == 0)
            {
                FatalErrorInFunction
                    << "No data points found in: " << expDataFile
                    << abort(FatalError);
            }

            if (fieldI == 0)
            {
                // First field sets the reference count for this simulation
                nPointsThisSim = nPointsInFile;
                Info << "  " << simName << ": " << nPointsInFile
                     << " observation points (from field " << fieldName << ")"
                     << endl;
            }
            else
            {
                // All fields within the same simulation must have the same count
                if (nPointsInFile != nPointsThisSim)
                {
                    FatalErrorInFunction
                        << "Inconsistent observation points within simulation "
                        << simName << nl
                        << "  Field " << observationFields_[0] << ": "
                        << nPointsThisSim << " points" << nl
                        << "  Field " << fieldName << ": "
                        << nPointsInFile << " points" << nl
                        << "All fields within the same simulation must have "
                        << "the same number of observation points."
                        << abort(FatalError);
                }
            }

            Info << "  Successfully read " << fieldName
                 << " for " << simName << endl;
        }

        nObservationPointsPerSim_[simI] = nPointsThisSim;
        nObservables_ += nPointsThisSim * observationFields_.size();
    }

    Info << "\nObservation point counts per simulation:" << endl;
    for (label simI = 0; simI < nSimulations_; simI++)
    {
        Info << "  " << simulationDirs_[simI].name() << ": "
             << nObservationPointsPerSim_[simI] << " points x "
             << observationFields_.size() << " fields = "
             << nObservationPointsPerSim_[simI] * observationFields_.size()
             << " observations" << endl;
    }
    Info << "Total observations across all simulations and fields: "
         << nObservables_ << endl;

    // Validate priorUncertainty_ size
    if (priorUncertainty_.size() != nModelParams_ * nClusters_)
    {
        Info << "Warning: priorUncertainty_ size mismatch ("
             << priorUncertainty_.size() << " vs " << nModelParams_
             << "). Resizing with default 0.5." << endl;
        priorUncertainty_.setSize(nModelParams_ * nClusters_);
        forAll(priorUncertainty_, i)
        {
            priorUncertainty_[i] = 0.5;
        }
    }

    // Validate paramPerturbRange_ size
    if (paramPerturbRange_.size() != nModelParams_ * nClusters_)
    {
        Info << "Warning: paramPerturbRange_ size mismatch ("
             << paramPerturbRange_.size() << " vs " << nModelParams_
             << "). Resizing with default 0.5." << endl;
        paramPerturbRange_.setSize(nModelParams_ * nClusters_);
        forAll(paramPerturbRange_, i)
        {
            paramPerturbRange_[i] = 0.5;
        }
    }

    // Hard bounds (required in EnKFDict, validated by EnKFControls)
    paramMin_ = controls.paramMin();
    paramMax_ = controls.paramMax();

    // Member failure handling: all members valid until detectFailedMembers()
    maxFailedFraction_ = controls.maxFailedFraction();
    nValid_            = nEnsembles_;
    validMembers_      = identity(nEnsembles_);

}

// ------------------------------------------------------------------
// Helper: getFieldType
//   Type and assimilated component of an observation field, from
//   EnKFDict's sampledFieldTypes.
// ------------------------------------------------------------------
void EnKF_multiVariable_multiDomain_clustering::getFieldType
(
    const word& fieldName,
    bool&       isVector,
    label&      component
) const
{
    const label idx = findIndex(sampledFieldNames_, fieldName);

    if (idx < 0)
    {
        FatalErrorInFunction
            << "Field '" << fieldName << "' has no entry in EnKFDict's "
            << "'sampledFieldTypes'." << abort(FatalError);
    }

    isVector  = sampledFieldIsVector_[idx];
    component = sampledFieldComponent_[idx];
}


// ------------------------------------------------------------------
// Helper: readXYHeader
//   Parses the first non-empty line of a raw .xy sample file,
//   "# x y z <col> <col> ...", into column name -> absolute index.
// ------------------------------------------------------------------
HashTable<label> EnKF_multiVariable_multiDomain_clustering::readXYHeader
(
    const fileName& xyFile
) const
{
    std::ifstream f(xyFile.c_str());
    if (!f.is_open())
    {
        FatalErrorInFunction
            << "Cannot open: " << xyFile << abort(FatalError);
    }

    std::string line;
    bool foundHeader = false;

    while (std::getline(f, line))
    {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;

        if (line[first] != '#')
        {
            FatalErrorInFunction
                << "No header line ('# x y z ...') before the data in "
                << xyFile << nl
                << "Sample sets must use 'setFormat raw;' and 'axis xyz;'."
                << abort(FatalError);
        }

        foundHeader = true;
        break;
    }

    if (!foundHeader)
    {
        FatalErrorInFunction
            << "Empty sample file: " << xyFile << abort(FatalError);
    }

    std::istringstream iss(line.substr(line.find('#') + 1));
    std::string tok;
    HashTable<label> cols;
    label idx = 0;

    while (iss >> tok)
    {
        if (!cols.insert(word(tok), idx))
        {
            FatalErrorInFunction
                << "Duplicate column '" << tok << "' in header of "
                << xyFile << abort(FatalError);
        }
        idx++;
    }

    if
    (
        idx < 4
     || !cols.found("x") || cols["x"] != 0
     || !cols.found("y") || cols["y"] != 1
     || !cols.found("z") || cols["z"] != 2
    )
    {
        FatalErrorInFunction
            << "Header of " << xyFile << " must start with 'x y z' "
            << "followed by at least one sampled column." << nl
            << "Header read: '" << line << "'" << nl
            << "Sample sets must use 'axis xyz;'."
            << abort(FatalError);
    }

    return cols;
}


// ------------------------------------------------------------------
// Helper: xyColumnIndex
//   Absolute column index of one sampled quantity in a .xy file.
//   Scalar f -> "f";  vector f, component c -> "f_x" / "f_y" / "f_z".
// ------------------------------------------------------------------
label EnKF_multiVariable_multiDomain_clustering::xyColumnIndex
(
    const fileName& xyFile,
    const word&     fieldName,
    const bool      isVector,
    const label     component
) const
{
    word colName = fieldName;

    if (isVector)
    {
        if (component < 0 || component > 2)
        {
            FatalErrorInFunction
                << "Invalid component " << component << " for vector field '"
                << fieldName << "' (must be 0, 1 or 2)." << abort(FatalError);
        }
        static const char* suffix[3] = {"_x", "_y", "_z"};
        colName = word(fieldName + suffix[component]);
    }

    HashTable<label> cols = readXYHeader(xyFile);

    if (!cols.found(colName))
    {
        // Rebuild the column list in file order for the error message
        wordList ordered(cols.size());
        forAllConstIter(HashTable<label>, cols, iter)
        {
            ordered[*iter] = iter.key();
        }

        FatalErrorInFunction
            << "Column '" << colName << "' not found in " << xyFile << nl
            << "Available columns: " << ordered << nl
            << "Check that '" << fieldName << "' is listed in this "
            << "simulation's system/sampleDict 'fields' entry"
            << (isVector ? "" : " and, for cluster distances, that the "
                                "solver writes it at every checkpoint")
            << "." << abort(FatalError);
    }

    return cols[colName];
}

void EnKF_multiVariable_multiDomain_clustering::resolveUpdateIntervals()
{
    if (updateInterval_.size() == 1)
    {
        const scalar n = updateInterval_[0];
        updateInterval_.setSize(nSimulations_, n);
        Info << "updateInterval: scalar " << n
             << " replicated to " << nSimulations_ << " domains." << endl;
    }
    else if (updateInterval_.size() != nSimulations_)
    {
        FatalErrorInFunction
            << "updateInterval has " << updateInterval_.size()
            << " entries but nSimulations=" << nSimulations_
            << ". Must be 1 or nSimulations." << nl
            << "Entries are matched to sim directories in numerical "
            << "order (sim1, sim2, ...)."
            << abort(FatalError);
    }

    Info << "updateInterval per domain:" << endl;
    forAll(simulationDirs_, simI)
    {
        Info << "  " << simulationDirs_[simI].name()
             << ": " << updateInterval_[simI]
             << " time/cycle  (current checkpoint = "
             << expectedTimestep(simI) << ")" << endl;
    }
    Info << "nCycles: " << nCycles_ << endl;
}

void EnKF_multiVariable_multiDomain_clustering::readPerSimDecomposition()
{
    simNProcs_.setSize(nSimulations_, 1);

    if (!parallelRun_) return;

    forAll(simulationDirs_, simI)
    {
        fileName decompDictPath =
            simulationDirs_[simI]/"system"/"decomposeParDict";

        if (!isFile(decompDictPath))
        {
            FatalErrorInFunction
                << "parallelRun is true but decomposeParDict not found: "
                << decompDictPath << abort(FatalError);
        }

        IOdictionary decompDict
        (
            IOobject
            (
                "decomposeParDict",
                simulationDirs_[simI]/"system",
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );

        simNProcs_[simI] = readLabel(decompDict.lookup("numberOfSubdomains"));

        Info << "  " << simulationDirs_[simI].name()
             << ": numberOfSubdomains = " << simNProcs_[simI] << endl;
    }
}

void EnKF_multiVariable_multiDomain_clustering::checkMemberCase
(
    const fileName& memberPath,
    const word&     simName
) const
{
    DynamicList<fileName> required;
    required.append("system/controlDict");
    required.append("system/sampleDict");
    required.append(parameterFile_);
    if (parallelRun_)
    {
        required.append("system/decomposeParDict");
    }

    forAll(required, i)
    {
        if (!isFile(memberPath/required[i]))
        {
            FatalErrorInFunction
                << "Member " << memberPath << " (" << simName << ") has no "
                << required[i] << "." << nl
                << "Add the file, or the folder that contains it, to "
                << "'memberFiles' in EnKFDict. Current list: " << memberFiles_
                << abort(FatalError);
        }
    }


    // The initial time T0 must have been copied into the member
    {
        bool foundT0 = false;
        const fileNameList dirs = readDir(memberPath, fileType::directory);

        forAll(dirs, i)
        {
            scalar t;
            if (parseTimeName(dirs[i], t) && mag(t - initialTime_) < 1e-9)
            {
                foundT0 = true;
                break;
            }
        }

        if (!foundT0)
        {
            FatalErrorInFunction
                << "Member " << memberPath << " (" << simName << ") has no "
                << "initial time directory " << initialTime_ << "." << nl
                << "Add it to 'memberFiles' in EnKFDict. Current list: "
                << memberFiles_ << abort(FatalError);
        }
    }
}

void EnKF_multiVariable_multiDomain_clustering::setupEnsembleCases()
{
    if (casesSetup_)
    {
        Info << "Ensemble cases have already been set up." << endl;
        return;
    }

    // Detect all simulation directories (sim1, sim2, ...)
    detectSimulationDirectories();

    if (simulationDirs_.empty())
    {
        FatalErrorInFunction
            << "No simulation directories found matching pattern 'sim[0-9]+'"
            << abort(FatalError);
    }

    Info << "Setting up ensemble cases for " << nSimulations_ << " simulations" << endl;
    Info << "Each simulation will have " << nEnsembles_ << " ensemble members" << endl;
    Info << "Observation fields: " << observationFields_ << endl;
    Info << "Total ensemble members: " << (nSimulations_ * nEnsembles_) << endl;

    // Validate that all simulations have consistent EnKF parameters
    validateEnKFConsistency();

    // Resize memberCasePaths_ to accommodate all simulations x members
    memberCasePaths_.setSize(nSimulations_ * nEnsembles_);

    for (label simI = 0; simI < nSimulations_; simI++)
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName = simDir.name();

        Info << "\n=== Setting up ensemble for simulation: " << simName << " ===" << endl;

        // Create the EnKFmembers directory within this simulation
        fileName enkfMembersDir = simDir/"EnKFmembers";
        if (!isDir(enkfMembersDir))
        {
            mkDir(enkfMembersDir);
            Info << "Created EnKFmembers directory: " << enkfMembersDir << endl;
        }

        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            fileName memberName = "member_" + Foam::name(ensI);
            fileName memberPath = enkfMembersDir/memberName;

            label globalMemberIndex = simI * nEnsembles_ + ensI;
            memberCasePaths_[globalMemberIndex] = memberPath;

            if (!isDir(memberPath))
            {
                mkDir(memberPath);
                Info << "Created directory: " << memberPath << endl;
            }

            // Copy the configured items (EnKFDict: memberFiles).
            // Items already present in the member are left untouched.
            forAll(memberFiles_, itemI)
            {
                const fileName& item = memberFiles_[itemI];
                const fileName  src  = simDir/item;
                const fileName  dest = memberPath/item;

                if (!exists(src))
                {
                    FatalErrorInFunction
                        << "memberFiles entry '" << item << "' not found in "
                        << simName << ": " << src << nl
                        << "Every memberFiles entry must exist in every sim."
                        << abort(FatalError);
                }

                if (exists(dest)) continue;

                mkDir(dest.path());   // parent dirs for nested entries

                const std::string cmd =
                    "cp -r \"" + std::string(src) + "\" \"" + std::string(dest) + "\"";

                if (system(cmd.c_str()) != 0)
                {
                    FatalErrorInFunction
                        << "Failed to copy " << src << " -> " << dest
                        << abort(FatalError);
                }

                if (ensI == 0)
                {
                    Info << "  Copied " << item << " (and likewise for every member)" << endl;
                }
            }

            checkMemberCase(memberPath, simName);
        }



        // Decompose parallel cases if needed
        if (parallelRun_)
        {
            decomposeSimulationMembers(simI);
        }

        // Modify controlDict for all members in this simulation
        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            label globalMemberIndex = simI * nEnsembles_ + ensI;
            modifyControlDict(globalMemberIndex);
        }
    }

    // Perturb parameters once, from the first simulation, then copy the
    // identical perturbations to all other simulations. Parameters are
    // shared across domains (same set calibrated jointly).
    Info << "Perturbing parameters for first simulation "
         << "(will be copied identically to all other domains)" << endl;
    perturbParametersForSimulation(0);

    if (nSimulations_ > 1)
    {
        copyParametersAcrossSimulations();
    }

    
    casesSetup_ = true;

    Info << "\n=== Multi-variable multi-domain ensemble setup complete ===" << endl;
    Info << "Total simulations: " << nSimulations_ << endl;
    Info << "Observation fields per simulation: " << observationFields_ << endl;
    Info << "Members per simulation: " << nEnsembles_ << endl;
    Info << "Total ensemble members: " << (nSimulations_ * nEnsembles_) << endl;
    Info << "Parameters shared across all domains: " << nModelParams_ << endl;
}

void EnKF_multiVariable_multiDomain_clustering::generateAndStorePerturbations()
{
    Info << "Generating perturbations for " << nSimulations_ << " simulations, "
         << observationFields_.size() << " fields" << endl;
    Info << "Observation points per simulation:" << endl;
    for (label simI = 0; simI < nSimulations_; simI++)
    {
        Info << "  " << simulationDirs_[simI].name() << ": "
             << nObservationPointsPerSim_[simI] << " points" << endl;
    }
    Info << "Total observations: " << nObservables_ << endl;

    perturbedObsInBIA_.setSize(nObservables_ * nEnsembles_);

    Random rand(clock::getTime() + Pstream::myProcNo());

    // ------------------------------------------------------------------
    // Generate shared parameter priors
    //
    // Storage layout: paramPrior[(paramI * nClusters_ + clusterI) * nEnsembles_ + ensI]
    //
    // Each (paramI, clusterI) pair now has its own default and perturbRange,
    // so each cluster gets an INDEPENDENT perturbation draw around its own
    // cluster-specific prior value.
    // ------------------------------------------------------------------
    Info << "Generating perturbed priors for " << nModelParams_
         << " shared parameters x " << nClusters_ << " clusters and "
         << nEnsembles_ << " ensemble members..." << endl;

    paramPrior.setSize(nModelParams_ * nClusters_ * nEnsembles_);

    // perturbedPriorsList[paramI][clusterI][ensI] — for writing to file
    // Outer: paramI, inner: clusterI, innermost: ensI
    List<List<List<scalar>>> perturbedPriorsList(nModelParams_);
    forAll(perturbedPriorsList, paramI)
    {
        perturbedPriorsList[paramI].setSize(nClusters_);
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            perturbedPriorsList[paramI][clusterI].setSize(nEnsembles_);
        }
    }

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        Info << "  Parameter " << paramNames_[paramI] << ":" << endl;

        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            // Cluster-specific default and perturbRange
            label flatParamIdx = paramI * nClusters_ + clusterI;
            scalar priorValue  = paramDefaults_[flatParamIdx];
            scalar pertRange   = paramPerturbRange_[flatParamIdx];
            scalar priorUnc    = priorUncertainty_[flatParamIdx];

            Info << "    cluster " << clusterI
                 << ": default=" << priorValue
                 << "  perturbRange=" << pertRange
                 << "  priorUncertainty=" << priorUnc << endl;

            for (label ensI = 0; ensI < nEnsembles_; ensI++)
            {
                // Independent draw per (paramI, clusterI, ensI)
                scalar perturbedPrior =
                    priorValue * (1.0 + rand.scalarNormal() * priorUnc);

                label flatIdx = flatParamIdx * nEnsembles_ + ensI;
                paramPrior[flatIdx] = perturbedPrior;
                perturbedPriorsList[paramI][clusterI][ensI] = perturbedPrior;

                if (ensI < 3)
                {
                    Info << "      member " << ensI
                         << ": " << perturbedPrior << endl;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Generate observation perturbations per simulation per field
    // (unchanged from original — no cluster indexing here)
    // ------------------------------------------------------------------
    label globalObsOffset = 0;

    for (label simI = 0; simI < nSimulations_; simI++)
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName = simDir.name();
        const label nPointsThisSim = nObservationPointsPerSim_[simI];

        Info << "\n=== Generating observation perturbations for simulation: "
             << simName << " (" << nPointsThisSim << " points) ===" << endl;

        List<List<List<scalar>>> perturbedObsLists(observationFields_.size());

        forAll(observationFields_, fieldI)
        {
            const word& fieldName = observationFields_[fieldI];

            fileName expDataFile =
                simDir/"constant"/("experimentalData_" + fieldName);

            if (!isFile(expDataFile))
            {
                FatalErrorInFunction
                    << "Experimental data file not found: " << expDataFile
                    << " for field " << fieldName
                    << " in simulation " << simName
                    << abort(FatalError);
            }

            IFstream is(expDataFile);
            if (!is.good())
            {
                FatalErrorInFunction
                    << "Cannot open file: " << expDataFile
                    << abort(FatalError);
            }

            dictionary fileDict(is);
            List<scalar> expPoints(fileDict.lookup("experimentalPoints"));
            label nPointsInFile = expPoints.size() / 4;

            if (nPointsInFile != nPointsThisSim)
            {
                FatalErrorInFunction
                    << "Observation point count mismatch in " << expDataFile << nl
                    << "  Expected (from constructor): " << nPointsThisSim << nl
                    << "  Found in file: " << nPointsInFile
                    << abort(FatalError);
            }

            perturbedObsLists[fieldI].setSize(nPointsThisSim);

            scalar sigMin = GREAT, sigMax = -GREAT, sigSum = 0.0;
            label  nAtFloor = 0;

            Info << "  Field " << fieldName << ": processing "
                 << nPointsThisSim << " points "
                 << "(globalObsOffset starts at " << globalObsOffset << ")" << endl;

            for (label pointI = 0; pointI < nPointsThisSim; pointI++)
            {
                perturbedObsLists[fieldI][pointI].setSize(nEnsembles_);

                scalar obsValue = expPoints[pointI * 4 + 3];

                const scalar uncertainty = obsSigma(fieldI, obsValue);

                sigMin  = min(sigMin, uncertainty);
                sigMax  = max(sigMax, uncertainty);
                sigSum += uncertainty;
                if (sigmaRel_[fieldI]*mag(obsValue) <= sigmaAbs_[fieldI])
                {
                    nAtFloor++;
                }

                for (label ensI = 0; ensI < nEnsembles_; ensI++)
                {
                    scalar perturbedObs =
                        obsValue + rand.scalarNormal() * uncertainty;

                    label globalObsIndex = globalObsOffset * nEnsembles_ + ensI;

                    if (globalObsIndex >= perturbedObsInBIA_.size())
                    {
                        FatalErrorInFunction
                            << "globalObsIndex " << globalObsIndex
                            << " out of bounds (size="
                            << perturbedObsInBIA_.size() << ")"
                            << " at simI=" << simI
                            << " fieldI=" << fieldI
                            << " pointI=" << pointI
                            << " ensI=" << ensI
                            << abort(FatalError);
                    }

                    perturbedObsInBIA_[globalObsIndex] = perturbedObs;
                    perturbedObsLists[fieldI][pointI][ensI] = perturbedObs;
                }

                globalObsOffset++;
            }

            Info << "  Finished field " << fieldName
                 << " (globalObsOffset now " << globalObsOffset << ")" << endl;
            
            Info << "  Field " << fieldName << " obs error sigma: min=" << sigMin
                << "  max=" << sigMax
                << "  avg=" << sigSum/nPointsThisSim
                << "  at floor (sigmaAbs): " << nAtFloor
                << "/" << nPointsThisSim << endl;
        }

        // ------------------------------------------------------------------
        // Write per-simulation perturbation file
        // ------------------------------------------------------------------
        fileName constantDir = simDir/"constant";
        if (!isDir(constantDir))
        {
            mkDir(constantDir);
        }

        fileName perturbFile = constantDir/"enkfPerturbations";
        OFstream os(perturbFile);

        if (!os.good())
        {
            FatalErrorInFunction
                << "Could not open perturbations file for writing: " << perturbFile
                << abort(FatalError);
        }

        os << "FoamFile" << nl
           << "{" << nl
           << "    version     2.0;" << nl
           << "    format      ascii;" << nl
           << "    class       dictionary;" << nl
           << "    object      enkfPerturbations;" << nl
           << "}" << nl << nl;

        // Per-field observation perturbations
        forAll(observationFields_, fieldI)
        {
            word dictKey = "perturbedObservations_" + observationFields_[fieldI];
            os << dictKey << nl
               << perturbedObsLists[fieldI] << ";" << nl << nl;
        }

        // ------------------------------------------------------------------
        // Write shared priors.
        //
        // NEW format: perturbedPriors is now List< List< List<scalar> > >
        //   outer index:  paramI
        //   middle index: clusterI
        //   inner index:  ensI
        //
        // readStoredPerturbations must be updated to match (step 4 will
        // handle that, but the key word "perturbedPriors" is preserved so
        // the old reader will produce a warning rather than a silent wrong
        // result if accidentally used with this new file).
        // ------------------------------------------------------------------
        os << "perturbedPriors" << nl
           << perturbedPriorsList << ";" << nl << nl;

        os << "observationFields   " << observationFields_ << ";" << nl;
        os << "nEnsembles          " << nEnsembles_ << ";" << nl;
        os << "nObservationPoints  " << nPointsThisSim << ";" << nl;
        os << "nFields             " << observationFields_.size() << ";" << nl;
        os << "nTotalObservations  "
           << (nPointsThisSim * observationFields_.size()) << ";" << nl;
        os << "nParameters         " << nModelParams_ << ";" << nl;
        os << "nClusters           " << nClusters_ << ";" << nl;
        os << "simulationIndex     " << simI << ";" << nl;
        os << "simulationName      " << simName << ";" << nl;
        os << "generatedTime       " << mesh_.time().timeName() << ";" << nl;

        os << "obsErrorModel       perField;" << nl;
        os << "obsSigmaRel         " << sigmaRel_ << ";" << nl;
        os << "obsSigmaAbs         " << sigmaAbs_ << ";" << nl;

        Info << "Written perturbations to: " << perturbFile << endl;
    }

    Info << "\n=== Perturbation generation complete ===" << endl;
    Info << "Simulations: " << nSimulations_ << endl;
    Info << "Fields per simulation: " << observationFields_ << endl;
    for (label simI = 0; simI < nSimulations_; simI++)
    {
        Info << "  " << simulationDirs_[simI].name() << ": "
             << nObservationPointsPerSim_[simI] << " points x "
             << observationFields_.size() << " fields = "
             << nObservationPointsPerSim_[simI] * observationFields_.size()
             << " observations" << endl;
    }
    Info << "Total observations: " << nObservables_ << endl;
    Info << "Shared parameters: " << nModelParams_ << endl;
    Info << "Clusters: " << nClusters_ << endl;
    Info << "Ensemble members: " << nEnsembles_ << endl;
}





void EnKF_multiVariable_multiDomain_clustering::decomposeSimulationMembers(const label simI)
{
    const fileName& simDir = simulationDirs_[simI];
    const word simName = simDir.name();
    
    Info << "Decomposing parallel cases for simulation: " << simName << endl;
    
    for (label ensI = 0; ensI < nEnsembles_; ensI++)
    {
        label globalMemberIndex = simI * nEnsembles_ + ensI;
        const fileName& memberPath = memberCasePaths_[globalMemberIndex];
        
        // Check if decomposeParDict exists in system directory
        fileName decomposeDict = memberPath/"system"/"decomposeParDict";
        if (!isFile(decomposeDict))
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::decomposeSimulationMembers")
                << "decomposeParDict not found at: " << decomposeDict
                << " for " << simName << " member " << ensI << ". Skipping decomposition." << endl;
            continue;
        }
        
        // Run decomposePar
        Info << "  Decomposing " << simName << " member " << ensI << "..." << endl;
        
        // Create and execute decomposePar command
        string decomposeCmd = "cd " + memberPath + " && decomposePar > log.decomposePar 2>&1";
        if (system(decomposeCmd.c_str()) != 0)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::decomposeSimulationMembers")
                << "Decomposition failed for " << simName << " member " << ensI << endl;
        }
        else
        {
            Info << "  " << simName << " member " << ensI << " successfully decomposed" << endl;
        }
    }
}

void EnKF_multiVariable_multiDomain_clustering::copyParametersAcrossSimulations()
{
    Info << "\nCopying parameters from first simulation to other simulations..." << endl;
    
    for (label simI = 1; simI < nSimulations_; simI++)  // Start from simI = 1
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName = simDir.name();
        
        Info << "Copying parameters to simulation: " << simName << endl;
        
        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            // Source paths (from first simulation)
            label sourceMemberIndex = 0 * nEnsembles_ + ensI;  // simI=0
            const fileName& sourceMemberPath = memberCasePaths_[sourceMemberIndex];
            
            // Destination paths (current simulation)
            label destMemberIndex = simI * nEnsembles_ + ensI;
            const fileName& destMemberPath = memberCasePaths_[destMemberIndex];
            
            // Copy turbulenceProperties
            fileName sourceParamFile = sourceMemberPath/parameterFile_;
            fileName destParamFile   = destMemberPath/parameterFile_;

            if (isFile(sourceParamFile))
            {
                string copyCommand = "cp " + sourceParamFile + " " + destParamFile;
                if (system(copyCommand.c_str()) != 0)
                {
                    FatalErrorInFunction
                        << "Failed to copy parameter file from " << sourceParamFile
                        << " to " << destParamFile
                        << abort(FatalError);
                }
                Info << "    Copied parameter file to member " << ensI << endl;
            }
            
        }
    }
    
    Info << "Parameter copying complete!" << endl;
}

void EnKF_multiVariable_multiDomain_clustering::detectSimulationDirectories()
{
    // Get the parent directory (multiSimulations)
    const fileName workingDir = mesh_.time().rootPath()/mesh_.time().caseName();
    const fileName multiSimsDir = workingDir;  
    
    Info << "Detecting simulation directories in: " << multiSimsDir << endl;
    
    // Alternative approach: try to read directory contents using different methods
    fileNameList allEntries;
    
    // Method 1: Try standard readDir
    try
    {
        allEntries = readDir(multiSimsDir);
        Info << "Method 1 (standard readDir): Found " << allEntries.size() << " entries" << endl;
    }
    catch (...)
    {
        Info << "Method 1 failed, trying alternative..." << endl;
        
        // Method 2: Use system command to list directories
        Info << "Trying system-based approach..." << endl;
        
        // Use system command to list directories
        string lsCommand = "ls -1 " + string(multiSimsDir) + " | grep '^sim[0-9]'";
        FILE* pipe = popen(lsCommand.c_str(), "r");
        if (pipe)
        {
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), pipe) != NULL)
            {
                string dirName = buffer;
                // Remove newline
                if (!dirName.empty() && dirName.back() == '\n')
                {
                    dirName.pop_back();
                }
                
                fileName fullPath = multiSimsDir/fileName(dirName);
                if (isDir(fullPath))
                {
                    allEntries.append(fileName(dirName));
                }
            }
            pclose(pipe);
            Info << "Method 2 (system command): Found " << allEntries.size() << " entries" << endl;
        }
        else
        {
            FatalErrorInFunction
                << "All directory reading methods failed for: " << multiSimsDir
                << abort(FatalError);
        }
    }
    
    List<fileName> tempSimDirs;
    
    forAll(allEntries, entryI)
    {
        const fileName& entryName = allEntries[entryI];
        const fileName fullPath = multiSimsDir/entryName;
        
        Info << "Checking entry: \"" << entryName << "\"" << endl;
        
        // Check if it's a directory first
        if (!isDir(fullPath))
        {
            Info << "  Skipping \"" << entryName << "\" (not a directory)" << endl;
            continue;
        }
        
        // Convert fileName to string for pattern matching
        string dirNameStr = string(entryName);
        
        // Check if it starts with "sim"
        if (dirNameStr.length() < 4 || dirNameStr.substr(0, 3) != "sim")
        {
            Info << "  Skipping \"" << entryName << "\" (doesn't start with 'sim')" << endl;
            continue;
        }
        
        // Extract number part and validate
        string numberPart = dirNameStr.substr(3);
        bool isValidNumber = true;
        
        if (numberPart.empty())
        {
            Info << "  Skipping \"" << entryName << "\" (no number after 'sim')" << endl;
            continue;
        }
        
        for (char c : numberPart)
        {
            if (!std::isdigit(c))
            {
                isValidNumber = false;
                break;
            }
        }
        
        if (isValidNumber)
        {
            tempSimDirs.append(fullPath);
            Info << "  Found valid simulation directory: \"" << entryName << "\"" << endl;
        }
        else
        {
            Info << "  Skipping \"" << entryName << "\" (invalid number format)" << endl;
        }
    }
    
    // If still no directories found, try a more direct approach
    if (tempSimDirs.empty())
    {
        Info << "No directories found with readDir, trying direct path checking..." << endl;
        
        // Try checking for sim directories directly
        for (label i = 1; i <= 100; i++)  // Check sim1 through sim100
        {
            fileName simDirName = "sim" + Foam::name(i);
            fileName simPath = multiSimsDir/simDirName;
            
            if (isDir(simPath))
            {
                tempSimDirs.append(simPath);
                Info << "  Found simulation directory by direct checking: " << simDirName << endl;
            }
        }
    }
    
    if (tempSimDirs.empty())
    {
        // Final attempt: use shell command to get directory listing
        Info << "Trying final shell-based directory detection..." << endl;
        
        string findCommand = "find " + string(multiSimsDir) + " -maxdepth 1 -type d -name 'sim[0-9]*' | sort";
        FILE* pipe = popen(findCommand.c_str(), "r");
        if (pipe)
        {
            char buffer[1024];
            while (fgets(buffer, sizeof(buffer), pipe) != NULL)
            {
                string fullPath = buffer;
                // Remove newline
                if (!fullPath.empty() && fullPath.back() == '\n')
                {
                    fullPath.pop_back();
                }
                
                if (isDir(fileName(fullPath)))
                {
                    tempSimDirs.append(fileName(fullPath));
                    Info << "  Found directory with find command: " << fullPath << endl;
                }
            }
            pclose(pipe);
        }
    }
    
    if (tempSimDirs.empty())
    {
        FatalErrorInFunction
            << "No simulation directories found matching pattern 'sim[0-9]+' in " << multiSimsDir << nl
            << "Found " << allEntries.size() << " total entries with readDir" << nl
            << "Troubleshooting steps:" << nl
            << "  1. Check directory permissions: ls -la " << multiSimsDir << nl
            << "  2. Verify directory names: ls -1 " << multiSimsDir << " | grep sim" << nl
            << "  3. Ensure directories are named: sim1, sim2, sim3, etc." << nl
            << "  4. Check for hidden characters in directory names"
            << abort(FatalError);
    }
    
    // Sort directories by number
    simulationDirs_.setSize(tempSimDirs.size());
    List<label> sortIndices(tempSimDirs.size());
    
    // Extract numbers and create sort indices
    forAll(tempSimDirs, i)
    {
        sortIndices[i] = i;
    }
    
    // Simple bubble sort by simulation number
    for (label i = 0; i < sortIndices.size() - 1; i++)
    {
        for (label j = 0; j < sortIndices.size() - 1 - i; j++)
        {
            const fileName& dir1 = tempSimDirs[sortIndices[j]];
            const fileName& dir2 = tempSimDirs[sortIndices[j + 1]];
            
            // Extract numbers for comparison
            string name1 = string(dir1.name());
            string name2 = string(dir2.name());
            
            label num1 = std::stoi(name1.substr(3));
            label num2 = std::stoi(name2.substr(3));
            
            if (num1 > num2)
            {
                std::swap(sortIndices[j], sortIndices[j + 1]);
            }
        }
    }
    
    // Copy sorted directories
    forAll(sortIndices, i)
    {
        simulationDirs_[i] = tempSimDirs[sortIndices[i]];
    }
    
    nSimulations_ = simulationDirs_.size();
    
    Info << "Successfully detected " << nSimulations_ << " simulation directories:" << endl;
    forAll(simulationDirs_, i)
    {
        Info << "  " << (i + 1) << ". " << simulationDirs_[i].name() << endl;
    }

}

void EnKF_multiVariable_multiDomain_clustering::validateEnKFConsistency()
{
    Info << "Validating EnKF parameter consistency across all simulations..." << endl;
    
    // Read EnKFDict from the first simulation
    const fileName firstEnKFDict = simulationDirs_[0]/"system"/"EnKFDict";
    
    if (!isFile(firstEnKFDict))
    {
        FatalErrorInFunction
            << "EnKFDict not found in first simulation: " << firstEnKFDict
            << abort(FatalError);
    }
    
    // Parameters that must be consistent across all simulations
    List<word> consistencyParams =
    {
        "nEnsembles",
        "nCycles",
        "nModelParams"
    };
    
    // Read reference values from first simulation
    IOdictionary firstDict
    (
        IOobject
        (
            "EnKFDict",
            simulationDirs_[0]/"system",
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    
    // Check consistency with other simulations
    for (label simI = 1; simI < nSimulations_; simI++)
    {
        const fileName currentEnKFDict = simulationDirs_[simI]/"system"/"EnKFDict";
        
        if (!isFile(currentEnKFDict))
        {
            FatalErrorInFunction
                << "EnKFDict not found in simulation " << simulationDirs_[simI].name()
                << ": " << currentEnKFDict
                << abort(FatalError);
        }
        
        IOdictionary currentDict
        (
            IOobject
            (
                "EnKFDict",
                simulationDirs_[simI]/"system",
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        
        // Check each parameter
        forAll(consistencyParams, paramI)
        {
            const word& paramName = consistencyParams[paramI];
            
            if (firstDict.found(paramName) && currentDict.found(paramName))
            {
                scalar firstValue = readScalar(firstDict.lookup(paramName));
                scalar currentValue = readScalar(currentDict.lookup(paramName));
                
                if (mag(firstValue - currentValue) > SMALL)
                {
                    FatalErrorInFunction
                        << "Parameter " << paramName << " mismatch between simulations!" << nl
                        << "  " << simulationDirs_[0].name() << ": " << firstValue << nl
                        << "  " << simulationDirs_[simI].name() << ": " << currentValue << nl
                        << "All simulations must have consistent EnKF parameters."
                        << abort(FatalError);
                }
            }
        }
    }
    
    Info << "EnKF parameter consistency validation passed." << endl;
}

void EnKF_multiVariable_multiDomain_clustering::perturbParametersForSimulation(const label simI)
{
    const fileName& simDir = simulationDirs_[simI];
    const word simName = simDir.name();
    
    Info << "Perturbing parameters for simulation: " << simName << endl;
    
    // Create a Random number generator (same seed for all simulations to ensure identical perturbations)
    Random rnd(clock::getTime());
    
    for (label ensI = 0; ensI < nEnsembles_; ensI++)
    {
        label globalMemberIndex = simI * nEnsembles_ + ensI;
        const fileName& memberPath = memberCasePaths_[globalMemberIndex];
        
        Info << "  Processing member " << ensI << " parameters..." << endl;
        
        // Path to turbulenceProperties file
        fileName paramFile = memberPath/parameterFile_;
        // Apply the same parameter perturbation logic as before
        // (keeping the existing parameter perturbation code structure)
        perturbMemberParameters(globalMemberIndex, paramFile, rnd);
    }
}


void EnKF_multiVariable_multiDomain_clustering::perturbMemberParameters
(
    const label globalMemberIndex,
    const fileName& paramFile,
    Random& rnd
)
{
    if (!isFile(paramFile))
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::perturbMemberParameters")
            << "Parameter file not found at: " << paramFile << endl;
        return;
    }

    std::ifstream inFile(paramFile);
    if (!inFile.is_open())
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::perturbMemberParameters")
            << "Could not open file: " << paramFile << endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inFile, line))
    {
        lines.push_back(line);
    }
    inFile.close();

    // memberParamValues layout: paramI * nClusters_ + clusterI
    // (kept for potential downstream use such as betaStar nut update)
    scalarField memberParamValues(nModelParams_ * nClusters_, 0.0);
    boolList paramFound(nModelParams_, false);

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        const std::string paramName = std::string(paramNames_[paramI]);

        // ------------------------------------------------------------------
        // Build per-cluster values.
        //
        // Each (paramI, clusterI) pair now has its own default and
        // perturbRange, so every cluster gets an independent draw around
        // its own cluster-specific prior value.
        // ------------------------------------------------------------------
        scalarField clusterValues(nClusters_);
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label flatIdx      = paramI * nClusters_ + clusterI;
            scalar defaultVal  = paramDefaults_[flatIdx];
            scalar perturbRange = paramPerturbRange_[flatIdx];

            // Prior draw, then hard bounds (EnKFDict min/max)
            const scalar draw =
                defaultVal * (1.0 + perturbRange * rnd.scalarNormal());

            clusterValues[clusterI] = boundParameter
            (
                draw,
                paramMin_[flatIdx],
                paramMax_[flatIdx],
                perturbRange,
                mag(rnd.scalarNormal())
            );

            if (clusterValues[clusterI] != draw)
            {
                Info << "    [BOUNDED, setup] " << paramName << "_c" << clusterI
                     << " member " << globalMemberIndex << ": " << draw
                     << " -> " << clusterValues[clusterI]
                     << "  [" << paramMin_[flatIdx] << ", "
                     << paramMax_[flatIdx] << "]" << endl;
            }

            memberParamValues[flatIdx] = clusterValues[clusterI];
        }

        std::ostringstream listStream;
        if (nClusters_ == 1)
        {
            // Global parameter: plain scalar, readable by standard models
            listStream << std::fixed << std::setprecision(8) << clusterValues[0];
        }
        else
        {
            listStream << "(";
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                if (clusterI > 0) listStream << " ";
                listStream << std::fixed << std::setprecision(8)
                           << clusterValues[clusterI];
            }
            listStream << ")";
        }
        std::string listString = listStream.str();

        for (std::string& l : lines)
        {
            // Skip comment lines
            std::string trimmed = l;
            trimmed.erase(0, trimmed.find_first_not_of(" \t"));
            if (trimmed.size() >= 2 && trimmed.substr(0, 2) == "//") continue;

            std::size_t pos = l.find(paramName);
            if (pos == std::string::npos) continue;

            // Whole-word check
            bool validStart = (pos == 0)
                || std::isspace(l[pos - 1])
                || l[pos - 1] == '{';

            std::size_t afterPos = pos + paramName.length();
            bool validEnd = (afterPos >= l.length())
                || std::isspace(l[afterPos])
                || l[afterPos] == '\t';

            if (!validStart || !validEnd) continue;

            // Must contain a semicolon
            if (l.find(';') == std::string::npos) continue;

            // Replace entire line with list format
            std::string newLine = "    " + paramName;
            while (newLine.length() < 20) newLine += " ";
            newLine += listString + ";";

            l = newLine;
            paramFound[paramI] = true;

            Info << "  Updated " << paramName << " = " << listString << endl;
            Info << "    per-cluster defaults: ";
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                label flatIdx = paramI * nClusters_ + clusterI;
                Info << "c" << clusterI << "=" << paramDefaults_[flatIdx] << "  ";
            }
            Info << endl;
            break;
        }

        if (!paramFound[paramI])
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::perturbMemberParameters")
                << "Could not find parameter '" << paramName
                << "' in: " << paramFile << endl;
        }
    }

    // Write modified lines back to file
    std::ofstream outFile(paramFile);
    if (outFile.is_open())
    {
        for (const std::string& l : lines)
        {
            outFile << l << "\n";
        }
        outFile.close();
        Info << "  Written updated parameter file for member "
             << globalMemberIndex << endl;
    }
    else
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::perturbMemberParameters")
            << "Could not write to: " << paramFile << endl;
    }
}

void EnKF_multiVariable_multiDomain_clustering::generateMemberSubmissionScript()
{
    label totalMembers = nSimulations_ * nEnsembles_;

    fileName masterDir = mesh_.time().rootPath()/mesh_.time().caseName();
    fileName workerScript = masterDir/"run_member.sh";

    Info << "Generating SLURM submission scripts in: " << masterDir << endl;

    // ---------------------------------------------------------------
    // Worker script (run_member.sh) — no quoting issues
    // ---------------------------------------------------------------
    std::ofstream workerFile(workerScript);

    workerFile << "#!/bin/bash\n\n"
            << "SIM_IDX=$1\n"
            << "MEMBER=$2\n"
            << "NPROCS=$3\n\n"
            << "# Initial time T0: never deleted (restartMode firstTime restarts from it)\n"
            << "INITIAL_TIME=" << std::setprecision(12) << initialTime_ << "\n"
            << "is_t0() { awk -v a=\"$1\" -v b=\"$INITIAL_TIME\" "
            << "'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<1e-9)}'; }\n\n"
            << "SIM_DIR=$(eval echo \"\\$SIMDIR_${SIM_IDX}\")\n"
            << "CASE_DIR=\"${SIM_DIR}/EnKFmembers/member_${MEMBER}\"\n\n"
            << "cd \"$CASE_DIR\" || { echo \"ERROR: cannot cd to $CASE_DIR\"; exit 1; }\n\n"
            << "echo \"Running sim${SIM_IDX}_member_${MEMBER} on $(hostname), NPROCS=$NPROCS, rank=${SLURM_PROCID:-0}\"\n\n"
            << "# --- Solver ---\n"
            << "if [ \"$NPROCS\" -gt 1 ]; then\n"
            << "    " << clusterSolverExecutable_ << " -parallel > log." << clusterSolverExecutable_ << " 2>&1\n"
            << "else\n"
            << "    " << clusterSolverExecutable_ << " > log." << clusterSolverExecutable_ << " 2>&1\n"
            << "fi\n\n"
            << "# --- Everything below runs once per member, not once per rank ---\n"
            << "if [ \"${SLURM_PROCID:-0}\" = \"0\" ]; then\n\n"
            << "if [ \"$NPROCS\" -gt 1 ]; then\n"
            << "    reconstructPar -latestTime > log.reconstructPar 2>&1\n"
            << "fi\n\n"
            << "LATEST_TIME=$(find . -maxdepth 1 -type d | \\\n"
            << "    grep -E \"^\\./[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?$\" | \\\n"
            << "    sed 's|^./||' | sort -g | tail -1)\n\n"
            << "echo \"Latest time: $LATEST_TIME\"\n\n"
            << "postProcess -func sampleDict -latestTime > log.postProcess 2>&1\n\n"
            << "SAMPLE_DIR=\"${CASE_DIR}/postProcessing/sampleDict/${LATEST_TIME}\"\n\n"
            << "# --- Keep only latest timestep (reconstructed case) ---\n"
            << "if [ -n \"$LATEST_TIME\" ] && [ \"$LATEST_TIME\" != \"0\" ]; then\n"
            << "    for dir in \"$CASE_DIR\"/*/; do\n"
            << "        base=$(basename \"$dir\")\n"
            << "        if echo \"$base\" | grep -qE '^[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?$'; then\n"
            << "            if [ \"$base\" != \"0\" ] && [ \"$base\" != \"$LATEST_TIME\" ] && ! is_t0 \"$base\"; then\n"
            << "                rm -rf \"$dir\"\n"
            << "            fi\n"
            << "        fi\n"
            << "    done\n\n"
            << "    # --- Keep only latest timestep (decomposed processor dirs) ---\n"
            << "    if [ \"$NPROCS\" -gt 1 ]; then\n"
            << "        for procDir in \"$CASE_DIR\"/processor*/; do\n"
            << "            for dir in \"$procDir\"*/; do\n"
            << "                base=$(basename \"$dir\")\n"
            << "                if echo \"$base\" | grep -qE '^[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?$'; then\n"
            << "                    if [ \"$base\" != \"0\" ] && [ \"$base\" != \"$LATEST_TIME\" ] && ! is_t0 \"$base\"; then\n"
            << "                        rm -rf \"$dir\"\n"
            << "                    fi\n"
            << "                fi\n"
            << "            done\n"
            << "        done\n"
            << "    fi\n"
            << "fi\n\n"
            << "echo \"Done: sim${SIM_IDX}_member_${MEMBER}\"\n"
            << "fi\n";

    workerFile.close();
    system(("chmod +x " + workerScript).c_str());

    // ---------------------------------------------------------------
    // Master script (run_all_members.sh)
    // ---------------------------------------------------------------
    
    // ---------------------------------------------------------------
    // Pack members onto nodes (first-fit, in launch order).
    // A member never spans two nodes.
    // ---------------------------------------------------------------
    const label coresPerNode = clusterTasksPerNode_;

    List<labelList>    memberNode(simulationDirs_.size());
    DynamicList<label> nodeLoad;

    forAll(simulationDirs_, simI)
    {
        const label nProcs = simNProcs_[simI];

        if (nProcs > coresPerNode)
        {
            FatalErrorInFunction
                << "Simulation " << simulationDirs_[simI].name()
                << " uses " << nProcs << " processes but cluster.tasksPerNode = "
                << coresPerNode << ". A member must fit on one node."
                << abort(FatalError);
        }

        memberNode[simI].setSize(nEnsembles_);

        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            label node = -1;
            forAll(nodeLoad, n)
            {
                if (nodeLoad[n] + nProcs <= coresPerNode)
                {
                    node = n;
                    break;
                }
            }
            if (node < 0)
            {
                node = nodeLoad.size();
                nodeLoad.append(0);
            }
            nodeLoad[node] += nProcs;
            memberNode[simI][ensI] = node;
        }
    }

    const label nNodes = nodeLoad.size();
    label maxLoad = 0;
    forAll(nodeLoad, n)
    {
        maxLoad = max(maxLoad, nodeLoad[n]);
    }

    Info << "SLURM packing: " << nNodes << " nodes, up to "
         << maxLoad << " tasks per node (node size " << coresPerNode << ")"
         << endl;

    const std::string memOpt =
        clusterMemPerCpu_.empty()
      ? std::string("")
      : std::string(" --mem-per-cpu=") + clusterMemPerCpu_;

    fileName masterScript = masterDir/"run_all_members.sh";
    std::ofstream scriptFile(masterScript);

    scriptFile << "#!/bin/bash\n"
            << "#SBATCH --job-name=" << masterDir.name() << "\n"
            << "#SBATCH --time=" << clusterTimeLimit_ << "\n"
            << "#SBATCH --nodes=" << nNodes << "\n"
            << "#SBATCH --ntasks-per-node=" << maxLoad << "\n"
            << (clusterMemPerCpu_.empty() ? std::string("")
                : "#SBATCH --mem-per-cpu=" + clusterMemPerCpu_ + "\n")
            << "#SBATCH --partition=" << clusterPartition_ << "\n"
            << "#SBATCH --output=" << masterDir << "/slurm-%j.out\n"
            << "#SBATCH --error=" << masterDir << "/slurm-%j.err\n"
            << "#SBATCH --account=" << clusterAccount_ << "\n\n"
            << "source " << clusterEnvScript_ << "\n\n"
            << "export OMP_NUM_THREADS=1\n"
            << "export SLURM_CPU_BIND=cores\n\n"
            << "NODES=($(scontrol show hostnames \"$SLURM_JOB_NODELIST\"))\n\n"
            << "BASE=\"" << masterDir << "\"\n\n";

    forAll(simulationDirs_, simI)
    {
        scriptFile << "export SIMDIR_" << simI
                << "=\"" << simulationDirs_[simI] << "\"\n";
    }

    scriptFile << "\nPIDS=()\n\n";

    forAll(simulationDirs_, simI)
    {
        label nProcs = simNProcs_[simI];
        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            scriptFile << "srun --exact --nodes=1 --ntasks=" << nProcs
                    << " --nodelist=\"${NODES[" << memberNode[simI][ensI] << "]}\""
                    << memOpt
                    << " bash " << workerScript
                    << " " << simI << " " << ensI << " " << nProcs
                    << " &\n"
                    << "PIDS+=($!)\n";
        }
    }

    scriptFile << "\nfor pid in \"${PIDS[@]}\"; do\n"
            << "    wait \"$pid\"\n"
            << "done\n\n";

    // completion markers — unchanged from before

    // Completion markers
    for (label simI = 0; simI < nSimulations_; simI++)
    {
        const fileName& simDir = simulationDirs_[simI];
        scriptFile << "echo \"" << simDir.name()
                   << " completed at $(date)\" > \""
                   << simDir << "/simulation_complete.txt\"\n";
    }

    scriptFile << "\n"
               << "COMPLETION_MARKER=\"" << masterDir << "/all_members_complete.txt\"\n"
               << "echo \"All member simulations completed at $(date)\" > \"$COMPLETION_MARKER\"\n"
               << "echo \"Total simulations: " << nSimulations_ << "\" >> \"$COMPLETION_MARKER\"\n"
               << "echo \"Total ensemble members: " << totalMembers << "\" >> \"$COMPLETION_MARKER\"\n"
               << "echo \"Observation fields: ";
    forAll(observationFields_, fI)
    {
        scriptFile << observationFields_[fI];
        if (fI < observationFields_.size() - 1) scriptFile << ", ";
    }
    scriptFile << "\" >> \"$COMPLETION_MARKER\"\n\n"
               << "echo \"Multi-variable multi-domain EnKF member phase completed.\"\n";

    scriptFile.close();
    system(("chmod +x " + masterScript).c_str());

    Info << "\n==============================================================\n"
         << "IMPORTANT: Two-phase process for EnKF\n"
         << "==============================================================\n"
         << "Phase 1: Setup (COMPLETED)\n"
         << "  - " << nSimulations_ << " simulation domains detected\n"
         << "  - " << totalMembers << " member directories created\n"
         << "  - Parameters perturbed and shared across all domains\n"
         << "  - Observation fields: ";
    forAll(observationFields_, fI)
    {
        Info << observationFields_[fI];
        if (fI < observationFields_.size() - 1) Info << ", ";
    }
    Info << "\n\n"
         << "Phase 2: Execution (MANUAL STEP REQUIRED)\n"
         << "  - Submit all simulations with:\n"
         << "    sbatch " << masterScript << "\n"
         << "  - Wait for all members to reach their first checkpoint:\n";
            forAll(simulationDirs_, simI)
            {
                Info << "      " << simulationDirs_[simI].name()
                    << ": time " << updateInterval_[simI] << "\n";
            }
            Info
         << "  - Restart this EnKF application for the analysis step\n"
         << "==============================================================\n" << endl;
}


void EnKF_multiVariable_multiDomain_clustering::modifyControlDict(const label memberID)
{
    const label simI = memberID / nEnsembles_;

    if (simI < 0 || simI >= updateInterval_.size())
    {
        FatalErrorInFunction
            << "memberID " << memberID << " maps to simI=" << simI
            << " but updateInterval_ has " << updateInterval_.size()
            << " entries. Was resolveUpdateIntervals() called?"
            << abort(FatalError);
    }

    const scalar interval = updateInterval_[simI];
    const fileName controlDictFile =
        memberCasePaths_[memberID]/"system"/"controlDict";

    if (!isFile(controlDictFile))
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::modifyControlDict")
            << "controlDict file not found at: " << controlDictFile << endl;
        return;
    }

    std::ifstream inFile(controlDictFile);
    std::string content
    (
        (std::istreambuf_iterator<char>(inFile)),
        std::istreambuf_iterator<char>()
    );
    inFile.close();

    // --- Read what is needed to choose the write control ----------------
    std::string val;

    bool adaptive = false;
    if (topLevelEntry(content, "adjustTimeStep", nullptr, &val))
    {
        adaptive = parseBoolWord(val);
    }

    scalar deltaT = -1;
    if (topLevelEntry(content, "deltaT", nullptr, &val))
    {
        deltaT = std::atof(val.c_str());
    }

    label timePrecision = 6;
    if (topLevelEntry(content, "timePrecision", nullptr, &val))
    {
        timePrecision = std::atoi(val.c_str());
    }

    // --- Consistency checks ---------------------------------------------
    if (adaptive)
    {
        if (deltaT > 0 && interval < deltaT)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::modifyControlDict")
                << "Member " << memberID << ": updateInterval " << interval
                << " < initial deltaT " << deltaT << endl;
        }
    }
    else if (deltaT > 0)
    {
        // Fixed step: the checkpoint must be a whole number of steps
        const scalar ratio = interval/deltaT;
        if (ratio < 0.5 || mag(ratio - std::round(ratio)) > 1e-6)
        {
            FatalErrorInFunction
                << "updateInterval " << interval << " (sim " << (simI + 1)
                << ") is not a multiple of deltaT " << deltaT
                << " (adjustTimeStep is off): interval/deltaT = " << ratio
                << abort(FatalError);
        }
    }

    // --- Apply ----------------------------------------------------------
    // First checkpoint T0 + interval (same in both restart modes);
    // write every interval; always start from the latest time (= T0 after
    // a firstTime reset, the previous checkpoint in latestTime mode)
    const std::string endTimeStr   = fmtNum(initialTime_ + interval);
    const std::string intervalStr  = fmtNum(interval);
    const std::string writeCtrl    = adaptive ? "adjustableRunTime" : "runTime";
    const std::string precisionStr = Foam::name(max(timePrecision, label(10)));
    const std::string startFromStr = "latestTime";

    topLevelEntry(content, "startFrom",     &startFromStr, nullptr);
    topLevelEntry(content, "endTime",       &endTimeStr,   nullptr);
    topLevelEntry(content, "writeControl",  &writeCtrl,    nullptr);
    topLevelEntry(content, "writeInterval", &intervalStr,  nullptr);
    topLevelEntry(content, "timePrecision", &precisionStr, nullptr);

    Info << "Member " << memberID << " (sim" << (simI + 1) << "): endTime="
         << endTimeStr << "  startFrom=latestTime  writeControl=" << writeCtrl
         << "  writeInterval=" << intervalStr
         << "  timePrecision=" << precisionStr
         << (adaptive ? "  (adaptive dt)" : "  (fixed dt)") << endl;

    std::ofstream outFile(controlDictFile);
    outFile << content;
    outFile.close();
}



void EnKF_multiVariable_multiDomain_clustering::update(scalarField& ensemble)
{
    Info << "Check member outputs" << endl;
    detectFailedMembers();

    createEnsemble(ensemble);

    Info << "Read stored perturbations" << endl;
    readStoredPerturbations();
    compactStoredPerturbations();

    // NEW — compute localisation weights before Kalman gain
    if (featureSpaceLocalisation_)
    {
        Info << "Computing feature-space localisation weights" << endl;
        computeLocalisationWeights();
    }

    Info << "Compute ensemble mean" << endl;
    scalarField ensembleLocalMean(nBIACells_, 0.0);
    computeEnsembleMean(ensemble, ensembleLocalMean);

    Info << "Compute ensemble perturbation" << endl;
    scalarField ensemblePerturbation(ensemble.size());
    computeEnsemblePerturbations(ensemble, ensembleLocalMean, ensemblePerturbation);

    Info << "Compute Kalman gain" << endl;
    scalarField localK(nBIACells_, 0.0);
    cacheExperimentalValues();
    computeKalmanGain(ensemble, ensembleLocalMean, ensemblePerturbation, localK);

    Info << "Update ensemble" << endl;
    updateEnsemble(ensemble, localK);

    Info << "Update members" << endl;
    updateMemberParameters(ensemble);
}


void EnKF_multiVariable_multiDomain_clustering::createEnsemble(scalarField& ensemble)
{
    Info << "Creating multi-variable multi-domain ensemble matrix..." << endl;
    Info << "Simulations: " << nSimulations_ << endl;
    Info << "Observation fields: " << observationFields_ << endl;
    Info << "Ensemble members: " << nEnsembles_ << endl;

    // ------------------------------------------------------------------
    // Step 0: Detect simulation directories if not already done
    // ------------------------------------------------------------------
    if (simulationDirs_.empty())
    {
        detectSimulationDirectories();
    }

    // Repopulate memberCasePaths_ from disk — needed when running as a
    // separate solver invocation where setupEnsembleCases was not called.
    memberCasePaths_.setSize(nSimulations_ * nEnsembles_);
    forAll(simulationDirs_, simI)
    {
        fileName enkfMembersDir = simulationDirs_[simI]/"EnKFmembers";
        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            label globalMemberIndex = simI * nEnsembles_ + ensI;
            memberCasePaths_[globalMemberIndex] =
                enkfMembersDir/("member_" + Foam::name(ensI));
        }
    }

    Info << "Repopulated memberCasePaths_ for "
        << nSimulations_ << " sims x " << nEnsembles_
        << " members = " << memberCasePaths_.size() << " entries." << endl;
    

    const label nFields = observationFields_.size();
    const label totalCols = nValid_;


    // ------------------------------------------------------------------
    // Step 2: First pass — determine domainCells_[simI] for every sim
    //         by reading the cell count from member_0's U field file.
    //         Also resolves/renames time directories to expectedTimestep.
    // ------------------------------------------------------------------
    domainCells_.setSize(nSimulations_, 0);

    forAll(simulationDirs_, simI)
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName = simDir.name();

        // First VALID member (member_0 may have failed)
        fileName member0Dir =
            simDir/"EnKFmembers"/("member_" + Foam::name(validMembers_[0]));

        if (!isDir(member0Dir))
        {
            FatalErrorInFunction
                << "member_0 directory not found: " << member0Dir
                << abort(FatalError);
        }

        const scalar tStep = expectedTimestep(simI);
        Info << "  " << simName << ": expected timestep " << tStep
             << " (interval " << updateInterval_[simI] << ")" << endl;

        fileName timeDir = resolveTimeDirectory(member0Dir, simI);

        // Read cell count from U field
        fileName uFieldFile = member0Dir/timeDir/"U";
        if (!isFile(uFieldFile))
        {
            FatalErrorInFunction
                << "U field file not found: " << uFieldFile
                << abort(FatalError);
        }

        label simCellCount = readCellCountFromUField(uFieldFile);

        if (simCellCount <= 0)
        {
            FatalErrorInFunction
                << "Could not determine cell count for simulation: " << simName
                << abort(FatalError);
        }

        domainCells_[simI] = simCellCount;

        Info << "  " << simName << ": " << simCellCount << " cells, "
             << nObservationPointsPerSim_[simI] << " obs points/field" << endl;
    }

    // ------------------------------------------------------------------
    // Step 3: Compute block offsets
    //
    // Layout (Option B — domains-then-fields):
    //
    //   phi block:
    //     for simI in [0, nSimulations_):
    //       for fieldI in [0, nFields):
    //         domainCells_[simI] rows
    //
    //   alpha block:
    //     nModelParams_ rows  (shared, sim-independent)
    //
    //   q block:
    //     for simI in [0, nSimulations_):
    //       for fieldI in [0, nFields):
    //         nObservationPointsPerSim_[simI] rows
    // ------------------------------------------------------------------
    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    label totalQRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalQRows += nObservationPointsPerSim_[simI] * nFields;
    }

    label alphaRows      = nModelParams_ * nClusters_;
    label phiRowOffset   = 0;
    label alphaRowOffset = totalPhiRows;
    label qRowOffset     = totalPhiRows + alphaRows;
    label totalRows      = qRowOffset + totalQRows;
    label matrixSize     = totalRows * totalCols;

    // Update class members
    nBIACells_ = totalPhiRows;   // Total phi rows (all sims, all fields)

    Info << "\nEnsemble matrix dimensions:" << endl;
    Info << "  phi block   (state):  " << totalPhiRows   << " rows" << endl;
    Info << "  alpha block (params): " << alphaRows
     << " rows (" << nModelParams_ << " params x "
     << nClusters_ << " clusters)" << endl;
    Info << "  q block     (obs):    " << totalQRows     << " rows" << endl;
    Info << "  Total rows:           " << totalRows      << endl;
    Info << "  Total columns:        " << totalCols      << endl;
    Info << "  Matrix size:          " << matrixSize     << " elements" << endl;

    Info << "\nBlock row offsets:" << endl;
    Info << "  phi   starts at row: " << phiRowOffset   << endl;
    Info << "  alpha starts at row: " << alphaRowOffset << endl;
    Info << "  q     starts at row: " << qRowOffset     << endl;

    // Resize and zero ensemble
    ensemble.setSize(matrixSize);
    ensemble = 0.0;

    // ------------------------------------------------------------------
    // Step 4: Second pass — fill ensemble matrix column by column (ensI)
    //         and, within each column, domain by domain (simI).
    // ------------------------------------------------------------------
    for (label ensI = 0; ensI < nValid_; ensI++)   // ensI = analysis column
    {
        const label memberI = validMembers_[ensI];
        Info << "\nProcessing ensemble member " << memberI
            << " (column " << ensI << ")..." << endl;

        // Accumulate phi/q row offsets per sim as we iterate
        label phiSimOffset = phiRowOffset;
        label qSimOffset   = qRowOffset;

        forAll(simulationDirs_, simI)
        {
            const fileName& simDir = simulationDirs_[simI];
            const word simName = simDir.name();
            const label nCells    = domainCells_[simI];
            const label nObsSim   = nObservationPointsPerSim_[simI];

            fileName memberDir =
                simDir/"EnKFmembers"/("member_" + Foam::name(memberI));

            if (!isDir(memberDir))
            {
                FatalErrorInFunction
                    << "Member directory not found: " << memberDir
                    << abort(FatalError);
            }

            // Resolve time directory for this member
            fileName timeDir =
                resolveTimeDirectory(memberDir, simI);

            // ------------------------------------------------------
            // 4a. phi block — one sub-block per field
            //     row = phiSimOffset + fieldI*nCells + cellI
            // ------------------------------------------------------
            forAll(observationFields_, fieldI)
            {
                const word& fieldName = observationFields_[fieldI];

                fileName fieldFile = memberDir/timeDir/fieldName;
                if (!isFile(fieldFile))
                {
                    FatalErrorInFunction
                        << fieldName << " field file not found: " << fieldFile
                        << " for " << simName << " member " << memberI
                        << abort(FatalError);
                }

                label rowBase = phiSimOffset + fieldI * nCells;

                bool isVec; label comp;
                getFieldType(fieldName, isVec, comp);

                readFieldComponent(fieldFile, nCells, ensemble,
                                    rowBase, totalCols, ensI, isVec, comp);
            }

            // Advance phi offset past all fields of this sim
            phiSimOffset += nCells * nFields;

            // ------------------------------------------------------
            // 4b. alpha block — only from sim0 (parameters shared)
            //     row = alphaRowOffset + paramI
            // ------------------------------------------------------
            if (simI == 0)
            {
                fileName paramFile = memberDir/parameterFile_;

                if (!isFile(paramFile))
                {
                    FatalErrorInFunction
                        << "turbulenceProperties not found: " << paramFile
                        << abort(FatalError);
                }

                scalarField paramValues(nModelParams_, 0.0);
                readParameterValues(memberDir, paramValues);

                for (label paramI = 0; paramI < nModelParams_; paramI++)
                {
                    for (label clusterI = 0; clusterI < nClusters_; clusterI++)
                    {
                        label row = alphaRowOffset + paramI * nClusters_ + clusterI;
                        label matrixIndex = row * totalCols + ensI;
                        ensemble[matrixIndex] =
                            paramValues[paramI * nClusters_ + clusterI];
                    }
                }

                Info << "  " << simName << ": stored " << nModelParams_
                    << " params x " << nClusters_ << " clusters = "
                    << nModelParams_ * nClusters_ << " alpha rows" << endl;
            }

            // ------------------------------------------------------
            // 4c. q block — sampled observations, one sub-block per field
            //     row = qSimOffset + fieldI*nObsSim + obsI
            //     file: postProcessing/sampleDict/<time>/<setName>.xy
            //     columns located by name from the file header
            // ------------------------------------------------------
            fileName postProcDir = memberDir/"postProcessing"/"sampleDict";
            if (!isDir(postProcDir))
            {
                FatalErrorInFunction
                    << "PostProcessing directory not found: " << postProcDir
                    << " for " << simName << " member " << memberI
                    << abort(FatalError);
            }

            // Resolve postprocessing time directory
            fileName postTimeDir =
                resolveTimeDirectory(postProcDir, simI);

            const fileName obsFile =
                postProcDir/postTimeDir/fileName(featureSampleSetName_ + ".xy");

            if (!isFile(obsFile))
            {
                FatalErrorInFunction 
                    << "Sample file not found: " << obsFile
                    << " for " << simName << " member " << memberI
                    << abort(FatalError);
            }

            forAll(observationFields_, fieldI)
            {
                const word& fieldName = observationFields_[fieldI];

                bool isVec; label comp;
                getFieldType(fieldName, isVec, comp);

                const label col =
                    xyColumnIndex(obsFile, fieldName, isVec, comp);

                const label rowBase = qSimOffset + fieldI * nObsSim;

                readXYColumn(obsFile, nObsSim, col,
                             ensemble, rowBase, totalCols, ensI);
            }

            // Advance q offset past all fields of this sim
            qSimOffset += nObsSim * nFields;

        } // end simI loop

    } // end ensI loop

    // ------------------------------------------------------------------
    // Step 5: Verification and diagnostics
    // ------------------------------------------------------------------
    Info << "\nEnsemble matrix creation completed." << endl;
    Info << "Matrix size: " << totalRows << " x " << totalCols
         << " = " << matrixSize << " elements" << endl;

    if (ensemble.size() != matrixSize)
    {
        FatalErrorInFunction
            << "Matrix size mismatch. Expected: " << matrixSize
            << ", Got: " << ensemble.size()
            << abort(FatalError);
    }

    // Sample values for verification
    Info << "\nSample ensemble matrix values:" << endl;
    Info << "  phi[0,0]   = " << ensemble[phiRowOffset   * totalCols + 0] << endl;
    Info << "  alpha[0,0] = " << ensemble[alphaRowOffset  * totalCols + 0] << endl;
    Info << "  q[0,0]     = " << ensemble[qRowOffset      * totalCols + 0] << endl;


}

// ------------------------------------------------------------------
// readXYColumn (standalone output)
//   Reads absolute column 'column' of every data row of a raw .xy
//   file. Exactly nObs data rows are required; blank lines and '#'
//   comment lines are skipped. Fatal on short rows, non-numeric
//   values (e.g. nan) or a row-count mismatch.
// ------------------------------------------------------------------
void EnKF_multiVariable_multiDomain_clustering::readXYColumn
(
    const fileName& xyFile,
    const label     nObs,
    const label     column,
    scalarField&    output
)
{
    output.setSize(nObs);
    output = 0.0;

    std::ifstream f(xyFile.c_str());
    if (!f.is_open())
    {
        FatalErrorInFunction << "Cannot open: " << xyFile << abort(FatalError);
    }

    std::string line;
    label row    = 0;
    label lineNo = 0;

    while (std::getline(f, line))
    {
        lineNo++;

        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;   // blank line
        if (line[first] == '#') continue;           // header / comment

        std::istringstream iss(line);
        scalar v;
        label  c   = 0;
        bool   got = false;

        while (iss >> v)
        {
            if (c == column) { got = true; break; }
            c++;
        }

        if (!got)
        {
            FatalErrorInFunction
                << "Line " << lineNo << " of " << xyFile
                << " has no numeric value in column " << column
                << " (short row or non-numeric entry such as nan)." << nl
                << "Line: '" << line << "'" << abort(FatalError);
        }

        if (row >= nObs)
        {
            FatalErrorInFunction
                << xyFile << " has more than the expected " << nObs
                << " data rows. The sampleDict points and the "
                << "experimentalData files disagree." << abort(FatalError);
        }

        output[row++] = v;
    }

    if (row != nObs)
    {
        FatalErrorInFunction
            << "Read " << row << " data rows from " << xyFile
            << " but expected " << nObs << "." << abort(FatalError);
    }
}


// ------------------------------------------------------------------
// readXYColumn (into the ensemble matrix)
// ------------------------------------------------------------------
void EnKF_multiVariable_multiDomain_clustering::readXYColumn
(
    const fileName& xyFile,
    const label     nObs,
    const label     column,
    scalarField&    ensemble,
    const label     rowBase,
    const label     totalCols,
    const label     ensI
)
{
    scalarField vals;
    readXYColumn(xyFile, nObs, column, vals);

    forAll(vals, i)
    {
        ensemble[(rowBase + i) * totalCols + ensI] = vals[i];
    }
}

bool EnKF_multiVariable_multiDomain_clustering::tryResolveTimeDirectory
(
    const fileName& caseDir,
    const label     simI,
    fileName&       timeDir,
    std::string&    why
)
{
    const scalar interval = updateInterval_[simI];
    const scalar tExp     = expectedTimestep(simI);
    const scalar tPrev    = tExp - interval;
    const scalar tol      = 1e-3*interval;

    const fileNameList entries = readDir(caseDir, fileType::directory);

    scalar bestErr = GREAT, latestT = -GREAT;
    fileName bestDir, latestDir;

    forAll(entries, i)
    {
        const std::string raw = entries[i].name();
        scalar t;
        if (!parseTimeName(raw, t)) continue;

        if (mag(t - tExp) < bestErr) { bestErr = mag(t - tExp); bestDir = raw; }
        if (t > latestT)             { latestT = t;             latestDir = raw; }
    }

    // Normal case: a time directory within tol of the expected checkpoint
    if (!bestDir.empty() && bestErr <= tol)
    {
        timeDir = bestDir;
        return true;
    }

    // Member stopped early (e.g. residualControl) but advanced in this
    // cycle: rename its latest time to the canonical checkpoint name.
    // A crashed member (latest <= previous checkpoint) is NOT renamed.
    if (!latestDir.empty() && latestT > tPrev + tol && latestT < tExp)
    {
        const fileName canonical = Time::timeName(tExp, 10);
        Info << "  Renaming " << caseDir/latestDir << " -> "
             << caseDir/canonical << endl;
        mv(caseDir/latestDir, caseDir/canonical);
        timeDir = canonical;
        return true;
    }

    std::ostringstream oss;
    oss << "no time directory within " << tol << " of checkpoint " << tExp
        << " in " << std::string(caseDir) << " (latest: "
        << (latestDir.empty() ? std::string("none") : std::string(latestDir))
        << ", previous checkpoint " << tPrev << ")";
    why = oss.str();
    return false;
}


fileName EnKF_multiVariable_multiDomain_clustering::resolveTimeDirectory
(
    const fileName& caseDir,
    const label simI
)
{
    fileName    timeDir;
    std::string why;

    if (!tryResolveTimeDirectory(caseDir, simI, timeDir, why))
    {
        FatalErrorInFunction << why << abort(FatalError);
    }
    return timeDir;
}


bool EnKF_multiVariable_multiDomain_clustering::xyFileIsFinite
(
    const fileName& xyFile,
    const label     nObs,
    std::string&    why
) const
{
    std::ifstream f(xyFile.c_str());
    if (!f.is_open())
    {
        why = "cannot open " + std::string(xyFile);
        return false;
    }

    std::string line;
    label rows = 0, lineNo = 0;

    while (std::getline(f, line))
    {
        lineNo++;
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;

        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok)
        {
            char* end = nullptr;
            const double v = std::strtod(tok.c_str(), &end);
            if (end == tok.c_str() || *end != '\0' || !std::isfinite(v))
            {
                why = "non-finite value '" + tok + "' at line "
                    + std::to_string(lineNo) + " of " + std::string(xyFile);
                return false;
            }
        }
        rows++;
    }

    if (rows != nObs)
    {
        why = std::to_string(rows) + " data rows in " + std::string(xyFile)
            + ", expected " + std::to_string(nObs);
        return false;
    }
    return true;
}


void EnKF_multiVariable_multiDomain_clustering::detectFailedMembers()
{
    if (simulationDirs_.empty())
    {
        detectSimulationDirectories();
    }

    // reason[ensI] empty = valid. A member failing in ANY sim is excluded
    // everywhere (its parameters are shared across domains).
    List<std::string> reason(nEnsembles_);

    forAll(simulationDirs_, simI)
    {
        const word simName = simulationDirs_[simI].name();

        for (label ensI = 0; ensI < nEnsembles_; ensI++)
        {
            if (!reason[ensI].empty()) continue;

            const fileName memberDir =
                simulationDirs_[simI]/"EnKFmembers"/("member_" + Foam::name(ensI));

            std::string why;
            fileName    timeDir, postTime;

            if (!isDir(memberDir))
            {
                why = "member directory missing";
            }
            else if (tryResolveTimeDirectory(memberDir, simI, timeDir, why))
            {
                forAll(observationFields_, fI)
                {
                    if (!isFile(memberDir/timeDir/observationFields_[fI]))
                    {
                        why = "field " + std::string(observationFields_[fI])
                            + " missing at time " + std::string(timeDir);
                        break;
                    }
                }
            }

            if (why.empty())
            {
                const fileName ppDir = memberDir/"postProcessing"/"sampleDict";

                if (!isDir(ppDir))
                {
                    why = "no postProcessing/sampleDict";
                }
                else if (tryResolveTimeDirectory(ppDir, simI, postTime, why))
                {
                    const fileName xy =
                        ppDir/postTime/fileName(featureSampleSetName_ + ".xy");

                    if (!isFile(xy))
                    {
                        why = "sample file missing: " + std::string(xy);
                    }
                    else
                    {
                        xyFileIsFinite(xy, nObservationPointsPerSim_[simI], why);
                    }
                }
            }

            if (!why.empty())
            {
                reason[ensI] = std::string(simName) + ": " + why;
            }
        }
    }

    validMembers_.clear();
    failedMembers_.clear();
    failReasons_.clear();

    forAll(reason, ensI)
    {
        if (reason[ensI].empty())
        {
            validMembers_.append(ensI);
        }
        else
        {
            failedMembers_.append(ensI);
            failReasons_.append(reason[ensI]);
        }
    }

    nValid_ = validMembers_.size();
    const label nFailed = failedMembers_.size();

    Info << "Member check: " << nValid_ << " valid, " << nFailed
         << " failed (of " << nEnsembles_ << ")" << endl;

    if (nFailed == 0) return;

    forAll(failedMembers_, f)
    {
        WarningInFunction
            << "member_" << failedMembers_[f] << " FAILED ("
            << failReasons_[f].c_str() << "). Excluded from this analysis; "
            << "new parameters will be drawn from the analysis ensemble."
            << endl;
    }

    if (!restartFromFirstTime_)
    {
        FatalErrorInFunction
            << nFailed << " member(s) failed. Member failures are only "
            << "tolerated with restartMode firstTime (a failed member has "
            << "no valid state to continue from)." << abort(FatalError);
    }

    if (scalar(nFailed) > maxFailedFraction_*nEnsembles_ + SMALL)
    {
        FatalErrorInFunction
            << nFailed << "/" << nEnsembles_ << " members failed, above "
            << "maxFailedFraction = " << maxFailedFraction_
            << ". Something systematic is wrong; check the member logs."
            << abort(FatalError);
    }

    if (nValid_ < 2)
    {
        FatalErrorInFunction
            << "Only " << nValid_ << " valid member(s): cannot compute "
            << "covariances." << abort(FatalError);
    }
}


void EnKF_multiVariable_multiDomain_clustering::compactStoredPerturbations()
{
    if (nValid_ == nEnsembles_) return;

    scalarField obsC(nObservables_*nValid_);
    for (label j = 0; j < nObservables_; j++)
    {
        for (label k = 0; k < nValid_; k++)
        {
            obsC[j*nValid_ + k] =
                perturbedObsInBIA_[j*nEnsembles_ + validMembers_[k]];
        }
    }
    perturbedObsInBIA_.transfer(obsC);

    const label nAlpha = nModelParams_*nClusters_;
    scalarField priorC(nAlpha*nValid_);
    for (label a = 0; a < nAlpha; a++)
    {
        for (label k = 0; k < nValid_; k++)
        {
            priorC[a*nValid_ + k] = paramPrior[a*nEnsembles_ + validMembers_[k]];
        }
    }
    paramPrior.transfer(priorC);

    Info << "Compacted stored perturbations to " << nValid_
         << " valid members." << endl;
}


// ------------------------------------------------------------------
// Helper: readCellCountFromUField
//   Reads the integer cell count that appears after "internalField"
//   in an OpenFOAM U field file.
// ------------------------------------------------------------------
label EnKF_multiVariable_multiDomain_clustering::readCellCountFromUField
(
    const fileName& uFieldFile
)
{
    std::ifstream f(uFieldFile);
    std::string line;
    bool foundInternalField = false;

    while (std::getline(f, line))
    {
        if (line.find("internalField") != std::string::npos)
        {
            foundInternalField = true;
            continue;
        }
        if (foundInternalField)
        {
            std::istringstream iss(line);
            label count;
            if (iss >> count)
            {
                return count;
            }
        }
    }
    return 0;
}


void EnKF_multiVariable_multiDomain_clustering::readFieldComponent
(
    const fileName& fieldFile,
    const label     nCells,
    scalarField&    ensemble,
    const label     rowBase,
    const label     totalCols,
    const label     ensI,
    const bool      isVector,
    const label     component
)
{
    std::ifstream f(fieldFile);
    std::string line;
    bool foundInternalField = false;
    bool foundCount         = false;
    bool inDataBlock        = false;
    label cellIndex         = 0;

    while (std::getline(f, line) && cellIndex < nCells)
    {
        if (line.find("internalField") != std::string::npos)
        {
            foundInternalField = true;
            continue;
        }

        if (foundInternalField && !foundCount)
        {
            std::istringstream iss(line);
            label count;
            if (iss >> count) { foundCount = true; }
            continue;
        }

        if (isVector)
        {
            if (foundCount && line.find('(') != std::string::npos
                && line.find(')') != std::string::npos)
            {
                size_t start = line.find('(');
                size_t end   = line.find(')');
                std::string vec = line.substr(start + 1, end - start - 1);
                std::istringstream viss(vec);
                scalar v0, v1, v2;
                if (viss >> v0 >> v1 >> v2)
                {
                    scalar vals[3] = {v0, v1, v2};
                    label matrixIndex = (rowBase + cellIndex) * totalCols + ensI;
                    ensemble[matrixIndex] = vals[component];
                    cellIndex++;
                }
            }
        }
        else
        {
            if (foundCount && !inDataBlock)
            {
                if (line.find('(') != std::string::npos) { inDataBlock = true; }
                continue;
            }

            if (inDataBlock)
            {
                if (line.find(')') != std::string::npos) break;

                std::istringstream iss(line);
                scalar val;
                if (iss >> val)
                {
                    label matrixIndex = (rowBase + cellIndex) * totalCols + ensI;
                    ensemble[matrixIndex] = val;
                    cellIndex++;
                }
            }
        }
    }

    if (cellIndex != nCells)
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::readFieldComponent")
            << "Expected " << nCells << " cells, read " << cellIndex
            << " from " << fieldFile << endl;
    }
}



void EnKF_multiVariable_multiDomain_clustering::readParameterValues
(
    const fileName& memberDir,
    scalarField& paramValues
)
{
    // paramValues layout: paramI * nClusters_ + clusterI
    paramValues.setSize(nModelParams_ * nClusters_);

    // ------------------------------------------------------------------
    // Initialise with cluster-specific defaults.
    //
    // paramDefaults_ is now flat of size nModelParams_ * nClusters_,
    // indexed as paramI * nClusters_ + clusterI — same layout as paramValues.
    // ------------------------------------------------------------------
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label flatIdx = paramI * nClusters_ + clusterI;
            paramValues[flatIdx] = paramDefaults_[flatIdx];
        }
    }

    fileName paramFile = memberDir/parameterFile_;
    if (!isFile(paramFile))
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::readParameterValues")
            << "Parameter file not found in: " << memberDir
            << ", using default values" << endl;
        return;
    }

    std::ifstream inFile(paramFile);
    std::string fileLine;
    boolList foundParams(nModelParams_, false);

    while (std::getline(inFile, fileLine))
    {
        // Trim leading whitespace
        std::string trimmed = fileLine;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));

        // Skip comment lines
        if (trimmed.size() >= 2 && trimmed.substr(0, 2) == "//") continue;

        for (label paramI = 0; paramI < nModelParams_; paramI++)
        {
            if (foundParams[paramI]) continue;

            const std::string paramName = std::string(paramNames_[paramI]);
            const std::size_t nameLen   = paramName.length();

            // Line must start with paramName
            if (trimmed.compare(0, nameLen, paramName) != 0) continue;

            // Character after paramName must be whitespace
            if (nameLen < trimmed.size() && !std::isspace(trimmed[nameLen])) continue;

            std::size_t openParen  = trimmed.find('(');
            std::size_t closeParen = trimmed.find(')');
            std::string listStr;

            if (openParen != std::string::npos && closeParen != std::string::npos)
            {
                listStr = trimmed.substr(openParen + 1, closeParen - openParen - 1);
            }
            else if (nClusters_ == 1)
            {
                // Global parameter written as a plain scalar: "name   value;"
                std::size_t semiPos = trimmed.find(';');
                if (semiPos == std::string::npos)
                {
                    WarningIn("EnKF_multiVariable_multiDomain_clustering::readParameterValues")
                        << "Parameter '" << paramName
                        << "' has no terminating ';'. Skipping." << endl;
                    continue;
                }
                listStr = trimmed.substr(nameLen, semiPos - nameLen);
            }
            else
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::readParameterValues")
                    << "Parameter '" << paramName
                    << "' found but not in list format (...) although nClusters="
                    << nClusters_ << ". Skipping." << endl;
                continue;
            }

            std::istringstream iss(listStr);

            label clusterI = 0;
            scalar val;
            while (iss >> val && clusterI < nClusters_)
            {
                paramValues[paramI * nClusters_ + clusterI] = val;
                clusterI++;
            }

            if (clusterI != nClusters_)
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::readParameterValues")
                    << "Parameter '" << paramName << "': expected "
                    << nClusters_ << " cluster values, found "
                    << clusterI << ". Remaining filled with cluster-specific defaults."
                    << endl;
                // Remaining clusters already hold their defaults from initialisation
            }

            foundParams[paramI] = true;
        }
    }
    inFile.close();

    // Warn about missing parameters — show all cluster defaults for context
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        if (!foundParams[paramI])
        {
            Info << "Warning: could not find parameter '" << paramNames_[paramI]
                 << "' in: " << paramFile
                 << ". Using cluster-specific defaults:";
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                label flatIdx = paramI * nClusters_ + clusterI;
                Info << "  c" << clusterI << "=" << paramDefaults_[flatIdx];
            }
            Info << endl;
        }
    }
}

void EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations()
{
    Info << "Reading stored perturbations from " << nSimulations_
         << " simulation domains..." << endl;
    Info << "  nEnsembles_:   " << nEnsembles_   << endl;
    Info << "  nObservables_: " << nObservables_  << endl;
    Info << "  nModelParams_: " << nModelParams_  << endl;
    Info << "  nClusters_:    " << nClusters_     << endl;

    if (simulationDirs_.empty())
    {
        detectSimulationDirectories();
    }

    // ------------------------------------------------------------------
    // Resize and zero output arrays
    //   perturbedObsInBIA_ : [globalObsIndex * nEnsembles_ + ensI]
    //   paramPrior         : [(paramI * nClusters_ + clusterI) * nEnsembles_ + ensI]
    // ------------------------------------------------------------------
    perturbedObsInBIA_.setSize(nObservables_ * nEnsembles_);
    paramPrior.setSize(nModelParams_ * nClusters_ * nEnsembles_);
    perturbedObsInBIA_ = 0.0;
    paramPrior         = 0.0;

    bool parametersRead   = false;
    label globalObsOffset = 0;

    forAll(simulationDirs_, simI)
    {
        const fileName& simDir = simulationDirs_[simI];
        const word simName     = simDir.name();
        const label nObsSim    = nObservationPointsPerSim_[simI];

        fileName perturbFile = simDir/"constant"/"enkfPerturbations";

        Info << "\nReading perturbations for simulation: " << simName
             << " (" << nObsSim << " obs points/field)" << endl;

        if (!isFile(perturbFile))
        {
            FatalErrorInFunction
                << "Perturbation file not found: " << perturbFile
                << "\nRun setup phase first to generate perturbations."
                << abort(FatalError);
        }

        std::ifstream inFile(perturbFile);
        std::string fileContent(
            (std::istreambuf_iterator<char>(inFile)),
             std::istreambuf_iterator<char>());
        inFile.close();
        
        if (fileContent.find("obsErrorModel") == std::string::npos)
        {
            FatalErrorInFunction
                << perturbFile << " was generated with the old uniform "
                << "observation error (obsUncertainty)." << nl
                << "Regenerate it (rerun the setup phase) so the perturbed "
                << "observations use the per-field sigma."
                << abort(FatalError);
        }

        // ------------------------------------------------------------------
        // Validate header dimensions
        // ------------------------------------------------------------------
        label fileEnsembles = parseScalarEntry(fileContent, "nEnsembles");
        label fileNObs      = parseScalarEntry(fileContent, "nObservationPoints");
        label fileNFields   = parseScalarEntry(fileContent, "nFields");
        label fileNParams   = parseScalarEntry(fileContent, "nParameters");
        label fileClusters  = parseScalarEntry(fileContent, "nClusters");

        if (fileEnsembles > 0 && fileEnsembles != nEnsembles_)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                << simName << ": nEnsembles mismatch — stored=" << fileEnsembles
                << ", current=" << nEnsembles_ << endl;
        }
        if (fileNObs > 0 && fileNObs != nObsSim)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                << simName << ": nObservationPoints mismatch — stored=" << fileNObs
                << ", current=" << nObsSim << endl;
        }
        if (fileNFields > 0 && fileNFields != observationFields_.size())
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                << simName << ": nFields mismatch — stored=" << fileNFields
                << ", current=" << observationFields_.size() << endl;
        }
        if (fileNParams > 0 && fileNParams != nModelParams_)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                << simName << ": nParameters mismatch — stored=" << fileNParams
                << ", current=" << nModelParams_ << endl;
        }
        if (fileClusters > 0 && fileClusters != nClusters_)
        {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                << simName << ": nClusters mismatch — stored=" << fileClusters
                << ", current=" << nClusters_ << endl;
        }

        // ------------------------------------------------------------------
        // Read per-field observation perturbations (format unchanged)
        // ------------------------------------------------------------------
        forAll(observationFields_, fieldI)
        {
            const word& fieldName = observationFields_[fieldI];
            // Trailing newline: the key is written as "<key>\n", so this
            // prevents T matching Trms, F matching Frms, etc.
            std::string searchKey =
                "perturbedObservations_" + std::string(fieldName) + "\n";

            Info << "  Reading " << searchKey << "..." << endl;

            std::size_t keyPos = fileContent.find(searchKey);
            if (keyPos == std::string::npos)
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                    << "Key '" << searchKey << "' not found in: " << perturbFile
                    << endl;
                globalObsOffset += nObsSim;
                continue;
            }

            std::size_t outerOpenPos  = fileContent.find('(', keyPos);
            std::size_t outerClosePos =
                findMatchingParenthesis(fileContent, outerOpenPos);

            if (outerOpenPos  == std::string::npos
             || outerClosePos == std::string::npos)
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                    << "Could not find data block for: " << searchKey << endl;
                globalObsOffset += nObsSim;
                continue;
            }

            std::string outerData = fileContent.substr(
                outerOpenPos + 1, outerClosePos - outerOpenPos - 1);

            std::istringstream outerStream(outerData);

            for (label pointI = 0; pointI < nObsSim; pointI++)
            {
                label numElements = 0;
                outerStream >> numElements;

                char paren;
                outerStream >> paren;   // '('

                label globalObsIndex = globalObsOffset + fieldI * nObsSim + pointI;

                for (label ensI = 0; ensI < numElements; ensI++)
                {
                    scalar value;
                    outerStream >> value;

                    if (ensI < nEnsembles_)
                    {
                        label flatIdx = globalObsIndex * nEnsembles_ + ensI;
                        if (flatIdx < perturbedObsInBIA_.size())
                        {
                            perturbedObsInBIA_[flatIdx] = value;
                        }
                    }
                }

                outerStream >> paren;   // ')'
            }

            Info << "  Successfully read " << nObsSim
                 << " observation points for field " << fieldName << endl;
        }

        globalObsOffset += nObsSim * observationFields_.size();

        // ------------------------------------------------------------------
        // Read shared parameter perturbations — first simulation only.
        //
        // NEW file format: List< List< List<scalar> > >
        //   outer level:  paramI        (fileNParams entries)
        //   middle level: clusterI      (fileClusters entries)
        //   inner level:  ensI          (fileEnsembles entries)
        //
        // Each level is written by OpenFOAM as:  N ( v0 v1 ... vN-1 )
        // ------------------------------------------------------------------
        if (simI == 0 && !parametersRead)
        {
            Info << "  Reading shared parameter perturbations from "
                 << simName << " (new cluster-aware format)..." << endl;

            std::size_t priorsKeyPos = fileContent.find("perturbedPriors");
            if (priorsKeyPos == std::string::npos)
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                    << "Key 'perturbedPriors' not found in: " << perturbFile
                    << endl;
            }
            else
            {
                // Locate the outermost '(' of the entire List<List<List<scalar>>>
                std::size_t outerOpenPos =
                    fileContent.find('(', priorsKeyPos);
                std::size_t outerClosePos =
                    findMatchingParenthesis(fileContent, outerOpenPos);

                if (outerOpenPos  == std::string::npos
                 || outerClosePos == std::string::npos)
                {
                    WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
                        << "Could not find outer data block for perturbedPriors in: "
                        << perturbFile << endl;
                }
                else
                {
                    std::string outerData = fileContent.substr(
                        outerOpenPos + 1,
                        outerClosePos - outerOpenPos - 1);

                    std::istringstream outerStream(outerData);

                    // -------------------------------------------------------
                    // Outer loop: paramI
                    // Each param entry is written as:
                    //   nClusters_ ( <cluster0_list> <cluster1_list> ... )
                    // -------------------------------------------------------
                    for (label paramI = 0; paramI < fileNParams; paramI++)
                    {
                        label numClusters = 0;
                        outerStream >> numClusters;

                        char paren;
                        outerStream >> paren;   // '(' opening clusters list

                        // ---------------------------------------------------
                        // Middle loop: clusterI
                        // Each cluster entry is written as:
                        //   nEnsembles_ ( v0 v1 ... vN-1 )
                        // ---------------------------------------------------
                        for (label clusterI = 0; clusterI < numClusters; clusterI++)
                        {
                            label numElements = 0;
                            outerStream >> numElements;

                            outerStream >> paren;   // '(' opening ensemble values

                            for (label ensI = 0; ensI < numElements; ensI++)
                            {
                                scalar value;
                                outerStream >> value;

                                if (   paramI   < nModelParams_
                                    && clusterI < nClusters_
                                    && ensI     < nEnsembles_)
                                {
                                    label flatIdx =
                                        (paramI * nClusters_ + clusterI)
                                        * nEnsembles_ + ensI;
                                    paramPrior[flatIdx] = value;
                                }
                            }

                            outerStream >> paren;   // ')' closing ensemble values
                        }

                        outerStream >> paren;   // ')' closing clusters list
                    }

                    parametersRead = true;
                    Info << "  Successfully read " << fileNParams
                         << " parameters x " << fileClusters
                         << " clusters." << endl;
                }
            }
        }
        else if (simI > 0)
        {
            Info << "  Skipping parameters — using shared values from "
                 << simulationDirs_[0].name() << endl;
        }

    } // end simI loop

    // ------------------------------------------------------------------
    // Verify globalObsOffset matches nObservables_
    // ------------------------------------------------------------------
    if (globalObsOffset != nObservables_)
    {
        WarningIn("EnKF_multiVariable_multiDomain_clustering::readStoredPerturbations")
            << "Final globalObsOffset=" << globalObsOffset
            << " does not match nObservables_=" << nObservables_
            << ". Check perturbation files." << endl;
    }

    // ------------------------------------------------------------------
    // Diagnostics — sample values per domain and per field
    // ------------------------------------------------------------------
    Info << "\nSample perturbed observation values:" << endl;
    label sampleOffset = 0;
    forAll(simulationDirs_, simI)
    {
        const label nObsSim = nObservationPointsPerSim_[simI];
        forAll(observationFields_, fieldI)
        {
            label globalObsIndex = sampleOffset + fieldI * nObsSim;
            Info << "  " << simulationDirs_[simI].name()
                 << " / " << observationFields_[fieldI]
                 << " point 0:" << endl;
            for (label ensI = 0; ensI < min(3, nEnsembles_); ensI++)
            {
                Info << "    member " << ensI << ": "
                     << perturbedObsInBIA_[globalObsIndex * nEnsembles_ + ensI]
                     << endl;
            }
        }
        sampleOffset += nObsSim * observationFields_.size();
    }

    Info << "\nSample parameter priors (all clusters, first 3 members):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        Info << "  " << paramNames_[paramI] << ":" << endl;
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            Info << "    cluster " << clusterI << ":";
            for (label ensI = 0; ensI < min(3, nEnsembles_); ensI++)
            {
                label flatIdx = (paramI * nClusters_ + clusterI) * nEnsembles_ + ensI;
                Info << "  member" << ensI << "=" << paramPrior[flatIdx];
            }
            Info << endl;
        }
    }

    Info << "\nreadStoredPerturbations complete." << endl;
    Info << "  Observations read: " << nObservables_ << " points x "
         << nEnsembles_ << " members" << endl;
    Info << "  Parameters read:   " << nModelParams_ << " params x "
         << nClusters_ << " clusters x "
         << nEnsembles_ << " members" << endl;
}


// ------------------------------------------------------------------
// Helper: parseScalarEntry
//   Finds "keyword   value;" in fileContent and returns value as label.
//   Returns -1 if not found.
// ------------------------------------------------------------------
label EnKF_multiVariable_multiDomain_clustering::parseScalarEntry
(
    const std::string& content,
    const std::string& keyword
)
{
    std::size_t keyPos = content.find(keyword);
    if (keyPos == std::string::npos) return -1;

    std::size_t valueStart =
        content.find_first_not_of(" \t", keyPos + keyword.length());
    std::size_t valueEnd = content.find(';', valueStart);

    if (valueStart == std::string::npos || valueEnd == std::string::npos)
        return -1;

    std::string valueStr = content.substr(valueStart, valueEnd - valueStart);

    // Trim trailing whitespace
    valueStr.erase(valueStr.find_last_not_of(" \t\r\n") + 1);

    try { return std::stoi(valueStr); }
    catch (...) { return -1; }
}

// Helper method to find matching parenthesis (add this to your class)
std::size_t EnKF_multiVariable_multiDomain_clustering::findMatchingParenthesis(const std::string& text, std::size_t start)
{
    if (start >= text.size() || text[start] != '(')
        return std::string::npos;
    
    label count = 1;
    std::size_t pos = start + 1;
    
    while (pos < text.size() && count > 0)
    {
        if (text[pos] == '(')
            count++;
        else if (text[pos] == ')')
            count--;
        pos++;
    }
    
    return (count == 0) ? pos - 1 : std::string::npos;
}


void EnKF_multiVariable_multiDomain_clustering::computeEnsembleMean(
    scalarField& ensemble,
    scalarField& ensembleLocalMean)
{
    const label nFields   = observationFields_.size();
    const label totalCols = nValid_;

    // ------------------------------------------------------------------
    // Recompute block sizes — must match createEnsemble exactly
    //
    //   phi block:   sum_simI [ domainCells_[simI] * nFields ]
    //   alpha block: nModelParams_
    //   q block:     sum_simI [ nObservationPointsPerSim_[simI] * nFields ]
    // ------------------------------------------------------------------
    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    label totalQRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalQRows += nObservationPointsPerSim_[simI] * nFields;
    }

    label alphaRows      = nModelParams_ * nClusters_;
    label phiRowOffset   = 0;
    label alphaRowOffset = totalPhiRows;
    label qRowOffset     = totalPhiRows + alphaRows;
    label totalRows      = qRowOffset + totalQRows;

    // Validate incoming ensemble size
    if (ensemble.size() != totalRows * totalCols)
    {
        FatalErrorInFunction
            << "Wrong ensemble size."
            << " Expected: " << totalRows * totalCols
            << ", Got: " << ensemble.size()
            << abort(FatalError);
    }

    // Resize and zero the mean vector
    ensembleLocalMean.setSize(totalRows);
    ensembleLocalMean = 0.0;

    Info << "Computing ensemble mean:" << endl;
    Info << "  phi block   (state):  " << totalPhiRows  << " rows" << endl;
    Info << "  alpha block (params): " << nModelParams_ << " rows" << endl;
    Info << "  q block     (obs):    " << totalQRows    << " rows" << endl;
    Info << "  Total rows:           " << totalRows     << endl;

    // ------------------------------------------------------------------
    // Core computation — identical for all blocks:
    //   mean[row] = (1/N) * sum_{ensI} ensemble[row * N + ensI]
    //
    // The multiVariable version correctly identified that a single loop
    // over all rows is sufficient regardless of block structure.
    // ------------------------------------------------------------------
    for (label row = 0; row < totalRows; row++)
    {
        scalar meanValue = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            meanValue += ensemble[row * totalCols + ensI];
        }
        ensembleLocalMean[row] = meanValue / nValid_;
    }

    // ------------------------------------------------------------------
    // Diagnostics — block-aware summary statistics
    // ------------------------------------------------------------------

    // phi block: per-sim, per-field summary
    Info << "\nphi block means (per sim, per field):" << endl;
    label phiSimOffset = phiRowOffset;
    forAll(simulationDirs_, simI)
    {
        const label nCells  = domainCells_[simI];
        const word  simName = simulationDirs_[simI].name();

        forAll(observationFields_, fieldI)
        {
            label rowBase = phiSimOffset + fieldI * nCells;

            scalar minVal = GREAT, maxVal = -GREAT, sumVal = 0.0;
            for (label cellI = 0; cellI < nCells; cellI++)
            {
                scalar v = ensembleLocalMean[rowBase + cellI];
                minVal = min(minVal, v);
                maxVal = max(maxVal, v);
                sumVal += v;
            }
            Info << "  " << simName << " / " << observationFields_[fieldI]
                 << ": min=" << minVal
                 << "  max=" << maxVal
                 << "  avg=" << sumVal / nCells << endl;
        }

        phiSimOffset += nCells * nFields;
    }

    // alpha block: one value per parameter per cluster
    Info << "\nalpha block means (shared parameters):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI << ": "
                << ensembleLocalMean[alphaRowOffset + alphaIdx] << endl;
        }
    }

    // q block: per-sim, per-field summary
    Info << "\nq block means (per sim, per field):" << endl;
    label qSimOffset = qRowOffset;
    forAll(simulationDirs_, simI)
    {
        const label nObs    = nObservationPointsPerSim_[simI];
        const word  simName = simulationDirs_[simI].name();

        forAll(observationFields_, fieldI)
        {
            label rowBase = qSimOffset + fieldI * nObs;

            scalar minVal = GREAT, maxVal = -GREAT, sumVal = 0.0;
            for (label obsI = 0; obsI < nObs; obsI++)
            {
                scalar v = ensembleLocalMean[rowBase + obsI];
                minVal = min(minVal, v);
                maxVal = max(maxVal, v);
                sumVal += v;
            }
            Info << "  " << simName << " / " << observationFields_[fieldI]
                 << ": min=" << minVal
                 << "  max=" << maxVal
                 << "  avg=" << sumVal / nObs << endl;
        }

        qSimOffset += nObs * nFields;
    }

    Info << "\ncomputeEnsembleMean completed." << endl;
}

void EnKF_multiVariable_multiDomain_clustering::computeEnsemblePerturbations(
    const scalarField& ensemble,
    const scalarField& ensembleLocalMean,
    scalarField& ensemblePerturbation)
{
    const label nFields   = observationFields_.size();
    const label totalCols = nValid_;

    // ------------------------------------------------------------------
    // Recompute block sizes — must match createEnsemble / computeEnsembleMean
    // ------------------------------------------------------------------
    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    label totalQRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalQRows += nObservationPointsPerSim_[simI] * nFields;
    }

    label alphaRows      = nModelParams_ * nClusters_;
    label alphaRowOffset = totalPhiRows;
    label qRowOffset     = totalPhiRows + alphaRows;
    label phiRowOffset   = 0;

    Info << "Computing ensemble perturbations:" << endl;
    Info << "  phi block   (state):  " << totalPhiRows << " rows" << endl;
    Info << "  alpha block (params): " << alphaRows    << " rows (" << nModelParams_ << " params x " << nClusters_ << " clusters)" << endl;
    Info << "  q block     (obs):    " << totalQRows   << " rows" << endl;

    // Resize perturbation array to match ensemble
    ensemblePerturbation.setSize(ensemble.size());
    ensemblePerturbation = 0.0;

    // ------------------------------------------------------------------
    // Core computation — single flat loop, valid for all blocks:
    //   perturbation[row * N + ensI] = ensemble[row * N + ensI] - mean[row]
    //
    // stateIndex = i / nValid_ gives the row for any flat index i.
    // ------------------------------------------------------------------
    for (label i = 0; i < ensemble.size(); i++)
    {
        label row = i / totalCols;
        ensemblePerturbation[i] = ensemble[i] - ensembleLocalMean[row];
    }

    // ------------------------------------------------------------------
    // Validation — check for NaN / Inf (from multiVariable)
    // ------------------------------------------------------------------
    label invalidCount = 0;
    forAll(ensemblePerturbation, i)
    {
        if (std::isnan(ensemblePerturbation[i]) || std::isinf(ensemblePerturbation[i]))
        {
            invalidCount++;
            if (invalidCount <= 10)
            {
                WarningIn
                (
                    "EnKF_multiVariable_multiDomain_clustering::computeEnsemblePerturbations"
                )
                    << "Invalid perturbation at flat index " << i
                    << " (row=" << i / totalCols
                    << ", ensI=" << i % totalCols
                    << "): " << ensemblePerturbation[i] << endl;
            }
        }
    }

    if (invalidCount > 0)
    {
        WarningIn
        (
            "EnKF_multiVariable_multiDomain_clustering::computeEnsemblePerturbations"
        )
            << "Total invalid perturbation values: " << invalidCount << endl;
    }

    // ------------------------------------------------------------------
    // Diagnostics — verify mean of perturbations is ~0 for first row
    // of each block, and print per-block statistics
    // ------------------------------------------------------------------


    // Zero-mean check for first row of each block
    auto checkZeroMean = [&](label row, const std::string& label_)
    {
        scalar meanPert = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            meanPert += ensemblePerturbation[row * totalCols + ensI];
        }
        meanPert /= nValid_;
        Info << "  " << label_ << " mean perturbation: " << meanPert
             << " (should be ~0)" << endl;
    };

    Info << "\nZero-mean check (first row of each block):" << endl;
    if (totalPhiRows  > 0) checkZeroMean(phiRowOffset,   "phi[0]  ");
    if (nModelParams_ > 0) checkZeroMean(alphaRowOffset, "alpha[0]");
    if (totalQRows    > 0) checkZeroMean(qRowOffset,     "q[0]    ");

    // Per-block min/max/avg of absolute perturbations
    auto blockStats = [&](label rowStart, label nRows, const std::string& blockLabel)
    {
        scalar minAbs = GREAT, maxAbs = 0.0, sumAbs = 0.0;
        label  count  = nRows * nValid_;

        for (label row = rowStart; row < rowStart + nRows; row++)
        {
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar absPert =
                    Foam::mag(ensemblePerturbation[row * totalCols + ensI]);
                minAbs  = min(minAbs, absPert);
                maxAbs  = max(maxAbs, absPert);
                sumAbs += absPert;
            }
        }

        Info << "  " << blockLabel
             << ": |pert| min=" << minAbs
             << "  max=" << maxAbs
             << "  avg=" << sumAbs / count << endl;
    };

    Info << "\nAbsolute perturbation statistics:" << endl;
    if (totalPhiRows  > 0) blockStats(phiRowOffset,   totalPhiRows,  "phi  ");
    if (nModelParams_ > 0) blockStats(alphaRowOffset, alphaRows, "alpha");
    if (totalQRows    > 0) blockStats(qRowOffset,     totalQRows,    "q    ");

    Info << "\ncomputeEnsemblePerturbations completed." << endl;
}

void EnKF_multiVariable_multiDomain_clustering::cacheExperimentalValues()
{
    Info << "\ncacheExperimentalValues: populating expValues_ ("
         << nObservables_ << " entries)..." << endl;

    expValues_.setSize(nObservables_);
    expValues_ = 0.0;

    // expValues_ ordering must match globalHx_ q block ordering:
    //   domains-then-fields, i.e.:
    //   [ sim0/field0 | sim0/field1 | sim1/field0 | sim1/field1 | ... ]
    //
    // Each simDir has its own experimentalData_<fieldName> file.
    // File format (OpenFOAM dictionary):
    //   experimentalPoints
    //   (
    //       x  y  z  value
    //       ...
    //   )
    // We read the 4th column (value) in order, one entry per obs point.
    // nObservationPointsPerSim_[simI] tells us how many points to expect.

    label globalObsOffset = 0;

    forAll(simulationDirs_, simI)
    {
        const label nObsSim = nObservationPointsPerSim_[simI];
        const word  simName = simulationDirs_[simI].name();

        forAll(observationFields_, fieldI)
        {
            const word fieldName = observationFields_[fieldI];
            fileName dataFile = simulationDirs_[simI]
                              / "constant"
                              / ("experimentalData_" + fieldName);

            if (!isFile(dataFile))
            {
                FatalErrorInFunction
                    << "Experimental data file not found: " << dataFile
                    << abort(FatalError);
            }

            std::ifstream inFile(dataFile);
            if (!inFile.good())
            {
                FatalErrorInFunction
                    << "Cannot open experimental data file: " << dataFile
                    << abort(FatalError);
            }

            // Parse OpenFOAM dictionary — skip until '(' after
            // "experimentalPoints", then read x y z value lines until ')'
            std::string line;
            bool foundList = false;

            while (std::getline(inFile, line))
            {
                // Strip leading whitespace
                size_t first = line.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) continue;
                line = line.substr(first);

                if (line.find("experimentalPoints") != std::string::npos)
                {
                    foundList = true;
                    continue;
                }

                if (foundList && line.find('(') != std::string::npos)
                {
                    // Opening parenthesis found — start reading data
                    break;
                }
            }

            if (!foundList)
            {
                FatalErrorInFunction
                    << "Could not find 'experimentalPoints' in: " << dataFile
                    << abort(FatalError);
            }

            // Read nObsSim data lines: x y z value
            label pointsRead = 0;
            while (std::getline(inFile, line) && pointsRead < nObsSim)
            {
                // Strip leading whitespace
                size_t first = line.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) continue;
                line = line.substr(first);

                // Skip closing parenthesis or empty lines
                if (line.empty() || line[0] == ')' || line[0] == ';')
                    break;

                // Parse 4 columns: x y z value
                std::istringstream iss(line);
                scalar x, y, z, value;
                if (!(iss >> x >> y >> z >> value))
                {
                    WarningIn
                    (
                        "EnKF_multiVariable_multiDomain_clustering::cacheExperimentalValues"
                    )
                        << "Could not parse line in " << dataFile
                        << ": '" << line << "'" << endl;
                    continue;
                }

                label globalObsIdx = globalObsOffset
                                   + fieldI * nObsSim
                                   + pointsRead;

                if (globalObsIdx >= nObservables_)
                {
                    FatalErrorInFunction
                        << "expValues_ index out of bounds: "
                        << globalObsIdx << " >= " << nObservables_
                        << abort(FatalError);
                }

                expValues_[globalObsIdx] = value;
                pointsRead++;
            }

            inFile.close();

            if (pointsRead != nObsSim)
            {
                FatalErrorInFunction
                    << "Read " << pointsRead << " points from " << dataFile
                    << " but expected " << nObsSim
                    << abort(FatalError);
            }

            // Diagnostics
            scalar vMin = GREAT, vMax = -GREAT, vSum = 0.0;
            for (label i = 0; i < nObsSim; i++)
            {
                scalar v = expValues_[globalObsOffset + fieldI * nObsSim + i];
                vMin = min(vMin, v);
                vMax = max(vMax, v);
                vSum += v;
            }
            Info << "  " << simName << " / " << fieldName
                 << ": " << pointsRead << " points read"
                 << "  range=[" << vMin << ", " << vMax << "]"
                 << "  avg=" << vSum / nObsSim << endl;
        }

        globalObsOffset += nObsSim * observationFields_.size();
    }

    // Final validation — no zeros left from uninitialised slots
    label zeroCount = 0;
    forAll(expValues_, i)
    {
        if (mag(expValues_[i]) < SMALL) zeroCount++;
    }
    if (zeroCount > 0)
    {
        WarningIn
        (
            "EnKF_multiVariable_multiDomain_clustering::cacheExperimentalValues"
        )
            << zeroCount << " expValues_ entries are zero or near-zero. "
            << "Check experimental data files for missing points." << endl;
    }

    Info << "cacheExperimentalValues complete. "
         << nObservables_ << " values cached." << endl;
}


void EnKF_multiVariable_multiDomain_clustering::computeKalmanGain
(
    scalarField& ensemble,
    const scalarField& ensembleMean,
    scalarField& ensemblePerturbation,
    scalarField& localK
)
{
    Info << "\n=== computeKalmanGain (multi-variable multi-domain) ===" << endl;

    // ------------------------------------------------------------------
    // Step 0: Block offset arithmetic
    //         Must match createEnsemble / computeEnsembleMean exactly.
    // ------------------------------------------------------------------
    const label nFields   = observationFields_.size();
    const label totalCols = nValid_;

    // phi block size
    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    // q block size  (= nObservables_)
    label totalQRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalQRows += nObservationPointsPerSim_[simI] * nFields;
    }

    // Sanity check
    if (totalQRows != nObservables_)
    {
        FatalErrorInFunction
            << "totalQRows=" << totalQRows
            << " does not match nObservables_=" << nObservables_
            << abort(FatalError);
    }

    // Row offsets in ensemble matrix
    label alphaRows      = nModelParams_ * nClusters_;

    label alphaRowOffset = totalPhiRows;
    label qRowOffset     = totalPhiRows + alphaRows;

    label totalHXSize = totalQRows + alphaRows;

    Info << "Block dimensions:" << endl;
    Info << "  phi rows:   " << totalPhiRows << " (skipped in PHt)" << endl;
    Info << "  alpha rows: " << alphaRows << " (= nModelParams * nClusters)" << endl;
    Info << "  q rows:     " << totalQRows  << " (= nObservables_)" << endl;
    Info << "  totalHXSize (nObs+nParams*nClusters): " << totalHXSize << endl;
    Info << "  nEnsembles: " << nValid_ << endl;


    // ------------------------------------------------------------------
    // Step 0b: Compute observation activity mask
    //
    //   For each sim/field block, compute fieldAvgMag = avg(|expValues_|).
    //   Flag obs j as inactive if |expValues_[j]| < obsUncertainty_ * fieldAvgMag.
    //   Build compressed index maps activeToGlobal_ and globalToActive_.
    // ------------------------------------------------------------------
    Info << "\n=== Observation activity mask ===" << endl;

    List<bool> obsActive(totalQRows, true);

    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                // Compute field average magnitude
                scalar fieldAvgMag = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    fieldAvgMag += mag(expValues_[globalQRow + obsI]);
                }
                fieldAvgMag /= nObs;

                scalar threshold = min(0 * fieldAvgMag, 0.5);

                label nMasked = 0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label j = globalQRow + obsI;
                    if (mag(expValues_[j]) < threshold)
                    {
                        obsActive[j] = false;
                        nMasked++;
                        Info << "  MASKED: " << simName << "/"
                            << observationFields_[fieldI]
                            << " obsI=" << obsI
                            << " globalIdx=" << j
                            << " expVal=" << expValues_[j]
                            << " threshold=" << threshold << endl;
                    }
                }

                Info << "  " << simName << "/" << observationFields_[fieldI]
                    << ": fieldAvgMag=" << fieldAvgMag
                    << "  threshold=" << threshold
                    << "  nObs=" << nObs
                    << "  nMasked=" << nMasked
                    << "  nActive=" << (nObs - nMasked) << endl;

                globalQRow += nObs;
            }
        }
    }

    // Build compressed index maps
    nActiveObs_ = 0;
    activeToGlobal_.clear();
    globalToActive_.setSize(totalQRows, -1);

    for (label j = 0; j < totalQRows; j++)
    {
        if (obsActive[j])
        {
            globalToActive_[j] = nActiveObs_;
            activeToGlobal_.append(j);
            nActiveObs_++;
        }
    }

    Info << "\n  totalQRows=" << totalQRows
        << "  nActiveObs=" << nActiveObs_
        << "  nMasked=" << (totalQRows - nActiveObs_) << endl;
    Info << "=== Observation activity mask complete ===" << endl;
    
    // Compressed observation space size — used from Step 3 onwards
    label totalHXSize_active = nActiveObs_ + alphaRows;
    // ------------------------------------------------------------------
    // Step 1: Build globalHx_  size = totalHXSize * nValid_
    //
    //   Layout:  [ q_block | alpha_block ] × nValid_
    //
    //   q_block:     totalQRows rows, ordered domains-then-fields
    //                (identical ordering to ensemble q block)
    //   alpha_block: nModelParams_ rows (shared, from sim0 only)
    //
    //   Flat index: row * nValid_ + ensI
    // ------------------------------------------------------------------
    globalHx_.setSize(totalHXSize * nValid_);
    globalHx_ = 0.0;

    Info << "\nStep 1: Building globalHx_ (" << totalHXSize
         << " x " << nValid_ << ")..." << endl;

    // ------------------------------------------------------------------
    // Step 1a: q block — copy directly from ensemble q rows
    //
    //   ensemble row (qRowOffset + globalQRow) maps 1-to-1 to
    //   globalHx_ row globalQRow, since both use domains-then-fields
    //   ordering established in createEnsemble.
    // ------------------------------------------------------------------
    Info << "  Step 1a: filling q block from ensemble q rows..." << endl;

    for (label globalQRow = 0; globalQRow < totalQRows; globalQRow++)
    {
        label ensembleRow = qRowOffset + globalQRow;

        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label ensembleIdx = ensembleRow * totalCols + ensI;
            label globalHxIdx = globalQRow * nValid_ + ensI;

            if (ensembleIdx >= ensemble.size())
            {
                FatalErrorInFunction
                    << "Ensemble index out of bounds in Step 1a:"
                    << " ensembleIdx=" << ensembleIdx
                    << " ensemble.size()=" << ensemble.size()
                    << abort(FatalError);
            }

            globalHx_[globalHxIdx] = ensemble[ensembleIdx];
        }
    }

    // Diagnostics for q block
    {
        scalar qMin = GREAT, qMax = -GREAT;
        for (label i = 0; i < totalQRows * nValid_; i++)
        {
            qMin = min(qMin, globalHx_[i]);
            qMax = max(qMax, globalHx_[i]);
        }
        Info << "  globalHx_ q block range: [" << qMin << ", " << qMax << "]" << endl;

        // Sample: first obs, first 3 members
        Info << "  globalHx_ q sample (row 0, members 0-2): ";
        for (label ensI = 0; ensI < min(3, nValid_); ensI++)
        {
            Info << globalHx_[ensI] << " ";
        }
        Info << endl;
    }

    Info << "  Step 1a complete: " << totalQRows << " q rows filled." << endl;

    // ------------------------------------------------------------------
    // Step 1b: alpha block — copy from ensemble alpha rows
    //
    //   globalHx_ row (totalQRows + paramI) ← ensemble row (alphaRowOffset + paramI)
    // ------------------------------------------------------------------

    Info << "  Step 1b: filling alpha block from ensemble alpha rows..." << endl;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx    = paramI * nClusters_ + clusterI;
            label ensembleRow = alphaRowOffset + alphaIdx;
            label globalHxRow = totalQRows + alphaIdx;

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label ensembleIdx = ensembleRow * totalCols   + ensI;
                label globalHxIdx = globalHxRow * nValid_ + ensI;

                if (ensembleIdx >= ensemble.size())
                {
                    FatalErrorInFunction
                        << "Ensemble index out of bounds in Step 1b:"
                        << " paramI=" << paramI
                        << " clusterI=" << clusterI
                        << " ensembleIdx=" << ensembleIdx
                        << " ensemble.size()=" << ensemble.size()
                        << abort(FatalError);
                }

                globalHx_[globalHxIdx] = ensemble[ensembleIdx];
            }

            // Debug: first 3 members of first 3 alpha rows
            if (alphaIdx < 3)
            {
                Info << "  globalHx_ alpha[" << paramNames_[paramI]
                    << " cluster" << clusterI << "] (members 0-2): ";
                for (label ensI = 0; ensI < min(3, nValid_); ensI++)
                {
                    Info << globalHx_[globalHxRow * nValid_ + ensI] << " ";
                }
                Info << endl;
            }
        }
    }

    Info << "  Step 1b complete: " << alphaRows << " alpha rows filled." << endl;
    // ------------------------------------------------------------------
    // Step 2: Compute globalHxMean and globalHxPerturbations
    //
    //   globalHxMean         size: totalHXSize
    //   globalHxPerturbations size: totalHXSize * nValid_
    //
    //   Both use the same flat indexing as globalHx_:
    //     row * nValid_ + ensI
    //
    //   Mean is computed uniformly across all rows — no domain-specific
    //   splitting needed here because the mean-of-perturbations = 0
    //   identity holds row-by-row regardless of block structure.
    // ------------------------------------------------------------------
    Info << "\nStep 2: Computing globalHxMean and globalHxPerturbations..." << endl;

    scalarField globalHxMean(totalHXSize, 0.0);
    scalarField globalHxPerturbations(globalHx_.size(), 0.0);

    // 2a: Compute mean for each row
    for (label row = 0; row < totalHXSize; row++)
    {
        scalar sum = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            sum += globalHx_[row * nValid_ + ensI];
        }
        globalHxMean[row] = sum / nValid_;
    }

    // 2b: Compute perturbations = globalHx_ - mean
    for (label row = 0; row < totalHXSize; row++)
    {
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label idx = row * nValid_ + ensI;
            globalHxPerturbations[idx] = globalHx_[idx] - globalHxMean[row];
        }
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    // q block means — per sim, per field
    Info << "  globalHxMean — q block (per sim, per field):" << endl;
    label qMeanOffset = 0;
    forAll(simulationDirs_, simI)
    {
        const label nObs    = nObservationPointsPerSim_[simI];
        const word  simName = simulationDirs_[simI].name();

        forAll(observationFields_, fieldI)
        {
            label rowBase = qMeanOffset + fieldI * nObs;

            scalar minVal = GREAT, maxVal = -GREAT, sumVal = 0.0;
            for (label obsI = 0; obsI < nObs; obsI++)
            {
                scalar v = globalHxMean[rowBase + obsI];
                minVal = min(minVal, v);
                maxVal = max(maxVal, v);
                sumVal += v;
            }
            Info << "    " << simName << " / " << observationFields_[fieldI]
                 << ": min=" << minVal
                 << "  max=" << maxVal
                 << "  avg=" << sumVal / nObs << endl;
        }

        qMeanOffset += nObs * nFields;
    }

    // alpha block means — one per parameter per cluster
    Info << "  globalHxMean — alpha block:" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI << ": "
                << globalHxMean[totalQRows + alphaIdx] << endl;
        }
    }

    // Zero-mean check on perturbations (first row of each block)
    Info << "  Zero-mean check on globalHxPerturbations:" << endl;
    {
        // First q row
        scalar sumPert = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            sumPert += globalHxPerturbations[ensI];
        }
        Info << "    q[0]     mean pert: " << sumPert / nValid_
             << " (should be ~0)" << endl;
    }
    {
        // First alpha row
        scalar sumPert = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            sumPert += globalHxPerturbations[totalQRows * nValid_ + ensI];
        }
        Info << "    alpha[0] mean pert: " << sumPert / nValid_
             << " (should be ~0)" << endl;
    }

    // Perturbation range per block
    {
        scalar qPertMin = GREAT, qPertMax = -GREAT;
        for (label i = 0; i < totalQRows * nValid_; i++)
        {
            qPertMin = min(qPertMin, globalHxPerturbations[i]);
            qPertMax = max(qPertMax, globalHxPerturbations[i]);
        }
        Info << "  globalHxPerturbations q block range: ["
             << qPertMin << ", " << qPertMax << "]" << endl;
    }
    {
        scalar aPertMin = GREAT, aPertMax = -GREAT;
        for (label i = totalQRows * nValid_; i < globalHxPerturbations.size(); i++)
        {
            aPertMin = min(aPertMin, globalHxPerturbations[i]);
            aPertMax = max(aPertMax, globalHxPerturbations[i]);
        }
        Info << "  globalHxPerturbations alpha block range: ["
             << aPertMin << ", " << aPertMax << "]" << endl;
    }

    Info << "Step 2 complete." << endl;


    scalarField alphaMeanScale(alphaRows, 1.0);
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            alphaMeanScale[alphaIdx] = max(mag(globalHxMean[totalQRows + alphaIdx]), 1e-6);
            label row = totalQRows + alphaIdx;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                globalHxPerturbations[row * nValid_ + ensI] /= alphaMeanScale[alphaIdx];
            }
        }
    }
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                ensemblePerturbation[ensemblePertRow * totalCols + ensI] /= alphaMeanScale[alphaIdx];
            }
        }
    }

    // Normalization diagnostics
    Info << "\n--- Alpha normalization scales ---" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": mean=" << globalHxMean[totalQRows + alphaIdx]
                << "  scale=" << alphaMeanScale[alphaIdx] << endl;
        }
    }

    Info << "\n--- globalHxPerturbations alpha block AFTER normalization ---" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label row      = totalQRows + alphaIdx;

            scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = globalHxPerturbations[row * nValid_ + ensI];
                minPert    = min(minPert, v);
                maxPert    = max(maxPert, v);
                sumAbsPert += mag(v);
            }
            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": min=" << minPert
                << "  max=" << maxPert
                << "  avgMag=" << sumAbsPert / nValid_
                << "  (scale=" << alphaMeanScale[alphaIdx] << ")" << endl;
        }
    }

    // ------------------------------------------------------------------
    // obsScale[a] = sigma_j of active obs a (physical units).
    // Normalizing by sigma_j gives R = I in normalized space (Step 4)
    // and R_eff = diag(sigma_j^2) in physical space.
    // ------------------------------------------------------------------
    labelList globalQRowToFieldI(totalQRows, -1);
    {
        label g = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs = nObservationPointsPerSim_[simI];
            forAll(observationFields_, fieldI)
            {
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    globalQRowToFieldI[g++] = fieldI;
                }
            }
        }
    }

    scalarField obsScale(nActiveObs_);
    for (label a = 0; a < nActiveObs_; a++)
    {
        const label g = activeToGlobal_[a];
        obsScale[a] = obsSigma(globalQRowToFieldI[g], expValues_[g]);
    }

    // Diagnostics: SNR_j = model spread / sigma_j,
    //              trust_j = SNR^2 / (SNR^2 + 1)
    // (q rows of globalHxPerturbations are still in physical units here)
    Info << "  obsScale = per-observation sigma_j:" << endl;
    {
        label g0 = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs = nObservationPointsPerSim_[simI];
            forAll(observationFields_, fieldI)
            {
                scalar sMin = GREAT, sMax = -GREAT;
                scalar snrMin = GREAT, snrMax = -GREAT;
                scalar snrSum = 0.0, trustSum = 0.0;
                label  nAct = 0;

                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    const label g = g0 + obsI;
                    const label a = globalToActive_[g];
                    if (a < 0) continue;

                    scalar var = 0.0;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        var += sqr(globalHxPerturbations[g*nValid_ + ensI]);
                    }
                    const scalar spread = Foam::sqrt(var/(nValid_ - 1));
                    const scalar snr    = spread/obsScale[a];

                    sMin   = min(sMin, obsScale[a]);
                    sMax   = max(sMax, obsScale[a]);
                    snrMin = min(snrMin, snr);
                    snrMax = max(snrMax, snr);
                    snrSum   += snr;
                    trustSum += sqr(snr)/(sqr(snr) + 1.0);
                    nAct++;
                }

                Info << "    " << simulationDirs_[simI].name() << "/"
                     << observationFields_[fieldI]
                     << ": sigma=[" << sMin << ", " << sMax << "]"
                     << "  SNR=[" << snrMin << ", " << snrMax << "]"
                     << " avg=" << snrSum/max(nAct, 1)
                     << "  avg trust=" << trustSum/max(nAct, 1)
                     << "  nActive=" << nAct << endl;

                g0 += nObs;
            }
        }
    }


    // Verify no zeros
    {
        label nZero = 0;
        scalar minS = GREAT, maxS = -GREAT;
        forAll(obsScale, i)
        {
            if (obsScale[i] < SMALL) nZero++;
            minS = min(minS, obsScale[i]);
            maxS = max(maxS, obsScale[i]);
        }
        Info << "  obsScale global range: [" << minS << ", " << maxS << "]" << endl;
        Info << "  obsScale zero entries: " << nZero << endl;

        if (nZero > 0)
        {
            FatalErrorInFunction
                << "obsScale contains " << nZero << " zero entries."
                << abort(FatalError);
        }
    }

    Info << "--- obsScale computation complete ---" << endl;
    



    // ------------------------------------------------------------------
    // Normalize globalHxPerturbations q rows by obsScale
    // Analogous to alpha row normalization already applied above.
    // After this, Cdd will become uniform sigma_rel^2 (fixed in Step 4).
    // ------------------------------------------------------------------
    Info << "\n--- Normalizing globalHxPerturbations q rows by obsScale ---" << endl;

    // Diagnostics BEFORE normalization
    Info << "  globalHxPerturbations q block BEFORE normalization:" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label row = globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        scalar v = globalHxPerturbations[row * nValid_ + ensI];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_) << endl;

                globalQRow += nObs;
            }
        }
    }

    // Apply normalization — active obs only
    for (label activeIdx = 0; activeIdx < nActiveObs_; activeIdx++)
    {
        label globalQRow = activeToGlobal_[activeIdx];

        if (obsScale[activeIdx] < SMALL)
        {
            FatalErrorInFunction
                << "obsScale[" << activeIdx << "] = " << obsScale[activeIdx]
                << " is zero or near-zero at normalization point."
                << abort(FatalError);
        }

        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label idx = globalQRow * nValid_ + ensI;
            globalHxPerturbations[idx] /= obsScale[activeIdx];
        }
    }

    // Diagnostics AFTER normalization
    Info << "  globalHxPerturbations q block AFTER normalization:" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label row = globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        scalar v = globalHxPerturbations[row * nValid_ + ensI];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }

                // Zero-mean check on first row of this field block
                scalar meanPert = 0.0;
                for (label ensI = 0; ensI < nValid_; ensI++)
                {
                    meanPert += globalHxPerturbations[globalQRow * nValid_ + ensI];
                }
                meanPert /= nValid_;

                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_)
                    << "  firstRowMean=" << meanPert
                    << " (should be ~0)" << endl;

                globalQRow += nObs;
            }
        }
    }

    // Verify alpha block of globalHxPerturbations is untouched
    Info << "  globalHxPerturbations alpha block (should be unchanged):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label row      = totalQRows + alphaIdx;
            scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = globalHxPerturbations[row * nValid_ + ensI];
                minPert    = min(minPert, v);
                maxPert    = max(maxPert, v);
                sumAbsPert += mag(v);
            }
            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": min=" << minPert
                << "  max=" << maxPert
                << "  avgMag=" << sumAbsPert / nValid_ << endl;
        }
    }

    Info << "--- globalHxPerturbations q normalization complete ---" << endl;

    // ------------------------------------------------------------------
    // Normalize ensemblePerturbation q rows by obsScale
    // Analogous to alpha row normalization already applied above.
    // qRowOffset = totalPhiRows + nModelParams_ — must match createEnsemble.
    // ------------------------------------------------------------------

    Info << "\n--- Normalizing ensemblePerturbation q rows by obsScale ---" << endl;
    Info << "  qRowOffset=" << qRowOffset
        << "  totalQRows=" << totalQRows
        << "  totalCols=" << totalCols << endl;
    Info << "  ensemblePerturbation.size()=" << ensemblePerturbation.size() << endl;
    Info << "  Expected q block flat range: ["
        << qRowOffset * totalCols
        << ", " << (qRowOffset + totalQRows) * totalCols << ")" << endl;

    // Diagnostics BEFORE normalization
    Info << "  ensemblePerturbation q block BEFORE normalization:" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label ensemblePertRow = qRowOffset + globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        label idx = ensemblePertRow * totalCols + ensI;

                        if (idx >= ensemblePerturbation.size())
                        {
                            FatalErrorInFunction
                                << "ensemblePerturbation index out of bounds "
                                << "(BEFORE diag): idx=" << idx
                                << " size=" << ensemblePerturbation.size()
                                << " ensemblePertRow=" << ensemblePertRow
                                << " simI=" << simI
                                << " fieldI=" << fieldI
                                << " obsI=" << obsI
                                << " ensI=" << ensI
                                << abort(FatalError);
                        }

                        scalar v = ensemblePerturbation[idx];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_) << endl;

                globalQRow += nObs;
            }
        }
    }

    // Apply normalization — active obs only
    for (label activeIdx = 0; activeIdx < nActiveObs_; activeIdx++)
    {
        label globalQRow     = activeToGlobal_[activeIdx];
        label ensemblePertRow = qRowOffset + globalQRow;

        if (obsScale[activeIdx] < SMALL)
        {
            FatalErrorInFunction
                << "obsScale[" << activeIdx << "] = " << obsScale[activeIdx]
                << " is zero or near-zero at ensemblePerturbation normalization."
                << abort(FatalError);
        }

        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label idx = ensemblePertRow * totalCols + ensI;
            ensemblePerturbation[idx] /= obsScale[activeIdx];
        }
    }
    Info << "  Normalization applied to " << nActiveObs_ << " active q rows." << endl;
    
    // Diagnostics AFTER normalization
    Info << "  ensemblePerturbation q block AFTER normalization:" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label ensemblePertRow = qRowOffset + globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        scalar v = ensemblePerturbation[ensemblePertRow * totalCols + ensI];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_) << endl;

                // Zero-mean check — normalization must preserve zero mean
                scalar meanPert = 0.0;
                label  firstRow = qRowOffset + globalQRow;
                for (label ensI = 0; ensI < nValid_; ensI++)
                {
                    meanPert += ensemblePerturbation[firstRow * totalCols + ensI];
                }
                meanPert /= nValid_;
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << " first row mean after normalization: " << meanPert
                    << " (should be ~0)" << endl;

                globalQRow += nObs;
            }
        }
    }

    // Verify alpha block of ensemblePerturbation is untouched
    Info << "  ensemblePerturbation alpha block (should be unchanged by obs norm):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;
            scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = ensemblePerturbation[ensemblePertRow * totalCols + ensI];
                minPert    = min(minPert, v);
                maxPert    = max(maxPert, v);
                sumAbsPert += mag(v);
            }
            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": min=" << minPert
                << "  max=" << maxPert
                << "  avgMag=" << sumAbsPert / nValid_
                << "  (already normalized by alphaMeanScale="
                << alphaMeanScale[alphaIdx] << ")" << endl;
        }
    }

    // Verify phi block is untouched (spot check first and last row)
    Info << "  ensemblePerturbation phi block spot check (should be unchanged):" << endl;
    {
        // First phi row
        scalar meanFirst = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
            meanFirst += ensemblePerturbation[ensI];
        meanFirst /= nValid_;
        Info << "    phi row 0 mean=" << meanFirst << " (should be ~0)" << endl;

        // Last phi row
        label lastPhiRow = totalPhiRows - 1;
        scalar meanLast = 0.0;
        for (label ensI = 0; ensI < nValid_; ensI++)
            meanLast += ensemblePerturbation[lastPhiRow * totalCols + ensI];
        meanLast /= nValid_;
        Info << "    phi last row mean=" << meanLast << " (should be ~0)" << endl;
    }

    Info << "--- ensemblePerturbation q normalization complete ---" << endl;



    // --- Observation residuals: exp vs ensemble mean (assesses previous iteration) ---
    Info << "\n--- Observation residuals (expValue - ensembleMean, assesses previous update) ---" << endl;

    scalar totalRMSE = 0.0;
    label  totalCount = 0;

    label residOffset = 0;
    forAll(simulationDirs_, simI)
    {
        const label nObs    = nObservationPointsPerSim_[simI];
        const word  simName = simulationDirs_[simI].name();

        forAll(observationFields_, fieldI)
        {
            label rowBase = residOffset + fieldI * nObs;

            scalar rmse = 0.0, meanResid = 0.0, maxResid = -GREAT, minResid = GREAT;
            scalar chi2 = 0.0;

            label nActiveField = 0;
            for (label obsI = 0; obsI < nObs; obsI++)
            {
                label globalQRow = rowBase + obsI;
                if (!obsActive[globalQRow]) continue;  // skip masked

                scalar expVal    = expValues_[globalQRow];
                scalar modelMean = globalHxMean[globalQRow];
                scalar resid     = expVal - modelMean;
                chi2 += sqr(resid/obsScale[globalToActive_[globalQRow]]);

                rmse     += sqr(resid);
                meanResid += resid;
                maxResid  = max(maxResid, resid);
                minResid  = min(minResid, resid);

                totalRMSE += sqr(resid);
                totalCount++;
                nActiveField++;
            }

            rmse      = Foam::sqrt(rmse / max(nActiveField, 1));
            meanResid /= max(nActiveField, 1);

            Info << "  " << simName << "/" << observationFields_[fieldI]
                << ":  RMSE=" << rmse
                << "  RMS(resid/sigma)="
                << Foam::sqrt(chi2/max(nActiveField, 1))
                << "  meanResid=" << meanResid
                << "  range=[" << minResid << ", " << maxResid << "]" << endl;
        }

        residOffset += nObs * nFields;
    }

    Info << "  Overall RMSE (all obs): " << Foam::sqrt(totalRMSE / totalCount) << endl;



    // ------------------------------------------------------------------
    // Step 3: Compute PHt — compressed observation space
    //
    //   PHt size: (nActiveObs_ + nModelParams_) × totalHXSize_active
    //
    //   Row layout:
    //     rows [0,           nActiveObs_)          ← active q rows only
    //     rows [nActiveObs_, nActiveObs_+nParams)  ← alpha rows
    //
    //   Column layout (compressed):
    //     cols [0,            nActiveObs_)          ← active HX q columns
    //     cols [nActiveObs_,  totalHXSize_active)   ← HX alpha columns
    //
    //   For row i (q rows):     ensemblePertRow = qRowOffset + activeToGlobal_[i]
    //   For row i (alpha rows): ensemblePertRow = alphaRowOffset + (i - nActiveObs_)
    //   For col j (q cols):     hxRow = activeToGlobal_[j]
    //   For col j (alpha cols): hxRow = totalQRows + (j - nActiveObs_)
    //
    //   Flat storage: row-major, index = i * totalHXSize_active + j
    // ------------------------------------------------------------------
    Info << "\nStep 3: Computing PHt ("
        << (nActiveObs_ + alphaRows) << " x " << totalHXSize_active << ")..." << endl;

    label PHtRows = nActiveObs_ + alphaRows;
    label PHtSize = PHtRows * totalHXSize_active;

    localPHt_.setSize(PHtSize);
    localPHt_ = 0.0;

    Info << "  PHt dimensions: " << PHtRows << " x " << totalHXSize_active
        << " = " << PHtSize << " elements" << endl;
    Info << "  Row layout: active q rows [0," << nActiveObs_ << ") + "
        << "alpha rows [" << nActiveObs_ << "," << PHtRows << ")" << endl;
    // ------------------------------------------------------------------
    // 3a: Active q rows
    //     i in [0, nActiveObs_)
    //     ensemblePertRow = qRowOffset + activeToGlobal_[i]
    // ------------------------------------------------------------------
    Info << "  Step 3a: computing active q rows..." << endl;

    for (label i = 0; i < nActiveObs_; i++)
    {
        label ensemblePertRow = qRowOffset + activeToGlobal_[i];

        for (label j = 0; j < totalHXSize_active; j++)
        {
            label hxRow = (j < nActiveObs_)
                        ? activeToGlobal_[j]
                        : totalQRows + (j - nActiveObs_);

            scalar sum = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label pertIdx = ensemblePertRow * totalCols + ensI;
                label hxIdx   = hxRow * nValid_ + ensI;

                sum += ensemblePerturbation[pertIdx]
                    * globalHxPerturbations[hxIdx];
            }

            label phtIdx = i * totalHXSize_active + j;
            localPHt_[phtIdx] = sum / (nValid_ - 1);

            if (mag(localPHt_[phtIdx]) > 1e10)
            {
                localPHt_[phtIdx] =
                    1e10 * (localPHt_[phtIdx] / mag(localPHt_[phtIdx]));
            }
        }

        if (nActiveObs_ > 10 && i % (nActiveObs_ / 10) == 0)
        {
            Info << "    active q row progress: " << i << "/" << nActiveObs_ << endl;
        }
    }

    Info << "  Step 3a complete: " << nActiveObs_ << " active q rows." << endl;

    // ------------------------------------------------------------------
    // 3b: Alpha rows
    //     paramI in [0, nModelParams_)
    //     phtRow = nActiveObs_ + paramI
    //     ensemblePertRow = alphaRowOffset + paramI
    // ------------------------------------------------------------------
    Info << "  Step 3b: computing alpha rows..." << endl;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;
            label phtRow          = nActiveObs_ + alphaIdx;

            for (label j = 0; j < totalHXSize_active; j++)
            {
                label hxRow = (j < nActiveObs_)
                            ? activeToGlobal_[j]
                            : totalQRows + (j - nActiveObs_);

                scalar sum = 0.0;
                for (label ensI = 0; ensI < nValid_; ensI++)
                {
                    label pertIdx = ensemblePertRow * totalCols + ensI;
                    label hxIdx   = hxRow * nValid_ + ensI;

                    sum += ensemblePerturbation[pertIdx]
                        * globalHxPerturbations[hxIdx];
                }

                label phtIdx = phtRow * totalHXSize_active + j;
                localPHt_[phtIdx] = sum / (nValid_ - 1);

                if (mag(localPHt_[phtIdx]) > 1e10)
                {
                    localPHt_[phtIdx] =
                        1e10 * (localPHt_[phtIdx] / mag(localPHt_[phtIdx]));
                }
            }

            // Debug: first 3 alpha rows
            if (alphaIdx < 3)
            {
                Info << "    PHt alpha[" << paramNames_[paramI]
                    << " cluster" << clusterI << "] first 3 cols: ";
                for (label j = 0; j < min(3, totalHXSize_active); j++)
                {
                    Info << localPHt_[phtRow * totalHXSize_active + j] << " ";
                }
                Info << endl;
            }
        }
    }

    Info << "  Step 3b complete: " << alphaRows << " alpha rows." << endl;

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    // q rows stats
    {
        scalar phtQMin = GREAT, phtQMax = -GREAT;
        label  nanCount = 0;
        for (label i = 0; i < nActiveObs_ * totalHXSize_active; i++)
        {
            if (std::isnan(localPHt_[i]) || std::isinf(localPHt_[i]))
                nanCount++;
            else
            {
                phtQMin = min(phtQMin, localPHt_[i]);
                phtQMax = max(phtQMax, localPHt_[i]);
            }
        }
        Info << "  PHt q rows range:     [" << phtQMin << ", " << phtQMax << "]"
            << "  NaN/Inf: " << nanCount << endl;
    }

    // alpha rows stats
    {
        scalar phtAMin = GREAT, phtAMax = -GREAT;
        label  nanCount = 0;
        for (label i = nActiveObs_ * totalHXSize_active; i < PHtSize; i++)
        {
            if (std::isnan(localPHt_[i]) || std::isinf(localPHt_[i]))
                nanCount++;
            else
            {
                phtAMin = min(phtAMin, localPHt_[i]);
                phtAMax = max(phtAMax, localPHt_[i]);
            }
        }
        Info << "  PHt alpha rows range: [" << phtAMin << ", " << phtAMax << "]"
            << "  NaN/Inf: " << nanCount << endl;
    }

    // Pqq diagonal sample (first 3 active obs)
    Info << "  PHt Pqq diagonal sample (first 3 active obs):" << endl;
    for (label i = 0; i < min(3, nActiveObs_); i++)
    {
        Info << "    Pqq[" << i << "," << i << "] = "
            << localPHt_[i * totalHXSize_active + i]
            << "  (globalIdx=" << activeToGlobal_[i] << ")" << endl;
    }

    Info << "  PHt Paa diagonal (all params):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label phtIdx   = (nActiveObs_ + alphaIdx) * totalHXSize_active
                        + (nActiveObs_ + alphaIdx);
            Info << "    Paa[" << paramNames_[paramI]
                << " cluster" << clusterI << "] = "
                << localPHt_[phtIdx] << endl;
        }
    }

    Info << "\n--- PHt Paa diagonal in NORMALIZED space ---" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx   = paramI * nClusters_ + clusterI;
            label phtIdx     = (nActiveObs_ + alphaIdx) * totalHXSize_active
                            + (nActiveObs_ + alphaIdx);
            scalar paaPhysical = localPHt_[phtIdx]
                            * alphaMeanScale[alphaIdx] * alphaMeanScale[alphaIdx];
            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": Paa_normalized=" << localPHt_[phtIdx]
                << "  Paa_physical=" << paaPhysical
                << "  scale=" << alphaMeanScale[alphaIdx] << endl;
        }
    }

    // Paa alpha-alpha subblock
    Info << "\n  PHt alpha-alpha subblock (normalized space, diagonal only after zeroing):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdxI = paramI * nClusters_ + clusterI;
            Info << "  " << paramNames_[paramI] << " cluster" << clusterI << ": ";
            for (label paramJ = 0; paramJ < nModelParams_; paramJ++)
            {
                for (label clusterJ = 0; clusterJ < nClusters_; clusterJ++)
                {
                    label alphaIdxJ = paramJ * nClusters_ + clusterJ;
                    label phtIdx    = (nActiveObs_ + alphaIdxI) * totalHXSize_active
                                    + (nActiveObs_ + alphaIdxJ);
                    Info << localPHt_[phtIdx] << " ";
                }
            }
            Info << endl;
        }
    }

    Info << "\n--- PHt alpha row norms ---" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label phtRow   = nActiveObs_ + alphaIdx;

            scalar normQ = 0.0, normA = 0.0;
            for (label j = 0; j < nActiveObs_; j++)
                normQ += sqr(localPHt_[phtRow * totalHXSize_active + j]);
            for (label j = nActiveObs_; j < totalHXSize_active; j++)
                normA += sqr(localPHt_[phtRow * totalHXSize_active + j]);

            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": ||Paq||=" << Foam::sqrt(normQ)
                << "  ||Paa||=" << Foam::sqrt(normA) << endl;
        }
    }

    Info << "Step 3 complete. PHt size: " << PHtSize << " elements." << endl;


    // ------------------------------------------------------------------
    // Step 4: Build HPHt [totalHXSize_active × totalHXSize_active]
    //
    //   HPHt = [ Pqq + Cdd    Pqα          ]
    //          [ Pαq           Pαα + Cpp    ]
    //
    //   All blocks copied from compressed localPHt_.
    //   Cdd diagonal: uniform sqr(obsUncertainty_) for active obs only.
    //   Cpp diagonal: C_pp_[paramI][paramI].
    //   Frobenius regularisation added to full diagonal.
    // ------------------------------------------------------------------
    Info << "\nStep 4: Building HPHt ("
        << totalHXSize_active << " x " << totalHXSize_active << ")..." << endl;

    if (expValues_.size() != nObservables_)
    {
        FatalErrorInFunction
            << "expValues_ not cached or wrong size."
            << abort(FatalError);
    }

    if (C_pp_.n() != alphaRows)
    {
        C_pp_ = scalarSquareMatrix(alphaRows, 0.0);
    }

    // Compute C_pp_ from Paa diagonal
    const scalar minPriorVar = 1e-9;
    Info << "  Computing C_pp_ from Paa diagonal:" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label phtIdx   = (nActiveObs_ + alphaIdx) * totalHXSize_active
                        + (nActiveObs_ + alphaIdx);
            scalar paaValue = mag(localPHt_[phtIdx]);

            C_pp_[alphaIdx][alphaIdx] = max(
                sqr(priorUncertainty_[alphaIdx]),
                minPriorVar
            );

            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": Paa=" << paaValue
                << "  priorUnc=" << priorUncertainty_[paramI]
                << "  Cpp=" << C_pp_[alphaIdx][alphaIdx] << endl;
        }
    }

    scalarSquareMatrix HPHt(totalHXSize_active, 0.0);

    // ------------------------------------------------------------------
    // Block 1 (top-left): Pqq   [0,nActiveObs_) × [0,nActiveObs_)
    // ------------------------------------------------------------------
    Info << "  Block 1: Pqq..." << endl;
    for (label i = 0; i < nActiveObs_; i++)
    {
        for (label j = 0; j < nActiveObs_; j++)
        {
            label phtIdx = i * totalHXSize_active + j;
            HPHt[i][j] = localPHt_[phtIdx];
            if (mag(HPHt[i][j]) > 1e10)
                HPHt[i][j] = 1e10 * (HPHt[i][j] / mag(HPHt[i][j]));
        }
    }

    // ------------------------------------------------------------------
    // Block 2 (top-right): Pqα  [0,nActiveObs_) × [nActiveObs_,totalHXSize_active)
    // ------------------------------------------------------------------
    Info << "  Block 2: Pqa..." << endl;
    for (label i = 0; i < nActiveObs_; i++)
    {
        for (label alphaIdx = 0; alphaIdx < alphaRows; alphaIdx++)
        {
            label j      = nActiveObs_ + alphaIdx;
            label phtIdx = i * totalHXSize_active + j;
            HPHt[i][j]   = localPHt_[phtIdx];
            if (mag(HPHt[i][j]) > 1e10)
                HPHt[i][j] = 1e10 * (HPHt[i][j] / mag(HPHt[i][j]));
        }
    }

    // ------------------------------------------------------------------
    // Block 3 (bottom-left): Pαq  [nActiveObs_,totalHXSize_active) × [0,nActiveObs_)
    // ------------------------------------------------------------------
    Info << "  Block 3: Paq..." << endl;
    for (label alphaIdx = 0; alphaIdx < alphaRows; alphaIdx++)
    {
        label i = nActiveObs_ + alphaIdx;
        for (label j = 0; j < nActiveObs_; j++)
        {
            label phtIdx = i * totalHXSize_active + j;
            HPHt[i][j]   = localPHt_[phtIdx];
            if (mag(HPHt[i][j]) > 1e10)
                HPHt[i][j] = 1e10 * (HPHt[i][j] / mag(HPHt[i][j]));
        }
    }

    // ------------------------------------------------------------------
    // Block 4 (bottom-right): Pαα  diagonal only (off-diagonal zeroed in localPHt_)
    // ------------------------------------------------------------------
    Info << "  Block 4: Paa..." << endl;
    for (label alphaIdxI = 0; alphaIdxI < alphaRows; alphaIdxI++)
    {
        label i = nActiveObs_ + alphaIdxI;
        for (label alphaIdxJ = 0; alphaIdxJ < alphaRows; alphaIdxJ++)
        {
            label j      = nActiveObs_ + alphaIdxJ;
            label phtIdx = i * totalHXSize_active + j;
            HPHt[i][j]   = localPHt_[phtIdx];
            if (mag(HPHt[i][j]) > 1e10)
                HPHt[i][j] = 1e10 * (HPHt[i][j] / mag(HPHt[i][j]));
        }
    }

    // R in normalized space: S^-1 diag(sigma_j^2) S^-1 = I, since
    // obsScale = sigma_j. Physical: R_eff = diag(sigma_j^2).
    Info << "  Adding R = I to Pqq diagonal (normalized space)..." << endl;
    {
        const scalar cddNormalized = 1.0;

        scalar hphtDiagMin = GREAT, hphtDiagMax = -GREAT;
        for (label i = 0; i < nActiveObs_; i++)
        {
            HPHt[i][i] += cddNormalized;
            hphtDiagMin = min(hphtDiagMin, HPHt[i][i]);
            hphtDiagMax = max(hphtDiagMax, HPHt[i][i]);
        }
        Info << "  HPHt Pqq diagonal range after R: ["
             << hphtDiagMin << ", " << hphtDiagMax << "]"
             << "  (= SNR^2 + 1, expect roughly O(1) to O(100))" << endl;
    }

    // ------------------------------------------------------------------
    // Add Cpp to Pαα diagonal
    // ------------------------------------------------------------------
    Info << "  Adding Cpp to Paa diagonal..." << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label i        = nActiveObs_ + alphaIdx;

            scalar before = HPHt[i][i];
            scalar cpp    = C_pp_[alphaIdx][alphaIdx];
            HPHt[i][i]   += cpp;

            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": Paa=" << before
                << "  + Cpp=" << cpp
                << "  = " << HPHt[i][i] << endl;
        }
    }

    // ------------------------------------------------------------------
    // Frobenius regularisation
    // ------------------------------------------------------------------
    Info << "  Computing Frobenius norm for regularisation..." << endl;
    scalar frobNorm = 0.0;
    for (label i = 0; i < totalHXSize_active; i++)
        for (label j = 0; j < totalHXSize_active; j++)
            frobNorm += sqr(HPHt[i][j]);
    frobNorm = Foam::sqrt(frobNorm);

    scalar epsilon = 1e-5 * frobNorm / totalHXSize_active;

    Info << "  Frobenius norm: " << frobNorm << endl;
    Info << "  Regularisation epsilon: " << epsilon << endl;
    
    label nEpsQ = 0, nEpsA = 0;
    for (label i = 0; i < totalHXSize_active; i++)
    {
        const scalar before = HPHt[i][i];
        HPHt[i][i] += epsilon;
        if (epsilon > before/10)
        {
            if (i < nActiveObs_) nEpsQ++; else nEpsA++;
        }
    }
    Info << "  epsilon > 0.1*diag on " << nEpsQ << "/" << nActiveObs_
         << " obs rows and " << nEpsA << "/" << alphaRows
         << " param rows (should be 0/0 with per-field sigma)" << endl;

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    scalar maxAsymmetry = 0.0;
    for (label i = 0; i < totalHXSize_active; i++)
        for (label j = i + 1; j < totalHXSize_active; j++)
            maxAsymmetry = max(maxAsymmetry, mag(HPHt[i][j] - HPHt[j][i]));
    Info << "  HPHt max asymmetry: " << maxAsymmetry
        << (maxAsymmetry < 1e-10 ? " (symmetric)" : " (WARNING: asymmetric)") << endl;

    scalar qDiagMin = GREAT, qDiagMax = -GREAT;
    scalar aDiagMin = GREAT, aDiagMax = -GREAT;
    for (label i = 0; i < nActiveObs_; i++)
    {
        qDiagMin = min(qDiagMin, HPHt[i][i]);
        qDiagMax = max(qDiagMax, HPHt[i][i]);
    }
    for (label alphaIdx = 0; alphaIdx < alphaRows; alphaIdx++)
    {
        label i = nActiveObs_ + alphaIdx;
        aDiagMin = min(aDiagMin, HPHt[i][i]);
        aDiagMax = max(aDiagMax, HPHt[i][i]);
    }
    Info << "  HPHt Pqq diagonal range: [" << qDiagMin << ", " << qDiagMax << "]" << endl;
    Info << "  HPHt Paa diagonal range: [" << aDiagMin << ", " << aDiagMax << "]" << endl;

    scalar diagMax = max(qDiagMax, aDiagMax);
    scalar diagMin = min(qDiagMin, aDiagMin);
    scalar condEst = (diagMin > SMALL) ? diagMax / diagMin : GREAT;
    Info << "  HPHt condition number estimate: " << condEst << endl;
    if (condEst > 1e12)
    {
        WarningIn("EnKF_multiVariable_multiDomain::computeKalmanGain")
            << "HPHt appears ill-conditioned (condition ~ " << condEst << ")." << endl;
    }

    label nanCount = 0;
    for (label i = 0; i < totalHXSize_active; i++)
        for (label j = 0; j < totalHXSize_active; j++)
            if (std::isnan(HPHt[i][j]) || std::isinf(HPHt[i][j]))
            {
                nanCount++;
                if (nanCount <= 5)
                    Info << "  ERROR: HPHt[" << i << "][" << j << "] = "
                        << HPHt[i][j] << endl;
            }
    if (nanCount > 0)
    {
        FatalErrorInFunction
            << "HPHt contains " << nanCount << " NaN/Inf values."
            << abort(FatalError);
    }

    Info << "Step 4 complete." << endl;

    // ------------------------------------------------------------------
    // Step 5: Invert HPHt → HPHtInv
    //
    //   HPHt is totalHXSize × totalHXSize (271×271 in current setup).
    //   Inversion by column-wise LUsolve: for each column c, solve
    //     HPHt * x = e_c  (identity column)
    //   and store x as column c of HPHtInv.
    //
    //   Only master computes the inverse; result broadcast to all ranks.
    // ------------------------------------------------------------------
    Info << "\nStep 5: Inverting HPHt (" << totalHXSize_active << " x " << totalHXSize_active << ")..." << endl;


    scalarSquareMatrix HPHtInv(totalHXSize_active, 0.0);

    if (!Pstream::parRun() || Pstream::master())
    {
        // Pre-inversion validation
        bool hasInvalid = false;
        for (label i = 0; i < totalHXSize_active && !hasInvalid; i++)
        {
            for (label j = 0; j < totalHXSize_active && !hasInvalid; j++)
            {
                if (std::isnan(HPHt[i][j]) || std::isinf(HPHt[i][j]))
                {
                    FatalErrorInFunction
                        << "Invalid value in HPHt[" << i << "][" << j
                        << "] = " << HPHt[i][j]
                        << " before inversion." << abort(FatalError);
                }
            }
        }

        label failedCols = 0;

        for (label col = 0; col < totalHXSize_active; col++)
        {
            scalarField identityCol(totalHXSize_active, 0.0);
            identityCol[col] = 1.0;

            // Fresh copy per column — LUsolve modifies the matrix in place
            scalarSquareMatrix HPHtCopy(HPHt);

            bool colOk = true;

            try
            {
                LUsolve(HPHtCopy, identityCol);
            }
            catch (...)
            {
                WarningIn
                (
                    "EnKF_multiVariable_multiDomain::computeKalmanGain"
                )
                    << "LUsolve failed for HPHtInv column " << col
                    << ". Using identity fallback." << endl;
                colOk = false;
                failedCols++;
            }

            if (colOk)
            {
                // Validate and clamp solution
                bool colHasInvalid = false;
                for (label row = 0; row < totalHXSize_active; row++)
                {
                    if (std::isnan(identityCol[row]) || std::isinf(identityCol[row]))
                    {
                        if (!colHasInvalid)
                        {
                            WarningIn
                            (
                                "EnKF_multiVariable_multiDomain::computeKalmanGain"
                            )
                                << "Invalid value in HPHtInv column " << col
                                << " row " << row << ": " << identityCol[row]
                                << ". Replacing with 0." << endl;
                        }
                        identityCol[row] = 0.0;
                        colHasInvalid = true;
                    }

                    // Clamp extreme values
                    if (mag(identityCol[row]) > 1e8)
                    {
                        identityCol[row] =
                            1e8 * (identityCol[row] / mag(identityCol[row]));
                    }
                }

                if (colHasInvalid) failedCols++;
            }
            else
            {
                // Identity fallback
                for (label row = 0; row < totalHXSize_active; row++)
                {
                    identityCol[row] = (row == col) ? 1.0 : 0.0;
                }
            }

            for (label row = 0; row < totalHXSize_active; row++)
            {
                HPHtInv[row][col] = identityCol[row];
            }
        }

        if (failedCols > 0)
        {
            WarningIn
            (
                "EnKF_multiVariable_multiDomain::computeKalmanGain"
            )
                << failedCols << " columns of HPHtInv had issues." << endl;
        }

        // Verify A * A^-1 ≈ I on diagonal (first 5 and last 5 elements)
        Info << "  Inverse accuracy check (A * A^-1 diagonal, should be ~1):" << endl;
        scalar maxDiagError = 0.0;

        for (label i = 0; i < totalHXSize_active; i++)
        {
            scalar diagSum = 0.0;
            for (label k = 0; k < totalHXSize_active; k++)
            {
                diagSum += HPHt[i][k] * HPHtInv[k][i];
            }
            scalar err = mag(diagSum - 1.0);
            maxDiagError = max(maxDiagError, err);

            if (i < 3 || i >= nActiveObs_)
            {
                word label_;
                if (i < nActiveObs_)
                {
                    label_ = "q[" + Foam::name(i) + "]";
                }
                else
                {
                    label alphaIdx  = i - nActiveObs_;
                    label paramI    = alphaIdx / nClusters_;
                    label clusterI  = alphaIdx % nClusters_;
                    label_ = paramNames_[paramI] + "_c" + Foam::name(clusterI);
                }
                Info << "    (A*A^-1)[" << label_ << "] = " << diagSum
                    << "  err=" << err << endl;
            }
        }

        Info << "  Max diagonal error: " << maxDiagError << endl;
        if (maxDiagError > 0.01)
        {
            WarningIn 
            (
                "EnKF_multiVariable_multiDomain::computeKalmanGain"
            )
                << "Large inversion error detected (max diagonal error = "
                << maxDiagError << "). Kalman gain may be unreliable." << endl;
        }
    }

    // Broadcast to all ranks
    if (Pstream::parRun())
    {
        Pstream::scatter(HPHtInv);
    }

    // Post-broadcast validation
    {
        scalar invMin = GREAT, invMax = -GREAT;
        label  nanCount = 0, extremeCount = 0;

        for (label i = 0; i < totalHXSize_active; i++)
        {
            for (label j = 0; j < totalHXSize_active; j++)
            {
                scalar v = HPHtInv[i][j];
                if (std::isnan(v) || std::isinf(v))
                {
                    nanCount++;
                    HPHtInv[i][j] = 0.0;
                }
                else
                {
                    invMin = min(invMin, v);
                    invMax = max(invMax, v);
                    if (mag(v) > 1e8) extremeCount++;
                }
            }
        }

        Info << "  HPHtInv range: [" << invMin << ", " << invMax << "]" << endl;
        Info << "  HPHtInv extreme values (>1e8): " << extremeCount << endl;
        Info << "  HPHtInv NaN/Inf replaced:      " << nanCount << endl;
    }

    Info << "Step 5 complete." << endl;


    // ------------------------------------------------------------------
    // Step 6: Compute compact K_alpha in compressed space
    //
    //   K_α = PHt_alphaRows × HPHtInv
    //
    //   PHt_alphaRows: rows [nActiveObs_, PHtRows) of localPHt_
    //                  size: nModelParams_ × totalHXSize_active
    //
    //   K_α size: nModelParams_ × totalHXSize_active
    //   Flat storage row-major: localK[paramI * totalHXSize_active + j]
    // ------------------------------------------------------------------
    Info << "\nStep 6: Computing compact K_alpha ("
        << alphaRows << " x " << totalHXSize_active << ")..." << endl;

    label localKSize = alphaRows * totalHXSize_active;
    localK.setSize(localKSize);
    localK = 0.0;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label phtRow   = nActiveObs_ + alphaIdx;

            for (label j = 0; j < totalHXSize_active; j++)
            {
                scalar sum = 0.0;

                for (label k = 0; k < totalHXSize_active; k++)
                {
                    label phtIdx = phtRow * totalHXSize_active + k;
                    sum += localPHt_[phtIdx] * HPHtInv[k][j];
                }

                if (mag(sum) > 1e10)
                    sum = 1e10 * (sum / mag(sum));

                if (std::isnan(sum) || std::isinf(sum))
                {
                    WarningIn
                    (
                        "EnKF_multiVariable_multiDomain::computeKalmanGain"
                    )
                        << "Invalid K_alpha[" << paramI << "][" << clusterI
                        << "][" << j << "] = " << sum << ". Setting to 0." << endl;
                    sum = 0.0;
                }

                localK[alphaIdx * totalHXSize_active + j] = sum;
            }
        }
    }


    // ------------------------------------------------------------------
    // Denormalize K_alpha back to physical space
    //
    // K_physical = T · K_normalized · Λ⁻¹
    //
    // where:
    //   T   = diag(alphaMeanScale[i])  — parameter row scaling
    //   Λ   = diag(obsScale[j], alphaMeanScale[j])  — full column scaling
    //
    // Expanding block by block:
    //   obs columns   (j < totalQRows):    K_physical[i,j] = K_norm[i,j] * tᵢ / sⱼ
    //   param columns (j >= totalQRows):   K_physical[i,j] = K_norm[i,j] * tᵢ / tⱼ
    //
    // Note: param columns formula unchanged from before.
    //       obs columns: previously * tᵢ, now * tᵢ / sⱼ
    // ------------------------------------------------------------------
    Info << "\n--- K_alpha denormalization (physical space) ---" << endl;
    Info << "  Formula: K_physical = T · K_normalized · Lambda^-1" << endl;
    Info << "  Obs    columns: K_physical[i,j] = K_norm[i,j] * tᵢ / sⱼ" << endl;
    Info << "  Param  columns: K_physical[i,j] = K_norm[i,j] * tᵢ / tⱼ" << endl;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            scalar tI      = alphaMeanScale[alphaIdx];

            Info << "\n  param=" << paramNames_[paramI]
                << " cluster" << clusterI << "  tI=" << tI << endl;

            scalar maxObsGainBefore = 0.0, maxObsGainAfter = 0.0;
            label  maxObsColBefore  = -1,  maxObsColAfter  = -1;

            // Obs columns: multiply by tᵢ / sⱼ
            // j runs over [0, nActiveObs_) in compressed space
            for (label j = 0; j < nActiveObs_; j++)
            {
                label kIdx = alphaIdx * totalHXSize_active + j;

                if (obsScale[j] < SMALL)
                {
                    FatalErrorInFunction
                        << "obsScale[" << j << "] = " << obsScale[j]
                        << " is zero at K denormalization."
                        << abort(FatalError);
                }

                scalar kBefore = localK[kIdx];
                scalar factor  = tI / obsScale[j];
                localK[kIdx]  *= factor;
                scalar kAfter  = localK[kIdx];

                if (mag(kBefore) > mag(maxObsGainBefore))
                {
                    maxObsGainBefore = kBefore;
                    maxObsColBefore  = j;
                }
                if (mag(kAfter) > mag(maxObsGainAfter))
                {
                    maxObsGainAfter = kAfter;
                    maxObsColAfter  = j;
                }

                if (j < 3 || j >= nActiveObs_ - 3)
                {
                    Info << "    obs col j=" << j
                        << " (globalIdx=" << activeToGlobal_[j] << ")"
                        << ": K_norm=" << kBefore
                        << "  tI=" << tI
                        << "  sJ=" << obsScale[j]
                        << "  factor=tI/sJ=" << factor
                        << "  K_phys=" << kAfter << endl;
                }
            }
        

            Info << "    Obs cols summary:"
                << "  maxGain_before=" << maxObsGainBefore
                << " (col=" << maxObsColBefore << ")"
                << "  maxGain_after=" << maxObsGainAfter
                << " (col=" << maxObsColAfter << ")" << endl;
                
            // ----------------------------------------------------------
            // Param columns: multiply by tᵢ / tⱼ  (unchanged from before)
            // ----------------------------------------------------------
            Info << "    Param cols:" << endl;
            for (label paramJ = 0; paramJ < nModelParams_; paramJ++)
            {
                for (label clusterJ = 0; clusterJ < nClusters_; clusterJ++)
                {
                    label alphaIdxJ = paramJ * nClusters_ + clusterJ;
                    label j         = nActiveObs_ + alphaIdxJ;
                    label kIdx      = alphaIdx * totalHXSize_active + j;

                    if (kIdx >= localK.size())
                    {
                        FatalErrorInFunction
                            << "localK index out of bounds (param denorm): "
                            << "kIdx=" << kIdx
                            << " localK.size()=" << localK.size()
                            << " paramI=" << paramI
                            << " clusterI=" << clusterI
                            << " paramJ=" << paramJ
                            << " clusterJ=" << clusterJ
                            << abort(FatalError);
                    }

                    scalar tJ      = alphaMeanScale[alphaIdxJ];
                    scalar kBefore = localK[kIdx];
                    scalar factor  = tI / tJ;
                    localK[kIdx]  *= factor;
                    scalar kAfter  = localK[kIdx];

                    bool isSelf = (paramI == paramJ && clusterI == clusterJ);
                    Info << "      paramJ=" << paramNames_[paramJ]
                        << " cluster" << clusterJ
                        << ": K_norm=" << kBefore
                        << "  tI=" << tI
                        << "  tJ=" << tJ
                        << "  factor=tI/tJ=" << factor
                        << "  K_phys=" << kAfter
                        << (isSelf ? "  (self-gain)" : "  (cross-gain)")
                        << endl;
                }
            }
        }
    }


    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdxI = paramI * nClusters_ + clusterI;
            for (label paramJ = 0; paramJ < nModelParams_; paramJ++)
            {
                for (label clusterJ = 0; clusterJ < nClusters_; clusterJ++)
                {
                    label alphaIdxJ = paramJ * nClusters_ + clusterJ;
                    if (alphaIdxI != alphaIdxJ)
                    {
                        label j    = nActiveObs_ + alphaIdxJ;
                        label kIdx = alphaIdxI * totalHXSize_active + j;
                        localK[kIdx] = 0.0;
                    }
                }
            }
        }
    }


    // ------------------------------------------------------------------
    // Feature-space localisation — applied after K denormalisation,
    // before cross-gain zeroing.
    //
    // For each (paramI, clusterI) row of K_alpha:
    //   for each active obs column j:
    //     (simI, obsI) = obsIndexMap[activeToGlobal_[j]]
    //     K[alphaIdx, j] *= obsLocWeights_[simI][obsI][clusterI]
    //
    // K param columns (j >= nActiveObs_) are NOT localised —
    // the cross-gain zeroing block handles those.
    // ------------------------------------------------------------------
    if (featureSpaceLocalisation_)
    {
        Info << "\n--- Applying feature-space localisation to K_alpha ---" << endl;

        // Build globalQRow -> (simI, obsI) map once
        // Layout: domains-then-fields, matching createEnsemble q block
        // globalQRow = simOffset + fieldI*nObs + obsI
        // We map each globalQRow to (simI, obsI) ignoring fieldI because
        // obsLocWeights_ is indexed by obsI within sim only —
        // all fields share the same spatial location → same cluster weight
        List<label> globalQRowToSimI(totalQRows, -1);
        List<label> globalQRowToObsI(totalQRows, -1);

        {
            label globalQRow = 0;
            forAll(simulationDirs_, simI)
            {
                const label nObs = nObservationPointsPerSim_[simI];
                forAll(observationFields_, fieldI)
                {
                    for (label obsI = 0; obsI < nObs; obsI++)
                    {
                        globalQRowToSimI[globalQRow] = simI;
                        globalQRowToObsI[globalQRow] = obsI;
                        globalQRow++;
                    }
                }
            }
        }

        // Apply weights
        for (label paramI = 0; paramI < nModelParams_; paramI++)
        {
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                label alphaIdx = paramI * nClusters_ + clusterI;

                scalar minWBefore = GREAT, maxWBefore = -GREAT;
                scalar minWAfter  = GREAT, maxWAfter  = -GREAT;

                for (label j = 0; j < nActiveObs_; j++)
                {
                    label globalQRow = activeToGlobal_[j];
                    label simI       = globalQRowToSimI[globalQRow];
                    label obsI       = globalQRowToObsI[globalQRow];

                    if (simI < 0 || obsI < 0)
                    {
                        FatalErrorInFunction
                            << "Invalid (simI,obsI) mapping for globalQRow="
                            << globalQRow << abort(FatalError);
                    }

                    scalar weight = obsLocWeights_[simI][obsI][clusterI];
                    label  kIdx   = alphaIdx * totalHXSize_active + j;

                    scalar kBefore = localK[kIdx];
                    localK[kIdx]  *= weight;

                    minWBefore = min(minWBefore, mag(kBefore));
                    maxWBefore = max(maxWBefore, mag(kBefore));
                    minWAfter  = min(minWAfter,  mag(localK[kIdx]));
                    maxWAfter  = max(maxWAfter,  mag(localK[kIdx]));
                }

                Info << "  " << paramNames_[paramI] << "_c" << clusterI
                    << ": |K_obs| before=[" << minWBefore << ", " << maxWBefore << "]"
                    << "  after=[" << minWAfter  << ", " << maxWAfter  << "]"
                    << endl;
            }
        }

        Info << "--- Feature-space localisation applied ---" << endl;
    }


    // ----------------------------------------------------------
    // Post-denormalization diagnostics
    // ----------------------------------------------------------
    Info << "\n--- K_alpha AFTER denormalization (physical space) ---" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;

            scalar maxObsGain = 0.0;
            label  maxObsCol  = -1;
            scalar minObsGain = GREAT;
            label  minObsCol  = -1;

            for (label j = 0; j < nActiveObs_; j++)
            {
                scalar g = mag(localK[alphaIdx * totalHXSize_active + j]);
                if (g > maxObsGain) { maxObsGain = g; maxObsCol = j; }
                if (g < minObsGain) { minObsGain = g; minObsCol = j; }
            }

            scalar selfGain = localK[alphaIdx * totalHXSize_active + nActiveObs_ + alphaIdx];

            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": maxObsGain=" << maxObsGain
                << " (col=" << maxObsCol << ")"
                << "  minObsGain=" << minObsGain
                << " (col=" << minObsCol << ")"
                << "  selfGain=" << selfGain
                << "  tI=" << alphaMeanScale[alphaIdx] << endl;

            if (alphaRows > 1)
            {
                Info << "    Cross-gains: ";
                for (label paramJ = 0; paramJ < nModelParams_; paramJ++)
                {
                    for (label clusterJ = 0; clusterJ < nClusters_; clusterJ++)
                    {
                        label alphaIdxJ = paramJ * nClusters_ + clusterJ;
                        if (alphaIdxJ != alphaIdx)
                        {
                            scalar crossGain =
                                localK[alphaIdx * totalHXSize_active + nActiveObs_ + alphaIdxJ];
                            Info << paramNames_[paramJ] << "_c" << clusterJ
                                << "=" << crossGain << "  ";
                        }
                    }
                }
                Info << endl;
            }
        }
    }

    // Global K stats
    {
        scalar kMin = GREAT, kMax = -GREAT, kAvgMag = 0.0;
        label  extremeCount = 0;
        label  localKSize   = nModelParams_ * totalHXSize_active;

        for (label i = 0; i < localKSize; i++)
        {
            scalar g = localK[i];
            kMin     = min(kMin, g);
            kMax     = max(kMax, g);
            kAvgMag += mag(g);
            if (mag(g) > 10.0) extremeCount++;
        }
        kAvgMag /= localKSize;

        Info << "  K_alpha global range:         [" << kMin << ", " << kMax << "]" << endl;
        Info << "  K_alpha avg magnitude:        " << kAvgMag << endl;
        Info << "  K_alpha extreme gains (>10):  " << extremeCount << endl;

        if (extremeCount > 0)
        {
            WarningIn("EnKF_multiVariable_multiDomain::computeKalmanGain")
                << extremeCount << " extreme Kalman gains detected (>10). "
                << "Consider increasing sigmaRel/sigmaAbs or priorUncertainty_."
                << endl;
        }
    }

    Info << "--- K_alpha denormalization complete ---" << endl;
    Info << "\n--- ensemblePerturbation alpha rows AFTER restore (should match pre-normalization) ---" << endl;

    // Restore ensemblePerturbation alpha rows to physical space
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                ensemblePerturbation[ensemblePertRow * totalCols + ensI] *= alphaMeanScale[alphaIdx];
            }
        }
    }

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;

            scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = ensemblePerturbation[ensemblePertRow * totalCols + ensI];
                minPert    = min(minPert, v);
                maxPert    = max(maxPert, v);
                sumAbsPert += mag(v);
            }
            Info << "  " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": min=" << minPert
                << "  max=" << maxPert
                << "  avgMag=" << sumAbsPert / nValid_ << endl;
        }
    }

    // ------------------------------------------------------------------
    // Restore ensemblePerturbation q rows to physical space
    // Analogous to alpha row restore already done above.
    // Must multiply back by obsScale to undo the normalization from step 3.
    // ------------------------------------------------------------------
    Info << "\n--- Restoring ensemblePerturbation q rows to physical space ---" << endl;
    Info << "  qRowOffset=" << qRowOffset
        << "  totalQRows=" << totalQRows
        << "  totalCols=" << totalCols << endl;

    // Diagnostics BEFORE restore
    Info << "  ensemblePerturbation q block BEFORE restore (normalized):" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label ensemblePertRow = qRowOffset + globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        label idx = ensemblePertRow * totalCols + ensI;

                        if (idx >= ensemblePerturbation.size())
                        {
                            FatalErrorInFunction
                                << "ensemblePerturbation index out of bounds "
                                << "(BEFORE restore): idx=" << idx
                                << " size=" << ensemblePerturbation.size()
                                << " ensemblePertRow=" << ensemblePertRow
                                << " simI=" << simI
                                << " fieldI=" << fieldI
                                << " obsI=" << obsI
                                << " ensI=" << ensI
                                << abort(FatalError);
                        }

                        scalar v    = ensemblePerturbation[idx];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_) << endl;

                globalQRow += nObs;
            }
        }
    }

    // Apply restore — active obs only
    for (label activeIdx = 0; activeIdx < nActiveObs_; activeIdx++)
    {
        label globalQRow      = activeToGlobal_[activeIdx];
        label ensemblePertRow = qRowOffset + globalQRow;

        if (obsScale[activeIdx] < SMALL)
        {
            FatalErrorInFunction
                << "obsScale[" << activeIdx << "] = " << obsScale[activeIdx]
                << " is zero at ensemblePerturbation restore."
                << abort(FatalError);
        }

        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label idx = ensemblePertRow * totalCols + ensI;
            ensemblePerturbation[idx] *= obsScale[activeIdx];
        }
    }
    Info << "  Restore applied to " << nActiveObs_ << " active q rows." << endl;
        
    // Diagnostics AFTER restore
    Info << "  ensemblePerturbation q block AFTER restore (should match pre-normalization):" << endl;
    {
        label globalQRow = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label ensemblePertRow = qRowOffset + globalQRow + obsI;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        scalar v = ensemblePerturbation[ensemblePertRow * totalCols + ensI];
                        minPert    = min(minPert, v);
                        maxPert    = max(maxPert, v);
                        sumAbsPert += mag(v);
                    }
                }
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << ": min=" << minPert
                    << "  max=" << maxPert
                    << "  avgMag=" << sumAbsPert / (nObs * nValid_) << endl;

                // Zero-mean check — must still hold after restore
                scalar meanPert = 0.0;
                label  firstRow = qRowOffset + globalQRow;
                for (label ensI = 0; ensI < nValid_; ensI++)
                {
                    meanPert += ensemblePerturbation[firstRow * totalCols + ensI];
                }
                meanPert /= nValid_;
                Info << "    " << simName << "/" << observationFields_[fieldI]
                    << " first row mean after restore: " << meanPert
                    << " (should be ~0)" << endl;

                globalQRow += nObs;
            }
        }
    }

    // Verify alpha block still correct after q restore
    Info << "  ensemblePerturbation alpha block after q restore (should be unchanged):" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx        = paramI * nClusters_ + clusterI;
            label ensemblePertRow = alphaRowOffset + alphaIdx;
            scalar minPert = GREAT, maxPert = -GREAT, sumAbsPert = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = ensemblePerturbation[ensemblePertRow * totalCols + ensI];
                minPert    = min(minPert, v);
                maxPert    = max(maxPert, v);
                sumAbsPert += mag(v);
            }
            Info << "    " << paramNames_[paramI]
                << " cluster" << clusterI
                << ": min=" << minPert
                << "  max=" << maxPert
                << "  avgMag=" << sumAbsPert / nValid_
                << "  (should match pre-normalization physical values)" << endl;
        }
    }

    // Final cross-check: compare AFTER restore vs original
    // pre-normalization values stored in computeEnsemblePerturbations
    // The avgMag here should match the "q" avgMag printed by
    // computeEnsemblePerturbations at the start of the run.
    Info << "  NOTE: avgMag above should match 'q' block avgMag from" << endl;
    Info << "        computeEnsemblePerturbations output at start of iteration." << endl;

    Info << "--- ensemblePerturbation q restore complete ---" << endl;

    Info << "=== computeKalmanGain complete ===" << endl;
}


void EnKF_multiVariable_multiDomain_clustering::updateEnsemble
(
    scalarField& ensemble,
    const scalarField& localK
)
{
    Info << "\n=== updateEnsemble (multi-variable multi-domain clustering) ===" << endl;

    // ------------------------------------------------------------------
    // Block offsets — must match createEnsemble exactly
    // ------------------------------------------------------------------
    const label nFields   = observationFields_.size();
    const label totalCols = nValid_;

    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    label totalQRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalQRows += nObservationPointsPerSim_[simI] * nFields;
    }

    label alphaRows          = nModelParams_ * nClusters_;
    label alphaRowOffset     = totalPhiRows;
    label totalHXSize_active = nActiveObs_ + alphaRows;

    Info << "  nFields=" << nFields
        << "  totalQRows=" << totalQRows
        << "  nActiveObs=" << nActiveObs_
        << "  nModelParams_=" << nModelParams_
        << "  nClusters_=" << nClusters_
        << "  alphaRows=" << alphaRows
        << "  totalHXSize_active=" << totalHXSize_active
        << "  nValid_=" << nValid_ << endl;

    // ------------------------------------------------------------------
    // Step 1: Build innovation vector (compressed obs space)
    //
    //   innovation size: totalHXSize_active * nValid_
    //   Layout:
    //     rows [0,         nActiveObs_)    <- active obs innovations
    //     rows [nActiveObs_, totalHXSize_active) <- param-cluster innovations
    //
    //   obs innovation[activeIdx][ensI] =
    //       perturbedObs[activeToGlobal_[activeIdx]] - modelObs[...]
    //   param innovation[nActiveObs_+alphaIdx][ensI] =
    //       paramPrior[(paramI*nClusters_+clusterI)*nValid_+ensI]
    //       - globalHx_[(totalQRows+alphaIdx)*nValid_+ensI]
    // ------------------------------------------------------------------
    Info << "\nStep 1: Building innovation vector (compressed)..." << endl;

    scalarField innovation(totalHXSize_active * nValid_, 0.0);

    // --- 1a: Active observation innovations ---
    for (label activeIdx = 0; activeIdx < nActiveObs_; activeIdx++)
    {
        label globalQRow = activeToGlobal_[activeIdx];

        for (label ensI = 0; ensI < nValid_; ensI++)
        {
            label globalIdx = globalQRow * nValid_ + ensI;

            scalar pertObs  = perturbedObsInBIA_[globalIdx];
            scalar modelObs = globalHx_[globalIdx];
            scalar innov    = pertObs - modelObs;

            if (std::isnan(innov) || std::isinf(innov))
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::updateEnsemble")
                    << "Invalid obs innovation at activeIdx=" << activeIdx
                    << " globalQRow=" << globalQRow
                    << " ensI=" << ensI << ": " << innov
                    << ". Setting to 0." << endl;
                innov = 0.0;
            }

            innovation[activeIdx * nValid_ + ensI] = innov;
        }
    }

    // --- 1b: Parameter-cluster innovations ---
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx  = paramI * nClusters_ + clusterI;
            label innovRow  = nActiveObs_ + alphaIdx;

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label priorIdx   = alphaIdx * nValid_ + ensI;
                label globalHxIdx = (totalQRows + alphaIdx) * nValid_ + ensI;

                scalar prior   = paramPrior[priorIdx];
                scalar current = globalHx_[globalHxIdx];
                scalar innov   = prior - current;

                if (std::isnan(innov) || std::isinf(innov))
                {
                    WarningIn("EnKF_multiVariable_multiDomain_clustering::updateEnsemble")
                        << "Invalid param innovation at paramI=" << paramI
                        << " clusterI=" << clusterI
                        << " ensI=" << ensI << ": " << innov
                        << ". Setting to 0." << endl;
                    innov = 0.0;
                }

                innovation[innovRow * nValid_ + ensI] = innov;
            }
        }
    }

    // --- Innovation diagnostics ---
    Info << "  Active observation innovations (per sim, per field):" << endl;
    {
        label globalQRowBase = 0;
        forAll(simulationDirs_, simI)
        {
            const label nObs    = nObservationPointsPerSim_[simI];
            const word  simName = simulationDirs_[simI].name();

            forAll(observationFields_, fieldI)
            {
                scalar minV = GREAT, maxV = -GREAT, sumV = 0.0;
                label  nActive = 0;
                label  rowBase = globalQRowBase + fieldI * nObs;

                for (label obsI = 0; obsI < nObs; obsI++)
                {
                    label globalIdx = rowBase + obsI;
                    label activeIdx = globalToActive_[globalIdx];
                    if (activeIdx < 0) continue;

                    for (label ensI = 0; ensI < nValid_; ensI++)
                    {
                        scalar v = innovation[activeIdx * nValid_ + ensI];
                        minV = min(minV, v);
                        maxV = max(maxV, v);
                        sumV += v;
                    }
                    nActive++;
                }

                Info << "    " << simName << " / " << observationFields_[fieldI]
                    << ": mean=" << sumV / max(nActive * nValid_, 1)
                    << "  range=[" << minV << ", " << maxV << "]"
                    << "  nActive=" << nActive << endl;
            }
            globalQRowBase += nObs * nFields;  // advance per sim, outside fieldI loop
        }
    }

    Info << "  Parameter-cluster innovations:" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            label innovRow = nActiveObs_ + alphaIdx;
            scalar minV = GREAT, maxV = -GREAT, sumV = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar v = innovation[innovRow * nValid_ + ensI];
                minV = min(minV, v);
                maxV = max(maxV, v);
                sumV += v;
            }
            Info << "    " << paramNames_[paramI] << "_c" << clusterI
                << ": mean=" << sumV / nValid_
                << "  range=[" << minV << ", " << maxV << "]" << endl;
        }
    }

    Info << "Step 1 complete." << endl;

    // ------------------------------------------------------------------
    // Step 2: Compute parameter-cluster updates
    //
    //   paramUpdates layout: alphaRows * nValid_
    //   flat index: alphaIdx * nValid_ + ensI
    //
    //   update[alphaIdx][ensI] = sum_j( K[alphaIdx][j] * innovation[j][ensI] )
    //
    //   K indexing: localK[alphaIdx * totalHXSize_active + j]
    // ------------------------------------------------------------------
    Info << "\nStep 2: Computing parameter-cluster updates..." << endl;
    Info << "  DEBUG: nModelParams_=" << nModelParams_
         << "  nClusters_=" << nClusters_
         << "  alphaRows=" << alphaRows
         << "  nValid_=" << nValid_
         << "  totalHXSize_active=" << totalHXSize_active
         << "  totalQRows=" << totalQRows
         << "  simulationDirs_.size()=" << simulationDirs_.size()
         << "  nFields=" << nFields << endl;

    Info << "  DEBUG: innovation.size()=" << innovation.size()
         << "  localK.size()=" << localK.size()
         << "  expected localK size=" << (alphaRows * totalHXSize_active) << endl;

    if (localK.size() != alphaRows * totalHXSize_active)
    {
        FatalErrorInFunction
            << "localK size mismatch: got " << localK.size()
            << " expected " << alphaRows * totalHXSize_active
            << abort(FatalError);
    }

    scalarField paramUpdates(alphaRows * nValid_, 0.0);

    const label topN = 10;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;

            // --------------------------------------------------------------
            // Per-ensemble contribution trackers
            // --------------------------------------------------------------
            label sfSize = simulationDirs_.size() * nFields;
            List<scalarField> simFieldContrib(nValid_);
            List<scalarField> alphaContrib(nValid_);

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                simFieldContrib[ensI].setSize(sfSize, 0.0);
                alphaContrib[ensI].setSize(alphaRows, 0.0);
            }

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                scalar update = 0.0;

                // --- Observation contributions ---
                label globalQRowBase = 0;
                forAll(simulationDirs_, simI)
                {
                    const label nObs = nObservationPointsPerSim_[simI];
                    forAll(observationFields_, fieldI)
                    {
                        scalar fieldContrib = 0.0;
                        label rowBase = globalQRowBase + fieldI * nObs;

                        for (label obsI = 0; obsI < nObs; obsI++)
                        {
                            label globalIdx = rowBase + obsI;
                            label activeIdx = globalToActive_[globalIdx];

                            if (activeIdx < 0) continue;

                            label kIdx     = alphaIdx * totalHXSize_active + activeIdx;
                            label innovIdx = activeIdx * nValid_ + ensI;

                            if (kIdx >= localK.size())
                            {
                                FatalErrorInFunction
                                    << "localK index out of bounds: kIdx=" << kIdx
                                    << " activeIdx=" << activeIdx
                                    << abort(FatalError);
                            }
                            if (innovIdx >= innovation.size())
                            {
                                FatalErrorInFunction
                                    << "innovation index out of bounds: innovIdx=" << innovIdx
                                    << " activeIdx=" << activeIdx
                                    << abort(FatalError);
                            }

                            scalar contrib = localK[kIdx] * innovation[innovIdx];
                            fieldContrib  += contrib;
                            update        += contrib;
                        }

                        label sfIdx = simI * nFields + fieldI;
                        simFieldContrib[ensI][sfIdx] = fieldContrib;
                    }
                    globalQRowBase += nObs * nFields;  // advance per sim, outside fieldI loop
                }

                // --- Parameter-cluster contributions ---
                for (label paramK = 0; paramK < nModelParams_; paramK++)
                {
                    for (label clusterK = 0; clusterK < nClusters_; clusterK++)
                    {
                        label alphaIdxK = paramK * nClusters_ + clusterK;
                        label j         = nActiveObs_ + alphaIdxK;
                        label kIdx      = alphaIdx * totalHXSize_active + j;
                        label innovIdx  = j * nValid_ + ensI;

                        if (innovIdx >= innovation.size())
                        {
                            FatalErrorInFunction
                                << "innovation index out of bounds (param): innovIdx="
                                << innovIdx
                                << " innovation.size()=" << innovation.size()
                                << abort(FatalError);
                        }

                        scalar contrib = localK[kIdx] * innovation[innovIdx];
                        alphaContrib[ensI][alphaIdxK] = contrib;
                        update += contrib;
                    }
                }

                paramUpdates[alphaIdx * nValid_ + ensI] = update;
            }

            // ----------------------------------------------------------
            // Diagnostics — avg source breakdown
            // ----------------------------------------------------------
            scalar avgUpdate = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
                avgUpdate += paramUpdates[alphaIdx * nValid_ + ensI];
            avgUpdate /= nValid_;

            Info << "\n  Parameter: " << paramNames_[paramI]
                 << "_c" << clusterI
                 << "  avg update: " << avgUpdate << endl;

            Info << "    Obs contributions (avg):" << endl;
            forAll(simulationDirs_, simI)
            {
                forAll(observationFields_, fieldI)
                {
                    scalar avg = 0.0;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                        avg += simFieldContrib[ensI][simI * nFields + fieldI];
                    avg /= nValid_;
                    Info << "      " << simulationDirs_[simI].name()
                         << "/" << observationFields_[fieldI] << ": " << avg << endl;
                }
            }

            Info << "    Alpha coupling (avg):" << endl;
            for (label paramK = 0; paramK < nModelParams_; paramK++)
            {
                for (label clusterK = 0; clusterK < nClusters_; clusterK++)
                {
                    label alphaIdxK = paramK * nClusters_ + clusterK;
                    scalar avg = 0.0;
                    for (label ensI = 0; ensI < nValid_; ensI++)
                        avg += alphaContrib[ensI][alphaIdxK];
                    avg /= nValid_;
                    if (mag(avg) > SMALL)
                    {
                        Info << "      " << paramNames_[paramK]
                             << "_c" << clusterK << ": " << avg
                             << (alphaIdxK == alphaIdx ? " (self)" : " (cross)") << endl;
                    }
                }
            }

            // Top-N contributions
            Info << "    Top-" << topN << " contributions:" << endl;
            scalarField contribVals(totalHXSize_active, 0.0);
            for (label j = 0; j < totalHXSize_active; j++)
            {
                scalar avgInnov = 0.0;
                for (label ensI = 0; ensI < nValid_; ensI++)
                    avgInnov += innovation[j * nValid_ + ensI];
                avgInnov /= nValid_;

                scalar contrib = localK[alphaIdx * totalHXSize_active + j] * avgInnov;
                contribVals[j] = contrib;
            }

            for (label n = 0; n < topN; n++)
            {
                scalar maxMag = -GREAT;
                label  maxJ   = -1;
                for (label j = 0; j < totalHXSize_active; j++)
                {
                    if (mag(contribVals[j]) > maxMag)
                    { maxMag = mag(contribVals[j]); maxJ = j; }
                }
                if (maxJ < 0 || maxMag < SMALL) break;

                if (maxJ < nActiveObs_)
                {
                    label globalJ = activeToGlobal_[maxJ];
                    label gRow    = 0;
                    bool  found   = false;
                    forAll(simulationDirs_, simI)
                    {
                        const label nObs = nObservationPointsPerSim_[simI];
                        forAll(observationFields_, fieldI)
                        {
                            if (globalJ >= gRow && globalJ < gRow + nObs)
                            {
                                Info << "      #" << (n+1) << " "
                                    << simulationDirs_[simI].name()
                                    << "/" << observationFields_[fieldI]
                                    << " obs[" << (globalJ - gRow) << "]"
                                    << " (globalIdx=" << globalJ << ")"
                                    << " = " << contribVals[maxJ]
                                    << " (K=" << localK[alphaIdx * totalHXSize_active + maxJ]
                                    << ")" << endl;
                                found = true; break;
                            }
                            gRow += nObs;
                        }
                        if (found) break;
                    }
                }
                else
                {
                    label alphaIdxK = maxJ - nActiveObs_;
                    label paramK    = alphaIdxK / nClusters_;
                    label clusterK  = alphaIdxK % nClusters_;
                    Info << "      #" << (n+1) << " param/"
                        << paramNames_[paramK] << "_c" << clusterK
                        << " = " << contribVals[maxJ]
                        << " (K=" << localK[alphaIdx * totalHXSize_active + maxJ]
                        << ")" << endl;
                }
                contribVals[maxJ] = 0.0;
            }
        }
    }

    // Synchronise
    if (Pstream::parRun())
    {
        reduce(paramUpdates, sumOp<scalarField>());
        paramUpdates /= Pstream::nProcs();
    }

    Info << "\nStep 2 complete." << endl;

    // ------------------------------------------------------------------
    // Step 3: Apply updates to ensemble with hard bounds (EnKFDict min/max)
    //         and covariance inflation. Bounds are enforced after the
    //         update AND again after inflation.
    //   Arrays are flat of size nModelParams_ * nClusters_,
    //   indexed as paramI * nClusters_ + clusterI  (= alphaIdx).
    // ------------------------------------------------------------------
    Info << "\nStep 3: Applying updates to ensemble..." << endl;

    const scalarField& paramBoundsMin = paramMin_;
    const scalarField& paramBoundsMax = paramMax_;

    // Log bounds for verification
    Info << "  Parameter bounds per cluster:" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        Info << "  " << paramNames_[paramI] << ":" << endl;
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label flatIdx = paramI * nClusters_ + clusterI;
            Info << "    c" << clusterI
                 << ": default=" << paramDefaults_[flatIdx]
                 << "  min=" << paramBoundsMin[flatIdx]
                 << "  max=" << paramBoundsMax[flatIdx] << endl;
        }
    }

    Random masterRand(clock::getTime());
    label  numResamplings = 0;



    // Bound hits per (param, cluster) this cycle, update + inflation
    labelList nBounded(alphaRows, 0);

    // Enforce the hard bounds on one value, log it, count it
    auto enforceBounds = [&]
    (
        scalar& v,
        const label paramI,
        const label clusterI,
        const label ensI,
        const char* stage
    )
    {
        const label  alphaIdx = paramI * nClusters_ + clusterI;
        const scalar lo = paramBoundsMin[alphaIdx];
        const scalar hi = paramBoundsMax[alphaIdx];

        if (v >= lo && v <= hi)
        {
            return;
        }

        scalar absN = 0.0;
        if (!Pstream::parRun() || Pstream::master())
        {
            absN = mag(masterRand.scalarNormal());
        }
        if (Pstream::parRun())
        {
            reduce(absN, maxOp<scalar>());   // absN >= 0, so maxOp is exact
        }

        const scalar before = v;
        v = boundParameter(v, lo, hi, paramPerturbRange_[alphaIdx], absN);

        nBounded[alphaIdx]++;
        numResamplings++;

        Info << "  [BOUNDED, " << stage << "] " << paramNames_[paramI]
                << "_c" << clusterI << " member_" << validMembers_[ensI] << ": " << before
             << " -> " << v << "  [" << lo << ", " << hi << "]" << endl;
    };

    // Store pre-update values for per-member reporting
    scalarField paramBeforeValues(alphaRows * nValid_, 0.0);
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label ensRow = alphaRowOffset + alphaIdx;
                label idx    = ensRow * totalCols + ensI;
                paramBeforeValues[alphaIdx * nValid_ + ensI] = ensemble[idx];
            }
        }
    }

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx  = paramI * nClusters_ + clusterI;
            scalar beforeSum = 0.0, afterSum = 0.0;

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label ensRow = alphaRowOffset + alphaIdx;
                label idx    = ensRow * totalCols + ensI;

                scalar before = paramBeforeValues[alphaIdx * nValid_ + ensI];
                beforeSum += before;

                ensemble[idx] += paramUpdates[alphaIdx * nValid_ + ensI];

                enforceBounds(ensemble[idx], paramI, clusterI, ensI, "update");

                afterSum += ensemble[idx];
            }

            // Covariance inflation around updated mean
            scalar afterMean = afterSum / nValid_;
            if (inflationFactor_ > 1.0)
            {
                scalar inflatedSum = 0.0;
                for (label ensI = 0; ensI < nValid_; ensI++)
                {
                    label ensRow = alphaRowOffset + alphaIdx;
                    label idx    = ensRow * totalCols + ensI;
                    ensemble[idx] = afterMean
                                  + inflationFactor_ * (ensemble[idx] - afterMean);

                    enforceBounds(ensemble[idx], paramI, clusterI, ensI, "inflation");

                    inflatedSum  += ensemble[idx];
                }
                afterMean = inflatedSum / nValid_;
            }

        }
    }

    Info << "  Total resamplings: " << numResamplings << endl;

    forAll(nBounded, alphaIdx)
    {
        if (nBounded[alphaIdx] > 0.3*nValid_)
        {
            WarningInFunction
                << paramNames_[alphaIdx / nClusters_] << "_c"
                << alphaIdx % nClusters_ << ": " << nBounded[alphaIdx]
                << " bound hits for " << nValid_ << " members this cycle "
                << "(update + inflation). The bounds or the prior may be "
                << "fighting the data." << endl;
        }
    }

    
    // ------------------------------------------------------------------
    // Step 4: Final parameter statistics
    // ------------------------------------------------------------------
    Info << "\nStep 4: Final parameter statistics:" << endl;

    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            scalar minV = GREAT, maxV = -GREAT, sumV = 0.0;

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label ensRow = alphaRowOffset + alphaIdx;
                label idx    = ensRow * totalCols + ensI;
                scalar v = ensemble[idx];
                minV  = min(minV, v);
                maxV  = max(maxV, v);
                sumV += v;
            }

            scalar meanV  = sumV / nValid_;
            scalar varSum = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label ensRow = alphaRowOffset + alphaIdx;
                label idx    = ensRow * totalCols + ensI;
                varSum += sqr(ensemble[idx] - meanV);
            }
            scalar stdDev = Foam::sqrt(varSum / nValid_);

            Info << "  " << paramNames_[paramI] << "_c" << clusterI
                 << ": mean=" << meanV
                 << "  std=" << stdDev
                 << "  range=[" << minV << ", " << maxV << "]"
                 << "  CoV=" << stdDev / max(mag(meanV), SMALL) << endl;
        }
    }

    Info << "\n=== updateEnsemble complete ===" << endl;
}


void EnKF_multiVariable_multiDomain_clustering::updateMemberParameters
(
    const scalarField& ensemble
)
{
    Info << "\n=== updateMemberParameters (multi-variable multi-domain clustering) ===" << endl;

    // ------------------------------------------------------------------
    // Block offset — must match createEnsemble and updateEnsemble
    // ------------------------------------------------------------------
    const label nFields = observationFields_.size();

    label totalPhiRows = 0;
    forAll(simulationDirs_, simI)
    {
        totalPhiRows += domainCells_[simI] * nFields;
    }

    label alphaRowOffset = totalPhiRows;
    label totalCols      = nValid_;

    Info << "  alphaRowOffset=" << alphaRowOffset
         << "  nModelParams_=" << nModelParams_
         << "  nClusters_=" << nClusters_
         << "  nEnsembles_=" << nEnsembles_
         << "  nDomains=" << simulationDirs_.size() << endl;

    // ------------------------------------------------------------------
    // Formatting helper — builds "paramName<spaces>(v0 v1 ... vN);"
    // Column width derived from the longest parameter name (no hardcoded list).
    // ------------------------------------------------------------------
    label maxNameLen = 0;
    forAll(paramNames_, i)
    {
        maxNameLen = max(maxNameLen, label(std::string(paramNames_[i]).length()));
    }
    const label nameColWidth = maxNameLen + 4;   // +4 spaces of breathing room

    auto formatLine =
    [&](const word& name, const scalarField& vals, label nVals) -> std::string
    {
        std::string nameStr(name);
        std::ostringstream oss;
        oss << nameStr << std::string(nameColWidth - nameStr.length(), ' ');
        if (nVals == 1)
        {
            // Global parameter: plain scalar, readable by standard models
            oss << std::fixed << std::setprecision(8) << vals[0] << ";";
        }
        else
        {
            oss << "(";
            for (label i = 0; i < nVals; i++)
            {
                if (i > 0) oss << " ";
                oss << std::fixed << std::setprecision(8) << vals[i];
            }
            oss << ");";
        }
        return oss.str();
    };

    // ------------------------------------------------------------------
    // Find-and-replace helper — finds "name" in fileContent, then
    // replaces everything up to and including the next ";" with newLine.
    // Handles both scalar format  "name   value;"
    // and list format             "name   (v0 v1 ... vN);"
    // ------------------------------------------------------------------
    auto findAndReplace =
    [&]
    (
        std::string&       fileContent,
        const word&        paramName,
        const std::string& newLine,
        label              memberI,
        label              simI
    ) -> bool
    {
        const std::string searchName(paramName);
        const label       searchLen = searchName.size();

        std::size_t pos = 0;
        bool found = false;

        while ((pos = fileContent.find(searchName, pos)) != std::string::npos)
        {
            // Check left boundary — must be start-of-line, space, or '{'
            if (pos == 0
             || std::isspace(fileContent[pos-1])
             || fileContent[pos-1] == '{')
            {
                // Check right boundary — must be space or tab after name
                if (pos + searchLen < fileContent.length()
                 && (std::isspace(fileContent[pos + searchLen])
                  || fileContent[pos + searchLen] == '\t'))
                {
                    std::size_t semiPos = fileContent.find(";", pos);
                    if (semiPos != std::string::npos)
                    {
                        std::string oldLine =
                            fileContent.substr(pos, semiPos - pos + 1);

                        fileContent.replace(pos, oldLine.length(), newLine);

                        Info << "    [sim" << (simI+1)
                             << " member_" << memberI << "] "
                             << paramName << ": "
                             << oldLine << " -> " << newLine << endl;

                        found = true;
                        break;
                    }
                }
            }
            pos += searchLen;
        }

        return found;
    };

    // ------------------------------------------------------------------
    // Counters for final summary
    // ------------------------------------------------------------------
    label totalParamFileAttempted  = 0;
    label totalParamFileSuccessful = 0;

    Info << "  DIAG alphaRowOffset=" << alphaRowOffset << endl;
    Info << "  DIAG ensemble.size()=" << ensemble.size() << endl;

    // ------------------------------------------------------------------
    // Main loop: for each ensemble member, for each domain
    // ------------------------------------------------------------------
    for (label ensI = 0; ensI < nValid_; ensI++)   // ensI = analysis column
    {
        const label memberI = validMembers_[ensI];
        Info << "\n  member_" << memberI << ":" << endl;

        // Extract this member's parameter values for all (paramI, clusterI)
        // memberParams layout: paramI * nClusters_ + clusterI
        scalarField memberParams(nModelParams_ * nClusters_);
        for (label paramI = 0; paramI < nModelParams_; paramI++)
        {
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                label alphaIdx = paramI * nClusters_ + clusterI;
                label idx      = (alphaRowOffset + alphaIdx) * totalCols + ensI;
                memberParams[alphaIdx] = ensemble[idx];
            }
        }

        // Print parameter values for this member
        for (label paramI = 0; paramI < nModelParams_; paramI++)
        {
            Info << "    " << paramNames_[paramI] << ":";
            for (label clusterI = 0; clusterI < nClusters_; clusterI++)
            {
                label alphaIdx = paramI * nClusters_ + clusterI;
                Info << "  c" << clusterI << "=" << memberParams[alphaIdx];
            }
            Info << endl;
        }

        // --------------------------------------------------------------
        // Update each simulation domain
        // --------------------------------------------------------------
        forAll(simulationDirs_, simI)
        {
            const word simName = simulationDirs_[simI].name();
            fileName memberDir = simulationDirs_[simI]
                               / "EnKFmembers"
                               / ("member_" + Foam::name(memberI));

            Info << "  " << simName << " / member_" << memberI << ":" << endl;

            if (!isDir(memberDir))
            {
                WarningIn
                (
                    "EnKF_multiVariable_multiDomain_clustering::updateMemberParameters"
                )
                    << "Member directory not found: " << memberDir << endl;
                continue;
            }

            // ----------------------------------------------------------
            // turbulenceProperties
            // ----------------------------------------------------------
            fileName paramFile = memberDir/parameterFile_;
            totalParamFileAttempted++;

            if (!isFile(paramFile))
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::updateMemberParameters")
                    << "Parameter file not found: " << paramFile
                    << endl;
                continue;
            }

            std::ifstream inFile(paramFile);
            if (!inFile.is_open())
            {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::updateMemberParameters")
                    << "Cannot open: " << paramFile << endl;
                continue;
            }

            std::string fileContent
            (
                (std::istreambuf_iterator<char>(inFile)),
                std::istreambuf_iterator<char>()
            );
            inFile.close();

            bool fileModified = false;

            for (label paramI = 0; paramI < nModelParams_; paramI++)
            {
                // Build the list of nClusters_ values for this parameter
                scalarField clusterVals(nClusters_);
                for (label clusterI = 0; clusterI < nClusters_; clusterI++)
                {
                    label alphaIdx    = paramI * nClusters_ + clusterI;
                    clusterVals[clusterI] = memberParams[alphaIdx];
                }

                std::string newLine =
                    formatLine(paramNames_[paramI], clusterVals, nClusters_);

                bool ok = findAndReplace
                (
                    fileContent,
                    paramNames_[paramI],
                    newLine,
                    memberI,
                    simI
                );

                if (ok)
                {
                    fileModified = true;
                }
                else
                {
                    WarningIn
                    (
                        "EnKF_multiVariable_multiDomain_clustering::updateMemberParameters"
                    )
                        << "Could not find " << paramNames_[paramI]
                        << " in " << paramFile << endl;
                }
            }

            if (fileModified)
            {
                std::ofstream outFile(paramFile);
                if (outFile.is_open())
                {
                outFile << fileContent;
                outFile.close();
                totalParamFileSuccessful++;
                Info << "    Parameter file written ("
                    << nModelParams_ << " params x "
                    << nClusters_ << " clusters)." << endl;
                }
                else
                {
                WarningIn("EnKF_multiVariable_multiDomain_clustering::updateMemberParameters")
                    << "Cannot write: " << paramFile << endl;
                }
            }
            else
            {
            WarningIn("EnKF_multiVariable_multiDomain_clustering::updateMemberParameters")
                << "No parameters updated in " << paramFile << endl;
            }
        }

        Info << "  member_" << memberI << " complete across "
             << simulationDirs_.size() << " domains." << endl;
    }

    // ------------------------------------------------------------------
    // Failed members (excluded from this analysis): new parameters drawn
    // from the analysis ensemble of the valid members,
    //     v = mean + A w / sqrt(nValid - 1),   w ~ N(0, I)
    // i.e. a sample of N(mean, C_analysis), keeping the correlations
    // between parameters; then the EnKFDict hard bounds.
    // ------------------------------------------------------------------
    if (failedMembers_.size())
    {
        const label  alphaRows = nModelParams_*nClusters_;
        const label  nFailed   = failedMembers_.size();
        const label  stride    = nValid_ + alphaRows;
        const scalar c         = 1.0/Foam::sqrt(scalar(nValid_ - 1));

        Info << "\n  Replacing parameters of " << nFailed
             << " failed member(s):" << endl;

        scalarField meanA(alphaRows, 0.0);
        for (label a = 0; a < alphaRows; a++)
        {
            for (label k = 0; k < nValid_; k++)
            {
                meanA[a] += ensemble[(alphaRowOffset + a)*totalCols + k];
            }
            meanA[a] /= nValid_;
        }

        // Draw on the master, broadcast: identical values on every rank
        scalarField draws(nFailed*stride, 0.0);
        if (Pstream::master())
        {
            Random rnd(clock::getTime() + 7919*iterationCount_);
            forAll(draws, i)
            {
                draws[i] = rnd.scalarNormal();
            }
        }
        Pstream::scatter(draws);

        std::ofstream failLog;
        if (Pstream::master())
        {
            const fileName failLogFile =
                mesh_.time().rootPath()/mesh_.time().caseName()
               /"EnKFmemberFailures.log";
            failLog.open(failLogFile.c_str(), std::ios::app);
        }

        forAll(failedMembers_, f)
        {
            const label memberI = failedMembers_[f];
            const label base    = f*stride;

            scalarField memberParams(alphaRows);
            for (label a = 0; a < alphaRows; a++)
            {
                scalar v = meanA[a];
                for (label k = 0; k < nValid_; k++)
                {
                    v += c
                       *(ensemble[(alphaRowOffset + a)*totalCols + k] - meanA[a])
                       *draws[base + k];
                }

                memberParams[a] = boundParameter
                (
                    v,
                    paramMin_[a],
                    paramMax_[a],
                    paramPerturbRange_[a],
                    mag(draws[base + nValid_ + a])
                );
            }

            Info << "  member_" << memberI << " (" << failReasons_[f].c_str()
                 << "):" << endl;
            for (label a = 0; a < alphaRows; a++)
            {
                Info << "    " << paramNames_[a/nClusters_] << "_c"
                     << a % nClusters_ << " = " << memberParams[a]
                     << "  (analysis mean " << meanA[a] << ")" << endl;
            }

            forAll(simulationDirs_, simI)
            {
                const fileName paramFile =
                    simulationDirs_[simI]/"EnKFmembers"
                   /("member_" + Foam::name(memberI))/parameterFile_;

                std::ifstream inFile(paramFile.c_str());
                if (!inFile.is_open())
                {
                    FatalErrorInFunction
                        << "Cannot open " << paramFile
                        << " to reset failed member_" << memberI
                        << abort(FatalError);
                }
                std::string fileContent
                (
                    (std::istreambuf_iterator<char>(inFile)),
                    std::istreambuf_iterator<char>()
                );
                inFile.close();

                for (label paramI = 0; paramI < nModelParams_; paramI++)
                {
                    scalarField clusterVals(nClusters_);
                    for (label clusterI = 0; clusterI < nClusters_; clusterI++)
                    {
                        clusterVals[clusterI] =
                            memberParams[paramI*nClusters_ + clusterI];
                    }

                    const std::string newLine =
                        formatLine(paramNames_[paramI], clusterVals, nClusters_);

                    if
                    (
                        !findAndReplace
                        (
                            fileContent, paramNames_[paramI], newLine,
                            memberI, simI
                        )
                    )
                    {
                        FatalErrorInFunction
                            << "Could not find " << paramNames_[paramI]
                            << " in " << paramFile
                            << " (failed member would rerun with its old "
                            << "parameters)." << abort(FatalError);
                    }
                }

                std::ofstream outFile(paramFile.c_str());
                if (!outFile.is_open())
                {
                    FatalErrorInFunction
                        << "Cannot write " << paramFile << abort(FatalError);
                }
                outFile << fileContent;
            }

            if (Pstream::master())
            {
                failLog << "cycle " << iterationCount_
                        << "  member_" << memberI
                        << "  [" << failReasons_[f] << "]  ->";
                for (label a = 0; a < alphaRows; a++)
                {
                    failLog << "  " << paramNames_[a/nClusters_] << "_c"
                            << a % nClusters_ << "=" << memberParams[a];
                }
                failLog << "  (nValid=" << nValid_ << ")\n";
            }
        }
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    Info << "\n=== updateMemberParameters summary ===" << endl;
    Info << "  Parameter file: "
        << totalParamFileSuccessful << "/" << totalParamFileAttempted
        << " (" << 100.0*totalParamFileSuccessful/max(totalParamFileAttempted,1)
        << "% success)" << endl;

    // Final parameter statistics across all clusters
    Info << "\n  Final parameter statistics:" << endl;
    for (label paramI = 0; paramI < nModelParams_; paramI++)
    {
        for (label clusterI = 0; clusterI < nClusters_; clusterI++)
        {
            label alphaIdx = paramI * nClusters_ + clusterI;
            scalar minV = GREAT, maxV = -GREAT, sumV = 0.0;

            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label idx = (alphaRowOffset + alphaIdx) * totalCols + ensI;
                scalar v = ensemble[idx];
                minV  = min(minV, v);
                maxV  = max(maxV, v);
                sumV += v;
            }
            scalar meanV  = sumV / nValid_;
            scalar varSum = 0.0;
            for (label ensI = 0; ensI < nValid_; ensI++)
            {
                label idx = (alphaRowOffset + alphaIdx) * totalCols + ensI;
                varSum += sqr(ensemble[idx] - meanV);
            }
            scalar stdDev = Foam::sqrt(varSum / nValid_);

            Info << "  " << paramNames_[paramI] << "_c" << clusterI
                 << ": mean=" << meanV
                 << "  std=" << stdDev
                 << "  range=[" << minV << ", " << maxV << "]" << endl;
        }
    }

    Info << "=== updateMemberParameters complete ===" << endl;
}


void EnKF_multiVariable_multiDomain_clustering::computeLocalisationWeights()
{
    Info << "\n=== computeLocalisationWeights (solver distance fields) ===" << endl;
    Info << "  distance fields: " << clusterDistFieldNames_ << endl;
    Info << "  L per domain:    " << featureSpaceLocalisationRadius_ << endl;
    Info << "  weight:          w_c = exp(-dbar_c^2 / 2L^2), "
         << "dbar_c = mean over " << nEnsembles_ << " members" << endl;

    if (simulationDirs_.empty())
    {
        detectSimulationDirectories();
    }

    obsLocWeights_.setSize(nSimulations_);

    forAll(simulationDirs_, simI)
    {
        const label  nObs    = nObservationPointsPerSim_[simI];
        const word   simName = simulationDirs_[simI].name();
        const scalar L       = featureSpaceLocalisationRadius_[simI];
        const scalar twoL2   = 2.0 * L * L;

        Info << "\n  Simulation: " << simName
             << " (" << nObs << " obs points)  L=" << L << endl;

        // meanDist[c][obsI] — ensemble-mean distance to cluster c
        List<scalarField> meanDist(nClusters_, scalarField(nObs, 0.0));

        // --------------------------------------------------------------
        // Accumulate distances over all members
        // --------------------------------------------------------------
        for (label k = 0; k < nValid_; k++)
        {
            const label ensI = validMembers_[k];
            const fileName postProcDir =
                simulationDirs_[simI]
              / "EnKFmembers"
              / ("member_" + Foam::name(ensI))
              / "postProcessing"
              / "sampleDict";

            if (!isDir(postProcDir))
            {
                FatalErrorInFunction
                    << "postProcessing directory not found: " << postProcDir
                    << abort(FatalError);
            }

            const fileName postTimeDir =
                resolveTimeDirectory(postProcDir, simI);

            const fileName xyFile =
                postProcDir/postTimeDir/fileName(featureSampleSetName_ + ".xy");

            if (!isFile(xyFile))
            {
                FatalErrorInFunction
                    << "Sample file not found: " << xyFile
                    << abort(FatalError);
            }

            for (label c = 0; c < nClusters_; c++)
            {
                const label col =
                    xyColumnIndex(xyFile, clusterDistFieldNames_[c], false, 0);

                scalarField d;
                readXYColumn(xyFile, nObs, col, d);

                forAll(d, obsI)
                {
                    if (d[obsI] < 0)
                    {
                        FatalErrorInFunction
                            << "Negative distance " << d[obsI] << " in column '"
                            << clusterDistFieldNames_[c] << "', obs " << obsI
                            << ", file " << xyFile << nl
                            << "The solver must write d >= 0 (not squared, "
                            << "not signed)." << abort(FatalError);
                    }
                }

                meanDist[c] += d;
            }
        }

        for (label c = 0; c < nClusters_; c++)
        {
            meanDist[c] /= scalar(nValid_);
        }

        // --------------------------------------------------------------
        // Weights
        // --------------------------------------------------------------
        obsLocWeights_[simI].setSize(nObs);

        scalarField minW(nClusters_,  GREAT), maxW(nClusters_, -GREAT), sumW(nClusters_, 0.0);
        scalarField minD(nClusters_,  GREAT), maxD(nClusters_, -GREAT), sumD(nClusters_, 0.0);
        labelList   nNearest(nClusters_, 0);
        label       nDiscarded = 0;   // obs with all weights < 1e-3

        for (label obsI = 0; obsI < nObs; obsI++)
        {
            obsLocWeights_[simI][obsI].setSize(nClusters_);

            scalar maxWobs  = 0.0;
            label  nearestC = 0;

            for (label c = 0; c < nClusters_; c++)
            {
                const scalar dbar = meanDist[c][obsI];
                const scalar w    = Foam::exp(-sqr(dbar) / twoL2);

                obsLocWeights_[simI][obsI][c] = w;

                minW[c] = min(minW[c], w);  maxW[c] = max(maxW[c], w);  sumW[c] += w;
                minD[c] = min(minD[c], dbar); maxD[c] = max(maxD[c], dbar); sumD[c] += dbar;

                if (dbar < meanDist[nearestC][obsI]) nearestC = c;
                maxWobs = max(maxWobs, w);
            }

            nNearest[nearestC]++;
            if (maxWobs < 1e-3) nDiscarded++;
        }

        // --------------------------------------------------------------
        // Diagnostics
        // --------------------------------------------------------------
        Info << "    Per cluster (ensemble-mean distance / weight):" << endl;
        for (label c = 0; c < nClusters_; c++)
        {
            Info << "      cluster " << c
                 << ": d=[" << minD[c] << ", " << maxD[c] << "] avg=" << sumD[c] / nObs
                 << "   w=[" << minW[c] << ", " << maxW[c] << "] avg=" << sumW[c] / nObs
                 << "   nearest for " << nNearest[c] << " obs" << endl;
        }

        Info << "    Obs with all weights < 1e-3 (effectively unused): "
             << nDiscarded << " / " << nObs << endl;

        Info << "    Sample weights (first 3 obs points):" << endl;
        for (label obsI = 0; obsI < min(3, nObs); obsI++)
        {
            Info << "      obsI=" << obsI << ": ";
            for (label c = 0; c < nClusters_; c++)
            {
                Info << "c" << c << "=" << obsLocWeights_[simI][obsI][c] << "  ";
            }
            Info << endl;
        }
    }

    Info << "=== computeLocalisationWeights complete ===" << endl;
}