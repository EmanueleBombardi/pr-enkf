#ifndef EnKFControls_H
#define EnKFControls_H

#include "IOdictionary.H"
#include "Time.H"
#include "scalarField.H"
#include "List.H"

namespace Foam
{
    class EnKFControls
    {
        const Time& runTime_;
        IOdictionary dict_;

        label iterationCount_;

        // Per-domain update interval (solver iterations between analyses).
        // Read as a plain label (single-element list, replicated to all
        // domains later) or as a list of length nSimulations_. Length
        // validation is deferred to the main EnKF class once nSimulations_
        // is known — same pattern as featureSpaceLocalisationRadius_.
        scalarList updateInterval_;

        // Number of EnKF analysis cycles. Common to all domains: every
        // cycle, domain k advances updateInterval_[k] iterations, then one
        // joint analysis is performed. Replaces the old 'endEnKF'.
        label nCycles_;


        // Restart strategy between EnKF cycles:
        //   latestTime : each cycle continues from the previous checkpoint
        //   firstTime  : each cycle restarts from initialTime
        word   restartMode_;

        // Time directory the members start from (T0)
        scalar initialTime_;
        scalar cfdUncertainty_;      // NOTE: read from dict, but not yet
                                      // consumed by EnKF_multiVariable_
                                      // multiDomain_clustering's constructor.
                                      // Kept here in case it's meant to be
                                      // wired in later; remove if genuinely
                                      // unneeded.
        scalar obsUncertainty_;
        label nEnsembles_;
        scalar inflationFactor_;
        bool parallelRun_;

        word  clusterScheduler_;
        word  clusterPartition_;
        word  clusterTimeLimit_;
        label clusterTasksPerNode_;
        word  clusterSolverExecutable_;
        word clusterAccount_;
        fileName clusterEnvScript_;
        word     clusterMemPerCpu_;
        label nClusters_;
        label nModelParams_;
        List<word> paramNames_;
        fileName parameterFile_;
        fileNameList memberFiles_;

        // All three are flat scalarFields of size nModelParams_ * nClusters_,
        // indexed as paramI * nClusters_ + clusterI.
        // A scalar entry in the dictionary is silently replicated to all clusters.
        scalarField paramDefaults_;
        scalarField paramPerturbRange_;
        scalarField priorUncertainty_;

        // Hard bounds per (param, cluster), same flat layout as above.
        // REQUIRED: 'min' and 'max' in every 'parameters' entry
        // (scalar, or a list of length nClusters).
        scalarField paramMin_;
        scalarField paramMax_;


        // Distance-based cluster localisation (optional).
        // The solver writes one volScalarField per cluster,
        // <clusterDistanceField_>0 ... <clusterDistanceField_>(nClusters-1),
        // holding each cell's distance d (NOT squared) to that cluster.
        // The EnKF samples them at the observation points, averages over
        // members, and uses w_c = exp(-d_c^2 / 2L^2).
        bool             featureSpaceLocalisation_;

        // Per-domain localisation radii L, in the solver's distance units.
        // Read as a plain scalar (single-element list) or a list of length
        // nSimulations_; length validated later by the main EnKF class.
        // Only read when featureSpaceLocalisation_ is true.
        scalarList       featureSpaceLocalisationRadius_;

        // Prefix of the per-cluster distance fields (default "clusterDist").
        // Only read when featureSpaceLocalisation_ is true.
        word             clusterDistanceField_;

        List<word>  sampledFieldNames_;
        List<bool>  sampledFieldIsVector_;
        List<label> sampledFieldComponent_;
        scalarList  sampledFieldSigmaRel_;
        scalarList  sampledFieldSigmaAbs_;

        word        featureSampleSetName_;

        // Max fraction of members allowed to fail in one cycle
        // (restartMode firstTime only). 0 = any failure is fatal.
        scalar      maxFailedFraction_;


        // ------------------------------------------------------------------
        // Helper: read a scalar-or-list entry from a sub-dictionary.
        //
        // If the value is a plain scalar  →  replicate to all nClusters slots.
        // If the value is a list of length nClusters  →  use directly.
        // Any other list length  →  FatalError.
        //
        // Result written into 'out' starting at index 'offset'
        // (i.e. out[offset + clusterI] for clusterI in [0, nClusters_)).
        // ------------------------------------------------------------------
        void readScalarOrList
        (
            const dictionary& subDict,
            const word&       keyword,
            const label       nClusters,
            scalarField&      out,
            const label       offset
        ) const
        {
            const entry* ePtr = subDict.lookupEntryPtr(keyword, false, false);

            if (!ePtr)
            {
                FatalErrorInFunction
                    << "Keyword '" << keyword
                    << "' not found in parameter sub-dictionary."
                    << abort(FatalError);
            }

            ITstream& is = ePtr->stream();
            token firstTok(is);
            is.rewind();

            if (firstTok == token::BEGIN_LIST)
            {
                scalarList vals(is);

                if (vals.size() == nClusters)
                {
                    forAll(vals, c)
                    {
                        out[offset + c] = vals[c];
                    }
                }
                else if (vals.size() == 1)
                {
                    for (label c = 0; c < nClusters; c++)
                    {
                        out[offset + c] = vals[0];
                    }
                }
                else
                {
                    FatalErrorInFunction
                        << "Keyword '" << keyword << "': list has "
                        << vals.size() << " elements but nClusters="
                        << nClusters << ". Must be exactly 1 or nClusters."
                        << abort(FatalError);
                }
            }
            else
            {
                scalar val = readScalar(is);
                for (label c = 0; c < nClusters; c++)
                {
                    out[offset + c] = val;
                }
            }
        }

        // ------------------------------------------------------------------
        // Helper: read featureSpaceLocalisationRadius as a scalar or list.
        // Only called when localisation is enabled, so it is REQUIRED.
        //
        // Plain scalar  →  single-element scalarList.
        // List          →  stored as-is, length validated later by the main
        //                  EnKF class once nSimulations_ is known.
        // Every entry must be strictly positive.
        // ------------------------------------------------------------------
        scalarList readLocalisationRadius(const dictionary& d) const
        {
            const entry* ePtr =
                d.lookupEntryPtr("featureSpaceLocalisationRadius", false, false);

            if (!ePtr)
            {
                FatalErrorInFunction
                    << "featureSpaceLocalisation is true but "
                    << "'featureSpaceLocalisationRadius' is not set." << nl
                    << "It is expressed in the units of the distance fields "
                    << "written by the solver." << abort(FatalError);
            }

            ITstream& is = ePtr->stream();
            token firstTok(is);
            is.rewind();

            scalarList vals;

            if (firstTok == token::BEGIN_LIST)
            {
                vals = scalarList(is);
            }
            else
            {
                vals = scalarList(1, readScalar(is));
            }

            if (vals.empty())
            {
                FatalErrorInFunction
                    << "featureSpaceLocalisationRadius is an empty list."
                    << abort(FatalError);
            }

            forAll(vals, i)
            {
                if (vals[i] <= 0)
                {
                    FatalErrorInFunction
                        << "featureSpaceLocalisationRadius entry " << i
                        << " = " << vals[i] << " must be > 0."
                        << abort(FatalError);
                }
            }

            return vals;
        }

        // ------------------------------------------------------------------
        // Helper: read updateInterval as a label or a list of labels.
        //
        // Plain label  →  single-element labelList.
        // List         →  stored as-is; length validated later by the main
        //                  EnKF class once nSimulations_ is known.
        // Every entry must be strictly positive.
        // ------------------------------------------------------------------
        scalarList readUpdateInterval(const dictionary& d) const
        {
            const entry* ePtr = d.lookupEntryPtr("updateInterval", false, false);

            if (!ePtr)
            {
                return scalarList(1, scalar(40));
            }

            ITstream& is = ePtr->stream();
            token firstTok(is);
            is.rewind();

            scalarList vals;

            if (firstTok == token::BEGIN_LIST)
            {
                vals = scalarList(is);
            }
            else
            {
                vals = scalarList(1, readScalar(is));
            }

            if (vals.empty())
            {
                FatalErrorInFunction
                    << "updateInterval is an empty list." << abort(FatalError);
            }

            forAll(vals, i)
            {
                if (vals[i] <= 0)
                {
                    FatalErrorInFunction
                        << "updateInterval entry " << i << " = " << vals[i]
                        << " must be > 0." << abort(FatalError);
                }
            }

            return vals;
        }

        // ------------------------------------------------------------------
        // Helper: read memberFiles — a parenthesised list of words, strings
        // or plain numbers (so both 0 and "0" work).
        // Default: ("0" "constant" "system").
        // ------------------------------------------------------------------
        fileNameList readMemberFiles(const dictionary& d) const
        {
            const entry* ePtr = d.lookupEntryPtr("memberFiles", false, false);

            if (!ePtr)
            {
                fileNameList def(3);
                def[0] = "0";  def[1] = "constant";  def[2] = "system";
                return def;
            }

            ITstream& is = ePtr->stream();
            token first(is);

            if (!(first == token::BEGIN_LIST))
            {
                FatalErrorInFunction
                    << "memberFiles must be a list in parentheses, e.g." << nl
                    << "    memberFiles (\"0\" \"constant\" \"system\");"
                    << abort(FatalError);
            }

            DynamicList<fileName> items;

            while (true)
            {
                token tok(is);

                if (!tok.good())
                {
                    FatalErrorInFunction
                        << "memberFiles: unterminated list." << abort(FatalError);
                }
                if (tok == token::END_LIST) break;

                fileName item;
                if (tok.isWord())        item = tok.wordToken();
                else if (tok.isString()) item = tok.stringToken();
                else if (tok.isLabel())  item = Foam::name(tok.labelToken());
                else
                {
                    FatalErrorInFunction
                        << "memberFiles: unsupported entry " << tok
                        << ". Use quoted paths." << abort(FatalError);
                }

                if (item.empty() || item.isAbsolute() || item.find("..") != std::string::npos)
                {
                    FatalErrorInFunction
                        << "memberFiles entry '" << item << "' must be a non-empty "
                        << "path relative to the sim directory, without '..'."
                        << abort(FatalError);
                }

                items.append(item);
            }

            if (items.empty())
            {
                FatalErrorInFunction
                    << "memberFiles is empty." << abort(FatalError);
            }

            return fileNameList(items);
        }

    public:
        EnKFControls(const Time& runTime)
        :
            runTime_(runTime),
            dict_
            (
                IOobject
                (
                    "EnKFDict",
                    runTime.system(),
                    runTime,
                    IOobject::MUST_READ_IF_MODIFIED,
                    IOobject::NO_WRITE
                )
            ),
            iterationCount_(0),
            updateInterval_(1, scalar(40)),
            nCycles_(1),
            restartMode_("latestTime"),
            initialTime_(0),
            cfdUncertainty_(0.04),
            obsUncertainty_(0.02),
            nEnsembles_(40),
            inflationFactor_(1.2),
            parallelRun_(true),
            nClusters_(1),
            nModelParams_(0),
            parameterFile_(fileName("constant/turbulenceProperties")),
            featureSpaceLocalisation_(false),
            featureSpaceLocalisationRadius_(1, scalar(1.0)),  // unused unless enabled
            clusterDistanceField_(word("clusterDist"))
        {
            readControls();
        }

        void readControls()
        {
            iterationCount_        = dict_.lookupOrDefault<label>("iterationCount", 0);

            // Per-domain update interval — label or list, validated later
            updateInterval_        = readUpdateInterval(dict_);

            // Number of analysis cycles (replaces the old endEnKF)
            if (dict_.found("nCycles"))
            {
                nCycles_ = readLabel(dict_.lookup("nCycles"));

                if (nCycles_ <= 0)
                {
                    FatalErrorInFunction
                        << "nCycles = " << nCycles_ << " must be > 0."
                        << abort(FatalError);
                }
            }
            else if (dict_.found("endEnKF"))
            {
                FatalErrorInFunction
                    << "EnKFDict uses the obsolete 'endEnKF' keyword." << nl
                    << "It has been replaced by 'nCycles' (number of EnKF "
                    << "analysis cycles, common to all domains)." << nl
                    << "Old equivalent: nCycles = endEnKF / updateInterval."
                    << abort(FatalError);
            }
            else
            {
                FatalErrorInFunction
                    << "Required keyword 'nCycles' not found in EnKFDict."
                    << abort(FatalError);
            }

            if (iterationCount_ > nCycles_)
            {
                WarningInFunction
                    << "iterationCount (" << iterationCount_
                    << ") > nCycles (" << nCycles_ << ")." << endl;
            }

            restartMode_ = dict_.lookupOrDefault<word>
            (
                "restartMode", word("latestTime")
            );

            if (restartMode_ != "latestTime" && restartMode_ != "firstTime")
            {
                FatalErrorInFunction
                    << "restartMode must be 'latestTime' or 'firstTime', got '"
                    << restartMode_ << "'" << abort(FatalError);
            }

            initialTime_ = dict_.lookupOrDefault<scalar>("initialTime", 0.0);

            if (initialTime_ < 0)
            {
                FatalErrorInFunction
                    << "initialTime = " << initialTime_ << " must be >= 0."
                    << abort(FatalError);
            }

            Info << "EnKFControls: restartMode = " << restartMode_
                 << "  initialTime (T0) = " << initialTime_ << endl;

            Info << "EnKFControls: updateInterval = " << updateInterval_
                 << (updateInterval_.size() == 1
                     ? " (scalar, replicated to all domains)"
                     : " (per domain)")
                 << "  nCycles = " << nCycles_ << endl;
            cfdUncertainty_        = dict_.lookupOrDefault<scalar>("cfdUncertainty", 0.04);
            obsUncertainty_        = dict_.lookupOrDefault<scalar>("obsUncertainty", 0.02);
            nEnsembles_            = dict_.lookupOrDefault<label>("nEnsembles", 40);
            inflationFactor_       = dict_.lookupOrDefault<scalar>("inflationFactor", 1.2);
            maxFailedFraction_ =
                dict_.lookupOrDefault<scalar>("maxFailedFraction", 0.0);

            if (maxFailedFraction_ < 0 || maxFailedFraction_ >= 1)
            {
                FatalErrorInFunction
                    << "maxFailedFraction = " << maxFailedFraction_
                    << " must be in [0, 1)." << abort(FatalError);
            }
            Info << "EnKFControls: maxFailedFraction = " << maxFailedFraction_
                 << endl;
            parallelRun_           = dict_.lookupOrDefault<bool>("parallelRun", true);
            nClusters_             = dict_.lookupOrDefault<label>("nClusters", 1);

            {
                dictionary clusterDict = dict_.subOrEmptyDict("cluster");

                clusterScheduler_ = clusterDict.lookupOrDefault<word>
                (
                    "scheduler", word("slurm")
                );

                if (clusterScheduler_ != "slurm")
                {
                    FatalErrorInFunction
                        << "EnKFDict cluster.scheduler = '" << clusterScheduler_ 
                        << "' is not supported. Only 'slurm' is currently implemented."
                        << abort(FatalError);
                }

                clusterPartition_        = clusterDict.lookupOrDefault<word>("partition", word("cpu"));
                clusterTimeLimit_        = clusterDict.lookupOrDefault<word>("timeLimit", word("12:00:00"));
                clusterTasksPerNode_     = clusterDict.lookupOrDefault<label>("tasksPerNode", 15);
                clusterSolverExecutable_ = clusterDict.lookupOrDefault<word>("solverExecutable", word("simpleFoam"));
                clusterAccount_   = clusterDict.lookupOrDefault<word>("account", word(""));
                clusterEnvScript_ = clusterDict.lookupOrDefault<fileName>("envScript", fileName("$HOME/.bashrc"));
                clusterMemPerCpu_ = clusterDict.lookupOrDefault<word>("memPerCpu", word(""));
                Info << "EnKFControls: cluster settings — scheduler=" << clusterScheduler_
                    << " partition=" << clusterPartition_
                    << " timeLimit=" << clusterTimeLimit_
                    << " tasksPerNode=" << clusterTasksPerNode_
                    << " solverExecutable=" << clusterSolverExecutable_ << endl;
            }

            // ------------------------------------------------------------------
            // Read parameters block.
            // ------------------------------------------------------------------
            const List<dictionary> paramList(dict_.lookup("parameters"));
            nModelParams_ = paramList.size();

            paramNames_.setSize(nModelParams_);
            paramDefaults_.setSize(nModelParams_ * nClusters_);
            paramPerturbRange_.setSize(nModelParams_ * nClusters_);
            priorUncertainty_.setSize(nModelParams_ * nClusters_);
            
            paramMin_.setSize(nModelParams_ * nClusters_);
            paramMax_.setSize(nModelParams_ * nClusters_);

            forAll(paramList, i)
            {
                paramNames_[i] = word(paramList[i].lookup("name"));

                label offset = i * nClusters_;

                readScalarOrList(paramList[i], "default",          nClusters_, paramDefaults_,    offset);
                readScalarOrList(paramList[i], "perturbRange",     nClusters_, paramPerturbRange_, offset);
                readScalarOrList(paramList[i], "priorUncertainty", nClusters_, priorUncertainty_,  offset);


                // Hard bounds: required
                if (!paramList[i].found("min") || !paramList[i].found("max"))
                {
                    FatalErrorInFunction
                        << "Parameter '" << paramNames_[i] << "': 'min' and "
                        << "'max' are required (hard bounds; scalar or one "
                        << "value per cluster)." << abort(FatalError);
                }

                readScalarOrList(paramList[i], "min", nClusters_, paramMin_, offset);
                readScalarOrList(paramList[i], "max", nClusters_, paramMax_, offset);

                for (label c = 0; c < nClusters_; c++)
                {
                    const scalar lo = paramMin_[offset + c];
                    const scalar hi = paramMax_[offset + c];
                    const scalar d  = paramDefaults_[offset + c];

                    if (!(lo < d && d < hi))
                    {
                        FatalErrorInFunction
                            << "Parameter '" << paramNames_[i] << "' cluster "
                            << c << ": need min < default < max, got min="
                            << lo << "  default=" << d << "  max=" << hi
                            << abort(FatalError);
                    }
                }
            }

            // Print resolved values for verification
            Info << "EnKFControls: parameter defaults per cluster:" << nl;
            forAll(paramNames_, i)
            {
                Info << "  " << paramNames_[i] << ":";
                for (label c = 0; c < nClusters_; c++)
                {
                    Info << "  c" << c << "=" << paramDefaults_[i * nClusters_ + c];
                }
                Info << nl;
            }
            Info << "EnKFControls: perturbRange per cluster:" << nl;
            forAll(paramNames_, i)
            {
                Info << "  " << paramNames_[i] << ":";
                for (label c = 0; c < nClusters_; c++)
                {
                    Info << "  c" << c << "=" << paramPerturbRange_[i * nClusters_ + c];
                }
                Info << nl;
            }
            Info << "EnKFControls: priorUncertainty per cluster:" << nl;
            forAll(paramNames_, i)
            {
                Info << "  " << paramNames_[i] << ":";
                for (label c = 0; c < nClusters_; c++)
                {
                    Info << "  c" << c << "=" << priorUncertainty_[i * nClusters_ + c];
                }
                Info << nl;
            }

            Info << "EnKFControls: hard bounds [min, max] per cluster:" << nl;
            forAll(paramNames_, i)
            {
                Info << "  " << paramNames_[i] << ":";
                for (label c = 0; c < nClusters_; c++)
                {
                    Info << "  c" << c << "=[" << paramMin_[i * nClusters_ + c]
                         << ", " << paramMax_[i * nClusters_ + c] << "]";
                }
                Info << nl;
            }

            // ------------------------------------------------------------------
            // Reject obsolete centroid-based localisation keywords, whatever
            // the value of featureSpaceLocalisation, so a stale dict can never
            // run with settings silently ignored.
            // ------------------------------------------------------------------
            {
                const wordList obsoleteKeys
                ({
                    "featureList", "featureMeans", "featureStds", "centroids"
                });

                wordList foundObsolete;
                forAll(obsoleteKeys, i)
                {
                    if (dict_.found(obsoleteKeys[i]))
                    {
                        foundObsolete.append(obsoleteKeys[i]);
                    }
                }

                if (foundObsolete.size())
                {
                    FatalErrorInFunction
                        << "EnKFDict contains obsolete keywords: "
                        << foundObsolete << nl
                        << "Centroid-based localisation has been replaced by "
                        << "solver-provided distance fields." << nl
                        << "Remove these entries. If localisation is enabled, "
                        << "set 'clusterDistanceField <prefix>;' (default "
                        << "clusterDist) and make the solver write "
                        << "<prefix>0 ... <prefix>" << nClusters_ - 1
                        << " (distance d, not squared)."
                        << abort(FatalError);
                }
            }

            // ------------------------------------------------------------------
            // Distance-based cluster localisation (optional)
            // ------------------------------------------------------------------
            featureSpaceLocalisation_ =
                dict_.lookupOrDefault<bool>("featureSpaceLocalisation", false);

            if (featureSpaceLocalisation_)
            {
                featureSpaceLocalisationRadius_ = readLocalisationRadius(dict_);

                clusterDistanceField_ = dict_.lookupOrDefault<word>
                (
                    "clusterDistanceField", word("clusterDist")
                );

                Info << "EnKFControls: cluster localisation ENABLED" << nl
                     << "  distance fields: " << clusterDistanceField_ << "0 ... "
                     << clusterDistanceField_ << nClusters_ - 1 << nl
                     << "  weight:          w_c = exp(-d_c^2 / 2L^2), "
                     << "d_c = ensemble-mean distance" << nl
                     << "  L per domain:    " << featureSpaceLocalisationRadius_
                     << endl;
            }
            else
            {
                if
                (
                    dict_.found("featureSpaceLocalisationRadius")
                 || dict_.found("clusterDistanceField")
                )
                {
                    Info << "EnKFControls: featureSpaceLocalisationRadius / "
                         << "clusterDistanceField present but ignored "
                         << "(featureSpaceLocalisation is false)." << endl;
                }

                Info << "EnKFControls: cluster localisation DISABLED" << endl;
            }
            parameterFile_ = dict_.lookupOrDefault<fileName>
            (
                "parameterFile",
                fileName("constant/turbulenceProperties")
            );

            memberFiles_ = readMemberFiles(dict_);
            Info << "EnKFControls: memberFiles = " << memberFiles_ << endl;

            Info << "EnKFControls: parameterFile = " << parameterFile_ << endl;

            Info << "EnKFControls: read " << nModelParams_
                 << " physical parameters x " << nClusters_ << " clusters." << endl;

            // sampledFieldTypes — type (scalar/vector) and, for vectors, the
            // component to assimilate, for every OBSERVATION field (master
            // system/sampleDict 'fields'). Order does not matter: columns are
            // located by name in the .xy header. Cluster distance fields need
            // no entry (always scalar).
            {
                const List<dictionary> sampledList
                (
                    dict_.lookupOrDefault<List<dictionary>>
                    (
                        "sampledFieldTypes", List<dictionary>()
                    )
                );

                sampledFieldNames_.setSize(sampledList.size());
                sampledFieldIsVector_.setSize(sampledList.size());
                sampledFieldComponent_.setSize(sampledList.size());
                sampledFieldSigmaRel_.setSize(sampledList.size());
                sampledFieldSigmaAbs_.setSize(sampledList.size());

                forAll(sampledList, i)
                {
                    sampledFieldNames_[i] = word(sampledList[i].lookup("name"));
                    
                    sampledFieldSigmaRel_[i] =
                        sampledList[i].lookupOrDefault<scalar>("sigmaRel", -1);
                    sampledFieldSigmaAbs_[i] =
                        sampledList[i].lookupOrDefault<scalar>("sigmaAbs", -1);

                    word typeStr(sampledList[i].lookup("type"));
                    if (typeStr == "vector")
                    {
                        sampledFieldIsVector_[i]  = true;
                        sampledFieldComponent_[i] =
                            sampledList[i].lookupOrDefault<label>("component", 0);
                    }
                    else if (typeStr == "scalar")
                    {
                        sampledFieldIsVector_[i]  = false;
                        sampledFieldComponent_[i] = 0;
                    }
                    else
                    {
                        FatalErrorInFunction
                            << "sampledFieldTypes entry '" << sampledFieldNames_[i]
                            << "': type must be 'scalar' or 'vector', got '"
                            << typeStr << "'" << abort(FatalError);
                    }
                }

                Info << "EnKFControls: sampledFieldTypes:" << nl;
                forAll(sampledFieldNames_, i)
                {
                    Info << "  " << sampledFieldNames_[i]
                         << "  type=" << (sampledFieldIsVector_[i] ? "vector" : "scalar")
                         << "  component=" << sampledFieldComponent_[i]
                         << "  sigmaRel=" << sampledFieldSigmaRel_[i]
                         << "  sigmaAbs=" << sampledFieldSigmaAbs_[i] << nl;
                }
                Info << endl;
            }

            featureSampleSetName_ = dict_.lookupOrDefault<word>
            (
                "featureSampleSetName", word("enkfPoints")
            );
        }

        // Accessors
        const label&  iterationCount()          const { return iterationCount_; }
        const scalarList& updateInterval() const { return updateInterval_; }
        const label&     nCycles()              const { return nCycles_; }
        const word&      restartMode()          const { return restartMode_; }
        const scalar&    initialTime()          const { return initialTime_; }
        const scalar& cfdUncertainty()          const { return cfdUncertainty_; }
        const scalar& obsUncertainty()          const { return obsUncertainty_; }
        const label&  nEnsembles()              const { return nEnsembles_; }
        const scalar& inflationFactor()         const { return inflationFactor_; }
        const scalar& maxFailedFraction()       const { return maxFailedFraction_; }
        const bool&   parallelRun()             const { return parallelRun_; }

        const word&   clusterScheduler()        const { return clusterScheduler_; }
        const word&   clusterPartition()        const { return clusterPartition_; }
        const word&   clusterTimeLimit()        const { return clusterTimeLimit_; }
        const label&  clusterTasksPerNode()     const { return clusterTasksPerNode_; }
        const word&   clusterSolverExecutable() const { return clusterSolverExecutable_; }
        const word&   clusterAccount()          const { return clusterAccount_; }
        const fileName& clusterEnvScript() const { return clusterEnvScript_; }
        const word& clusterMemPerCpu() const { return clusterMemPerCpu_; }

        const label&       nClusters()          const { return nClusters_; }
        const label&       nModelParams()       const { return nModelParams_; }
        const List<word>&  paramNames()         const { return paramNames_; }

        const scalarField&  paramDefaults()     const { return paramDefaults_; }
        const scalarField&  paramPerturbRange() const { return paramPerturbRange_; }
        const scalarField&  priorUncertainty()  const { return priorUncertainty_; }

        const scalarField&  paramMin()          const { return paramMin_; }
        const scalarField&  paramMax()          const { return paramMax_; }

        // Feature-space localisation accessors
        const bool&             featureSpaceLocalisation()       const { return featureSpaceLocalisation_; }
        const scalarList&       featureSpaceLocalisationRadius() const { return featureSpaceLocalisationRadius_; }

        const fileName& parameterFile() const { return parameterFile_; }
        const fileNameList& memberFiles() const { return memberFiles_; }

        const List<word>&  sampledFieldNames()     const { return sampledFieldNames_; }
        const List<bool>&  sampledFieldIsVector()  const { return sampledFieldIsVector_; }
        const List<label>& sampledFieldComponent() const { return sampledFieldComponent_; }
        const scalarList&  sampledFieldSigmaRel()  const { return sampledFieldSigmaRel_; }
        const scalarList&  sampledFieldSigmaAbs()  const { return sampledFieldSigmaAbs_; }
        const word&        featureSampleSetName()  const { return featureSampleSetName_; }
        const word&        clusterDistanceField()  const { return clusterDistanceField_; }
    };
}

#endif